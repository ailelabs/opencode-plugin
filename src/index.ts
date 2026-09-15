/**
 * @ailelabs/opencode-plugin — a zero-config Aile provider for the opencode CLI.
 *
 * Aile (https://aile.sh — the marketplace relay) exposes an OpenAI- AND
 * Anthropic-compatible surface under `/v1/*`:
 *   - `GET  /v1/models`            → catalog (raw, public, un-enveloped)
 *   - `POST /v1/chat/completions`  → OpenAI chat surface
 *   - `POST /v1/messages`          → Anthropic messages surface
 * authenticated with a per-buyer `sk-aile-…` key (accepted as either
 * `Authorization: Bearer` or `x-api-key`). This plugin wires that surface into
 * opencode: it prompts for the key on `opencode auth login`, discovers models
 * dynamically from `/v1/models`, and injects the bearer on outbound inference
 * calls.
 *
 * Ported (heavily trimmed) from `@omniroute/opencode-plugin`. The OmniRoute
 * plugin carries combos / enrichment / gemini-sanitisation / compression /
 * auto-combos / debug-logging / allowlists — Aile exposes none of those today,
 * so they are intentionally omitted. What remains is the auth hook, the dynamic
 * provider hook (+ in-memory TTL cache and best-effort disk snapshot fallback),
 * the fetch interceptor, and a static config-shim for old opencode builds.
 *
 * ── THE ONE CRITICAL DIFFERENCE FROM OmniRoute (model id keying) ────────────
 * opencode dispatches the models-map KEY (== `ModelV2.id`) VERBATIM as the wire
 * `model` field, after stripping only the provider's own first segment
 * (`hook.id`). OmniRoute's `/v1/models` returns BARE ids (`gpt-4o`), so its
 * plugin PREFIXES them (`omniroute/gpt-4o`) to give the wire a provider segment.
 *
 * Aile's `/v1/models` already returns FULLY-QUALIFIED ids (`claude/claude-…`,
 * `kiro/…`, `openai/…`) and Aile's `/v1/*` router expects that exact string.
 * So here `ModelV2.id` == the raw id VERBATIM — never re-prefixed. Prefixing
 * would put `opencode-aile/claude/claude-…` on the wire and Aile would fail to
 * route it. (This is exactly OmniRoute's `raw.id.includes("/") ? raw.id : …`
 * true-branch, which fires for every Aile id.)
 *
 * Zero runtime dependencies: Node built-ins only. `@opencode-ai/plugin` is an
 * OPTIONAL peer dep provided by the opencode runtime; we model its contract
 * with local `import type`-erased shims so this package builds and tests with
 * nothing installed.
 */

import { createHash } from "node:crypto";
import { mkdir, readFile, unlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

// ────────────────────────────────────────────────────────────────────────────
// Local shims for the opencode plugin contract (@opencode-ai/plugin).
// Declared here (not imported) so the package has zero installed dependencies.
// All are erased at build time; the opencode runtime supplies the real shapes.
// ────────────────────────────────────────────────────────────────────────────

type Awaitable<T> = T | Promise<T>;

/**
 * A fetch-shaped function — what the AI-SDK accepts for its `fetch` option.
 * Deliberately narrower than the global `typeof fetch` (which carries a
 * `preconnect` member the SDK never calls), so an ordinary async closure
 * satisfies it.
 */
export type FetchLike = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;

/** An `{type:"api"}` auth method as read from opencode's `auth login` UX. */
export interface AuthMethodApi {
  type: "api";
  label: string;
  prompts?: Array<{ type: "text" | "password"; key: string; message: string }>;
}

/** What an auth loader returns → forwarded to the AI-SDK provider factory. */
export interface AuthLoaderOptions {
  apiKey?: string;
  baseURL?: string;
  fetch?: FetchLike;
  [k: string]: unknown;
}

/** The credential record opencode hands back through `getAuth()` / `ctx.auth`. */
export type AuthInfo =
  | { type: "api"; key: string; baseURL?: string; [k: string]: unknown }
  | { type: string; [k: string]: unknown }
  | Record<string, unknown>;

export interface AuthHook {
  provider: string;
  methods: AuthMethodApi[];
  loader: (getAuth: () => Awaitable<AuthInfo | undefined>, provider: unknown) => Awaitable<AuthLoaderOptions>;
}

export interface ProviderModelsContext {
  auth?: AuthInfo;
  [k: string]: unknown;
}

/** The rich model shape opencode's provider hook returns (a superset-tolerant view). */
export interface ModelV2 {
  id: string;
  name: string;
  providerID: string;
  api: { id: string; url: string; npm: string };
  capabilities: {
    temperature: boolean;
    reasoning: boolean;
    attachment: boolean;
    toolcall: boolean;
    input: { text: boolean; audio: boolean; image: boolean; video: boolean; pdf: boolean };
    output: { text: boolean; audio: boolean; image: boolean; video: boolean; pdf: boolean };
    interleaved: boolean;
  };
  cost: { input: number; output: number; cache: { read: number; write: number } };
  limit: { context: number; input?: number; output: number };
  status: "active" | "alpha" | "beta" | "deprecated";
  options: Record<string, unknown>;
  headers: Record<string, string>;
  release_date: string;
}

export interface ProviderHook {
  id: string;
  models: (provider: unknown, ctx: ProviderModelsContext) => Awaitable<Record<string, ModelV2>>;
}

/** The opencode config object mutated by the config hook. */
export interface OpencodeConfig {
  provider?: Record<string, unknown>;
  [k: string]: unknown;
}

export type ConfigHook = (input: OpencodeConfig) => Awaitable<void>;

export interface Hooks {
  auth?: AuthHook;
  provider?: ProviderHook;
  config?: ConfigHook;
  [k: string]: unknown;
}

export interface PluginInput {
  directory?: string;
  worktree?: string;
  [k: string]: unknown;
}

export type Plugin = (input: PluginInput, options?: unknown) => Awaitable<Hooks>;

// ────────────────────────────────────────────────────────────────────────────
// Options
// ────────────────────────────────────────────────────────────────────────────

/**
 * Upstream provider ids whose Aile serve format is Anthropic ("claude"). Models
 * whose id begins with one of these (`<prefix>/<model>`) are surfaced through
 * `@ai-sdk/anthropic` (Aile's `/v1/messages` surface); everything else through
 * `@ai-sdk/openai-compatible` (`/v1/chat/completions`).
 *
 * Sourced from the merged Aile catalog (`apps/api/src/lib/catalog.omniroute.json`,
 * every provider whose `transport.format === "claude"`). `kiro` and `gitlab-duo`
 * are deliberately ABSENT: their Phase-4 serve translators accept OpenAI-chat
 * input only, so they must resolve to the openai-compatible adapter.
 */
export const DEFAULT_ANTHROPIC_PROVIDER_IDS: readonly string[] = [
  "agentrouter",
  "anthropic",
  "bailian-coding-plan",
  "claude",
  "kimi-coding",
  "kimi-coding-apikey",
  "minimax",
  "minimax-cn",
  "wafer",
  "zai",
];

const DEFAULT_MODEL_CACHE_TTL_MS = 5 * 60 * 1000;
const OPENCODE_PROVIDER_PREFIX = "opencode-";

export interface AilePluginFeatures {
  /** Inject the bearer key on outbound inference calls. Default: true. */
  fetchInterceptor?: boolean;
  /** Persist / read a last-known-good catalog snapshot to disk. Default: true. */
  diskCache?: boolean;
}

export interface AilePluginOptions {
  /**
   * Aile relay base URL — the PUBLIC relay origin's `/v1`, e.g.
   * `https://relay.aile.example/v1`. NOT the web app's `/relay` proxy. When
   * omitted here it may still be resolved from the stored credential's
   * `baseURL` (set during `auth login`).
   */
  baseURL?: string;
  /** Provider id (default `"aile"`). Auto-prefixed with `opencode-` for the hook. */
  providerId?: string;
  /** Display name in opencode's model picker. Default `"Aile"`. */
  displayName?: string;
  /** In-memory catalog cache TTL (ms). Default 5 minutes. */
  modelCacheTtl?: number;
  /** Upstream provider ids routed via the Anthropic adapter. Default: {@link DEFAULT_ANTHROPIC_PROVIDER_IDS}. */
  anthropicProviderIds?: readonly string[];
  features?: AilePluginFeatures;
}

export interface ResolvedAilePluginOptions {
  baseURL?: string;
  /** OC-native-adapter-gate id (`opencode-<x>`). Used as `hook.id` / `AuthHook.provider`. */
  providerId: string;
  /**
   * UNPREFIXED id (`<x>`). Used for `ModelV2.providerID`, the disk-snapshot
   * filename, and any value that must NOT carry the `opencode-` gate prefix.
   */
  aileProviderId: string;
  displayName: string;
  modelCacheTtl: number;
  anthropicProviderIds: readonly string[];
  features: Required<AilePluginFeatures>;
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** Strip a single leading `opencode-` (the OC-native-adapter-gate prefix). */
function stripOpencodePrefix(id: string): string {
  return id.startsWith(OPENCODE_PROVIDER_PREFIX) ? id.slice(OPENCODE_PROVIDER_PREFIX.length) : id;
}

/**
 * Validate + normalise plugin options. Throws on structurally-invalid input so
 * a typo in `opencode.json` surfaces loudly rather than silently disabling the
 * provider. `baseURL` is validated as a URL only when present.
 */
export function resolveAilePluginOptions(opts?: AilePluginOptions): ResolvedAilePluginOptions {
  const o = opts ?? {};
  if (!isPlainObject(o)) {
    throw new TypeError("@ailelabs/opencode-plugin: options must be an object");
  }

  let baseURL: string | undefined;
  if (o.baseURL !== undefined) {
    if (typeof o.baseURL !== "string" || o.baseURL.trim().length === 0) {
      throw new TypeError("@ailelabs/opencode-plugin: baseURL must be a non-empty string");
    }
    const trimmed = o.baseURL.trim();
    try {
      // eslint-disable-next-line no-new
      new URL(trimmed);
    } catch {
      throw new Error(`@ailelabs/opencode-plugin: baseURL is not a valid URL: ${JSON.stringify(o.baseURL)}`);
    }
    baseURL = trimmed;
  }

  const rawProviderId =
    typeof o.providerId === "string" && o.providerId.trim().length > 0 ? o.providerId.trim() : "aile";
  const aileProviderId = stripOpencodePrefix(rawProviderId);
  const providerId = rawProviderId.startsWith(OPENCODE_PROVIDER_PREFIX)
    ? rawProviderId
    : `${OPENCODE_PROVIDER_PREFIX}${rawProviderId}`;

  const displayName =
    typeof o.displayName === "string" && o.displayName.trim().length > 0 ? o.displayName.trim() : "Aile";

  const modelCacheTtl =
    typeof o.modelCacheTtl === "number" && Number.isFinite(o.modelCacheTtl) && o.modelCacheTtl >= 0
      ? o.modelCacheTtl
      : DEFAULT_MODEL_CACHE_TTL_MS;

  const anthropicProviderIds =
    Array.isArray(o.anthropicProviderIds) && o.anthropicProviderIds.every((x) => typeof x === "string")
      ? (o.anthropicProviderIds as readonly string[])
      : DEFAULT_ANTHROPIC_PROVIDER_IDS;

  const featuresIn = isPlainObject(o.features) ? (o.features as AilePluginFeatures) : {};
  const features: Required<AilePluginFeatures> = {
    fetchInterceptor: featuresIn.fetchInterceptor !== false,
    diskCache: featuresIn.diskCache !== false,
  };

  return {
    baseURL,
    providerId,
    aileProviderId,
    displayName,
    modelCacheTtl,
    anthropicProviderIds,
    features,
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Pure URL / id helpers
// ────────────────────────────────────────────────────────────────────────────

export function trimTrailingSlashes(url: string): string {
  let end = url.length;
  while (end > 0 && url[end - 1] === "/") end--;
  return end < url.length ? url.slice(0, end) : url;
}

/**
 * Ensure a baseURL ends in exactly one `/v1`. Both Aile surfaces live under
 * `/v1`: the openai-compatible adapter appends `/chat/completions` and the
 * anthropic adapter appends `/messages`, so BOTH want the `/v1` suffix present.
 * Idempotent — never produces `/v1/v1`.
 */
export function ensureV1Suffix(url: string): string {
  const trimmed = trimTrailingSlashes(url);
  return /\/v\d+$/.test(trimmed) ? trimmed : `${trimmed}/v1`;
}

/**
 * Resolve the AI-SDK adapter block for a model id. The id's first segment
 * (`<upstream>/<model>`) decides the surface: anthropic-format upstreams →
 * `@ai-sdk/anthropic`, everything else → `@ai-sdk/openai-compatible`.
 *
 * Unlike OmniRoute (whose anthropic branch omits `/v1`), BOTH Aile branches use
 * `ensureV1Suffix(baseURL)`: Aile serves anthropic at `/v1/messages` and openai
 * at `/v1/chat/completions`, so both adapters need the `/v1` base.
 */
export function resolveApiBlock(
  modelId: string,
  baseURL: string,
  anthropicProviderIds: readonly string[] = DEFAULT_ANTHROPIC_PROVIDER_IDS
): { id: string; url: string; npm: string } {
  const slash = modelId.indexOf("/");
  const prefix = slash === -1 ? modelId : modelId.slice(0, slash);
  const isAnthropic = anthropicProviderIds.includes(prefix);
  const url = ensureV1Suffix(baseURL);
  return isAnthropic
    ? { id: "anthropic", url, npm: "@ai-sdk/anthropic" }
    : { id: "openai-compatible", url, npm: "@ai-sdk/openai-compatible" };
}

// ────────────────────────────────────────────────────────────────────────────
// Raw catalog entry → ModelV2
// ────────────────────────────────────────────────────────────────────────────

export interface AileRawModelEntry {
  id: string;
  name?: string;
  context_length?: number;
  max_input_tokens?: number;
  max_output_tokens?: number;
  release_date?: string;
  capabilities?: {
    temperature?: boolean;
    reasoning?: boolean;
    thinking?: boolean;
    attachment?: boolean;
    vision?: boolean;
    tool_calling?: boolean;
  };
  input_modalities?: string[];
  output_modalities?: string[];
  [k: string]: unknown;
}

/**
 * Map a raw `/v1/models` entry → `ModelV2`.
 *
 * `id` is the raw id VERBATIM (see the file header): Aile ids are already
 * `<upstream>/<model>` and Aile routes on that exact string. We only synthesise
 * a prefix in the degenerate case of a bare (slash-less) id, which Aile's
 * catalog does not currently emit — there we fall back to
 * `<aileProviderId>/<id>` so opencode's `(providerID, modelID)` split still
 * yields two parts.
 */
export function mapRawModelToModelV2(
  raw: AileRawModelEntry,
  ctx: { aileProviderId: string; baseURL: string; anthropicProviderIds?: readonly string[] }
): ModelV2 {
  const rawId = raw.id;
  const modelId = rawId.includes("/") ? rawId : `${ctx.aileProviderId}/${rawId}`;
  const caps = raw.capabilities ?? {};
  const inMods = new Set(raw.input_modalities ?? ["text"]);
  const outMods = new Set(raw.output_modalities ?? ["text"]);

  return {
    id: modelId,
    name: typeof raw.name === "string" && raw.name.trim().length > 0 ? raw.name.trim() : rawId,
    providerID: ctx.aileProviderId,
    api: resolveApiBlock(rawId, ctx.baseURL, ctx.anthropicProviderIds),
    capabilities: {
      temperature: caps.temperature ?? true,
      reasoning: Boolean(caps.reasoning || caps.thinking),
      attachment: Boolean(caps.attachment ?? caps.vision ?? false),
      toolcall: Boolean(caps.tool_calling ?? true),
      input: {
        text: inMods.has("text"),
        audio: inMods.has("audio"),
        image: inMods.has("image"),
        video: inMods.has("video"),
        pdf: inMods.has("pdf"),
      },
      output: {
        text: outMods.has("text"),
        audio: outMods.has("audio"),
        image: outMods.has("image"),
        video: outMods.has("video"),
        pdf: outMods.has("pdf"),
      },
      interleaved: Boolean(caps.thinking),
    },
    // Aile's /v1/models does not surface pricing — emit a zeroed cost block
    // (the field is required by ModelV2). Live pricing is Aile's own concern at
    // serve time; opencode reads this for display only.
    cost: { input: 0, output: 0, cache: { read: 0, write: 0 } },
    limit: {
      context: typeof raw.context_length === "number" ? raw.context_length : 0,
      ...(typeof raw.max_input_tokens === "number" ? { input: raw.max_input_tokens } : {}),
      output: typeof raw.max_output_tokens === "number" ? raw.max_output_tokens : 0,
    },
    status: "active",
    options: {},
    headers: {},
    release_date: typeof raw.release_date === "string" ? raw.release_date : "",
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Models fetcher — GET {baseURL}/v1/models
// ────────────────────────────────────────────────────────────────────────────

export type AileModelsFetcher = (
  baseURL: string,
  apiKey: string,
  timeoutMs?: number
) => Promise<AileRawModelEntry[]>;

export const defaultAileModelsFetcher: AileModelsFetcher = async (baseURL, apiKey, timeoutMs = 10_000) => {
  if (!apiKey) throw new Error("@ailelabs/opencode-plugin: apiKey required to fetch /v1/models");
  if (!baseURL) throw new Error("@ailelabs/opencode-plugin: baseURL required to fetch /v1/models");

  const trimmed = trimTrailingSlashes(baseURL);
  // Tolerate both `https://host` and `https://host/v1`; never produce `/v1/v1`.
  const url = /\/v\d+$/.test(trimmed) ? `${trimmed}/models` : `${trimmed}/v1/models`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: "GET",
      headers: { Authorization: `Bearer ${apiKey}`, Accept: "application/json" },
      signal: controller.signal,
    });
    if (!res.ok) {
      throw new Error(`@ailelabs/opencode-plugin: GET ${url} failed: ${res.status} ${res.statusText}`);
    }
    const body = (await res.json()) as unknown;
    // Aile's /v1/models returns `{ object:"list", data:[{id,...}] }`; also
    // tolerate a bare array for forward-compat.
    const rawList: unknown[] = Array.isArray(body)
      ? body
      : isPlainObject(body) && Array.isArray((body as { data?: unknown }).data)
        ? ((body as { data: unknown[] }).data)
        : [];
    const out: AileRawModelEntry[] = [];
    for (const r of rawList) {
      if (isPlainObject(r) && typeof (r as { id?: unknown }).id === "string") {
        out.push(r as AileRawModelEntry);
      }
    }
    return out;
  } finally {
    clearTimeout(timer);
  }
};

// ────────────────────────────────────────────────────────────────────────────
// Fetch interceptor — inject the bearer on Aile's inference paths ONLY.
// ────────────────────────────────────────────────────────────────────────────

/**
 * The returned fetch is handed to the AI-SDK in place of global fetch. It
 * attaches `Authorization: Bearer <key>` (and defaults Content-Type when a body
 * is present) to requests whose origin AND path match Aile's inference surface:
 * `/v1/chat/completions`, `/v1/messages`, `/v1/models`. Every other request —
 * different host, different path — is forwarded untouched, so the buyer key
 * NEVER leaks off the configured relay origin.
 */
export function createAileFetchInterceptor(config: { apiKey: string; baseURL: string }): FetchLike {
  let baseOrigin: string | undefined;
  const inferencePaths = new Set<string>();
  try {
    const baseUrl = new URL(config.baseURL);
    baseOrigin = baseUrl.origin;
    const basePath = ensureV1Suffix(baseUrl.pathname);
    inferencePaths.add(`${basePath}/chat/completions`);
    inferencePaths.add(`${basePath}/messages`);
    inferencePaths.add(`${basePath}/models`);
  } catch {
    // Malformed baseURL disables injection rather than broadening scope.
  }

  return async (input, init = {}) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;

    let requestUrl: URL | undefined;
    try {
      requestUrl = new URL(url);
    } catch {
      // Relative/malformed URL — forward untouched; do not attach credentials.
    }
    const normalizedPath = requestUrl ? trimTrailingSlashes(requestUrl.pathname) || "/" : undefined;
    const targetsInference =
      requestUrl?.origin === baseOrigin &&
      normalizedPath !== undefined &&
      inferencePaths.has(normalizedPath);
    if (!targetsInference) {
      return fetch(input, init);
    }

    // Merge order: Request-attached headers → init.headers → our injected
    // headers last (so the key we own always wins over a caller-supplied one).
    const headers = new Headers(input instanceof Request ? input.headers : undefined);
    if (init.headers) {
      const initHeaders = new Headers(init.headers);
      initHeaders.forEach((value, key) => headers.set(key, value));
    }
    headers.set("Authorization", `Bearer ${config.apiKey}`);
    const hasBody = init.body != null || input instanceof Request;
    if (!headers.has("Content-Type") && hasBody) {
      headers.set("Content-Type", "application/json");
    }
    return fetch(input, { ...init, headers });
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Disk snapshot — last-known-good catalog fallback when /v1/models is offline.
// Lives at ${OPENCODE_DATA_DIR ?? ~/.local/share/opencode}/plugins/aile-<id>.json
// (dir 0o700, file 0o600 — same posture as opencode's auth.json).
// ────────────────────────────────────────────────────────────────────────────

interface AileDiskSnapshot {
  v: 1;
  identityFingerprint: string;
  rawModels: AileRawModelEntry[];
  writtenAt: number;
}

export function diskSnapshotPath(aileProviderId: string): string {
  const dir = process.env.OPENCODE_DATA_DIR ?? path.join(os.homedir(), ".local/share/opencode");
  return path.join(dir, "plugins", `aile-${aileProviderId}.json`);
}

/** Bind a snapshot to (normalised baseURL, apiKey) without persisting the key. */
export function diskSnapshotIdentityFingerprint(baseURL: string, apiKey: string): string {
  let normalizedBaseURL: string;
  try {
    const parsed = new URL(baseURL);
    parsed.hash = "";
    parsed.pathname = trimTrailingSlashes(parsed.pathname) || "/";
    normalizedBaseURL = parsed.toString();
  } catch {
    normalizedBaseURL = trimTrailingSlashes(baseURL);
  }
  return createHash("sha256").update(JSON.stringify([normalizedBaseURL, apiKey])).digest("hex");
}

export type AileDiskSnapshotWriter = (
  aileProviderId: string,
  rawModels: AileRawModelEntry[],
  identityFingerprint: string,
  now: number
) => Promise<void>;

export type AileDiskSnapshotReader = (
  aileProviderId: string,
  identityFingerprint: string
) => Promise<AileRawModelEntry[] | undefined>;

/** Best-effort disk write. Soft-fails on any I/O error (never throws). */
export const defaultDiskSnapshotWriter: AileDiskSnapshotWriter = async (
  aileProviderId,
  rawModels,
  identityFingerprint,
  now
) => {
  try {
    const file = diskSnapshotPath(aileProviderId);
    await mkdir(path.dirname(file), { recursive: true, mode: 0o700 });
    const snapshot: AileDiskSnapshot = { v: 1, identityFingerprint, rawModels, writtenAt: now };
    await writeFile(file, JSON.stringify(snapshot), { encoding: "utf8", mode: 0o600 });
  } catch {
    // Soft-fail; the in-memory cache is authoritative.
  }
};

/** Best-effort disk read. Returns undefined when missing / corrupt / mismatched. */
export const defaultDiskSnapshotReader: AileDiskSnapshotReader = async (aileProviderId, identityFingerprint) => {
  try {
    const file = diskSnapshotPath(aileProviderId);
    const body = await readFile(file, "utf8");
    const parsed = JSON.parse(body) as Partial<AileDiskSnapshot>;
    if (!parsed || parsed.v !== 1 || parsed.identityFingerprint !== identityFingerprint) {
      return undefined;
    }
    return Array.isArray(parsed.rawModels) ? parsed.rawModels : [];
  } catch {
    return undefined;
  }
};

/** Best-effort delete (force-sync). */
export async function clearDiskSnapshot(aileProviderId: string): Promise<boolean> {
  try {
    await unlink(diskSnapshotPath(aileProviderId));
    return true;
  } catch {
    return false;
  }
}

/** No-op disk pair — used by tests to avoid filesystem side effects. */
export const noopDiskSnapshotWriter: AileDiskSnapshotWriter = async () => {};
export const noopDiskSnapshotReader: AileDiskSnapshotReader = async () => undefined;

// ────────────────────────────────────────────────────────────────────────────
// auth.json reader (config hook has no getAuth()).
// ────────────────────────────────────────────────────────────────────────────

export interface AuthJsonShape {
  [providerId: string]: { type?: string; key?: string; baseURL?: string; [k: string]: unknown } | undefined;
}

export type AileReadAuthJson = () => Promise<AuthJsonShape | undefined | null>;

export const defaultReadAuthJson: AileReadAuthJson = async () => {
  const dir = process.env.OPENCODE_DATA_DIR ?? path.join(os.homedir(), ".local/share/opencode");
  const file = path.join(dir, "auth.json");
  let body: string;
  try {
    body = await readFile(file, "utf8");
  } catch {
    return undefined; // missing/unreadable — expected before `auth login`.
  }
  try {
    const parsed = JSON.parse(body) as unknown;
    return isPlainObject(parsed) ? (parsed as AuthJsonShape) : null;
  } catch {
    return null; // exists but corrupt.
  }
};

// ────────────────────────────────────────────────────────────────────────────
// In-memory catalog cache
// ────────────────────────────────────────────────────────────────────────────

interface AileCacheEntry {
  rawModels: AileRawModelEntry[];
  expiresAt: number;
}
export type AileFetchCache = Map<string, AileCacheEntry>;

/** In-memory cache key: `${baseURL}::sha256(apiKey)` — never stores the raw key. */
export function modelsCacheKey(baseURL: string, apiKey: string): string {
  const h = createHash("sha256").update(apiKey).digest("hex");
  return `${baseURL}::${h}`;
}

/** Narrow an opencode auth record to a usable `{type:"api", key}` credential. */
function extractApiKey(auth: unknown): string | undefined {
  if (
    isPlainObject(auth) &&
    (auth as { type?: unknown }).type === "api" &&
    typeof (auth as { key?: unknown }).key === "string" &&
    (auth as { key: string }).key.length > 0
  ) {
    return (auth as { key: string }).key;
  }
  return undefined;
}

// ────────────────────────────────────────────────────────────────────────────
// Auth hook
// ────────────────────────────────────────────────────────────────────────────

/**
 * `{type:"api"}` auth: prompts for the `sk-aile-…` key on `opencode auth login`.
 * The loader projects the stored credential into AI-SDK options and, when a
 * baseURL is resolvable, wraps in the bearer-injecting fetch interceptor.
 * Non-api / empty-key credentials → `{}` (opencode then surfaces the connect
 * flow instead of dispatching with bad creds).
 */
export function createAileAuthHook(opts?: AilePluginOptions): AuthHook {
  const resolved = resolveAilePluginOptions(opts);
  return {
    provider: resolved.providerId,
    methods: [
      {
        type: "api",
        label: `${resolved.displayName} API Key`,
        prompts: [{ type: "text", key: "apiKey", message: `${resolved.displayName} API key (sk-aile-…)` }],
      },
    ],
    loader: async (getAuth) => {
      const auth = await getAuth();
      const apiKey = extractApiKey(auth);
      if (!apiKey) return {};

      const authBaseURL = isPlainObject(auth) ? (auth as { baseURL?: unknown }).baseURL : undefined;
      const resolvedBaseURL =
        resolved.baseURL ?? (typeof authBaseURL === "string" && authBaseURL.length > 0 ? authBaseURL : "");
      if (!resolvedBaseURL) {
        // No baseURL → the interceptor can't scope itself; fall back to
        // apiKey-only and let the SDK use its default fetch.
        return { apiKey };
      }
      if (!resolved.features.fetchInterceptor) {
        return { apiKey, baseURL: resolvedBaseURL };
      }
      return {
        apiKey,
        baseURL: resolvedBaseURL,
        fetch: createAileFetchInterceptor({ apiKey, baseURL: resolvedBaseURL }),
      };
    },
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Provider hook — dynamic catalog from /v1/models
// ────────────────────────────────────────────────────────────────────────────

function resolveBaseURL(
  resolved: ResolvedAilePluginOptions,
  auth: unknown,
  provider: unknown
): string {
  const authBaseURL = isPlainObject(auth) ? (auth as { baseURL?: unknown }).baseURL : undefined;
  const providerBaseURL = isPlainObject(provider)
    ? ((provider as { options?: { baseURL?: unknown } }).options?.baseURL)
    : undefined;
  return (
    resolved.baseURL ??
    (typeof authBaseURL === "string" && authBaseURL.length > 0 ? authBaseURL : undefined) ??
    (typeof providerBaseURL === "string" && providerBaseURL.length > 0 ? providerBaseURL : undefined) ??
    ""
  );
}

export function createAileProviderHook(
  opts?: AilePluginOptions,
  deps: {
    fetcher?: AileModelsFetcher;
    now?: () => number;
    cache?: AileFetchCache;
    diskSnapshotReader?: AileDiskSnapshotReader;
    diskSnapshotWriter?: AileDiskSnapshotWriter;
    logger?: { warn: (...args: unknown[]) => void };
  } = {}
): ProviderHook {
  const resolved = resolveAilePluginOptions(opts);
  const fetcher = deps.fetcher ?? defaultAileModelsFetcher;
  const now = deps.now ?? Date.now;
  const cache: AileFetchCache = deps.cache ?? new Map();
  const wantDisk = resolved.features.diskCache;
  const diskReader = deps.diskSnapshotReader ?? defaultDiskSnapshotReader;
  const diskWriter = deps.diskSnapshotWriter ?? defaultDiskSnapshotWriter;
  const logger = deps.logger ?? console;

  return {
    id: resolved.providerId,
    async models(provider, ctx) {
      const apiKey = extractApiKey(ctx?.auth);
      if (!apiKey) return {}; // opencode exposes the connect flow instead.

      const baseURL = resolveBaseURL(resolved, ctx?.auth, provider);
      if (!baseURL) {
        logger.warn(
          `[aile-plugin] provider.models(${resolved.providerId}): no baseURL resolvable — ` +
            `set baseURL in opencode.json plugin options or during \`opencode auth login\`.`
        );
        return {};
      }

      const cacheKey = modelsCacheKey(baseURL, apiKey);
      const t = now();
      const cached = cache.get(cacheKey);

      let rawModels: AileRawModelEntry[];
      if (cached && cached.expiresAt > t) {
        rawModels = cached.rawModels;
      } else {
        try {
          rawModels = await fetcher(baseURL, apiKey, 10_000);
          cache.set(cacheKey, { rawModels, expiresAt: t + resolved.modelCacheTtl });
          if (wantDisk) {
            const fp = diskSnapshotIdentityFingerprint(baseURL, apiKey);
            await diskWriter(resolved.aileProviderId, rawModels, fp, t);
          }
        } catch (err) {
          // Fetch failed (offline, IP-block, relay down). Fall back to the
          // last-known-good disk snapshot so the picker still populates.
          if (wantDisk) {
            const fp = diskSnapshotIdentityFingerprint(baseURL, apiKey);
            const snap = await diskReader(resolved.aileProviderId, fp);
            if (snap && snap.length > 0) {
              logger.warn(
                `[aile-plugin] /v1/models fetch failed; serving last-known-good disk snapshot (${snap.length} models)`
              );
              rawModels = snap;
            } else {
              throw err;
            }
          } else {
            throw err;
          }
        }
      }

      const models: Record<string, ModelV2> = {};
      for (const raw of rawModels) {
        if (!raw.id) continue;
        const model = mapRawModelToModelV2(raw, {
          aileProviderId: resolved.aileProviderId,
          baseURL,
          anthropicProviderIds: resolved.anthropicProviderIds,
        });
        // Key by model.id (== raw id verbatim for Aile) — this is the string
        // opencode dispatches on the wire; see the file header.
        models[model.id] = model;
      }
      return models;
    },
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Config-shim hook — static provider block for opencode builds that predate the
// dynamic provider hook. Runs BEFORE auth.loader, so it reads auth.json itself.
// Emits the openai-compatible single-npm block; the dynamic hook (when present)
// wins by opencode's own merge rule and applies the proper per-model
// anthropic/openai split. On very old opencode, claude-native models are served
// through the openai-compatible surface (Aile accepts OpenAI chat for them too).
// ────────────────────────────────────────────────────────────────────────────

interface AileStaticModelEntry {
  name: string;
  attachment?: boolean;
  reasoning?: boolean;
  temperature?: boolean;
  tool_call?: boolean;
  limit?: { context: number; output: number };
}

interface AileStaticProviderEntry {
  npm: string;
  name: string;
  options: { baseURL: string };
  models: Record<string, AileStaticModelEntry>;
}

export function buildStaticProviderEntry(
  rawModels: AileRawModelEntry[],
  resolved: ResolvedAilePluginOptions,
  baseURL: string
): AileStaticProviderEntry {
  const models: Record<string, AileStaticModelEntry> = {};
  for (const raw of rawModels) {
    if (!raw.id) continue;
    const caps = raw.capabilities ?? {};
    const entry: AileStaticModelEntry = {
      name: typeof raw.name === "string" && raw.name.trim().length > 0 ? raw.name.trim() : raw.id,
    };
    const attachment = caps.attachment ?? caps.vision;
    if (typeof attachment === "boolean") entry.attachment = attachment;
    if (typeof caps.reasoning === "boolean" || typeof caps.thinking === "boolean") {
      entry.reasoning = Boolean(caps.reasoning || caps.thinking);
    }
    if (typeof caps.temperature === "boolean") entry.temperature = caps.temperature;
    if (typeof caps.tool_calling === "boolean") entry.tool_call = caps.tool_calling;
    if (
      typeof raw.context_length === "number" &&
      raw.context_length > 0 &&
      typeof raw.max_output_tokens === "number" &&
      raw.max_output_tokens > 0
    ) {
      entry.limit = { context: raw.context_length, output: raw.max_output_tokens };
    }
    // Key by the raw id VERBATIM — opencode dispatches a static model key
    // verbatim as the wire `model` (only the top-level provider[<id>] segment
    // is stripped). Aile routes on that exact string.
    models[raw.id] = entry;
  }
  return {
    npm: "@ai-sdk/openai-compatible",
    name: resolved.displayName,
    options: { baseURL: ensureV1Suffix(baseURL) },
    models,
  };
}

export function createAileConfigHook(
  opts?: AilePluginOptions,
  deps: {
    readAuthJson?: AileReadAuthJson;
    fetcher?: AileModelsFetcher;
    now?: () => number;
    cache?: AileFetchCache;
    logger?: { warn: (...args: unknown[]) => void };
  } = {}
): ConfigHook {
  const resolved = resolveAilePluginOptions(opts);
  const readAuthJson = deps.readAuthJson ?? defaultReadAuthJson;
  const fetcher = deps.fetcher ?? defaultAileModelsFetcher;
  const now = deps.now ?? Date.now;
  const cache: AileFetchCache = deps.cache ?? new Map();
  const logger = deps.logger ?? console;

  return async (input) => {
    if (!isPlainObject(input)) return;
    const providerMap = isPlainObject(input.provider) ? (input.provider as Record<string, unknown>) : undefined;
    // Operator override wins — never clobber a manually-curated block.
    if (providerMap && providerMap[resolved.providerId] !== undefined) return;

    const authJson = await readAuthJson();
    if (!authJson) return; // missing/corrupt → no-op (fresh install).
    const record = authJson[resolved.providerId] ?? authJson[resolved.aileProviderId];
    const apiKey = extractApiKey(record);
    if (!apiKey) return;

    const authBaseURL = isPlainObject(record) ? (record as { baseURL?: unknown }).baseURL : undefined;
    const baseURL =
      resolved.baseURL ?? (typeof authBaseURL === "string" && authBaseURL.length > 0 ? authBaseURL : "");
    if (!baseURL) return;

    let rawModels: AileRawModelEntry[] = [];
    const cacheKey = modelsCacheKey(baseURL, apiKey);
    const t = now();
    const cached = cache.get(cacheKey);
    if (cached && cached.expiresAt > t) {
      rawModels = cached.rawModels;
    } else {
      try {
        rawModels = await fetcher(baseURL, apiKey, 10_000);
        cache.set(cacheKey, { rawModels, expiresAt: t + resolved.modelCacheTtl });
      } catch (err) {
        // Fail-open: publish a complete-shape stub block so opencode still has
        // a renderable provider entry; the dynamic hook fills it in later.
        logger.warn("[aile-plugin] config hook: /v1/models fetch failed; publishing empty static block", err);
        rawModels = [];
      }
    }

    const entry = buildStaticProviderEntry(rawModels, resolved, baseURL);
    const nextProvider = providerMap ?? {};
    nextProvider[resolved.providerId] = entry;
    input.provider = nextProvider;
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Plugin factory + default export
// ────────────────────────────────────────────────────────────────────────────

/** Package id — the value opencode records as this plugin's source. */
export const AILE_PLUGIN_ID = "@ailelabs/opencode-plugin" as const;

/**
 * The opencode Plugin. opencode passes options as the SECOND argument (from the
 * `[name, opts]` tuple in opencode.json), so multi-instance setups get their own
 * options bag per registration. All three hooks share ONE in-memory cache so a
 * cold start hits `/v1/models` once per TTL rather than once per hook.
 */
export const AilePlugin: Plugin = async (_input, options) => {
  const opts = (options ?? undefined) as AilePluginOptions | undefined;
  const cache: AileFetchCache = new Map();
  return {
    auth: createAileAuthHook(opts),
    provider: createAileProviderHook(opts, { cache }),
    config: createAileConfigHook(opts, { cache }),
  };
};

export default { id: AILE_PLUGIN_ID, server: AilePlugin };
