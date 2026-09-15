/**
 * Unit tests for @ailelabs/opencode-plugin. Every test drives pure helpers or
 * hooks with INJECTED deps — no real network, no real filesystem.
 */

import { describe, expect, test } from "bun:test";
import {
  AILE_PLUGIN_ID,
  AilePlugin,
  buildStaticProviderEntry,
  createAileAuthHook,
  createAileConfigHook,
  createAileFetchInterceptor,
  createAileProviderHook,
  DEFAULT_ANTHROPIC_PROVIDER_IDS,
  defaultAileModelsFetcher,
  diskSnapshotIdentityFingerprint,
  ensureV1Suffix,
  mapRawModelToModelV2,
  modelsCacheKey,
  noopDiskSnapshotReader,
  noopDiskSnapshotWriter,
  resolveAilePluginOptions,
  resolveApiBlock,
  trimTrailingSlashes,
  type AileModelsFetcher,
  type AileRawModelEntry,
} from "./index";

const BASE = "https://relay.aile.test/v1";

// ── options ─────────────────────────────────────────────────────────────────

describe("resolveAilePluginOptions", () => {
  test("defaults", () => {
    const r = resolveAilePluginOptions();
    expect(r.providerId).toBe("opencode-aile");
    expect(r.aileProviderId).toBe("aile");
    expect(r.displayName).toBe("Aile");
    expect(r.modelCacheTtl).toBe(5 * 60 * 1000);
    expect(r.features.fetchInterceptor).toBe(true);
    expect(r.features.diskCache).toBe(true);
    expect(r.anthropicProviderIds).toEqual(DEFAULT_ANTHROPIC_PROVIDER_IDS);
  });

  test("providerId is prefixed for the OC gate but unprefixed id retained", () => {
    expect(resolveAilePluginOptions({ providerId: "aile" }).providerId).toBe("opencode-aile");
    // idempotent: an already-prefixed id is not double-prefixed
    const r = resolveAilePluginOptions({ providerId: "opencode-aile" });
    expect(r.providerId).toBe("opencode-aile");
    expect(r.aileProviderId).toBe("aile");
  });

  test("custom providerId", () => {
    const r = resolveAilePluginOptions({ providerId: "myrelay" });
    expect(r.providerId).toBe("opencode-myrelay");
    expect(r.aileProviderId).toBe("myrelay");
  });

  test("rejects a non-URL baseURL", () => {
    expect(() => resolveAilePluginOptions({ baseURL: "not a url" })).toThrow();
    expect(() => resolveAilePluginOptions({ baseURL: "   " })).toThrow();
  });

  test("features can be disabled", () => {
    const r = resolveAilePluginOptions({ features: { fetchInterceptor: false, diskCache: false } });
    expect(r.features.fetchInterceptor).toBe(false);
    expect(r.features.diskCache).toBe(false);
  });
});

// ── url helpers ───────────────────────────────────────────────────────────────

describe("trimTrailingSlashes / ensureV1Suffix", () => {
  test("trim", () => {
    expect(trimTrailingSlashes("https://x/v1///")).toBe("https://x/v1");
    expect(trimTrailingSlashes("https://x/v1")).toBe("https://x/v1");
  });
  test("ensureV1Suffix adds when missing, idempotent when present", () => {
    expect(ensureV1Suffix("https://relay.aile.test")).toBe("https://relay.aile.test/v1");
    expect(ensureV1Suffix("https://relay.aile.test/")).toBe("https://relay.aile.test/v1");
    expect(ensureV1Suffix("https://relay.aile.test/v1")).toBe("https://relay.aile.test/v1");
    expect(ensureV1Suffix("https://relay.aile.test/v1/")).toBe("https://relay.aile.test/v1");
    // any /vN is treated as already-suffixed (never doubles)
    expect(ensureV1Suffix("https://relay.aile.test/v2")).toBe("https://relay.aile.test/v2");
  });
});

// ── resolveApiBlock ───────────────────────────────────────────────────────────

describe("resolveApiBlock", () => {
  test("claude-format prefix → anthropic adapter (baseURL keeps /v1)", () => {
    const b = resolveApiBlock("claude/claude-sonnet-4-5", BASE);
    expect(b.id).toBe("anthropic");
    expect(b.npm).toBe("@ai-sdk/anthropic");
    expect(b.url).toBe(BASE); // /v1 present → hits /v1/messages
  });

  test("openai prefix → openai-compatible", () => {
    const b = resolveApiBlock("openai/gpt-4o", BASE);
    expect(b.id).toBe("openai-compatible");
    expect(b.npm).toBe("@ai-sdk/openai-compatible");
    expect(b.url).toBe(BASE); // /v1 present → hits /v1/chat/completions
  });

  test("kiro and gitlab route to openai-compatible (their translators take OpenAI chat)", () => {
    expect(resolveApiBlock("kiro/claude-sonnet-4-5", BASE).id).toBe("openai-compatible");
    expect(resolveApiBlock("gitlab-duo/code-completion", BASE).id).toBe("openai-compatible");
  });

  test("every declared claude-format prefix resolves to anthropic", () => {
    for (const p of DEFAULT_ANTHROPIC_PROVIDER_IDS) {
      expect(resolveApiBlock(`${p}/some-model`, BASE).id).toBe("anthropic");
    }
  });

  test("both branches add /v1 when the baseURL lacks it", () => {
    const bare = "https://relay.aile.test";
    expect(resolveApiBlock("claude/x", bare).url).toBe(`${bare}/v1`);
    expect(resolveApiBlock("openai/x", bare).url).toBe(`${bare}/v1`);
  });
});

// ── mapRawModelToModelV2 (THE keying contract) ────────────────────────────────

describe("mapRawModelToModelV2", () => {
  const ctx = { aileProviderId: "aile", baseURL: BASE };

  test("already-qualified id is kept VERBATIM (never re-prefixed)", () => {
    const m = mapRawModelToModelV2({ id: "claude/claude-sonnet-4-5" }, ctx);
    expect(m.id).toBe("claude/claude-sonnet-4-5"); // NOT aile/claude/... or opencode-aile/...
    expect(m.providerID).toBe("aile");
    expect(m.api.id).toBe("anthropic");
  });

  test("openai model id kept verbatim, openai-compatible adapter", () => {
    const m = mapRawModelToModelV2({ id: "openai/gpt-4o", name: "GPT-4o" }, ctx);
    expect(m.id).toBe("openai/gpt-4o");
    expect(m.name).toBe("GPT-4o");
    expect(m.api.npm).toBe("@ai-sdk/openai-compatible");
  });

  test("bare (slash-less) id falls back to <aileProviderId>/<id>", () => {
    const m = mapRawModelToModelV2({ id: "mystery-model" }, ctx);
    expect(m.id).toBe("aile/mystery-model");
  });

  test("cost is always zeroed (Aile does not surface pricing here)", () => {
    const m = mapRawModelToModelV2({ id: "openai/gpt-4o" }, ctx);
    expect(m.cost).toEqual({ input: 0, output: 0, cache: { read: 0, write: 0 } });
  });

  test("limits and capabilities are projected", () => {
    const m = mapRawModelToModelV2(
      {
        id: "openai/gpt-4o",
        context_length: 128000,
        max_output_tokens: 16384,
        capabilities: { reasoning: true, vision: true, tool_calling: true },
        input_modalities: ["text", "image"],
      },
      ctx
    );
    expect(m.limit.context).toBe(128000);
    expect(m.limit.output).toBe(16384);
    expect(m.capabilities.reasoning).toBe(true);
    expect(m.capabilities.attachment).toBe(true);
    expect(m.capabilities.input.image).toBe(true);
    expect(m.capabilities.input.audio).toBe(false);
    expect(m.status).toBe("active");
  });
});

// ── cache key ─────────────────────────────────────────────────────────────────

describe("modelsCacheKey", () => {
  test("does not embed the raw key and varies by key + base", () => {
    const k1 = modelsCacheKey(BASE, "sk-aile-AAA");
    const k2 = modelsCacheKey(BASE, "sk-aile-BBB");
    expect(k1).not.toContain("sk-aile-AAA");
    expect(k1).not.toBe(k2);
    expect(modelsCacheKey("https://other/v1", "sk-aile-AAA")).not.toBe(k1);
  });
});

describe("diskSnapshotIdentityFingerprint", () => {
  test("stable across trailing-slash noise, varies by key", () => {
    const a = diskSnapshotIdentityFingerprint("https://relay.aile.test/v1", "sk-aile-A");
    const b = diskSnapshotIdentityFingerprint("https://relay.aile.test/v1/", "sk-aile-A");
    expect(a).toBe(b);
    expect(diskSnapshotIdentityFingerprint(BASE, "sk-aile-B")).not.toBe(a);
  });
});

// ── fetch interceptor (credential-scoping) ────────────────────────────────────

describe("createAileFetchInterceptor", () => {
  function capturingFetch() {
    const calls: Array<{ url: string; auth: string | null }> = [];
    const orig = globalThis.fetch;
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
      const headers = new Headers(
        init?.headers ?? (input instanceof Request ? input.headers : undefined)
      );
      calls.push({ url, auth: headers.get("Authorization") });
      return new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } });
    }) as typeof fetch;
    return { calls, restore: () => (globalThis.fetch = orig) };
  }

  test("injects bearer on the three inference paths", async () => {
    const cap = capturingFetch();
    try {
      const f = createAileFetchInterceptor({ apiKey: "sk-aile-XYZ", baseURL: BASE });
      await f(`${BASE}/chat/completions`, { method: "POST", body: "{}" });
      await f(`${BASE}/messages`, { method: "POST", body: "{}" });
      await f(`${BASE}/models`, { method: "GET" });
      expect(cap.calls.map((c) => c.auth)).toEqual([
        "Bearer sk-aile-XYZ",
        "Bearer sk-aile-XYZ",
        "Bearer sk-aile-XYZ",
      ]);
    } finally {
      cap.restore();
    }
  });

  test("does NOT inject on other paths or cross-origin hosts", async () => {
    const cap = capturingFetch();
    try {
      const f = createAileFetchInterceptor({ apiKey: "sk-aile-XYZ", baseURL: BASE });
      await f(`${BASE}/some-other-endpoint`, { method: "POST", body: "{}" });
      await f("https://evil.example.com/v1/chat/completions", { method: "POST", body: "{}" });
      await f("https://evil.example.com/v1/models", { method: "GET" });
      expect(cap.calls.map((c) => c.auth)).toEqual([null, null, null]);
    } finally {
      cap.restore();
    }
  });

  test("tolerates trailing slash on the matched path", async () => {
    const cap = capturingFetch();
    try {
      const f = createAileFetchInterceptor({ apiKey: "sk-aile-XYZ", baseURL: BASE });
      await f(`${BASE}/chat/completions/`, { method: "POST", body: "{}" });
      expect(cap.calls[0]?.auth).toBe("Bearer sk-aile-XYZ");
    } finally {
      cap.restore();
    }
  });
});

// ── models fetcher (parse) ────────────────────────────────────────────────────

describe("defaultAileModelsFetcher", () => {
  function withFetch(handler: (url: string) => Response, fn: () => Promise<void>) {
    const orig = globalThis.fetch;
    globalThis.fetch = (async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
      return handler(url);
    }) as typeof fetch;
    return fn().finally(() => (globalThis.fetch = orig));
  }

  test("parses {data:[...]} and drops entries without a string id", async () => {
    await withFetch(
      (url) => {
        expect(url).toBe(`${BASE}/models`);
        return new Response(
          JSON.stringify({ object: "list", data: [{ id: "openai/gpt-4o" }, { noid: true }, { id: 5 }] }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      },
      async () => {
        const out = await defaultAileModelsFetcher(BASE, "sk-aile-A");
        expect(out.map((m) => m.id)).toEqual(["openai/gpt-4o"]);
      }
    );
  });

  test("tolerates a bare array and a base without /v1", async () => {
    await withFetch(
      (url) => {
        expect(url).toBe("https://relay.aile.test/v1/models");
        return new Response(JSON.stringify([{ id: "claude/claude-sonnet-4-5" }]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      },
      async () => {
        const out = await defaultAileModelsFetcher("https://relay.aile.test", "sk-aile-A");
        expect(out.map((m) => m.id)).toEqual(["claude/claude-sonnet-4-5"]);
      }
    );
  });

  test("throws on non-2xx", async () => {
    await withFetch(
      () => new Response("nope", { status: 401 }),
      async () => {
        await expect(defaultAileModelsFetcher(BASE, "sk-aile-A")).rejects.toThrow();
      }
    );
  });
});

// ── auth hook ─────────────────────────────────────────────────────────────────

describe("createAileAuthHook", () => {
  test("provider is the prefixed OC-gate id; method is api", () => {
    const hook = createAileAuthHook({ baseURL: BASE });
    expect(hook.provider).toBe("opencode-aile");
    expect(hook.methods[0]?.type).toBe("api");
    expect(hook.methods[0]?.prompts?.[0]?.key).toBe("apiKey");
  });

  test("loader returns {} for a non-api / empty credential", async () => {
    const hook = createAileAuthHook({ baseURL: BASE });
    expect(await hook.loader(async () => undefined, {})).toEqual({});
    expect(await hook.loader(async () => ({ type: "oauth" }) as any, {})).toEqual({});
    expect(await hook.loader(async () => ({ type: "api", key: "" }) as any, {})).toEqual({});
  });

  test("loader returns apiKey + baseURL + a fetch interceptor for a valid key", async () => {
    const hook = createAileAuthHook({ baseURL: BASE });
    const opts = await hook.loader(async () => ({ type: "api", key: "sk-aile-A" }), {});
    expect(opts.apiKey).toBe("sk-aile-A");
    expect(opts.baseURL).toBe(BASE);
    expect(typeof opts.fetch).toBe("function");
  });

  test("fetch interceptor omitted when disabled", async () => {
    const hook = createAileAuthHook({ baseURL: BASE, features: { fetchInterceptor: false } });
    const opts = await hook.loader(async () => ({ type: "api", key: "sk-aile-A" }), {});
    expect(opts.fetch).toBeUndefined();
    expect(opts.baseURL).toBe(BASE);
  });
});

// ── provider hook ─────────────────────────────────────────────────────────────

describe("createAileProviderHook", () => {
  const rawModels: AileRawModelEntry[] = [
    { id: "claude/claude-sonnet-4-5", name: "Claude Sonnet 4.5" },
    { id: "openai/gpt-4o", name: "GPT-4o" },
  ];
  const makeFetcher = (): { fetcher: AileModelsFetcher; count: () => number } => {
    let n = 0;
    return { fetcher: async () => ((n++), rawModels), count: () => n };
  };

  test("returns {} without a usable credential", async () => {
    const hook = createAileProviderHook({ baseURL: BASE }, { fetcher: makeFetcher().fetcher });
    expect(await hook.models({}, { auth: undefined })).toEqual({});
  });

  test("maps models keyed by verbatim id and caches within TTL", async () => {
    const f = makeFetcher();
    const hook = createAileProviderHook(
      { baseURL: BASE },
      {
        fetcher: f.fetcher,
        now: () => 1000,
        diskSnapshotReader: noopDiskSnapshotReader,
        diskSnapshotWriter: noopDiskSnapshotWriter,
      }
    );
    const models = await hook.models({}, { auth: { type: "api", key: "sk-aile-A" } });
    expect(Object.keys(models).sort()).toEqual(["claude/claude-sonnet-4-5", "openai/gpt-4o"]);
    expect(models["claude/claude-sonnet-4-5"]?.api.id).toBe("anthropic");
    expect(models["openai/gpt-4o"]?.api.id).toBe("openai-compatible");
    // second call within TTL → cache hit (fetcher not called again)
    await hook.models({}, { auth: { type: "api", key: "sk-aile-A" } });
    expect(f.count()).toBe(1);
  });

  test("falls back to the disk snapshot when the fetch fails", async () => {
    const hook = createAileProviderHook(
      { baseURL: BASE },
      {
        fetcher: async () => {
          throw new Error("offline");
        },
        now: () => 1000,
        diskSnapshotReader: async () => rawModels,
        diskSnapshotWriter: noopDiskSnapshotWriter,
        logger: { warn: () => {} },
      }
    );
    const models = await hook.models({}, { auth: { type: "api", key: "sk-aile-A" } });
    expect(Object.keys(models).length).toBe(2);
  });

  test("rethrows when the fetch fails and no snapshot exists", async () => {
    const hook = createAileProviderHook(
      { baseURL: BASE },
      {
        fetcher: async () => {
          throw new Error("offline");
        },
        now: () => 1000,
        diskSnapshotReader: noopDiskSnapshotReader,
        diskSnapshotWriter: noopDiskSnapshotWriter,
        logger: { warn: () => {} },
      }
    );
    await expect(hook.models({}, { auth: { type: "api", key: "sk-aile-A" } })).rejects.toThrow("offline");
  });

  test("hook id is the prefixed OC-gate id", () => {
    expect(createAileProviderHook({ baseURL: BASE }).id).toBe("opencode-aile");
  });
});

// ── config hook (static shim) ─────────────────────────────────────────────────

describe("createAileConfigHook / buildStaticProviderEntry", () => {
  const rawModels: AileRawModelEntry[] = [
    { id: "claude/claude-sonnet-4-5", name: "Claude Sonnet 4.5", context_length: 200000, max_output_tokens: 8192 },
    { id: "openai/gpt-4o", name: "GPT-4o" },
  ];

  test("static block keys models by raw id verbatim, single openai-compatible npm", () => {
    const resolved = resolveAilePluginOptions({ baseURL: BASE });
    const entry = buildStaticProviderEntry(rawModels, resolved, BASE);
    expect(entry.npm).toBe("@ai-sdk/openai-compatible");
    expect(Object.keys(entry.models).sort()).toEqual(["claude/claude-sonnet-4-5", "openai/gpt-4o"]);
    expect(entry.models["claude/claude-sonnet-4-5"]?.limit).toEqual({ context: 200000, output: 8192 });
    expect(entry.options.baseURL).toBe(BASE);
  });

  test("hook injects the provider block from auth.json + /v1/models", async () => {
    const hook = createAileConfigHook(
      { baseURL: BASE },
      {
        readAuthJson: async () => ({ "opencode-aile": { type: "api", key: "sk-aile-A" } }),
        fetcher: async () => rawModels,
        now: () => 1000,
      }
    );
    const config: { provider?: Record<string, unknown> } = {};
    await hook(config);
    expect(config.provider?.["opencode-aile"]).toBeDefined();
  });

  test("does not clobber an operator-provided provider block", async () => {
    const hook = createAileConfigHook(
      { baseURL: BASE },
      {
        readAuthJson: async () => ({ "opencode-aile": { type: "api", key: "sk-aile-A" } }),
        fetcher: async () => rawModels,
        now: () => 1000,
      }
    );
    const config = { provider: { "opencode-aile": { npm: "custom" } } };
    await hook(config);
    expect(config.provider["opencode-aile"]).toEqual({ npm: "custom" });
  });

  test("no-op when auth.json lacks the provider", async () => {
    const hook = createAileConfigHook(
      { baseURL: BASE },
      { readAuthJson: async () => ({}), fetcher: async () => rawModels, now: () => 1000 }
    );
    const config: { provider?: Record<string, unknown> } = {};
    await hook(config);
    expect(config.provider).toBeUndefined();
  });
});

// ── plugin wiring ─────────────────────────────────────────────────────────────

describe("AilePlugin default export + factory", () => {
  test("factory returns all three hooks sharing options", async () => {
    const hooks = await AilePlugin({ directory: "/tmp" }, { baseURL: BASE });
    expect(hooks.auth?.provider).toBe("opencode-aile");
    expect(hooks.provider?.id).toBe("opencode-aile");
    expect(typeof hooks.config).toBe("function");
  });

  test("plugin id constant", () => {
    expect(AILE_PLUGIN_ID).toBe("@ailelabs/opencode-plugin");
  });
});
