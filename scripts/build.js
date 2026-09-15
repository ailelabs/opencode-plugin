#!/usr/bin/env bun
/**
 * Bundle the opencode plugin into dist/index.js and emit dist/index.d.ts.
 *
 * There is no tsup in this repo (see apps/aile.sh/scripts/build.js), so we drive
 * the bundle with Bun.build directly and emit declarations with `tsc`.
 *
 * The published artifact opencode actually loads is dist/index.js. The peer
 * `@opencode-ai/plugin` (and the `@ai-sdk/*` adapters we only NAME as strings)
 * are provided by the opencode runtime, never bundled — this plugin's own
 * runtime dependencies are Node built-ins only.
 */

import fs from "node:fs";
import path from "node:path";

const root = path.join(import.meta.dirname, "..");
const outdir = path.join(root, "dist");

fs.rmSync(outdir, { recursive: true, force: true });
fs.mkdirSync(outdir, { recursive: true });

const result = await Bun.build({
  entrypoints: [path.join(root, "src/index.ts")],
  outdir,
  target: "node",
  format: "esm",
  naming: "index.js",
  // Never bundle the opencode runtime contract; it's provided by the host.
  external: ["@opencode-ai/plugin", "@opencode-ai/sdk"],
});

if (!result.success) {
  for (const log of result.logs) console.error(log);
  process.exit(1);
}

const outFile = path.join(outdir, "index.js");
const bytes = fs.statSync(outFile).size;
console.log(`Built dist/index.js — ${(bytes / 1024).toFixed(1)} KB`);

// Declarations: best-effort. dist/index.js is the artifact opencode loads, so a
// missing tsc (constrained CI, offline install) must not fail the JS build — but
// when the toolchain is present we ship accurate .d.ts for editor tooling.
try {
  const proc = Bun.spawnSync(["bun", "x", "tsc", "--project", path.join(root, "tsconfig.json")], {
    cwd: root,
    stdout: "inherit",
    stderr: "inherit",
  });
  if (proc.exitCode === 0 && fs.existsSync(path.join(outdir, "index.d.ts"))) {
    console.log("Emitted dist/index.d.ts");
  } else {
    console.warn("warning: tsc did not emit declarations (dist/index.js still built)");
  }
} catch {
  console.warn("warning: tsc unavailable — skipped declaration emit (dist/index.js still built)");
}
