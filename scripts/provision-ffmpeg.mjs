#!/usr/bin/env node
// Provision + verify the Tauri client's bundled FFmpeg sidecar, checksum-pinned.
//
// The Tauri config declares `externalBin: ["ffmpeg"]`, so Tauri resolves the sidecar
// as `apps/client/src-tauri/ffmpeg-<target-triple>[.exe]` at bundle time. This script is
// the COMMITTED, reproducible mechanism that guarantees the bytes on disk are exactly the
// pinned build — never an unverified download, never a fabricated hash (No-Fake-Success).
//
// Modes:
//   node scripts/provision-ffmpeg.mjs           ensure the sidecar exists AND matches the
//                                               pinned sha256; download+verify if a real
//                                               `url` (or FFMPEG_URL_<triple> env) is set.
//   node scripts/provision-ffmpeg.mjs --check    verify an EXISTING sidecar only (no
//                                               download). Use in `verify` / pre-build.
//   --manifest <path>                            override manifest (tests / alt pins).
//   --help
//
// Exit codes: 0 ok · 2 hash/size mismatch · 3 unsupported triple · 4 missing (--check)
//             · 5 missing + no pinned source · 6 download failed · 7 bad usage.
import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { stat, rename, unlink, mkdir } from "node:fs/promises";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import { tmpdir } from "node:os";

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs(argv) {
  const opts = { check: false, manifest: join(REPO_ROOT, "scripts", "ffmpeg-manifest.json") };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--check") opts.check = true;
    else if (a === "--help" || a === "-h") opts.help = true;
    else if (a === "--manifest") opts.manifest = resolve(argv[++i] ?? "");
    else return { error: `unknown argument: ${a}` };
  }
  return opts;
}

// Map the host (Node platform/arch) to a Rust target triple + sidecar extension.
function hostTriple() {
  const p = process.platform;
  const a = process.arch;
  if (p === "win32" && a === "x64") return { triple: "x86_64-pc-windows-msvc", ext: ".exe" };
  if (p === "darwin" && a === "arm64") return { triple: "aarch64-apple-darwin", ext: "" };
  if (p === "darwin" && a === "x64") return { triple: "x86_64-apple-darwin", ext: "" };
  if (p === "linux" && a === "x64") return { triple: "x86_64-unknown-linux-gnu", ext: "" };
  if (p === "linux" && a === "arm64") return { triple: "aarch64-unknown-linux-gnu", ext: "" };
  return { triple: null, ext: "" };
}

async function sha256File(path) {
  const hash = createHash("sha256");
  await pipeline(createReadStream(path), hash);
  return hash.digest("hex");
}

async function fileSize(path) {
  try {
    return (await stat(path)).size;
  } catch (e) {
    if (e.code === "ENOENT") return null;
    throw e;
  }
}

function die(code, msg) {
  console.error(`[provision-ffmpeg] ${msg}`);
  process.exit(code);
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.error) die(7, opts.error);
  if (opts.help) {
    console.log(
      "Verify/provision the checksum-pinned FFmpeg sidecar.\n" +
        "  (default)      ensure sidecar present AND sha256-verified (download if url pinned)\n" +
        "  --check        verify an existing sidecar only; never downloads\n" +
        "  --manifest P   use manifest at P (default scripts/ffmpeg-manifest.json)\n",
    );
    process.exit(0);
  }

  let manifest;
  try {
    const { readFile } = await import("node:fs/promises");
    manifest = JSON.parse(await readFile(opts.manifest, "utf8"));
  } catch (e) {
    die(7, `cannot read manifest ${opts.manifest}: ${e.message}`);
  }

  const { triple, ext } = hostTriple();
  if (!triple) {
    die(3, `unsupported host platform/arch: ${process.platform}/${process.arch}`);
  }
  const entry = manifest.targets?.[triple];
  if (!entry) {
    die(
      3,
      `no pinned FFmpeg for target '${triple}'. Refusing to fabricate a build ` +
        `(No-Fake-Success). Add a real {sha256,bytes,url} entry to ${opts.manifest}.`,
    );
  }

  const sidecarPath = join(REPO_ROOT, manifest.sidecarDir, `${manifest.sidecarBase}-${triple}${entry.ext ?? ext}`);
  const size = await fileSize(sidecarPath);

  if (size !== null) {
    // Present -> verify size then hash, fail-closed on any mismatch.
    if (entry.bytes != null && size !== entry.bytes) {
      die(2, `size mismatch for ${sidecarPath}: expected ${entry.bytes} bytes, got ${size}. Refusing (fail-closed).`);
    }
    const got = await sha256File(sidecarPath);
    if (got !== entry.sha256) {
      die(2, `sha256 mismatch for ${sidecarPath}:\n  expected ${entry.sha256}\n  got      ${got}\nRefusing (fail-closed) — delete and re-provision.`);
    }
    console.log(`[provision-ffmpeg] OK: ${manifest.sidecarBase}-${triple}${entry.ext ?? ext} verified (${got.slice(0, 16)}…, ${size} bytes).`);
    process.exit(0);
  }

  // Missing.
  if (opts.check) {
    die(
      4,
      `sidecar missing: ${sidecarPath}\n  Provide it via 'git lfs pull' (if LFS-tracked) or ` +
        `'node scripts/provision-ffmpeg.mjs' with a pinned url, then re-run --check.`,
    );
  }

  const envKey = `FFMPEG_URL_${triple.replace(/[^A-Za-z0-9]/g, "_")}`;
  const url = process.env[envKey] || entry.url;
  if (!url) {
    die(
      5,
      `sidecar missing and no pinned source for '${triple}'.\n` +
        `  Choose one: (a) 'git lfs pull' if the binary is LFS-tracked,\n` +
        `             (b) set ${envKey}=<immutable url whose sha256==${entry.sha256}>,\n` +
        `             (c) drop the verified build at ${sidecarPath}.\n` +
        `  Not faking success: the sidecar is genuinely absent.`,
    );
  }

  // Download -> temp -> verify -> atomic move. Fail-closed on any hash/size mismatch.
  console.log(`[provision-ffmpeg] downloading ${triple} sidecar from ${envKey === undefined ? "manifest url" : url} …`);
  await mkdir(join(REPO_ROOT, manifest.sidecarDir), { recursive: true });
  const tmp = join(tmpdir(), `ffmpeg-${triple}-${process.pid}.part`);
  try {
    const res = await fetch(url);
    if (!res.ok || !res.body) die(6, `download failed: HTTP ${res.status}`);
    await pipeline(Readable.fromWeb(res.body), createWriteStream(tmp));
  } catch (e) {
    await unlink(tmp).catch(() => {});
    die(6, `download error: ${e.message}`);
  }
  const dlSize = await fileSize(tmp);
  const dlHash = await sha256File(tmp);
  if ((entry.bytes != null && dlSize !== entry.bytes) || dlHash !== entry.sha256) {
    await unlink(tmp).catch(() => {});
    die(2, `downloaded artifact failed verification (got ${dlHash}, ${dlSize} bytes; expected ${entry.sha256}, ${entry.bytes}). Refusing (fail-closed).`);
  }
  await rename(tmp, sidecarPath);
  console.log(`[provision-ffmpeg] OK: provisioned + verified ${sidecarPath}.`);
  process.exit(0);
}

main().catch((e) => die(1, `unexpected: ${e.stack || e}`));
