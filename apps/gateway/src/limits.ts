// Edge input-bound constants for /api/jobs/create validation.
//
// These live in their OWN module (not index.ts) on purpose: a Worker ENTRYPOINT
// module (the one wrangler bundles as `main`) may only export handlers/functions/
// ExportedHandler/classes. A plain-value named export (e.g. `export const MAX_x = 200`)
// from the entrypoint makes the workerd runtime refuse to start:
//   "Incorrect type for map entry 'MAX_FREETEXT_CHARS': the provided value is not of
//    type 'function or ExportedHandler'. The Workers runtime failed to start."
// `wrangler deploy --dry-run` and vitest (which drives the Hono app via app.request()
// in Node) BOTH miss this — only the real runtime (`wrangler dev`/deploy) catches it.
// index.ts imports these internally; tests import them from here.

// Đợt 17 F3/F4 — bound job input at the EDGE (defense-in-depth mirror of the
// worker's pydantic gate). An untrusted-but-registered device (Zero-Trust: ALL
// registered devices are untrusted) can sign a valid payload with a huge `segments`
// array (count or per-segment text). The worker folds ALL segments into ONE Qwen
// prompt then tokenizes + generates once -> VRAM OOM or a hang past MAX_PLAUSIBLE_MS
// -> Station 3 quarantines the (globally URL-keyed) worker for 24h -> one bad device
// DoSes EVERY tenant. Refusing here with a fast 400 turns "hang the cluster" into
// "one rejected request" without ever touching the worker or the quarantine machinery.
export const MAX_SEGMENTS = 2_000;               // max approved segments per job
export const MAX_SEGMENT_TEXT_CHARS = 2_000;     // max chars in a single segment's text
export const MAX_TOTAL_TEXT_CHARS = 200_000;     // max chars summed across all segments
export const MAX_FREETEXT_CHARS = 200;           // target/style/source language free-text cap
// Đợt 18 F6: the worker also embeds each segment's `id` and `speaker`/`speaker_id` into the
// single Qwen prompt (translation_service builds them into model_inputs), yet the Đợt-17
// bound measured only `text`. A signed payload with a tiny `text` but a giant `id`/`speaker`
// slips both edges, bloats the prompt, and OOMs/hangs the worker -> 24h cross-tenant
// quarantine. Bound each id/speaker string AND count it toward the same total budget.
export const MAX_SEGMENT_META_CHARS = 256;       // max chars in a segment's id or speaker label
// Đợt 32 F-R2-01: in /api/uploads/presign the jobId is interpolated into a HIERARCHICAL
// R2 object key `audio/<deviceId>/<jobId>.wav` (unlike the FLAT KV key `job:<dev>:<id>` at
// job creation, where '/' and '.' are inert). Bound its length so the full object key stays
// well under R2's 1024-byte key limit — an over-long jobId would mint a 200 for a key R2
// later rejects on PUT (No-Fake-Success). The one client ever mints `JOB-<epoch_ms>` (~17 chars).
export const MAX_JOBID_CHARS = 128;              // max chars in a presign jobId (URL-path safe)
