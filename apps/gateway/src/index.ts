import { Hono } from 'hono';
import { cors } from 'hono/cors';
import * as jose from 'jose';
import { verifySignature } from '@dichomnion/crypto-utils';
import { JobRequest, deterministicStringify } from '@dichomnion/shared-types';
import { mintJobAudioUrls, amzDate } from './r2presign';
// Edge input-bound constants live in a SEPARATE module: a Worker entrypoint may only
// export handlers/functions — a plain-value named export here fails workerd startup
// (see limits.ts). Import (do NOT re-export) so index.ts exports only default + functions.
import {
  MAX_SEGMENTS,
  MAX_SEGMENT_TEXT_CHARS,
  MAX_TOTAL_TEXT_CHARS,
  MAX_FREETEXT_CHARS,
  MAX_SEGMENT_META_CHARS,
  MAX_JOBID_CHARS,
} from './limits';
// M1 (ADR 0001) — durable job coordination. JobCoordinator is the atomic source of
// truth (per-(device,job) Durable Object); the Queue consumer drives dispatch with
// durable redelivery. Re-export the DO class at the bottom so wrangler can bind it.
import {
  JobCoordinator,
  coordinatorCreate,
  coordinatorTransition,
  coordinatorGet,
  isTerminal,
  type JobStatus,
} from './coordinator';

type Bindings = {
  KV_CACHE: KVNamespace;
  // M1 (ADR 0001) — Durable Object namespace for JobCoordinator (atomic job state)
  // and the producer handle for the durable dispatch Queue. Both are OPTIONAL on the
  // type so the many existing tests that build a KV-only env keep type-checking; the
  // /create producer path degrades gracefully when they are absent (see below).
  JOB_COORDINATOR?: DurableObjectNamespace;
  JOB_DISPATCH_QUEUE?: Queue<QueueJob>;
  // Station 2 (asymmetric): PKCS8 PEM of the gateway's ES256 (P-256) PRIVATE key.
  // Provided as a Wrangler secret. The Worker holds only the matching PUBLIC key.
  GATEWAY_JWT_PRIVATE_KEY?: string;
  WORKER_URL?: string;            // e.g. http://127.0.0.1:8000 (private worker)
  TURNSTILE_SECRET?: string;      // Cloudflare Turnstile secret key
  ALLOWED_ORIGINS?: string;       // comma-separated list of allowed client origins
  ADMIN_TOKEN?: string;           // shared secret for the Financial Kill Switch monitor
  // Station 3 timing bounds are tunable per deployment (worker tier / content length).
  MIN_PLAUSIBLE_MS?: string;      // override the "too fast => fraud" floor
  MAX_RENDER_MS?: string;         // override the hard render timeout
  // Đợt 17 F3/F4: input-size bounds (defense-in-depth mirror of the worker gate).
  MAX_SEGMENTS?: string;          // override max approved segments per job
  MAX_SEGMENT_TEXT_CHARS?: string;// override max chars for a single segment's text
  MAX_TOTAL_TEXT_CHARS?: string;  // override max chars summed across all segments
  MAX_SEGMENT_META_CHARS?: string;// Đợt 18 F6: override max chars for a segment's id/speaker
  // Đợt 17 F5: per-device job-creation throttle (bounds GPU spend / KV writes).
  JOBS_RATE_LIMIT?: string;       // override max NEW jobs per device per window
  JOBS_WINDOW_S?: string;         // override the job-creation throttle window (s)
  // Đợt 30 — R2 presign (Option A: Gateway mints a per-job upload/download URL
  // pair so audio flows client→R2→worker and NEVER through the Gateway). The two
  // *_KEY values are Wrangler secrets; the rest are plain vars.
  R2_ACCOUNT_ID?: string;         // → host `${R2_ACCOUNT_ID}.r2.cloudflarestorage.com`
  R2_BUCKET?: string;             // target R2 bucket
  R2_ACCESS_KEY_ID?: string;      // R2 S3 access key id (secret)
  R2_SECRET_ACCESS_KEY?: string;  // R2 S3 secret access key (secret)
  R2_REGION?: string;             // S3 region label; R2 uses 'auto' (default)
  R2_PRESIGN_EXPIRES_S?: string;  // presigned URL lifetime (default 7200s = 2h)
  PRESIGN_RATE_LIMIT?: string;    // override max mints per device per window
  PRESIGN_WINDOW_S?: string;      // override the mint-throttle window (s)
};

// ---- Security / anti-fraud tuning -----------------------------------------
const REPLAY_WINDOW_MS = 30_000;          // reject requests older/newer than this
const REGISTER_RATE_LIMIT = 5;            // max registrations per IP per window
const REGISTER_WINDOW_S = 3_600;          // registration rate-limit window (1h)
// Đợt 17 F5: a registered-but-untrusted device could otherwise create UNLIMITED
// (bounded) jobs — each a real GPU render + KV write — to burn spend (criterion #4).
// Registration is throttled per IP, but nothing capped jobs-per-device post-register.
// Generous default so ordinary users are never throttled; env-tunable per deployment.
const JOBS_RATE_LIMIT = 60;               // max NEW jobs per device per window
const JOBS_WINDOW_S = 3_600;              // job-creation throttle window (1h)
const JOB_TTL_S = 86_400;                 // job/result records live 24h
// Đợt 30 — R2 presign defaults. The URL must outlive upload + queue wait + the
// worker's fetch; 2h is generous yet well under JOB_TTL_S. Integrity is bound by
// the signed videoAudioMd5 the worker verifies, so a moderate window is safe.
const PRESIGN_EXPIRES_S = 7_200;          // default presigned URL lifetime (2h)
// A registered-but-untrusted device (Zero-Trust) could otherwise mint unlimited
// upload URLs and PUT unlimited R2 objects (storage Denial-of-Wallet) WITHOUT
// ever creating a job. Bound mints per device independently of JOBS_RATE_LIMIT.
const PRESIGN_RATE_LIMIT = 60;            // max mints per device per window
const PRESIGN_WINDOW_S = 3_600;           // mint-throttle window (1h)
// ETA hint for the client poller: base overhead (download + model warmup + mix)
// plus a coarse per-segment translate/TTS cost. This is an ESTIMATE surfaced in the
// 202 accept and echoed by polling — NOT a guarantee. Real timing is enforced by
// Station 3, which rejects results that come back implausibly fast or hang.
const ETA_BASE_S = 20;
const ETA_PER_SEGMENT_S = 3;
// Auto-retry: a transiently dead worker (network error or 5xx) is a blip, not a
// verdict — re-dispatch up to this many total attempts before giving up.
const MAX_DISPATCH_ATTEMPTS = 3;
// Station 3 (Anti-Fraud timing bounds). A genuine 10-minute render takes tens of
// seconds; a result returned "impossibly fast" signals a worker faking output
// (compute theft), and one that never returns signals a hung/abusive worker.
const MIN_PLAUSIBLE_MS_FLOOR = 2_000;     // absolute floor for "too fast"
const PER_SEGMENT_MIN_MS = 150;           // scale the floor with real work to do
const MAX_PLAUSIBLE_MS = 15 * 60_000;     // hard timeout -> terminate worker
const KILL_SWITCH_KEY = 'system:kill_switch';

// Edge input-bound constants (MAX_SEGMENTS, MAX_SEGMENT_TEXT_CHARS, MAX_TOTAL_TEXT_CHARS,
// MAX_FREETEXT_CHARS, MAX_SEGMENT_META_CHARS) are imported from ./limits above — see that
// module for the full Đợt 17 F3/F4 + Đợt 18 F6 rationale and why they cannot be exported here.

const app = new Hono<{ Bindings: Bindings }>();

// CORS: restrict to configured client origins (fall back to localhost dev ports).
// Never reflect an arbitrary Origin in production.
app.use('/api/*', cors({
  origin: (origin, c) => {
    const configured = (c.env.ALLOWED_ORIGINS || '')
      .split(',').map((o: string) => o.trim()).filter(Boolean);
    const allow = configured.length
      ? configured
      : ['http://localhost:1420', 'http://localhost:5173', 'tauri://localhost', 'https://tauri.localhost'];
    if (!origin) return allow[0];          // non-browser callers (no Origin header)
    return allow.includes(origin) ? origin : null;
  },
  allowHeaders: ['Content-Type', 'X-ECDSA-Signature', 'X-Device-Id'],
  allowMethods: ['POST', 'GET', 'OPTIONS'],
}));

app.get('/', (c) => c.text('OmniVoice Gateway is running! Secure Edge Active.'));

// --- helpers ---------------------------------------------------------------

/** Run a promise in the background without blocking the response. Test envs have
 *  no ExecutionContext, so fall back to a fire-and-forget with error swallowing. */
function background(c: any, p: Promise<unknown>): void {
  const guarded = Promise.resolve(p).catch(() => {
    // Zero-logging: never print payloads/urls/tokens. Status is tracked in KV.
    console.error('[dispatch] background job failed');
  });
  try {
    c.executionCtx.waitUntil(guarded);
  } catch {
    // no-op: running under vitest / direct app.request without a ctx.
  }
}

/** Verify a Cloudflare Turnstile token. Fail-CLOSED when configured; when no
 *  secret is set (local dev) verification is skipped and documented as such. */
async function verifyTurnstile(env: Bindings, token?: string): Promise<boolean> {
  const secret = env.TURNSTILE_SECRET;
  if (!secret) return true;                // dev: Turnstile not configured
  if (!token) return false;
  try {
    const form = new FormData();
    form.append('secret', secret);
    form.append('response', token);
    const r = await fetch('https://challenges.cloudflare.com/turnstile/v0/siteverify', {
      method: 'POST',
      body: form,
    });
    const data = (await r.json()) as { success?: boolean };
    return data.success === true;
  } catch {
    return false;                          // network failure -> fail closed
  }
}

async function isKillSwitchActive(env: Bindings): Promise<boolean> {
  return (await env.KV_CACHE.get(KILL_SWITCH_KEY)) === '1';
}

/** Coarse ETA (seconds) for the async job, from the number of approved segments.
 *  Surfaced in the 202 accept and persisted so polling echoes it. */
function estimateEtaSeconds(payload: JobRequest): number {
  const n = Array.isArray(payload.segments) ? payload.segments.length : 0;
  return ETA_BASE_S + n * ETA_PER_SEGMENT_S;
}

/** Đợt 19 F8 — mirror the worker's to_seconds() finiteness. The wire contract is a
 *  finite number of seconds (shared-types), but a signed device can smuggle a STRING
 *  that the worker's lenient to_seconds() normalizes to ±Infinity/NaN ("1e999", "inf",
 *  "nan"). That later detonates int(inf*1000) in mix_audio AFTER the full ASR+Qwen+TTS
 *  pipeline -> 500 -> the gateway reruns the whole dispatch up to 3×. (A JSON *number*
 *  1e999 is neutralised by the JSON round-trip — JSON.stringify(Infinity) === "null" —
 *  so the surviving vector is a string.) Return true when to_seconds would yield a
 *  FINITE value (accept), false ONLY for the inf/nan-producing inputs (reject). Junk
 *  strings ("abc") -> worker to_seconds falls back to 0.0 (finite) -> accept here too,
 *  so we never false-reject a value the worker tolerates. */
function timecodeIsFinite(v: unknown): boolean {
  if (typeof v === 'boolean') return true; // to_seconds(bool) -> 0.0
  if (typeof v === 'number') return Number.isFinite(v);
  if (typeof v !== 'string') return true; // list/object -> to_seconds -> 0.0 (finite)
  const s = v.trim().replace(/,/g, '.');
  if (s === '') return true; // falsy -> 0.0
  // Explicit non-finite tokens that Python float() accepts (case-insensitive, optional sign).
  if (/^[+-]?(inf|infinity|nan)$/i.test(s)) return false;
  // Remaining non-finite case is magnitude overflow ("1e999" -> Infinity). Split on ':'
  // so a timecode part that overflows is caught too; junk parts -> Number()=NaN (!==
  // ±Infinity) -> accept, mirroring to_seconds' per-part float() + 0.0 fallback exactly.
  return s.split(':').every((p) => {
    const n = Number(p.trim());
    return n !== Infinity && n !== -Infinity;
  });
}

/** Đợt 24 CC23-01 — faithful mirror of the worker's to_seconds() so the gateway can check
 *  the SAME arithmetic the audio sink performs, not just seconds-scale finiteness. mix_audio
 *  does int(to_seconds(start)*1000) (audio_engine.py:177) and int((end-start)*1000) (:191): a
 *  value FINITE as seconds but whose ×1000 (or whose end-start difference ×1000) overflows to
 *  ±Infinity makes int(inf) raise OverflowError → 500 → 3× retry. timecodeIsFinite only proves
 *  finiteness AT the seconds scale, so it accepts "1e306" / 1e306; this returns the numeric
 *  seconds so the caller can test the ×1000 / difference overflow the worker now also rejects.
 *  MUST NOT be stricter than the worker: junk/empty/absent → 0.0 exactly like to_seconds, so a
 *  worker-tolerated input is never false-rejected. Only meaningful once timecodeIsFinite(v) is
 *  true (inf/nan tokens & per-part overflow already rejected upstream). */
function timecodeSeconds(v: unknown): number {
  if (typeof v === 'boolean') return 0; // to_seconds(bool) -> 0.0
  if (typeof v === 'number') return v; // finite by the time this is reached
  if (typeof v !== 'string') return 0; // list/object -> 0.0
  const s = v.trim().replace(/,/g, '.');
  if (s === '') return 0; // falsy -> 0.0
  const direct = Number(s); // Python float(s) first ("1e306" -> 1e306)
  if (!Number.isNaN(direct)) return direct;
  let sec = 0; // else HH:MM:SS accumulate: sec*60 + part (mirror to_seconds)
  for (const p of s.split(':')) {
    const t = p.trim();
    if (t === '') return 0; // Python float("") raises -> to_seconds 0.0
    const n = Number(t);
    if (Number.isNaN(n)) return 0; // junk part -> to_seconds 0.0
    sec = sec * 60 + n;
  }
  return sec;
}

/** Đợt 17 F3/F4 — reject oversized job input at the edge so a crafted payload can
 *  never reach the worker's single Qwen prompt and hang the cluster. Mirrors the
 *  worker's pydantic gate (defense-in-depth): a crafted request is refused fast
 *  here even if the worker gate were bypassed/misconfigured. Returns an error
 *  string to surface as a 400, or null when the payload is within bounds. Uses
 *  optional access throughout so a malformed-but-signed body never throws. */
// Đợt 24 F13 — true iff `s` contains a LONE UTF-16 surrogate (a half of a surrogate
// pair with no valid partner, e.g. "\ud800"). Such a string is not well-formed UTF: it
// passes every typeof/length gate (still a `string`) and re-encodes to U+FFFD identically
// on both client and gateway (so ECDSA verify still matches), but on the worker the Qwen
// fast-tokenizer's str→Rust-String conversion raises UnicodeEncodeError "surrogates not
// allowed" — thrown OUTSIDE translation_service's retry try/except → HTTP 500 → this gateway
// treats 5xx as retryable → re-runs the whole pipeline (incl. Whisper ASR) up to 3× per
// malicious job. Reject pre-dispatch (400) so it never reaches the worker. Manual scan (not
// ES2024 String.isWellFormed) for runtime portability; matches exactly the set Python's
// str.encode('utf-8') / the tokenizer reject, so worker (422) and gateway (400) stay in lockstep.
function hasLoneSurrogate(s: string): boolean {
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff) {
      // high surrogate: valid ONLY if immediately followed by a low surrogate
      const next = i + 1 < s.length ? s.charCodeAt(i + 1) : 0;
      if (next < 0xdc00 || next > 0xdfff) return true;
      i++; // consume the valid low surrogate of the pair
    } else if (c >= 0xdc00 && c <= 0xdfff) {
      return true; // low surrogate with no preceding high surrogate
    }
  }
  return false;
}

function validateJobSize(env: Bindings, payload: JobRequest): string | null {
  const maxSegments = Number(env.MAX_SEGMENTS) || MAX_SEGMENTS;
  const maxSegChars = Number(env.MAX_SEGMENT_TEXT_CHARS) || MAX_SEGMENT_TEXT_CHARS;
  const maxTotalChars = Number(env.MAX_TOTAL_TEXT_CHARS) || MAX_TOTAL_TEXT_CHARS;
  const maxMetaChars = Number(env.MAX_SEGMENT_META_CHARS) || MAX_SEGMENT_META_CHARS;

  const segments = payload.segments;
  if (segments !== undefined && segments !== null) {
    if (!Array.isArray(segments)) return 'segments must be an array';
    if (segments.length > maxSegments) return `too many segments (>${maxSegments})`;
    let total = 0;
    for (const seg of segments) {
      // F8b/F9: each segment MUST be an object. A primitive/array element (["x"],[1],[null])
      // would pass every field check below (all `(seg as any)?.field` -> undefined) yet the
      // worker's translate_segments calls seg.get(...) on it -> AttributeError -> 500 -> 3×
      // retry. Reject non-objects here so the worker never sees them (mirror worker 422).
      if (typeof seg !== 'object' || seg === null || Array.isArray(seg)) {
        return 'segment must be an object';
      }
      // Worker reads seg.text (falling back to original_text); mirror that so the
      // edge bound measures the SAME field that grows the worker's prompt.
      // Đợt 22 F12: `??` cũ nuốt `text:null`→'' rồi typeof-string PASS, nhưng worker consumer
      // (translation_service.py:265 `.get("text", .get("original_text",""))`) đọc None -> _merge
      // dựng TranslatedSegment(original_text=None) [str bắt buộc] -> ValidationError -> 500 -> retry×3.
      // Phản chiếu CHÍNH XÁC .get của Python: key HIỆN DIỆN (kể cả null) dùng value đó (KHÔNG fallback),
      // key VẮNG mới fallback; `undefined` = key vắng sau JSON.parse, `null` = key hiện diện null.
      const rawText = (seg as any)?.text;
      const rawOrig = (seg as any)?.original_text;
      const text = rawText !== undefined ? rawText : (rawOrig !== undefined ? rawOrig : '');
      if (typeof text !== 'string') return 'segment text must be a string';
      if (text.length > maxSegChars) return `segment text too long (>${maxSegChars} chars)`;
      if (hasLoneSurrogate(text)) return 'segment text is not well-formed UTF';  // Đợt 24 F13
      total += text.length;
      // F6: the worker also embeds each segment's id + speaker/speaker_id into the prompt.
      // Bound each string and fold into the same total (mirrors worker _bound_segments).
      // F11 (Đợt 21): length alone is NOT enough — id/speaker also flow VERBATIM into the
      // worker's translation_service `_merge`, which builds TranslatedSegment (id: int|str,
      // speaker_id: str) OUTSIDE its retry try/except. A wrong-TYPE value (object/array/null/
      // boolean) passes the old length-only gate, then detonates as an uncaught pydantic
      // ValidationError → HTTP 500 → this gateway retries the whole pipeline 3×. Enforce the
      // TYPE here pre-dispatch (mirror F9/F10). Accept string|number for id (worker backstops a
      // fractional-number id with a clean 422); speaker must be a string.
      const segId = (seg as any)?.id;
      if (segId !== undefined && segId !== null && typeof segId !== 'string' && typeof segId !== 'number') {
        return 'segment id must be a string or number';
      }
      if (typeof segId === 'string') {
        if (segId.length > maxMetaChars) return `segment id too long (>${maxMetaChars} chars)`;
        if (hasLoneSurrogate(segId)) return 'segment id is not well-formed UTF';  // Đợt 24 F13
        total += segId.length;
      }
      const speaker = (seg as any)?.speaker ?? (seg as any)?.speaker_id;
      if (speaker !== undefined && speaker !== null && typeof speaker !== 'string') {
        return 'segment speaker must be a string';
      }
      if (typeof speaker === 'string') {
        if (speaker.length > maxMetaChars) return `segment speaker too long (>${maxMetaChars} chars)`;
        if (hasLoneSurrogate(speaker)) return 'segment speaker is not well-formed UTF';  // Đợt 24 F13
        total += speaker.length;
      }
      // F8: start/end/duration must normalise to FINITE seconds (mirrors worker gate).
      // Absent fields are fine (worker defaults them); only a present-but-non-finite
      // value is rejected — the vector that detonates int(inf*1000) / round(inf) later.
      for (const f of ['start', 'end'] as const) {
        const tv = (seg as any)?.[f];
        if (tv === undefined || tv === null) continue;
        if (!timecodeIsFinite(tv)) return `segment ${f} must be a finite number`;
      }
      // Đợt 24 CC23-01 — COMPLETE F8: the loop above proves each timecode is finite AS SECONDS,
      // but the sink is int(to_seconds(start)*1000) / int((end-start)*1000). A finite-as-seconds
      // value whose ×1000 (e.g. 1e306) — or whose end-start difference ×1000 (start=-1e305,
      // end=1e305 → 2e308) — overflows to inf makes int(inf) raise OverflowError → 500 → 3× retry.
      // Test the exact products the sink computes (absent → 0, no false-reject). Mirrors the
      // worker's extended _bound_segments gate; NOT a new axis — same sink, completing F8.
      const s0 = timecodeSeconds((seg as any)?.start);
      const e0 = timecodeSeconds((seg as any)?.end);
      if (!Number.isFinite(s0 * 1000) || !Number.isFinite(e0 * 1000) || !Number.isFinite((e0 - s0) * 1000)) {
        return 'segment start/end too large (×1000 overflows)';
      }
      const dur = (seg as any)?.duration;
      if (dur !== undefined && dur !== null && (typeof dur !== 'number' || !Number.isFinite(dur))) {
        return 'segment duration must be a finite number';
      }
    }
    if (total > maxTotalChars) return `total segment text too long (>${maxTotalChars} chars)`;
  }

  // Free-text fields are also interpolated into the prompt — bound them too.
  const cfg = payload.config as any;
  for (const name of ['targetLanguage', 'translationStyle', 'sourceLanguage'] as const) {
    const val = cfg?.[name];
    if (val === undefined || val === null) continue;
    if (typeof val !== 'string') return `${name} must be a string`;
    if (val.length > MAX_FREETEXT_CHARS) return `${name} too long (>${MAX_FREETEXT_CHARS} chars)`;
    if (hasLoneSurrogate(val)) return `${name} is not well-formed UTF`;  // Đợt 24 F13
  }

  // F10: speakerMapping is forwarded verbatim as the worker's voice_map, whose values reach
  // _resolve_voice -> VOICE_ID_GENDER.get(value) (crashes on a non-hashable list/dict) or
  // voice.startswith(...) (crashes on a non-string). The wire type Record<string,string> is
  // compile-time only; enforce value TYPE at runtime so a crafted map can't crash-and-retry
  // the worker. (TYPE only — the map's ENTRY COUNT is not a bloat axis; that was rejected in
  // Đợt 19 because _resolve_voice only looks up per-speaker_id, already bounded by segments.)
  const vmap = (payload as any).speakerMapping;
  if (vmap !== undefined && vmap !== null) {
    if (typeof vmap !== 'object' || Array.isArray(vmap)) return 'speakerMapping must be an object';
    for (const val of Object.values(vmap)) {
      if (typeof val !== 'string') return 'speakerMapping values must be strings';
    }
  }
  return null;
}

/** Constant-time secret comparison for the admin token. Both sides are hashed to
 *  fixed-length SHA-256 digests first, so (a) the compare length never varies with
 *  the secret and (b) a byte mismatch reveals nothing about the real token (SHA-256
 *  is preimage-resistant). A plain `a !== b` on the raw strings short-circuits at the
 *  first differing byte — a timing oracle that leaks the token one byte at a time. */
async function safeTokenEqual(a: string, b: string): Promise<boolean> {
  const enc = new TextEncoder();
  const [ha, hb] = await Promise.all([
    crypto.subtle.digest('SHA-256', enc.encode(a)),
    crypto.subtle.digest('SHA-256', enc.encode(b)),
  ]);
  const va = new Uint8Array(ha);
  const vb = new Uint8Array(hb);
  let diff = 0;
  for (let i = 0; i < va.length; i++) diff |= va[i] ^ vb[i];
  return diff === 0;
}

// --- Financial Kill Switch admin endpoint ----------------------------------
// The standalone billing monitor (scripts/kill-switch-monitor.mjs) flips this
// flag when spend crosses the threshold; every job/register call then 503s.
app.post('/api/admin/kill-switch', async (c) => {
  const token = c.req.header('X-Admin-Token');
  // Fail-closed: unset server token or missing header => reject BEFORE any compare
  // (a missing header is not a secret-dependent branch — the caller already knows it).
  if (!c.env.ADMIN_TOKEN || !token || !(await safeTokenEqual(token, c.env.ADMIN_TOKEN))) {
    return c.json({ error: 'Forbidden' }, 403);
  }
  // Fail-CLOSED on a malformed request: a garbled/empty body must NEVER be read as
  // "clear the switch". Previously a parse failure fell to `{}` (active !== true) and
  // silently DISARMED the financial kill switch — a mangled re-arm call would turn off
  // the spend guard. Require an explicit boolean `active`; anything else is a 400 that
  // leaves the current switch state untouched.
  const body = (await c.req.json().catch(() => null)) as { active?: unknown } | null;
  if (!body || typeof body.active !== 'boolean') {
    return c.json({ error: 'Missing or invalid "active" flag (boolean required)' }, 400);
  }
  if (body.active === true) {
    await c.env.KV_CACHE.put(KILL_SWITCH_KEY, '1');
    return c.json({ killSwitch: 'ACTIVE' });
  }
  await c.env.KV_CACHE.delete(KILL_SWITCH_KEY);
  return c.json({ killSwitch: 'CLEARED' });
});

// --- Device registration (Station 1 enrolment) -----------------------------
app.post('/api/auth/register', async (c) => {
  if (await isKillSwitchActive(c.env)) {
    return c.json({ error: 'Service temporarily unavailable' }, 503);
  }

  const body = await c.req.json().catch(() => null);
  if (!body || !body.publicKeyJwk) {
    return c.json({ error: 'Missing Public Key' }, 400);
  }

  // Bot protection (Turnstile) before we spend any KV writes.
  if (!(await verifyTurnstile(c.env, body.turnstileToken))) {
    return c.json({ error: 'Bot verification failed' }, 403);
  }

  // Registration throttle per client IP.
  const ip = c.req.header('CF-Connecting-IP') || 'unknown';
  const rlKey = `rl:register:${ip}`;
  const count = parseInt((await c.env.KV_CACHE.get(rlKey)) || '0', 10);
  if (count >= REGISTER_RATE_LIMIT) {
    return c.json({ error: 'Too Many Registrations. Please try again later.' }, 429);
  }
  await c.env.KV_CACHE.put(rlKey, String(count + 1), { expirationTtl: REGISTER_WINDOW_S });

  const deviceId = crypto.randomUUID();
  await c.env.KV_CACHE.put(`device:${deviceId}`, JSON.stringify(body.publicKeyJwk));

  return c.json({ deviceId, message: 'Device Registered Successfully' }, 201);
});

// --- Job creation (Station 1 verification + async accept) ------------------
app.post('/api/jobs/create', async (c) => {
  if (await isKillSwitchActive(c.env)) {
    return c.json({ error: 'Service temporarily unavailable (Kill Switch active)' }, 503);
  }

  const signature = c.req.header('X-ECDSA-Signature');
  const deviceId = c.req.header('X-Device-Id');
  if (!signature || !deviceId) {
    return c.json({ error: 'Missing Zero-Trust Signature or Device ID' }, 401);
  }

  // Never trust the header key — look the public key up in the registry.
  const publicKeyJwk = await c.env.KV_CACHE.get<JsonWebKey>(`device:${deviceId}`, { type: 'json' });
  if (!publicKeyJwk) {
    return c.json({ error: 'Unauthorized Device. Public Key not found in Registry.' }, 401);
  }

  const rawBody = await c.req.text();
  let payloadObj: JobRequest;
  try {
    payloadObj = JSON.parse(rawBody);
  } catch {
    return c.json({ error: 'Invalid JSON' }, 400);
  }

  // Verify the ECDSA signature over the deterministic serialization.
  const payloadStr = deterministicStringify(payloadObj);
  const isValid = await verifySignature(payloadStr, signature, publicKeyJwk);
  if (!isValid) {
    return c.json({ error: 'Tampering Detected. Signature Invalid.' }, 403);
  }

  // Replay protection — NaN-safe: a missing/non-numeric timestamp is rejected,
  // and future-dated timestamps are treated the same as expired ones.
  const ts = payloadObj.timestamp;
  if (typeof ts !== 'number' || !Number.isFinite(ts) || Math.abs(Date.now() - ts) > REPLAY_WINDOW_MS) {
    return c.json({ error: 'Request Expired. Replay attack prevented.' }, 403);
  }

  if (!payloadObj.jobId || typeof payloadObj.jobId !== 'string') {
    return c.json({ error: 'Missing jobId' }, 400);
  }
  // Đợt 25 AMP-JOBID-SURROGATE-01 — jobId is the one client-controlled string F13 (Đợt 24) left
  // ungated at BOTH tiers. A lone surrogate ("job-\ud800") passes the type-check above, is signed
  // (deterministicStringify escapes it to \udXXX identically both sides) and forwarded VERBATIM
  // into both the ES256 JWT `jobId` claim and the worker request body job_id (JobPayload.job_id is
  // a bare `str`). It survives token-binding, runs the FULL pipeline, then detonates when the worker
  // renders its JSON reply: `return {"job_id": "job-\ud800", ...}` -> json.dumps(...).encode('utf-8')
  // raises UnicodeEncodeError AFTER the handler returns (outside its try/except) = an uncaught 500,
  // which this gateway treats as retryable (res.status>=500) and re-dispatches 3×, re-running the
  // whole GPU pipeline (post-render crash = maximal Denial-of-Wallet). Reject pre-dispatch (400
  // terminal) mirroring the worker's job_id gate (422). DIFFERENT field + DIFFERENT sink than F13
  // (response serializer, not the Qwen tokenizer) — a genuinely new (field, sink), not a re-fix.
  if (hasLoneSurrogate(payloadObj.jobId)) {
    return c.json({ error: 'jobId is not well-formed UTF' }, 400);
  }

  // Đợt 17 F3/F4: bound input size at the edge BEFORE any expensive work (idempotency
  // KV read / dispatch). A crafted oversized payload from an untrusted-but-registered
  // device is refused fast here so it can never hang the worker and quarantine it.
  const sizeError = validateJobSize(c.env, payloadObj);
  if (sizeError) {
    return c.json({ error: `Job input too large: ${sizeError}` }, 400);
  }

  const jobKey = `job:${deviceId}:${payloadObj.jobId}`;
  const etaSeconds = estimateEtaSeconds(payloadObj);

  // Đợt 17 F5 per-device job-creation throttle config (shared by both dispatch
  // paths). Bounds a registered-but-untrusted device from spinning up unlimited
  // real GPU renders. The quota is consumed only for GENUINELY NEW jobIds — a
  // client retrying the SAME job (returned below as idempotent) never counts.
  const jobsLimit = Number(c.env.JOBS_RATE_LIMIT) || JOBS_RATE_LIMIT;
  const jobsWindow = Number(c.env.JOBS_WINDOW_S) || JOBS_WINDOW_S;
  const jobsRlKey = `rl:jobs:${deviceId}`;

  // ── M1 (ADR 0001): DURABLE dispatch path ─────────────────────────────────
  // When the JobCoordinator DO + dispatch Queue are bound (production, wired in
  // wrangler.toml), the DO is the ATOMIC source of truth and the Queue is the
  // durable hand-off. This replaces the fragile `202 + waitUntil` dispatch:
  //   • W1 — an evicted Worker can no longer silently drop the job (the Queue
  //          redelivers to the consumer instead of a fire-and-forget waitUntil);
  //   • W3 — DO.create is an atomic check-and-set, ending the KV get-then-put
  //          TOCTOU that could double-dispatch a re-sent jobId.
  // Absent the bindings we fall through to the legacy KV + background path below.
  if (c.env.JOB_COORDINATOR && c.env.JOB_DISPATCH_QUEUE) {
    const stub = coordinatorStub(c.env.JOB_COORDINATOR, deviceId, payloadObj.jobId);

    // Repeat? A re-sent job returns its CURRENT state (read-only) and consumes no
    // quota — mirroring the legacy "idempotency before throttle" ordering.
    const seen = await coordinatorGet(stub);
    if (seen) {
      return c.json(
        {
          message: 'Job Accepted securely!',
          jobId: payloadObj.jobId,
          status: seen.status,
          etaSeconds: seen.etaSeconds,
          idempotent: true,
        },
        202,
      );
    }

    // Genuinely new → throttle (CHECK only; the quota is consumed AFTER we confirm
    // WE created the record, so a lost race neither enqueues nor charges quota).
    const jobsCount = parseInt((await c.env.KV_CACHE.get(jobsRlKey)) || '0', 10);
    if (jobsCount >= jobsLimit) {
      return c.json({ error: 'Too Many Jobs. Rate limit exceeded; please try again later.' }, 429);
    }

    // Atomic check-and-set (W3 fix). Also seeds the KV `job:` projection to QUEUED
    // so the existing poll/download contract keeps reading KV unchanged.
    const created = await coordinatorCreate(stub, { deviceId, jobId: payloadObj.jobId, etaSeconds });
    if (created.idempotent) {
      // Lost a concurrent race for the same jobId — the winner enqueued & charged
      // quota; we must not double-dispatch (no double GPU render).
      return c.json(
        {
          message: 'Job Accepted securely!',
          jobId: payloadObj.jobId,
          status: created.record.status,
          etaSeconds: created.record.etaSeconds,
          idempotent: true,
        },
        202,
      );
    }

    await c.env.KV_CACHE.put(jobsRlKey, String(jobsCount + 1), { expirationTtl: jobsWindow });
    // Durable hand-off: the Queue consumer (handleJobQueue) drives dispatch with
    // at-least-once redelivery, so an evicted Worker can never silently lose the job.
    await c.env.JOB_DISPATCH_QUEUE.send({ deviceId, jobId: payloadObj.jobId, payload: payloadObj });
    return c.json(
      { message: 'Job Accepted securely!', jobId: payloadObj.jobId, status: 'QUEUED', etaSeconds },
      202,
    );
  }

  // ── Legacy path (no durable bindings): KV idempotency + background dispatch ──
  // A real, still-functional fallback (the pre-M1 behavior). Production binds the
  // DO + Queue, so this runs only in KV-only environments (e.g. the existing unit
  // tests). Idempotency: a re-sent job returns the existing record, never re-dispatches.
  const existing = await c.env.KV_CACHE.get<{ status: string; etaSeconds?: number }>(jobKey, { type: 'json' });
  if (existing) {
    return c.json(
      {
        message: 'Job Accepted securely!',
        jobId: payloadObj.jobId,
        status: existing.status,
        etaSeconds: existing.etaSeconds,
        idempotent: true,
      },
      202,
    );
  }

  const jobsCount = parseInt((await c.env.KV_CACHE.get(jobsRlKey)) || '0', 10);
  if (jobsCount >= jobsLimit) {
    return c.json({ error: 'Too Many Jobs. Rate limit exceeded; please try again later.' }, 429);
  }
  await c.env.KV_CACHE.put(jobsRlKey, String(jobsCount + 1), { expirationTtl: jobsWindow });

  await c.env.KV_CACHE.put(
    jobKey,
    JSON.stringify({ status: 'QUEUED', createdAt: Date.now(), etaSeconds }),
    { expirationTtl: JOB_TTL_S },
  );

  // Dispatch to the GPU worker asynchronously (Graceful Degradation: accept now,
  // process later). We never block the client on GPU time.
  background(c, dispatchToWorker(c.env, payloadObj, jobKey));

  return c.json(
    { message: 'Job Accepted securely!', jobId: payloadObj.jobId, status: 'QUEUED', etaSeconds },
    202,
  );
});

// --- Per-job R2 upload URL minting (Đợt 30 — Option A) ----------------------
// The client asks the Gateway for a short-lived presigned PUT (to upload the
// extracted audio straight to R2) plus a presigned GET (for the GPU worker to
// fetch it back). The audio bytes NEVER traverse the Gateway (Zero-Logging /
// Zero-Trust). Auth mirrors /api/jobs/create exactly: only a registered device
// with a valid ECDSA signature over { jobId, timestamp } may mint, and the
// object key is namespaced by (device, job) so a minted URL can only ever
// address its own job's object — never another device's or another job's.
app.post('/api/uploads/presign', async (c) => {
  if (await isKillSwitchActive(c.env)) {
    return c.json({ error: 'Service temporarily unavailable (Kill Switch active)' }, 503);
  }

  const signature = c.req.header('X-ECDSA-Signature');
  const deviceId = c.req.header('X-Device-Id');
  if (!signature || !deviceId) {
    return c.json({ error: 'Missing Zero-Trust Signature or Device ID' }, 401);
  }

  // Never trust the header key — look the public key up in the registry.
  const publicKeyJwk = await c.env.KV_CACHE.get<JsonWebKey>(`device:${deviceId}`, { type: 'json' });
  if (!publicKeyJwk) {
    return c.json({ error: 'Unauthorized Device. Public Key not found in Registry.' }, 401);
  }

  const rawBody = await c.req.text();
  let reqObj: { jobId?: unknown; timestamp?: unknown };
  try {
    reqObj = JSON.parse(rawBody);
  } catch {
    return c.json({ error: 'Invalid JSON' }, 400);
  }

  // Verify the ECDSA signature over the deterministic serialization.
  const payloadStr = deterministicStringify(reqObj as Record<string, unknown>);
  const isValid = await verifySignature(payloadStr, signature, publicKeyJwk);
  if (!isValid) {
    return c.json({ error: 'Tampering Detected. Signature Invalid.' }, 403);
  }

  // Replay protection — identical NaN-safe window to job creation.
  const ts = reqObj.timestamp;
  if (typeof ts !== 'number' || !Number.isFinite(ts) || Math.abs(Date.now() - ts) > REPLAY_WINDOW_MS) {
    return c.json({ error: 'Request Expired. Replay attack prevented.' }, 403);
  }

  const jobId = reqObj.jobId;
  if (!jobId || typeof jobId !== 'string') {
    return c.json({ error: 'Missing jobId' }, 400);
  }
  // Đợt 32 F-R2-01 — UNLIKE job creation (where jobId is a FLAT KV key `job:<dev>:<id>`,
  // so '/' and '.' are inert), here jobId is interpolated into a HIERARCHICAL R2 object key
  // `audio/<deviceId>/<jobId>.wav` and thence into the SigV4 canonical URI path (r2presign
  // keeps '/' literal and treats '.' as unreserved). A '/' reshapes the key — silently
  // breaking the one-key-per-job invariant asserted above — and a '/../' or '/./' dot-segment
  // survives into the SIGNED path yet the WHATWG URL parser normalises it away on the wire, so
  // R2 answers SignatureDoesNotMatch even though we already returned 200 (No-Fake-Success).
  // Bound jobId to a URL-path-safe allowlist; the only client ever mints `JOB-<epoch_ms>`.
  // This allowlist strictly subsumes the lone-surrogate gate used at job creation.
  if (!/^[A-Za-z0-9_-]+$/.test(jobId) || jobId.length > MAX_JOBID_CHARS) {
    return c.json({ error: 'jobId has invalid characters' }, 400);
  }

  // Fail closed if R2 is not fully provisioned — never hand back a URL that
  // points nowhere (No-Fake-Success). All four fields are required to sign.
  const { R2_ACCOUNT_ID, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY } = c.env;
  if (!R2_ACCOUNT_ID || !R2_BUCKET || !R2_ACCESS_KEY_ID || !R2_SECRET_ACCESS_KEY) {
    return c.json({ error: 'Upload storage not provisioned' }, 503);
  }

  // Per-device mint throttle (bounds R2 object creation independently of the
  // job-creation throttle). Checked AFTER provisioning so a misconfigured
  // deployment never silently burns a device's quota.
  const presignLimit = Number(c.env.PRESIGN_RATE_LIMIT) || PRESIGN_RATE_LIMIT;
  const presignWindow = Number(c.env.PRESIGN_WINDOW_S) || PRESIGN_WINDOW_S;
  const rlKey = `rl:presign:${deviceId}`;
  const rlCount = parseInt((await c.env.KV_CACHE.get(rlKey)) || '0', 10);
  if (rlCount >= presignLimit) {
    return c.json({ error: 'Too Many Upload Requests. Rate limit exceeded; please try again later.' }, 429);
  }
  await c.env.KV_CACHE.put(rlKey, String(rlCount + 1), { expirationTtl: presignWindow });

  const expiresSeconds = Number(c.env.R2_PRESIGN_EXPIRES_S) || PRESIGN_EXPIRES_S;

  try {
    const urls = await mintJobAudioUrls({
      config: {
        accountId: R2_ACCOUNT_ID,
        bucket: R2_BUCKET,
        accessKeyId: R2_ACCESS_KEY_ID,
        secretAccessKey: R2_SECRET_ACCESS_KEY,
        region: c.env.R2_REGION,
      },
      deviceId,
      jobId,
      amzDate: amzDate(new Date()),
      expiresSeconds,
    });
    return c.json(urls, 200);
  } catch {
    // Only reached on misconfigured expiry / signing failure — fail closed
    // without leaking internals (Zero-Logging).
    return c.json({ error: 'Upload storage misconfigured' }, 503);
  }
});

// --- Job status polling (client-facing) ------------------------------------
app.get('/api/jobs/:jobId', async (c) => {
  const deviceId = c.req.header('X-Device-Id');
  if (!deviceId) return c.json({ error: 'Missing Device ID' }, 401);
  const jobId = c.req.param('jobId');
  const record = (await c.env.KV_CACHE.get(`job:${deviceId}:${jobId}`, { type: 'json' })) as any;
  if (!record) return c.json({ error: 'Job not found' }, 404);

  // When finished, surface the worker's HONEST result metadata (pipeline actually
  // run, distinct voices, watermark/separation flags) so the client can display it
  // and knows the artifact is retrievable. We deliberately do NOT leak the worker's
  // internal temp path (dubbed_audio) — the client fetches the bytes through the
  // authenticated gateway proxy (/download) instead. Result is DEVICE-SCOPED so one
  // device can never read another's output by guessing a jobId.
  if (record.status === 'DONE') {
    const stored = (await c.env.KV_CACHE.get(`result:${deviceId}:${jobId}`, { type: 'json' })) as any;
    // Belt-and-suspenders for KV propagation skew across PoPs: if the terminal status
    // is visible but the result key has not yet propagated to this edge, report a
    // NON-terminal status so the client keeps polling instead of latching a DONE it
    // cannot fulfil. The dispatch path commits the result BEFORE DONE, so this only
    // fires on cross-PoP eventual-consistency lag, never on a genuinely finished job.
    if (!stored) {
      return c.json({ jobId, ...record, status: 'FINALIZING' });
    }
    const inner = stored.result ?? {};
    // Allowlist RÕ RÀNG các trường trả cho client — KHÔNG denylist "trừ dubbed_audio"
    // (denylist bỏ sót nếu kho lưu thêm trường nhạy cảm mới). Tuyệt đối không trả
    // dubbed_audio (đường dẫn temp nội bộ) — client tải bytes qua proxy /download.
    return c.json({
      jobId,
      ...record,
      result: {
        status: inner.status,
        message: inner.message,
        device_used: inner.device_used,
        pipeline: inner.pipeline,
        separated: inner.separated,
        watermarked: inner.watermarked,
        distinct_voices: inner.distinct_voices,
        notes: inner.notes,
        artifactReady: Boolean(inner.dubbed_audio),
        downloadUrl: inner.dubbed_audio ? `/api/jobs/${jobId}/download` : undefined,
      },
    });
  }
  return c.json({ jobId, ...record });
});

// --- Authenticated result download (client-facing) -------------------------
// Closes the async loop: a client that polled DONE retrieves the dubbed AUDIO
// here. The gateway proxies the worker's JWT-protected file endpoint; the raw
// worker temp path is never exposed to the client, and the lookup is DEVICE-SCOPED.
app.get('/api/jobs/:jobId/download', async (c) => {
  const deviceId = c.req.header('X-Device-Id');
  if (!deviceId) return c.json({ error: 'Missing Device ID' }, 401);
  const jobId = c.req.param('jobId');

  const record = (await c.env.KV_CACHE.get(`job:${deviceId}:${jobId}`, { type: 'json' })) as any;
  if (!record) return c.json({ error: 'Job not found' }, 404);
  if (record.status !== 'DONE') return c.json({ error: 'Job not ready', status: record.status }, 409);

  const stored = (await c.env.KV_CACHE.get(`result:${deviceId}:${jobId}`, { type: 'json' })) as any;
  const path = stored?.result?.dubbed_audio;
  if (!path) return c.json({ error: 'Result artifact unavailable' }, 404);

  // Station 2: prove gateway identity to the worker with the asymmetric ES256 JWT.
  const jwt = await signGatewayJwt(c.env, jobId);
  if (!jwt) return c.json({ error: 'Gateway not provisioned to fetch result' }, 503);

  const workerUrl = (c.env.WORKER_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
  // Bound only the CONNECT/headers phase: a hung worker must not stall the client's
  // download forever, but once headers arrive we clear the timer so a large audio
  // body streams to completion without being aborted mid-transfer.
  const controller = new AbortController();
  const connectTimeout = setTimeout(() => controller.abort(), 15_000);
  let res: Response;
  try {
    res = await fetch(`${workerUrl}/api/worker/download?path=${encodeURIComponent(path)}`, {
      headers: { Authorization: `Bearer ${jwt}` },
      signal: controller.signal,
    });
  } catch {
    return c.json({ error: 'Worker unreachable' }, 502);
  } finally {
    clearTimeout(connectTimeout);
  }
  if (!res.ok) return c.json({ error: 'Worker download failed', code: res.status }, 502);

  // Stream the audio straight back to the owning client.
  return new Response(res.body, {
    status: 200,
    headers: {
      'Content-Type': res.headers.get('Content-Type') || 'audio/wav',
      'Content-Disposition': `attachment; filename="dubbed-${jobId}.wav"`,
    },
  });
});

// --- Station 2 + Station 3 dispatch -----------------------------------------
// Exported for direct testing of the anti-fraud timing enforcement.
export async function dispatchToWorker(env: Bindings, payload: JobRequest, jobKey: string): Promise<void> {
  const workerUrl = (env.WORKER_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

  // Station 3: never dispatch to a worker previously QUARANTINED for anomaly
  // (fabricated/too-fast result or hang). A known-bad instance must not receive
  // new jobs; re-queueing to a FRESH instance is the infra worker-pool's job
  // (residual_hardware G3). Faking a redispatch to the same bad URL is dishonest.
  if (await env.KV_CACHE.get(`quarantine:${workerUrl}`)) {
    await setJob(env, jobKey, { status: 'FAILED', reason: 'worker_quarantined' });
    console.error('[station3] refused dispatch to quarantined worker');
    return;
  }

  // Station 2: prove identity to the Worker with an ASYMMETRIC ES256 JWT.
  // The Worker verifies with the matching public key — no shared secret exists,
  // so a leaked worker image cannot forge gateway authority.
  // Sign once here purely as a FAIL-CLOSED GATE: if the signing key is missing/invalid
  // we refuse to touch the worker at all (and report which). The token actually SENT is
  // re-minted per attempt inside the loop so a slow render / retry never ships a token
  // that has already crossed its short 2m exp.
  const gate = await signGatewayJwt(env, payload.jobId);
  if (!gate) {
    const reason = env.GATEWAY_JWT_PRIVATE_KEY ? 'gateway_key_invalid' : 'gateway_key_missing';
    await setJob(env, jobKey, { status: 'FAILED', reason });
    console.error(`[station2] cannot dispatch job: ${reason}`);
    return;
  }

  // Station 3: bound the round-trip. Too fast => faked result; too slow => hung.
  // Bounds default to the tuned constants but can be overridden per deployment.
  const floor = Number(env.MIN_PLAUSIBLE_MS) || MIN_PLAUSIBLE_MS_FLOOR;
  const maxMs = Number(env.MAX_RENDER_MS) || MAX_PLAUSIBLE_MS;
  const rawFloor = Math.max(
    floor,
    (payload.segments?.length || 0) * PER_SEGMENT_MIN_MS,
  );
  // Structural guarantee: the "too fast => fraud" floor must never reach the hang
  // timeout, or the plausible window [floor, maxMs] inverts and EVERY honest result
  // is rejected as fraud (or times out). Input bounds keep segments*PER_SEGMENT_MIN_MS
  // well under maxMs by default (2000*150ms = 5m vs 15m); this cap makes the invariant
  // hold even under env MISCONFIGURATION (huge MAX_SEGMENTS + tiny MAX_RENDER_MS). It
  // does NOT fix the DoS — validateJobSize does; this keeps fraud detection self-consistent.
  const minPlausibleMs = Math.min(rawFloor, Math.floor(maxMs / 2));
  // Retry policy (Graceful Degradation without ever trusting a bad result):
  //   • network error / 5xx  -> transient worker blip, RE-DISPATCH (worker still alive)
  //   • 4xx                  -> hard client/auth error, TERMINAL (retry is pointless)
  //   • REJECTED_FRAUD       -> "impossibly fast" result, TERMINAL (never re-trust)
  //   • TIMEOUT              -> hung worker, terminate+quarantine, TERMINAL. We do NOT
  //     re-dispatch to the SAME now-quarantined URL; re-queueing to a FRESH instance
  //     belongs to the infra worker-pool (residual_hardware G3), so faking it here
  //     would be dishonest.
  for (let attempt = 1; attempt <= MAX_DISPATCH_ATTEMPTS; attempt++) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), maxMs);

    try {
      // Re-mint the JWT for EACH attempt: the token carries a short 2m exp, so a slow
      // render or a retry after backoff can outlive a token signed once before the loop,
      // turning an otherwise-retryable worker into a 401 hard-auth failure. Re-signing
      // keeps every attempt's credential fresh (mirrors terminateWorker below).
      const jwt = await signGatewayJwt(env, payload.jobId);
      if (!jwt) {
        // Key validated at the pre-loop gate; a null here is a transient signing failure
        // — fail closed rather than dispatch to the worker unauthenticated.
        await setJob(env, jobKey, { status: 'FAILED', reason: 'gateway_key_invalid', attempts: attempt });
        return;
      }
      const startTime = Date.now();
      const res = await fetch(`${workerUrl}/api/worker/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${jwt}` },
        body: JSON.stringify({
          job_id: payload.jobId,
          audio_url: payload.videoAudioUrl,
          // Chuyển tiếp md5 ĐÃ KÝ của bytes audio (ràng buộc toàn vẹn). Object key R2 dùng
          // chung cho mọi job nên worker phải tự kiểm bytes tải-về khớp md5 client đã ký,
          // fail-closed khi lệch (chống tải nhầm/rò rỉ audio chéo tenant — xem worker).
          audio_md5: payload.videoAudioMd5,
          target_language: payload.config.targetLanguage,
          translation_style: payload.config.translationStyle,
          // Chuyển tiếp ánh xạ ĐA GIỌNG người dùng gán ở client (speaker_id -> voice).
          // Bản cũ bỏ rơi speakerMapping ở đây nên đa giọng không bao giờ tới worker.
          voice_map: payload.speakerMapping || {},
          source_language: payload.config.sourceLanguage,
          segments: payload.segments || [],
        }),
        signal: controller.signal,
      });

      const elapsed = Date.now() - startTime;

      if (!res.ok) {
        // 5xx: transient -> retry if attempts remain. 4xx: hard error -> stop.
        if (res.status >= 500 && attempt < MAX_DISPATCH_ATTEMPTS) {
          await setJob(env, jobKey, { status: 'RETRYING', attempt, code: res.status });
          continue;
        }
        await setJob(env, jobKey, { status: 'FAILED', code: res.status, attempts: attempt });
        return;
      }

      // Anti-Fraud: an implausibly fast success is a fabricated result. Terminal.
      if (elapsed < minPlausibleMs) {
        await terminateWorker(env, workerUrl, payload.jobId, 'anomaly_too_fast');
        await setJob(env, jobKey, { status: 'REJECTED_FRAUD', elapsed, attempts: attempt });
        console.error('[station3] rejected worker result: impossibly fast');
        return;
      }

      const workerResponse = (await res.json()) as any;
      const raw = workerResponse?.result ?? {};
      // Zero-Logging AT REST: allowlist các trường TRƯỚC khi lưu KV 24h. Chỉ giữ metadata
      // an toàn + dubbed_audio (đường dẫn nội bộ để /download proxy file — KHÔNG bao giờ
      // gửi cho client). Loại mọi kịch bản/bản dịch plaintext để không lưu trữ nội dung
      // nhạy cảm; dùng allowlist (không denylist) nên trường nhạy cảm mới cũng không lọt.
      const storedResult = {
        status: raw.status,
        message: raw.message,
        device_used: raw.device_used,
        pipeline: raw.pipeline,
        separated: raw.separated,
        watermarked: raw.watermarked,
        distinct_voices: raw.distinct_voices,
        notes: raw.notes,
        dubbed_audio: raw.dubbed_audio,
      };
      // DEVICE-SCOPED result key (derived from jobKey `job:<device>:<jobId>`) so the
      // stored output is only retrievable by the owning device, not by jobId guess.
      // Commit the ARTIFACT *before* flipping status to DONE: any client that observes
      // DONE must be able to read the result. The reverse order left an in-request
      // window where DONE was visible but the result key was not yet written, so the
      // client latched a terminal DONE with no artifact (a dead end). The status
      // read-side is belt-and-suspenders for residual cross-PoP KV propagation skew.
      const resultKey = jobKey.replace(/^job:/, 'result:');
      await env.KV_CACHE.put(
        resultKey,
        JSON.stringify({ job_id: workerResponse?.job_id, result: storedResult }),
        { expirationTtl: JOB_TTL_S },
      );
      await setJob(env, jobKey, { status: 'DONE', elapsed, attempts: attempt });
      return;
    } catch (e: any) {
      if (controller.signal.aborted) {
        // Hung worker: terminate + quarantine. Terminal (see policy note above).
        await terminateWorker(env, workerUrl, payload.jobId, 'timeout');
        await setJob(env, jobKey, { status: 'TERMINATED_TIMEOUT', attempts: attempt });
        console.error('[station3] worker timed out; termination signalled');
        return;
      }
      // Network error: transient -> retry if attempts remain.
      if (attempt < MAX_DISPATCH_ATTEMPTS) {
        await setJob(env, jobKey, { status: 'RETRYING', attempt, reason: 'network' });
        continue;
      }
      await setJob(env, jobKey, { status: 'ERROR', attempts: attempt });
      return;
    } finally {
      clearTimeout(timeout);
    }
  }
}

async function setJob(env: Bindings, jobKey: string, record: object): Promise<void> {
  await env.KV_CACHE.put(jobKey, JSON.stringify(record), { expirationTtl: JOB_TTL_S });
}

/** Station 2: sign an asymmetric ES256 gateway→worker JWT. Returns null (fail
 *  closed) when the private key is missing or invalid, so callers refuse to
 *  dispatch/serve rather than proceed unauthenticated. */
async function signGatewayJwt(env: Bindings, jobId: string): Promise<string | null> {
  const pem = env.GATEWAY_JWT_PRIVATE_KEY;
  if (!pem) return null;
  try {
    const privateKey = await jose.importPKCS8(pem, 'ES256');
    return await new jose.SignJWT({ role: 'gateway', jobId })
      .setProtectedHeader({ alg: 'ES256' })
      .setIssuedAt()
      .setExpirationTime('2m')
      .sign(privateKey);
  } catch {
    return null;
  }
}

/** Station 3 enforcement: signal the offending worker to shut down (best effort)
 *  and quarantine it so no further results are trusted. Real cluster teardown is
 *  performed by the infra provider's API (RunPod/Modal) driven off the quarantine
 *  flag; here we send the in-band terminate signal and record the quarantine. */
async function terminateWorker(env: Bindings, workerUrl: string, jobId: string, reason: string): Promise<void> {
  // Quarantine is the DURABLE signal (infra teardown keys off it); set it first so
  // it stands even if the in-band terminate can't authenticate or the worker is gone.
  await env.KV_CACHE.put(`quarantine:${workerUrl}`, reason, { expirationTtl: JOB_TTL_S });
  // Sign a FRESH JWT rather than reuse the dispatch token: on the timeout path the
  // dispatch JWT (exp 2m) is long expired by the time a MAX_RENDER_MS (up to 15m)
  // hang fires, so the worker would reject the terminate as expired and the shutdown
  // signal would be silently dropped. Re-signing keeps the in-band terminate valid.
  const jwt = await signGatewayJwt(env, jobId);
  if (!jwt) return;                          // cannot authenticate; quarantine still stands
  try {
    await fetch(`${workerUrl}/api/worker/terminate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${jwt}` },
      body: JSON.stringify({ reason }),
    });
  } catch {
    // Worker may already be gone; the quarantine flag is the durable signal.
  }
}

// --- M1: durable dispatch Queue consumer (ADR 0001) ------------------------
// The message a /create producer enqueues. The full payload rides the message so
// the consumer can dispatch without a second read; input size is already bounded by
// validateJobSize (well under the 128 KiB queue-message limit).
export interface QueueJob {
  deviceId: string;
  jobId: string;
  payload: JobRequest;
}

/** Resolve the JobCoordinator DO stub for one (device, job). One object per pair. */
function coordinatorStub(ns: DurableObjectNamespace, deviceId: string, jobId: string) {
  return ns.get(ns.idFromName(`${deviceId}:${jobId}`));
}

/** Queue consumer: the durable replacement for `background(dispatchToWorker)` in
 *  `waitUntil`. On Worker eviction mid-consume the message is NOT acked, so the Queue
 *  redelivers it — a job can never be silently lost (the core W1 fix). The DO's
 *  terminal-sticky state makes at-least-once redelivery idempotent. */
export async function handleJobQueue(batch: MessageBatch<QueueJob>, env: Bindings): Promise<void> {
  for (const message of batch.messages) {
    const { deviceId, jobId, payload } = message.body;
    try {
      if (!env.JOB_COORDINATOR) throw new Error('JOB_COORDINATOR binding missing');
      const stub = coordinatorStub(env.JOB_COORDINATOR, deviceId, jobId);

      // Idempotent consume: at-least-once delivery can redeliver a finished job. The
      // DO is the authority — if it is already terminal, ACK without re-dispatching.
      const current = await coordinatorGet(stub);
      if (current && isTerminal(current.status)) {
        message.ack();
        continue;
      }

      // Mark in-flight (QUEUED→DISPATCHING; a no-op if a prior delivery already did).
      await coordinatorTransition(stub, 'DISPATCHING');

      // Unchanged dispatch: owns the worker round-trip, Station-3 timing bounds, the
      // in-request transient-retry loop, and the KV job:/result: projection writes.
      // When it returns, the KV `job:` status is always terminal.
      const jobKey = `job:${deviceId}:${jobId}`;
      await dispatchToWorker(env, payload, jobKey);

      // Sync the DO authority to the terminal state the dispatch landed on, so any
      // later redelivery short-circuits at the idempotent-consume guard above.
      const finalKv = (await env.KV_CACHE.get(jobKey, { type: 'json' })) as
        | ({ status?: JobStatus } & Record<string, unknown>)
        | null;
      if (finalKv?.status) {
        await coordinatorTransition(stub, finalKv.status, finalKv);
      }
      message.ack();
    } catch {
      // Catastrophic failure (DO/KV unreachable, Worker evicted mid-consume). Do NOT
      // ack — let the Queue redeliver so the job survives (durable retry `waitUntil`
      // could never provide). Zero-Logging: never print payload/url/token.
      console.error('[queue] dispatch consume failed; will redeliver');
      message.retry();
    }
  }
}

// Serve BOTH roles from the one Worker: `.fetch` (the Hono router) and `.queue`
// (the dispatch consumer). Tests keep importing the default app and using .request;
// the runtime additionally invokes .queue for the job-dispatch consumer.
(app as unknown as { queue: typeof handleJobQueue }).queue = handleJobQueue;

// Re-export the Durable Object class so wrangler's [[durable_objects.bindings]] +
// [[migrations]] can find it as a named export on the Worker entrypoint.
export { JobCoordinator };

export default app;
