import { Hono } from 'hono';
import { cors } from 'hono/cors';
import * as jose from 'jose';
import { verifySignature } from '@dichomnion/crypto-utils';
import { JobRequest, deterministicStringify } from '@dichomnion/shared-types';

type Bindings = {
  KV_CACHE: KVNamespace;
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
};

// ---- Security / anti-fraud tuning -----------------------------------------
const REPLAY_WINDOW_MS = 30_000;          // reject requests older/newer than this
const REGISTER_RATE_LIMIT = 5;            // max registrations per IP per window
const REGISTER_WINDOW_S = 3_600;          // registration rate-limit window (1h)
const JOB_TTL_S = 86_400;                 // job/result records live 24h
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
  const body = await c.req.json().catch(() => ({}));
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

  // Idempotency: a re-sent job returns the existing record, never double-dispatches.
  const jobKey = `job:${deviceId}:${payloadObj.jobId}`;
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

  const etaSeconds = estimateEtaSeconds(payloadObj);
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
    const inner = stored?.result ?? {};
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
  const jwt = await signGatewayJwt(env, payload.jobId);
  if (!jwt) {
    const reason = env.GATEWAY_JWT_PRIVATE_KEY ? 'gateway_key_invalid' : 'gateway_key_missing';
    await setJob(env, jobKey, { status: 'FAILED', reason });
    console.error(`[station2] cannot dispatch job: ${reason}`);
    return;
  }

  // Station 3: bound the round-trip. Too fast => faked result; too slow => hung.
  // Bounds default to the tuned constants but can be overridden per deployment.
  const floor = Number(env.MIN_PLAUSIBLE_MS) || MIN_PLAUSIBLE_MS_FLOOR;
  const maxMs = Number(env.MAX_RENDER_MS) || MAX_PLAUSIBLE_MS;
  const minPlausibleMs = Math.max(
    floor,
    (payload.segments?.length || 0) * PER_SEGMENT_MIN_MS,
  );
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
    const startTime = Date.now();

    try {
      const res = await fetch(`${workerUrl}/api/worker/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${jwt}` },
        body: JSON.stringify({
          job_id: payload.jobId,
          audio_url: payload.videoAudioUrl,
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
      await setJob(env, jobKey, { status: 'DONE', elapsed, attempts: attempt });
      // DEVICE-SCOPED result key (derived from jobKey `job:<device>:<jobId>`) so the
      // stored output is only retrievable by the owning device, not by jobId guess.
      const resultKey = jobKey.replace(/^job:/, 'result:');
      await env.KV_CACHE.put(
        resultKey,
        JSON.stringify({ job_id: workerResponse?.job_id, result: storedResult }),
        { expirationTtl: JOB_TTL_S },
      );
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

export default app;
