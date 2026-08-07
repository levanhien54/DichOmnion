import { Hono } from 'hono';
import { cors } from 'hono/cors';
import * as jose from 'jose';
import {
  JobRequest,
  JobRequestSchema,
  WORKER_JWT_AUDIENCE,
  WorkerJwtClaimsSchema,
  WorkerJwtAct,
  WorkerResponseSchema,
  FAILURE_REASONS,
  // M4-S3 — the ANALYZE (Human-in-the-Loop) contract. AnalyzeRequest rides the analyze
  // submission route + dispatch message; WorkerAnalyzeResponseSchema is the FULL-body gate
  // the Gateway parses before it rests a job at AWAITING_REVIEW (never latch a bare 200).
  AnalyzeRequest,
  AnalyzeRequestSchema,
  EncryptedArtifactSchema,
  WorkerAnalyzeResponseSchema,
  // M4-S4 — the APPROVE contract. ApproveRequest carries the canonical client-approved
  // manifest; the Gateway recomputes its hash (deterministicStringify) and mints an
  // ApprovedRevision that a later RENDER submission binds to (ADR 0002 §4).
  ApproveRequest,
  ApproveRequestSchema,
  // M4-S4 — the RENDER contract. RenderRequest binds a render to an APPROVED manifest
  // (analyzeJobId + approvedManifestHash); the render dispatcher fetches that manifest
  // from R2, RE-HASHES it, and drives /api/worker/render with its human-approved
  // translatedText VERBATIM (no Qwen re-call). ApprovedManifest is the canonical shape.
  RenderRequest,
  RenderRequestSchema,
  ApprovedManifest,
  ApprovedManifestSchema,
  VOICE_CATALOG_SCHEMA_VERSION,
  WorkerVoiceCapabilitiesSchema,
  WorkerReadinessSchema,
  JobProgressSchema,
  type VoiceCatalog,
  type JobProgress,
  type JobProgressStage,
  MAX_QWEN_CUSTOM_INSTRUCTION_CHARS,
  deterministicStringify,
} from '@dichomnion/shared-types';
// M4-S4 — sha256Hex recomputes the canonical manifest hash server-side so a tampered
// manifest can't ride a stale/forged hash into an ApprovedRevision.
import { sha256Hex } from '@dichomnion/crypto-utils';
import { mintJobAudioUrls, presignS3Url, amzDate } from './r2presign';
// M3-S2 (item 4): CANONICAL Zero-Trust auth. authenticateSignedRequest verifies an
// ECDSA signature over METHOD+PATH+bodyHash+X-Timestamp+X-Nonce (see auth.ts), so a
// captured signature can't be borrowed onto another route/method. REPLAY_WINDOW_MS
// moves there with it (it is now checked on the SIGNED X-Timestamp header).
import { authenticateSignedRequest } from './auth';
// M2-S5d — the orphan sweeper (scheduled/cron backstop). Reclaims R2 objects that
// outlived their job, protecting the winning artifact + every live-job object.
import { sweepOrphans } from './sweeper';
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
  MAX_INPUT_BYTES,
  MAX_QUEUE_PAYLOAD_BYTES,
} from './limits';
// M1 (ADR 0001) — durable job coordination. JobCoordinator is the atomic source of
// truth (per-(device,job) Durable Object); the Queue consumer drives dispatch with
// durable redelivery. Re-export the DO class at the bottom so wrangler can bind it.
import {
  JobCoordinator,
  coordinatorCreate,
  coordinatorTransition,
  coordinatorGet,
  coordinatorMarkEnqueued,
  coordinatorAcquireDispatch,
  coordinatorCancel,
  coordinatorApprove,
  coordinatorGetApproval,
  coordinatorPrepareAnalyzeRerun,
  coordinatorCompleteAnalyzeRerun,
  coordinatorRollbackAnalyzeRerun,
  coordinatorClaimRender,
  coordinatorFinishRender,
  coordinatorReleaseRender,
  isTerminal,
  type JobStatus,
} from './coordinator';
// M3-S4 (ADR-Zero-Trust) — ReplayGuard is the strong-consistent nonce single-use store
// (per-device Durable Object). Re-exported at the bottom so wrangler can bind the class.
import { ReplayGuard } from './replayguard';
// M3-S5 (Bug D) — RateLimiter is the ATOMIC fixed-window throttle (per-key Durable
// Object) that replaces the get-then-put KV TOCTOU on register/job/presign. enforceRateLimit
// routes to the DO when bound and degrades to KV otherwise. Class re-exported at the bottom.
import { RateLimiter, enforceRateLimit } from './ratelimiter';
// M3-S6 — KillSwitch is the STRONG-consistent Financial Kill Switch state (single global-shard
// Durable Object). readKillSwitch/setKillSwitch route to the DO when bound and degrade to the
// legacy KV flag otherwise; triggerProviderTeardown fires the optional scale-to-zero hook.
// Class re-exported at the bottom so wrangler can bind it.
import { KillSwitch, readKillSwitch, setKillSwitch, triggerProviderTeardown } from './killswitch';
import {
  WorkerTargetRegistry,
  WorkerTargetValidationError,
  acquireWorkerControlLease,
  acquireWorkerProvisionLease,
  clearWorkerProvisionIntent,
  readWorkerProvisionIntent,
  clearWorkerTarget,
  publishWorkerTarget,
  resolveWorkerTarget,
  setWorkerControlBlocked,
  workerTargetSupportsRequest,
  type ResolvedWorkerTarget,
} from './worker_target';

type Bindings = {
  KV_CACHE: KVNamespace;
  // M1 (ADR 0001) — Durable Object namespace for JobCoordinator (atomic job state)
  // and the producer handle for the durable dispatch Queue. Both are OPTIONAL on the
  // type so the many existing tests that build a KV-only env keep type-checking; the
  // /create producer path degrades gracefully when they are absent (see below).
  JOB_COORDINATOR?: DurableObjectNamespace;
  JOB_DISPATCH_QUEUE?: Queue<QueueJob>;
  // M3-S4 — ReplayGuard Durable Object namespace (nonce single-use store, sharded by
  // deviceId). OPTIONAL: when absent, authenticateSignedRequest degrades to the
  // timestamp-only replay window; it is NEVER substituted with KV. See src/replayguard.ts.
  REPLAY_GUARD?: DurableObjectNamespace;
  // M3-S5 (Bug D) — RateLimiter Durable Object namespace (atomic fixed-window throttle,
  // sharded by the full rate-limit key). OPTIONAL: when absent, enforceRateLimit degrades
  // to the legacy KV get-then-put (a real limiter sequentially, TOCTOU-prone under
  // concurrency). Production binds it so register/job/presign are lost-update-free.
  RATE_LIMITER?: DurableObjectNamespace;
  // M3-S6 — KillSwitch Durable Object namespace (single global shard = the whole-service
  // Financial Kill Switch). OPTIONAL: when absent, readKillSwitch/setKillSwitch degrade to
  // the legacy `system:kill_switch` KV flag. Production binds it so a flip is strongly
  // consistent across PoPs (a financial decision must not ride KV eventual-consistency).
  KILL_SWITCH?: DurableObjectNamespace;
  // Strong single-shard registry for the RunPod controller's short-lived target
  // heartbeats. Optional only for KV-backed tests and rolling migration.
  WORKER_TARGET_REGISTRY?: DurableObjectNamespace;
  // M3-S6 — optional provider scale-to-zero hook fired when the kill switch is ARMED. BOTH
  // required to fire; absent → honest no-op. The URL is the infra provider's teardown
  // endpoint (RunPod/Modal/etc.); the token authenticates the call (a Wrangler secret).
  PROVIDER_TEARDOWN_URL?: string;
  PROVIDER_TEARDOWN_TOKEN?: string;
  // Station 2 (asymmetric): PKCS8 PEM of the gateway's ES256 (P-256) PRIVATE key.
  // Provided as a Wrangler secret. The Worker holds only the matching PUBLIC key.
  GATEWAY_JWT_PRIVATE_KEY?: string;
  WORKER_URL?: string;            // static fallback; dynamic controller target wins while live
  WORKER_CAPABILITY_TIMEOUT_MS?: string; // bounded timeout for the public voice-catalog proxy
  WORKER_CAPABILITY_CACHE_MS?: string; // short isolate-local cache; bounded to 5 seconds
  TURNSTILE_SECRET?: string;      // Cloudflare Turnstile secret key
  ALLOWED_ORIGINS?: string;       // comma-separated list of allowed client origins
  ADMIN_TOKEN?: string;           // shared secret for the Financial Kill Switch monitor
  WORKER_TARGET_ADMIN_TOKEN?: string; // separate least-privilege secret for RunPod controller
  // Station 3 timing bounds are tunable per deployment (worker tier / content length).
  MIN_PLAUSIBLE_MS?: string;      // override the "too fast => fraud" floor
  MAX_RENDER_MS?: string;         // override the hard render timeout
  // Opt-in async analyze protocol: submit returns quickly, then Gateway polls short
  // status requests so RunPod's 90s proxy cap never covers model inference.
  ASYNC_ANALYZE?: string;
  ASYNC_ANALYZE_MAX_MS?: string;
  ASYNC_ANALYZE_ENABLED?: string; // submit+poll ANALYZE instead of one long proxy request
  ASYNC_ANALYZE_TIMEOUT_MS?: string; // total async compute deadline (default 15 minutes)
  ASYNC_ANALYZE_POLL_MS?: string; // bounded worker-status polling cadence
  MAX_INPUT_BYTES?: string;       // M2-S5b: override the 30 MB input ceiling
  // Đợt 17 F3/F4: input-size bounds (defense-in-depth mirror of the worker gate).
  MAX_SEGMENTS?: string;          // override max approved segments per job
  MAX_SEGMENT_TEXT_CHARS?: string;// override max chars for a single segment's text
  MAX_TOTAL_TEXT_CHARS?: string;  // override max chars summed across all segments
  MAX_SEGMENT_META_CHARS?: string;// Đợt 18 F6: override max chars for a segment's id/speaker
  // M3-S9: byte-level bounds on the create request (see ./limits). MAX_REQUEST_BODY_BYTES is
  // read in auth.ts (raw-body DoS backstop); MAX_QUEUE_PAYLOAD_BYTES is the dispatch Queue-fit
  // guard read here. Both optional → compiled-in defaults when unset.
  MAX_REQUEST_BODY_BYTES?: string;// override the raw-body cap (default 256 KiB)
  MAX_QUEUE_PAYLOAD_BYTES?: string;// override the dispatch Queue-fit bound (default 120 KiB)
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
  DISPATCH_GET_EXPIRES_S?: string; // M2-S2: dispatch-time GET lifetime (default 900s = 15m)
  PRESIGN_RATE_LIMIT?: string;    // override max mints per device per window
  PRESIGN_WINDOW_S?: string;      // override the mint-throttle window (s)
  // M2 — native R2 bucket binding (SAME bucket the presign vars name). Presigned
  // URLs let the client/worker read+write R2 from OUTSIDE the Worker; this binding
  // lets the GATEWAY itself operate on the bucket (delete a job's input after it is
  // terminal, HEAD-verify a result before DONE, sweep orphans) WITHOUT threading
  // SigV4 query-credentials through KV/DO. Optional on the type so the KV-only test
  // envs and a pre-R2 deploy keep working: R2-side lifecycle no-ops when unbound.
  R2?: R2Bucket;
};

// ---- Security / anti-fraud tuning -----------------------------------------
// (REPLAY_WINDOW_MS now lives in auth.ts alongside the canonical verifier — M3-S2.)
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
// M2-S2 — the GET the Gateway hands the worker at DISPATCH is minted fresh and
// lives only long enough to fetch: it never sits at rest in the durable Queue
// message (unlike the client's long-lived upload URL used to). Keep it short so a
// leaked queue/DO record grants at most this window of read access. 15m covers
// download + a slow start under load; bounded by the same 604800 SigV4 ceiling.
const DISPATCH_GET_EXPIRES_S = 900;       // dispatch-time GET lifetime (15m)
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

function workerRequestTimeoutMs(env: Pick<Bindings, 'MAX_RENDER_MS'>): number {
  const configured = Number(env.MAX_RENDER_MS);
  return Number.isFinite(configured) && configured > 0
    ? Math.floor(configured)
    : MAX_PLAUSIBLE_MS;
}

function asyncAnalyzeEnabled(env: Pick<Bindings, 'ASYNC_ANALYZE_ENABLED'>): boolean {
  return ['1', 'true', 'yes', 'on'].includes(
    (env.ASYNC_ANALYZE_ENABLED ?? '').trim().toLowerCase(),
  );
}

function asyncAnalyzeTimeoutMs(env: Pick<Bindings, 'ASYNC_ANALYZE_TIMEOUT_MS'>): number {
  const configured = Number(env.ASYNC_ANALYZE_TIMEOUT_MS);
  if (!Number.isFinite(configured) || configured <= 0) return MAX_PLAUSIBLE_MS;
  return Math.min(MAX_PLAUSIBLE_MS, Math.max(60_000, Math.floor(configured)));
}

function asyncAnalyzePollMs(env: Pick<Bindings, 'ASYNC_ANALYZE_POLL_MS'>): number {
  const configured = Number(env.ASYNC_ANALYZE_POLL_MS);
  if (!Number.isFinite(configured) || configured <= 0) return 2_500;
  return Math.min(10_000, Math.max(500, Math.floor(configured)));
}
// M3-S6: the kill-switch state now lives behind readKillSwitch/setKillSwitch (strong DO when
// bound, legacy `system:kill_switch` KV flag otherwise); the constant moved to ./killswitch.

// Edge input-bound constants (MAX_SEGMENTS, MAX_SEGMENT_TEXT_CHARS, MAX_TOTAL_TEXT_CHARS,
// MAX_FREETEXT_CHARS, MAX_SEGMENT_META_CHARS, MAX_JOBID_CHARS, MAX_INPUT_BYTES) are imported
// from ./limits above — see that module for the full Đợt 17 F3/F4 + Đợt 18 F6 + M2-S5b
// rationale and why a plain-value named export from this entrypoint cannot exist here.

function stageProgress(stage: JobProgressStage, attempt?: number, updatedAt = Date.now()): JobProgress {
  return {
    mode: 'indeterminate',
    stage,
    updatedAt,
    ...(attempt && attempt > 0 ? { attempt } : {}),
  };
}

function completeProgress(attempt?: number): JobProgress {
  return {
    mode: 'determinate',
    stage: 'complete',
    updatedAt: Date.now(),
    ...(attempt && attempt > 0 ? { attempt } : {}),
    completed: 1,
    total: 1,
    unit: 'steps',
  };
}

function publicJobProgress(value: unknown): JobProgress | undefined {
  const parsed = JobProgressSchema.safeParse(value);
  return parsed.success ? parsed.data : undefined;
}

const app = new Hono<{ Bindings: Bindings }>();

// CORS: restrict to configured client origins (fall back to localhost dev ports).
// Never reflect an arbitrary Origin in production.
app.use('/api/*', cors({
  origin: (origin, c) => {
    const configured = (c.env.ALLOWED_ORIGINS || '')
      .split(',').map((o: string) => o.trim()).filter(Boolean);
    const allow = configured.length
      ? configured
      : [
          'http://localhost:1420',
          'http://127.0.0.1:1420',
          'http://localhost:5173',
          'http://127.0.0.1:5173',
          'tauri://localhost',
          'https://tauri.localhost',
        ];
    if (!origin) return allow[0];          // non-browser callers (no Origin header)
    return allow.includes(origin) ? origin : null;
  },
  // M3-S2: X-Timestamp / X-Nonce carry the signed replay-window timestamp + per-request
  // nonce that are now folded into the canonical signature (were body fields before).
  allowHeaders: ['Content-Type', 'X-ECDSA-Signature', 'X-Device-Id', 'X-Timestamp', 'X-Nonce'],
  allowMethods: ['POST', 'GET', 'OPTIONS'],
}));

app.get('/', (c) => c.text('OmniVoice Gateway is running! Secure Edge Active.'));

/**
 * Public, sanitized deployment probe for the desktop Settings screen.
 * It reports presence/absence only; secrets, worker URLs and topology never
 * leave the Gateway. A degraded configuration is still HTTP 200 so clients
 * can render actionable diagnostics instead of collapsing everything into a
 * generic network error.
 */
app.get('/api/health', async (c) => {
  const privateKeyPresent = Boolean(c.env.GATEWAY_JWT_PRIVATE_KEY?.trim());
  const probeJwt = privateKeyPresent
    ? await signGatewayJwt(c.env, 'gateway-readiness', 'probe')
    : null;
  const authStatus = !privateKeyPresent ? 'missing' : probeJwt ? 'configured' : 'invalid';
  const authCacheKey = probeJwt
    ? await sha256Hex(c.env.GATEWAY_JWT_PRIVATE_KEY!.trim())
    : '';
  const uploadStorageConfigured = [
    c.env.R2_ACCOUNT_ID,
    c.env.R2_BUCKET,
    c.env.R2_ACCESS_KEY_ID,
    c.env.R2_SECRET_ACCESS_KEY,
  ].every((value) => Boolean(value?.trim()));
  const workerConfig = await resolveWorkerTarget(c.env);
  let workerStatus: WorkerOperationalStatus;
  if (workerConfig.status !== 'valid') {
    workerStatus = workerConfig.status;
  } else if (!probeJwt) {
    workerStatus = 'auth_failed';
  } else {
    const readiness = await cachedWorkerReadiness(
      workerConfig.url,
      probeJwt,
      authCacheKey,
      workerCapabilityTimeoutMs(c.env.WORKER_CAPABILITY_TIMEOUT_MS),
      workerCapabilityCacheMs(c.env.WORKER_CAPABILITY_CACHE_MS),
    );
    workerStatus = readiness.status === 'ready'
      && !workerTargetSupportsRequest(workerConfig, workerRequestTimeoutMs(c.env))
      ? 'transport_limited'
      : readiness.status;
  }
  const ready = authStatus === 'configured' && uploadStorageConfigured && workerStatus === 'ready';

  c.header('Cache-Control', 'no-store');
  return c.json({
    status: ready ? 'ready' : 'degraded',
    checks: {
      gateway: 'ok',
      auth: authStatus,
      uploadStorage: uploadStorageConfigured ? 'configured' : 'missing',
      worker: workerStatus,
    },
  });
});

const DEFAULT_WORKER_CAPABILITY_TIMEOUT_MS = 3_500;
const MAX_WORKER_CAPABILITY_TIMEOUT_MS = 10_000;
const DEFAULT_WORKER_CAPABILITY_CACHE_MS = 1_000;
const MAX_WORKER_CAPABILITY_CACHE_MS = 5_000;
const VOICE_CAPABILITY_UNAVAILABLE_REASON = 'capability_unavailable' as const;
const MAX_WORKER_CAPABILITY_RESPONSE_BYTES = 128 * 1024;
const MAX_WORKER_CAPABILITY_CACHE_KEYS = 8;

type WorkerOperationalStatus =
  | 'missing'
  | 'invalid'
  | 'unreachable'
  | 'auth_failed'
  | 'not_ready'
  | 'contract_invalid'
  | 'unavailable'
  | 'transport_limited'
  | 'ready';

type WorkerReadinessResult = {
  status: Exclude<WorkerOperationalStatus, 'missing' | 'invalid'>;
  catalog?: VoiceCatalog;
};

type CachedWorkerReadiness = { expiresAt: number; result: WorkerReadinessResult };
const workerReadinessCache = new Map<string, CachedWorkerReadiness>();
const workerReadinessInflight = new Map<string, Promise<WorkerReadinessResult>>();

export function resetVoiceCapabilityCacheForTests(): void {
  workerReadinessCache.clear();
  workerReadinessInflight.clear();
}

function unavailableVoiceCatalog(): VoiceCatalog {
  return {
    schema_version: VOICE_CATALOG_SCHEMA_VERSION,
    revision: 'unavailable',
    ready: false,
    localOnly: true,
    reason: VOICE_CAPABILITY_UNAVAILABLE_REASON,
    defaultProfileId: null,
    profiles: [],
  };
}

function workerCapabilityTimeoutMs(raw: string | undefined): number {
  const configured = Number(raw);
  if (!Number.isFinite(configured) || configured <= 0) {
    return DEFAULT_WORKER_CAPABILITY_TIMEOUT_MS;
  }
  return Math.min(Math.floor(configured), MAX_WORKER_CAPABILITY_TIMEOUT_MS);
}

function workerCapabilityCacheMs(raw: string | undefined): number {
  const configured = Number(raw);
  if (!Number.isFinite(configured) || configured <= 0) {
    return DEFAULT_WORKER_CAPABILITY_CACHE_MS;
  }
  return Math.min(Math.floor(configured), MAX_WORKER_CAPABILITY_CACHE_MS);
}

async function readBoundedWorkerCapability(response: Response): Promise<unknown> {
  const declared = Number(response.headers.get('content-length'));
  if (Number.isFinite(declared) && declared > MAX_WORKER_CAPABILITY_RESPONSE_BYTES) {
    throw new Error('worker capability response too large');
  }
  if (!response.body) throw new Error('worker capability response is empty');

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > MAX_WORKER_CAPABILITY_RESPONSE_BYTES) {
      await reader.cancel();
      throw new Error('worker capability response too large');
    }
    chunks.push(value);
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return JSON.parse(new TextDecoder().decode(bytes)) as unknown;
}

async function fetchWorkerReadiness(
  workerUrl: string,
  probeJwt: string,
  timeoutMs: number,
): Promise<WorkerReadinessResult> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    let response: Response;
    try {
      response = await fetch(`${workerUrl}/api/worker/readiness`, {
        method: 'GET',
        headers: {
          Accept: 'application/json',
          Authorization: `Bearer ${probeJwt}`,
        },
        signal: controller.signal,
      });
    } catch {
      return { status: 'unreachable' };
    }
    if (response.status === 401 || response.status === 403) return { status: 'auth_failed' };

    // Older, already-published worker images expose the versioned voice catalog but
    // predate the aggregate readiness route. Keep that image usable during a rolling
    // upgrade, while still requiring its public health contract below before reporting
    // the worker as ready. A 404 is the only condition that activates this compatibility
    // path; malformed/new readiness responses remain fail-closed.
    if (response.status === 404) {
      return await fetchLegacyWorkerReadiness(workerUrl, probeJwt, controller.signal);
    }

    if (!response.ok) {
      if (response.status !== 503) return { status: 'contract_invalid' };
      try {
        const body = await readBoundedWorkerCapability(response);
        const detail = typeof body === 'object' && body !== null
          ? (body as { detail?: unknown }).detail
          : undefined;
        return detail === 'Worker not provisioned with Gateway public key'
          ? { status: 'auth_failed' }
          : { status: 'unreachable' };
      } catch {
        return { status: 'unreachable' };
      }
    }

    let body: unknown;
    try {
      body = await readBoundedWorkerCapability(response);
    } catch {
      return { status: controller.signal.aborted ? 'unreachable' : 'contract_invalid' };
    }
    const parsed = WorkerReadinessSchema.safeParse(body);
    if (!parsed.success) return { status: 'contract_invalid' };
    return {
      status: parsed.data.ready && parsed.data.catalog.ready ? 'ready' : 'not_ready',
      catalog: parsed.data.catalog,
    };
  } catch {
    return { status: 'unreachable' };
  } finally {
    clearTimeout(timeout);
  }
}

type LegacyHealthCapability = {
  available?: unknown;
  required?: unknown;
  pipeline_loaded?: unknown;
};

function legacyHealthReady(body: unknown): boolean | null {
  if (typeof body !== 'object' || body === null || Array.isArray(body)) return null;
  const health = body as Record<string, unknown>;
  if (health.status !== 'ok' && health.status !== 'not_ready') return null;
  if (typeof health.models_loaded !== 'boolean') return null;
  if (health.device !== 'cuda' && health.device !== 'cpu' && health.device !== 'unknown') return null;

  const diarization = health.diarization;
  if (typeof diarization !== 'object' || diarization === null || Array.isArray(diarization)) return null;
  const diarizationCapability = diarization as Record<string, unknown>;
  if (typeof diarizationCapability.available !== 'boolean'
    || typeof diarizationCapability.pipeline_loaded !== 'boolean') return null;

  const enhancements = health.audio_enhancements;
  if (typeof enhancements !== 'object' || enhancements === null || Array.isArray(enhancements)) return null;
  for (const name of ['audioseal', 'demucs']) {
    const capability = (enhancements as Record<string, unknown>)[name] as LegacyHealthCapability | undefined;
    if (typeof capability !== 'object' || capability === null || Array.isArray(capability)
      || typeof capability.available !== 'boolean' || typeof capability.required !== 'boolean') return null;
    if (capability.required === true && capability.available !== true) return false;
  }

  const tts = health.tts as LegacyHealthCapability | undefined;
  if (typeof tts !== 'object' || tts === null || Array.isArray(tts)
    || typeof tts.available !== 'boolean' || typeof tts.required !== 'boolean') return null;

  // This deployment requires local TTS and diarization. The older health response has
  // no separate readiness version, so these fields are the minimum safe equivalent.
  const ready = health.status === 'ok'
    && health.models_loaded === true
    && health.device === 'cuda'
    && diarizationCapability.available === true
    && diarizationCapability.pipeline_loaded === true
    && tts.available === true;
  return ready;
}

async function fetchLegacyWorkerReadiness(
  workerUrl: string,
  probeJwt: string,
  signal: AbortSignal,
): Promise<WorkerReadinessResult> {
  let catalogResponse: Response;
  try {
    catalogResponse = await fetch(`${workerUrl}/api/worker/voices`, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
        Authorization: `Bearer ${probeJwt}`,
      },
      signal,
    });
  } catch {
    return { status: signal.aborted ? 'unreachable' : 'unreachable' };
  }
  if (catalogResponse.status === 401 || catalogResponse.status === 403) return { status: 'auth_failed' };
  if (!catalogResponse.ok) return { status: catalogResponse.status === 503 ? 'unreachable' : 'contract_invalid' };

  let catalog: VoiceCatalog;
  try {
    const body = await readBoundedWorkerCapability(catalogResponse);
    const parsed = WorkerVoiceCapabilitiesSchema.safeParse(body);
    if (!parsed.success) return { status: 'contract_invalid' };
    catalog = parsed.data.catalog;
  } catch {
    return { status: signal.aborted ? 'unreachable' : 'contract_invalid' };
  }

  let healthResponse: Response;
  try {
    healthResponse = await fetch(`${workerUrl}/health`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal,
    });
  } catch {
    return { status: 'unreachable', catalog };
  }
  if (healthResponse.status === 401 || healthResponse.status === 403) return { status: 'auth_failed', catalog };

  let healthBody: unknown;
  try {
    healthBody = await readBoundedWorkerCapability(healthResponse);
  } catch {
    return { status: signal.aborted ? 'unreachable' : 'contract_invalid', catalog };
  }
  // A non-2xx health response is never allowed to become ready, even if a stale or
  // contradictory body happens to say `status: ok`. Preserve a useful not_ready
  // result for the documented 503 shape; all other statuses are a contract fault.
  if (!healthResponse.ok && healthResponse.status !== 503) {
    return { status: 'contract_invalid', catalog };
  }
  const healthReady = legacyHealthReady(healthBody);
  if (healthReady === null) return { status: 'contract_invalid', catalog };
  if (!healthResponse.ok) return { status: 'not_ready', catalog };
  return { status: healthReady && catalog.ready ? 'ready' : 'not_ready', catalog };
}

async function cachedWorkerReadiness(
  workerUrl: string,
  probeJwt: string,
  authCacheKey: string,
  timeoutMs: number,
  cacheMs: number,
): Promise<WorkerReadinessResult> {
  // Include a one-way key fingerprint so a private-key rotation cannot inherit a
  // stale successful auth result, while health and voices still share one probe.
  const key = `${workerUrl}\n${timeoutMs}\n${authCacheKey}`;
  const cached = workerReadinessCache.get(key);
  if (cached && cached.expiresAt > Date.now()) return cached.result;
  if (cached) workerReadinessCache.delete(key);

  let pending = workerReadinessInflight.get(key);
  if (!pending) {
    pending = fetchWorkerReadiness(workerUrl, probeJwt, timeoutMs);
    if (workerReadinessInflight.size >= MAX_WORKER_CAPABILITY_CACHE_KEYS) {
      const oldest = workerReadinessInflight.keys().next().value as string | undefined;
      if (oldest !== undefined) workerReadinessInflight.delete(oldest);
    }
    workerReadinessInflight.set(key, pending);
  }

  let result: WorkerReadinessResult;
  try {
    result = await pending;
  } catch {
    result = { status: 'unreachable' };
  } finally {
    if (workerReadinessInflight.get(key) === pending) workerReadinessInflight.delete(key);
  }
  if (workerReadinessCache.size >= MAX_WORKER_CAPABILITY_CACHE_KEYS && !workerReadinessCache.has(key)) {
    const oldest = workerReadinessCache.keys().next().value as string | undefined;
    if (oldest !== undefined) workerReadinessCache.delete(oldest);
  }
  workerReadinessCache.set(key, { expiresAt: Date.now() + cacheMs, result });
  return result;
}

/** Public, sanitized catalog proxy. It reuses the authenticated readiness probe, so
 * the Settings health state and the catalog cannot disagree or probe the Worker twice. */
app.get('/api/voices', async (c) => {
  c.header('Cache-Control', 'no-store');
  const workerConfig = await resolveWorkerTarget(c.env);
  if (workerConfig.status !== 'valid') return c.json(unavailableVoiceCatalog());
  const probeJwt = await signGatewayJwt(c.env, 'gateway-readiness', 'probe');
  if (!probeJwt) return c.json(unavailableVoiceCatalog());
  const authCacheKey = await sha256Hex(c.env.GATEWAY_JWT_PRIVATE_KEY!.trim());

  const readiness = await cachedWorkerReadiness(
    workerConfig.url,
    probeJwt,
    authCacheKey,
    workerCapabilityTimeoutMs(c.env.WORKER_CAPABILITY_TIMEOUT_MS),
    workerCapabilityCacheMs(c.env.WORKER_CAPABILITY_CACHE_MS),
  );
  return c.json(readiness.catalog ?? unavailableVoiceCatalog());
});

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

/** Is the Financial Kill Switch active? Fails CLOSED: if the strong-state source (the DO,
 *  when bound) cannot answer, treat the switch as ACTIVE (block) rather than assume it is off
 *  — a financial guard that cannot answer must never wave spend through (M3-S6). Callers that
 *  see `true` return 503. Delegates state to ./killswitch (DO when bound, KV flag otherwise). */
function killSwitchBlocks(env: Bindings): Promise<boolean> {
  return readKillSwitch(env).catch(() => true);
}

/** Coarse ETA (seconds) for the async job, from the number of approved segments.
 *  Surfaced in the 202 accept and persisted so polling echoes it. Accepts `unknown`
 *  and reads only `.segments` internally so BOTH the render JobRequest (which has
 *  segments) and the ANALYZE AnalyzeRequest (M4-S3, which has NONE — analyze PRODUCES
 *  them) can be estimated without those two unrelated request types sharing a base:
 *  an analyze job has no segments yet, so it lands on the flat ETA_BASE_S. */
function estimateEtaSeconds(payload: unknown): number {
  const segments = (payload as { segments?: unknown } | null)?.segments;
  const n = Array.isArray(segments) ? segments.length : 0;
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
      if (
        (seg as any)?.start !== undefined
        && (seg as any)?.end !== undefined
        && e0 <= s0
      ) {
        return 'segment start must be before end';
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
  const customInstructions = cfg?.promptProfile?.customInstructions;
  if (customInstructions !== undefined && customInstructions !== null) {
    if (typeof customInstructions !== 'string') return 'customInstructions must be a string';
    if (customInstructions.length > MAX_QWEN_CUSTOM_INSTRUCTION_CHARS) {
      return `customInstructions too long (>${MAX_QWEN_CUSTOM_INSTRUCTION_CHARS} chars)`;
    }
    if (hasLoneSurrogate(customInstructions)) return 'customInstructions is not well-formed UTF';
    if (/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(customInstructions)) {
      return 'customInstructions contains an unsupported control character';
    }
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

async function isAuthorizedAdminRequest(c: any, expectedToken: string | undefined): Promise<boolean> {
  const token = c.req.header('X-Admin-Token');
  return Boolean(
    expectedToken
    && token
    && await safeTokenEqual(token, expectedToken),
  );
}

// --- Financial Kill Switch admin endpoint ----------------------------------
// The standalone billing monitor (scripts/kill-switch-monitor.mjs) flips this
// flag when spend crosses the threshold; every job/register call then 503s.
app.post('/api/admin/kill-switch', async (c) => {
  // Fail-closed: unset server token or missing header => reject BEFORE any compare
  // (a missing header is not a secret-dependent branch — the caller already knows it).
  if (!(await isAuthorizedAdminRequest(c, c.env.ADMIN_TOKEN))) {
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
  // Coordinate the spend guard with the RunPod control lease. Arming first blocks and
  // revokes the controller lease, then commits the global switch. Clearing reverses the
  // order: jobs may queue while the controller remains blocked, but no Pod can resume
  // until both stores confirm the cleared state. Partial failure therefore fails closed.
  try {
    if (body.active) {
      await setWorkerControlBlocked(c.env, true);
      await setKillSwitch(c.env, true);
    } else {
      await setKillSwitch(c.env, false);
      await setWorkerControlBlocked(c.env, false);
    }
  } catch {
    if (body.active) background(c, triggerProviderTeardown(c.env));
    return c.json({ error: 'Kill switch control plane unavailable. Please retry.' }, 503);
  }
  if (body.active === true) {
    // M3-S6 (plan line 343-344): a tripped switch must stop paying for idle GPUs, not just new
    // dispatches. Fire the provider scale-to-zero hook in the BACKGROUND — the durable guard is
    // the switch STATE (already committed above), so a slow/unconfigured/failing teardown must
    // never delay or undo the ACTIVE response. Honest no-op when the hook is not configured.
    background(c, triggerProviderTeardown(c.env));
    return c.json({ killSwitch: 'ACTIVE' });
  }
  return c.json({ killSwitch: 'CLEARED' });
});

// RunPod controller heartbeat. The URL is accepted only through this authenticated
// server-to-server route, validated again by the registry, and never reflected by a
// public API. Generation ordering prevents a delayed old controller from replacing a
// newer target; the short validUntil makes a dead controller fail closed automatically.
app.post('/api/admin/worker-target/lease', async (c) => {
  if (!(await isAuthorizedAdminRequest(c, c.env.WORKER_TARGET_ADMIN_TOKEN))) {
    return c.json({ error: 'Forbidden' }, 403);
  }
  const body = await c.req.json().catch(() => null);
  if (!body) return c.json({ error: 'Invalid worker control lease contract' }, 400);

  try {
    if (await readKillSwitch(c.env)) {
      return c.json({ granted: false, outcome: 'kill_switch_active' }, 423);
    }
  } catch {
    return c.json({ error: 'Financial control state unavailable' }, 503);
  }

  try {
    const result = await acquireWorkerControlLease(c.env, body, Date.now());
    if (result.outcome === 'blocked') {
      return c.json({ granted: false, outcome: 'control_blocked' }, 423);
    }
    if (result.outcome === 'stale' || result.outcome === 'conflict') {
      return c.json({ granted: false, outcome: result.outcome }, 409);
    }
    return c.json({ granted: true, outcome: result.outcome });
  } catch (error) {
    if (error instanceof WorkerTargetValidationError) {
      return c.json({ error: 'Invalid worker control lease contract' }, 400);
    }
    return c.json({ error: 'Worker target registry unavailable' }, 503);
  }
});

// RunPod provisioning lock. This is intentionally separate from the Pod control
// lease above: a controller needs a lock before it knows the Pod ID returned by
// POST /pods. The stable slot/spec binding prevents duplicate paid creates.
app.post('/api/admin/worker-provision/lease', async (c) => {
  if (!(await isAuthorizedAdminRequest(c, c.env.WORKER_TARGET_ADMIN_TOKEN))) {
    return c.json({ error: 'Forbidden' }, 403);
  }
  const body = await c.req.json().catch(() => null);
  if (!body) return c.json({ error: 'Invalid worker provision lease contract' }, 400);

  try {
    if (await readKillSwitch(c.env)) {
      return c.json({ granted: false, outcome: 'kill_switch_active' }, 423);
    }
  } catch {
    return c.json({ error: 'Financial control state unavailable' }, 503);
  }

  try {
    const result = await acquireWorkerProvisionLease(c.env, body, Date.now());
    if (result.outcome === 'blocked') {
      return c.json({ granted: false, outcome: 'control_blocked' }, 423);
    }
    if (result.outcome === 'stale' || result.outcome === 'conflict' || result.outcome === 'recovery_required') {
      return c.json({ granted: false, outcome: result.outcome }, 409);
    }
    return c.json({ granted: true, outcome: result.outcome });
  } catch (error) {
    if (error instanceof WorkerTargetValidationError) {
      return c.json({ error: 'Invalid worker provision lease contract' }, 400);
    }
    return c.json({ error: 'Worker target registry unavailable' }, 503);
  }
});

app.get('/api/admin/worker-provision/lease', async (c) => {
  if (!(await isAuthorizedAdminRequest(c, c.env.WORKER_TARGET_ADMIN_TOKEN))) {
    return c.json({ error: 'Forbidden' }, 403);
  }
  try {
    const snapshot = await readWorkerProvisionIntent(c.env, Date.now());
    return c.json(snapshot);
  } catch {
    return c.json({ error: 'Worker target registry unavailable' }, 503);
  }
});

// Destructive recovery unlock. Operators must verify RunPod inventory and cleanup
// the exact intent-owned Pod before using this endpoint; the explicit confirmation
// prevents an accidental dashboard DELETE from erasing an unresolved create fence.
app.delete('/api/admin/worker-provision/lease', async (c) => {
  if (!(await isAuthorizedAdminRequest(c, c.env.WORKER_TARGET_ADMIN_TOKEN))) {
    return c.json({ error: 'Forbidden' }, 403);
  }
  const body = await c.req.json().catch(() => null);
  if (!body) return c.json({ error: 'Invalid worker provision recovery contract' }, 400);
  try {
    const result = await clearWorkerProvisionIntent(c.env, body);
    if (result.outcome === 'conflict') {
      return c.json({ cleared: false, outcome: result.outcome }, 409);
    }
    return c.json({ cleared: result.outcome === 'accepted', outcome: result.outcome });
  } catch (error) {
    if (error instanceof WorkerTargetValidationError) {
      return c.json({ error: 'Invalid worker provision recovery contract' }, 400);
    }
    return c.json({ error: 'Worker target registry unavailable' }, 503);
  }
});

app.post('/api/admin/worker-target', async (c) => {
  if (!(await isAuthorizedAdminRequest(c, c.env.WORKER_TARGET_ADMIN_TOKEN))) {
    return c.json({ error: 'Forbidden' }, 403);
  }
  const body = await c.req.json().catch(() => null);
  if (!body) return c.json({ error: 'Invalid worker target contract' }, 400);
  try {
    if (await readKillSwitch(c.env)) {
      return c.json({ accepted: false, outcome: 'kill_switch_active' }, 423);
    }
  } catch {
    return c.json({ error: 'Financial control state unavailable' }, 503);
  }
  try {
    const result = await publishWorkerTarget(c.env, body, Date.now());
    if (result.outcome === 'blocked') {
      return c.json({ accepted: false, outcome: 'control_blocked' }, 423);
    }
    if (
      result.outcome === 'stale'
      || result.outcome === 'conflict'
      || result.outcome === 'lease_required'
    ) {
      return c.json({ accepted: false, outcome: result.outcome }, 409);
    }
    return c.json({ accepted: true, outcome: result.outcome });
  } catch (error) {
    if (error instanceof WorkerTargetValidationError) {
      return c.json({ error: 'Invalid worker target contract' }, 400);
    }
    return c.json({ error: 'Worker target registry unavailable' }, 503);
  }
});

// Clear is generation-scoped. A late shutdown from an old controller cannot clear
// a target already owned by a newer controller generation.
app.delete('/api/admin/worker-target', async (c) => {
  if (!(await isAuthorizedAdminRequest(c, c.env.WORKER_TARGET_ADMIN_TOKEN))) {
    return c.json({ error: 'Forbidden' }, 403);
  }
  const body = await c.req.json().catch(() => null);
  try {
    const result = await clearWorkerTarget(c.env, body);
    if (result.outcome === 'stale' || result.outcome === 'conflict') {
      return c.json({ cleared: false, outcome: result.outcome }, 409);
    }
    return c.json({ cleared: true, outcome: result.outcome });
  } catch (error) {
    if (error instanceof WorkerTargetValidationError) {
      return c.json({ error: 'Invalid worker target clear contract' }, 400);
    }
    return c.json({ error: 'Worker target registry unavailable' }, 503);
  }
});

// --- Device registration (Station 1 enrolment) -----------------------------
app.post('/api/auth/register', async (c) => {
  if (await killSwitchBlocks(c.env)) {
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

  // Registration throttle per client IP (M3-S5: atomic check-and-consume via the
  // RateLimiter DO when bound; KV get-then-put fallback otherwise). Fail CLOSED (503) if a
  // bound limiter errors — a throttle that cannot answer must not wave registrations through.
  const ip = c.req.header('CF-Connecting-IP') || 'unknown';
  let registerAllowed: boolean;
  try {
    ({ allowed: registerAllowed } = await enforceRateLimit(
      c.env,
      `rl:register:${ip}`,
      REGISTER_RATE_LIMIT,
      REGISTER_WINDOW_S * 1000,
      Date.now(),
    ));
  } catch {
    return c.json({ error: 'Rate limiter unavailable. Please try again later.' }, 503);
  }
  if (!registerAllowed) {
    return c.json({ error: 'Too Many Registrations. Please try again later.' }, 429);
  }

  const deviceId = crypto.randomUUID();
  await c.env.KV_CACHE.put(`device:${deviceId}`, JSON.stringify(body.publicKeyJwk));

  return c.json({ deviceId, message: 'Device Registered Successfully' }, 201);
});

// --- Job creation (Station 1 verification + async accept) ------------------
app.post('/api/jobs/create', async (c) => {
  if (await killSwitchBlocks(c.env)) {
    return c.json({ error: 'Service temporarily unavailable (Kill Switch active)' }, 503);
  }

  // M3-S2 (item 4): CANONICAL Zero-Trust auth. One helper does header presence,
  // device-key lookup, ts/nonce presence + bound, signature-verify over
  // METHOD+PATH+bodyHash+X-Timestamp+X-Nonce, and the replay window on the SIGNED
  // timestamp — returning the authenticated device + the raw body it already read,
  // so we neither re-read the request nor re-derive the public key.
  const auth = await authenticateSignedRequest(c);
  if (!auth.ok) {
    return c.json({ error: auth.error }, auth.status);
  }
  const { deviceId, rawBody } = auth;

  let payloadObj: JobRequest;
  try {
    payloadObj = JSON.parse(rawBody);
  } catch {
    return c.json({ error: 'Invalid JSON' }, 400);
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

  // M3-S1 (item 1 / acceptance A1): shared RUNTIME-schema gate. A signed payload
  // that is MISSING or wrong-typing a REQUIRED field (config, videoAudioKey,
  // videoAudioMd5, speakerMapping, …) is rejected HERE — still before the
  // idempotency read / throttle / dispatch (jobKey below) — instead of slipping
  // through validateJobSize's optional chaining, returning 202, and only
  // detonating as a TypeError inside the background dispatch. The single schema
  // lives in @dichomnion/shared-types so client and Gateway can't drift.
  //
  // ORDERING: this runs AFTER validateJobSize on purpose. validateJobSize owns the
  // deep segment/voice-map bounds AND emits SPECIFIC diagnostic messages (e.g.
  // "speakerMapping values must be strings", Đợt 20 F10) for fields that are
  // PRESENT-but-malformed; letting it run first preserves those precise messages.
  // The schema then adds only the NET-NEW rejections validateJobSize's optional
  // chaining lets slip — a top-level required field that is entirely absent. Its
  // message is GENERIC (no zod field detail) so it never echoes user data back,
  // pre-satisfying item 11's sanitized-reason spirit.
  if (!JobRequestSchema.safeParse(payloadObj).success) {
    return c.json({ error: 'Invalid request: required field missing or malformed' }, 400);
  }

  // M3-S9 — DISPATCH-TRANSPORT bound. Every accepted job's payload later rides a Cloudflare
  // Queue message (ADR 0001 durable dispatch) as `{ deviceId, jobId, payload }`, whose HARD
  // limit is 128 KiB. The CHAR-level field caps above miss two byte-level vectors that pass
  // every one of them: (a) the `original_text` hole (validateJobSize measures `text`, never
  // original_text when text is present) and (b) sheer segment COUNT (2000 minimal segments
  // ≈ 132 KiB). Measure the EXACT envelope that dispatch would send and refuse an
  // undispatchable payload HERE — before the idempotency read / DO create / throttle / send —
  // so a `.send()` that would THROW (stranding the job as an un-rendered orphan) becomes a
  // clean 413. Enforced uniformly (both the DO/Queue and legacy paths) so any accepted job is
  // guaranteed dispatchable; env MAX_QUEUE_PAYLOAD_BYTES tunes the margin per deploy.
  const maxQueueBytes = Number(c.env.MAX_QUEUE_PAYLOAD_BYTES) || MAX_QUEUE_PAYLOAD_BYTES;
  const dispatchEnvelope = JSON.stringify({ deviceId, jobId: payloadObj.jobId, payload: payloadObj });
  if (new TextEncoder().encode(dispatchEnvelope).length > maxQueueBytes) {
    return c.json({ error: 'Job payload too large to dispatch' }, 413);
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
      // Orphan heal (M1 review, Bug A). coordinatorCreate commits the DO record
      // BEFORE the dispatch send, and those two are non-atomic: a send that fails
      // after the commit leaves a QUEUED job that nothing will ever render. A
      // still-QUEUED record WITHOUT the `enqueued` confirmation is exactly that
      // orphan — re-enqueue it (the consumer's dispatch lease dedupes any redundant
      // message, so a double-send can never cause a double GPU render).
      if (seen.status === 'QUEUED' && seen.enqueued !== true) {
        await c.env.JOB_DISPATCH_QUEUE.send({ deviceId, jobId: payloadObj.jobId, payload: payloadObj });
        try {
          await coordinatorMarkEnqueued(stub);
        } catch {
          /* best-effort: an unmarked-but-enqueued job just re-enqueues once more (deduped). */
        }
      }
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

    // Genuinely new → throttle. M3-S5 (Bug D): this is now an ATOMIC check-AND-consume in
    // the RateLimiter DO (no more get-then-put lost-update). Fail CLOSED (503) if a bound
    // limiter errors. A same-jobId race loser that gets rejected by coordinatorCreate below
    // has already consumed one quota unit here — a rare, harmless over-count (that device
    // asked for exactly that job) traded for closing the concurrent under-count that let a
    // burst blow past the cap.
    let jobsAllowed: boolean;
    try {
      ({ allowed: jobsAllowed } = await enforceRateLimit(
        c.env,
        jobsRlKey,
        jobsLimit,
        jobsWindow * 1000,
        Date.now(),
      ));
    } catch {
      return c.json({ error: 'Rate limiter unavailable. Please try again later.' }, 503);
    }
    if (!jobsAllowed) {
      // M2-S5g (#2): this create is REFUSED, so its already-uploaded input is orphaned.
      // Reclaim it — but only if no record committed meanwhile (a concurrent winner would
      // own a live input). coordinatorGet is strongly consistent (DO storage), so this
      // re-check is authoritative, unlike the eventually-consistent legacy KV path.
      if (!(await coordinatorGet(stub))) await reapUncreatedInput(c.env, deviceId, payloadObj.jobId);
      return c.json({ error: 'Too Many Jobs. Rate limit exceeded; please try again later.' }, 429);
    }

    // Atomic check-and-set (W3 fix). Also seeds the KV `job:` projection to QUEUED
    // so the existing poll/download contract keeps reading KV unchanged.
    const created = await coordinatorCreate(stub, { deviceId, jobId: payloadObj.jobId, etaSeconds });
    if (created.idempotent) {
      // Lost a concurrent race for the same jobId — the winner enqueued; we must not
      // double-dispatch (no double GPU render). Quota was consumed at the gate above.
      return c.json(
        {
          message: 'Job Accepted securely!',
          jobId: payloadObj.jobId,
          status: created.record.status,
          etaSeconds: created.record.etaSeconds,
          progress: publicJobProgress(created.record.progress),
          idempotent: true,
        },
        202,
      );
    }

    // Durable hand-off: the Queue consumer (handleJobQueue) drives dispatch with
    // at-least-once redelivery, so an evicted Worker can never silently lose the job.
    await c.env.JOB_DISPATCH_QUEUE.send({ deviceId, jobId: payloadObj.jobId, payload: payloadObj });
    // Confirm the hand-off (M1 review, Bug A) so a later peek can distinguish this
    // job from an orphan a failed send left behind. Best-effort: if the send above
    // succeeded but this mark fails, a retry simply re-enqueues once (consumer dedupes).
    try {
      await coordinatorMarkEnqueued(stub);
    } catch {
      /* best-effort marker */
    }
    return c.json(
      {
        message: 'Job Accepted securely!',
        jobId: payloadObj.jobId,
        status: 'QUEUED',
        etaSeconds,
        progress: publicJobProgress(created.record.progress),
      },
      202,
    );
  }

  // ── Legacy path (no durable bindings): KV idempotency + background dispatch ──
  // A real, still-functional fallback (the pre-M1 behavior). Production binds the
  // DO + Queue, so this runs only in KV-only environments (e.g. the existing unit
  // tests). Idempotency: a re-sent job returns the existing record, never re-dispatches.
  const existing = await c.env.KV_CACHE.get<{ status: string; etaSeconds?: number; progress?: JobProgress }>(jobKey, { type: 'json' });
  if (existing) {
    return c.json(
      {
        message: 'Job Accepted securely!',
        jobId: payloadObj.jobId,
        status: existing.status,
        etaSeconds: existing.etaSeconds,
        progress: publicJobProgress(existing.progress),
        idempotent: true,
      },
      202,
    );
  }

  // M3-S5 (Bug D): atomic check-AND-consume (RateLimiter DO when bound; KV fallback
  // otherwise). Fail CLOSED (503) if a bound limiter errors.
  let jobsAllowedLegacy: boolean;
  try {
    ({ allowed: jobsAllowedLegacy } = await enforceRateLimit(
      c.env,
      jobsRlKey,
      jobsLimit,
      jobsWindow * 1000,
      Date.now(),
    ));
  } catch {
    return c.json({ error: 'Rate limiter unavailable. Please try again later.' }, 503);
  }
  if (!jobsAllowedLegacy) {
    // M2-S5g (#2): refused → reclaim this job's orphaned input. We reach here only after
    // the idempotency read above returned null (no record owns the input in this isolate).
    await reapUncreatedInput(c.env, deviceId, payloadObj.jobId);
    return c.json({ error: 'Too Many Jobs. Rate limit exceeded; please try again later.' }, 429);
  }

  const initialProgress = stageProgress('queued');
  await c.env.KV_CACHE.put(
    jobKey,
    JSON.stringify({ status: 'QUEUED', createdAt: Date.now(), etaSeconds, progress: initialProgress }),
    { expirationTtl: JOB_TTL_S },
  );

  // Dispatch to the GPU worker asynchronously (Graceful Degradation: accept now,
  // process later). We never block the client on GPU time.
  background(c, dispatchToWorker(c.env, payloadObj, jobKey));

  return c.json(
    {
      message: 'Job Accepted securely!',
      jobId: payloadObj.jobId,
      status: 'QUEUED',
      etaSeconds,
      progress: initialProgress,
    },
    202,
  );
});

// --- M4-S3: ANALYZE submission (Human-in-the-Loop entry point) --------------
// POST /api/jobs/analyze is the FIRST of the two-step HITL pipeline (ADR 0002). It
// mirrors /api/jobs/create's Zero-Trust surface EXACTLY — canonical signature verify,
// jobId presence + lone-surrogate gate, shared runtime-schema validation, DO atomic
// idempotency, durable Queue hand-off — the ONLY differences are:
//   • the request schema is AnalyzeRequestSchema (phase literal 'ANALYZE', and it carries
//     the client's per-device ECDH encryptionPublicKey, ADR 0002); and
//   • the enqueued message is PHASE-TAGGED so the consumer drives /api/worker/analyze
//     (whose compute rests at AWAITING_REVIEW), never the render pipeline (which ends DONE).
// The Gateway forwards encryptionPublicKey verbatim and only ever sees ciphertext (M4
// mandate #4). Analyze REQUIRES the durable bindings (the DO is the lineage authority and
// the Queue is the hand-off) — absent them we 503 HONESTLY rather than fake a KV-only
// analyze (No-Fake-Success); there is deliberately no legacy KV path here.
type AnalyzeRerunRequest = AnalyzeRequest & { baseRevision: number };
const AnalyzeRerunRequestSchema = AnalyzeRequestSchema.extend({
  baseRevision: ApproveRequestSchema.shape.baseRevision,
});

app.post('/api/jobs/analyze', async (c) => {
  if (await killSwitchBlocks(c.env)) {
    return c.json({ error: 'Service temporarily unavailable (Kill Switch active)' }, 503);
  }

  const auth = await authenticateSignedRequest(c);
  if (!auth.ok) {
    return c.json({ error: auth.error }, auth.status);
  }
  const { deviceId, rawBody } = auth;

  let payloadObj: AnalyzeRequest;
  try {
    payloadObj = JSON.parse(rawBody);
  } catch {
    return c.json({ error: 'Invalid JSON' }, 400);
  }

  if (!payloadObj.jobId || typeof payloadObj.jobId !== 'string') {
    return c.json({ error: 'Missing jobId' }, 400);
  }
  // Đợt 25 AMP-JOBID-SURROGATE-01 (mirrored): a lone-surrogate jobId would ride the JWT
  // `jobId` claim + worker body job_id verbatim and detonate on the worker's JSON reply
  // serialization (uncaught 500 → retry-amplified GPU spend). Reject pre-dispatch, 400.
  if (hasLoneSurrogate(payloadObj.jobId)) {
    return c.json({ error: 'jobId is not well-formed UTF' }, 400);
  }

  // Shared runtime-schema gate (M3-S1 discipline) BEFORE any durable work. AnalyzeRequestSchema
  // pins phase to the literal 'ANALYZE' and REQUIRES a non-empty encryptionPublicKey, so this one
  // check subsumes both "missing enc pubkey → 400" and "a render (phase!=='ANALYZE') payload cannot
  // ride the analyze route → 400". Generic message (no zod field detail) → never echoes user data.
  const parsedAnalyze = AnalyzeRequestSchema.safeParse(payloadObj);
  if (!parsedAnalyze.success) {
    return c.json({ error: 'Invalid analyze request: required field missing or malformed' }, 400);
  }
  payloadObj = parsedAnalyze.data;

  // Analyze REQUIRES durable coordination (DO lineage authority + Queue hand-off). No
  // legacy KV analyze exists — refuse honestly rather than half-run one.
  if (!c.env.JOB_COORDINATOR || !c.env.JOB_DISPATCH_QUEUE) {
    return c.json({ error: 'Analyze requires durable coordination (unavailable)' }, 503);
  }

  // Bind idempotency to the normalized business payload, not just jobId. Timestamp is an
  // authentication freshness field and may legitimately change when retrying an ambiguous
  // response; every compute-affecting field remains hash-bound.
  const { timestamp: _analyzeTimestamp, ...stableAnalyzeRequest } = payloadObj;
  void _analyzeTimestamp;
  const requestFingerprint = await sha256Hex(deterministicStringify(stableAnalyzeRequest));

  // M3-S9 dispatch-transport bound — measure the EXACT phase-tagged envelope the consumer
  // will send and refuse an undispatchable payload here (a .send() that would throw would
  // strand the job as an un-rendered orphan). AnalyzeRequest is small (no segments), but the
  // guard is uniform with /create so every accepted job is guaranteed dispatchable.
  const maxQueueBytes = Number(c.env.MAX_QUEUE_PAYLOAD_BYTES) || MAX_QUEUE_PAYLOAD_BYTES;
  const analyzeEnvelope = JSON.stringify({
    deviceId,
    jobId: payloadObj.jobId,
    phase: 'ANALYZE',
    payload: payloadObj,
  });
  if (new TextEncoder().encode(analyzeEnvelope).length > maxQueueBytes) {
    return c.json({ error: 'Analyze payload too large to dispatch' }, 413);
  }

  const etaSeconds = estimateEtaSeconds(payloadObj);
  const jobsLimit = Number(c.env.JOBS_RATE_LIMIT) || JOBS_RATE_LIMIT;
  const jobsWindow = Number(c.env.JOBS_WINDOW_S) || JOBS_WINDOW_S;
  const jobsRlKey = `rl:jobs:${deviceId}`;

  const stub = coordinatorStub(c.env.JOB_COORDINATOR, deviceId, payloadObj.jobId);

  // Idempotency BEFORE throttle (mirrors /create): a re-sent analyze returns its CURRENT
  // state read-only and consumes no quota. Orphan heal: a still-QUEUED record WITHOUT the
  // `enqueued` confirmation is an orphan a failed send left behind — re-enqueue (the
  // consumer's dispatch lease dedupes, so a double-send can never double-run analyze).
  const seen = await coordinatorGet(stub);
  if (seen) {
    if (seen.phase !== 'ANALYZE' || seen.requestFingerprint !== requestFingerprint) {
      return c.json({ error: 'jobId is already bound to a different request' }, 409);
    }
    if (seen.status === 'QUEUED' && seen.enqueued !== true) {
      await c.env.JOB_DISPATCH_QUEUE.send({
        deviceId,
        jobId: payloadObj.jobId,
        phase: 'ANALYZE',
        payload: payloadObj,
      });
      try {
        await coordinatorMarkEnqueued(stub);
      } catch {
        /* best-effort: an unmarked-but-enqueued job just re-enqueues once more (deduped). */
      }
    }
    return c.json(
      {
        message: 'Analyze Accepted securely!',
        jobId: payloadObj.jobId,
        status: seen.status,
        etaSeconds: seen.etaSeconds,
        idempotent: true,
      },
      202,
    );
  }

  // Genuinely new → atomic check-AND-consume throttle (M3-S5 RateLimiter DO). Fail CLOSED
  // (503) if a bound limiter errors.
  let analyzeAllowed: boolean;
  try {
    ({ allowed: analyzeAllowed } = await enforceRateLimit(
      c.env,
      jobsRlKey,
      jobsLimit,
      jobsWindow * 1000,
      Date.now(),
    ));
  } catch {
    return c.json({ error: 'Rate limiter unavailable. Please try again later.' }, 503);
  }
  if (!analyzeAllowed) {
    // Refused → reclaim this job's already-uploaded input (only if no record committed
    // meanwhile — coordinatorGet is strongly consistent, so this re-check is authoritative).
    if (!(await coordinatorGet(stub))) await reapUncreatedInput(c.env, deviceId, payloadObj.jobId);
    return c.json({ error: 'Too Many Jobs. Rate limit exceeded; please try again later.' }, 429);
  }

  // Atomic check-and-set, TAGGED as the ANALYZE phase so the DO record + KV projection
  // both carry phase='ANALYZE' (the poll/render-lineage contract reads it).
  const created = await coordinatorCreate(stub, {
    deviceId,
    jobId: payloadObj.jobId,
    etaSeconds,
    phase: 'ANALYZE',
    meta: { requestFingerprint, analyzeRevision: 0 },
  });
  if (created.idempotent) {
    // Lost a concurrent race. Only the byte-equivalent business request is idempotent;
    // a different payload under the same jobId must never inherit the winner's state.
    if (created.record.phase !== 'ANALYZE' || created.record.requestFingerprint !== requestFingerprint) {
      return c.json({ error: 'jobId is already bound to a different request' }, 409);
    }
    return c.json(
      {
        message: 'Analyze Accepted securely!',
        jobId: payloadObj.jobId,
        status: created.record.status,
        etaSeconds: created.record.etaSeconds,
        progress: publicJobProgress(created.record.progress),
        idempotent: true,
      },
      202,
    );
  }

  // Durable hand-off: a PHASE-TAGGED message so handleJobQueue routes to the analyze dispatch.
  await c.env.JOB_DISPATCH_QUEUE.send({
    deviceId,
    jobId: payloadObj.jobId,
    phase: 'ANALYZE',
    payload: payloadObj,
  });
  try {
    await coordinatorMarkEnqueued(stub);
  } catch {
    /* best-effort marker */
  }
  return c.json(
    {
      message: 'Analyze Accepted securely!',
      jobId: payloadObj.jobId,
      status: 'QUEUED',
      etaSeconds,
      progress: publicJobProgress(created.record.progress),
    },
    202,
  );
});

// Explicit producer semantics for re-analysis on the SAME lineage. The signed body is
// the normal AnalyzeRequest plus baseRevision; jobId must equal the path lineage. The
// lineage DO is the only component allowed to reopen AWAITING_REVIEW, and does so with a
// strong CAS. A successful worker attempt becomes a revision strictly greater than base.
app.post('/api/jobs/:analyzeJobId/rerun', async (c) => {
  if (await killSwitchBlocks(c.env)) {
    return c.json({ error: 'Service temporarily unavailable (Kill Switch active)' }, 503);
  }
  const auth = await authenticateSignedRequest(c);
  if (!auth.ok) return c.json({ error: auth.error }, auth.status);
  const { deviceId, rawBody } = auth;
  const analyzeJobId = c.req.param('analyzeJobId');

  let payloadObj: AnalyzeRerunRequest;
  try {
    payloadObj = AnalyzeRerunRequestSchema.parse(JSON.parse(rawBody)) as AnalyzeRerunRequest;
  } catch {
    return c.json({ error: 'Invalid analyze rerun request: required field missing or malformed' }, 400);
  }
  if (payloadObj.jobId !== analyzeJobId || hasLoneSurrogate(payloadObj.jobId)) {
    return c.json({ error: 'Rerun lineage does not match this analyze job' }, 400);
  }
  if (!c.env.JOB_COORDINATOR || !c.env.JOB_DISPATCH_QUEUE) {
    return c.json({ error: 'Analyze rerun requires durable coordination (unavailable)' }, 503);
  }

  const revisionBase = payloadObj.baseRevision;
  const envelope = JSON.stringify({
    deviceId,
    jobId: analyzeJobId,
    phase: 'ANALYZE',
    revisionBase,
    payload: payloadObj,
  });
  if (
    new TextEncoder().encode(envelope).length >
    (Number(c.env.MAX_QUEUE_PAYLOAD_BYTES) || MAX_QUEUE_PAYLOAD_BYTES)
  ) {
    return c.json({ error: 'Analyze rerun payload too large to dispatch' }, 413);
  }

  const { timestamp: _timestamp, ...stableRerun } = payloadObj;
  void _timestamp;
  const requestFingerprint = await sha256Hex(deterministicStringify(stableRerun));
  const stub = coordinatorStub(c.env.JOB_COORDINATOR, deviceId, analyzeJobId);
  const before = await coordinatorGet(stub);

  // Idempotent orphan healing must not spend rate-limit quota. A new generation does.
  const alreadyPrepared =
    before?.phase === 'ANALYZE' &&
    before.status === 'QUEUED' &&
    before.rerunBaseRevision === payloadObj.baseRevision &&
    before.requestFingerprint === requestFingerprint;
  if (!alreadyPrepared) {
    let allowed: boolean;
    try {
      ({ allowed } = await enforceRateLimit(
        c.env,
        `rl:jobs:${deviceId}`,
        Number(c.env.JOBS_RATE_LIMIT) || JOBS_RATE_LIMIT,
        (Number(c.env.JOBS_WINDOW_S) || JOBS_WINDOW_S) * 1000,
        Date.now(),
      ));
    } catch {
      return c.json({ error: 'Rate limiter unavailable. Please try again later.' }, 503);
    }
    if (!allowed) {
      return c.json({ error: 'Too Many Jobs. Rate limit exceeded; please try again later.' }, 429);
    }
  }

  const prepared = await coordinatorPrepareAnalyzeRerun(stub, {
    baseRevision: payloadObj.baseRevision,
    requestFingerprint,
  });
  if (prepared.outcome === 'not_found') return c.json({ error: 'Job not found' }, 404);
  if (prepared.outcome === 'stale') {
    return c.json(
      {
        error: 'Stale rerun: analyze revision has moved',
        currentRevision: prepared.currentRevision,
        baseRevision: payloadObj.baseRevision,
      },
      409,
    );
  }
  if (prepared.outcome === 'render_active') {
    return c.json({ error: 'Analyze lineage has an active render' }, 409);
  }
  if (prepared.outcome === 'conflict') {
    return c.json({ error: 'Analyze rerun is already bound to a different request' }, 409);
  }
  if (prepared.outcome === 'not_awaiting_review') {
    return c.json({ error: 'Job is not awaiting review', status: prepared.record?.status }, 409);
  }

  if (prepared.record?.enqueued !== true) {
    await c.env.JOB_DISPATCH_QUEUE.send({
      deviceId,
      jobId: analyzeJobId,
      phase: 'ANALYZE',
      revisionBase,
      payload: payloadObj,
    });
    try {
      await coordinatorMarkEnqueued(stub);
    } catch {
      /* best-effort marker; exact retry heals the hand-off */
    }
  }
  return c.json(
    {
      message: 'Analyze rerun accepted securely!',
      jobId: analyzeJobId,
      status: 'QUEUED',
      baseRevision: payloadObj.baseRevision,
      targetRevision: prepared.targetRevision,
      ...(prepared.outcome === 'idempotent' ? { idempotent: true } : {}),
    },
    202,
  );
});

// --- HITL RENDER submission (M4-S4) -----------------------------------------
// The SECOND half of the Human-in-the-Loop flow. After a client reviews an ANALYZE result
// and APPROVES a canonical manifest (POST /api/jobs/:analyzeJobId/approve), it submits a
// RenderRequest BOUND to that approval by (analyzeJobId, approvedManifestHash). This mirrors
// /api/jobs/analyze's Zero-Trust surface exactly (canonical signature, jobId presence +
// lone-surrogate gate, schema validation, durable hand-off, idempotency, kill-switch) and
// adds ONE render-specific gate: it refuses (409) to enqueue a render that does not match a
// stored ApprovedRevision for THIS device's analyze lineage — you cannot render before
// approving, and you cannot ride a DIFFERENT approval's hash. The created job carries
// phase='RENDER' + its immutable lineage in the coordinator meta, so the queue consumer
// routes it to the render dispatch (a separate increment) with the approval it is provably
// bound to.
app.post('/api/jobs/render', async (c) => {
  if (await killSwitchBlocks(c.env)) {
    return c.json({ error: 'Service temporarily unavailable (Kill Switch active)' }, 503);
  }

  const auth = await authenticateSignedRequest(c);
  if (!auth.ok) {
    return c.json({ error: auth.error }, auth.status);
  }
  const { deviceId, rawBody } = auth;

  let payloadObj: RenderRequest;
  try {
    payloadObj = JSON.parse(rawBody);
  } catch {
    return c.json({ error: 'Invalid JSON' }, 400);
  }

  if (!payloadObj.jobId || typeof payloadObj.jobId !== 'string') {
    return c.json({ error: 'Missing jobId' }, 400);
  }
  // Mirror AMP-JOBID-SURROGATE-01: a lone-surrogate jobId would ride the JWT claim + worker
  // body verbatim and detonate on JSON serialization (uncaught 500 → retry-amplified spend).
  if (hasLoneSurrogate(payloadObj.jobId)) {
    return c.json({ error: 'jobId is not well-formed UTF' }, 400);
  }

  // Shared runtime-schema gate (M3-S1 discipline). RenderRequestSchema pins phase to the
  // literal 'RENDER', so this one check subsumes both "malformed render payload → 400" and
  // "an analyze (phase!=='RENDER') payload cannot ride the render route → 400". Generic
  // message (no zod field detail) so it never echoes user data.
  const parsedRender = RenderRequestSchema.safeParse(payloadObj);
  if (!parsedRender.success) {
    return c.json({ error: 'Invalid render request: required field missing or malformed' }, 400);
  }
  payloadObj = parsedRender.data;

  // Render REQUIRES durable coordination (DO lineage authority + Queue hand-off). No legacy
  // KV render exists — refuse honestly rather than half-run one.
  if (!c.env.JOB_COORDINATOR || !c.env.JOB_DISPATCH_QUEUE) {
    return c.json({ error: 'Render requires durable coordination (unavailable)' }, 503);
  }

  const { timestamp: _renderTimestamp, ...stableRenderRequest } = payloadObj;
  void _renderTimestamp;
  const requestFingerprint = await sha256Hex(deterministicStringify(stableRenderRequest));

  // Approval authorization comes from the ANALYZE lineage DO, never the eventually
  // consistent KV projection. The dispatcher still re-hashes the R2 manifest bytes.
  const lineageStub = coordinatorStub(c.env.JOB_COORDINATOR, deviceId, payloadObj.analyzeJobId);
  const approval = await coordinatorGetApproval(lineageStub);
  if (!approval || approval.approvedManifestHash !== payloadObj.approvedManifestHash) {
    return c.json({ error: 'No matching approval for this analyze lineage' }, 409);
  }

  // M3-S9 dispatch-transport bound — measure the EXACT phase-tagged envelope the consumer will
  // send and refuse an undispatchable payload here (a .send() that would throw would strand
  // the job as an un-rendered orphan).
  const maxQueueBytes = Number(c.env.MAX_QUEUE_PAYLOAD_BYTES) || MAX_QUEUE_PAYLOAD_BYTES;
  const renderEnvelope = JSON.stringify({
    deviceId,
    jobId: payloadObj.jobId,
    phase: 'RENDER',
    payload: payloadObj,
  });
  if (new TextEncoder().encode(renderEnvelope).length > maxQueueBytes) {
    return c.json({ error: 'Render payload too large to dispatch' }, 413);
  }

  const etaSeconds = estimateEtaSeconds(payloadObj);
  const jobsLimit = Number(c.env.JOBS_RATE_LIMIT) || JOBS_RATE_LIMIT;
  const jobsWindow = Number(c.env.JOBS_WINDOW_S) || JOBS_WINDOW_S;
  const jobsRlKey = `rl:jobs:${deviceId}`;

  const stub = coordinatorStub(c.env.JOB_COORDINATOR, deviceId, payloadObj.jobId);

  // Idempotency BEFORE throttle (mirrors analyze): a re-sent render returns its CURRENT state
  // read-only and consumes no quota. Orphan heal: a still-QUEUED record WITHOUT the `enqueued`
  // confirmation is an orphan a failed send left behind — re-enqueue (the consumer's dispatch
  // lease dedupes, so a double-send can never double-run the render).
  const seen = await coordinatorGet(stub);
  if (seen) {
    if (seen.phase !== 'RENDER' || seen.requestFingerprint !== requestFingerprint) {
      return c.json({ error: 'jobId is already bound to a different request' }, 409);
    }
    if (seen.status === 'QUEUED' && seen.enqueued !== true) {
      await c.env.JOB_DISPATCH_QUEUE.send({
        deviceId,
        jobId: payloadObj.jobId,
        phase: 'RENDER',
        payload: payloadObj,
      });
      try {
        await coordinatorMarkEnqueued(stub);
      } catch {
        /* best-effort: an unmarked-but-enqueued job just re-enqueues once more (deduped). */
      }
    }
    return c.json(
      {
        message: 'Render Accepted securely!',
        jobId: payloadObj.jobId,
        status: seen.status,
        etaSeconds: seen.etaSeconds,
        idempotent: true,
      },
      202,
    );
  }

  // Genuinely new → atomic check-AND-consume throttle (M3-S5 RateLimiter DO). Fail CLOSED
  // (503) if a bound limiter errors. On refusal we do NOT reap input: a render reuses the
  // ANALYZE lineage's input object (uploaded once under the analyze key) — it is not this
  // render job's to reclaim (that would delete an object a re-approved render still needs).
  let renderAllowed: boolean;
  try {
    ({ allowed: renderAllowed } = await enforceRateLimit(
      c.env,
      jobsRlKey,
      jobsLimit,
      jobsWindow * 1000,
      Date.now(),
    ));
  } catch {
    return c.json({ error: 'Rate limiter unavailable. Please try again later.' }, 503);
  }
  if (!renderAllowed) {
    return c.json({ error: 'Too Many Jobs. Rate limit exceeded; please try again later.' }, 429);
  }

  // Strong single-active policy on the ANALYZE lineage. This claim is taken BEFORE the
  // render job is created, so two distinct render jobIds can never both become runnable.
  const renderClaim = await coordinatorClaimRender(lineageStub, {
    renderJobId: payloadObj.jobId,
    approvedManifestHash: payloadObj.approvedManifestHash,
    now: Date.now(),
  });
  if (renderClaim.outcome !== 'acquired' && renderClaim.outcome !== 'idempotent') {
    if (renderClaim.outcome === 'no_approval' || renderClaim.outcome === 'not_found') {
      return c.json({ error: 'No matching approval for this analyze lineage' }, 409);
    }
    if (renderClaim.outcome === 'busy') {
      return c.json({ error: 'Another render is active for this analyze lineage' }, 409);
    }
    if (renderClaim.outcome === 'consumed') {
      return c.json({ error: 'Analyze revision has already been consumed by a render' }, 409);
    }
    return c.json({ error: 'Analyze lineage is not awaiting review' }, 409);
  }

  // Atomic check-and-set, TAGGED phase='RENDER' with the IMMUTABLE approval lineage seeded in
  // meta so the DO record + KV projection both bind the render to its approved analyze
  // revision — a render job is provably bound to its approval before it ever dispatches.
  let created;
  try {
    created = await coordinatorCreate(stub, {
      deviceId,
      jobId: payloadObj.jobId,
      etaSeconds,
      phase: 'RENDER',
      meta: {
        analyzeJobId: payloadObj.analyzeJobId,
        approvedManifestHash: payloadObj.approvedManifestHash,
        requestFingerprint,
      },
    });
  } catch (error) {
    if (renderClaim.outcome === 'acquired') {
      await coordinatorReleaseRender(lineageStub, { renderJobId: payloadObj.jobId }).catch(() => {});
    }
    throw error;
  }
  if (created.idempotent) {
    if (created.record.phase !== 'RENDER' || created.record.requestFingerprint !== requestFingerprint) {
      if (renderClaim.outcome === 'acquired') {
        await coordinatorReleaseRender(lineageStub, { renderJobId: payloadObj.jobId }).catch(() => {});
      }
      return c.json({ error: 'jobId is already bound to a different request' }, 409);
    }
    return c.json(
      {
        message: 'Render Accepted securely!',
        jobId: payloadObj.jobId,
        status: created.record.status,
        etaSeconds: created.record.etaSeconds,
        progress: publicJobProgress(created.record.progress),
        idempotent: true,
      },
      202,
    );
  }

  // Durable hand-off: a PHASE-TAGGED message so handleJobQueue routes to the render dispatch.
  await c.env.JOB_DISPATCH_QUEUE.send({
    deviceId,
    jobId: payloadObj.jobId,
    phase: 'RENDER',
    payload: payloadObj,
  });
  try {
    await coordinatorMarkEnqueued(stub);
  } catch {
    /* best-effort marker */
  }
  return c.json(
    {
      message: 'Render Accepted securely!',
      jobId: payloadObj.jobId,
      status: 'QUEUED',
      etaSeconds,
      progress: publicJobProgress(created.record.progress),
    },
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
  if (await killSwitchBlocks(c.env)) {
    return c.json({ error: 'Service temporarily unavailable (Kill Switch active)' }, 503);
  }

  // M3-S2 (item 4): CANONICAL Zero-Trust auth — SAME verifier as /api/jobs/create,
  // now binding method+path so a presign signature can't be borrowed onto create
  // (or vice versa). Returns the authenticated device + the raw body already read.
  const auth = await authenticateSignedRequest(c);
  if (!auth.ok) {
    return c.json({ error: auth.error }, auth.status);
  }
  const { deviceId, rawBody } = auth;

  let reqObj: { jobId?: unknown; timestamp?: unknown };
  try {
    reqObj = JSON.parse(rawBody);
  } catch {
    return c.json({ error: 'Invalid JSON' }, 400);
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
  // M3-S5 (Bug D): atomic check-AND-consume (RateLimiter DO when bound; KV fallback
  // otherwise). Fail CLOSED (503) if a bound limiter errors.
  const presignLimit = Number(c.env.PRESIGN_RATE_LIMIT) || PRESIGN_RATE_LIMIT;
  const presignWindow = Number(c.env.PRESIGN_WINDOW_S) || PRESIGN_WINDOW_S;
  let presignAllowed: boolean;
  try {
    ({ allowed: presignAllowed } = await enforceRateLimit(
      c.env,
      `rl:presign:${deviceId}`,
      presignLimit,
      presignWindow * 1000,
      Date.now(),
    ));
  } catch {
    return c.json({ error: 'Rate limiter unavailable. Please try again later.' }, 503);
  }
  if (!presignAllowed) {
    return c.json({ error: 'Too Many Upload Requests. Rate limit exceeded; please try again later.' }, 429);
  }

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
  // M3-S3 (A2): the caller must PROVE ownership of the device with a canonical
  // signature (method+path+bodyHash+ts+nonce), not merely assert an X-Device-Id
  // header. We then scope the KV lookup to the AUTHENTICATED device, so knowing a
  // victim's (deviceId, jobId) no longer leaks their job — a spoofed X-Device-Id
  // fails signature verification (403) before any lookup happens.
  const auth = await authenticateSignedRequest(c);
  if (!auth.ok) return c.json({ error: auth.error }, auth.status);
  const deviceId = auth.deviceId;
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
      const terminalProgress = publicJobProgress(record.progress);
      return c.json({
        jobId,
        ...record,
        status: 'FINALIZING',
        progress: stageProgress('finalizing', terminalProgress?.attempt, terminalProgress?.updatedAt),
      });
    }
    const inner = stored.result ?? {};
    // Allowlist RÕ RÀNG các trường trả cho client — KHÔNG denylist (denylist bỏ sót nếu
    // kho lưu thêm trường nhạy cảm mới). Artifact readiness/tải xuống dựa trên result_key
    // (khóa R2 bền vững đã được HEAD-xác minh), KHÔNG phải đường dẫn temp của worker. Bản
    // thân result_key cũng KHÔNG lộ cho client — chỉ trả downloadUrl (proxy /download đọc
    // bytes trực tiếp từ R2), nên restart worker sau DONE vẫn tải được (acceptance #1).
    return c.json({
      jobId,
      ...record,
      progress: publicJobProgress(record.progress),
      result: {
        status: inner.status,
        message: inner.message,
        device_used: inner.device_used,
        pipeline: inner.pipeline,
        separated: inner.separated,
        watermarked: inner.watermarked,
        distinct_voices: inner.distinct_voices,
        notes: inner.notes,
        artifactReady: Boolean(inner.result_key),
        downloadUrl: inner.result_key ? `/api/jobs/${jobId}/download` : undefined,
        artifactSizeBytes: inner.result_key ? inner.result_size : undefined,
        artifactSha256: inner.result_key ? inner.result_sha256 : undefined,
      },
    });
  }

  // M4-S3: an ANALYZE job rests at AWAITING_REVIEW with a SEALED artifact pending human
  // review. Surface a `review` pointer built from the `artifact:` record so the client can
  // fetch + decrypt the ciphertext. We expose ONLY ciphertext-referencing metadata — never
  // the internal R2 object key, and (by construction) NO plaintext transcript/translation
  // (M4 #4) — plus an artifactUrl the client GETs next.
  if (record.status === 'AWAITING_REVIEW') {
    const ptr = (await c.env.KV_CACHE.get(`artifact:${deviceId}:${jobId}`, { type: 'json' })) as any;
    // Cross-PoP skew: the status flipped but the pointer has not propagated to THIS edge yet.
    // Report a NON-terminal status so the client keeps polling instead of latching a review it
    // cannot fulfil (mirrors the DONE→FINALIZING belt-and-suspenders above).
    if (!ptr) {
      const terminalProgress = publicJobProgress(record.progress);
      return c.json({
        jobId,
        ...record,
        status: 'FINALIZING',
        progress: stageProgress('finalizing', terminalProgress?.attempt, terminalProgress?.updatedAt),
      });
    }
    return c.json({
      jobId,
      ...record,
      progress: publicJobProgress(record.progress),
      review: {
        revision: ptr.revision,
        diarization: ptr.diarization,
        segment_count: ptr.segment_count,
        alg: ptr.alg,
        artifact_md5: ptr.artifact_md5,
        artifact_size: ptr.artifact_size,
        artifactReady: Boolean(ptr.artifact_key),
        artifactUrl: ptr.artifact_key ? `/api/jobs/${jobId}/artifact` : undefined,
      },
    });
  }
  return c.json({ jobId, ...record, progress: publicJobProgress(record.progress) });
});

// --- Authenticated result download (client-facing) -------------------------
// Closes the async loop: a client that polled DONE retrieves the dubbed AUDIO
// here. M2-S4: the artifact is a DURABLE R2 object (results/<device>/<job>/<attempt>.wav)
// that the Gateway HEAD-verified at DONE — so we stream its bytes STRAIGHT FROM R2,
// not by proxying an ephemeral worker temp file. This makes the download restart-safe
// (acceptance #1: the worker can be recreated after DONE and the output still downloads)
// and means no worker temp path is ever stored or exposed (acceptance #7). Lookup is
// DEVICE-SCOPED, and a missing object (never uploaded, or reaped by lifecycle) is a
// generic fail-closed 404 that leaks nothing.
app.get('/api/jobs/:jobId/download', async (c) => {
  // M3-S3 (A2): the download streams the owner's dubbed-audio BYTES, so it must be
  // owner-authenticated. A canonical signature over METHOD+PATH (path includes the
  // jobId) proves the caller holds the device's private key; we then scope the
  // lookup to the AUTHENTICATED device. An unsigned/ spoofed request is rejected
  // (401/403) BEFORE we ever touch KV or R2 — no victim bytes can leak.
  const auth = await authenticateSignedRequest(c);
  if (!auth.ok) return c.json({ error: auth.error }, auth.status);
  const deviceId = auth.deviceId;
  const jobId = c.req.param('jobId');

  const record = (await c.env.KV_CACHE.get(`job:${deviceId}:${jobId}`, { type: 'json' })) as any;
  if (!record) return c.json({ error: 'Job not found' }, 404);
  if (record.status !== 'DONE') return c.json({ error: 'Job not ready', status: record.status }, 409);

  const stored = (await c.env.KV_CACHE.get(`result:${deviceId}:${jobId}`, { type: 'json' })) as any;
  const key = stored?.result?.result_key;
  // No durable artifact recorded (legacy/no-R2 dispatch) or the R2 binding is absent →
  // there is nothing to serve. Same generic 404 either way — never disclose internals.
  if (!key || !c.env.R2) return c.json({ error: 'Result artifact unavailable' }, 404);

  const obj = await c.env.R2.get(key);
  // Key recorded but object gone (lifecycle reap / retention expiry). Clean 404, not a 500.
  if (!obj) return c.json({ error: 'Result artifact unavailable' }, 404);

  const storedSize = stored?.result?.result_size;
  const storedSha256 = stored?.result?.result_sha256;
  if (
    !Number.isSafeInteger(storedSize) ||
    storedSize <= 0 ||
    obj.size !== storedSize ||
    typeof storedSha256 !== 'string' ||
    !/^[a-f0-9]{64}$/.test(storedSha256)
  ) {
    return c.json({ error: 'Result artifact metadata unavailable' }, 409);
  }

  // Stream the R2 body straight back to the owning client (memory-bound — the Gateway
  // never buffers the whole artifact). Results are always WAV (the object key is *.wav).
  return new Response(obj.body, {
    status: 200,
    headers: {
      'Content-Type': 'audio/wav',
      'Content-Disposition': `attachment; filename="dubbed-${jobId}.wav"`,
      'Content-Length': String(storedSize),
      'X-Artifact-Sha256': storedSha256,
    },
  });
});

// --- Authenticated ANALYZE artifact download (client-facing) ---------------
// M4-S3: the sealed ECIES ciphertext of an Analyze result. A client that polled
// AWAITING_REVIEW fetches the artifact here and DECRYPTS it locally — the Gateway only
// ever handles ciphertext, so no plaintext transcript/translation ever passes through it
// (M4 #4). Owner-authenticated + device-scoped EXACTLY like /download (a canonical
// signature over METHOD+PATH proves the caller holds the device key), and streamed
// straight from the durable R2 object (memory-bound). Served only while the job is at
// AWAITING_REVIEW (409 otherwise); a missing pointer / reaped object is a generic
// fail-closed 404 that leaks nothing.
app.get('/api/jobs/:jobId/artifact', async (c) => {
  const auth = await authenticateSignedRequest(c);
  if (!auth.ok) return c.json({ error: auth.error }, auth.status);
  const deviceId = auth.deviceId;
  const jobId = c.req.param('jobId');

  const record = (await c.env.KV_CACHE.get(`job:${deviceId}:${jobId}`, { type: 'json' })) as any;
  if (!record) return c.json({ error: 'Job not found' }, 404);
  if (record.status !== 'AWAITING_REVIEW') {
    return c.json({ error: 'Artifact not ready', status: record.status }, 409);
  }

  const ptr = (await c.env.KV_CACHE.get(`artifact:${deviceId}:${jobId}`, { type: 'json' })) as any;
  const key = ptr?.artifact_key;
  // No pointer (not propagated / legacy) or no R2 binding → nothing to serve. Same generic
  // 404 either way — never disclose internals.
  if (!key || !c.env.R2) return c.json({ error: 'Artifact unavailable' }, 404);

  const obj = await c.env.R2.get(key);
  // Key recorded but object gone (lifecycle reap / retention expiry). Clean 404, not a 500.
  if (!obj) return c.json({ error: 'Artifact unavailable' }, 404);

  // Stream the sealed ciphertext straight back to the owning client (memory-bound — the
  // Gateway never buffers the whole artifact). The integrity md5 + alg ride as headers so
  // the client can verify + decrypt without a second poll; both reference ciphertext only.
  return new Response(obj.body, {
    status: 200,
    headers: {
      'Content-Type': 'application/octet-stream',
      'Content-Disposition': `attachment; filename="artifact-${jobId}.bin"`,
      'X-Artifact-Md5': String(ptr.artifact_md5 ?? ''),
      'X-Artifact-Alg': String(ptr.alg ?? ''),
    },
  });
});

// --- Authenticated manifest approval (client-facing) -----------------------
// M4-S4 (ADR 0002 §4): the ONLY gate that mints an ApprovedRevision — the durable
// authority a RENDER submission binds to. The owner POSTs a CANONICAL, client-approved
// manifest (edited segments + resolved voices) here; the Gateway proves two properties
// before persisting it and refuses everything else:
//
//   • INTEGRITY — it recomputes sha256Hex(deterministicStringify(manifest)) and rejects a
//     body whose claimed manifestHash disagrees (400). Render binds to THIS hash, so it
//     must provably be the manifest's own.
//   • OPTIMISTIC CONCURRENCY — baseRevision must equal the analyze job's CURRENT revision;
//     if the analyze re-ran since the edits were derived (M4-S9 conflict), the stale
//     approval is rejected (409) with no silent overwrite of the newer analysis.
//
// Persistence honours M4 item 379 (NO plaintext transcript/translation in KV/log): the
// approved manifest — which legitimately carries sourceText/translatedText — is written as
// an R2 OBJECT (the sanctioned durable store, exactly like the input WAV), and KV holds only
// a NON-plaintext pointer (hash + revision + r2Key). This medium split is flagged for the
// M4-S10 adversarial review.
app.post('/api/jobs/:analyzeJobId/approve', async (c) => {
  const auth = await authenticateSignedRequest(c);
  if (!auth.ok) return c.json({ error: auth.error }, auth.status);
  const { deviceId, rawBody } = auth;
  const analyzeJobId = c.req.param('analyzeJobId');

  // Owner-scoped lookup: the analyze job must exist for THIS device and rest at
  // AWAITING_REVIEW. A missing record (incl. another device's job) → generic 404.
  // Parse the ApproveRequest from the SIGNED body before any durable work. Generic message
  // (no zod field detail) so it never echoes user data.
  let approve: ApproveRequest;
  try {
    approve = ApproveRequestSchema.parse(JSON.parse(rawBody));
  } catch {
    return c.json({ error: 'Invalid approve request: required field missing or malformed' }, 400);
  }

  // Lineage bind: the manifest (and the request envelope) must name the SAME analyze job as
  // the route it rides, and the manifest's analyzeRevision must equal the baseRevision it
  // claims to derive from. A manifest for a different analyze — or internally inconsistent —
  // can never be approved against this job.
  if (approve.analyzeJobId !== analyzeJobId || approve.manifest.analyzeJobId !== analyzeJobId) {
    return c.json({ error: 'Manifest lineage does not match this analyze job' }, 400);
  }
  if (approve.manifest.analyzeRevision !== approve.baseRevision) {
    return c.json({ error: 'Manifest revision does not match baseRevision' }, 400);
  }

  // INTEGRITY: recompute the canonical hash and refuse a mismatch (400). Identical
  // deterministicStringify both sides ⇒ a truthful client always matches.
  const computedHash = await sha256Hex(deterministicStringify(approve.manifest));
  if (computedHash !== approve.manifestHash) {
    return c.json({ error: 'manifestHash does not match manifest' }, 400);
  }

  if (!c.env.JOB_COORDINATOR) {
    return c.json({ error: 'Approval requires durable coordination (unavailable)' }, 503);
  }
  if (!c.env.R2) return c.json({ error: 'Approval storage unavailable' }, 503);
  const stub = coordinatorStub(c.env.JOB_COORDINATOR, deviceId, analyzeJobId);
  const record = await coordinatorGet(stub);
  if (!record) return c.json({ error: 'Job not found' }, 404);
  if (record.phase !== 'ANALYZE' || record.status !== 'AWAITING_REVIEW') {
    return c.json({ error: 'Job is not awaiting review', status: record.status }, 409);
  }
  const currentRevision =
    typeof record.analyzeRevision === 'number' ? (record.analyzeRevision as number) : undefined;
  if (currentRevision !== approve.baseRevision) {
    return c.json(
      { error: 'Stale approval: analyze revision has moved', currentRevision, baseRevision: approve.baseRevision },
      409,
    );
  }

  const r2Key = `approved/${deviceId}/${analyzeJobId}/${computedHash}.json`;
  await c.env.R2.put(r2Key, deterministicStringify(approve.manifest));
  const decision = await coordinatorApprove(stub, {
    baseRevision: approve.baseRevision,
    approvedManifestHash: computedHash,
    r2Key,
    approvedAt: Date.now(),
  });
  if (decision.outcome === 'not_found') return c.json({ error: 'Job not found' }, 404);
  if (decision.outcome === 'not_awaiting_review') {
    return c.json({ error: 'Job is not awaiting review', status: decision.record?.status }, 409);
  }
  if (decision.outcome === 'stale') {
    return c.json(
      {
        error: 'Stale approval: analyze revision has moved',
        currentRevision: decision.currentRevision,
        baseRevision: approve.baseRevision,
      },
      409,
    );
  }
  if (decision.outcome === 'conflict') {
    return c.json(
      {
        error: 'Approval already committed for this analyze revision',
        currentRevision: decision.currentRevision,
        baseRevision: approve.baseRevision,
      },
      409,
    );
  }

  const approvedRevision = decision.approval!;
  return c.json({
    approved: true,
    ...(decision.outcome === 'idempotent' ? { idempotent: true } : {}),
    analyzeJobId,
    revision: approvedRevision.analyzeRevision,
    approvedRevision: approvedRevision.approvedRevision,
    approvedManifestHash: approvedRevision.approvedManifestHash,
  });
});

// --- Authenticated job cancellation (client-facing) ------------------------
// M3-S3 (A2 / plan line 334): let the OWNER cancel a not-yet-running job. Like poll
// and download, it demands a canonical signature (method+path+bodyHash+ts+nonce) and
// scopes everything to the AUTHENTICATED device — no unsigned/spoofed request can
// cancel a victim's job. POST (not DELETE) keeps the CORS allow-list unchanged while
// still binding a DISTINCT path into the signature.
//
// Cancellation is only meaningful BEFORE a GPU render is dispatched: we flip
// QUEUED|RETRYING → CANCELLED (a terminal state). Because CANCELLED is terminal and
// acquireDispatch returns 'terminal' for it, the queue consumer that later drains the
// message ACKS it WITHOUT firing a render — the anti-spend guarantee. A job already
// PROCESSING cannot be cancelled here (409); tearing down an in-flight render is the
// job of the kill-switch/teardown path (M3-S6). Terminal jobs report cancelled:false.
app.post('/api/jobs/:jobId/cancel', async (c) => {
  const auth = await authenticateSignedRequest(c);
  if (!auth.ok) return c.json({ error: auth.error }, auth.status);
  const deviceId = auth.deviceId;
  const jobId = c.req.param('jobId');
  const jobKey = `job:${deviceId}:${jobId}`;

  // Durable path: the JobCoordinator is the atomic authority. Its stub is addressed by
  // idFromName("<authDeviceId>:<jobId>"), so identity is already proven — cancel() flips
  // the record and projects CANCELLED to KV in one place. We still reclaim the uploaded
  // input here (the DO does not touch R2), mirroring setJob's terminal-input cleanup.
  if (c.env.JOB_COORDINATOR) {
    const stub = coordinatorStub(c.env.JOB_COORDINATOR, deviceId, jobId);
    const out = await coordinatorCancel(stub);
    if (!out.record || out.reason === 'not_found') return c.json({ error: 'Job not found' }, 404);
    if (out.applied) {
      const inputOwnerJobId =
        out.record.phase === 'RENDER' && typeof out.record.analyzeJobId === 'string'
          ? out.record.analyzeJobId
          : undefined;
      await deleteJobInputIfTerminal(c.env, jobKey, 'CANCELLED', inputOwnerJobId);
      return c.json({ jobId, status: 'CANCELLED', cancelled: true });
    }
    if (out.reason === 'in_flight') {
      return c.json(
        { jobId, status: out.record.status, error: 'Job is already processing and cannot be cancelled' },
        409,
      );
    }
    // reason === 'terminal': already DONE/FAILED/CANCELLED/… — report it, cancelled:false.
    return c.json({ jobId, status: out.record.status, cancelled: false, message: 'Job already finished' });
  }

  // Legacy KV-only path (no DO binding — pre-M1 deploys / KV-only tests). Read-modify-write
  // the projection; setJob flips KV to CANCELLED and (being terminal) reaps the input R2 object.
  return withLegacyJobLock(jobKey, async () => {
    const record = (await c.env.KV_CACHE.get(jobKey, { type: 'json' })) as any;
    if (!record) return c.json({ error: 'Job not found' }, 404);
    if (isTerminal(record.status)) {
      return c.json({ jobId, status: record.status, cancelled: false, message: 'Job already finished' });
    }
    if (record.status !== 'QUEUED' && record.status !== 'RETRYING') {
      return c.json(
        { jobId, status: record.status, error: 'Job is already processing and cannot be cancelled' },
        409,
      );
    }
    const inputOwnerJobId =
      record.phase === 'RENDER' && typeof record.analyzeJobId === 'string'
        ? record.analyzeJobId
        : undefined;
    await setJobUnlocked(
      c.env,
      jobKey,
      { ...record, status: 'CANCELLED', cancelledAt: Date.now() },
      inputOwnerJobId,
    );
    return c.json({ jobId, status: 'CANCELLED', cancelled: true });
  });
});

// --- Station 2 + Station 3 dispatch -----------------------------------------
// Exported for direct testing of the anti-fraud timing enforcement.
export async function dispatchToWorker(
  env: Bindings,
  payload: JobRequest,
  jobKey: string,
  resolvedTarget?: ResolvedWorkerTarget,
): Promise<void> {
  const maxMs = workerRequestTimeoutMs(env);
  const workerConfig = resolvedTarget ?? await resolveWorkerTarget(env);
  if (workerConfig.status === 'unavailable') {
    console.error('[dispatch] worker target registry unavailable; deferring');
    throw new Error('worker target registry unavailable');
  }
  if (workerConfig.status !== 'valid') {
    await setJob(env, jobKey, {
      status: 'FAILED',
      reason: FAILURE_REASONS.WORKER_CONFIGURATION_INVALID,
    });
    console.error('[dispatch] worker configuration missing or invalid');
    return;
  }
  if (!workerTargetSupportsRequest(workerConfig, maxMs)) {
    await setJob(env, jobKey, {
      status: 'FAILED',
      reason: FAILURE_REASONS.WORKER_TRANSPORT_UNSUPPORTED,
    });
    console.error('[dispatch] worker transport cannot hold the configured request timeout');
    return;
  }
  const workerUrl = workerConfig.url;

  // M2-S2: the worker's input GET is DERIVED from the authenticated (deviceId, jobId)
  // carried in the jobKey — not from any client-supplied URL/key — and minted fresh
  // per attempt below (never stored at rest). jobKey is always well-formed here (it
  // comes from job creation), so a null parse degrades to an empty URL (fail-safe).
  const ids = parseJobKey(jobKey);

  // Station 3: never dispatch to a worker previously QUARANTINED for anomaly
  // (fabricated/too-fast result or hang). A known-bad instance must not receive
  // new jobs; re-queueing to a FRESH instance is the infra worker-pool's job
  // (residual_hardware G3). Faking a redispatch to the same bad URL is dishonest.
  if (await env.KV_CACHE.get(`quarantine:${workerUrl}`)) {
    await setJob(env, jobKey, { status: 'FAILED', reason: FAILURE_REASONS.WORKER_QUARANTINED });
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
  const gate = await signGatewayJwt(env, payload.jobId, 'dispatch', 1, '0'.repeat(64));
  if (!gate) {
    const reason = env.GATEWAY_JWT_PRIVATE_KEY
      ? FAILURE_REASONS.GATEWAY_KEY_INVALID
      : FAILURE_REASONS.GATEWAY_KEY_MISSING;
    await setJob(env, jobKey, { status: 'FAILED', reason });
    console.error(`[station2] cannot dispatch job: ${reason}`);
    return;
  }

  // M2-S5b (#5 + #4): enforce the 30 MB INPUT ceiling at the EDGE, before any GPU
  // spend. The size is read from the NATIVE R2 BINDING (authoritative object
  // metadata) — never by probing a signed URL — so an expired/re-signed GET can
  // never be mistaken for object state (acceptance #4). Oversize is TERMINAL
  // (reason `input_too_large`); setJob's terminal-delete then reaps the oversized
  // object. Only a CONFIRMED-oversize head fails: a null head (input still
  // uploading / already reaped) is left to the worker's own GET + streaming
  // hard-gate, so a create→dispatch race never false-fails an honest job. Legacy
  // KV-only envs (no R2 binding) skip this — the worker's stream cap still holds.
  if (ids && env.R2) {
    const maxInputBytes = Number(env.MAX_INPUT_BYTES) > 0 ? Number(env.MAX_INPUT_BYTES) : MAX_INPUT_BYTES;
    const inHead = await env.R2.head(inputAudioKey(ids.deviceId, ids.jobId));
    const inSize = inHead ? (inHead as { size?: number }).size : undefined;
    if (typeof inSize === 'number' && inSize > maxInputBytes) {
      await setJob(env, jobKey, { status: 'FAILED', reason: FAILURE_REASONS.INPUT_TOO_LARGE });
      console.error('[m2-s5b] input exceeds cap; refusing dispatch');
      return;
    }
  }

  // Station 3: bound the round-trip. Too fast => faked result; too slow => hung.
  // Bounds default to the tuned constants but can be overridden per deployment.
  const floor = Number(env.MIN_PLAUSIBLE_MS) || MIN_PLAUSIBLE_MS_FLOOR;
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
      // Mint the SHORT-LIVED input GET for THIS attempt (mirrors the per-attempt JWT
      // above): a retry after backoff always ships a URL that is still valid, and the
      // ephemeral credential never outlives the dispatch. Derived from the job's own
      // (device, id), so it can only ever address this job's input object.
      const audioUrl = ids ? await mintDispatchAudioUrl(env, ids.deviceId, ids.jobId) : '';
      // M2-S3: also mint a SHORT-LIVED PUT for THIS attempt's result object so the
      // worker can upload its rendered output to R2 (results/<dev>/<job>/<attempt>.wav)
      // WITHOUT ever holding R2 credentials. The window must outlive the render (the
      // upload happens after rendering), so size it to the dispatch timeout + margin.
      const resultPutExpiresS = Math.ceil(maxMs / 1000) + 300;
      const resultUploadUrl = ids
        ? await mintDispatchResultUploadUrl(env, ids.deviceId, ids.jobId, attempt, resultPutExpiresS)
        : '';
      const resultObjKey = ids ? resultObjectKey(ids.deviceId, ids.jobId, attempt) : '';
      const workerBody = JSON.stringify({
        job_id: payload.jobId,
        attempt,
        audio_url: audioUrl,
        result_upload_url: resultUploadUrl,
        result_key: resultObjKey,
        audio_md5: payload.videoAudioMd5,
        target_language: payload.config.targetLanguage,
        translation_style: payload.config.translationStyle,
        voice_map: payload.speakerMapping || {},
        source_language: payload.config.sourceLanguage,
        prompt_profile: payload.config.promptProfile ? {
          preset_id: payload.config.promptProfile.presetId,
          preset_revision: payload.config.promptProfile.presetRevision,
          custom_instructions: payload.config.promptProfile.customInstructions,
        } : undefined,
        segments: payload.segments || [],
      });
      const jwt = await signGatewayJwt(
        env,
        payload.jobId,
        'dispatch',
        attempt,
        await sha256Hex(workerBody),
      );
      if (!jwt) {
        await setJob(env, jobKey, { status: 'FAILED', reason: FAILURE_REASONS.GATEWAY_KEY_INVALID, attempts: attempt });
        return;
      }
      const processingClaimed = await setJob(env, jobKey, {
        status: 'PROCESSING',
        attempt,
        progress: stageProgress('rendering', attempt),
      });
      // A stale legacy dispatch can outlive an owner cancellation. If the terminal
      // projection won the per-job lock, do not contact the worker or spend GPU.
      if (!processingClaimed) return;
      const startTime = Date.now();
      const res = await fetch(`${workerUrl}/api/worker/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${jwt}` },
        body: workerBody,
        signal: controller.signal,
      });

      const elapsed = Date.now() - startTime;

      if (!res.ok) {
        // 5xx: transient -> retry if attempts remain. 4xx: hard error -> stop.
        // M3-S10: a worker non-2xx is stamped with the CANONICAL reason 'worker_error',
        // not a raw numeric `code: res.status` (every other branch already uses a
        // sanitized `reason` string — this was the lone inconsistency). The raw status
        // is not user data, so it goes to the server log for diagnosis, never the record.
        if (res.status >= 500 && attempt < MAX_DISPATCH_ATTEMPTS) {
          await setJob(env, jobKey, {
            status: 'RETRYING',
            attempt,
            reason: FAILURE_REASONS.WORKER_ERROR,
            progress: stageProgress('retrying', attempt),
          });
          continue;
        }
        console.error('[dispatch] worker returned non-ok status; failing', res.status);
        await setJob(env, jobKey, { status: 'FAILED', reason: FAILURE_REASONS.WORKER_ERROR, attempts: attempt });
        return;
      }

      // Anti-Fraud: an implausibly fast success is a fabricated result. Terminal.
      if (elapsed < minPlausibleMs) {
        await terminateWorker(env, workerUrl, payload.jobId, 'anomaly_too_fast');
        await setJob(env, jobKey, { status: 'REJECTED_FRAUD', elapsed, attempts: attempt });
        console.error('[station3] rejected worker result: impossibly fast');
        return;
      }

      const workerResponse = (await res.json()) as unknown;

      // M3-S8: VALIDATE THE FULL RESPONSE BEFORE DONE. A 200 status only proves the
      // worker answered — not that it answered for THIS job, succeeded, or spoke a
      // contract version we understand. Parse the body against the shared schema
      // (schema_version literal, non-empty job_id, positive attempt, result.status
      // === 'success'); any deviation is a contract violation, not a render we can
      // trust. Then cross-check the job_id/attempt ECHO against the dispatch we
      // actually sent — the schema proves shape, these prove IDENTITY (a stale or
      // misrouted body carrying another job's/attempt's success can't latch DONE
      // here). Both are TERMINAL: re-dispatching only burns GPU on a worker that
      // just broke contract, and the too-fast fraud gate already ran. Zero-Logging:
      // the console line names only the failure class, never the body.
      const parsed = WorkerResponseSchema.safeParse(workerResponse);
      if (!parsed.success) {
        await setJob(env, jobKey, { status: 'FAILED', reason: FAILURE_REASONS.WORKER_RESPONSE_INVALID, attempts: attempt });
        console.error('[m3] worker response failed schema validation; refusing DONE');
        return;
      }
      const resp = parsed.data;
      if (resp.job_id !== payload.jobId || resp.attempt !== attempt) {
        await setJob(env, jobKey, { status: 'FAILED', reason: FAILURE_REASONS.WORKER_RESPONSE_MISMATCH, attempts: attempt });
        console.error('[m3] worker response job/attempt mismatch; refusing DONE');
        return;
      }
      const raw = resp.result as Record<string, any>;
      await setJob(env, jobKey, {
        status: 'PROCESSING',
        attempt,
        progress: stageProgress('finalizing', attempt),
      });

      // M2-S3: VERIFY-BEFORE-DONE. When the worker participated in the R2-result
      // protocol (it reported a result_md5 — i.e. it uploaded to the PUT URL we minted),
      // the Gateway MUST confirm the artifact actually landed in R2 with the matching
      // checksum BEFORE flipping DONE. A client that observes DONE must be able to read
      // a real, intact result; a worker that merely CLAIMS success (no upload, or a
      // corrupt/substituted object) must never mint a DONE. The key is DERIVED from the
      // authenticated (device, job, attempt) — the worker's own result_key is ignored
      // here — so verification addresses only this job's own output (F-R2-01 discipline).
      // Missing/mismatch is TERMINAL: re-running only burns GPU on an untrustworthy
      // worker, and the too-fast fraud gate already ran. Legacy responses (no result_md5,
      // e.g. KV-only envs) skip verify and keep the prior temp-path behavior.
      const reportedMd5 = typeof raw.result_md5 === 'string' ? raw.result_md5.toLowerCase() : '';
      const reportedSha256 = typeof raw.result_sha256 === 'string' ? raw.result_sha256.toLowerCase() : '';
      const reportedSize = raw.result_size;
      let verifiedResultKey = '';
      let verifiedResultSize: number | undefined;
      if (reportedMd5 && ids && env.R2) {
        if (!/^[a-f0-9]{64}$/.test(reportedSha256) || !Number.isSafeInteger(reportedSize) || reportedSize <= 0) {
          await setJob(env, jobKey, { status: 'FAILED', reason: FAILURE_REASONS.RESULT_UNVERIFIED, attempts: attempt });
          console.error('[m6] result artifact metadata invalid; refusing DONE');
          return;
        }
        const key = resultObjectKey(ids.deviceId, ids.jobId, attempt);
        const head = await env.R2.head(key);
        if (!head || r2ObjectMd5(head) !== reportedMd5 || (head as { size?: number }).size !== reportedSize) {
          await setJob(env, jobKey, { status: 'FAILED', reason: FAILURE_REASONS.RESULT_UNVERIFIED, attempts: attempt });
          console.error('[m2] result artifact failed verification; refusing DONE');
          return;
        }
        verifiedResultKey = key;
        verifiedResultSize = (head as { size?: number }).size;
      }

      // Zero-Logging AT REST: allowlist các trường TRƯỚC khi lưu KV 24h. Chỉ giữ metadata
      // an toàn + tham chiếu artifact R2 đã xác minh. Loại mọi kịch bản/bản dịch plaintext
      // để không lưu trữ nội dung nhạy cảm; dùng allowlist (không denylist) nên trường nhạy
      // cảm mới cũng không lọt. M2-S4: TUYỆT ĐỐI KHÔNG lưu dubbed_audio (đường dẫn temp của
      // worker) — nó là ephemeral (mất khi worker restart) và là đường dẫn nội bộ; nguồn
      // chân lý để tải là object R2 tại result_key (acceptance #7).
      const storedResult = {
        status: raw.status,
        message: raw.message,
        device_used: raw.device_used,
        pipeline: raw.pipeline,
        separated: raw.separated,
        watermarked: raw.watermarked,
        distinct_voices: raw.distinct_voices,
        notes: raw.notes,
        // M2-S3/S4: the VERIFIED durable artifact — key/checksum/size the Gateway confirmed
        // exist in R2. The /download endpoint streams bytes from this key (restart-safe).
        // Empty for legacy/no-R2 dispatches (then there is no downloadable artifact).
        result_key: verifiedResultKey || undefined,
        result_md5: verifiedResultKey ? reportedMd5 : undefined,
        result_sha256: verifiedResultKey ? reportedSha256 : undefined,
        result_size: verifiedResultSize,
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
        JSON.stringify({ job_id: resp.job_id, result: storedResult }),
        { expirationTtl: JOB_TTL_S },
      );
      await setJob(env, jobKey, {
        status: 'DONE',
        elapsed,
        attempts: attempt,
        progress: completeProgress(attempt),
      });
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
        await setJob(env, jobKey, {
          status: 'RETRYING',
          attempt,
          reason: FAILURE_REASONS.NETWORK,
          progress: stageProgress('retrying', attempt),
        });
        continue;
      }
      // M3-S10: the network-exhausted terminal carries the same canonical 'network'
      // reason as its intermediate RETRYING branch (every terminal has a sanitized reason).
      await setJob(env, jobKey, { status: 'ERROR', reason: FAILURE_REASONS.NETWORK, attempts: attempt });
      return;
    } finally {
      clearTimeout(timeout);
    }
  }
}

type AsyncAnalyzeRoundTrip = {
  response: Response;
  elapsed: number;
};

function asyncAbortError(): DOMException {
  return new DOMException('async analyze deadline exceeded', 'AbortError');
}

async function sleepWithAbort(milliseconds: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) throw asyncAbortError();
  await new Promise<void>((resolve, reject) => {
    let settled = false;
    const cleanup = () => {
      clearTimeout(timer);
      signal.removeEventListener('abort', onAbort);
    };
    const finish = () => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve();
    };
    const timer = setTimeout(finish, milliseconds);
    const onAbort = () => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(asyncAbortError());
    };
    signal.addEventListener('abort', onAbort, { once: true });
  });
}

/**
 * RunPod's public HTTP proxy bounds one response, not the lifetime of a worker
 * task.  The async worker contract therefore uses one short POST and repeated
 * short GETs.  Every individual request is bounded below the proxy ceiling and
 * the caller's AbortSignal still owns the total deadline.
 */
async function dispatchAnalyzeAsyncRoundTrip(
  env: Bindings,
  workerUrl: string,
  workerBody: string,
  submitJwt: string,
  jobId: string,
  attempt: number,
  outerSignal: AbortSignal,
): Promise<AsyncAnalyzeRoundTrip> {
  const startedAt = Date.now();
  const totalMs = asyncAnalyzeTimeoutMs(env);
  const pollMs = asyncAnalyzePollMs(env);
  const requestMs = Math.min(80_000, Math.max(5_000, workerRequestTimeoutMs(env) - 1_000));
  const deadline = startedAt + totalMs;

  const boundedFetch = async (
    url: string,
    init: RequestInit,
  ): Promise<Response> => {
    if (outerSignal.aborted) throw asyncAbortError();
    const local = new AbortController();
    const relayAbort = () => local.abort();
    outerSignal.addEventListener('abort', relayAbort, { once: true });
    const timer = setTimeout(() => local.abort(), requestMs);
    try {
      return await fetch(url, { ...init, signal: local.signal });
    } catch {
      if (outerSignal.aborted) throw asyncAbortError();
      return new Response(JSON.stringify({ error: 'worker_unavailable' }), {
        status: 503,
        headers: { 'Content-Type': 'application/json' },
      });
    } finally {
      clearTimeout(timer);
      outerSignal.removeEventListener('abort', relayAbort);
    }
  };

  const submit = await boundedFetch(`${workerUrl}/api/worker/analyze/submit`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      Authorization: `Bearer ${submitJwt}`,
    },
    body: workerBody,
  });
  if (!submit.ok) {
    // Preserve a normal Response so the caller applies the existing sanitized
    // 4xx/5xx policy.  In particular, a missing async endpoint is never silently
    // treated as a successful submission.
    return { response: submit, elapsed: Date.now() - startedAt };
  }
  let submitted: unknown;
  try {
    submitted = await submit.json();
  } catch {
    return {
      response: new Response(JSON.stringify({ error: 'worker_response_invalid' }), { status: 502 }),
      elapsed: Date.now() - startedAt,
    };
  }
  if (
    typeof submitted !== 'object'
    || submitted === null
    || (submitted as Record<string, unknown>).job_id !== jobId
    || (submitted as Record<string, unknown>).attempt !== attempt
  ) {
    return {
      response: new Response(JSON.stringify({ error: 'worker_response_mismatch' }), { status: 502 }),
      elapsed: Date.now() - startedAt,
    };
  }

  while (Date.now() < deadline) {
    // Status remains bound to the exact job/attempt. The empty GET body is
    // represented by SHA-256(empty), so a readiness probe token cannot be reused.
    const statusJwt = await signGatewayJwt(env, jobId, 'analyze', attempt, await sha256Hex(''));
    if (!statusJwt) {
      return {
        response: new Response(JSON.stringify({ error: FAILURE_REASONS.GATEWAY_KEY_INVALID }), { status: 503 }),
        elapsed: Date.now() - startedAt,
      };
    }
    const poll = await boundedFetch(
      `${workerUrl}/api/worker/analyze/status/${encodeURIComponent(jobId)}/${attempt}`,
      {
        method: 'GET',
        headers: { Accept: 'application/json', Authorization: `Bearer ${statusJwt}` },
      },
    );
    if (poll.ok) {
      let body: unknown;
      try {
        body = await poll.json();
      } catch {
        return {
          response: new Response(JSON.stringify({ error: 'worker_response_invalid' }), { status: 502 }),
          elapsed: Date.now() - startedAt,
        };
      }
      if (typeof body !== 'object' || body === null) {
        return {
          response: new Response(JSON.stringify({ error: 'worker_response_invalid' }), { status: 502 }),
          elapsed: Date.now() - startedAt,
        };
      }
      const status = (body as Record<string, unknown>).status;
      if (status === 'completed') {
        const response = (body as Record<string, unknown>).response;
        return {
          response: new Response(JSON.stringify(response), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          }),
          elapsed: Date.now() - startedAt,
        };
      }
      if (status === 'failed') {
        return {
          response: new Response(JSON.stringify({ error: 'worker_error' }), { status: 500 }),
          elapsed: Date.now() - startedAt,
        };
      }
      if (status !== 'queued' && status !== 'running') {
        return {
          response: new Response(JSON.stringify({ error: 'worker_response_invalid' }), { status: 502 }),
          elapsed: Date.now() - startedAt,
        };
      }
    } else if (poll.status === 401 || poll.status === 403 || poll.status === 404) {
      return { response: poll, elapsed: Date.now() - startedAt };
    }
    await sleepWithAbort(Math.min(pollMs, Math.max(0, deadline - Date.now())), outerSignal);
  }

  throw asyncAbortError();
}

// --- M4-S3: ANALYZE dispatch + encrypted-artifact channel -------------------
// The ANALYZE twin of dispatchToWorker. It drives the worker's /api/worker/analyze
// endpoint (ASR + diarization + translate), whose result the worker SEALS to the
// client's per-device ECDH public key (ECIES, ADR 0002) and uploads as CIPHERTEXT to
// R2 at a DERIVED artifact key. The Gateway — exactly as it HEAD-verifies a render
// result before DONE — HEAD-verifies the ciphertext object exists with the reported
// md5 BEFORE resting the job at AWAITING_REVIEW (never latch a bare 200). It only ever
// sees ciphertext: no transcript/translation ever lands in Gateway KV/log (M4 #4).
//
// Two deliberate divergences from the render dispatch:
//   • the SUCCESS sink is AWAITING_REVIEW (compute done, human review pending), written
//     by a DIRECT KV put — NOT setJob — so it does NOT reap the input: the RENDER phase
//     (a separate job in the same lineage) reads the SAME input object; reaping here
//     would strand render. FAILED terminals still go through setJob (reaping a dead
//     analyze's input is correct — no render will ever follow it).
//   • verification is MANDATORY, not conditional: the sealed artifact IS analyze's whole
//     product, so a dispatch that cannot verify it in R2 fails closed rather than resting
//     an unverified AWAITING_REVIEW (render's verify is optional only for legacy no-md5).
// The quarantine / fail-closed-gate / input-cap / per-attempt retry / too-fast-fraud /
// timeout scaffolding MIRRORS dispatchToWorker's already-proven structure (same GPU-worker
// threat model); only the endpoint, the JWT act ('analyze'), the request body, the
// response schema, and the artifact-verify + AWAITING_REVIEW sink are analyze-specific.
// Exported for direct testing of the encrypted-artifact channel (see m4_analyze_dispatch).
export async function dispatchAnalyzeToWorker(
  env: Bindings,
  payload: AnalyzeRequest,
  jobKey: string,
  revisionBase = 0,
  resolvedTarget?: ResolvedWorkerTarget,
): Promise<void> {
  const ids = parseJobKey(jobKey);
  const finishAnalyzeFailure = async (failure: Record<string, unknown>): Promise<void> => {
    if (revisionBase > 0 && env.JOB_COORDINATOR && ids) {
      const stub = coordinatorStub(env.JOB_COORDINATOR, ids.deviceId, ids.jobId);
      const rollbackInput = { baseRevision: revisionBase, failure };
      let rollback;
      try {
        rollback = await coordinatorRollbackAnalyzeRerun(stub, rollbackInput);
      } catch {
        // The DO storage commit may have succeeded while a following KV projection
        // failed. The idempotent retry heals every old-review projection.
        rollback = await coordinatorRollbackAnalyzeRerun(stub, rollbackInput);
      }
      if (rollback.outcome === 'rolled_back' || rollback.outcome === 'idempotent') return;
      throw new Error(`analyze rerun rollback refused: ${rollback.outcome}`);
    }
    await setJob(env, jobKey, failure);
  };

  const maxMs = workerRequestTimeoutMs(env);
  const asyncMode = asyncAnalyzeEnabled(env);
  // In async mode maxMs remains the per-request transport budget, while the
  // background task gets its own bounded end-to-end deadline.
  const roundTripMaxMs = asyncMode ? asyncAnalyzeTimeoutMs(env) : maxMs;
  const workerConfig = resolvedTarget ?? await resolveWorkerTarget(env);
  if (workerConfig.status === 'unavailable') {
    console.error('[dispatch] analyze worker target registry unavailable; deferring');
    throw new Error('worker target registry unavailable');
  }
  if (workerConfig.status !== 'valid') {
    await finishAnalyzeFailure({
      status: 'FAILED',
      reason: FAILURE_REASONS.WORKER_CONFIGURATION_INVALID,
    });
    console.error('[dispatch] analyze worker configuration missing or invalid');
    return;
  }
  if (!workerTargetSupportsRequest(workerConfig, maxMs)) {
    await finishAnalyzeFailure({
      status: 'FAILED',
      reason: FAILURE_REASONS.WORKER_TRANSPORT_UNSUPPORTED,
    });
    console.error('[dispatch] analyze worker transport cannot hold the configured request timeout');
    return;
  }
  const workerUrl = workerConfig.url;

  // Station 3: never dispatch to a worker previously QUARANTINED for anomaly.
  if (await env.KV_CACHE.get(`quarantine:${workerUrl}`)) {
    await finishAnalyzeFailure({ status: 'FAILED', reason: FAILURE_REASONS.WORKER_QUARANTINED });
    console.error('[station3] refused analyze dispatch to quarantined worker');
    return;
  }

  // Station 2 fail-closed GATE: refuse to touch the worker without a valid signing key.
  const gate = await signGatewayJwt(env, payload.jobId, 'analyze', 1, '0'.repeat(64));
  if (!gate) {
    const reason = env.GATEWAY_JWT_PRIVATE_KEY
      ? FAILURE_REASONS.GATEWAY_KEY_INVALID
      : FAILURE_REASONS.GATEWAY_KEY_MISSING;
    await finishAnalyzeFailure({ status: 'FAILED', reason });
    console.error(`[station2] cannot dispatch analyze: ${reason}`);
    return;
  }

  // M2-S5b (#5 + #4): enforce the input ceiling at the EDGE, read from the NATIVE R2
  // binding — never by probing a signed URL. Only a CONFIRMED-oversize head fails; a
  // null head (still uploading / already reaped) defers to the worker's own stream cap.
  if (ids && env.R2) {
    const maxInputBytes = Number(env.MAX_INPUT_BYTES) > 0 ? Number(env.MAX_INPUT_BYTES) : MAX_INPUT_BYTES;
    const inHead = await env.R2.head(inputAudioKey(ids.deviceId, ids.jobId));
    const inSize = inHead ? (inHead as { size?: number }).size : undefined;
    if (typeof inSize === 'number' && inSize > maxInputBytes) {
      await finishAnalyzeFailure({ status: 'FAILED', reason: FAILURE_REASONS.INPUT_TOO_LARGE });
      console.error('[m2-s5b] analyze input exceeds cap; refusing dispatch');
      return;
    }
  }

  // Station 3 timing bounds. Analyze carries no segments (it PRODUCES them), so the
  // fraud floor is the flat MIN_PLAUSIBLE_MS — no per-segment scaling. The min() keeps
  // the plausible window [floor, maxMs] non-inverted even under env misconfiguration.
  const floor = Number(env.MIN_PLAUSIBLE_MS) || MIN_PLAUSIBLE_MS_FLOOR;
  const minPlausibleMs = Math.min(floor, Math.floor(roundTripMaxMs / 2));

  for (let attempt = 1; attempt <= MAX_DISPATCH_ATTEMPTS; attempt++) {
    // Retry attempts and analyze revisions share one monotonic sequence. Initial analyze
    // starts at zero; an explicit rerun starts at its CAS base revision.
    const workerAttempt = revisionBase + attempt;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), roundTripMaxMs);

    try {
      // Re-mint the JWT per attempt (short 2m exp): a retry after backoff must never ship
      // an already-expired token. Scoped act='analyze' + this attempt (M3-S7).
      const audioUrl = ids ? await mintDispatchAudioUrl(env, ids.deviceId, ids.jobId) : '';
      // Mint the SHORT-LIVED PUT the worker uploads the CIPHERTEXT to (it never holds R2
      // creds). The window must outlive analyze, so size it to the timeout + margin.
      const artifactPutExpiresS = Math.ceil(roundTripMaxMs / 1000) + 300;
      const artifactUploadUrl = ids
        ? await mintDispatchArtifactUploadUrl(env, ids.deviceId, ids.jobId, workerAttempt, artifactPutExpiresS)
        : '';
      const artifactKey = ids ? artifactObjectKey(ids.deviceId, ids.jobId, workerAttempt) : '';
      const workerBody = JSON.stringify({
        job_id: payload.jobId,
        attempt: workerAttempt,
        audio_url: audioUrl,
        audio_md5: payload.videoAudioMd5,
        artifact_upload_url: artifactUploadUrl,
        artifact_key: artifactKey,
        encryption_public_key: payload.encryptionPublicKey,
        target_language: payload.config.targetLanguage,
        translation_style: payload.config.translationStyle,
        source_language: payload.config.sourceLanguage,
        prompt_profile: payload.config.promptProfile ? {
          preset_id: payload.config.promptProfile.presetId,
          preset_revision: payload.config.promptProfile.presetRevision,
          custom_instructions: payload.config.promptProfile.customInstructions,
        } : undefined,
      });
      const jwt = await signGatewayJwt(
        env,
        payload.jobId,
        'analyze',
        workerAttempt,
        await sha256Hex(workerBody),
      );
      if (!jwt) {
        await finishAnalyzeFailure({ status: 'FAILED', reason: FAILURE_REASONS.GATEWAY_KEY_INVALID, attempts: attempt });
        return;
      }
      const processingClaimed = await setJob(env, jobKey, {
        status: 'PROCESSING',
        phase: 'ANALYZE',
        attempt,
        progress: stageProgress('analyzing', attempt),
      });
      if (!processingClaimed) return;
      const startTime = Date.now();
      const roundTrip = asyncMode
        ? await dispatchAnalyzeAsyncRoundTrip(
          env,
          workerUrl,
          workerBody,
          jwt,
          payload.jobId,
          workerAttempt,
          controller.signal,
        )
        : {
          response: await fetch(`${workerUrl}/api/worker/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${jwt}` },
            body: workerBody,
            signal: controller.signal,
          }),
          elapsed: Date.now() - startTime,
        };
      const res = roundTrip.response;
      const elapsed = asyncMode ? roundTrip.elapsed : Date.now() - startTime;

      if (!res.ok) {
        if (res.status >= 500 && attempt < MAX_DISPATCH_ATTEMPTS) {
          await setJob(env, jobKey, {
            status: 'RETRYING',
            phase: 'ANALYZE',
            attempt,
            reason: FAILURE_REASONS.WORKER_ERROR,
            progress: stageProgress('retrying', attempt),
          });
          continue;
        }
        console.error('[analyze] worker returned non-ok status; failing', res.status);
        await finishAnalyzeFailure({ status: 'FAILED', reason: FAILURE_REASONS.WORKER_ERROR, attempts: attempt });
        return;
      }

      // Anti-Fraud: a real ASR + diarization + translate cannot finish instantly. Terminal.
      if (elapsed < minPlausibleMs) {
        await terminateWorker(env, workerUrl, payload.jobId, 'anomaly_too_fast');
        await finishAnalyzeFailure({ status: 'REJECTED_FRAUD', elapsed, attempts: attempt });
        console.error('[station3] rejected analyze result: impossibly fast');
        return;
      }

      // M3-S8: VALIDATE THE FULL RESPONSE. A 200 only proves the worker answered. Parse
      // against the shared analyze schema, then cross-check the job_id/attempt ECHO. Both
      // TERMINAL (re-dispatch only burns GPU on a worker that just broke contract).
      const parsed = WorkerAnalyzeResponseSchema.safeParse(await res.json());
      if (!parsed.success) {
        await finishAnalyzeFailure({ status: 'FAILED', reason: FAILURE_REASONS.WORKER_RESPONSE_INVALID, attempts: attempt });
        console.error('[m4] analyze response failed schema validation; refusing AWAITING_REVIEW');
        return;
      }
      const resp = parsed.data;
      if (resp.job_id !== payload.jobId || resp.attempt !== workerAttempt) {
        await finishAnalyzeFailure({ status: 'FAILED', reason: FAILURE_REASONS.WORKER_RESPONSE_MISMATCH, attempts: attempt });
        console.error('[m4] analyze response job/attempt mismatch; refusing AWAITING_REVIEW');
        return;
      }
      const result = resp.result;
      await setJob(env, jobKey, {
        status: 'PROCESSING',
        phase: 'ANALYZE',
        attempt,
        progress: stageProgress('finalizing', attempt),
      });

      // VERIFY-BEFORE-AWAITING_REVIEW (MANDATORY). The sealed artifact IS the product;
      // confirm the ciphertext actually landed in R2 at the DERIVED key with the reported
      // md5 before resting the job. Without an R2 binding there is no artifact channel to
      // verify against → fail closed (never rest an unverified AWAITING_REVIEW). The key
      // is DERIVED from the authenticated (device, job, attempt) — the worker's own
      // artifact_key is ignored here — so verification addresses only this job's output.
      const reportedMd5 = result.artifact_md5.toLowerCase();
      if (!ids || !env.R2) {
        await finishAnalyzeFailure({ status: 'FAILED', reason: FAILURE_REASONS.RESULT_UNVERIFIED, attempts: attempt });
        console.error('[m4] analyze artifact channel unavailable; refusing AWAITING_REVIEW');
        return;
      }
      const verifiedKey = artifactObjectKey(ids.deviceId, ids.jobId, workerAttempt);
      const artifactObject = await env.R2.get(verifiedKey);
      if (!artifactObject || r2ObjectMd5(artifactObject) !== reportedMd5) {
        await finishAnalyzeFailure({ status: 'FAILED', reason: FAILURE_REASONS.RESULT_UNVERIFIED, attempts: attempt });
        console.error('[m4] analyze artifact failed verification; refusing AWAITING_REVIEW');
        return;
      }
      // MD5 proves byte integrity, not usability. Parse only the non-secret ECIES envelope
      // (never decrypt ciphertext) so an arbitrary blob with a matching checksum cannot
      // strand the client in AWAITING_REVIEW with an undecryptable artifact.
      let envelopeAlg: string;
      try {
        const envelope = EncryptedArtifactSchema.strict().safeParse(
          JSON.parse(await artifactObject.text()),
        );
        if (!envelope.success) {
          await finishAnalyzeFailure({ status: 'FAILED', reason: FAILURE_REASONS.RESULT_UNVERIFIED, attempts: attempt });
          console.error('[m4] analyze artifact envelope invalid; refusing AWAITING_REVIEW');
          return;
        }
        if (
          envelope.data.context.analyzeJobId !== ids.jobId ||
          envelope.data.context.analyzeRevision !== resp.attempt ||
          envelope.data.context.alg !== envelope.data.alg
        ) {
          await finishAnalyzeFailure({ status: 'FAILED', reason: FAILURE_REASONS.RESULT_UNVERIFIED, attempts: attempt });
          console.error('[m4] analyze artifact context mismatch; refusing AWAITING_REVIEW');
          return;
        }
        envelopeAlg = envelope.data.alg;
      } catch {
        await finishAnalyzeFailure({ status: 'FAILED', reason: FAILURE_REASONS.RESULT_UNVERIFIED, attempts: attempt });
        console.error('[m4] analyze artifact envelope unreadable; refusing AWAITING_REVIEW');
        return;
      }
      const verifiedSize = (artifactObject as { size?: number }).size;

      // Commit the ARTIFACT POINTER before flipping the job to AWAITING_REVIEW: any
      // observer of AWAITING_REVIEW must be able to read the pointer. Only ciphertext-
      // referencing metadata is stored — NO plaintext transcript/translation ever lands
      // in Gateway KV (M4 #4). The worker seals `analyzeRevision = attempt` into the
      // ciphertext, so the public pointer MUST carry the same successful attempt. A retry
      // can produce a different stochastic analysis; hard-coding 1 would make its draft
      // impossible to approve because the client would present a newer base revision.
      const artifactKvKey = `artifact:${ids.deviceId}:${ids.jobId}`;
      const artifactPointer = {
        phase: 'ANALYZE',
        status: 'AWAITING_REVIEW',
        revision: resp.attempt,
        artifact_key: verifiedKey,
        artifact_md5: reportedMd5,
        artifact_size: verifiedSize,
        alg: envelopeAlg,
        diarization: result.diarization,
        segment_count: result.segment_count,
        createdAt: Date.now(),
      };
      await env.KV_CACHE.put(
        artifactKvKey,
        JSON.stringify(artifactPointer),
        { expirationTtl: JOB_TTL_S },
      );

      // Commit revision + terminal state to the lineage DO before exposing
      // AWAITING_REVIEW. Direct-dispatch tests/legacy callers without a DO retain the KV
      // fallback, but production Analyze requires the binding at submission time.
      if (env.JOB_COORDINATOR) {
        const analyzeStub = coordinatorStub(env.JOB_COORDINATOR, ids.deviceId, ids.jobId);
        try {
          if (revisionBase > 0) {
            const completion = await coordinatorCompleteAnalyzeRerun(analyzeStub, {
              baseRevision: revisionBase,
              analyzeRevision: resp.attempt,
              artifactPointer,
              meta: { elapsed, attempts: attempt },
            });
            if (completion.outcome !== 'completed' && completion.outcome !== 'idempotent') {
              throw new Error(`analyze rerun completion refused: ${completion.outcome}`);
            }
          } else {
            await coordinatorTransition(analyzeStub, 'AWAITING_REVIEW', {
              elapsed,
              attempts: attempt,
              analyzeRevision: resp.attempt,
              artifactPointer,
              progress: completeProgress(attempt),
            });
          }
        } catch (error) {
          // A KV projection failure can happen after the DO storage commit. Do not rerun
          // paid GPU work if the strong record already contains this exact completion.
          const authoritative = await coordinatorGet(analyzeStub).catch(() => null);
          if (
            authoritative?.status !== 'AWAITING_REVIEW' ||
            authoritative.analyzeRevision !== resp.attempt
          ) {
            throw error;
          }
          await env.KV_CACHE.put(
            jobKey,
            JSON.stringify({
              status: 'AWAITING_REVIEW',
              phase: 'ANALYZE',
              elapsed,
              attempts: attempt,
              analyzeRevision: resp.attempt,
              progress: completeProgress(attempt),
            }),
            { expirationTtl: JOB_TTL_S },
          );
        }
      } else {
        await env.KV_CACHE.put(
          jobKey,
          JSON.stringify({
            status: 'AWAITING_REVIEW',
            phase: 'ANALYZE',
            elapsed,
            attempts: attempt,
            analyzeRevision: resp.attempt,
            progress: completeProgress(attempt),
          }),
          { expirationTtl: JOB_TTL_S },
        );
      }
      return;
    } catch {
      if (controller.signal.aborted) {
        await terminateWorker(env, workerUrl, payload.jobId, 'timeout');
        await finishAnalyzeFailure({ status: 'TERMINATED_TIMEOUT', attempts: attempt });
        console.error('[station3] analyze worker timed out; termination signalled');
        return;
      }
      if (attempt < MAX_DISPATCH_ATTEMPTS) {
        await setJob(env, jobKey, {
          status: 'RETRYING',
          phase: 'ANALYZE',
          attempt,
          reason: FAILURE_REASONS.NETWORK,
          progress: stageProgress('retrying', attempt),
        });
        continue;
      }
      await finishAnalyzeFailure({ status: 'ERROR', reason: FAILURE_REASONS.NETWORK, attempts: attempt });
      return;
    } finally {
      clearTimeout(timeout);
    }
  }
}

// --- M4-S4: RENDER dispatch, bound to a human-approved manifest --------------
// The RENDER twin of dispatchToWorker. Where the legacy render pipeline trusts the
// client's raw /create config, the HITL render is driven ENTIRELY by the manifest a
// human approved. It re-fetches that canonical manifest from R2 at the content-addressed
// key approved/<device>/<analyzeJobId>/<hash>.json, RE-HASHES it (deterministicStringify →
// sha256) and refuses to render unless the digest equals the bound approvedManifestHash —
// defense-in-depth so a tampered/substituted R2 object can never reach the worker. It then
// drives /api/worker/render under an act='render' JWT, forwarding the approved segments'
// translatedText VERBATIM (no Qwen re-call) plus the resolved voice map.
//
// Three deliberate divergences from dispatchToWorker:
//   • a NEW fail-closed MANIFEST gate (fetch + re-hash) runs BEFORE any worker contact;
//     missing object / unparseable-or-invalid bytes / hash mismatch → FAILED
//     manifest_unverified (the human-approved text is the whole trust anchor).
//   • the input the worker fetches is the ANALYZE lineage's input: the audio GET + the
//     input size-cap HEAD are minted from inputAudioKey(device, analyzeJobId), NOT the
//     render jobId (the client uploaded the input ONCE, under the analyze key). The RESULT
//     object stays render-job-specific (resultObjectKey(device, renderJobId, attempt)).
//   • the request body carries the manifest's config/speakerMapping/segments verbatim plus
//     analyze_job_id + approved_manifest_hash (the worker binds to the same lineage).
// The quarantine / signing-gate / input-cap / per-attempt retry / too-fast-fraud / timeout
// / verify-before-DONE scaffolding MIRRORS dispatchToWorker's already-proven structure. The
// Every terminal sink uses the ANALYZE lineage as the input owner, so shared input is
// retained through retries and reclaimed only after Render can no longer need it.
export async function dispatchRenderToWorker(
  env: Bindings,
  payload: RenderRequest,
  jobKey: string,
  resolvedTarget?: ResolvedWorkerTarget,
): Promise<void> {
  const ids = parseJobKey(jobKey);
  // A RENDER job has its own jobId, but consumes the ANALYZE lineage's input object.
  // Every terminal render outcome must therefore reap analyzeJobId, never render jobId.
  const setRenderJob = (record: object): Promise<boolean> =>
    setJob(env, jobKey, record, payload.analyzeJobId);

  const maxMs = workerRequestTimeoutMs(env);
  const workerConfig = resolvedTarget ?? await resolveWorkerTarget(env);
  if (workerConfig.status === 'unavailable') {
    console.error('[dispatch] render worker target registry unavailable; deferring');
    throw new Error('worker target registry unavailable');
  }
  if (workerConfig.status !== 'valid') {
    await setRenderJob({
      status: 'FAILED',
      reason: FAILURE_REASONS.WORKER_CONFIGURATION_INVALID,
    });
    console.error('[dispatch] render worker configuration missing or invalid');
    return;
  }
  if (!workerTargetSupportsRequest(workerConfig, maxMs)) {
    await setRenderJob({
      status: 'FAILED',
      reason: FAILURE_REASONS.WORKER_TRANSPORT_UNSUPPORTED,
    });
    console.error('[dispatch] render worker transport cannot hold the configured request timeout');
    return;
  }
  const workerUrl = workerConfig.url;

  // Station 3: never dispatch to a worker previously QUARANTINED for anomaly.
  if (await env.KV_CACHE.get(`quarantine:${workerUrl}`)) {
    await setRenderJob({ status: 'FAILED', reason: FAILURE_REASONS.WORKER_QUARANTINED });
    console.error('[station3] refused render dispatch to quarantined worker');
    return;
  }

  // Station 2 fail-closed GATE: refuse to touch the worker without a valid signing key.
  const gate = await signGatewayJwt(env, payload.jobId, 'render', 1, '0'.repeat(64));
  if (!gate) {
    const reason = env.GATEWAY_JWT_PRIVATE_KEY
      ? FAILURE_REASONS.GATEWAY_KEY_INVALID
      : FAILURE_REASONS.GATEWAY_KEY_MISSING;
    await setRenderJob({ status: 'FAILED', reason });
    console.error(`[station2] cannot dispatch render: ${reason}`);
    return;
  }

  // M4-S4 MANIFEST GATE (fail-closed, BEFORE any worker contact). Re-fetch the canonical
  // approved manifest this render is bound to, RE-HASH it, and refuse unless the digest
  // matches. The key is content-addressed from the authenticated device + the payload's
  // analyzeJobId + approvedManifestHash, so a render can only ever load its OWN lineage's
  // approved object. Missing object, unparseable/invalid bytes, or a re-hash mismatch
  // (someone rewrote the R2 object under the same key) is TERMINAL — without the exact
  // human-approved manifest there is nothing legitimate to render.
  if (!ids || !env.R2) {
    await setRenderJob({ status: 'FAILED', reason: FAILURE_REASONS.MANIFEST_UNVERIFIED });
    console.error('[m4] render manifest channel unavailable; refusing dispatch');
    return;
  }
  const manifestKey = `approved/${ids.deviceId}/${payload.analyzeJobId}/${payload.approvedManifestHash}.json`;
  let manifest: ApprovedManifest;
  try {
    const obj = await env.R2.get(manifestKey);
    if (!obj) {
      await setRenderJob({ status: 'FAILED', reason: FAILURE_REASONS.MANIFEST_UNVERIFIED });
      console.error('[m4] approved manifest absent from R2; refusing render');
      return;
    }
    const parsedManifest = ApprovedManifestSchema.safeParse(JSON.parse(await obj.text()));
    if (!parsedManifest.success) {
      await setRenderJob({ status: 'FAILED', reason: FAILURE_REASONS.MANIFEST_UNVERIFIED });
      console.error('[m4] approved manifest failed schema validation; refusing render');
      return;
    }
    // RE-HASH the canonical form and require equality with the bound hash: bytes rewritten
    // under the original content-addressed key (substitution) re-hash to a different digest.
    const rehash = await sha256Hex(deterministicStringify(parsedManifest.data));
    if (rehash !== payload.approvedManifestHash) {
      await setRenderJob({ status: 'FAILED', reason: FAILURE_REASONS.MANIFEST_UNVERIFIED });
      console.error('[m4] approved manifest re-hash mismatch; refusing render');
      return;
    }
    manifest = parsedManifest.data;
  } catch {
    await setRenderJob({ status: 'FAILED', reason: FAILURE_REASONS.MANIFEST_UNVERIFIED });
    console.error('[m4] approved manifest fetch/parse failed; refusing render');
    return;
  }

  // M2-S5b (#5 + #4): enforce the 30 MB INPUT ceiling at the EDGE from the NATIVE R2
  // binding — never by probing a signed URL. The input belongs to the ANALYZE lineage, so
  // the HEAD is on the ANALYZE key (render reuses it). Only a CONFIRMED-oversize head fails.
  {
    const maxInputBytes = Number(env.MAX_INPUT_BYTES) > 0 ? Number(env.MAX_INPUT_BYTES) : MAX_INPUT_BYTES;
    const inHead = await env.R2.head(inputAudioKey(ids.deviceId, payload.analyzeJobId));
    const inSize = inHead ? (inHead as { size?: number }).size : undefined;
    if (typeof inSize === 'number' && inSize > maxInputBytes) {
      await setRenderJob({ status: 'FAILED', reason: FAILURE_REASONS.INPUT_TOO_LARGE });
      console.error('[m2-s5b] render input exceeds cap; refusing dispatch');
      return;
    }
  }

  // Station 3 timing bounds. Flat fraud floor (min()'d against maxMs/2 so the plausible
  // window [floor, maxMs] never inverts under env misconfiguration) — the segment-scaled
  // floor is unnecessary here: MIN_PLAUSIBLE_MS_FLOOR already gates a fabricated instant result.
  const floor = Number(env.MIN_PLAUSIBLE_MS) || MIN_PLAUSIBLE_MS_FLOOR;
  const minPlausibleMs = Math.min(floor, Math.floor(maxMs / 2));

  for (let attempt = 1; attempt <= MAX_DISPATCH_ATTEMPTS; attempt++) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), maxMs);

    try {
      // Re-mint the JWT per attempt (short 2m exp): a retry after backoff must never ship
      // an already-expired token. Scoped act='render' + this attempt (M3-S7): a render
      // token cannot be replayed onto /process (act='dispatch') or /analyze (act='analyze').
      // Input GET is minted from the ANALYZE key (render reuses the analyze input); the
      // result PUT + key stay render-job-specific (a re-dispatch writes a fresh object).
      const audioUrl = await mintDispatchAudioUrl(env, ids.deviceId, payload.analyzeJobId);
      const resultPutExpiresS = Math.ceil(maxMs / 1000) + 300;
      const resultUploadUrl = await mintDispatchResultUploadUrl(env, ids.deviceId, payload.jobId, attempt, resultPutExpiresS);
      const resultObjKey = resultObjectKey(ids.deviceId, payload.jobId, attempt);
      const workerBody = JSON.stringify({
        job_id: payload.jobId,
        attempt,
        analyze_job_id: payload.analyzeJobId,
        approved_manifest_hash: payload.approvedManifestHash,
        audio_url: audioUrl,
        audio_md5: payload.videoAudioMd5,
        result_upload_url: resultUploadUrl,
        result_key: resultObjKey,
        target_language: manifest.config.targetLanguage,
        translation_style: manifest.config.translationStyle,
        source_language: manifest.config.sourceLanguage,
        voice_map: manifest.speakerMapping,
        segments: manifest.segments,
      });
      const jwt = await signGatewayJwt(
        env,
        payload.jobId,
        'render',
        attempt,
        await sha256Hex(workerBody),
      );
      if (!jwt) {
        await setRenderJob({ status: 'FAILED', reason: FAILURE_REASONS.GATEWAY_KEY_INVALID, attempts: attempt });
        return;
      }
      const processingClaimed = await setRenderJob({
        status: 'PROCESSING',
        phase: 'RENDER',
        attempt,
        progress: stageProgress('rendering', attempt),
      });
      if (!processingClaimed) return;
      const startTime = Date.now();
      const res = await fetch(`${workerUrl}/api/worker/render`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${jwt}` },
        body: workerBody,
        signal: controller.signal,
      });

      const elapsed = Date.now() - startTime;

      if (!res.ok) {
        if (res.status >= 500 && attempt < MAX_DISPATCH_ATTEMPTS) {
          await setRenderJob({
            status: 'RETRYING',
            phase: 'RENDER',
            attempt,
            reason: FAILURE_REASONS.WORKER_ERROR,
            progress: stageProgress('retrying', attempt),
          });
          continue;
        }
        console.error('[render] worker returned non-ok status; failing', res.status);
        await setRenderJob({ status: 'FAILED', reason: FAILURE_REASONS.WORKER_ERROR, attempts: attempt });
        return;
      }

      // Anti-Fraud: an implausibly fast success is a fabricated render. Terminal.
      if (elapsed < minPlausibleMs) {
        await terminateWorker(env, workerUrl, payload.jobId, 'anomaly_too_fast');
        await setRenderJob({ status: 'REJECTED_FRAUD', elapsed, attempts: attempt });
        console.error('[station3] rejected render result: impossibly fast');
        return;
      }

      // M3-S8: VALIDATE THE FULL RESPONSE BEFORE DONE (render reuses WorkerResponseSchema —
      // the /process contract shape is identical). A 200 only proves the worker answered.
      const parsed = WorkerResponseSchema.safeParse(await res.json());
      if (!parsed.success) {
        await setRenderJob({ status: 'FAILED', reason: FAILURE_REASONS.WORKER_RESPONSE_INVALID, attempts: attempt });
        console.error('[m3] render response failed schema validation; refusing DONE');
        return;
      }
      const resp = parsed.data;
      if (resp.job_id !== payload.jobId || resp.attempt !== attempt) {
        await setRenderJob({ status: 'FAILED', reason: FAILURE_REASONS.WORKER_RESPONSE_MISMATCH, attempts: attempt });
        console.error('[m3] render response job/attempt mismatch; refusing DONE');
        return;
      }
      const raw = resp.result as Record<string, any>;
      await setRenderJob({
        status: 'PROCESSING',
        phase: 'RENDER',
        attempt,
        progress: stageProgress('finalizing', attempt),
      });

      // M4: VERIFY-BEFORE-DONE. Confirm the reported result actually landed in R2 at the
      // DERIVED render-job key with the matching md5 before flipping DONE — the worker's own
      // result_key is ignored here, so verification addresses only this render's own output.
      // Missing, malformed, absent, or mismatched artifacts are terminal for this M4 path.
      // Legacy /process retains its compatibility behavior in dispatchToWorker.
      const reportedMd5 = typeof raw.result_md5 === 'string' ? raw.result_md5.toLowerCase() : '';
      const reportedSha256 = typeof raw.result_sha256 === 'string' ? raw.result_sha256.toLowerCase() : '';
      const reportedSize = raw.result_size;
      if (
        !/^[a-f0-9]{32}$/.test(reportedMd5) ||
        !/^[a-f0-9]{64}$/.test(reportedSha256) ||
        !Number.isSafeInteger(reportedSize) ||
        reportedSize <= 0
      ) {
        await setRenderJob({ status: 'FAILED', reason: FAILURE_REASONS.RESULT_UNVERIFIED, attempts: attempt });
        console.error('[m6] render result integrity metadata missing or invalid; refusing DONE');
        return;
      }

      const verifiedResultKey = resultObjectKey(ids.deviceId, payload.jobId, attempt);
      const head = await env.R2.head(verifiedResultKey);
      if (!head || r2ObjectMd5(head) !== reportedMd5 || (head as { size?: number }).size !== reportedSize) {
        await setRenderJob({ status: 'FAILED', reason: FAILURE_REASONS.RESULT_UNVERIFIED, attempts: attempt });
        console.error('[m4] render result artifact failed verification; refusing DONE');
        return;
      }
      const verifiedResultSize = (head as { size?: number }).size;

      // Zero-Logging AT REST: allowlist safe metadata + the VERIFIED R2 artifact reference
      // before the 24h KV put. Same allowlist as the render pipeline (no plaintext content).
      const storedResult = {
        status: raw.status,
        message: raw.message,
        device_used: raw.device_used,
        pipeline: raw.pipeline,
        separated: raw.separated,
        watermarked: raw.watermarked,
        distinct_voices: raw.distinct_voices,
        notes: raw.notes,
        result_key: verifiedResultKey,
        result_md5: reportedMd5,
        result_sha256: reportedSha256,
        result_size: verifiedResultSize,
      };
      // Commit the ARTIFACT before flipping DONE (any observer of DONE must be able to read
      // the result). The render-specific status writer then reaps the ANALYZE lineage input.
      const resultKey = jobKey.replace(/^job:/, 'result:');
      await env.KV_CACHE.put(
        resultKey,
        JSON.stringify({ job_id: resp.job_id, result: storedResult }),
        { expirationTtl: JOB_TTL_S },
      );
      await setRenderJob({
        status: 'DONE',
        phase: 'RENDER',
        elapsed,
        attempts: attempt,
        progress: completeProgress(attempt),
      });
      return;
    } catch {
      if (controller.signal.aborted) {
        await terminateWorker(env, workerUrl, payload.jobId, 'timeout');
        await setRenderJob({ status: 'TERMINATED_TIMEOUT', attempts: attempt });
        console.error('[station3] render worker timed out; termination signalled');
        return;
      }
      if (attempt < MAX_DISPATCH_ATTEMPTS) {
        await setRenderJob({
          status: 'RETRYING',
          phase: 'RENDER',
          attempt,
          reason: FAILURE_REASONS.NETWORK,
          progress: stageProgress('retrying', attempt),
        });
        continue;
      }
      await setRenderJob({ status: 'ERROR', reason: FAILURE_REASONS.NETWORK, attempts: attempt });
      return;
    } finally {
      clearTimeout(timeout);
    }
  }
}

// KV-only deployments have no compare-and-swap primitive. Serialize operations for a
// job within this isolate so cancellation and a late background dispatch cannot interleave
// their read/check/write sequences. The durable-object path remains the cross-isolate
// authority in production; this lock is a compatibility safeguard for legacy deployments.
const legacyJobLocks = new Map<string, Promise<void>>();

async function withLegacyJobLock<T>(jobKey: string, fn: () => Promise<T>): Promise<T> {
  const previous = legacyJobLocks.get(jobKey) ?? Promise.resolve();
  let release!: () => void;
  const current = new Promise<void>((resolve) => {
    release = resolve;
  });
  legacyJobLocks.set(jobKey, current);
  await previous;
  try {
    return await fn();
  } finally {
    release();
    if (legacyJobLocks.get(jobKey) === current) legacyJobLocks.delete(jobKey);
  }
}

async function setJob(
  env: Bindings,
  jobKey: string,
  record: object,
  inputOwnerJobId?: string,
): Promise<boolean> {
  return withLegacyJobLock(jobKey, () => setJobUnlocked(env, jobKey, record, inputOwnerJobId));
}

async function setJobUnlocked(
  env: Bindings,
  jobKey: string,
  record: object,
  inputOwnerJobId?: string,
): Promise<boolean> {
  // Keep every terminal projection sticky: late retries and stale waitUntil tasks may
  // observe it, but cannot regress it or replace its original terminal reason.
  const current = (await env.KV_CACHE.get(jobKey, { type: 'json' })) as { status?: unknown } | null;
  if (current && typeof current.status === 'string' && isTerminal(current.status)) return false;

  await env.KV_CACHE.put(jobKey, JSON.stringify(record), { expirationTtl: JOB_TTL_S });
  // M2 lifecycle: the KV projection is the single choke point every terminal status
  // flows through — reclaim the job's uploaded input audio the moment it turns terminal.
  await deleteJobInputIfTerminal(
    env,
    jobKey,
    (record as { status?: unknown }).status,
    inputOwnerJobId,
  );
  return true;
}

// M2 — R2 object key for a job's uploaded INPUT audio. MUST stay in lock-step with
// mintJobAudioUrls (r2presign.ts), which is what actually created the object.
export function inputAudioKey(deviceId: string, jobId: string): string {
  return `audio/${deviceId}/${jobId}.wav`;
}

// M2 — recover (deviceId, jobId) from a `job:<deviceId>:<jobId>` KV key. jobId is
// ^[A-Za-z0-9_-]+$ (never contains ':'), so splitting on the LAST ':' is exact even
// if deviceId itself held one. Returns null when the key is malformed (defensive).
export function parseJobKey(jobKey: string): { deviceId: string; jobId: string } | null {
  const rest = jobKey.startsWith('job:') ? jobKey.slice(4) : jobKey;
  const sep = rest.lastIndexOf(':');
  if (sep <= 0 || sep >= rest.length - 1) return null;
  return { deviceId: rest.slice(0, sep), jobId: rest.slice(sep + 1) };
}

// M2-S2 — mint the SHORT-LIVED presigned GET the worker uses to fetch a job's input
// audio. The key is DERIVED from the authenticated (deviceId, jobId) — never a
// client-supplied path string — so a minted GET can only ever address THIS job's
// object (owner/jobId binding; forecloses the F-R2-01 path-injection class entirely).
// Returns '' when R2 is not provisioned (KV-only test env / pre-R2 deploy) — those
// paths mock the worker fetch, so the empty URL is inert; production always has creds.
async function mintDispatchAudioUrl(env: Bindings, deviceId: string, jobId: string): Promise<string> {
  const { R2_ACCOUNT_ID, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY } = env;
  if (!R2_ACCOUNT_ID || !R2_BUCKET || !R2_ACCESS_KEY_ID || !R2_SECRET_ACCESS_KEY) return '';
  const expiresSeconds = Number(env.DISPATCH_GET_EXPIRES_S) || DISPATCH_GET_EXPIRES_S;
  return presignS3Url({
    accessKeyId: R2_ACCESS_KEY_ID,
    secretAccessKey: R2_SECRET_ACCESS_KEY,
    method: 'GET',
    host: `${R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
    path: `/${R2_BUCKET}/${inputAudioKey(deviceId, jobId)}`,
    region: env.R2_REGION ?? 'auto',
    amzDate: amzDate(new Date()),
    expiresSeconds,
  });
}

// M2-S3 — R2 object key for a job attempt's RESULT (dubbed) audio. The <attempt> is
// part of the key so a re-dispatch (transient worker blip) writes a fresh object
// instead of racing the previous attempt's upload; the Gateway HEAD-verifies the
// SPECIFIC attempt it accepted. Derived from the authenticated (deviceId, jobId) —
// never a worker-supplied string — so it only ever addresses this job's own output.
export function resultObjectKey(deviceId: string, jobId: string, attempt: number): string {
  return `results/${deviceId}/${jobId}/${attempt}.wav`;
}

// M2-S3 — mint the SHORT-LIVED presigned PUT the worker uses to UPLOAD its rendered
// result to R2. The worker never holds R2 credentials (Zero-Trust): it receives only
// this one-object, time-boxed URL. The window must outlive the render itself (the
// upload happens only after rendering finishes), so it is sized to the dispatch
// timeout plus an upload margin — still far under the SigV4 7-day ceiling. Returns ''
// when R2 is not provisioned (KV-only test env / pre-R2 deploy).
async function mintDispatchResultUploadUrl(
  env: Bindings,
  deviceId: string,
  jobId: string,
  attempt: number,
  expiresSeconds: number,
): Promise<string> {
  const { R2_ACCOUNT_ID, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY } = env;
  if (!R2_ACCOUNT_ID || !R2_BUCKET || !R2_ACCESS_KEY_ID || !R2_SECRET_ACCESS_KEY) return '';
  return presignS3Url({
    accessKeyId: R2_ACCESS_KEY_ID,
    secretAccessKey: R2_SECRET_ACCESS_KEY,
    method: 'PUT',
    host: `${R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
    path: `/${R2_BUCKET}/${resultObjectKey(deviceId, jobId, attempt)}`,
    region: env.R2_REGION ?? 'auto',
    amzDate: amzDate(new Date()),
    expiresSeconds,
  });
}

// M4-S3 — R2 object key for a job attempt's SEALED analyze artifact (ECIES ciphertext,
// ADR 0002). The `.bin` extension marks it as opaque ciphertext (not the `.wav` render
// result). The <attempt> is part of the key so a re-dispatch writes a fresh object
// instead of racing the previous attempt's upload; the Gateway HEAD-verifies the SPECIFIC
// attempt it accepted. Derived from the authenticated (deviceId, jobId) — never a
// worker-supplied string — so it only ever addresses this job's own artifact.
export function artifactObjectKey(deviceId: string, jobId: string, attempt: number): string {
  return `artifacts/${deviceId}/${jobId}/${attempt}.bin`;
}

// M4-S3 — mint the SHORT-LIVED presigned PUT the worker uses to UPLOAD its SEALED analyze
// artifact to R2. Mirrors mintDispatchResultUploadUrl exactly (the worker never holds R2
// credentials — it receives only this one-object, time-boxed URL), differing only in the
// object key (artifacts/*.bin vs results/*.wav). Returns '' when R2 is not provisioned.
async function mintDispatchArtifactUploadUrl(
  env: Bindings,
  deviceId: string,
  jobId: string,
  attempt: number,
  expiresSeconds: number,
): Promise<string> {
  const { R2_ACCOUNT_ID, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY } = env;
  if (!R2_ACCOUNT_ID || !R2_BUCKET || !R2_ACCESS_KEY_ID || !R2_SECRET_ACCESS_KEY) return '';
  return presignS3Url({
    accessKeyId: R2_ACCESS_KEY_ID,
    secretAccessKey: R2_SECRET_ACCESS_KEY,
    method: 'PUT',
    host: `${R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
    path: `/${R2_BUCKET}/${artifactObjectKey(deviceId, jobId, attempt)}`,
    region: env.R2_REGION ?? 'auto',
    amzDate: amzDate(new Date()),
    expiresSeconds,
  });
}

// M2-S3 — normalize an R2Object's md5 to lowercase hex for checksum comparison.
// R2 exposes it two ways depending on how the object was written: `checksums.md5`
// (an ArrayBuffer via the native binding; a hex string in the test harness) and the
// quoted-hex `httpEtag` (the md5 for a single-part S3/presigned PUT, which is how the
// worker uploads). Prefer the explicit checksum, fall back to the etag. Returns '' if
// neither is present (then verification cannot pass — fail closed).
function r2ObjectMd5(obj: unknown): string {
  const o = obj as { checksums?: { md5?: unknown }; httpEtag?: unknown; etag?: unknown } | null;
  const c = o?.checksums?.md5;
  if (typeof c === 'string') return c.toLowerCase();
  if (c && (c instanceof ArrayBuffer || ArrayBuffer.isView(c))) {
    const bytes = c instanceof ArrayBuffer ? new Uint8Array(c) : new Uint8Array((c as ArrayBufferView).buffer);
    return [...bytes].map((x) => x.toString(16).padStart(2, '0')).join('');
  }
  const etag = o?.httpEtag ?? o?.etag;
  return typeof etag === 'string' ? etag.replace(/"/g, '').toLowerCase() : '';
}

// M2 — once a job is TERMINAL it can never be re-dispatched, so its uploaded input
// audio is dead weight (and lingering plaintext audio at rest). Delete it. BEST-EFFORT
// on purpose: a failed delete must NOT block the terminal status write (that would wedge
// the job); the orphan sweeper (later M2 slice) + bucket lifecycle are the backstops.
// Idempotent — R2 delete of an absent key is a no-op, so a redelivered terminal is safe.
async function deleteJobInputIfTerminal(
  env: Bindings,
  jobKey: string,
  status: unknown,
  inputOwnerJobId?: string,
): Promise<void> {
  // AWAITING_REVIEW is terminal only for the ANALYZE queue attempt. Its input is still
  // required by the later RENDER job, so neither direct cleanup nor the sweeper may reap it.
  if (typeof status !== 'string' || status === 'AWAITING_REVIEW' || !isTerminal(status)) return;
  const bucket = env.R2;
  if (!bucket) return; // KV-only env (legacy tests / pre-R2 deploy) — nothing to reclaim
  const parsed = parseJobKey(jobKey);
  if (!parsed) return;
  const ownerJobId = inputOwnerJobId ?? parsed.jobId;
  try {
    // A RENDER consumes another job's (the ANALYZE lineage's) input. Atomically mark
    // that lineage consumed before deletion, so a second producer cannot claim the
    // audio in the release→delete window. Legacy/direct-dispatch callers with no lineage
    // record retain best-effort cleanup; an existing mismatched claim always blocks it.
    if (ownerJobId !== parsed.jobId && env.JOB_COORDINATOR) {
      const lineageStub = coordinatorStub(env.JOB_COORDINATOR, parsed.deviceId, ownerJobId);
      const finishInput = {
        renderJobId: parsed.jobId,
        now: Date.now(),
      };
      let finish;
      try {
        finish = await coordinatorFinishRender(lineageStub, finishInput);
      } catch {
        // If DO storage committed but its marker projection failed, the idempotent
        // retry repairs inputConsumedAt before any delete is attempted.
        finish = await coordinatorFinishRender(lineageStub, finishInput);
      }
      if (finish.outcome === 'mismatch' || finish.outcome === 'not_claimed') return;
    }
    await bucket.delete(inputAudioKey(parsed.deviceId, ownerJobId));
  } catch {
    // Zero-Logging: no key/URL in logs. The sweeper reclaims anything left behind.
    console.error('[m2] terminal input cleanup deferred (delete failed)');
  }
}

// M2-S5g (#2) — reclaim a job's uploaded INPUT when create is REFUSED. The client uploads
// audio/<deviceId>/<jobId>.wav BEFORE calling create (presign → PUT → create), so a
// refusal orphans that object (lingering plaintext audio at rest + storage cost). This is
// the failed-CREATE half of acceptance #2 (the dispatch half is deleteJobInputIfTerminal).
// Called ONLY from the rate-limit branch — the one refusal reached AFTER the idempotency
// check proved no job record exists — so no live job can own this input. The key is DERIVED
// from the AUTHENTICATED (deviceId, jobId), never the client's videoAudioKey string, so a
// refused create can only ever delete its OWN device+job object. Best-effort + idempotent
// (R2 delete of an absent key is a no-op); the orphan sweeper + bucket lifecycle back it up.
async function reapUncreatedInput(env: Bindings, deviceId: string, jobId: string): Promise<void> {
  const bucket = env.R2;
  if (!bucket) return; // KV-only env (legacy tests / pre-R2 deploy) — nothing to reclaim
  try {
    await bucket.delete(inputAudioKey(deviceId, jobId));
  } catch {
    console.error('[m2-s5g] refused-create input cleanup deferred (delete failed)');
  }
}

/** Station 2: sign an asymmetric ES256 gateway→worker JWT, SCOPED to a single
 *  action (M3-S7/A3). Returns null (fail closed) when the private key is missing
 *  or invalid — or when the claims don't satisfy the shared schema — so callers
 *  refuse to dispatch/serve rather than proceed with an unscoped/unauthenticated
 *  token. `attempt` is supplied ONLY for dispatch (each render attempt re-mints
 *  with its own number, so an attempt-1 token can't be replayed onto attempt 2). */
async function signGatewayJwt(
  env: Bindings,
  jobId: string,
  act: WorkerJwtAct,
  attempt?: number,
  bodyDigest?: string,
): Promise<string | null> {
  const pem = env.GATEWAY_JWT_PRIVATE_KEY;
  if (!pem) return null;
  try {
    const privateKey = await jose.importPKCS8(pem, 'ES256');
    // Self-validate the claim set against the SHARED schema before signing: a bad
    // act/attempt throws here and fails closed to null, rather than minting a token
    // whose scope the worker would then reject on the wire. `aud` is stamped so the
    // worker (which REQUIRES its audience) accepts it. attempt omitted → no claim.
    if (bodyDigest !== undefined && !/^[0-9a-f]{64}$/.test(bodyDigest)) return null;
    const validated = WorkerJwtClaimsSchema.parse({
      role: 'gateway',
      act,
      jobId,
      aud: WORKER_JWT_AUDIENCE,
      ...(attempt !== undefined ? { attempt } : {}),
      ...(bodyDigest !== undefined ? { bodyDigest } : {}),
    });
    return await new jose.SignJWT(validated)
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
  const jwt = await signGatewayJwt(env, jobId, 'terminate');
  if (!jwt) return;                          // cannot authenticate; quarantine still stands
  try {
    await fetch(`${workerUrl}/api/worker/terminate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${jwt}` },
      // job_id lets the worker bind token.jobId == body.job_id (M3-S7): a terminate
      // token for job A can't be redirected to quarantine under a job-B request.
      body: JSON.stringify({ reason, job_id: jobId }),
    });
  } catch {
    // Worker may already be gone; the quarantine flag is the durable signal.
  }
}

// --- M1: durable dispatch Queue consumer (ADR 0001) ------------------------
// The message a /create producer enqueues. The full payload rides the message so
// the consumer can dispatch without a second read. The CHAR-level field caps
// (validateJobSize) do NOT by themselves keep this under the 128 KiB queue-message
// limit — they miss byte-level vectors (the `original_text` hole; sheer segment
// count). The BYTE-level Queue-fit guard at /create (MAX_QUEUE_PAYLOAD_BYTES) is what
// guarantees this envelope fits, so the producer's `.send()` below can never overflow.
// M4-S3 — the message is a DISCRIMINATED UNION keyed on `phase`. A render job (the /create
// producer) carries no phase and a JobRequest; an ANALYZE job (the /analyze producer)
// carries phase:'ANALYZE' and an AnalyzeRequest. The consumer narrows on `phase` to route
// to the right dispatcher (dispatchToWorker vs dispatchAnalyzeToWorker) with the correctly
// typed payload — a render payload can never reach the analyze dispatch or vice-versa.
// M4-S4 — the union gains a RENDER arm (phase:'RENDER', a RenderRequest). Adding it here
// FORCES the consumer's phase-narrowing branch below to handle it (a render payload can
// never reach the legacy /process or the analyze dispatch, and vice-versa).
export type QueueJob =
  | { deviceId: string; jobId: string; phase?: undefined; payload: JobRequest }
  | { deviceId: string; jobId: string; phase: 'ANALYZE'; revisionBase?: number; payload: AnalyzeRequest }
  | { deviceId: string; jobId: string; phase: 'RENDER'; payload: RenderRequest };

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
    // Narrow on the whole message BEFORE touching `payload`: `body.phase` is the union
    // discriminant, so destructuring `payload` up-front would widen it to the union and
    // lose the phase→payload-type link the dispatch branch below relies on.
    const body = message.body;
    const { deviceId, jobId } = body;
    try {
      if (!env.JOB_COORDINATOR) throw new Error('JOB_COORDINATOR binding missing');
      const stub = coordinatorStub(env.JOB_COORDINATOR, deviceId, jobId);
      const jobKey = `job:${deviceId}:${jobId}`;

      // (1) DO-terminal guard. At-least-once delivery can redeliver a finished job; if
      // the DO authority is already terminal, ACK without re-dispatching.
      const current = await coordinatorGet(stub);
      if (current && isTerminal(current.status)) {
        message.ack();
        continue;
      }

      // (2) KV-terminal heal (M1 review, Bug C). A prior delivery may have completed the
      // render — committing a TERMINAL KV `job:` projection — but crashed before syncing
      // the DO (leaving it DISPATCHING, possibly with an expired lease). Re-dispatching
      // would burn a second GPU render AND drag the KV projection back off its terminal
      // state (a status regression the poll contract would surface). The terminal KV is
      // definitive proof the work finished: sync the DO to it (best-effort) and ACK.
      const kvRec = (await env.KV_CACHE.get(jobKey, { type: 'json' })) as
        | ({ status?: JobStatus } & Record<string, unknown>)
        | null;
      if (kvRec?.status && isTerminal(kvRec.status)) {
        try {
          await coordinatorTransition(stub, kvRec.status, kvRec);
        } catch {
          /* DO may be at an edge from which this heal is illegal; KV is the poll
             authority and is already terminal, so ACK regardless of the DO sync. */
        }
        message.ack();
        continue;
      }

      // (2.5) Financial Kill Switch re-check at the point of spend (M3-S6b). The create-time
      // guard blocks NEW enqueues, but a job enqueued moments BEFORE the switch tripped is
      // already in the queue; dispatching it now would burn a GPU render the switch is meant
      // to stop. Re-check here and DEFER (retry) while active — never ACK (that would silently
      // drop a legitimate job) — so the job resumes automatically once the switch clears. Runs
      // AFTER the terminal guards above so an already-finished job still ACKs (draining it
      // costs nothing) rather than being re-deferred forever. killSwitchBlocks FAILS CLOSED:
      // a switch-store read error → treat as active → defer, matching the HTTP guards.
      if (await killSwitchBlocks(env)) {
        message.retry();
        continue;
      }

      // Resolve the control-plane target before taking the long dispatch lease. A
      // transient registry outage is infrastructure backpressure, not a terminal job
      // configuration error; defer the Queue message and retry without consuming a lease.
      const workerTarget = await resolveWorkerTarget(env);
      if (workerTarget.status === 'unavailable') {
        message.retry();
        continue;
      }

      // (3) Dispatch lease (M1 review, Bug B). Atomically claim the render so a
      // concurrent redelivery cannot fire a SECOND GPU render (doubled spend). A FRESH
      // lease held by a live delivery ⇒ defer (retry). An EXPIRED lease ⇒ the owner died,
      // take over. Terminal ⇒ ACK. The lease must cover the worst-case render window,
      // including dispatchToWorker's internal transient-retry loop.
      const maxMs = workerRequestTimeoutMs(env);
      // An async ANALYZE owns the DO dispatch lease for the background deadline;
      // the individual RunPod POST/GET calls remain bounded by maxMs. Reserve one
      // transport window per possible attempt in addition to the long compute window:
      // an early submit/contract 5xx may retry before the final task is polled.
      const leaseMs = body.phase === 'ANALYZE' && asyncAnalyzeEnabled(env)
        ? asyncAnalyzeTimeoutMs(env) + MAX_DISPATCH_ATTEMPTS * maxMs + 60_000
        : MAX_DISPATCH_ATTEMPTS * maxMs;
      const claim = await coordinatorAcquireDispatch(stub, { now: Date.now(), leaseMs });
      if (claim.outcome === 'terminal') {
        message.ack();
        continue;
      }
      if (claim.outcome === 'busy') {
        // A live delivery owns the render — defer without re-dispatching. Redelivery
        // after the lease window lets a genuinely-dead owner be taken over.
        message.retry();
        continue;
      }

      // Acquired. Dispatch: owns the worker round-trip, Station-3 timing bounds, the
      // in-request transient-retry loop, and the KV job:/result:/artifact: projection
      // writes. When it returns, the KV `job:` status is always terminal (AWAITING_REVIEW
      // for analyze — a genuine sink — or DONE/FAILED/… for render).
      // Route by phase (M4-S3): an ANALYZE message drives the encrypted-artifact analyze
      // dispatch; a render message drives the render pipeline. `body.phase` narrows
      // `body.payload` to the matching request type, so neither can cross to the other.
      if (body.phase === 'ANALYZE') {
        await dispatchAnalyzeToWorker(
          env,
          body.payload,
          jobKey,
          body.revisionBase ?? 0,
          workerTarget,
        );
      } else if (body.phase === 'RENDER') {
        // M4-S4: a manifest-bound render (fetch+re-hash the approved manifest, then drive
        // /api/worker/render with its human-approved text verbatim).
        await dispatchRenderToWorker(env, body.payload, jobKey, workerTarget);
      } else {
        await dispatchToWorker(env, body.payload, jobKey, workerTarget);
      }

      // Sync the DO authority to the terminal state the dispatch landed on, so any
      // later redelivery short-circuits at the DO-terminal guard above.
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

// M2-S5d — the cron entrypoint. wrangler's [triggers] crons invokes `.scheduled` on a
// schedule; it runs the orphan sweeper as a best-effort backstop that reclaims R2 objects
// outliving their job. Zero-Logging: log only counts, never keys; swallow all errors so a
// bad tick can never wedge future ticks. Live cron firing stays BLOCKED_EXTERNAL.
async function handleScheduled(
  event: { scheduledTime?: number; cron?: string },
  env: Bindings,
  _ctx: ExecutionContext,
): Promise<void> {
  try {
    const s = await sweepOrphans(env, event.scheduledTime ?? 0);
    console.log(
      `[sweeper] scanned in=${s.scannedInputs} res=${s.scannedResults} ` +
        `art=${s.scannedArtifacts} app=${s.scannedApproved} ` +
        `deleted in=${s.deletedInputs} res=${s.deletedResults} ` +
        `art=${s.deletedArtifacts} app=${s.deletedApproved} ptr=${s.deletedPointers} ` +
        `kept=${s.kept}`,
    );
  } catch {
    console.error('[sweeper] run failed; next tick retries');
  }
}

// Serve ALL roles from the one Worker: `.fetch` (the Hono router), `.queue` (the dispatch
// consumer), and `.scheduled` (the cron sweeper). Tests keep importing the default app and
// using .request; the runtime additionally invokes .queue and .scheduled.
(app as unknown as { queue: typeof handleJobQueue }).queue = handleJobQueue;
(app as unknown as { scheduled: typeof handleScheduled }).scheduled = handleScheduled;

// Re-export the Durable Object class so wrangler's [[durable_objects.bindings]] +
// [[migrations]] can find it as a named export on the Worker entrypoint.
export { JobCoordinator, ReplayGuard, RateLimiter, KillSwitch, WorkerTargetRegistry };

export default app;
