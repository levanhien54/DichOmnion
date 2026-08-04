// JobCoordinator — Durable Object that is the ATOMIC source of truth for a job's
// lifecycle (M1, ADR 0001). One DO instance per (deviceId, jobId), addressed via
// idFromName("<deviceId>:<jobId>"). Because a DO is single-threaded per object, its
// create() check-and-set and transition() are atomic — this replaces the KV
// get-then-put idempotency (TOCTOU) and the KV last-writer-wins status writes.
//
// Deliberately a CLASSIC fetch()-based DO (no `cloudflare:workers` import): it needs
// no modern compatibility_date, and — crucially — it instantiates under plain
// node-vitest so its logic is unit-tested WITHOUT the workers-pool test runtime.
// The runtime and the tests both drive it through the SAME door: coordinator*()
// build the Request, call `.fetch()`, and parse the Response.

// The KV `job:<dev>:<id>` projection (what the poll endpoint reads) is written
// THROUGH every state change so the existing poll/download contract and its tests
// stay green. DO storage is the authority; KV is a read replica.

type Bindings = {
  KV_CACHE: KVNamespace;
};

export type JobStatus =
  | 'QUEUED'
  | 'DISPATCHING'
  | 'PROCESSING'
  | 'RETRYING'
  | 'DONE'
  | 'FAILED'
  | 'REJECTED_FRAUD'
  | 'TERMINATED_TIMEOUT'
  | 'ERROR';

// Terminal states are STICKY: once reached, no transition leaves them (a redelivered
// queue message or a late retry can never regress DONE → QUEUED). This is the W2 fix.
export const TERMINAL_STATES: readonly JobStatus[] = [
  'DONE',
  'FAILED',
  'REJECTED_FRAUD',
  'TERMINATED_TIMEOUT',
  'ERROR',
];

export function isTerminal(status: string): boolean {
  return (TERMINAL_STATES as readonly string[]).includes(status);
}

// Legal transitions from each NON-terminal state. Terminal states have no outgoing
// edges (absent key ⇒ sticky). QUEUED→DISPATCHING→(PROCESSING|terminal); a transient
// blip parks at RETRYING→DISPATCHING. PROCESSING is reserved for M4 progress signals.
export const ALLOWED_TRANSITIONS: Record<string, readonly JobStatus[]> = {
  QUEUED: ['DISPATCHING'],
  DISPATCHING: ['PROCESSING', 'RETRYING', 'DONE', 'FAILED', 'REJECTED_FRAUD', 'TERMINATED_TIMEOUT', 'ERROR'],
  PROCESSING: ['RETRYING', 'DONE', 'FAILED', 'REJECTED_FRAUD', 'TERMINATED_TIMEOUT', 'ERROR'],
  RETRYING: ['DISPATCHING'],
};

const JOB_TTL_S = 86_400; // mirror index.ts: job/result KV records live 24h

export interface JobRecord {
  deviceId: string;
  jobId: string;
  status: JobStatus;
  createdAt: number;
  updatedAt: number;
  etaSeconds?: number;
  [k: string]: unknown; // carries per-transition meta (elapsed, attempts, reason, code…)
}

const RECORD_KEY = 'record'; // one job per DO instance

export class JobCoordinator {
  private state: DurableObjectState;
  private env: Bindings;

  constructor(state: DurableObjectState, env: Bindings) {
    this.state = state;
    this.env = env;
  }

  private jobKvKey(r: { deviceId: string; jobId: string }): string {
    return `job:${r.deviceId}:${r.jobId}`;
  }

  // Write-through the KV projection the poll endpoint reads. Shape MUST stay
  // compatible with the legacy record: { status, createdAt, etaSeconds, ...meta }.
  // Internal-only fields (deviceId/jobId/updatedAt, plus the producer hand-off marker
  // `enqueued` and the consumer's `leaseExpiresAt`) are stripped so they never leak
  // into the client-facing poll contract.
  private async projectToKv(record: JobRecord): Promise<void> {
    const { deviceId, jobId, updatedAt, enqueued, leaseExpiresAt, ...rest } = record;
    void deviceId;
    void jobId;
    void updatedAt;
    void enqueued;
    void leaseExpiresAt;
    await this.env.KV_CACHE.put(this.jobKvKey(record), JSON.stringify(rest), {
      expirationTtl: JOB_TTL_S,
    } as any);
  }

  // Atomic check-and-set. Existing record ⇒ idempotent hit (never re-initialized).
  private async create(input: {
    deviceId: string;
    jobId: string;
    etaSeconds?: number;
  }): Promise<{ record: JobRecord; idempotent: boolean }> {
    const existing = await this.state.storage.get<JobRecord>(RECORD_KEY);
    if (existing) return { record: existing, idempotent: true };

    const now = Date.now();
    const record: JobRecord = {
      deviceId: input.deviceId,
      jobId: input.jobId,
      status: 'QUEUED',
      etaSeconds: input.etaSeconds,
      createdAt: now,
      updatedAt: now,
    };
    await this.state.storage.put(RECORD_KEY, record);
    await this.projectToKv(record);
    return { record, idempotent: false };
  }

  // Enforce the state machine. Returns { applied, record, reason }:
  //   • from a terminal state → applied:false, reason:'terminal' (sticky, NOT an error)
  //   • to the same state     → applied:false, reason:'noop' (redelivery safety)
  //   • illegal live edge     → throws (a coding bug worth surfacing)
  //   • legal edge            → applied:true, record updated + projected to KV
  private async transition(
    to: JobStatus,
    meta?: Record<string, unknown>,
  ): Promise<{ applied: boolean; record: JobRecord; reason?: string }> {
    const record = await this.state.storage.get<JobRecord>(RECORD_KEY);
    if (!record) throw new Error('transition on a job that was never created');

    if (record.status === to) return { applied: false, record, reason: 'noop' };
    if (isTerminal(record.status)) return { applied: false, record, reason: 'terminal' };

    const allowed = ALLOWED_TRANSITIONS[record.status] ?? [];
    if (!allowed.includes(to)) {
      throw new Error(`illegal transition ${record.status} → ${to}`);
    }

    const next: JobRecord = { ...record, ...(meta ?? {}), status: to, updatedAt: Date.now() };
    await this.state.storage.put(RECORD_KEY, next);
    await this.projectToKv(next);
    return { applied: true, record: next };
  }

  // M1 review (Bug A). The producer commits create() BEFORE it enqueues the dispatch
  // message, and those two durable writes are non-atomic: a send() that throws (or an
  // isolate evicted between them) leaves the job committed as QUEUED with NO message on
  // the queue — an orphan. Because the idempotency peek treats "a record exists" as
  // "already handed off", such an orphan would be re-accepted (202) forever, never
  // dispatched — the exact W1 silent-loss M1 exists to kill, reappearing on the
  // producer side. `enqueued` is the durable proof that a hand-off was CONFIRMED; the
  // producer sets it only AFTER a successful send, so a still-QUEUED record lacking it
  // is a heal candidate the peek re-enqueues. Not projected to KV (internal marker).
  private async markEnqueued(): Promise<{ record: JobRecord }> {
    const record = await this.state.storage.get<JobRecord>(RECORD_KEY);
    if (!record) throw new Error('markEnqueued on a job that was never created');
    if (record.enqueued === true) return { record };
    const next: JobRecord = { ...record, enqueued: true, updatedAt: Date.now() };
    await this.state.storage.put(RECORD_KEY, next);
    return { record: next };
  }

  // M1 review (Bug B). Cloudflare Queues deliver at-least-once, so the same job can be
  // handed to a second consumer while the first is still rendering (visibility-timeout
  // expiry during a multi-minute render, or the owner evicted mid-consume — the very
  // eviction the Queue exists to survive). The terminal-sticky guard alone does NOT
  // stop this: an in-flight job is DISPATCHING (non-terminal), so without a lease a
  // redelivery would fire a SECOND GPU render (doubled spend / Denial-of-Wallet).
  // acquireDispatch is the ATOMIC arbiter (single-threaded DO): a FRESH lease ⇒ a live
  // delivery owns the render ⇒ 'busy' (defer). An EXPIRED lease ⇒ the owner died ⇒ take
  // over. Terminal ⇒ 'terminal' (never re-dispatch). Lease is internal (not projected).
  private async acquireDispatch(input: {
    now: number;
    leaseMs: number;
  }): Promise<{ outcome: 'acquired' | 'busy' | 'terminal'; record: JobRecord }> {
    const record = await this.state.storage.get<JobRecord>(RECORD_KEY);
    if (!record) throw new Error('acquireDispatch on a job that was never created');
    if (isTerminal(record.status)) return { outcome: 'terminal', record };

    const inFlight = record.status === 'DISPATCHING' || record.status === 'PROCESSING';
    const leaseFresh =
      typeof record.leaseExpiresAt === 'number' && (record.leaseExpiresAt as number) > input.now;
    if (inFlight && leaseFresh) return { outcome: 'busy', record };

    // Acquire: a genuinely new (QUEUED/RETRYING) job, or takeover of an in-flight job
    // whose owner's lease has expired. Written directly (not via transition()) so a
    // takeover that keeps status DISPATCHING is not blocked by the state-machine edges.
    const next: JobRecord = {
      ...record,
      status: 'DISPATCHING',
      leaseExpiresAt: input.now + input.leaseMs,
      updatedAt: input.now,
    };
    await this.state.storage.put(RECORD_KEY, next);
    await this.projectToKv(next);
    return { outcome: 'acquired', record: next };
  }

  private async get(): Promise<JobRecord | null> {
    return (await this.state.storage.get<JobRecord>(RECORD_KEY)) ?? null;
  }

  // Runtime entry point. The gateway holds a stub and calls stub.fetch(); tests
  // call a JobCoordinator instance's .fetch() directly. Path selects the op.
  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    const op = url.pathname.replace(/^\//, '');
    const json = (body: unknown, status = 200) =>
      new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

    try {
      if (op === 'create') {
        const input = (await request.json()) as { deviceId: string; jobId: string; etaSeconds?: number };
        return json(await this.create(input));
      }
      if (op === 'transition') {
        const { to, meta } = (await request.json()) as { to: JobStatus; meta?: Record<string, unknown> };
        return json(await this.transition(to, meta));
      }
      if (op === 'markEnqueued') {
        return json(await this.markEnqueued());
      }
      if (op === 'acquireDispatch') {
        const { now, leaseMs } = (await request.json()) as { now: number; leaseMs: number };
        return json(await this.acquireDispatch({ now, leaseMs }));
      }
      if (op === 'get') {
        return json({ record: await this.get() });
      }
      return json({ error: 'unknown op' }, 404);
    } catch (e: any) {
      // Illegal-transition (and any storage) errors surface as 500 to the caller.
      return json({ error: String(e?.message ?? e) }, 500);
    }
  }
}

// ---- Stub-side helpers: used identically by the gateway (real DO stub) and by
// tests (a JobCoordinator instance). Anything with a `.fetch(Request)` works. ----

interface Fetcher {
  fetch(request: Request): Promise<Response>;
}

async function callOp<T>(stub: Fetcher, op: string, body: unknown): Promise<T> {
  const res = await stub.fetch(
    new Request(`https://coordinator/${op}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body ?? {}),
    }),
  );
  const data = (await res.json()) as any;
  if (!res.ok) throw new Error(data?.error ?? `coordinator ${op} failed (${res.status})`);
  return data as T;
}

export function coordinatorCreate(
  stub: Fetcher,
  input: { deviceId: string; jobId: string; etaSeconds?: number },
): Promise<{ record: JobRecord; idempotent: boolean }> {
  return callOp(stub, 'create', input);
}

export function coordinatorTransition(
  stub: Fetcher,
  to: JobStatus,
  meta?: Record<string, unknown>,
): Promise<{ applied: boolean; record: JobRecord; reason?: string }> {
  return callOp(stub, 'transition', { to, meta });
}

export async function coordinatorGet(stub: Fetcher): Promise<JobRecord | null> {
  const { record } = await callOp<{ record: JobRecord | null }>(stub, 'get', {});
  return record;
}

// M1 review (Bug A): confirm the durable hand-off. Called only AFTER a successful
// JOB_DISPATCH_QUEUE.send, so a still-QUEUED record without this marker is an orphan
// the idempotency peek must re-enqueue.
export function coordinatorMarkEnqueued(stub: Fetcher): Promise<{ record: JobRecord }> {
  return callOp(stub, 'markEnqueued', {});
}

// M1 review (Bug B): atomically claim the dispatch of a job (or learn it is already
// owned/terminal). `now`/`leaseMs` are passed in so callers (runtime + tests) control
// the clock. leaseMs must cover the worst-case render window (see handleJobQueue).
export function coordinatorAcquireDispatch(
  stub: Fetcher,
  input: { now: number; leaseMs: number },
): Promise<{ outcome: 'acquired' | 'busy' | 'terminal'; record: JobRecord }> {
  return callOp(stub, 'acquireDispatch', input);
}
