import { describe, it, expect } from 'vitest';
import { MemoryKV } from './setup';
import { FakeDOState } from './do-harness';
import {
  JobCoordinator,
  coordinatorCreate,
  coordinatorTransition,
  coordinatorGet,
  coordinatorMarkEnqueued,
  coordinatorAcquireDispatch,
  TERMINAL_STATES,
} from '../src/coordinator';

// Build a coordinator instance that quacks like a real DO stub (it has `.fetch`),
// backed by a shared MemoryKV so we can assert the write-through projection.
function makeCoordinator() {
  const kv = new MemoryKV();
  const coord = new JobCoordinator(new FakeDOState() as any, { KV_CACHE: kv } as any);
  return { coord, kv };
}

const DEV = 'device-abc';
const JOB = 'job-123';

describe('JobCoordinator.create — atomic idempotency + KV projection', () => {
  it('first create initializes QUEUED, is not idempotent, and seeds the KV projection', async () => {
    const { coord, kv } = makeCoordinator();

    const r = await coordinatorCreate(coord, { deviceId: DEV, jobId: JOB, etaSeconds: 29 });

    expect(r.idempotent).toBe(false);
    expect(r.record.status).toBe('QUEUED');
    expect(r.record.deviceId).toBe(DEV);
    expect(r.record.jobId).toBe(JOB);
    expect(r.record.etaSeconds).toBe(29);

    // Write-through: poll reads KV `job:<dev>:<id>` — its shape must match the
    // record the legacy /create handler wrote: { status, createdAt, etaSeconds }.
    const projected = await kv.get(`job:${DEV}:${JOB}`, { type: 'json' });
    expect(projected.status).toBe('QUEUED');
    expect(projected.etaSeconds).toBe(29);
    expect(typeof projected.createdAt).toBe('number');
  });

  it('a second create with the same (device, job) returns the existing record as idempotent, never re-initializing', async () => {
    const { coord } = makeCoordinator();

    const first = await coordinatorCreate(coord, { deviceId: DEV, jobId: JOB, etaSeconds: 29 });
    // Advance the job past QUEUED so we can prove the second create returns the
    // CURRENT record, not a fresh QUEUED one (TOCTOU double-init would reset it).
    await coordinatorTransition(coord, 'DISPATCHING');

    const second = await coordinatorCreate(coord, { deviceId: DEV, jobId: JOB, etaSeconds: 999 });

    expect(second.idempotent).toBe(true);
    expect(second.record.status).toBe('DISPATCHING'); // preserved, not reset to QUEUED
    expect(second.record.etaSeconds).toBe(29); // original eta, not the second call's 999
    expect(first.record.createdAt).toBe(second.record.createdAt);
  });
});

describe('JobCoordinator.transition — state machine enforcement', () => {
  it('walks the legal dispatch path QUEUED→DISPATCHING→PROCESSING→DONE, each applied, projecting to KV', async () => {
    const { coord, kv } = makeCoordinator();
    await coordinatorCreate(coord, { deviceId: DEV, jobId: JOB, etaSeconds: 20 });

    for (const to of ['DISPATCHING', 'PROCESSING', 'DONE'] as const) {
      const r = await coordinatorTransition(coord, to);
      expect(r.applied).toBe(true);
      expect(r.record.status).toBe(to);
      const projected = await kv.get(`job:${DEV}:${JOB}`, { type: 'json' });
      expect(projected.status).toBe(to);
    }
  });

  it('is terminal-STICKY: a transition out of DONE is a silent no-op, and never regresses the KV projection', async () => {
    const { coord, kv } = makeCoordinator();
    await coordinatorCreate(coord, { deviceId: DEV, jobId: JOB });
    await coordinatorTransition(coord, 'DISPATCHING');
    await coordinatorTransition(coord, 'DONE', { elapsed: 4321 });

    // A redelivered queue message tries to re-drive the job — must NOT resurrect it.
    const r = await coordinatorTransition(coord, 'DISPATCHING');
    expect(r.applied).toBe(false);
    expect(r.reason).toBe('terminal');
    expect(r.record.status).toBe('DONE');

    const projected = await kv.get(`job:${DEV}:${JOB}`, { type: 'json' });
    expect(projected.status).toBe('DONE'); // NOT dragged back to DISPATCHING/QUEUED
  });

  it('rejects an illegal live-state transition (QUEUED→DONE skips the machine) as an error', async () => {
    const { coord } = makeCoordinator();
    await coordinatorCreate(coord, { deviceId: DEV, jobId: JOB });

    await expect(coordinatorTransition(coord, 'DONE')).rejects.toThrow(/illegal transition/i);
  });

  it('treats a transition to the CURRENT state as an idempotent no-op (redelivery safety)', async () => {
    const { coord } = makeCoordinator();
    await coordinatorCreate(coord, { deviceId: DEV, jobId: JOB });
    await coordinatorTransition(coord, 'DISPATCHING');

    const r = await coordinatorTransition(coord, 'DISPATCHING');
    expect(r.applied).toBe(false);
    expect(r.reason).toBe('noop');
    expect(r.record.status).toBe('DISPATCHING');
  });

  it('merges per-transition meta into the record and the KV projection without dropping createdAt/etaSeconds', async () => {
    const { coord, kv } = makeCoordinator();
    await coordinatorCreate(coord, { deviceId: DEV, jobId: JOB, etaSeconds: 26 });
    await coordinatorTransition(coord, 'DISPATCHING');
    await coordinatorTransition(coord, 'DONE', { elapsed: 55000, attempts: 2 });

    const projected = await kv.get(`job:${DEV}:${JOB}`, { type: 'json' });
    expect(projected.status).toBe('DONE');
    expect(projected.elapsed).toBe(55000);
    expect(projected.attempts).toBe(2);
    expect(projected.etaSeconds).toBe(26); // seeded field preserved through transitions
    expect(typeof projected.createdAt).toBe('number');
  });

  it('throws when transitioning a job that was never created', async () => {
    const { coord } = makeCoordinator();
    await expect(coordinatorTransition(coord, 'DISPATCHING')).rejects.toThrow(/never created/i);
  });
});

describe('JobCoordinator.get', () => {
  it('returns null before create and the live record afterward', async () => {
    const { coord } = makeCoordinator();
    expect(await coordinatorGet(coord)).toBeNull();

    await coordinatorCreate(coord, { deviceId: DEV, jobId: JOB });
    const rec = await coordinatorGet(coord);
    expect(rec?.status).toBe('QUEUED');
    expect(rec?.jobId).toBe(JOB);
  });
});

// ── M1 review (Bug A): durable hand-off confirmation marker ──────────────────
// The producer commits the DO record BEFORE it enqueues the dispatch message, and
// those are non-atomic. To tell "created but never enqueued" (an orphan a failed
// send left behind) from "created and enqueued", the DO carries an internal
// `enqueued` marker set ONLY after a confirmed send. It must never leak into the KV
// poll projection (which the client reads).
describe('JobCoordinator.markEnqueued — durable hand-off confirmation (Bug A)', () => {
  it('sets the internal enqueued marker on the DO record WITHOUT projecting it to the KV poll contract', async () => {
    const { coord, kv } = makeCoordinator();
    await coordinatorCreate(coord, { deviceId: DEV, jobId: JOB, etaSeconds: 12 });

    expect((await coordinatorGet(coord))?.enqueued).toBeUndefined();

    await coordinatorMarkEnqueued(coord);

    expect((await coordinatorGet(coord))?.enqueued).toBe(true);
    // The KV projection the poll endpoint reads must stay clean (no internal leak).
    const projected = await kv.get(`job:${DEV}:${JOB}`, { type: 'json' });
    expect(projected.enqueued).toBeUndefined();
    expect(projected.status).toBe('QUEUED');
  });

  it('is idempotent: a second markEnqueued keeps enqueued=true', async () => {
    const { coord } = makeCoordinator();
    await coordinatorCreate(coord, { deviceId: DEV, jobId: JOB });
    await coordinatorMarkEnqueued(coord);
    await coordinatorMarkEnqueued(coord);
    expect((await coordinatorGet(coord))?.enqueued).toBe(true);
  });

  it('throws when marking a job that was never created', async () => {
    const { coord } = makeCoordinator();
    await expect(coordinatorMarkEnqueued(coord)).rejects.toThrow(/never created/i);
  });
});

// ── M1 review (Bug B): dispatch lease ────────────────────────────────────────
// At-least-once redelivery can hand the same job to a second consumer while the
// first is still rendering. acquireDispatch is the ATOMIC guard: a fresh lease means
// a live delivery owns the render (defer, do not double-dispatch); an EXPIRED lease
// means the owner died (take over). Terminal jobs never re-acquire. The lease is
// internal and must not leak into the KV poll projection.
describe('JobCoordinator.acquireDispatch — dispatch lease (Bug B)', () => {
  it('acquires a fresh QUEUED job: DISPATCHING + a lease, projecting DISPATCHING to KV (no lease leak)', async () => {
    const { coord, kv } = makeCoordinator();
    await coordinatorCreate(coord, { deviceId: DEV, jobId: JOB });

    const r = await coordinatorAcquireDispatch(coord, { leaseMs: 10_000, now: 1_000 });
    expect(r.outcome).toBe('acquired');
    expect(r.record.status).toBe('DISPATCHING');

    const projected = await kv.get(`job:${DEV}:${JOB}`, { type: 'json' });
    expect(projected.status).toBe('DISPATCHING');
    expect(projected.leaseExpiresAt).toBeUndefined(); // internal, not in the poll contract
  });

  it('reports BUSY when a fresh lease is already held (a concurrent redelivery must not double-dispatch)', async () => {
    const { coord } = makeCoordinator();
    await coordinatorCreate(coord, { deviceId: DEV, jobId: JOB });
    await coordinatorAcquireDispatch(coord, { leaseMs: 10_000, now: 1_000 }); // lease → 11_000

    const r = await coordinatorAcquireDispatch(coord, { leaseMs: 10_000, now: 5_000 });
    expect(r.outcome).toBe('busy');
    expect(r.record.status).toBe('DISPATCHING');
  });

  it('TAKES OVER when the prior lease has expired (the owning delivery died mid-render)', async () => {
    const { coord } = makeCoordinator();
    await coordinatorCreate(coord, { deviceId: DEV, jobId: JOB });
    await coordinatorAcquireDispatch(coord, { leaseMs: 10_000, now: 1_000 }); // lease → 11_000

    const r = await coordinatorAcquireDispatch(coord, { leaseMs: 10_000, now: 20_000 });
    expect(r.outcome).toBe('acquired'); // stale lease → takeover
    expect(r.record.status).toBe('DISPATCHING');
  });

  it('reports TERMINAL and never re-acquires a job that already finished', async () => {
    const { coord } = makeCoordinator();
    await coordinatorCreate(coord, { deviceId: DEV, jobId: JOB });
    await coordinatorAcquireDispatch(coord, { leaseMs: 10_000, now: 1_000 });
    await coordinatorTransition(coord, 'DONE', { elapsed: 5000 });

    const r = await coordinatorAcquireDispatch(coord, { leaseMs: 10_000, now: 2_000 });
    expect(r.outcome).toBe('terminal');
    expect(r.record.status).toBe('DONE');
  });

  it('throws when acquiring a job that was never created', async () => {
    const { coord } = makeCoordinator();
    await expect(coordinatorAcquireDispatch(coord, { leaseMs: 1, now: 1 })).rejects.toThrow(/never created/i);
  });
});

describe('exported invariants', () => {
  it('TERMINAL_STATES are exactly the sink states of the machine', () => {
    expect([...TERMINAL_STATES].sort()).toEqual(
      ['DONE', 'ERROR', 'FAILED', 'REJECTED_FRAUD', 'TERMINATED_TIMEOUT'].sort(),
    );
  });
});
