import { describe, it, expect, afterEach, vi } from 'vitest';
import app from '../src/index';
import { MemoryKV } from './setup';
import { makeCoordinatorNamespace } from './do-harness';
import { coordinatorGet } from '../src/coordinator';
import { generateECDSAKeyPair, signPayload } from '@dichomnion/crypto-utils';
import { JobRequest, deterministicStringify } from '@dichomnion/shared-types';

// M1 (ADR 0001) — the DURABLE /api/jobs/create path. When the JOB_COORDINATOR
// Durable Object + JOB_DISPATCH_QUEUE bindings are present (production, wired in
// wrangler.toml), /create must:
//   • create the job ATOMICALLY in the DO (fixes W3 TOCTOU idempotency),
//   • ENQUEUE a dispatch message (durable hand-off; the consumer renders — W1),
//   • NOT dispatch to the worker in-request (no more background/waitUntil — W1),
//   • return 202 QUEUED, and be idempotent on a re-sent jobId (no double enqueue).
// When those bindings are ABSENT, /create degrades to the legacy KV + background
// path — that fallback is covered by the existing gateway.test.ts suite.

// A recording Queue producer stub: captures every enqueued message body so we can
// assert the durable hand-off happened exactly once.
function makeDurableEnv() {
  const kv = new MemoryKV();
  const ns = makeCoordinatorNamespace({ KV_CACHE: kv });
  const sent: Array<{ deviceId: string; jobId: string; payload: JobRequest }> = [];
  const env = {
    KV_CACHE: kv,
    JOB_COORDINATOR: ns,
    JOB_DISPATCH_QUEUE: { send: async (body: any) => { sent.push(body); } },
  };
  return { env, kv, ns, sent };
}

async function registerDevice(env: any) {
  const { publicKeyJwk, privateKeyJwk } = await generateECDSAKeyPair();
  const reg = await app.request(
    '/api/auth/register',
    { method: 'POST', body: JSON.stringify({ publicKeyJwk }) },
    env,
  );
  const { deviceId } = await reg.json();
  return { deviceId, privateKeyJwk };
}

async function createJob(env: any, deviceId: string, privateKeyJwk: any, jobId: string) {
  const payload: JobRequest = {
    jobId,
    videoAudioUrl: 'https://r2.cloudflare.com/test.wav',
    videoAudioMd5: 'd41d8cd98f00b204e9800998ecf8427e',
    config: { targetLanguage: 'Vietnamese', translationStyle: 'Formal' },
    speakerMapping: { SPEAKER_01: 'Voice_Nam' },
    timestamp: Date.now(),
  };
  const body = deterministicStringify(payload);
  const signature = await signPayload(body, privateKeyJwk);
  const res = await app.request(
    '/api/jobs/create',
    { method: 'POST', headers: { 'X-ECDSA-Signature': signature, 'X-Device-Id': deviceId }, body },
    env,
  );
  return { res, payload };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('/api/jobs/create — durable path (DO + Queue)', () => {
  it('creates the job in the DO as QUEUED, enqueues one dispatch message, and does NOT dispatch to the worker in-request', async () => {
    const { env, kv, ns, sent } = makeDurableEnv();
    const { deviceId, privateKeyJwk } = await registerDevice(env);

    // The worker must NOT be touched by /create in durable mode — the consumer renders.
    const fetchMock = vi.fn(async () => new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    const { res } = await createJob(env, deviceId, privateKeyJwk, 'JOB-DURABLE');

    expect(res.status).toBe(202);
    const data = await res.json();
    expect(data.status).toBe('QUEUED');
    expect(typeof data.etaSeconds).toBe('number');
    expect(data.idempotent).toBeUndefined();

    // Durable hand-off: exactly one message, carrying the full payload.
    expect(sent.length).toBe(1);
    expect(sent[0].deviceId).toBe(deviceId);
    expect(sent[0].jobId).toBe('JOB-DURABLE');
    expect(sent[0].payload.jobId).toBe('JOB-DURABLE');

    // DO authority + KV projection both reflect QUEUED.
    const stub = ns.get(ns.idFromName(`${deviceId}:JOB-DURABLE`));
    expect((await coordinatorGet(stub as any))?.status).toBe('QUEUED');
    const kvRec = await kv.get(`job:${deviceId}:JOB-DURABLE`, { type: 'json' });
    expect(kvRec.status).toBe('QUEUED');

    // No in-request worker dispatch (no background/waitUntil in the durable path).
    expect(fetchMock.mock.calls.some(([u]) => String(u).includes('/api/worker/process'))).toBe(false);
  });

  it('is idempotent on a re-sent jobId: the second create returns idempotent and does NOT enqueue again', async () => {
    const { env, sent } = makeDurableEnv();
    const { deviceId, privateKeyJwk } = await registerDevice(env);

    const first = await createJob(env, deviceId, privateKeyJwk, 'JOB-IDEM');
    expect(first.res.status).toBe(202);
    expect((await first.res.json()).idempotent).toBeUndefined();
    expect(sent.length).toBe(1);

    const second = await createJob(env, deviceId, privateKeyJwk, 'JOB-IDEM');
    expect(second.res.status).toBe(202);
    const data = await second.res.json();
    expect(data.idempotent).toBe(true);
    expect(data.jobId).toBe('JOB-IDEM');
    // The atomic DO guard prevents a second durable hand-off (no double GPU dispatch).
    expect(sent.length).toBe(1);
  });

  // ── M1 review (Bug A): orphan heal ─────────────────────────────────────────
  // coordinatorCreate commits the DO record BEFORE the dispatch send, and those two
  // are non-atomic. If the send fails after the commit, the job is an ORPHAN: QUEUED
  // in the DO, but nothing on the queue will ever render it. The producer confirms a
  // successful hand-off with an internal `enqueued` marker; a later peek that finds a
  // QUEUED record WITHOUT that marker must re-enqueue instead of returning a job that
  // will never run.
  it('heals an orphaned job: when the durable hand-off send fails after the DO commit, a re-sent create re-enqueues it', async () => {
    const { env, ns, sent } = makeDurableEnv();
    const { deviceId, privateKeyJwk } = await registerDevice(env);

    // First create: the DO record commits QUEUED, then the dispatch send FAILS.
    let failNextSend = true;
    env.JOB_DISPATCH_QUEUE = {
      send: async (body: any) => {
        if (failNextSend) {
          failNextSend = false;
          throw new Error('queue temporarily unavailable');
        }
        sent.push(body);
      },
    };

    await createJob(env, deviceId, privateKeyJwk, 'JOB-ORPHAN');
    // The send threw → nothing was enqueued…
    expect(sent.length).toBe(0);
    // …but the atomic create already committed the DO record as QUEUED (an orphan).
    const stub = ns.get(ns.idFromName(`${deviceId}:JOB-ORPHAN`));
    const orphan = await coordinatorGet(stub as any);
    expect(orphan?.status).toBe('QUEUED');
    expect(orphan?.enqueued).not.toBe(true); // never confirmed enqueued

    // Client retries the SAME jobId. The peek must detect the orphan and RE-ENQUEUE.
    const second = await createJob(env, deviceId, privateKeyJwk, 'JOB-ORPHAN');
    expect(second.res.status).toBe(202);
    expect((await second.res.json()).idempotent).toBe(true);
    expect(sent.length).toBe(1); // healed: exactly one dispatch now on the queue
    expect(sent[0].jobId).toBe('JOB-ORPHAN');
    // Now confirmed enqueued, so a THIRD create must NOT re-enqueue.
    expect((await coordinatorGet(stub as any))?.enqueued).toBe(true);
    const third = await createJob(env, deviceId, privateKeyJwk, 'JOB-ORPHAN');
    expect(third.res.status).toBe(202);
    expect(sent.length).toBe(1); // still one — no double dispatch
  });
});
