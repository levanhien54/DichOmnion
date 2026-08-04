import { describe, it, expect, afterEach, vi } from 'vitest';
import * as jose from 'jose';
import { handleJobQueue } from '../src/index';
import { coordinatorCreate, coordinatorGet } from '../src/coordinator';
import { MemoryKV } from './setup';
import { makeCoordinatorNamespace } from './do-harness';
import type { JobRequest } from '@dichomnion/shared-types';

const DEV = 'DEV';
const JOB = 'JOB-QUEUE';

function basePayload(overrides: Partial<JobRequest> = {}): JobRequest {
  return {
    jobId: JOB,
    videoAudioUrl: 'https://r2.cloudflare.com/audio.wav',
    videoAudioMd5: 'd41d8cd98f00b204e9800998ecf8427e',
    config: { targetLanguage: 'Vietnamese', translationStyle: 'Formal' },
    speakerMapping: { SPEAKER_01: 'Voice_Nam' },
    timestamp: Date.now(),
    ...overrides,
  } as JobRequest;
}

async function gatewayPrivateKeyPem(): Promise<string> {
  const { privateKey } = await jose.generateKeyPair('ES256', { extractable: true });
  return jose.exportPKCS8(privateKey);
}

// A message/batch stub that records ack()/retry() so we can assert the consumer's
// durability decision (ack = handled, retry = redeliver).
function makeBatch(bodies: Array<{ deviceId: string; jobId: string; payload: JobRequest }>) {
  const acked: string[] = [];
  const retried: string[] = [];
  const messages = bodies.map((body, i) => ({
    id: `msg-${i}`,
    body,
    ack: () => acked.push(body.jobId),
    retry: () => retried.push(body.jobId),
  }));
  return { batch: { messages, queue: 'job-dispatch' } as any, acked, retried };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('handleJobQueue — durable dispatch consumer', () => {
  it('drives a QUEUED job to DONE: dispatches, syncs the DO to terminal, and ACKs the message', async () => {
    const kv = new MemoryKV();
    const pem = await gatewayPrivateKeyPem();
    const ns = makeCoordinatorNamespace({ KV_CACHE: kv });
    const env = {
      KV_CACHE: kv,
      JOB_COORDINATOR: ns,
      GATEWAY_JWT_PRIVATE_KEY: pem,
      WORKER_URL: 'http://127.0.0.1:8000',
      MIN_PLAUSIBLE_MS: '5',
      MAX_RENDER_MS: '5000',
    };

    // Producer step: the job is created in the DO before it is enqueued.
    const stub = ns.get(ns.idFromName(`${DEV}:${JOB}`));
    await coordinatorCreate(stub as any, { deviceId: DEV, jobId: JOB, etaSeconds: 23 });

    const fetchMock = vi.fn(async (url: string) => {
      if (url.endsWith('/api/worker/process')) {
        await new Promise((r) => setTimeout(r, 40)); // plausible render time
        return new Response(
          JSON.stringify({ job_id: JOB, result: { dubbed_audio: '/tmp/out.wav', distinct_voices: 1 } }),
          { status: 200 },
        );
      }
      return new Response('{}', { status: 200 });
    });
    vi.stubGlobal('fetch', fetchMock);

    const { batch, acked, retried } = makeBatch([{ deviceId: DEV, jobId: JOB, payload: basePayload() }]);
    await handleJobQueue(batch, env as any);

    // KV projection reflects DONE (poll contract), DO authority reflects DONE.
    const kvRec = await kv.get(`job:${DEV}:${JOB}`, { type: 'json' });
    expect(kvRec.status).toBe('DONE');
    const doRec = await coordinatorGet(stub as any);
    expect(doRec?.status).toBe('DONE');
    // Result artifact was committed and the worker was actually called.
    expect((await kv.get(`result:${DEV}:${JOB}`, { type: 'json' })).result.dubbed_audio).toBe('/tmp/out.wav');
    expect(fetchMock.mock.calls.some(([u]) => String(u).endsWith('/api/worker/process'))).toBe(true);
    // Handled → ACK, not retry.
    expect(acked).toEqual([JOB]);
    expect(retried).toEqual([]);
  });

  it('is idempotent on redelivery: a job already TERMINAL is ACKed WITHOUT re-dispatching to the worker', async () => {
    const kv = new MemoryKV();
    const pem = await gatewayPrivateKeyPem();
    const ns = makeCoordinatorNamespace({ KV_CACHE: kv });
    const env = {
      KV_CACHE: kv,
      JOB_COORDINATOR: ns,
      GATEWAY_JWT_PRIVATE_KEY: pem,
      WORKER_URL: 'http://127.0.0.1:8000',
    };

    // Pre-drive the DO to a terminal state (as if a prior delivery already finished).
    const stub = ns.get(ns.idFromName(`${DEV}:${JOB}`));
    await coordinatorCreate(stub as any, { deviceId: DEV, jobId: JOB });
    await stub.fetch(
      new Request('https://coordinator/transition', {
        method: 'POST',
        body: JSON.stringify({ to: 'DISPATCHING' }),
      }),
    );
    await stub.fetch(
      new Request('https://coordinator/transition', {
        method: 'POST',
        body: JSON.stringify({ to: 'DONE', meta: { elapsed: 1000 } }),
      }),
    );

    const fetchMock = vi.fn(async () => new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    const { batch, acked, retried } = makeBatch([{ deviceId: DEV, jobId: JOB, payload: basePayload() }]);
    await handleJobQueue(batch, env as any);

    // No re-dispatch: the worker was never called; state stays DONE; message ACKed.
    expect(fetchMock).not.toHaveBeenCalled();
    expect((await coordinatorGet(stub as any))?.status).toBe('DONE');
    expect(acked).toEqual([JOB]);
    expect(retried).toEqual([]);
  });

  it('RETRIES (redeliver) on a catastrophic failure — the DO/infra being unreachable must not silently drop the job', async () => {
    const kv = new MemoryKV();
    // A coordinator namespace whose stub throws models the DO being unreachable.
    const throwingNs = {
      idFromName: (name: string) => ({ name }) as any,
      get: () => ({ fetch: async () => { throw new Error('DO unreachable'); } }) as any,
    } as any;
    const env = {
      KV_CACHE: kv,
      JOB_COORDINATOR: throwingNs,
      WORKER_URL: 'http://127.0.0.1:8000',
    };

    const fetchMock = vi.fn(async () => new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    const { batch, acked, retried } = makeBatch([{ deviceId: DEV, jobId: JOB, payload: basePayload() }]);
    await handleJobQueue(batch, env as any);

    // Not acked (would drop the job); redelivered instead (queue durable retry — the W1 fix).
    expect(retried).toEqual([JOB]);
    expect(acked).toEqual([]);
  });
});
