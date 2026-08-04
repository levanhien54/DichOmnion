import { describe, it, expect } from 'vitest';
import { generateECDSAKeyPair, signPayload } from '@dichomnion/crypto-utils';
import { JobRequest, deterministicStringify } from '@dichomnion/shared-types';

// LIVE local proof (M1, ADR 0001) — drives a signed /create against a RUNNING
// `wrangler dev` (real Workerd + a locally-simulated Durable Object + Queue), NOT
// the in-process Hono app. It proves the durable producer path executes inside the
// Workerd DO/Queue runtime: DO.create (atomic) + KV projection + enqueue.
//
// Skipped unless LIVE_DEV_URL is set, so the default `pnpm vitest run` stays green
// when no dev server is up. Run it while `wrangler dev --port 8788` is live:
//   LIVE_DEV_URL=http://127.0.0.1:8788 pnpm vitest run tests/live_dev_smoke.test.ts
const BASE = process.env.LIVE_DEV_URL;

describe.skipIf(!BASE)('LIVE wrangler dev — durable /create on real Workerd', () => {
  it('registers, signs, and creates a job that lands QUEUED in the DO + KV projection, then is visible via poll', async () => {
    const { publicKeyJwk, privateKeyJwk } = await generateECDSAKeyPair();

    const reg = await fetch(`${BASE}/api/auth/register`, {
      method: 'POST',
      body: JSON.stringify({ publicKeyJwk }),
    });
    expect(reg.status).toBe(201);
    const { deviceId } = await reg.json();
    expect(typeof deviceId).toBe('string');

    const jobId = `LIVE-${deviceId.slice(0, 8)}`;
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

    const create = await fetch(`${BASE}/api/jobs/create`, {
      method: 'POST',
      headers: { 'X-ECDSA-Signature': signature, 'X-Device-Id': deviceId },
      body,
    });
    // The durable producer path accepted the job: DO.create seeded QUEUED and the
    // dispatch message was enqueued — all inside the Workerd runtime.
    expect(create.status).toBe(202);
    const created = await create.json();
    expect(created.status).toBe('QUEUED');
    expect(typeof created.etaSeconds).toBe('number');
    expect(created.idempotent).toBeUndefined();

    // Poll: the DO's write-through KV projection is live and readable. The queue
    // consumer may already be advancing it (DISPATCHING/RETRYING/ERROR — the GPU
    // worker at WORKER_URL is intentionally not running locally), so accept any of
    // the machine's states; the point is the projection exists and is device-scoped.
    const poll = await fetch(`${BASE}/api/jobs/${jobId}`, {
      headers: { 'X-Device-Id': deviceId },
    });
    expect(poll.status).toBe(200);
    const polled = await poll.json();
    expect(polled.jobId).toBe(jobId);
    expect(typeof polled.status).toBe('string');
    expect(
      ['QUEUED', 'DISPATCHING', 'PROCESSING', 'RETRYING', 'ERROR', 'FAILED', 'FINALIZING'].includes(
        polled.status,
      ),
    ).toBe(true);

    // Idempotency under real Workerd: re-sending the SAME signed job returns the
    // existing record (atomic DO guard), never a fresh QUEUED, never a 2nd enqueue.
    const again = await fetch(`${BASE}/api/jobs/create`, {
      method: 'POST',
      headers: { 'X-ECDSA-Signature': signature, 'X-Device-Id': deviceId },
      body,
    });
    expect(again.status).toBe(202);
    expect((await again.json()).idempotent).toBe(true);
  });
});
