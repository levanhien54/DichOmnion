import { describe, it, expect } from 'vitest';
import app from '../src/index';
import { MemoryKV } from './setup';
import { generateECDSAKeyPair, signPayload } from '@dichomnion/crypto-utils';
import { deterministicStringify } from '@dichomnion/shared-types';

// The Gateway mints a per-job presigned R2 upload/download URL pair for an
// AUTHENTICATED (registered) device. Auth mirrors /api/jobs/create exactly:
// X-Device-Id + X-ECDSA-Signature over deterministicStringify({ jobId, timestamp }).

const R2_ENV = {
  R2_ACCOUNT_ID: '385e2b411beb41a79d6b45477bc3f544',
  R2_BUCKET: 'dichomnion-audio',
  R2_ACCESS_KEY_ID: 'R2ACCESSKEYIDEXAMPLE',
  R2_SECRET_ACCESS_KEY: 'r2SecretKeyExampleValueForSigningOnlyNotReal',
};

async function seed(extraEnv: Record<string, unknown> = {}) {
  const kv = new MemoryKV();
  const { publicKeyJwk, privateKeyJwk } = await generateECDSAKeyPair();
  const deviceId = 'device-under-test';
  await kv.put(`device:${deviceId}`, JSON.stringify(publicKeyJwk));
  const env = { KV_CACHE: kv, ...R2_ENV, ...extraEnv };
  return { env, deviceId, privateKeyJwk, kv };
}

async function mint(
  env: Record<string, unknown>,
  deviceId: string | null,
  privateKeyJwk: JsonWebKey | null,
  body: unknown,
) {
  const bodyStr = deterministicStringify(body as any);
  const headers: Record<string, string> = {};
  if (deviceId) headers['X-Device-Id'] = deviceId;
  if (privateKeyJwk) headers['X-ECDSA-Signature'] = await signPayload(bodyStr, privateKeyJwk);
  return app.request(
    '/api/uploads/presign',
    { method: 'POST', headers, body: bodyStr },
    env,
  );
}

describe('POST /api/uploads/presign (per-job R2 URL minting)', () => {
  it('mints a presigned PUT+GET pair for a valid signed request', async () => {
    const { env, deviceId, privateKeyJwk } = await seed();
    const res = await mint(env, deviceId, privateKeyJwk, {
      jobId: 'job-xyz',
      timestamp: Date.now(),
    });
    expect(res.status).toBe(200);
    const data = (await res.json()) as any;

    expect(data.key).toBe('audio/device-under-test/job-xyz.wav');
    expect(typeof data.expiresSeconds).toBe('number');
    expect(data.expiresSeconds).toBeGreaterThan(0);

    const put = new URL(data.uploadUrl);
    const get = new URL(data.getUrl);
    expect(put.host).toBe('385e2b411beb41a79d6b45477bc3f544.r2.cloudflarestorage.com');
    expect(put.pathname).toBe('/dichomnion-audio/audio/device-under-test/job-xyz.wav');
    expect(put.searchParams.get('X-Amz-Signature')).toBeTruthy();
    expect(get.searchParams.get('X-Amz-Signature')).toBeTruthy();
    // PUT and GET are bound to different HTTP methods → different signatures.
    expect(put.searchParams.get('X-Amz-Signature')).not.toBe(
      get.searchParams.get('X-Amz-Signature'),
    );
  });

  it('rejects a request with no signature/device headers (401)', async () => {
    const { env } = await seed();
    const res = await mint(env, null, null, { jobId: 'j', timestamp: Date.now() });
    expect(res.status).toBe(401);
  });

  it('rejects an unregistered device (401)', async () => {
    const { env, privateKeyJwk } = await seed();
    const res = await mint(env, 'not-registered', privateKeyJwk, {
      jobId: 'j',
      timestamp: Date.now(),
    });
    expect(res.status).toBe(401);
  });

  it('rejects a tampered signature (403)', async () => {
    const { env, deviceId, privateKeyJwk } = await seed();
    // Sign one body, submit a different one → signature will not verify.
    const bodyStr = deterministicStringify({ jobId: 'real', timestamp: Date.now() });
    const signature = await signPayload(bodyStr, privateKeyJwk);
    const res = await app.request(
      '/api/uploads/presign',
      {
        method: 'POST',
        headers: { 'X-Device-Id': deviceId, 'X-ECDSA-Signature': signature },
        body: deterministicStringify({ jobId: 'tampered', timestamp: Date.now() }),
      },
      env,
    );
    expect(res.status).toBe(403);
  });

  it('rejects a replayed/expired timestamp (403)', async () => {
    const { env, deviceId, privateKeyJwk } = await seed();
    const res = await mint(env, deviceId, privateKeyJwk, {
      jobId: 'j',
      timestamp: Date.now() - 10 * 60_000, // 10 min old ≫ ±30s window
    });
    expect(res.status).toBe(403);
  });

  it('rejects a missing jobId (400)', async () => {
    const { env, deviceId, privateKeyJwk } = await seed();
    const res = await mint(env, deviceId, privateKeyJwk, { timestamp: Date.now() });
    expect(res.status).toBe(400);
  });

  it('rejects a lone-surrogate jobId (400)', async () => {
    const { env, deviceId, privateKeyJwk } = await seed();
    const res = await mint(env, deviceId, privateKeyJwk, {
      jobId: 'job-\ud800',
      timestamp: Date.now(),
    });
    expect(res.status).toBe(400);
  });

  // Đợt 32 F-R2-01 — jobId is interpolated into the HIERARCHICAL R2 object key
  // `audio/<deviceId>/<jobId>.wav` (unlike the flat KV key at job creation). A '/'
  // reshapes the key, breaking the documented one-key-per-job namespacing.
  it('rejects a jobId containing a path separator (400)', async () => {
    const { env, deviceId, privateKeyJwk } = await seed();
    const res = await mint(env, deviceId, privateKeyJwk, {
      jobId: 'a/b',
      timestamp: Date.now(),
    });
    expect(res.status).toBe(400);
  });

  // A '/../' dot-segment survives into the SIGNED canonical URI (r2presign keeps '/'
  // literal, '.' unreserved) but the WHATWG URL parser normalises it away on the wire,
  // so R2 answers SignatureDoesNotMatch — a 200 for a dead-on-arrival URL pair
  // (No-Fake-Success). Reject it at the edge instead of minting a fake success.
  it('rejects a jobId containing a dot-segment (400)', async () => {
    const { env, deviceId, privateKeyJwk } = await seed();
    const res = await mint(env, deviceId, privateKeyJwk, {
      jobId: 'a/../b',
      timestamp: Date.now(),
    });
    expect(res.status).toBe(400);
  });

  // The one legit client mints `JOB-${Date.now()}` — a hyphen/underscore/alnum id must
  // still mint 200 (locks the allowlist against over-tightening).
  it('mints for a client-shaped JOB-<epoch> jobId (200)', async () => {
    const { env, deviceId, privateKeyJwk } = await seed();
    const res = await mint(env, deviceId, privateKeyJwk, {
      jobId: 'JOB-1753600000000',
      timestamp: Date.now(),
    });
    expect(res.status).toBe(200);
  });

  it('fails closed with 503 when R2 is not provisioned', async () => {
    // Seed WITHOUT R2 creds.
    const kv = new MemoryKV();
    const { publicKeyJwk, privateKeyJwk } = await generateECDSAKeyPair();
    const deviceId = 'device-under-test';
    await kv.put(`device:${deviceId}`, JSON.stringify(publicKeyJwk));
    const env = { KV_CACHE: kv }; // no R2_* env
    const res = await mint(env, deviceId, privateKeyJwk, {
      jobId: 'j',
      timestamp: Date.now(),
    });
    expect(res.status).toBe(503);
  });

  it('throttles per-device minting (429) once the window limit is hit', async () => {
    const { env, deviceId, privateKeyJwk } = await seed({ PRESIGN_RATE_LIMIT: '1' });
    const first = await mint(env, deviceId, privateKeyJwk, {
      jobId: 'job-a',
      timestamp: Date.now(),
    });
    expect(first.status).toBe(200);
    const second = await mint(env, deviceId, privateKeyJwk, {
      jobId: 'job-b',
      timestamp: Date.now(),
    });
    expect(second.status).toBe(429);
  });

  it('honours the Financial Kill Switch (503)', async () => {
    const { env, deviceId, privateKeyJwk, kv } = await seed();
    await kv.put('system:kill_switch', '1');
    const res = await mint(env, deviceId, privateKeyJwk, {
      jobId: 'j',
      timestamp: Date.now(),
    });
    expect(res.status).toBe(503);
  });
});
