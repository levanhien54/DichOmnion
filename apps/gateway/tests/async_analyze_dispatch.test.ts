import { createHash } from 'node:crypto';
import { afterEach, describe, expect, it, vi } from 'vitest';
import * as jose from 'jose';
import {
  ARTIFACT_ALG,
  ARTIFACT_SCHEMA_VERSION,
  FAILURE_REASONS,
  WORKER_ANALYZE_RESPONSE_SCHEMA_VERSION,
  type AnalyzeRequest,
} from '@dichomnion/shared-types';
import {
  artifactObjectKey,
  dispatchAnalyzeToWorker,
  inputAudioKey,
} from '../src/index';
import { MemoryKV, MemoryR2 } from './setup';

const DEVICE_ID = 'ASYNC-DEVICE';
const JOB_ID = 'ASYNC-JOB';
const PUBLIC_KEY =
  'BMh2gSG9lbSvQ4fBqwoQVeGNdUT-ibkZ7dn2oPoHIYc6YVCZVlCxAIAcob6iRe7xGiMq4CEulfO5pjbMgjcgJ_E';

function artifactEnvelope(revision = 1): string {
  return JSON.stringify({
    schema_version: ARTIFACT_SCHEMA_VERSION,
    alg: ARTIFACT_ALG,
    context: {
      alg: ARTIFACT_ALG,
      analyzeJobId: JOB_ID,
      analyzeRevision: revision,
      artifactSchemaVersion: ARTIFACT_SCHEMA_VERSION,
      payloadSchemaVersion: 1,
    },
    ephemeralPublicKey: PUBLIC_KEY,
    iv: 'AAAAAAAAAAAAAAAA',
    ciphertext: 'Y2lwaGVydGV4dC13aXRoLWdjbS10YWc',
  });
}

function analyzePayload(): AnalyzeRequest {
  return {
    jobId: JOB_ID,
    phase: 'ANALYZE',
    videoAudioKey: inputAudioKey(DEVICE_ID, JOB_ID),
    videoAudioMd5: 'd41d8cd98f00b204e9800998ecf8427e',
    config: {
      sourceLanguage: 'Chinese',
      targetLanguage: 'Vietnamese',
      translationStyle: 'Natural',
    },
    encryptionPublicKey: PUBLIC_KEY,
    timestamp: Date.now(),
  } as AnalyzeRequest;
}

async function testEnv(kv: MemoryKV, r2: MemoryR2) {
  const { privateKey } = await jose.generateKeyPair('ES256', { extractable: true });
  return {
    KV_CACHE: kv,
    R2: r2,
    GATEWAY_JWT_PRIVATE_KEY: await jose.exportPKCS8(privateKey),
    WORKER_URL: 'http://127.0.0.1:8000',
    MIN_PLAUSIBLE_MS: '5',
    MAX_RENDER_MS: '5000',
    ASYNC_ANALYZE_ENABLED: 'true',
    ASYNC_ANALYZE_TIMEOUT_MS: '60000',
    ASYNC_ANALYZE_POLL_MS: '500',
    R2_ACCOUNT_ID: 'acct123',
    R2_BUCKET: 'sonsonjh',
    R2_ACCESS_KEY_ID: 'AKIAEXAMPLE',
    R2_SECRET_ACCESS_KEY: 'secretexample0000000000000000000000000000',
  };
}

afterEach(() => vi.unstubAllGlobals());

describe('async RunPod analyze transport', () => {
  it('uses authenticated short submit/status calls and never opens the long sync request', async () => {
    const kv = new MemoryKV();
    const r2 = new MemoryR2();
    await r2.put(inputAudioKey(DEVICE_ID, JOB_ID), 'input');
    const ciphertext = artifactEnvelope();
    const calls: Array<{ url: string; init?: any }> = [];
    let pollCount = 0;

    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: any) => {
      const value = String(url);
      calls.push({ url: value, init });
      if (value.endsWith('/api/worker/analyze/submit')) {
        const request = JSON.parse(String(init?.body ?? '{}'));
        const claims = jose.decodeJwt(String(init?.headers?.Authorization).replace(/^Bearer /, ''));
        expect(claims.act).toBe('analyze');
        expect(claims.jobId).toBe(JOB_ID);
        expect(claims.attempt).toBe(1);
        expect(claims.bodyDigest).toBe(
          createHash('sha256').update(String(init?.body ?? ''), 'utf8').digest('hex'),
        );
        return new Response(JSON.stringify({
          schema_version: WORKER_ANALYZE_RESPONSE_SCHEMA_VERSION,
          job_id: request.job_id,
          attempt: request.attempt,
          status: 'queued',
        }), { status: 202 });
      }
      if (value.includes('/api/worker/analyze/status/')) {
        const claims = jose.decodeJwt(String(init?.headers?.Authorization).replace(/^Bearer /, ''));
        expect(claims.act).toBe('analyze');
        expect(claims.jobId).toBe(JOB_ID);
        expect(claims.attempt).toBe(1);
        expect(claims.bodyDigest).toBe(createHash('sha256').update('').digest('hex'));
        pollCount += 1;
        if (pollCount === 1) {
          return new Response(JSON.stringify({
            job_id: JOB_ID,
            attempt: 1,
            status: 'running',
          }), { status: 200 });
        }
        await r2.put(artifactObjectKey(DEVICE_ID, JOB_ID, 1), ciphertext);
        return new Response(JSON.stringify({
          job_id: JOB_ID,
          attempt: 1,
          status: 'completed',
          response: {
            schema_version: WORKER_ANALYZE_RESPONSE_SCHEMA_VERSION,
            job_id: JOB_ID,
            attempt: 1,
            result: {
              status: 'success',
              artifact_key: artifactObjectKey(DEVICE_ID, JOB_ID, 1),
              artifact_md5: createHash('md5').update(ciphertext).digest('hex'),
              alg: ARTIFACT_ALG,
              diarization: { available: true, mode: 'full', speakerCount: 2 },
              segment_count: 3,
            },
          },
        }), { status: 200 });
      }
      return new Response('{}', { status: 404 });
    }));

    await dispatchAnalyzeToWorker(
      await testEnv(kv, r2) as any,
      analyzePayload(),
      `job:${DEVICE_ID}:${JOB_ID}`,
    );

    expect((await kv.get(`job:${DEVICE_ID}:${JOB_ID}`, { type: 'json' }) as any).status)
      .toBe('AWAITING_REVIEW');
    expect(calls.filter((call) => call.url.endsWith('/api/worker/analyze/submit'))).toHaveLength(1);
    expect(calls.filter((call) => call.url.includes('/api/worker/analyze/status/'))).toHaveLength(2);
    expect(calls.some((call) => call.url.endsWith('/api/worker/analyze'))).toBe(false);
  });

  it('fails closed instead of falling back to the long request when submit is unavailable', async () => {
    const kv = new MemoryKV();
    const r2 = new MemoryR2();
    await r2.put(inputAudioKey(DEVICE_ID, JOB_ID), 'input');
    const calls: string[] = [];
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      calls.push(String(url));
      return new Response('{}', { status: 404 });
    }));

    await dispatchAnalyzeToWorker(
      await testEnv(kv, r2) as any,
      analyzePayload(),
      `job:${DEVICE_ID}:${JOB_ID}`,
    );

    const record = await kv.get(`job:${DEVICE_ID}:${JOB_ID}`, { type: 'json' }) as any;
    expect(record.status).toBe('FAILED');
    expect(record.reason).toBe(FAILURE_REASONS.WORKER_ERROR);
    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatch(/\/api\/worker\/analyze\/submit$/);
  });
});
