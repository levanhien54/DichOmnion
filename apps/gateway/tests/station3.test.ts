import { describe, it, expect, afterEach, vi } from 'vitest';
import * as jose from 'jose';
import app, { dispatchToWorker } from '../src/index';
import { MemoryKV } from './setup';
import type { JobRequest } from '@dichomnion/shared-types';

const ctx = { waitUntil: (_p: Promise<unknown>) => {}, passThroughOnException: () => {} };

function basePayload(overrides: Partial<JobRequest> = {}): JobRequest {
  return {
    jobId: 'JOB-STATION3',
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

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('Trạm 3 — Phát hiện gian lận thời gian (Anti-Fraud timing)', () => {
  it('Kết quả trả về NHANH BẤT THƯỜNG bị coi là giả mạo → REJECTED_FRAUD + cách ly worker', async () => {
    const kv = new MemoryKV();
    const pem = await gatewayPrivateKeyPem();
    const env = {
      KV_CACHE: kv,
      GATEWAY_JWT_PRIVATE_KEY: pem,
      WORKER_URL: 'http://127.0.0.1:8000',
      // default floor (2000ms) stays; an instant response is "impossibly fast".
    };

    const calls: string[] = [];
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      calls.push(url);
      // Both /process and /terminate answer instantly.
      return new Response(JSON.stringify({ job_id: 'JOB-STATION3', result: {} }), { status: 200 });
    }));

    const jobKey = 'job:DEV:JOB-STATION3';
    await dispatchToWorker(env as any, basePayload(), jobKey);

    const record = await kv.get(jobKey, { type: 'json' });
    expect(record.status).toBe('REJECTED_FRAUD');
    // Worker was quarantined and the in-band terminate signal was sent.
    expect(await kv.get('quarantine:http://127.0.0.1:8000')).toBe('anomaly_too_fast');
    expect(calls.some((u) => u.endsWith('/api/worker/terminate'))).toBe(true);
  });

  it('Kết quả trong ngưỡng hợp lý → DONE + lưu kết quả', async () => {
    const kv = new MemoryKV();
    const pem = await gatewayPrivateKeyPem();
    const env = {
      KV_CACHE: kv,
      GATEWAY_JWT_PRIVATE_KEY: pem,
      WORKER_URL: 'http://127.0.0.1:8000',
      MIN_PLAUSIBLE_MS: '5',      // lower the floor so the test stays fast
      MAX_RENDER_MS: '5000',
    };

    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url.endsWith('/api/worker/process')) {
        await new Promise((r) => setTimeout(r, 40)); // 40ms > 5ms floor, < 5s cap
        return new Response(JSON.stringify({ job_id: 'JOB-STATION3', result: { dubbed_audio: '/tmp/x.wav' } }), { status: 200 });
      }
      return new Response('{}', { status: 200 });
    }));

    const jobKey = 'job:DEV:JOB-STATION3';
    await dispatchToWorker(env as any, basePayload(), jobKey);

    const record = await kv.get(jobKey, { type: 'json' });
    expect(record.status).toBe('DONE');
    // Result key is DEVICE-SCOPED (derived from jobKey job:DEV:JOB-STATION3).
    const result = await kv.get('result:DEV:JOB-STATION3', { type: 'json' });
    expect(result.result.dubbed_audio).toBe('/tmp/x.wav');
  });

  it('Lưu kết quả DÙNG ALLOWLIST: translated_segments (kịch bản plaintext) KHÔNG bao giờ ghi vào KV', async () => {
    const kv = new MemoryKV();
    const pem = await gatewayPrivateKeyPem();
    const env = {
      KV_CACHE: kv,
      GATEWAY_JWT_PRIVATE_KEY: pem,
      WORKER_URL: 'http://127.0.0.1:8000',
      MIN_PLAUSIBLE_MS: '5',
      MAX_RENDER_MS: '5000',
    };

    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url.endsWith('/api/worker/process')) {
        await new Promise((r) => setTimeout(r, 40));
        // Worker (giả sử bản lỗi/cũ) lỡ trả kèm kịch bản gốc + bản dịch plaintext.
        return new Response(
          JSON.stringify({
            job_id: 'JOB-STATION3',
            result: {
              dubbed_audio: '/tmp/x.wav',
              distinct_voices: 1,
              translated_segments: [{ original_text: 'TOP SECRET script', translated_text: 'KỊCH BẢN MẬT' }],
            },
          }),
          { status: 200 },
        );
      }
      return new Response('{}', { status: 200 });
    }));

    const jobKey = 'job:DEV:JOB-STATION3';
    await dispatchToWorker(env as any, basePayload(), jobKey);

    const stored = await kv.get('result:DEV:JOB-STATION3', { type: 'json' });
    // Allowlist: metadata an toàn + dubbed_audio được giữ...
    expect(stored.result.dubbed_audio).toBe('/tmp/x.wav');
    expect(stored.result.distinct_voices).toBe(1);
    // ...nhưng kịch bản plaintext TUYỆT ĐỐI không được lưu trữ (dù worker có trả).
    expect(stored.result.translated_segments).toBeUndefined();
    expect(JSON.stringify(stored)).not.toContain('TOP SECRET');
  });

  it('Worker treo quá thời hạn → hủy (abort) → TERMINATED_TIMEOUT + cách ly', async () => {
    const kv = new MemoryKV();
    const pem = await gatewayPrivateKeyPem();
    const env = {
      KV_CACHE: kv,
      GATEWAY_JWT_PRIVATE_KEY: pem,
      WORKER_URL: 'http://127.0.0.1:8000',
      MAX_RENDER_MS: '30',        // abort after 30ms
    };

    vi.stubGlobal('fetch', vi.fn((url: string, init: any) => {
      if (url.endsWith('/api/worker/process')) {
        // Never resolves on its own; rejects only when the gateway aborts.
        return new Promise((_resolve, reject) => {
          init.signal?.addEventListener('abort', () =>
            reject(new DOMException('Aborted', 'AbortError')),
          );
        });
      }
      return Promise.resolve(new Response('{}', { status: 200 }));
    }));

    const jobKey = 'job:DEV:JOB-STATION3';
    await dispatchToWorker(env as any, basePayload(), jobKey);

    const record = await kv.get(jobKey, { type: 'json' });
    expect(record.status).toBe('TERMINATED_TIMEOUT');
    expect(await kv.get('quarantine:http://127.0.0.1:8000')).toBe('timeout');
  });
});

describe('Trạm 3 — Cách ly (quarantine) + JWT terminate luôn tươi (G-01)', () => {
  it('Terminate ký JWT MỚI (không tái dùng token dispatch có thể đã hết hạn) và xác minh được', async () => {
    const kv = new MemoryKV();
    // Giữ public key để XÁC MINH token terminate là chứng thư gateway hợp lệ.
    const { publicKey, privateKey } = await jose.generateKeyPair('ES256', { extractable: true });
    const pem = await jose.exportPKCS8(privateKey);
    const env = {
      KV_CACHE: kv,
      GATEWAY_JWT_PRIVATE_KEY: pem,
      WORKER_URL: 'http://127.0.0.1:8000',
      // sàn mặc định 2000ms; trả về tức thì => "quá nhanh" => đi nhánh terminate.
    };

    const auths: Record<string, string> = {};
    vi.stubGlobal('fetch', vi.fn(async (url: string, init: any) => {
      if (url.endsWith('/api/worker/process')) auths.process = init?.headers?.Authorization || '';
      if (url.endsWith('/api/worker/terminate')) auths.terminate = init?.headers?.Authorization || '';
      return new Response(JSON.stringify({ job_id: 'JOB-STATION3', result: {} }), { status: 200 });
    }));

    await dispatchToWorker(env as any, basePayload(), 'job:DEV:JOB-STATION3');

    expect(auths.process?.startsWith('Bearer ')).toBe(true);
    expect(auths.terminate?.startsWith('Bearer ')).toBe(true);
    // Ký MỚI cho terminate: token khác token dispatch (ECDSA ngẫu nhiên hóa chữ ký,
    // và ở nhánh timeout token dispatch exp 2m sẽ hết hạn trước khi terminate chạy).
    expect(auths.terminate).not.toBe(auths.process);
    // Và phải TỰ xác minh được là chứng thư gateway hợp lệ cho ĐÚNG job này.
    const { payload } = await jose.jwtVerify(auths.terminate.slice('Bearer '.length), publicKey);
    expect(payload.role).toBe('gateway');
    expect(payload.jobId).toBe('JOB-STATION3');
  });

  it('KHÔNG dispatch tới worker đã bị cách ly → FAILED(worker_quarantined), không gọi worker', async () => {
    const kv = new MemoryKV();
    const pem = await gatewayPrivateKeyPem();
    const env = { KV_CACHE: kv, GATEWAY_JWT_PRIVATE_KEY: pem, WORKER_URL: 'http://127.0.0.1:8000' };
    // Một anomaly trước đó đã cách ly URL worker này.
    await kv.put('quarantine:http://127.0.0.1:8000', 'timeout');

    const fetchMock = vi.fn(async () => new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    await dispatchToWorker(env as any, basePayload(), 'job:DEV:JOB-STATION3');

    expect(fetchMock).not.toHaveBeenCalled();  // tuyệt đối không gửi job tới worker xấu
    const record = await kv.get('job:DEV:JOB-STATION3', { type: 'json' });
    expect(record.status).toBe('FAILED');
    expect(record.reason).toBe('worker_quarantined');
  });
});

describe('Trạm 3 — Tự động thử lại khi worker chết tạm thời (auto-retry / re-queue)', () => {
  function retryEnv(pem: string, kv: MemoryKV) {
    return {
      KV_CACHE: kv,
      GATEWAY_JWT_PRIVATE_KEY: pem,
      WORKER_URL: 'http://127.0.0.1:8000',
      MIN_PLAUSIBLE_MS: '5',   // hạ sàn để test nhanh
      MAX_RENDER_MS: '5000',
    };
  }

  it('5xx thoáng qua: thử lại và thành công ở lần thứ 3 → DONE (attempts=3)', async () => {
    const kv = new MemoryKV();
    const env = retryEnv(await gatewayPrivateKeyPem(), kv);

    let processCalls = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url.endsWith('/api/worker/process')) {
        processCalls += 1;
        if (processCalls < 3) return new Response('{"detail":"overloaded"}', { status: 503 });
        await new Promise((r) => setTimeout(r, 40)); // 40ms > sàn 5ms => không bị coi là giả
        return new Response(
          JSON.stringify({ job_id: 'JOB-STATION3', result: { dubbed_audio: '/tmp/x.wav' } }),
          { status: 200 },
        );
      }
      return new Response('{}', { status: 200 });
    }));

    await dispatchToWorker(env as any, basePayload(), 'job:DEV:JOB-STATION3');

    expect(processCalls).toBe(3);
    const record = await kv.get('job:DEV:JOB-STATION3', { type: 'json' });
    expect(record.status).toBe('DONE');
    expect(record.attempts).toBe(3);
    const result = await kv.get('result:DEV:JOB-STATION3', { type: 'json' });
    expect(result.result.dubbed_audio).toBe('/tmp/x.wav');
  });

  it('Ký JWT MỚI cho MỖI lần dispatch (không tái dùng token có thể hết hạn giữa các lần thử)', async () => {
    const kv = new MemoryKV();
    // Giữ public key để tự xác minh mỗi token gửi đi là chứng thư gateway hợp lệ.
    const { publicKey, privateKey } = await jose.generateKeyPair('ES256', { extractable: true });
    const pem = await jose.exportPKCS8(privateKey);
    const env = retryEnv(pem, kv);

    const processAuths: string[] = [];
    let processCalls = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string, init: any) => {
      if (url.endsWith('/api/worker/process')) {
        processCalls += 1;
        processAuths.push(init?.headers?.Authorization || '');
        if (processCalls < 2) return new Response('{"detail":"overloaded"}', { status: 503 }); // 5xx -> retry
        await new Promise((r) => setTimeout(r, 40));
        return new Response(
          JSON.stringify({ job_id: 'JOB-STATION3', result: { dubbed_audio: '/tmp/x.wav' } }),
          { status: 200 },
        );
      }
      return new Response('{}', { status: 200 });
    }));

    await dispatchToWorker(env as any, basePayload(), 'job:DEV:JOB-STATION3');

    expect(processCalls).toBe(2);
    // Mỗi lần thử ký MỚI: hai token KHÁC nhau (không tái dùng token dispatch cũ có thể đã
    // qua exp 2m). ECDSA ngẫu nhiên hóa chữ ký nên hai lần ký cho ra chuỗi khác nhau.
    expect(processAuths[0]).not.toBe(processAuths[1]);
    // Và cả hai đều tự xác minh được là chứng thư gateway hợp lệ cho ĐÚNG job.
    for (const a of processAuths) {
      expect(a.startsWith('Bearer ')).toBe(true);
      const { payload } = await jose.jwtVerify(a.slice('Bearer '.length), publicKey);
      expect(payload.role).toBe('gateway');
      expect(payload.jobId).toBe('JOB-STATION3');
    }
  });

  it('Lỗi mạng thoáng qua (không phải abort): thử lại và thành công → DONE', async () => {
    const kv = new MemoryKV();
    const env = retryEnv(await gatewayPrivateKeyPem(), kv);

    let processCalls = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url.endsWith('/api/worker/process')) {
        processCalls += 1;
        if (processCalls < 2) throw new TypeError('network down'); // fetch reject, KHÔNG abort
        await new Promise((r) => setTimeout(r, 40));
        return new Response(JSON.stringify({ job_id: 'JOB-STATION3', result: {} }), { status: 200 });
      }
      return new Response('{}', { status: 200 });
    }));

    await dispatchToWorker(env as any, basePayload(), 'job:DEV:JOB-STATION3');

    expect(processCalls).toBe(2);
    expect((await kv.get('job:DEV:JOB-STATION3', { type: 'json' })).status).toBe('DONE');
  });

  it('4xx là lỗi CỨNG (client/xác thực): KHÔNG thử lại → FAILED ngay', async () => {
    const kv = new MemoryKV();
    const env = retryEnv(await gatewayPrivateKeyPem(), kv);

    let processCalls = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url.endsWith('/api/worker/process')) {
        processCalls += 1;
        return new Response('{"detail":"bad request"}', { status: 400 });
      }
      return new Response('{}', { status: 200 });
    }));

    await dispatchToWorker(env as any, basePayload(), 'job:DEV:JOB-STATION3');

    expect(processCalls).toBe(1); // 4xx không được thử lại
    const record = await kv.get('job:DEV:JOB-STATION3', { type: 'json' });
    expect(record.status).toBe('FAILED');
    expect(record.code).toBe(400);
  });

  it('5xx dai dẳng: cạn lượt thử → FAILED (đã thử đủ MAX_DISPATCH_ATTEMPTS lần)', async () => {
    const kv = new MemoryKV();
    const env = retryEnv(await gatewayPrivateKeyPem(), kv);

    let processCalls = 0;
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url.endsWith('/api/worker/process')) {
        processCalls += 1;
        return new Response('{"detail":"still down"}', { status: 502 });
      }
      return new Response('{}', { status: 200 });
    }));

    await dispatchToWorker(env as any, basePayload(), 'job:DEV:JOB-STATION3');

    expect(processCalls).toBe(3); // thử hết 3 lượt rồi bỏ cuộc
    const record = await kv.get('job:DEV:JOB-STATION3', { type: 'json' });
    expect(record.status).toBe('FAILED');
    expect(record.attempts).toBe(3);
  });
});

describe('Kết quả BẤT ĐỒNG BỘ — client lấy được output (device-scoped, không lộ temp path)', () => {
  it('Poll DONE trả metadata TRUNG THỰC + downloadUrl, KHÔNG lộ dubbed_audio temp path', async () => {
    const kv = new MemoryKV();
    // Seed a finished job + its DEVICE-SCOPED result exactly as dispatch stores them.
    await kv.put('job:DEV:JOB-1', JSON.stringify({ status: 'DONE', elapsed: 4200, attempts: 1 }));
    await kv.put(
      'result:DEV:JOB-1',
      JSON.stringify({
        job_id: 'JOB-1',
        result: {
          status: 'success',
          distinct_voices: 2,
          watermarked: true,
          separated: true,
          pipeline: ['asr', 'translate', 'tts', 'mix'],
          dubbed_audio: '/tmp/secret-abc.wav',
        },
      }),
    );

    const res = await app.request('/api/jobs/JOB-1', { headers: { 'X-Device-Id': 'DEV' } }, { KV_CACHE: kv }, ctx);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.status).toBe('DONE');
    expect(body.result.distinct_voices).toBe(2);
    expect(body.result.watermarked).toBe(true);
    expect(body.result.artifactReady).toBe(true);
    expect(body.result.downloadUrl).toBe('/api/jobs/JOB-1/download');
    // Zero-Logging/privacy: the worker's internal temp path MUST NOT reach the client.
    expect(body.result.dubbed_audio).toBeUndefined();
    expect(JSON.stringify(body)).not.toContain('secret-abc');
  });

  it('Poll DÙNG ALLOWLIST: dữ liệu nhạy cảm còn sót trong KV (translated_segments) KHÔNG lọt ra client', async () => {
    const kv = new MemoryKV();
    await kv.put('job:DEV:JOB-1', JSON.stringify({ status: 'DONE', elapsed: 4200, attempts: 1 }));
    // Mô phỏng bản ghi CŨ còn kịch bản plaintext nằm sẵn trong KV (trước khi vá worker).
    await kv.put(
      'result:DEV:JOB-1',
      JSON.stringify({
        job_id: 'JOB-1',
        result: {
          distinct_voices: 1,
          dubbed_audio: '/tmp/secret-abc.wav',
          translated_segments: [{ original_text: 'LEAKED source line', translated_text: 'DÒNG BỊ LỘ' }],
        },
      }),
    );

    const res = await app.request('/api/jobs/JOB-1', { headers: { 'X-Device-Id': 'DEV' } }, { KV_CACHE: kv }, ctx);
    expect(res.status).toBe(200);
    const body = await res.json();
    // Allowlist chặn: dù translated_segments nằm sẵn trong storage, poll KHÔNG trả nó.
    expect(body.result.translated_segments).toBeUndefined();
    expect(JSON.stringify(body)).not.toContain('LEAKED source line');
    expect(body.result.dubbed_audio).toBeUndefined();
  });

  it('Poll DEVICE-SCOPED: thiết bị khác KHÔNG đọc được job của người khác → 404', async () => {
    const kv = new MemoryKV();
    await kv.put('job:DEV:JOB-1', JSON.stringify({ status: 'DONE' }));
    const res = await app.request('/api/jobs/JOB-1', { headers: { 'X-Device-Id': 'ATTACKER' } }, { KV_CACHE: kv }, ctx);
    expect(res.status).toBe(404);
  });

  it('Poll DONE nhưng result:key CHƯA lan tới edge (KV skew across PoP) → trả FINALIZING (non-terminal) để client poll tiếp, KHÔNG chốt DONE cụt', async () => {
    const kv = new MemoryKV();
    // Trạng thái DONE đã thấy ở edge này, nhưng result key chưa propagate tới (eventual
    // consistency giữa các PoP). Dispatch ghi result TRƯỚC DONE nên đây chỉ là skew đọc.
    await kv.put('job:DEV:JOB-1', JSON.stringify({ status: 'DONE', elapsed: 4200, attempts: 1 }));
    // Cố ý KHÔNG ghi result:DEV:JOB-1.

    const res = await app.request('/api/jobs/JOB-1', { headers: { 'X-Device-Id': 'DEV' } }, { KV_CACHE: kv }, ctx);
    expect(res.status).toBe(200);
    const body = await res.json();
    // KHÔNG được chốt DONE: client sẽ latch terminal + không có artifact = ngõ cụt.
    expect(body.status).toBe('FINALIZING');
    expect(body.status).not.toBe('DONE');
    // Không có kết quả để trả trong cửa sổ propagate này.
    expect(body.result).toBeUndefined();
  });

  it('Download proxy: DONE → ký JWT gateway → proxy worker → stream audio về client', async () => {
    const kv = new MemoryKV();
    const pem = await gatewayPrivateKeyPem();
    const env = { KV_CACHE: kv, GATEWAY_JWT_PRIVATE_KEY: pem, WORKER_URL: 'http://127.0.0.1:8000' };
    await kv.put('job:DEV:JOB-1', JSON.stringify({ status: 'DONE' }));
    await kv.put('result:DEV:JOB-1', JSON.stringify({ result: { dubbed_audio: '/tmp/out.wav' } }));

    let calledUrl = '';
    let auth = '';
    vi.stubGlobal('fetch', vi.fn(async (url: string, init: any) => {
      calledUrl = url;
      auth = init?.headers?.Authorization || '';
      return new Response('RIFFfakeaudio', { status: 200, headers: { 'Content-Type': 'audio/wav' } });
    }));

    const res = await app.request('/api/jobs/JOB-1/download', { headers: { 'X-Device-Id': 'DEV' } }, env, ctx);
    expect(res.status).toBe(200);
    expect(res.headers.get('Content-Type')).toBe('audio/wav');
    expect(await res.text()).toBe('RIFFfakeaudio');
    // Proxied to the worker's JWT-protected endpoint, carrying the temp path + Bearer token.
    expect(calledUrl).toContain('/api/worker/download');
    expect(calledUrl).toContain(encodeURIComponent('/tmp/out.wav'));
    expect(auth.startsWith('Bearer ')).toBe(true);
  });

  it('Download khi job CHƯA xong → 409 (chưa có gì để tải)', async () => {
    const kv = new MemoryKV();
    await kv.put('job:DEV:JOB-1', JSON.stringify({ status: 'QUEUED' }));
    const res = await app.request('/api/jobs/JOB-1/download', { headers: { 'X-Device-Id': 'DEV' } }, { KV_CACHE: kv }, ctx);
    expect(res.status).toBe(409);
  });
});

describe('Trạm 4 — Chống bot (Turnstile) + tiết lưu đăng ký', () => {
  it('Turnstile fail-closed: có cấu hình secret nhưng thiếu token → 403', async () => {
    const env = { KV_CACHE: new MemoryKV(), TURNSTILE_SECRET: 'server-secret' };
    const res = await app.request(
      '/api/auth/register',
      { method: 'POST', body: JSON.stringify({ publicKeyJwk: { kty: 'EC' } }) },
      env,
      ctx,
    );
    expect(res.status).toBe(403);
    expect((await res.json()).error).toBe('Bot verification failed');
  });

  it('Turnstile: token bị Cloudflare từ chối (success:false) → 403', async () => {
    const env = { KV_CACHE: new MemoryKV(), TURNSTILE_SECRET: 'server-secret' };
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ success: false }), { status: 200 })));
    const res = await app.request(
      '/api/auth/register',
      { method: 'POST', body: JSON.stringify({ publicKeyJwk: { kty: 'EC' }, turnstileToken: 'bad' }) },
      env,
      ctx,
    );
    expect(res.status).toBe(403);
  });

  it('Turnstile: token hợp lệ (success:true) → 201 đăng ký thành công', async () => {
    const env = { KV_CACHE: new MemoryKV(), TURNSTILE_SECRET: 'server-secret' };
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ success: true }), { status: 200 })));
    const res = await app.request(
      '/api/auth/register',
      { method: 'POST', body: JSON.stringify({ publicKeyJwk: { kty: 'EC' }, turnstileToken: 'good' }) },
      env,
      ctx,
    );
    expect(res.status).toBe(201);
    expect((await res.json()).deviceId).toBeTruthy();
  });

  it('Tiết lưu đăng ký: quá 5 lần/IP trong cửa sổ → 429', async () => {
    const env = { KV_CACHE: new MemoryKV() }; // no Turnstile secret in dev
    const headers = { 'Content-Type': 'application/json', 'CF-Connecting-IP': '203.0.113.7' };
    const body = JSON.stringify({ publicKeyJwk: { kty: 'EC' } });

    for (let i = 0; i < 5; i++) {
      const ok = await app.request('/api/auth/register', { method: 'POST', headers, body }, env, ctx);
      expect(ok.status).toBe(201);
    }
    const blocked = await app.request('/api/auth/register', { method: 'POST', headers, body }, env, ctx);
    expect(blocked.status).toBe(429);
  });
});

describe('Kill Switch tài chính (Financial Kill Switch)', () => {
  it('Endpoint admin phải chặn khi thiếu/sai X-Admin-Token → 403', async () => {
    const env = { KV_CACHE: new MemoryKV(), ADMIN_TOKEN: 'admin-secret' };
    const res = await app.request(
      '/api/admin/kill-switch',
      { method: 'POST', body: JSON.stringify({ active: true }) },
      env,
      ctx,
    );
    expect(res.status).toBe(403);
  });

  it('Token admin SAI (có header, khác secret) → 403 và KHÔNG bật switch', async () => {
    // Token cùng độ dài, chỉ khác byte cuối: đi qua nhánh so sánh hằng-thời-gian
    // (băm cả hai -> digest khác -> false). Khác với test thiếu-header ở trên vốn
    // dừng ở nhánh `!token`. Chứng minh token sai không kích hoạt được Kill Switch.
    const env = { KV_CACHE: new MemoryKV(), ADMIN_TOKEN: 'admin-secret' };
    const res = await app.request(
      '/api/admin/kill-switch',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Admin-Token': 'admin-secreX' },
        body: JSON.stringify({ active: true }),
      },
      env,
      ctx,
    );
    expect(res.status).toBe(403);
    expect(await env.KV_CACHE.get('system:kill_switch')).toBe(null);
  });

  it('Body HỎNG/thiếu cờ active (dù token đúng) → 400 và KHÔNG vô tình XÓA switch đang bật', async () => {
    // Fail-closed: một request re-arm bị méo body TUYỆT ĐỐI không được ngầm tắt kill switch.
    const env = { KV_CACHE: new MemoryKV(), ADMIN_TOKEN: 'admin-secret' };
    await env.KV_CACHE.put('system:kill_switch', '1'); // switch đang BẬT (đang chặn chi tiêu)

    const res = await app.request(
      '/api/admin/kill-switch',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Admin-Token': 'admin-secret' },
        body: 'khong-phai-json', // body hỏng -> parse ném -> null
      },
      env,
      ctx,
    );
    expect(res.status).toBe(400);
    // Switch vẫn BẬT: garbled body không được disarm safety tài chính.
    expect(await env.KV_CACHE.get('system:kill_switch')).toBe('1');
  });

  it('active KHÔNG phải boolean (vd chuỗi "true") → 400, không đổi trạng thái switch', async () => {
    const env = { KV_CACHE: new MemoryKV(), ADMIN_TOKEN: 'admin-secret' };
    await env.KV_CACHE.put('system:kill_switch', '1');
    const res = await app.request(
      '/api/admin/kill-switch',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Admin-Token': 'admin-secret' },
        body: JSON.stringify({ active: 'true' }), // chuỗi, không phải boolean
      },
      env,
      ctx,
    );
    expect(res.status).toBe(400);
    expect(await env.KV_CACHE.get('system:kill_switch')).toBe('1');
  });

  it('active:false TƯỜNG MINH → CLEARED (gỡ switch đúng chủ đích)', async () => {
    const env = { KV_CACHE: new MemoryKV(), ADMIN_TOKEN: 'admin-secret' };
    await env.KV_CACHE.put('system:kill_switch', '1');
    const res = await app.request(
      '/api/admin/kill-switch',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Admin-Token': 'admin-secret' },
        body: JSON.stringify({ active: false }),
      },
      env,
      ctx,
    );
    expect(res.status).toBe(200);
    expect((await res.json()).killSwitch).toBe('CLEARED');
    expect(await env.KV_CACHE.get('system:kill_switch')).toBe(null);
  });

  it('Khi Kill Switch bật: register và jobs/create đều bị chặn 503', async () => {
    const env = { KV_CACHE: new MemoryKV(), ADMIN_TOKEN: 'admin-secret' };

    const flip = await app.request(
      '/api/admin/kill-switch',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Admin-Token': 'admin-secret' },
        body: JSON.stringify({ active: true }),
      },
      env,
      ctx,
    );
    expect(flip.status).toBe(200);
    expect((await flip.json()).killSwitch).toBe('ACTIVE');

    const reg = await app.request(
      '/api/auth/register',
      { method: 'POST', body: JSON.stringify({ publicKeyJwk: { kty: 'EC' } }) },
      env,
      ctx,
    );
    expect(reg.status).toBe(503);

    const job = await app.request(
      '/api/jobs/create',
      { method: 'POST', headers: { 'X-ECDSA-Signature': 'x', 'X-Device-Id': 'y' }, body: '{}' },
      env,
      ctx,
    );
    expect(job.status).toBe(503);
  });
});

describe('Trạm 2 — dispatch FAIL-CLOSED khi thiếu/hỏng khóa ký gateway', () => {
  it('Thiếu GATEWAY_JWT_PRIVATE_KEY → job FAILED(gateway_key_missing), TUYỆT ĐỐI không gọi worker', async () => {
    const kv = new MemoryKV();
    // Cố ý KHÔNG cấp khóa ký: worker chỉ giữ public key nên nếu gateway không ký được
    // thì KHÔNG có cách nào chứng minh danh tính -> phải từ chối dispatch, không "chạy chui".
    const env = { KV_CACHE: kv, WORKER_URL: 'http://127.0.0.1:8000' };

    const fetchMock = vi.fn(async () => new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    await dispatchToWorker(env as any, basePayload(), 'job:DEV:JOB-STATION3');

    // Không có JWT -> không bao giờ chạm worker (không tốn GPU, không lộ audio_url).
    expect(fetchMock).not.toHaveBeenCalled();
    const record = await kv.get('job:DEV:JOB-STATION3', { type: 'json' });
    expect(record.status).toBe('FAILED');
    expect(record.reason).toBe('gateway_key_missing');
  });

  it('PEM sai định dạng → job FAILED(gateway_key_invalid), không gọi worker', async () => {
    const kv = new MemoryKV();
    // Khóa có mặt nhưng KHÔNG parse được (import ném lỗi) -> signGatewayJwt trả null.
    // Phân biệt với nhánh "thiếu khóa": reason phải là gateway_key_invalid, không phải missing.
    const env = { KV_CACHE: kv, GATEWAY_JWT_PRIVATE_KEY: 'khong-phai-PEM-hop-le', WORKER_URL: 'http://127.0.0.1:8000' };

    const fetchMock = vi.fn(async () => new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    await dispatchToWorker(env as any, basePayload(), 'job:DEV:JOB-STATION3');

    expect(fetchMock).not.toHaveBeenCalled();
    const record = await kv.get('job:DEV:JOB-STATION3', { type: 'json' });
    expect(record.status).toBe('FAILED');
    expect(record.reason).toBe('gateway_key_invalid');
  });
});

describe('Download proxy — nhánh lỗi FAIL-CLOSED (không lộ temp path nội bộ)', () => {
  const SECRET = '/tmp/secret-leak.wav';

  it('DONE nhưng result KHÔNG có dubbed_audio → 404, không lộ đường dẫn', async () => {
    const kv = new MemoryKV();
    await kv.put('job:DEV:JOB-1', JSON.stringify({ status: 'DONE' }));
    // Bản ghi kết quả thiếu artifact (vd worker báo thành công nhưng không đính file).
    await kv.put('result:DEV:JOB-1', JSON.stringify({ result: { distinct_voices: 1 } }));

    const res = await app.request(
      '/api/jobs/JOB-1/download',
      { headers: { 'X-Device-Id': 'DEV' } },
      { KV_CACHE: kv },
      ctx,
    );
    expect(res.status).toBe(404);
    expect((await res.json()).error).toBe('Result artifact unavailable');
  });

  it('DONE + có path nhưng gateway KHÔNG có khóa ký → 503, KHÔNG lộ temp path', async () => {
    const kv = new MemoryKV();
    await kv.put('job:DEV:JOB-1', JSON.stringify({ status: 'DONE' }));
    await kv.put('result:DEV:JOB-1', JSON.stringify({ result: { dubbed_audio: SECRET } }));
    // Không cấp GATEWAY_JWT_PRIVATE_KEY -> signGatewayJwt null -> 503 fail-closed.
    const env = { KV_CACHE: kv, WORKER_URL: 'http://127.0.0.1:8000' };

    const res = await app.request('/api/jobs/JOB-1/download', { headers: { 'X-Device-Id': 'DEV' } }, env, ctx);
    expect(res.status).toBe(503);
    expect((await res.json()).error).toBe('Gateway not provisioned to fetch result');
  });

  it('Worker không kết nối được (fetch ném lỗi) → 502, KHÔNG lộ temp path', async () => {
    const kv = new MemoryKV();
    const pem = await gatewayPrivateKeyPem();
    await kv.put('job:DEV:JOB-1', JSON.stringify({ status: 'DONE' }));
    await kv.put('result:DEV:JOB-1', JSON.stringify({ result: { dubbed_audio: SECRET } }));
    const env = { KV_CACHE: kv, GATEWAY_JWT_PRIVATE_KEY: pem, WORKER_URL: 'http://127.0.0.1:8000' };

    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('connection refused'); }));

    const res = await app.request('/api/jobs/JOB-1/download', { headers: { 'X-Device-Id': 'DEV' } }, env, ctx);
    expect(res.status).toBe(502);
    const body = await res.json();
    expect(body.error).toBe('Worker unreachable');
    // Zero-Logging: đường dẫn temp nội bộ TUYỆT ĐỐI không rò rỉ trong response lỗi.
    expect(JSON.stringify(body)).not.toContain('secret-leak');
  });

  it('Worker trả lỗi (không ok) → 502 kèm mã, KHÔNG lộ temp path', async () => {
    const kv = new MemoryKV();
    const pem = await gatewayPrivateKeyPem();
    await kv.put('job:DEV:JOB-1', JSON.stringify({ status: 'DONE' }));
    await kv.put('result:DEV:JOB-1', JSON.stringify({ result: { dubbed_audio: SECRET } }));
    const env = { KV_CACHE: kv, GATEWAY_JWT_PRIVATE_KEY: pem, WORKER_URL: 'http://127.0.0.1:8000' };

    // Worker từ chối (vd 403 do path không hợp lệ). Gateway KHÔNG chuyển tiếp mã lỗi thô
    // mà quy về 502 + code, không lộ chi tiết nội bộ.
    vi.stubGlobal('fetch', vi.fn(async () => new Response('Access Denied', { status: 403 })));

    const res = await app.request('/api/jobs/JOB-1/download', { headers: { 'X-Device-Id': 'DEV' } }, env, ctx);
    expect(res.status).toBe(502);
    const body = await res.json();
    expect(body.error).toBe('Worker download failed');
    expect(body.code).toBe(403);
    expect(JSON.stringify(body)).not.toContain('secret-leak');
  });
});
