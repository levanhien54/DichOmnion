import { describe, it, expect } from 'vitest';
import app from '../src/index';
import { MemoryKV } from './setup';
import { generateECDSAKeyPair, signPayload } from '@dichomnion/crypto-utils';
import { JobRequest, deterministicStringify } from '@dichomnion/shared-types';

describe('Kiểm thử Gateway API (Zero-Trust Endpoint)', () => {
  it('Phải từ chối Request nếu không có chữ ký (401 Unauthorized)', async () => {
    const res = await app.request('/api/jobs/create', {
      method: 'POST',
      body: JSON.stringify({ hello: 'world' })
    });
    expect(res.status).toBe(401);
    const data = await res.json();
    expect(data.error).toBe('Missing Zero-Trust Signature or Device ID');
  });

  it('Phải từ chối Request (401) nếu Device ID không tồn tại trong Database', async () => {
    const { privateKeyJwk } = await generateECDSAKeyPair();
    
    const payload: JobRequest = {
      jobId: 'TEST-123',
      videoAudioUrl: 'https://r2.cloudflare.com/test.wav',
      config: { targetLanguage: 'Vietnamese', translationStyle: 'Formal' },
      speakerMapping: { 'SPEAKER_01': 'Voice_Nam' },
      timestamp: Date.now()
    };
    
    const payloadStr = deterministicStringify(payload);
    const signature = await signPayload(payloadStr, privateKeyJwk);

    const res = await app.request('/api/jobs/create', {
      method: 'POST',
      headers: {
        'X-ECDSA-Signature': signature,
        'X-Device-Id': 'FAKE-DEVICE-ID-HACKER'
      },
      body: payloadStr
    });
    
    expect(res.status).toBe(401);
    const data = await res.json();
    expect(data.error).toBe('Unauthorized Device. Public Key not found in Registry.');
  });

  it('Phải chấp nhận Request (202) nếu Device ID hợp lệ và chữ ký khớp', async () => {
    const { publicKeyJwk, privateKeyJwk } = await generateECDSAKeyPair();
    
    // Bước 1: Đăng ký Device
    const regRes = await app.request('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({ publicKeyJwk })
    });
    const regData = await regRes.json();
    const deviceId = regData.deviceId;
    
    // Bước 2: Ký và Gửi
    const payload: JobRequest = {
      jobId: 'TEST-123',
      videoAudioUrl: 'https://r2.cloudflare.com/test.wav',
      config: { targetLanguage: 'Vietnamese', translationStyle: 'Formal' },
      speakerMapping: { 'SPEAKER_01': 'Voice_Nam' },
      timestamp: Date.now()
    };
    
    const payloadStr = deterministicStringify(payload);
    const signature = await signPayload(payloadStr, privateKeyJwk);

    const res = await app.request('/api/jobs/create', {
      method: 'POST',
      headers: {
        'X-ECDSA-Signature': signature,
        'X-Device-Id': deviceId
      },
      body: payloadStr
    });
    
    expect(res.status).toBe(202);
    const data = await res.json();
    expect(data.message).toBe('Job Accepted securely!');
    expect(data.jobId).toBe('TEST-123');
    // ETA phải có mặt trong 202 để client hiển thị thời gian chờ ước lượng.
    expect(typeof data.etaSeconds).toBe('number');
    expect(data.etaSeconds).toBeGreaterThan(0);
  });

  it('202 ETA tăng theo số segment (20s nền + 3s/segment)', async () => {
    const { publicKeyJwk, privateKeyJwk } = await generateECDSAKeyPair();

    const regRes = await app.request('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({ publicKeyJwk }),
    });
    const deviceId = (await regRes.json()).deviceId;

    const payload: JobRequest = {
      jobId: 'TEST-ETA',
      videoAudioUrl: 'https://r2.cloudflare.com/test.wav',
      config: { targetLanguage: 'Vietnamese', translationStyle: 'Formal' },
      speakerMapping: {},
      timestamp: Date.now(),
      // 5 segment hợp lệ theo hợp đồng ClientSegment (start/end là GIÂY). Test này chỉ
      // quan tâm SỐ LƯỢNG segment (ETA = 20s nền + 5*3s), nhưng vẫn dùng đúng kiểu.
      segments: [0, 1, 2, 3, 4].map((i) => ({
        id: `seg-${i}`,
        speaker: 'SPEAKER_01',
        text: `line ${i}`,
        start: i * 2,
        end: i * 2 + 1.5,
      })),
    };

    const payloadStr = deterministicStringify(payload);
    const signature = await signPayload(payloadStr, privateKeyJwk);

    const res = await app.request('/api/jobs/create', {
      method: 'POST',
      headers: { 'X-ECDSA-Signature': signature, 'X-Device-Id': deviceId },
      body: payloadStr,
    });
    expect(res.status).toBe(202);
    // 20s nền + 5 segment * 3s = 35s.
    expect((await res.json()).etaSeconds).toBe(35);
  });

  it('Poll (GET /api/jobs/:id) phản chiếu lại etaSeconds đã lưu trong record', async () => {
    // Kiểm thử độc lập đường poll: gieo sẵn record QUEUED có ETA rồi đọc lại. (Không
    // đi qua create để tránh việc dispatch nền ghi đè record thành FAILED trong test.)
    const kv = new MemoryKV();
    await kv.put('job:DEV-ETA:JOB-ETA', JSON.stringify({ status: 'QUEUED', etaSeconds: 42 }));

    const res = await app.request(
      '/api/jobs/JOB-ETA',
      { method: 'GET', headers: { 'X-Device-Id': 'DEV-ETA' } },
      { KV_CACHE: kv },
    );
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.status).toBe('QUEUED');
    expect(data.etaSeconds).toBe(42);
  });

  it('Phải từ chối (403) nếu Timestamp quá 30 giây (Replay Attack)', async () => {
    const { publicKeyJwk, privateKeyJwk } = await generateECDSAKeyPair();
    
    // Bước 1: Đăng ký Device
    const regRes = await app.request('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({ publicKeyJwk })
    });
    const regData = await regRes.json();
    const deviceId = regData.deviceId;
    
    const payload: JobRequest = {
      jobId: 'TEST-123',
      videoAudioUrl: 'https://r2.cloudflare.com/test.wav',
      config: { targetLanguage: 'Vietnamese', translationStyle: 'Formal' },
      speakerMapping: { 'SPEAKER_01': 'Voice_Nam' },
      timestamp: Date.now() - 31000 // Quá khứ 31 giây
    };
    
    const payloadStr = deterministicStringify(payload);
    const signature = await signPayload(payloadStr, privateKeyJwk);

    const res = await app.request('/api/jobs/create', {
      method: 'POST',
      headers: {
        'X-ECDSA-Signature': signature,
        'X-Device-Id': deviceId
      },
      body: payloadStr
    });
    
    expect(res.status).toBe(403);
    const data = await res.json();
    expect(data.error).toBe('Request Expired. Replay attack prevented.');
  });

  it('Phải từ chối (403) nếu chữ ký KHÔNG khớp thân request (Tampering Detected)', async () => {
    const { publicKeyJwk, privateKeyJwk } = await generateECDSAKeyPair();
    const regRes = await app.request('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({ publicKeyJwk }),
    });
    const deviceId = (await regRes.json()).deviceId;

    // Ký payload GỐC nhưng gửi payload đã BỊ SỬA (kẻ tấn công đổi ngôn ngữ đích sau khi ký).
    const signed: JobRequest = {
      jobId: 'TEST-TAMPER',
      videoAudioUrl: 'https://r2.cloudflare.com/test.wav',
      config: { targetLanguage: 'Vietnamese', translationStyle: 'Formal' },
      speakerMapping: {},
      timestamp: Date.now(),
    };
    const signature = await signPayload(deterministicStringify(signed), privateKeyJwk);
    const tampered = { ...signed, config: { ...signed.config, targetLanguage: 'English' } };

    const res = await app.request('/api/jobs/create', {
      method: 'POST',
      headers: { 'X-ECDSA-Signature': signature, 'X-Device-Id': deviceId },
      body: deterministicStringify(tampered), // thân KHÔNG khớp chữ ký gốc
    });
    expect(res.status).toBe(403);
    expect((await res.json()).error).toBe('Tampering Detected. Signature Invalid.');
  });

  it('Idempotency: gửi lại cùng job → lần 2 trả idempotent:true (không tạo job trùng)', async () => {
    const kv = new MemoryKV();
    const { publicKeyJwk, privateKeyJwk } = await generateECDSAKeyPair();
    // Đăng ký device trên CÙNG kv để 2 request dùng chung registry + job store.
    const reg = await app.request(
      '/api/auth/register',
      { method: 'POST', body: JSON.stringify({ publicKeyJwk }) },
      { KV_CACHE: kv },
    );
    const deviceId = (await reg.json()).deviceId;

    const payload: JobRequest = {
      jobId: 'TEST-IDEM',
      videoAudioUrl: 'https://r2.cloudflare.com/test.wav',
      config: { targetLanguage: 'Vietnamese', translationStyle: 'Formal' },
      speakerMapping: {},
      timestamp: Date.now(),
    };
    const body = deterministicStringify(payload);
    const signature = await signPayload(body, privateKeyJwk);
    const headers = { 'X-ECDSA-Signature': signature, 'X-Device-Id': deviceId };

    const first = await app.request('/api/jobs/create', { method: 'POST', headers, body }, { KV_CACHE: kv });
    expect(first.status).toBe(202);
    expect((await first.json()).idempotent).toBeUndefined();

    // Gửi LẠI y hệt: gateway thấy record cũ -> trả về idempotent, KHÔNG dispatch lần nữa.
    const second = await app.request('/api/jobs/create', { method: 'POST', headers, body }, { KV_CACHE: kv });
    expect(second.status).toBe(202);
    expect((await second.json()).idempotent).toBe(true);
  });
});
