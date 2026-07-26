import { describe, it, expect } from 'vitest';
import app, {
  MAX_SEGMENTS,
  MAX_SEGMENT_TEXT_CHARS,
  MAX_TOTAL_TEXT_CHARS,
  MAX_FREETEXT_CHARS,
  MAX_SEGMENT_META_CHARS,
} from '../src/index';
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
      videoAudioMd5: 'd41d8cd98f00b204e9800998ecf8427e',
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
      videoAudioMd5: 'd41d8cd98f00b204e9800998ecf8427e',
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
      videoAudioMd5: 'd41d8cd98f00b204e9800998ecf8427e',
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
      videoAudioMd5: 'd41d8cd98f00b204e9800998ecf8427e',
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
      videoAudioMd5: 'd41d8cd98f00b204e9800998ecf8427e',
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
      videoAudioMd5: 'd41d8cd98f00b204e9800998ecf8427e',
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

describe('Kiểm tra đầu vào (input validation) — chặn request méo mó trước khi tốn tài nguyên', () => {
  it('register thiếu publicKeyJwk → 400 (không ghi device rỗng vào registry)', async () => {
    // Body hợp lệ JSON nhưng THIẾU trường bắt buộc publicKeyJwk. Chặn TRƯỚC cả Turnstile
    // và throttle: không có khóa thì đăng ký vô nghĩa, không nên tiêu KV write.
    const res = await app.request(
      '/api/auth/register',
      { method: 'POST', body: JSON.stringify({ turnstileToken: 'whatever' }) },
      { KV_CACHE: new MemoryKV() },
    );
    expect(res.status).toBe(400);
    expect((await res.json()).error).toBe('Missing Public Key');
  });

  it('jobs/create body KHÔNG phải JSON hợp lệ → 400 (Invalid JSON), trước cả verify chữ ký', async () => {
    const kv = new MemoryKV();
    const { publicKeyJwk } = await generateECDSAKeyPair();
    // Cần device tồn tại để qua cửa 401; JSON.parse hỏng NẰM TRƯỚC verifySignature nên
    // chữ ký giả cũng không sao — điểm test là thân méo bị chặn 400, không phải 500.
    const reg = await app.request(
      '/api/auth/register',
      { method: 'POST', body: JSON.stringify({ publicKeyJwk }) },
      { KV_CACHE: kv },
    );
    const deviceId = (await reg.json()).deviceId;

    const res = await app.request(
      '/api/jobs/create',
      {
        method: 'POST',
        headers: { 'X-ECDSA-Signature': 'dummy', 'X-Device-Id': deviceId },
        body: 'khong-phai-json{',
      },
      { KV_CACHE: kv },
    );
    expect(res.status).toBe(400);
    expect((await res.json()).error).toBe('Invalid JSON');
  });

  it('jobs/create thiếu jobId (dù chữ ký + timestamp hợp lệ) → 400 (Missing jobId)', async () => {
    const kv = new MemoryKV();
    const { publicKeyJwk, privateKeyJwk } = await generateECDSAKeyPair();
    const reg = await app.request(
      '/api/auth/register',
      { method: 'POST', body: JSON.stringify({ publicKeyJwk }) },
      { KV_CACHE: kv },
    );
    const deviceId = (await reg.json()).deviceId;

    // Payload KÝ THẬT nhưng CỐ Ý bỏ jobId. Chữ ký khớp thân + timestamp mới => qua được
    // verifySignature và replay-check, rơi đúng vào nhánh kiểm jobId ở cuối.
    const payload = {
      videoAudioUrl: 'https://r2.cloudflare.com/test.wav',
      videoAudioMd5: 'd41d8cd98f00b204e9800998ecf8427e',
      config: { targetLanguage: 'Vietnamese', translationStyle: 'Formal' },
      speakerMapping: {},
      timestamp: Date.now(),
    };
    const payloadStr = deterministicStringify(payload as any);
    const signature = await signPayload(payloadStr, privateKeyJwk);

    const res = await app.request(
      '/api/jobs/create',
      { method: 'POST', headers: { 'X-ECDSA-Signature': signature, 'X-Device-Id': deviceId }, body: payloadStr },
      { KV_CACHE: kv },
    );
    expect(res.status).toBe(400);
    expect((await res.json()).error).toBe('Missing jobId');
  });
});

describe('Đợt 17 F3/F4 — bound job input tại BIÊN (chống DoS OOM/treo → quarantine chéo tenant)', () => {
  // Một thiết bị ĐÃ đăng ký nhưng KHÔNG đáng tin (Zero-Trust) KÝ hợp lệ được payload khổng
  // lồ. Worker gộp MỌI segment vào MỘT prompt Qwen rồi tokenize + generate một lần -> OOM/
  // treo tới timeout 15' -> Trạm 3 cách ly URL worker 24h -> DoS mọi tenant. validateJobSize
  // từ chối 400 TRƯỚC idempotency/dispatch, biến "treo cả cụm" thành "một request bị chặn"
  // (không bao giờ chạm worker hay bộ máy quarantine). Đây là lớp BIÊN, song song với cổng
  // pydantic của worker (defense-in-depth).

  // Gieo device thẳng vào KV (bỏ qua register throttle) rồi KÝ payload thật để vượt
  // verifySignature + replay + jobId, rơi đúng vào cổng bound.
  async function seedDeviceAndPost(kv: MemoryKV, payload: JobRequest) {
    const { publicKeyJwk, privateKeyJwk } = await generateECDSAKeyPair();
    const deviceId = 'DEV-BOUND';
    await kv.put(`device:${deviceId}`, JSON.stringify(publicKeyJwk));
    const payloadStr = deterministicStringify(payload);
    const signature = await signPayload(payloadStr, privateKeyJwk);
    const res = await app.request(
      '/api/jobs/create',
      { method: 'POST', headers: { 'X-ECDSA-Signature': signature, 'X-Device-Id': deviceId }, body: payloadStr },
      { KV_CACHE: kv },
    );
    return { res, deviceId };
  }

  const baseSeg = { id: 's', speaker: 'SPEAKER_01', text: 'hi', start: 0, end: 1 };
  function basePayload(overrides: Partial<JobRequest> = {}): JobRequest {
    return {
      jobId: 'JOB-BOUND',
      videoAudioUrl: 'https://r2.cloudflare.com/test.wav',
      videoAudioMd5: 'd41d8cd98f00b204e9800998ecf8427e',
      config: { targetLanguage: 'Vietnamese', translationStyle: 'Formal' },
      speakerMapping: {},
      timestamp: Date.now(),
      ...overrides,
    };
  }

  it('quá SỐ segment (>MAX_SEGMENTS) → 400, KHÔNG tạo job (không dispatch tới worker)', async () => {
    const kv = new MemoryKV();
    const segments = Array.from({ length: MAX_SEGMENTS + 1 }, (_, i) => ({ ...baseSeg, id: `s${i}` }));
    const { res, deviceId } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(400);
    expect((await res.json()).error).toContain('too many segments');
    // Bound chạy TRƯỚC khi ghi record job -> không có job -> không background dispatch.
    expect(await kv.get(`job:${deviceId}:JOB-BOUND`)).toBeNull();
  });

  it('text MỘT segment quá dài (>MAX_SEGMENT_TEXT_CHARS) → 400', async () => {
    const kv = new MemoryKV();
    const segments = [{ ...baseSeg, text: 'a'.repeat(MAX_SEGMENT_TEXT_CHARS + 1) }];
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(400);
    expect((await res.json()).error).toContain('segment text too long');
  });

  it('TỔNG text vượt trần dù mỗi segment vừa phải → 400 (chặn trục thứ 3)', async () => {
    const kv = new MemoryKV();
    const per = MAX_SEGMENT_TEXT_CHARS;
    const n = Math.floor(MAX_TOTAL_TEXT_CHARS / per) + 2;
    // Cô lập trục TỔNG: số lượng phải DƯỚI trần segment để không kích nhầm nhánh đếm.
    expect(n).toBeLessThanOrEqual(MAX_SEGMENTS);
    const segments = Array.from({ length: n }, (_, i) => ({ ...baseSeg, id: `s${i}`, text: 'a'.repeat(per) }));
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(400);
    expect((await res.json()).error).toContain('total segment text too long');
  });

  it('free-text targetLanguage phình (>MAX_FREETEXT_CHARS) → 400 (cũng nhúng vào prompt)', async () => {
    const kv = new MemoryKV();
    const payload = basePayload();
    payload.config = { targetLanguage: 'V'.repeat(MAX_FREETEXT_CHARS + 1), translationStyle: 'Formal' };
    const { res } = await seedDeviceAndPost(kv, payload);
    expect(res.status).toBe(400);
    expect((await res.json()).error).toContain('targetLanguage too long');
  });

  it('đúng trần MAX_SEGMENTS (biên) vẫn 202 — bound là ">" chứ không phải ">=" (không chặn nhầm job thật)', async () => {
    const kv = new MemoryKV();
    const segments = Array.from({ length: MAX_SEGMENTS }, (_, i) => ({ ...baseSeg, id: `s${i}` }));
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(202);
  });

  // --- Đợt 18 F6: id + speaker/speaker_id cũng nhúng vào prompt Qwen -> phải bound tại BIÊN ---
  // Bound Đợt-17 chỉ đo `text`, nên payload đã KÝ với `text` tí hon + id/speaker khổng lồ vẫn
  // lọt cả hai cổng, phình prompt, OOM/treo worker -> quarantine chéo tenant. Mirror ở gateway.

  it('Đợt 18 F6: id MỘT segment quá dài (>MAX_SEGMENT_META_CHARS) → 400', async () => {
    const kv = new MemoryKV();
    const segments = [{ ...baseSeg, id: 'z'.repeat(MAX_SEGMENT_META_CHARS + 1) }];
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(400);
    expect((await res.json()).error).toContain('segment id too long');
  });

  it('Đợt 18 F6: speaker quá dài (>MAX_SEGMENT_META_CHARS) → 400 (nhúng làm speaker_id trong prompt)', async () => {
    const kv = new MemoryKV();
    const segments = [{ ...baseSeg, speaker: 'S'.repeat(MAX_SEGMENT_META_CHARS + 1) }];
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(400);
    expect((await res.json()).error).toContain('segment speaker too long');
  });

  it('Đợt 18 F6: alias speaker_id (khi thiếu speaker) cũng bị bound → 400', async () => {
    const kv = new MemoryKV();
    const segments = [
      { id: 's', text: 'hi', start: 0, end: 1, speaker_id: 'S'.repeat(MAX_SEGMENT_META_CHARS + 1) } as any,
    ];
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(400);
    expect((await res.json()).error).toContain('segment speaker too long');
  });

  it('Đợt 18 F6: id/speaker vừa-mỗi-trường nhưng GỘP vượt TỔNG → 400 (trục count × field)', async () => {
    const kv = new MemoryKV();
    const per = MAX_SEGMENT_META_CHARS;
    // mỗi segment đóng góp id(per) + speaker(per) + text(1) vào total.
    const n = Math.floor(MAX_TOTAL_TEXT_CHARS / (2 * per)) + 2;
    expect(n).toBeLessThanOrEqual(MAX_SEGMENTS);
    const segments = Array.from({ length: n }, () => ({
      id: 'z'.repeat(per), speaker: 'S'.repeat(per), text: 'x', start: 0, end: 1,
    }));
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(400);
    expect((await res.json()).error).toContain('total segment text too long');
  });

  it('Đợt 18 F6: id/speaker ĐÚNG bằng trần (biên) vẫn 202 — ">" chứ không ">="', async () => {
    const kv = new MemoryKV();
    const segments = [{
      id: 'z'.repeat(MAX_SEGMENT_META_CHARS), speaker: 'S'.repeat(MAX_SEGMENT_META_CHARS),
      text: 'hi', start: 0, end: 1,
    }];
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(202);
  });

  // --- Đợt 19 F8: start/end/duration phải HỮU HẠN (chống crash muộn -> 500 -> retry 3×) ---
  // Hợp đồng wire là SỐ giây (shared-types), nhưng một device đã ký có thể nhét CHUỖI mà
  // to_seconds của worker chuẩn hóa ra ±Infinity/NaN ("1e999"/"inf"/"nan"). Bound Đợt-17/18
  // chỉ đo ĐỘ DÀI chuỗi, không đo TÍNH HỢP LỆ số, nên payload lọt cả hai cổng cũ, chạy trọn
  // ASR+Qwen+TTS rồi mới nổ int(inf*1000) tại mix_audio -> 500 -> gateway chạy lại 3 lần.

  it('Đợt 19 F8: start = "1e999" (tràn -> Infinity) → 400', async () => {
    const kv = new MemoryKV();
    const segments = [{ ...baseSeg, start: '1e999' } as any];
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(400);
    expect((await res.json()).error).toContain('segment start must be a finite number');
  });

  it('Đợt 19 F8: start = "inf" → 400 (token vô cực Python float() nhận)', async () => {
    const kv = new MemoryKV();
    const segments = [{ ...baseSeg, start: 'inf' } as any];
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(400);
    expect((await res.json()).error).toContain('segment start must be a finite number');
  });

  it('Đợt 19 F8: end = "nan" → 400', async () => {
    const kv = new MemoryKV();
    const segments = [{ ...baseSeg, end: 'nan' } as any];
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(400);
    expect((await res.json()).error).toContain('segment end must be a finite number');
  });

  it('Đợt 19 F8: duration = "5.0" (chuỗi) → 400 (round() tiêu thụ trực tiếp, phải là số)', async () => {
    const kv = new MemoryKV();
    const segments = [{ ...baseSeg, duration: '5.0' } as any];
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(400);
    expect((await res.json()).error).toContain('segment duration must be a finite number');
  });

  it('Đợt 19 F8: rác không-số ("abc") KHÔNG bị chặn nhầm — worker coi như 0.0, vẫn 202', async () => {
    const kv = new MemoryKV();
    const segments = [{ ...baseSeg, start: 'abc' } as any];
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(202);
  });

  it('Đợt 19 F8: timecode "HH:MM:SS" + số hữu hạn → 202 (không chặn hợp đồng hợp lệ)', async () => {
    const kv = new MemoryKV();
    const segments = [{ id: 's', speaker: 'SPEAKER_01', text: 'hi', start: '00:00:12', end: '00:00:15', duration: 3 } as any];
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(202);
  });

  // --- Đợt 24 CC23-01: HOÀN THIỆN F8 — chặn TRÀN-×1000 (hữu-hạn-ở-giây nhưng ×1000 -> inf) ---
  // F8 chỉ chứng minh HỮU HẠN Ở GIÂY. Nhưng sink là int(to_seconds(start)*1000) và
  // int((end-start)*1000): 1e306 lọt F8 (hữu hạn) nhưng ×1000 = inf -> int(inf) OverflowError
  // tại mix_audio -> 500 -> retry 3×. Số 1e306 SỐNG qua JSON round-trip (JSON.stringify(1e306)
  // = "1e+306", KHÁC Infinity -> "null"), nên là vector thật. timecodeSeconds phản chiếu
  // to_seconds để kiểm ĐÚNG phép nhân/hiệu sink làm — hoàn thiện F8, không phải trục mới.

  it.each([['start', 1e306], ['end', 1e306]])(
    'Đợt 24 CC23-01: %s = 1e306 (số, hữu hạn ở giây, ×1000 -> inf) → 400',
    async (field, val) => {
      const kv = new MemoryKV();
      const segments = [{ ...baseSeg, [field as string]: val } as any];
      const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
      expect(res.status).toBe(400);
      expect((await res.json()).error).toContain('×1000 overflows');
    },
  );

  it('Đợt 24 CC23-01: start = "1e306" (chuỗi float() nhận, hữu hạn) → 400', async () => {
    const kv = new MemoryKV();
    const segments = [{ ...baseSeg, start: '1e306' } as any];
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(400);
    expect((await res.json()).error).toContain('×1000 overflows');
  });

  it('Đợt 24 CC23-01: HIỆU chéo start=-1e305,end=1e305 (mỗi cái ×1000 hữu hạn, HIỆU ×1000 -> inf) → 400', async () => {
    const kv = new MemoryKV();
    const segments = [{ ...baseSeg, start: -1e305, end: 1e305 } as any];
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(400);
    expect((await res.json()).error).toContain('×1000 overflows');
  });

  it('Đợt 24 CC23-01: end = 1e300 (lớn NHƯNG ×1000 = 1e303 hữu hạn) KHÔNG bị chặn nhầm → 202', async () => {
    const kv = new MemoryKV();
    const segments = [{ ...baseSeg, end: 1e300 } as any];
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(202);
  });

  // --- Đợt 20 F9: PHẦN TỬ segment phải là object (chống crash .get muộn -> 500 -> retry 3×) ---
  // Phần tử phi-object (chuỗi/số/null/mảng) không có trường nào để mọi cổng field cũ đo, nên lọt
  // cả gateway lẫn worker; worker translate_segments seg.get(...) ném AttributeError -> 500 -> 3×.

  it.each([['string', 'x'], ['number', 1], ['null', null], ['array', ['n']]])(
    'Đợt 20 F9: phần tử segment kiểu %s → 400',
    async (_label, bad) => {
      const kv = new MemoryKV();
      const { res } = await seedDeviceAndPost(kv, basePayload({ segments: [bad] as any }));
      expect(res.status).toBe(400);
      expect((await res.json()).error).toContain('segment must be an object');
    },
  );

  it('Đợt 20 F9: phần tử rác lẫn giữa các dict hợp lệ vẫn → 400', async () => {
    const kv = new MemoryKV();
    const segments = [{ ...baseSeg, id: 'a' }, 'sneaky', { ...baseSeg, id: 'b' }] as any;
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(400);
    expect((await res.json()).error).toContain('segment must be an object');
  });

  // --- Đợt 20 F10: GIÁ TRỊ speakerMapping phải là chuỗi (chống crash _resolve_voice muộn) ---
  // speakerMapping chuyển verbatim thành voice_map của worker; giá trị tới _resolve_voice ->
  // VOICE_ID_GENDER.get(val) (TypeError nếu val không hashable như list/dict) hoặc
  // voice.startswith(...) (AttributeError nếu val phi-chuỗi). Record<string,string> chỉ là kiểu
  // biên dịch; ép KIỂU giá trị lúc chạy để map ác ý không thể crash-rồi-retry worker.

  it.each([['object', { x: 'y' }], ['array', ['a']], ['number', 42], ['boolean', true], ['null', null]])(
    'Đợt 20 F10: speakerMapping giá trị kiểu %s → 400',
    async (_label, bad) => {
      const kv = new MemoryKV();
      const speakerMapping = { SPEAKER_00: bad } as any;
      const { res } = await seedDeviceAndPost(kv, basePayload({ speakerMapping }));
      expect(res.status).toBe(400);
      expect((await res.json()).error).toContain('speakerMapping values must be strings');
    },
  );

  it('Đợt 20 F10: speakerMapping là mảng (không phải object) → 400', async () => {
    const kv = new MemoryKV();
    const { res } = await seedDeviceAndPost(kv, basePayload({ speakerMapping: ['a'] as any }));
    expect(res.status).toBe(400);
    expect((await res.json()).error).toContain('speakerMapping must be an object');
  });

  it('Đợt 20 F10: speakerMapping toàn chuỗi hợp lệ → 202 (không chặn hợp đồng đúng)', async () => {
    const kv = new MemoryKV();
    const speakerMapping = { SPEAKER_00: 'nam_tram', SPEAKER_01: 'vi-VN-NamMinhNeural' };
    const { res } = await seedDeviceAndPost(kv, basePayload({ speakerMapping }));
    expect(res.status).toBe(202);
  });

  // --- Đợt 21 F11: KIỂU của segment id/speaker (chống crash _merge muộn -> 500 -> retry 3×) ---
  // id/speaker đi VERBATIM vào worker _merge dựng TranslatedSegment (id: int|str, speaker_id: str)
  // NGOÀI khối retry; value sai-kiểu (object/array/boolean...) nổ thành ValidationError không bắt
  // -> 500 -> gateway retry 3×. Cổng F6 cũ chỉ đo độ-dài KHI đã là chuỗi. Ép kiểu pre-dispatch.

  it.each([['object', { x: 'y' }], ['array', ['SPEAKER_00']], ['number', 7], ['boolean', true]])(
    'Đợt 21 F11: speaker kiểu %s → 400',
    async (_label, bad) => {
      const kv = new MemoryKV();
      const segments = [{ ...baseSeg, speaker: bad }] as any;
      const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
      expect(res.status).toBe(400);
      expect((await res.json()).error).toContain('segment speaker must be a string');
    },
  );

  it.each([['object', { x: 'y' }], ['array', ['s1']], ['boolean', true]])(
    'Đợt 21 F11: id kiểu %s → 400',
    async (_label, bad) => {
      const kv = new MemoryKV();
      const segments = [{ ...baseSeg, id: bad }] as any;
      const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
      expect(res.status).toBe(400);
      expect((await res.json()).error).toContain('segment id must be a string or number');
    },
  );

  it('Đợt 21 F11: id + speaker chuỗi hợp lệ → 202 (không chặn hợp đồng đúng)', async () => {
    const kv = new MemoryKV();
    const segments = [{ ...baseSeg, id: 's1', speaker: 'SPEAKER_03' }];
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(202);
  });

  it('Đợt 21 F11: id số → 202 (gateway nới; worker chốt số-thập-phân bằng 422 sạch)', async () => {
    const kv = new MemoryKV();
    const segments = [{ ...baseSeg, id: 42 }] as any;
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(202);
  });

  // --- Đợt 22 F12: NULLABILITY của text/original_text (chống crash _merge muộn -> 500 -> retry 3×) ---
  // `?? ''` cũ chỉ nuốt null/undefined: `text:null` -> fallback original_text -> (vắng) '' -> PASS,
  // nhưng worker consumer translation_service.py:265 `.get("text", .get("original_text",""))` đọc None
  // (key "text" hiện diện) -> _merge dựng TranslatedSegment(original_text=None) [str bắt buộc] NGOÀI
  // retry -> ValidationError -> 500 -> retry 3×. Phản chiếu .get: key hiện diện (kể cả null) KHÔNG
  // fallback; ép typeof string pre-dispatch.

  it.each([['null', null], ['zero', 0], ['false', false]])(
    'Đợt 22 F12: text falsy %s → 400',
    async (_label, bad) => {
      const kv = new MemoryKV();
      const segments = [{ ...baseSeg, text: bad }] as any;
      const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
      expect(res.status).toBe(400);
      expect((await res.json()).error).toContain('segment text must be a string');
    },
  );

  it('Đợt 22 F12: text vắng + original_text:null → 400 (đối xứng .get đọc None)', async () => {
    const kv = new MemoryKV();
    const { text: _t, ...segNoText } = baseSeg as any;
    const segments = [{ ...segNoText, original_text: null }] as any;
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(400);
    expect((await res.json()).error).toContain('segment text must be a string');
  });

  it('Đợt 22 F12: text:null che original_text hợp lệ → 400 (key hiện diện, KHÔNG fallback)', async () => {
    const kv = new MemoryKV();
    const segments = [{ ...baseSeg, text: null, original_text: 'hi' }] as any;
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(400);
    expect((await res.json()).error).toContain('segment text must be a string');
  });

  it('Đợt 22 F12: text="" (chuỗi rỗng hợp lệ) → 202 (không chặn nhầm)', async () => {
    const kv = new MemoryKV();
    const segments = [{ ...baseSeg, text: '' }];
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(202);
  });

  it('Đợt 22 F12: text vắng + original_text chuỗi → 202 (fallback hợp lệ)', async () => {
    const kv = new MemoryKV();
    const { text: _t, ...segNoText } = baseSeg as any;
    const segments = [{ ...segNoText, original_text: 'xin chao' }] as any;
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(202);
  });

  it('Đợt 22 F12: thiếu cả text lẫn original_text → 202 (mặc định "")', async () => {
    const kv = new MemoryKV();
    const { text: _t, ...segNoText } = baseSeg as any;
    const segments = [segNoText] as any;
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(202);
  });

  // --- Đợt 24 F13: LONE SURROGATE (UTF không well-formed) — crash tokenizer muộn -> 500 -> 3× ---
  // "\ud800" (nửa cặp UTF-16 lẻ) lọt mọi cổng typeof/length của validateJobSize (vẫn là string),
  // qua ECDSA verify (deterministicStringify escape thành \udXXX, TextEncoder ra U+FFFD y hệt hai
  // phía), tới worker -> Qwen fast-tokenizer ném UnicodeEncodeError NGOÀI retry -> 500 -> Gateway
  // retry TOÀN BỘ pipeline 3× (re-run Whisper ASR = khuếch đại GPU). Chặn pre-dispatch (400) đối
  // xứng cổng worker (422). Unicode HỢP LỆ (emoji = cặp surrogate đủ, chữ có dấu) PHẢI qua (202).
  const loneSurrogates: [string, string][] = [
    ['high lẻ', '\ud800'],
    ['low lẻ', '\udfff'],
    ['giữa chuỗi', 'hi\udc00there'],
  ];

  it.each(loneSurrogates)('Đợt 24 F13: targetLanguage lone surrogate (%s) → 400', async (_label, bad) => {
    const kv = new MemoryKV();
    const payload = basePayload({ config: { targetLanguage: bad, translationStyle: 'Formal' } as any });
    const { res } = await seedDeviceAndPost(kv, payload);
    expect(res.status).toBe(400);
  });

  it.each(loneSurrogates)('Đợt 24 F13: translationStyle lone surrogate (%s) → 400', async (_label, bad) => {
    const kv = new MemoryKV();
    const payload = basePayload({ config: { targetLanguage: 'Vietnamese', translationStyle: bad } as any });
    const { res } = await seedDeviceAndPost(kv, payload);
    expect(res.status).toBe(400);
  });

  it.each(loneSurrogates)('Đợt 24 F13: sourceLanguage lone surrogate (%s) → 400', async (_label, bad) => {
    const kv = new MemoryKV();
    const payload = basePayload({ config: { targetLanguage: 'Vietnamese', translationStyle: 'Formal', sourceLanguage: bad } as any });
    const { res } = await seedDeviceAndPost(kv, payload);
    expect(res.status).toBe(400);
  });

  it.each(loneSurrogates)('Đợt 24 F13: segment text lone surrogate (%s) → 400', async (_label, bad) => {
    const kv = new MemoryKV();
    const segments = [{ ...baseSeg, text: bad }];
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(400);
  });

  it.each(loneSurrogates)('Đợt 24 F13: segment id lone surrogate (%s) → 400', async (_label, bad) => {
    const kv = new MemoryKV();
    const segments = [{ ...baseSeg, id: bad }];
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(400);
  });

  it.each(loneSurrogates)('Đợt 24 F13: segment speaker lone surrogate (%s) → 400', async (_label, bad) => {
    const kv = new MemoryKV();
    const segments = [{ ...baseSeg, speaker: bad }];
    const { res } = await seedDeviceAndPost(kv, basePayload({ segments }));
    expect(res.status).toBe(400);
  });

  it('Đợt 24 F13: Unicode HỢP LỆ (emoji cặp surrogate đủ + chữ có dấu) → 202 (không false-positive)', async () => {
    const kv = new MemoryKV();
    const segments = [{ id: 'đoạn-1', speaker: 'NGƯỜI_01', text: 'Xin chào 😀 thế giới', start: 0, end: 1 }];
    const payload = basePayload({
      segments,
      config: { targetLanguage: 'Tiếng Việt 🇻🇳', translationStyle: 'Trang trọng' } as any,
    });
    const { res } = await seedDeviceAndPost(kv, payload);
    expect(res.status).toBe(202);
  });

  // --- Đợt 25 AMP-JOBID-SURROGATE-01: LONE SURROGATE trong jobId (crash serializer response worker -> 500 -> 3×) ---
  // jobId là chuỗi client kiểm soát mà F13 BỎ SÓT: kiểm !jobId/typeof (dòng 460-462) KHÔNG chặn
  // lone surrogate, và validateJobSize KHÔNG rà jobId. Nó được ký + forward NGUYÊN VẸN vào cả claim
  // jobId của JWT lẫn body job_id (JobPayload.job_id là `str` trần) -> worker `return {"job_id": ...}`
  // nổ ở json.dumps(...).encode('utf-8') LÚC render (ngoài try/except) = 500 uncaught -> gateway retry
  // 5xx 3× toàn pipeline (crash SAU render = Denial-of-Wallet tối đa). Chặn pre-dispatch (400) đối
  // xứng cổng worker (422). Sink KHÁC F13 (serializer, không phải tokenizer) -> trường/sink MỚI.
  it.each(loneSurrogates)('Đợt 25 AMP-JOBID-SURROGATE-01: jobId lone surrogate (%s) → 400', async (_label, bad) => {
    const kv = new MemoryKV();
    const { res } = await seedDeviceAndPost(kv, basePayload({ jobId: `job-${bad}` }));
    expect(res.status).toBe(400);
    expect((await res.json()).error).toContain('jobId is not well-formed UTF');
  });

  it('Đợt 25 AMP-JOBID-SURROGATE-01: jobId Unicode HỢP LỆ (emoji cặp surrogate đủ) → 202 (không false-positive)', async () => {
    const kv = new MemoryKV();
    const { res } = await seedDeviceAndPost(kv, basePayload({ jobId: 'job-😀-2026' }));
    expect(res.status).toBe(202);
  });
});

describe('Đợt 17 F5 — throttle tạo job theo THIẾT BỊ (chống một device đốt GPU/KV không giới hạn)', () => {
  // Ngay cả khi input đã bị bound (F3/F4), một thiết bị đã đăng ký nhưng KHÔNG đáng tin vẫn
  // có thể tạo VÔ HẠN job thật (mỗi job = một render GPU + KV write) để đốt chi phí (tiêu
  // chí #4). Đăng ký bị throttle theo IP, nhưng số job/thiết bị SAU đăng ký thì trước đây
  // không có trần. Fixed-window per-device chặn điều đó; chỉ đếm job MỚI (idempotent re-send
  // trả về TRƯỚC throttle nên client retry sau blip mạng không tốn quota / không bị khoá).

  async function postJob(kv: MemoryKV, deviceId: string, privateKeyJwk: any, jobId: string, env: any) {
    const payload: JobRequest = {
      jobId,
      videoAudioUrl: 'https://r2.cloudflare.com/test.wav',
      videoAudioMd5: 'd41d8cd98f00b204e9800998ecf8427e',
      config: { targetLanguage: 'Vietnamese', translationStyle: 'Formal' },
      speakerMapping: {},
      timestamp: Date.now(),
    };
    const body = deterministicStringify(payload);
    const signature = await signPayload(body, privateKeyJwk);
    return app.request(
      '/api/jobs/create',
      { method: 'POST', headers: { 'X-ECDSA-Signature': signature, 'X-Device-Id': deviceId }, body },
      env,
    );
  }

  it('vượt trần JOBS_RATE_LIMIT trên cùng device → 429; thiết bị KHÁC (quota riêng) vẫn tạo được', async () => {
    const kv = new MemoryKV();
    const { publicKeyJwk, privateKeyJwk } = await generateECDSAKeyPair();
    const deviceId = 'DEV-RL';
    await kv.put(`device:${deviceId}`, JSON.stringify(publicKeyJwk));
    const env = { KV_CACHE: kv, JOBS_RATE_LIMIT: '2' };

    // 2 job MỚI đầu tiên qua (đúng trần).
    expect((await postJob(kv, deviceId, privateKeyJwk, 'RL-A', env)).status).toBe(202);
    expect((await postJob(kv, deviceId, privateKeyJwk, 'RL-B', env)).status).toBe(202);
    // Job MỚI thứ 3 vượt trần → 429.
    const third = await postJob(kv, deviceId, privateKeyJwk, 'RL-C', env);
    expect(third.status).toBe(429);
    expect((await third.json()).error).toContain('Too Many Jobs');

    // Thiết bị KHÁC có quota riêng (throttle keyed theo deviceId) → vẫn 202.
    const { publicKeyJwk: pk2, privateKeyJwk: sk2 } = await generateECDSAKeyPair();
    const dev2 = 'DEV-RL-2';
    await kv.put(`device:${dev2}`, JSON.stringify(pk2));
    expect((await postJob(kv, dev2, sk2, 'RL-D', env)).status).toBe(202);
  });

  it('idempotent re-send KHÔNG tốn quota (gửi lại job cũ vẫn 202 dù đã chạm trần)', async () => {
    const kv = new MemoryKV();
    const { publicKeyJwk, privateKeyJwk } = await generateECDSAKeyPair();
    const deviceId = 'DEV-RL-IDEM';
    await kv.put(`device:${deviceId}`, JSON.stringify(publicKeyJwk));
    const env = { KV_CACHE: kv, JOBS_RATE_LIMIT: '1' };

    // 1 job MỚI (chạm trần ngay).
    expect((await postJob(kv, deviceId, privateKeyJwk, 'IDEM-A', env)).status).toBe(202);
    // Job MỚI thứ 2 (jobId khác) → 429.
    expect((await postJob(kv, deviceId, privateKeyJwk, 'IDEM-B', env)).status).toBe(429);
    // GỬI LẠI job cũ 'IDEM-A': nhánh idempotency TRƯỚC throttle -> 202 idempotent, không 429.
    const resend = await postJob(kv, deviceId, privateKeyJwk, 'IDEM-A', env);
    expect(resend.status).toBe(202);
    expect((await resend.json()).idempotent).toBe(true);
  });
});
