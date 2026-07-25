import { describe, it, expect } from 'vitest';
import app from '../src/index';
import { generateECDSAKeyPair } from '@dichomnion/crypto-utils';
import { deterministicStringify } from '@dichomnion/shared-types';

describe('Bài Kiểm tra Khả năng Chịu tải (Stress Test / Crypto-DoS)', () => {
  it('Phải xử lý được 5.000 Request giả mạo trong vòng vài giây mà không bị sập', async () => {
    // 1. Khởi tạo khóa của Hacker
    const hackerKeys = await generateECDSAKeyPair();
    
    // 2. Payload rác
    const fakePayload = deterministicStringify({ spam: true, time: Date.now() });
    
    // Hacker dùng Device ID ngẫu nhiên và một chữ ký bậy bạ
    const fakeSignature = 'base64FakeSignatureStringThatIsLongEnoughToLookReal';
    const fakeDeviceId = 'HACKER-DEVICE-ID-9999';

    const TOTAL_REQUESTS = 5000;
    const CONCURRENCY = 500; // Bắn 500 req/lượt

    const startTime = performance.now();
    let rejectedCount = 0;

    // Chạy dội bom 5.000 requests
    for (let i = 0; i < TOTAL_REQUESTS; i += CONCURRENCY) {
      const promises = [];
      for (let j = 0; j < CONCURRENCY; j++) {
        promises.push(
          app.request('/api/jobs/create', {
            method: 'POST',
            headers: {
              'X-ECDSA-Signature': fakeSignature,
              'X-Device-Id': fakeDeviceId
            },
            body: fakePayload
          })
        );
      }
      
      const responses = await Promise.all(promises);
      for (const res of responses) {
        if (res.status === 403 || res.status === 400 || res.status === 401) {
          rejectedCount++;
        }
      }
    }

    const endTime = performance.now();
    const durationMs = endTime - startTime;
    const rps = (TOTAL_REQUESTS / (durationMs / 1000)).toFixed(2);
    const latencyPerReq = (durationMs / TOTAL_REQUESTS).toFixed(2);

    console.log(`\n================== BÁO CÁO STRESS TEST ==================`);
    console.log(`Tổng số Request: ${TOTAL_REQUESTS}`);
    console.log(`Số Request chặn thành công (Forbidden): ${rejectedCount}/${TOTAL_REQUESTS}`);
    console.log(`Thời gian hoàn thành: ${durationMs.toFixed(2)} ms`);
    console.log(`Tốc độ xử lý (RPS): ${rps} req/sec`);
    console.log(`Độ trễ trung bình: ${latencyPerReq} ms/req`);
    console.log(`=========================================================\n`);

    // Hệ thống chặn thành công 100% các request giả mạo
    expect(rejectedCount).toBe(TOTAL_REQUESTS);
    
    // Nếu hệ thống mất hơn 10 giây để từ chối 5k request thì nghĩa là quá yếu (bị Crypto-DoS thành công).
    // Kỳ vọng phải xử lý xong trong dưới 10 giây (tương đương 500 RPS trên single-thread)
    expect(durationMs).toBeLessThan(10000); 
  }, 30000); // Timeout 30s
});
