import { describe, it, expect } from 'vitest';
import {
  generateECDSAKeyPair,
  signPayload,
  verifySignature,
  importPrivateSigningKey,
  signPayloadWithKey,
} from '../src/ecdsa';
import { deterministicStringify } from '@dichomnion/shared-types';

describe('Cơ chế mã hóa ECDSA (Web Crypto API Edge-Compatible)', () => {
  it('Phải sinh được cặp khóa Public/Private dạng JWK', async () => {
    const { publicKeyJwk, privateKeyJwk } = await generateECDSAKeyPair();
    expect(publicKeyJwk.kty).toBe('EC');
    expect(publicKeyJwk.crv).toBe('P-256');
    expect(privateKeyJwk.d).toBeDefined(); // Private key field
  });

  it('Gateway phải verify được chữ ký của Client (Zero-Trust)', async () => {
    const { publicKeyJwk, privateKeyJwk } = await generateECDSAKeyPair();
    
    // Test cả chức năng Deterministic Stringify
    const mockData = { b: 2, a: 1 };
    const payloadStr = deterministicStringify(mockData);
    
    // Client ký
    const signature = await signPayload(payloadStr, privateKeyJwk);
    expect(signature).toBeTypeOf('string');

    // Gateway xác thực
    const isValid = await verifySignature(payloadStr, signature, publicKeyJwk);
    expect(isValid).toBe(true);
  });

  it('Đầu vào dị dạng bị coi là chữ ký KHÔNG hợp lệ (fail-closed, không ném 500)', async () => {
    const { publicKeyJwk } = await generateECDSAKeyPair();
    const payloadStr = deterministicStringify({ jobId: 'JOB-1' });

    // Chữ ký rác (không phải base64 hợp lệ / sai độ dài) -> false, KHÔNG throw.
    await expect(
      verifySignature(payloadStr, 'this-is-not-a-valid-signature!!', publicKeyJwk),
    ).resolves.toBe(false);

    // JWK công khai hỏng (thiếu tọa độ đường cong) -> false, KHÔNG throw.
    await expect(
      verifySignature(payloadStr, 'AAAA', { kty: 'EC', crv: 'P-256' } as JsonWebKey),
    ).resolves.toBe(false);
  });

  it('Hacker giả mạo Timestamp sẽ bị từ chối (Tampering Detection)', async () => {
    const { publicKeyJwk, privateKeyJwk } = await generateECDSAKeyPair();
    
    const originalPayload = deterministicStringify({ id: '123', time: 1000 });
    const hackedPayload = deterministicStringify({ id: '123', time: 9999 });

    const signature = await signPayload(originalPayload, privateKeyJwk);

    const isValid = await verifySignature(hackedPayload, signature, publicKeyJwk);
    expect(isValid).toBe(false);
  });

  it('Khóa riêng non-extractable vẫn ký được và Gateway verify được', async () => {
    const { publicKeyJwk, privateKeyJwk } = await generateECDSAKeyPair();

    // Client nhập khóa riêng ở dạng KHÔNG THỂ export (mô phỏng lưu IndexedDB).
    const nonExtractableKey = await importPrivateSigningKey(privateKeyJwk, false);
    expect(nonExtractableKey.extractable).toBe(false);

    // Không thể moi lại chất liệu khóa từ CryptoKey non-extractable.
    await expect(
      crypto.subtle.exportKey('jwk', nonExtractableKey),
    ).rejects.toThrow();

    // Nhưng vẫn ký được, và chữ ký verify hợp lệ bằng public key.
    const payloadStr = deterministicStringify({ jobId: 'JOB-1', ts: 1234 });
    const signature = await signPayloadWithKey(payloadStr, nonExtractableKey);
    expect(await verifySignature(payloadStr, signature, publicKeyJwk)).toBe(true);
  });
});

describe('deterministicStringify phản chiếu ngữ nghĩa JSON.stringify (an toàn khi round-trip)', () => {
  it('Bỏ khóa có giá trị undefined giống JSON.stringify (không sinh JSON hỏng)', () => {
    expect(deterministicStringify({ a: undefined, b: 1 })).toBe('{"b":1}');
    expect(deterministicStringify({ a: undefined })).toBe('{}');
    // Không còn dấu vết "undefined" (bug cũ sinh {"a":undefined} — JSON không hợp lệ).
    expect(deterministicStringify({ a: undefined, b: 2 })).not.toContain('undefined');
  });

  it('Phần tử mảng undefined/function quy về null (đúng như JSON.stringify)', () => {
    expect(deterministicStringify([undefined, 1])).toBe('[null,1]');
    expect(deterministicStringify([1])).toBe(JSON.stringify([1]));
  });

  it('Chữ ký sống sót qua round-trip JSON khi payload có field optional = undefined', async () => {
    const { publicKeyJwk, privateKeyJwk } = await generateECDSAKeyPair();

    // Client dựng payload với field optional bỏ trống một cách tường minh (undefined).
    const clientPayload = {
      jobId: 'JOB-XYZ',
      videoAudioUrl: 'https://r2/test.wav',
      segments: undefined, // optional — client không đính segments
      timestamp: 1234,
    };

    // Client ký trên chuỗi tất định.
    const signature = await signPayload(deterministicStringify(clientPayload), privateKeyJwk);

    // Gateway CHỈ thấy dữ liệu sau khi đi qua dây (JSON.stringify -> parse), lúc này
    // khóa `segments: undefined` đã bị JSON loại bỏ. Gateway dựng lại chuỗi tất định
    // rồi verify — phải KHỚP, vì cả hai phía cùng bỏ khóa undefined.
    const overWire = JSON.parse(JSON.stringify(clientPayload));
    const gatewayStr = deterministicStringify(overWire);

    expect(await verifySignature(gatewayStr, signature, publicKeyJwk)).toBe(true);
  });
});
