/**
 * Utility: Chuyển đổi Uint8Array sang Base64
 */
function bufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

/**
 * Utility: Chuyển đổi Base64 sang Uint8Array
 */
function base64ToBuffer(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

/**
 * [OPTIMIZED] Sinh cặp khóa ECDSA (Public / Private Key) sử dụng Web Crypto API.
 * Đảm bảo tương thích với Trình duyệt, Tauri và Cloudflare Workers.
 */
export async function generateECDSAKeyPair(): Promise<{ publicKeyJwk: JsonWebKey; privateKeyJwk: JsonWebKey }> {
  // eslint-disable-next-line no-undef
  const cryptoAPI = globalThis.crypto;
  const keyPair = await cryptoAPI.subtle.generateKey(
    { name: 'ECDSA', namedCurve: 'P-256' },
    true,
    ['sign', 'verify']
  );
  
  const publicKeyJwk = await cryptoAPI.subtle.exportKey('jwk', keyPair.publicKey);
  const privateKeyJwk = await cryptoAPI.subtle.exportKey('jwk', keyPair.privateKey);
  
  return { publicKeyJwk, privateKeyJwk };
}

/**
 * [OPTIMIZED] Ký một chuỗi Payload bằng Web Crypto API.
 */
export async function signPayload(payload: string, privateKeyJwk: JsonWebKey): Promise<string> {
  const cryptoAPI = globalThis.crypto;
  const privateKey = await cryptoAPI.subtle.importKey(
    'jwk',
    privateKeyJwk,
    { name: 'ECDSA', namedCurve: 'P-256' },
    true,
    ['sign']
  );
  
  const encoder = new TextEncoder();
  const data = encoder.encode(payload);
  
  const signatureBuffer = await cryptoAPI.subtle.sign(
    { name: 'ECDSA', hash: { name: 'SHA-256' } },
    privateKey,
    data
  );
  
  return bufferToBase64(signatureBuffer);
}

/**
 * Nhập khóa RIÊNG (private) từ JWK thành CryptoKey để KÝ.
 *
 * `extractable=false` (mặc định) tạo ra khóa KHÔNG THỂ export lại: sau khi nhập,
 * không script nào (kể cả chính app) đọc lại được chất liệu khóa. Client sinh khóa,
 * nhập bản riêng ở dạng non-extractable, lưu CryptoKey đó vào IndexedDB rồi vứt bỏ
 * JWK riêng — thay cho việc để nguyên JWK riêng trong localStorage (đọc được).
 */
export async function importPrivateSigningKey(
  privateKeyJwk: JsonWebKey,
  extractable = false,
): Promise<CryptoKey> {
  const cryptoAPI = globalThis.crypto;
  return cryptoAPI.subtle.importKey(
    'jwk',
    privateKeyJwk,
    { name: 'ECDSA', namedCurve: 'P-256' },
    extractable,
    ['sign'],
  );
}

/**
 * Ký payload bằng một CryptoKey RIÊNG có sẵn (thường là khóa non-extractable lấy
 * từ IndexedDB). Không cần — và không thể — biết chất liệu khóa, nên đây là đường
 * ký an toàn cho client Zero-Trust.
 */
export async function signPayloadWithKey(payload: string, privateKey: CryptoKey): Promise<string> {
  const cryptoAPI = globalThis.crypto;
  const data = new TextEncoder().encode(payload);
  const signatureBuffer = await cryptoAPI.subtle.sign(
    { name: 'ECDSA', hash: { name: 'SHA-256' } },
    privateKey,
    data,
  );
  return bufferToBase64(signatureBuffer);
}

/**
 * [OPTIMIZED] Xác thực chữ ký bằng Public Key (Web Crypto API).
 */
export async function verifySignature(payload: string, signatureBase64: string, publicKeyJwk: JsonWebKey): Promise<boolean> {
  try {
    const cryptoAPI = globalThis.crypto;
    const publicKey = await cryptoAPI.subtle.importKey(
      'jwk',
      publicKeyJwk,
      { name: 'ECDSA', namedCurve: 'P-256' },
      true,
      ['verify']
    );

    const encoder = new TextEncoder();
    const data = encoder.encode(payload);
    const signatureBytes = base64ToBuffer(signatureBase64);

    return await cryptoAPI.subtle.verify(
      { name: 'ECDSA', hash: { name: 'SHA-256' } },
      publicKey,
      signatureBytes as BufferSource,
      data
    );
  } catch {
    // Fail-closed: đầu vào dị dạng (JWK hỏng, chữ ký không phải base64, độ dài sai)
    // khiến importKey/atob/verify NÉM. Coi mọi trường hợp đó là chữ ký KHÔNG hợp lệ
    // và trả false, để Gateway trả 403 "Tampering Detected" sạch sẽ thay vì để lỗi
    // rò lên thành 500 (che giấu ý đồ giả mạo sau một lỗi máy chủ).
    return false;
  }
}
