/**
 * Keystore Zero-Trust của client.
 *
 * Khóa RIÊNG (private) được giữ dưới dạng CryptoKey non-extractable trong IndexedDB:
 * trình duyệt/Tauri lưu được object CryptoKey qua structured-clone, và vì nó
 * non-extractable nên KHÔNG script nào (kể cả app) đọc lại được chất liệu khóa —
 * an toàn hơn hẳn việc nhét JWK riêng vào localStorage (đọc/copy được).
 *
 * Khóa CÔNG KHAI (public) không bí mật nên lưu JWK ở localStorage để gửi Gateway.
 */

const DB_NAME = 'omnivoice-keystore';
const STORE = 'keys';
const PRIVATE_KEY_ID = 'device-private-key';

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function idbGet<T>(key: string): Promise<T | null> {
  return openDb().then(
    (db) =>
      new Promise<T | null>((resolve, reject) => {
        const tx = db.transaction(STORE, 'readonly');
        const req = tx.objectStore(STORE).get(key);
        req.onsuccess = () => resolve((req.result as T) ?? null);
        req.onerror = () => reject(req.error);
      }),
  );
}

function idbPut(key: string, value: unknown): Promise<void> {
  return openDb().then(
    (db) =>
      new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE, 'readwrite');
        tx.objectStore(STORE).put(value, key);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      }),
  );
}

/** Lưu CryptoKey riêng (non-extractable) vào IndexedDB. */
export async function savePrivateKey(key: CryptoKey): Promise<void> {
  await idbPut(PRIVATE_KEY_ID, key);
}

/** Đọc CryptoKey riêng (non-extractable) từ IndexedDB, hoặc null nếu chưa có. */
export async function loadPrivateKey(): Promise<CryptoKey | null> {
  return idbGet<CryptoKey>(PRIVATE_KEY_ID);
}
