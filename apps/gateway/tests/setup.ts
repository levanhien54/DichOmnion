import app from '../src/index';

/**
 * Minimal in-memory KVNamespace for tests. Implements the subset of the
 * Cloudflare KV API the gateway uses: get (text + json), put, delete.
 * TTLs are accepted and ignored — tests don't advance the clock.
 */
export class MemoryKV {
  private store = new Map<string, string>();

  async get(key: string, options?: { type?: 'text' | 'json' }): Promise<any> {
    const raw = this.store.has(key) ? this.store.get(key)! : null;
    if (raw === null) return null;
    if (options && options.type === 'json') {
      try { return JSON.parse(raw); } catch { return null; }
    }
    return raw;
  }

  async put(key: string, value: string): Promise<void> {
    this.store.set(key, value);
  }

  async delete(key: string): Promise<void> {
    this.store.delete(key);
  }
}

// Inject a fresh KV + a no-op ExecutionContext into every app.request() call
// that doesn't already supply them. The existing tests call app.request(path, init)
// with no env, so this bridges them without touching the test bodies.
const env = { KV_CACHE: new MemoryKV() };
const ctx = { waitUntil: (_p: Promise<unknown>) => {}, passThroughOnException: () => {} };

const original = app.request.bind(app);
// @ts-expect-error — widening the signature for test injection only
app.request = (input: any, init?: any, Env?: any, executionCtx?: any) =>
  original(input, init, Env ?? env, executionCtx ?? ctx);
