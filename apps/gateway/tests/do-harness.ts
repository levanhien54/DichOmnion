// Shared Durable Object test harness. Models the single-threaded storage semantics
// a real DO gives us (our tests await every call sequentially, so a plain Map is a
// faithful stand-in) and a DurableObjectNamespace that hands out JobCoordinator
// instances keyed by idFromName — each instance quacks like a stub (it has `.fetch`).
import { JobCoordinator } from '../src/coordinator';
import { MemoryKV } from './setup';

export class FakeDOStorage {
  private m = new Map<string, unknown>();
  async get<T>(key: string): Promise<T | undefined> {
    return this.m.has(key) ? (JSON.parse(JSON.stringify(this.m.get(key))) as T) : undefined;
  }
  async put<T>(key: string, value: T): Promise<void> {
    this.m.set(key, JSON.parse(JSON.stringify(value)));
  }
  async delete(key: string): Promise<boolean> {
    return this.m.delete(key);
  }
}

export class FakeDOState {
  storage = new FakeDOStorage();
}

/** A DurableObjectNamespace stub: idFromName → a stable id; get(id) → a per-name
 *  JobCoordinator backed by the SHARED env (so its KV write-through lands in the
 *  same KV the gateway/tests read). Instances persist per name for the harness's
 *  lifetime, modelling a durable object's identity across requests. */
export function makeCoordinatorNamespace(env: { KV_CACHE: MemoryKV }) {
  const instances = new Map<string, JobCoordinator>();
  return {
    idFromName(name: string) {
      return { name } as unknown as DurableObjectId;
    },
    get(id: DurableObjectId) {
      const name = (id as unknown as { name: string }).name;
      let inst = instances.get(name);
      if (!inst) {
        inst = new JobCoordinator(new FakeDOState() as any, env as any);
        instances.set(name, inst);
      }
      return inst as unknown as DurableObjectStub;
    },
  } as unknown as DurableObjectNamespace;
}
