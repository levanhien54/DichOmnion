import { describe, it, expect, vi, beforeEach } from 'vitest';

// The Rust audio-read IPC is mocked so the test exercises only the transport's
// HTTP orchestration (mint → PUT → return GET), not Tauri.
vi.mock('@tauri-apps/api/tauri', () => ({
  invoke: vi.fn(async () => btoa('FAKE-AUDIO-BYTES')),
}));

import { uploadAudioForWorker } from './transport';

const MINT = {
  key: 'audio/dev/job-1.wav',
  uploadUrl:
    'https://acc.r2.cloudflarestorage.com/bucket/audio/dev/job-1.wav?X-Amz-Signature=put',
  getUrl:
    'https://acc.r2.cloudflarestorage.com/bucket/audio/dev/job-1.wav?X-Amz-Signature=get',
  expiresSeconds: 7200,
};

describe('uploadAudioForWorker (per-job R2 URL minted by the Gateway)', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('mints a per-job URL, uploads the audio, and returns the GET url', async () => {
    const calls: Array<{ url: string; init: any }> = [];
    const fetchMock = vi.fn(async (url: string, init: any) => {
      calls.push({ url, init });
      if (url.endsWith('/api/uploads/presign')) {
        return new Response(JSON.stringify(MINT), { status: 200 });
      }
      return new Response(null, { status: 200 }); // the R2 PUT
    });
    vi.stubGlobal('fetch', fetchMock);
    const signPayload = vi.fn(async (s: string) => `SIG(${s})`);

    const getUrl = await uploadAudioForWorker({
      audioPath: '/tmp/a.wav',
      gatewayUrl: 'https://gw.example',
      deviceId: 'dev',
      jobId: 'job-1',
      signPayload,
    });

    // Returns the worker-facing GET url from the mint.
    expect(getUrl).toBe(MINT.getUrl);

    // 1st call: signed mint request to the Gateway, carrying THIS job's id.
    const presign = calls[0];
    expect(presign.url).toBe('https://gw.example/api/uploads/presign');
    expect(presign.init.method).toBe('POST');
    expect(presign.init.headers['X-Device-Id']).toBe('dev');
    expect(presign.init.headers['X-ECDSA-Signature']).toBeTruthy();
    const signedBody = JSON.parse(presign.init.body);
    expect(signedBody.jobId).toBe('job-1');
    expect(typeof signedBody.timestamp).toBe('number');
    // The EXACT bytes sent are what got signed (no signing of a different body).
    expect(signPayload).toHaveBeenCalledWith(presign.init.body);

    // 2nd call: PUT the audio bytes to the mint's uploadUrl (not a static env url).
    const put = calls[1];
    expect(put.url).toBe(MINT.uploadUrl);
    expect(put.init.method).toBe('PUT');
  });

  it('throws a clear error when the Gateway refuses to mint', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) =>
        url.endsWith('/api/uploads/presign')
          ? new Response(JSON.stringify({ error: 'Upload storage not provisioned' }), {
              status: 503,
            })
          : new Response(null, { status: 200 }),
      ),
    );

    await expect(
      uploadAudioForWorker({
        audioPath: '/tmp/a.wav',
        gatewayUrl: 'https://gw',
        deviceId: 'dev',
        jobId: 'job-1',
        signPayload: async () => 'sig',
      }),
    ).rejects.toThrow();
  });

  it('never uploads audio when minting fails (no PUT is attempted)', async () => {
    const fetchMock = vi.fn(async (url: string) =>
      url.endsWith('/api/uploads/presign')
        ? new Response(JSON.stringify({ error: 'nope' }), { status: 503 })
        : new Response(null, { status: 200 }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await expect(
      uploadAudioForWorker({
        audioPath: '/tmp/a.wav',
        gatewayUrl: 'https://gw',
        deviceId: 'dev',
        jobId: 'job-1',
        signPayload: async () => 'sig',
      }),
    ).rejects.toThrow();

    // Only the mint request was made — audio bytes never left the machine.
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
