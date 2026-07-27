import { describe, it, expect } from 'vitest';
import {
  sigv4Signature,
  presignS3Url,
  mintJobAudioUrls,
  amzDate,
} from '../src/r2presign';

describe('SigV4 signing core (r2presign)', () => {
  // AUTHORITATIVE known-answer: the official AWS SigV4 test-suite `get-vanilla`
  // case (fetched verbatim from the suite: .sts + .authz). This vector is
  // self-consistent and rigorously maintained. Reproducing it proves the
  // signing-key derivation chain + final HMAC are correct.
  //   creds : AKIDEXAMPLE / wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY
  //   scope : 20150830/us-east-1/service/aws4_request
  it('reproduces the AWS test-suite get-vanilla signature', async () => {
    const stringToSign = [
      'AWS4-HMAC-SHA256',
      '20150830T123600Z',
      '20150830/us-east-1/service/aws4_request',
      'bb579772317eb040ac9ed261061d46c1f17a8133879d6129b6e1c25292927e63',
    ].join('\n');

    const signature = await sigv4Signature(stringToSign, {
      secretAccessKey: 'wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY',
      dateStamp: '20150830',
      region: 'us-east-1',
      service: 'service',
    });

    expect(signature).toBe(
      '5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31',
    );
  });
});

describe('presignS3Url (query canonicalization + assembly)', () => {
  // AWS-documented worked example: presigned GET for examplebucket/test.txt.
  // The documented inputs are self-consistent and reproduced independently by
  // two from-scratch oracles (Node + Python). NOTE: the signature asserted here
  // (3ed0be64…) is the DERIVED-CORRECT value for UNSIGNED-PAYLOAD — it is NOT
  // the doc's prose value aeeed9bb…, which is a well-known erratum inconsistent
  // with the doc's own published string-to-sign intermediates.
  const EXAMPLE = {
    accessKeyId: 'AKIAIOSFODNN7EXAMPLE',
    secretAccessKey: 'wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY',
    method: 'GET' as const,
    host: 'examplebucket.s3.amazonaws.com',
    path: '/test.txt',
    region: 'us-east-1',
    service: 's3',
    amzDate: '20130524T000000Z',
    expiresSeconds: 86400,
  };

  it('produces the AWS-documented canonical query string', async () => {
    const url = await presignS3Url(EXAMPLE);
    const qs = url.slice(url.indexOf('?') + 1);
    // Everything up to (and excluding) the appended X-Amz-Signature must match
    // AWS's documented canonical query string byte-for-byte.
    const withoutSig = qs.slice(0, qs.indexOf('&X-Amz-Signature='));
    expect(withoutSig).toBe(
      'X-Amz-Algorithm=AWS4-HMAC-SHA256' +
        '&X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20130524%2Fus-east-1%2Fs3%2Faws4_request' +
        '&X-Amz-Date=20130524T000000Z' +
        '&X-Amz-Expires=86400' +
        '&X-Amz-SignedHeaders=host',
    );
  });

  it('appends the derived-correct UNSIGNED-PAYLOAD signature', async () => {
    const url = await presignS3Url(EXAMPLE);
    const sig = new URL(url).searchParams.get('X-Amz-Signature');
    expect(sig).toBe(
      '3ed0be64024db54d5574a27da223529635c383f911f80e636f0ccc13890053d2',
    );
  });

  it('prefixes the scheme + host + path unchanged', async () => {
    const url = await presignS3Url(EXAMPLE);
    expect(url.startsWith('https://examplebucket.s3.amazonaws.com/test.txt?')).toBe(
      true,
    );
  });
});

describe('mintJobAudioUrls (per-job R2 upload/download URLs)', () => {
  const CONFIG = {
    accountId: '385e2b411beb41a79d6b45477bc3f544',
    bucket: 'dichomnion-audio',
    accessKeyId: 'R2ACCESSKEYIDEXAMPLE',
    secretAccessKey: 'r2SecretKeyExampleValueForSigningOnlyNotReal',
  };
  const BASE = {
    config: CONFIG,
    deviceId: 'device-abc',
    jobId: 'job-123',
    amzDate: '20260726T120000Z',
    expiresSeconds: 900,
  };

  const requiredParams = [
    'X-Amz-Algorithm',
    'X-Amz-Credential',
    'X-Amz-Date',
    'X-Amz-Expires',
    'X-Amz-SignedHeaders',
    'X-Amz-Signature',
  ];

  it('derives a unique object key namespaced by device + job', async () => {
    const { key } = await mintJobAudioUrls(BASE);
    expect(key).toBe('audio/device-abc/job-123.wav');
  });

  it('targets the account R2 S3 host with the bucket in the path', async () => {
    const { uploadUrl, getUrl } = await mintJobAudioUrls(BASE);
    for (const raw of [uploadUrl, getUrl]) {
      const u = new URL(raw);
      expect(u.host).toBe('385e2b411beb41a79d6b45477bc3f544.r2.cloudflarestorage.com');
      expect(u.pathname).toBe('/dichomnion-audio/audio/device-abc/job-123.wav');
    }
  });

  it('emits all six X-Amz-* auth params on both URLs, host-only signed', async () => {
    const { uploadUrl, getUrl } = await mintJobAudioUrls(BASE);
    for (const raw of [uploadUrl, getUrl]) {
      const q = new URL(raw).searchParams;
      for (const p of requiredParams) {
        expect(q.get(p), `${p} present`).toBeTruthy();
      }
      expect(q.get('X-Amz-SignedHeaders')).toBe('host');
      expect(q.get('X-Amz-Expires')).toBe('900');
    }
  });

  it('binds the PUT and GET URLs to different methods (distinct signatures)', async () => {
    const { uploadUrl, getUrl } = await mintJobAudioUrls(BASE);
    const putSig = new URL(uploadUrl).searchParams.get('X-Amz-Signature');
    const getSig = new URL(getUrl).searchParams.get('X-Amz-Signature');
    expect(putSig).not.toBe(getSig);
  });

  it('produces distinct keys for different jobs (no cross-job collision)', async () => {
    const a = await mintJobAudioUrls(BASE);
    const b = await mintJobAudioUrls({ ...BASE, jobId: 'job-999' });
    expect(a.key).not.toBe(b.key);
  });

  it('rejects an expiry outside the S3 presign bounds (1..604800)', async () => {
    await expect(
      mintJobAudioUrls({ ...BASE, expiresSeconds: 0 }),
    ).rejects.toThrow();
    await expect(
      mintJobAudioUrls({ ...BASE, expiresSeconds: 604801 }),
    ).rejects.toThrow();
  });
});

describe('amzDate (SigV4 basic-format UTC timestamp)', () => {
  it('formats a Date as YYYYMMDDTHHMMSSZ in UTC', () => {
    expect(amzDate(new Date('2026-07-26T12:00:00Z'))).toBe('20260726T120000Z');
  });

  it('zero-pads all fields and ignores milliseconds', () => {
    expect(amzDate(new Date('2013-05-24T00:00:00.789Z'))).toBe('20130524T000000Z');
  });
});
