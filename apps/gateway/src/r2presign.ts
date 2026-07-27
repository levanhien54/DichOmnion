/**
 * SigV4 presigner for Cloudflare R2 (S3-compatible), implemented with WebCrypto
 * only — zero external dependencies, runs unchanged in Workers and Node ≥18.
 *
 * The Gateway uses this to mint short-lived, per-job presigned PUT/GET URLs so
 * the client can upload audio straight to R2 and the GPU worker can fetch it,
 * without the Gateway ever touching the audio bytes (Zero-Logging / Zero-Trust).
 *
 * Correctness is pinned by tests against the official AWS SigV4 test-suite
 * `get-vanilla` vector and the AWS-documented `examplebucket` presigned example.
 */

// Bare `crypto` global: typed by @cloudflare/workers-types in the Worker and
// present as a global in Node ≥20 (used the same way in index.ts).
const cryptoAPI = crypto;
const encoder = new TextEncoder();

function bytesToHex(bytes: Uint8Array): string {
  let hex = '';
  for (const b of bytes) hex += b.toString(16).padStart(2, '0');
  return hex;
}

async function hmac(key: Uint8Array, data: string): Promise<Uint8Array> {
  const cryptoKey = await cryptoAPI.subtle.importKey(
    'raw',
    key,
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  const sig = await cryptoAPI.subtle.sign('HMAC', cryptoKey, encoder.encode(data));
  return new Uint8Array(sig);
}

async function sha256Hex(data: string): Promise<string> {
  const digest = await cryptoAPI.subtle.digest('SHA-256', encoder.encode(data));
  return bytesToHex(new Uint8Array(digest));
}

/**
 * RFC3986 percent-encoding as required by SigV4. Every byte outside the
 * unreserved set [A-Za-z0-9-._~] is escaped; multi-byte UTF-8 characters are
 * escaped byte-by-byte. Slashes are escaped unless `encodeSlash` is false
 * (used for the canonical URI path, where segment separators stay literal).
 */
function uriEncode(str: string, encodeSlash = true): string {
  let out = '';
  for (const ch of str) {
    if (/[A-Za-z0-9\-._~]/.test(ch)) {
      out += ch;
    } else if (ch === '/' && !encodeSlash) {
      out += ch;
    } else {
      for (const b of encoder.encode(ch)) {
        out += '%' + b.toString(16).toUpperCase().padStart(2, '0');
      }
    }
  }
  return out;
}

/**
 * Format a Date as the SigV4 "basic format" UTC timestamp `YYYYMMDDTHHMMSSZ`
 * used for the X-Amz-Date parameter. Milliseconds and local offset are dropped.
 */
export function amzDate(date: Date): string {
  const iso = date.toISOString(); // e.g. 2026-07-26T12:00:00.000Z
  return iso.replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
}

export interface SigningScope {
  secretAccessKey: string;
  dateStamp: string; // YYYYMMDD
  region: string;
  service: string;
}

/**
 * Derive the SigV4 signing key and produce the final hex signature for a given
 * string-to-sign. This is the crypto core; it is deliberately exported so tests
 * can drive it directly with the authoritative get-vanilla vector.
 */
export async function sigv4Signature(
  stringToSign: string,
  { secretAccessKey, dateStamp, region, service }: SigningScope,
): Promise<string> {
  const kDate = await hmac(encoder.encode('AWS4' + secretAccessKey), dateStamp);
  const kRegion = await hmac(kDate, region);
  const kService = await hmac(kRegion, service);
  const kSigning = await hmac(kService, 'aws4_request');
  const signature = await hmac(kSigning, stringToSign);
  return bytesToHex(signature);
}

export interface PresignParams {
  accessKeyId: string;
  secretAccessKey: string;
  method: string; // 'GET' | 'PUT'
  host: string; // e.g. '<account>.r2.cloudflarestorage.com'
  path: string; // canonical URI, must begin with '/', e.g. '/bucket/key.wav'
  region: string; // 'auto' for R2
  service?: string; // default 's3'
  amzDate: string; // 'YYYYMMDDTHHMMSSZ'
  expiresSeconds: number; // 1 .. 604800
  signedHeaders?: string; // default 'host'
}

/**
 * Build a fully presigned S3/R2 URL (query-string auth, UNSIGNED-PAYLOAD). The
 * returned URL carries all X-Amz-* auth params plus the computed signature, so
 * the holder can perform exactly one `method` request against `path` until it
 * expires — nothing else is needed on the wire.
 */
export async function presignS3Url(p: PresignParams): Promise<string> {
  const service = p.service ?? 's3';
  const signedHeaders = p.signedHeaders ?? 'host';
  const dateStamp = p.amzDate.slice(0, 8);
  const credential = `${p.accessKeyId}/${dateStamp}/${p.region}/${service}/aws4_request`;

  const params: Record<string, string> = {
    'X-Amz-Algorithm': 'AWS4-HMAC-SHA256',
    'X-Amz-Credential': credential,
    'X-Amz-Date': p.amzDate,
    'X-Amz-Expires': String(p.expiresSeconds),
    'X-Amz-SignedHeaders': signedHeaders,
  };
  const canonicalQuerystring = Object.keys(params)
    .sort()
    .map((k) => `${uriEncode(k)}=${uriEncode(params[k])}`)
    .join('&');

  // Encode each path segment but keep the separating slashes literal.
  const canonicalUri = p.path
    .split('/')
    .map((seg) => uriEncode(seg))
    .join('/');

  const canonicalHeaders = `host:${p.host}\n`;
  const canonicalRequest = [
    p.method,
    canonicalUri,
    canonicalQuerystring,
    canonicalHeaders,
    signedHeaders,
    'UNSIGNED-PAYLOAD',
  ].join('\n');

  const stringToSign = [
    'AWS4-HMAC-SHA256',
    p.amzDate,
    `${dateStamp}/${p.region}/${service}/aws4_request`,
    await sha256Hex(canonicalRequest),
  ].join('\n');

  const signature = await sigv4Signature(stringToSign, {
    secretAccessKey: p.secretAccessKey,
    dateStamp,
    region: p.region,
    service,
  });

  return `https://${p.host}${canonicalUri}?${canonicalQuerystring}&X-Amz-Signature=${signature}`;
}

/** Cloudflare R2 access config the Gateway holds as secrets + bindings. */
export interface R2Config {
  accountId: string; // → host `${accountId}.r2.cloudflarestorage.com`
  bucket: string;
  accessKeyId: string;
  secretAccessKey: string;
  region?: string; // default 'auto' (R2)
}

export interface JobAudioUrls {
  key: string; // the unique R2 object key
  uploadUrl: string; // presigned PUT
  getUrl: string; // presigned GET
  expiresSeconds: number;
}

// AWS/S3 caps the presign validity window at 7 days.
const MAX_PRESIGN_EXPIRES = 604800;

/**
 * Mint the pair of presigned URLs for one job's audio: a PUT the client uses to
 * upload straight to R2, and a GET the GPU worker uses to fetch it back. The
 * object key is namespaced per (device, job) so jobs never collide and one
 * job's URL can never address another job's object. Both URLs are signed for a
 * single method and expire together.
 */
export async function mintJobAudioUrls(opts: {
  config: R2Config;
  deviceId: string;
  jobId: string;
  amzDate: string; // 'YYYYMMDDTHHMMSSZ' — injected so callers/tests control the clock
  expiresSeconds: number;
}): Promise<JobAudioUrls> {
  const { config, deviceId, jobId, amzDate, expiresSeconds } = opts;
  if (
    !Number.isInteger(expiresSeconds) ||
    expiresSeconds < 1 ||
    expiresSeconds > MAX_PRESIGN_EXPIRES
  ) {
    throw new Error(
      `expiresSeconds must be an integer in [1, ${MAX_PRESIGN_EXPIRES}]`,
    );
  }

  const region = config.region ?? 'auto';
  const host = `${config.accountId}.r2.cloudflarestorage.com`;
  const key = `audio/${deviceId}/${jobId}.wav`;
  const path = `/${config.bucket}/${key}`;

  const common = {
    accessKeyId: config.accessKeyId,
    secretAccessKey: config.secretAccessKey,
    host,
    path,
    region,
    amzDate,
    expiresSeconds,
  };

  const uploadUrl = await presignS3Url({ ...common, method: 'PUT' });
  const getUrl = await presignS3Url({ ...common, method: 'GET' });

  return { key, uploadUrl, getUrl, expiresSeconds };
}
