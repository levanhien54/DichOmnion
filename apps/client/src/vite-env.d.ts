/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** URL của Cloudflare Gateway (mặc định http://localhost:8787 khi dev). */
  readonly VITE_GATEWAY_URL?: string;
  /** Site key Cloudflare Turnstile — nếu đặt thì hiện widget chống bot khi đăng ký. */
  readonly VITE_TURNSTILE_SITE_KEY?: string;
  /** Presigned PUT URL để đẩy AUDIO đã tách lên lưu trữ đối tượng (R2/S3). */
  readonly VITE_AUDIO_UPLOAD_URL?: string;
  /** URL công khai để GPU worker tải AUDIO về (khớp với đối tượng vừa PUT). */
  readonly VITE_AUDIO_PUBLIC_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
