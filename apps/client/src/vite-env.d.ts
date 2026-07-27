/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** URL của Cloudflare Gateway (mặc định http://localhost:8787 khi dev). */
  readonly VITE_GATEWAY_URL?: string;
  /** Site key Cloudflare Turnstile — nếu đặt thì hiện widget chống bot khi đăng ký. */
  readonly VITE_TURNSTILE_SITE_KEY?: string;
  // R2 audio KHÔNG còn biến client nào (Đợt 30, Option A "Gateway ký URL mỗi job"):
  // mỗi job client gọi POST ${VITE_GATEWAY_URL}/api/uploads/presign để Gateway ký cặp
  // presigned PUT/GET. Cấu hình R2 nằm ở Gateway (apps/gateway/.dev.vars.example).
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
