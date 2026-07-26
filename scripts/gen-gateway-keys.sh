#!/usr/bin/env bash
#
# Sinh cặp khoá ECDSA P-256 (thuật toán JWT ES256) cho TRẠM 2 (Zero-Trust bất đối xứng):
#   - Gateway giữ PRIVATE (PKCS8) để KÝ token       -> Cloudflare secret GATEWAY_JWT_PRIVATE_KEY
#   - Worker  giữ PUBLIC  (SPKI)  để XÁC MINH token -> env worker      GATEWAY_JWT_PUBLIC_KEY
#
# Bất đối xứng nghĩa là dù ảnh worker bị lộ, kẻ tấn công chỉ có PUBLIC key -> KHÔNG thể giả
# mạo quyền Gateway (không có private để ký). TUYỆT ĐỐI KHÔNG commit khoá vào repo.
#
# Dùng:  scripts/gen-gateway-keys.sh [thư_mục_ra]
#   Không truyền thư mục -> ghi vào một thư mục tạm ngoài repo.
set -euo pipefail

command -v openssl >/dev/null 2>&1 || { echo "Cần openssl trong PATH." >&2; exit 1; }

OUT="${1:-$(mktemp -d)}"
mkdir -p "$OUT"
raw="$OUT/ec_raw.pem"
priv="$OUT/gateway_private_pkcs8.pem"
pub="$OUT/gateway_public_spki.pem"

# P-256 private key -> chuyển PKCS8 (định dạng cả PyJWT/cryptography lẫn WebCrypto đọc được).
openssl ecparam -name prime256v1 -genkey -noout -out "$raw"
openssl pkcs8 -topk8 -nocrypt -in "$raw" -out "$priv"
openssl ec -in "$raw" -pubout -out "$pub"
rm -f "$raw"
chmod 600 "$priv"

echo
echo "==================================================================================="
echo " PRIVATE KEY (Gateway KÝ) -> Cloudflare secret GATEWAY_JWT_PRIVATE_KEY"
echo "   cd apps/gateway && wrangler secret put GATEWAY_JWT_PRIVATE_KEY"
echo "   (dán TOÀN BỘ nội dung dưới đây, gồm cả dòng BEGIN/END)"
echo "-----------------------------------------------------------------------------------"
cat "$priv"
echo "==================================================================================="
echo " PUBLIC KEY (Worker XÁC MINH) -> env worker GATEWAY_JWT_PUBLIC_KEY"
echo "   (đặt trong .env / secret của RunPod-Modal; là PEM nhiều dòng)"
echo "-----------------------------------------------------------------------------------"
cat "$pub"
echo "==================================================================================="
echo
echo "Khoá đã ghi tại: $OUT  (NGOÀI repo)."
echo "Xoá sau khi đã nạp xong hai nơi:  rm -rf \"$OUT\""
