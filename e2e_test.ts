import { generateECDSAKeyPair, signPayload } from './packages/crypto-utils/src/ecdsa';
import { JobRequest, deterministicStringify } from './packages/shared-types/src/index';

async function runE2E() {
  console.log("🚀 Bắt đầu giả lập quá trình End-to-End (Client -> Gateway -> GPU Worker)...");
  
  // 1. Sinh khóa ECDSA
  console.log("\n[Client] 1. Khởi tạo Khóa bảo mật ECDSA...");
  const { publicKeyJwk, privateKeyJwk } = await generateECDSAKeyPair();
  
  // 2. Gọi Đăng ký lấy Device ID
  console.log("[Client] 2. Gọi Đăng ký Thiết bị tới Gateway (Cổng 8787)...");
  const regRes = await fetch('http://localhost:8787/api/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ publicKeyJwk })
  });
  
  if (!regRes.ok) {
    console.error("❌ Đăng ký thất bại", await regRes.text());
    return;
  }
  
  const regData = await regRes.json();
  const deviceId = regData.deviceId;
  console.log("✅ Đã nhận Thẻ Căn cước (Device ID):", deviceId);
  
  // 3. Tạo Gói tin (Payload) và ký
  console.log("\n[Client] 3. Đóng gói Audio và Ký Mật mã Zero-Trust...");
  const payload: JobRequest = {
    jobId: 'OMNI-JOB-' + Date.now(),
    videoAudioUrl: 'https://r2.cloudflare.com/e2e_test_audio.wav',
    config: { targetLanguage: 'Vietnamese', translationStyle: 'Formal' },
    speakerMapping: { 'SPEAKER_01': 'Voice_Nam' },
    timestamp: Date.now()
  };
  
  const payloadStr = deterministicStringify(payload);
  const signature = await signPayload(payloadStr, privateKeyJwk);
  console.log("✅ Chữ ký tạo thành công:", signature.substring(0, 50) + "...");
  
  // 4. Bắn Request qua Gateway
  console.log("\n[Client] 4. Gửi Job lên Gateway (Chờ Gateway thẩm định và đẩy qua GPU)...");
  const jobRes = await fetch('http://localhost:8787/api/jobs/create', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-ECDSA-Signature': signature,
      'X-Device-Id': deviceId
    },
    body: payloadStr
  });
  
  const result = await jobRes.json();
  
  if (jobRes.ok) {
    console.log("\n🎉 [SUCCESS] KẾT QUẢ END-TO-END TỪ GPU WORKER TRẢ VỀ:");
    console.log(JSON.stringify(result, null, 2));
  } else {
    console.error("\n❌ [ERROR] Lỗi từ hệ thống:", result);
  }
}

runE2E();
