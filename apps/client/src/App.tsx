import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import {
  Film, Mic2, ShieldCheck, Languages, KeyRound, CheckCircle2, ArrowRight,
  FileAudio, AlertTriangle, Plus, Loader2, XCircle, Download, Save, PauseCircle,
} from 'lucide-react';
import {
  generateECDSAKeyPair,
  importPrivateSigningKey,
  signPayloadWithKey,
} from '@dichomnion/crypto-utils';
import { JobRequest, deterministicStringify } from '@dichomnion/shared-types';

import { SubtitleEditor, Subtitle } from './components/SubtitleEditor';
import { VoiceMapper } from './components/VoiceMapper';
import { TurnstileWidget } from './components/TurnstileWidget';
import {
  isTauri, pickVideoFile, extractAudio, muxAudioToVideo, writeTempAudio,
  saveOutputVideo, cleanupTempFile, AudioInfo,
} from './lib/tauri';
import { uploadAudioForWorker } from './lib/transport';
import { savePrivateKey, loadPrivateKey } from './lib/keystore';

const GATEWAY_URL = import.meta.env.VITE_GATEWAY_URL || 'http://localhost:8787';
const TURNSTILE_SITE_KEY = import.meta.env.VITE_TURNSTILE_SITE_KEY;

// Danh sách ngôn ngữ dùng CHUNG cho cả select nguồn lẫn đích. `value` là tên tiếng
// Anh gửi thẳng cho worker (target_language -> prompt LLM; source_language ->
// count_syllables). `label` là nhãn hiển thị tiếng Việt.
const LANGUAGES: ReadonlyArray<{ value: string; label: string }> = [
  { value: 'Vietnamese', label: 'Tiếng Việt' },
  { value: 'English', label: 'Tiếng Anh' },
  { value: 'Japanese', label: 'Tiếng Nhật' },
  { value: 'Korean', label: 'Tiếng Hàn' },
  { value: 'Chinese', label: 'Tiếng Trung' },
  { value: 'French', label: 'Tiếng Pháp' },
  { value: 'Spanish', label: 'Tiếng Tây Ban Nha' },
  { value: 'German', label: 'Tiếng Đức' },
];

/**
 * Quy mốc thời gian người dùng nhập ("HH:MM:SS", "MM:SS", "SS", hoặc số/thập phân
 * kiểu "1,5") về GIÂY (số thực) TRƯỚC KHI KÝ. Hợp đồng `ClientSegment.start/end`
 * là số giây; worker (translation_service tính duration, audio_engine căn mix) đọc
 * SỐ — nếu gửi thẳng chuỗi "HH:MM:SS", `float()` phía worker sẽ hỏng rồi rơi về
 * pacing mặc định sai (CC-1). Không parse được -> 0 (không đoán bừa). Logic này
 * phản chiếu 1-1 src/timecode.py phía worker để hai đầu nhất quán tuyệt đối.
 */
export function timecodeToSeconds(tc: string): number {
  const s = (tc ?? '').trim().replace(',', '.');
  if (!s) return 0;
  const direct = Number(s);
  if (!Number.isNaN(direct)) return direct; // đã là số ("83", "1.5")
  const parts = s.split(':').map((p) => Number(p));
  if (parts.some((n) => Number.isNaN(n))) return 0; // rác -> 0
  return parts.reduce((acc, n) => acc * 60 + n, 0);
}

// Trạng thái công việc đã kết thúc (do Gateway trả về khi poll). Không có trạng
// thái "giả thành công" — client phản ánh đúng những gì server báo.
const TERMINAL_STATUSES = ['DONE', 'FAILED', 'REJECTED_FRAUD', 'TERMINATED_TIMEOUT', 'ERROR'];

// CLIENT-02: ngân sách poll phía client PHẢI bao trùm cửa sổ render tối đa của server
// (MAX_PLAUSIBLE_MS = 15') cộng đệm mạng, để client thấy được KẾT LUẬN của server
// (DONE / TERMINATED_TIMEOUT) thay vì tự cắt sớm ở 5' rồi báo "thất bại" oan. Hết
// ngân sách mà server chưa kết luận -> POLL_PAUSED (mềm, cho phép tiếp tục), KHÔNG
// coi là lỗi cứng.
const POLL_INTERVAL_MS = 2500;
const CLIENT_POLL_BUDGET_MS = 16 * 60_000;
const ACTIVE_JOB_KEY = 'omni_active_job';

// Lưu/xóa jobId đang chạy để RESUME được sau khi tải lại app (chỉ localStorage, không
// đụng state — an toàn khi gọi từ trong useCallback/closure).
function persistActiveJob(id: string) {
  try {
    localStorage.setItem(ACTIVE_JOB_KEY, id);
  } catch {
    /* storage đầy/tắt — resume là tiện ích, không chặn luồng chính */
  }
}
function clearActiveJob() {
  try {
    localStorage.removeItem(ACTIVE_JOB_KEY);
  } catch {
    /* no-op */
  }
}

// Kết quả TRUNG THỰC worker báo (gateway đã lược bỏ đường dẫn temp nội bộ, thêm
// artifactReady/downloadUrl). Client hiển thị ĐÚNG các cờ này — không tự quảng cáo
// "đã tách nền + watermark" khi bước đó thực tế không chạy (NFS-01).
interface JobResult {
  message?: string;
  device_used?: string;
  pipeline?: string[];
  separated?: boolean;
  watermarked?: boolean;
  distinct_voices?: number;
  notes?: string[];
  artifactReady?: boolean;
  downloadUrl?: string;
}

/** Chuyển Uint8Array -> base64 theo TỪNG KHỐI để không tràn stack với audio lớn
 *  (String.fromCharCode(...) trên mảng chục MB sẽ nổ "Maximum call stack"). */
function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

/** Cờ trạng thái TRUNG THỰC: ✓ nếu bước thực sự chạy, ✗ nếu không (không tô hồng). */
function Flag({ ok, label }: { ok?: boolean; label: string }) {
  return (
    <div className="flex items-center gap-1.5 text-muted-foreground">
      {ok ? (
        <CheckCircle2 className="w-3.5 h-3.5 text-primary shrink-0" />
      ) : (
        <XCircle className="w-3.5 h-3.5 text-muted-foreground/60 shrink-0" />
      )}
      <span className={ok ? 'text-foreground' : ''}>{label}</span>
    </div>
  );
}

type Identity = 'BOOTING' | 'GENERATING' | 'NEEDS_REGISTRATION' | 'REGISTERING' | 'READY' | 'ERROR';

function App() {
  const [step, setStep] = useState<1 | 2 | 3>(1);

  // Định danh Zero-Trust (khóa non-extractable trong IndexedDB).
  const [identity, setIdentity] = useState<Identity>('BOOTING');
  const [publicKeyJwk, setPublicKeyJwk] = useState<JsonWebKey | null>(null);
  const [deviceId, setDeviceId] = useState<string | null>(null);
  const [regError, setRegError] = useState<string | null>(null);
  const privateKeyRef = useRef<CryptoKey | null>(null);

  // Step 1 — Chuẩn bị & tách audio cục bộ
  const [videoPath, setVideoPath] = useState<string | null>(null);
  const [audioInfo, setAudioInfo] = useState<AudioInfo | null>(null);
  // Ngôn ngữ GỐC (nguồn) và ĐÍCH. `value` là tên tiếng Anh gửi thẳng cho worker:
  //  - targetLanguage -> bơm vào prompt LLM (dịch sang ngôn ngữ này).
  //  - sourceLanguage -> count_syllables ở worker (nhánh "vi" khi .startswith("vi"),
  //    còn lại dùng heuristic tiếng Anh) để căn lip-sync đúng.
  const [sourceLang, setSourceLang] = useState('English');
  const [targetLang, setTargetLang] = useState('Vietnamese');
  const [style, setStyle] = useState<'Formal' | 'Casual' | 'Slang'>('Formal');

  // Step 2 — Human-in-the-loop (tuỳ chọn) + gán giọng
  const [subtitles, setSubtitles] = useState<Subtitle[]>([]);
  const [mapping, setMapping] = useState<Record<string, string>>({});

  // Step 3 — Trạng thái công việc thực từ Gateway (poll)
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [jobResult, setJobResult] = useState<JobResult | null>(null);
  const [jobEta, setJobEta] = useState<number | null>(null); // ETA (giây) server ước tính
  const resumedRef = useRef(false); // chỉ RESUME job đang dang dở đúng 1 lần / phiên
  // "Vé sở hữu" vòng poll: mỗi lần pollJob chạy sẽ ++token và giành quyền ghi state.
  // Vòng poll CŨ (sau reset, hoặc khi submit/resume job khác) thấy token đã đổi thì
  // DỪNG — không để loop cũ ghi đè trạng thái job hiện tại hay hồi sinh state đã reset.
  const pollTokenRef = useRef(0);
  // Chỉ TỰ đăng ký (dev, không có Turnstile) đúng MỘT lần/phiên. register() khi lỗi sẽ
  // đặt identity về NEEDS_REGISTRATION — nếu không chốt, effect phụ thuộc identity sẽ
  // gọi lại register() liên tục (spam Gateway khi nó đang tắt). Muốn thử lại: tải lại trang.
  const autoRegisterAttemptedRef = useRef(false);

  // Lắp ráp cục bộ (CLIENT-01): tải track lồng tiếng -> mux vào video gốc -> lưu.
  const [assembling, setAssembling] = useState(false);
  const [outputVideoPath, setOutputVideoPath] = useState<string | null>(null); // temp mp4 đã mux, chờ lưu
  const [savedPath, setSavedPath] = useState<string | null>(null);
  const [assembleError, setAssembleError] = useState<string | null>(null);

  const [status, setStatus] = useState<string>('IDLE');
  const [error, setError] = useState<string | null>(null);

  // --- Đăng ký thiết bị với Gateway (Trạm 1) ---------------------------------
  const register = useCallback(
    async (turnstileToken?: string) => {
      if (!publicKeyJwk) return;
      setIdentity('REGISTERING');
      setRegError(null);
      try {
        const res = await fetch(`${GATEWAY_URL}/api/auth/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ publicKeyJwk, turnstileToken }),
        });
        if (!res.ok) {
          const messages: Record<number, string> = {
            403: 'Xác minh chống bot thất bại. Vui lòng thử lại.',
            429: 'Quá số lần đăng ký cho phép trên IP này. Thử lại sau.',
            503: 'Dịch vụ tạm ngưng (Kill Switch tài chính đang bật).',
          };
          setRegError(messages[res.status] || `Đăng ký thất bại (HTTP ${res.status}).`);
          setIdentity('NEEDS_REGISTRATION');
          return;
        }
        const data = await res.json();
        localStorage.setItem('omni_device_id', data.deviceId);
        setDeviceId(data.deviceId);
        setIdentity('READY');
      } catch {
        // Zero-Logging: không in payload/khóa/token ra console.
        setRegError('Không kết nối được Gateway. Kiểm tra dịch vụ Edge.');
        setIdentity('NEEDS_REGISTRATION');
      }
    },
    [publicKeyJwk],
  );

  // --- Khởi tạo định danh 1 lần: khóa non-extractable trong IndexedDB --------
  useEffect(() => {
    (async () => {
      try {
        const priv = await loadPrivateKey();
        const pubRaw = localStorage.getItem('omni_public_key');
        const dev = localStorage.getItem('omni_device_id');
        if (priv && pubRaw && dev) {
          privateKeyRef.current = priv;
          setPublicKeyJwk(JSON.parse(pubRaw));
          setDeviceId(dev);
          setIdentity('READY');
          return;
        }

        setIdentity('GENERATING');
        const { publicKeyJwk: pub, privateKeyJwk } = await generateECDSAKeyPair();
        // Nhập khóa RIÊNG ở dạng KHÔNG THỂ export rồi lưu IndexedDB; JWK riêng bị
        // vứt bỏ ngay (không hề chạm localStorage). An toàn hơn bản cũ lưu JWK.
        const nonExtractable = await importPrivateSigningKey(privateKeyJwk, false);
        await savePrivateKey(nonExtractable);
        privateKeyRef.current = nonExtractable;
        localStorage.setItem('omni_public_key', JSON.stringify(pub));
        setPublicKeyJwk(pub);
        setIdentity('NEEDS_REGISTRATION');
      } catch {
        setIdentity('ERROR');
      }
    })();
  }, []);

  // Không cấu hình Turnstile (dev): tự đăng ký ngay khi có khóa. Có cấu hình:
  // chờ người dùng vượt widget rồi mới đăng ký (register nhận token qua callback).
  useEffect(() => {
    if (
      identity === 'NEEDS_REGISTRATION' &&
      !TURNSTILE_SITE_KEY &&
      publicKeyJwk &&
      !autoRegisterAttemptedRef.current
    ) {
      autoRegisterAttemptedRef.current = true; // chốt: không tự retry vòng lặp khi lỗi
      register();
    }
  }, [identity, publicKeyJwk, register]);

  const uniqueSpeakers = useMemo(
    () => Array.from(new Set(subtitles.map((s) => s.speaker))),
    [subtitles],
  );

  const handleSubtitleChange = (id: string, newText: string) => {
    setSubtitles((prev) => prev.map((s) => (s.id === id ? { ...s, text: newText } : s)));
  };
  const handleSpeakerChange = (id: string, newSpeaker: string) => {
    setSubtitles((prev) => prev.map((s) => (s.id === id ? { ...s, speaker: newSpeaker } : s)));
  };
  const handleMappingChange = (speaker: string, voiceId: string) => {
    setMapping((prev) => ({ ...prev, [speaker]: voiceId }));
  };

  // --- Step 1: chọn video + tách audio CỤC BỘ (không upload video thô) -------
  const pickVideo = async () => {
    setError(null);
    try {
      const path = await pickVideoFile();
      if (path) {
        setVideoPath(path);
        setAudioInfo(null);
      }
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const startExtraction = async () => {
    if (!videoPath) return;
    setStatus('EXTRACTING');
    setError(null);
    try {
      // TR-2: nếu đã tách trước đó (đổi video / tách lại), dọn audio tạm cũ để
      // không tích tụ ~30MB mỗi lượt trong thư mục tạm. Fire-and-forget, best-effort.
      if (audioInfo?.audio_path) void cleanupTempFile(audioInfo.audio_path);
      // FFmpeg CỤC BỘ (Rust): tách 16kHz mono, trả md5 + kích thước thật.
      const info = await extractAudio(videoPath);
      setAudioInfo(info);
      setStep(2);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setStatus('IDLE');
    }
  };

  // Human-in-the-loop TUỲ CHỌN: người dùng có thể tự thêm câu thoại để định
  // hướng/ghi đè ASR. Bỏ trống -> worker tự bóc băng (Whisper). Không có dữ liệu giả.
  const addSegment = () => {
    setSubtitles((prev) => {
      // Người nói mặc định KẾ THỪA dòng trước (nhiều câu liền thường cùng một người);
      // dòng đầu -> SPEAKER_01. Ô người nói có thể sửa để tách thành nhiều giọng —
      // uniqueSpeakers/VoiceMapper tự cập nhật theo các tên duy nhất.
      const lastSpeaker = prev.length > 0 ? prev[prev.length - 1].speaker : '';
      return [
        ...prev,
        {
          id: `seg-${Date.now()}-${prev.length}`,
          speaker: lastSpeaker || 'SPEAKER_01',
          start: '00:00:00',
          end: '00:00:00',
          text: '',
        },
      ];
    });
  };

  // --- Step 2 -> submit: upload AUDIO, ký non-extractable, gửi, POLL thật ----
  const pollJob = useCallback(
    async (id: string, device: string) => {
      // Giành quyền sở hữu: token này chỉ còn "của mình" tới khi có pollJob khác chạy
      // hoặc resetAll bump token. Kiểm tra token sau MỖI await để loop cũ tự rút lui.
      const myToken = (pollTokenRef.current += 1);
      const owns = () => pollTokenRef.current === myToken;
      setStatus('PROCESSING');
      // CLIENT-02: ngân sách bao trùm cửa sổ render tối đa của server (15') + đệm.
      const deadline = Date.now() + CLIENT_POLL_BUDGET_MS;
      while (Date.now() < deadline) {
        if (!owns()) return; // bị thay thế (job mới / reset) -> ngừng, không ghi state
        try {
          const res = await fetch(`${GATEWAY_URL}/api/jobs/${encodeURIComponent(id)}`, {
            headers: { 'X-Device-Id': device },
          });
          if (!owns()) return; // re-check sau await mạng: tránh ghi đè job hiện tại
          if (res.ok) {
            const rec = await res.json();
            if (!owns()) return;
            setJobStatus(rec.status);
            if (typeof rec.etaSeconds === 'number') setJobEta(rec.etaSeconds);
            if (rec.result) setJobResult(rec.result as JobResult); // cờ trung thực khi DONE
            if (TERMINAL_STATUSES.includes(rec.status)) {
              clearActiveJob(); // đã kết luận (DONE/FAILED/...) — thôi resume
              setStatus('IDLE');
              return;
            }
          }
        } catch {
          // Lỗi mạng tạm thời — thử lại chu kỳ sau (không tự ý báo thành công).
        }
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
      }
      if (!owns()) return;
      // Hết ngân sách mà server CHƯA kết luận: KHÔNG coi là thất bại (server có thể
      // vẫn đang render). Chuyển sang trạng thái MỀM POLL_PAUSED, GIỮ jobId đã lưu
      // để người dùng bấm "Tiếp tục kiểm tra" (hoặc tự resume khi mở lại app).
      setJobStatus('POLL_PAUSED');
      setStatus('IDLE');
    },
    [],
  );

  // CLIENT-02: RESUME job dang dở. Nếu app bị đóng/tải lại khi một job còn đang
  // render trên server, ta còn jobId trong localStorage — khôi phục ngay khi định
  // danh sẵn sàng và tiếp tục poll (không bắt người dùng gửi lại từ đầu). Chỉ chạy
  // đúng 1 lần/phiên (resumedRef) để không nhân đôi vòng poll.
  useEffect(() => {
    if (resumedRef.current || identity !== 'READY' || !deviceId) return;
    let saved: string | null = null;
    try {
      saved = localStorage.getItem(ACTIVE_JOB_KEY);
    } catch {
      saved = null;
    }
    if (!saved) return;
    resumedRef.current = true;
    setJobId(saved);
    setStep(3);
    void pollJob(saved, deviceId);
  }, [identity, deviceId, pollJob]);

  const submitJob = async () => {
    if (!privateKeyRef.current || !deviceId || !audioInfo) return;

    // Nếu người dùng đã nhập câu thoại thì bắt buộc gán giọng cho mọi nhân vật.
    if (subtitles.length > 0 && !uniqueSpeakers.every((s) => mapping[s])) {
      setError('Vui lòng gán giọng cho tất cả nhân vật bạn đã nhập.');
      return;
    }

    setStatus('UPLOADING_AUDIO');
    setError(null);
    try {
      // Chỉ AUDIO rời máy (fail-closed nếu chưa cấu hình lưu trữ đối tượng).
      const audioUrl = await uploadAudioForWorker(audioInfo.audio_path);

      const payload: JobRequest = {
        jobId: `JOB-${Date.now()}`,
        videoAudioUrl: audioUrl,
        config: { targetLanguage: targetLang, translationStyle: style, sourceLanguage: sourceLang },
        speakerMapping: mapping,
        timestamp: Date.now(),
        // CC-1: quy "HH:MM:SS" -> giây và gửi ĐÚNG hình dạng ClientSegment (id,
        // speaker, text, start/end dạng số). Bản cũ nhét thẳng Subtitle với start/end
        // là chuỗi -> worker đọc số hỏng.
        ...(subtitles.length > 0
          ? {
              segments: subtitles.map((s) => ({
                id: s.id,
                speaker: s.speaker,
                text: s.text,
                start: timecodeToSeconds(s.start),
                end: timecodeToSeconds(s.end),
              })),
            }
          : {}),
      };

      const payloadStr = deterministicStringify(payload);
      // Ký bằng khóa non-extractable (không lộ chất liệu khóa). KHÔNG log payload.
      const signature = await signPayloadWithKey(payloadStr, privateKeyRef.current);

      setStatus('SUBMITTING');
      const res = await fetch(`${GATEWAY_URL}/api/jobs/create`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-ECDSA-Signature': signature,
          'X-Device-Id': deviceId,
        },
        body: payloadStr,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setError(`Gateway từ chối: ${err.error || `HTTP ${res.status}`}`);
        setStatus('IDLE');
        return;
      }

      const data = await res.json(); // { status: 'QUEUED', jobId, etaSeconds }
      setJobId(data.jobId);
      setJobStatus(data.status || 'QUEUED');
      if (typeof data.etaSeconds === 'number') setJobEta(data.etaSeconds);
      // CLIENT-02: ghi jobId để RESUME được nếu app đóng/tải lại giữa chừng.
      persistActiveJob(data.jobId);
      setStep(3);
      // TR-2: audio nguồn đã nằm an toàn trên R2 và job đã vào hàng đợi — dọn bản
      // tạm cục bộ ngay. (Bước mux cuối dùng VIDEO GỐC + audio lồng tiếng tải về,
      // KHÔNG cần audio nguồn này nữa.) Sau 202, UI sang bước 3 nên không resubmit.
      void cleanupTempFile(audioInfo.audio_path);
      // Hợp đồng BẤT ĐỒNG BỘ thật: Gateway trả 202 QUEUED rồi ta poll trạng thái.
      await pollJob(data.jobId, deviceId);
    } catch (e) {
      setError((e as Error).message);
      setStatus('IDLE');
    }
  };

  // Mở hộp thoại Save và chép video KẾT QUẢ (temp) ra nơi người dùng chọn. Hủy hộp
  // thoại -> giữ nguyên outputVideoPath để bấm "Lưu" lại (không mất công đã ghép).
  const saveAssembled = async (tempOut: string) => {
    const { save } = await import('@tauri-apps/api/dialog');
    // Mặc định mở tại thư mục Downloads: Rust chỉ cho phép chép kết quả VÀO Downloads
    // (chống ghi file tùy ý nếu webview bị XSS gọi thẳng IPC). defaultPath dẫn người
    // dùng về đúng thư mục hợp lệ; lưu ra ngoài Downloads sẽ bị Rust từ chối rõ ràng.
    const fileName = `omnivoice-dubbed-${jobId}.mp4`;
    let defaultPath = fileName;
    try {
      const { downloadDir, join } = await import('@tauri-apps/api/path');
      defaultPath = await join(await downloadDir(), fileName);
    } catch {
      // Không lấy được thư mục Downloads => để dialog tự chọn thư mục mặc định.
    }
    const dest = await save({
      defaultPath,
      filters: [{ name: 'Video', extensions: ['mp4'] }],
    });
    if (!dest) return;
    await saveOutputVideo(tempOut, dest);
    setSavedPath(dest);
    setOutputVideoPath(null);
    void cleanupTempFile(tempOut); // đã lưu ra ngoài -> dọn bản temp
  };

  // --- CLIENT-01: tải track lồng tiếng về + GHÉP (mux) vào video gốc CỤC BỘ -----
  // Đóng vòng bất đồng bộ phía client: kết quả worker là AUDIO đã mix+watermark; ta
  // tải qua proxy gateway (device-scoped), ghi ra temp rồi ffmpeg ghép vào VIDEO GỐC
  // (video KHÔNG rời máy). Không "giả xong" — chỉ khi ghép thật mới có file kết quả.
  const downloadAndMux = async () => {
    if (!jobId || !deviceId || !jobResult?.downloadUrl) return;
    if (!videoPath) {
      setAssembleError('Cần video gốc trên máy để ghép. Hãy chọn lại video ở bước 1.');
      return;
    }
    setAssembling(true);
    setAssembleError(null);
    setSavedPath(null);
    let dubbedPath: string | null = null;
    try {
      // 1) Tải AUDIO lồng tiếng qua proxy gateway (kèm X-Device-Id: device-scoped).
      const res = await fetch(`${GATEWAY_URL}${jobResult.downloadUrl}`, {
        headers: { 'X-Device-Id': deviceId },
      });
      if (!res.ok) throw new Error(`Tải track lồng tiếng thất bại (HTTP ${res.status}).`);
      const bytes = new Uint8Array(await res.arrayBuffer());
      if (bytes.length === 0) throw new Error('Track lồng tiếng tải về rỗng.');

      // 2) Ghi ra file tạm (Rust) để ffmpeg dùng, rồi 3) mux vào video gốc.
      dubbedPath = await writeTempAudio(bytesToBase64(bytes));
      const outPath = await muxAudioToVideo(videoPath, dubbedPath);
      setOutputVideoPath(outPath);

      // 4) Lưu ra vị trí người dùng chọn (Save dialog).
      await saveAssembled(outPath);
    } catch (e) {
      setAssembleError((e as Error).message);
    } finally {
      // Track lồng tiếng nguồn không còn cần sau khi đã mux (giữ lại video kết quả).
      if (dubbedPath) void cleanupTempFile(dubbedPath);
      setAssembling(false);
    }
  };

  const resetAll = () => {
    // Còn video kết quả tạm chưa lưu -> dọn để không rớt rác trong thư mục tạm.
    if (outputVideoPath) void cleanupTempFile(outputVideoPath);
    // Vô hiệu hóa vòng poll đang chạy (nếu có): bump token để loop cũ ngừng ghi state,
    // tránh nó hồi sinh trạng thái job cũ sau khi ta vừa xóa sạch bên dưới.
    pollTokenRef.current += 1;
    clearActiveJob(); // bỏ job đã lưu -> không auto-resume job cũ nữa
    setStep(1);
    setVideoPath(null);
    setAudioInfo(null);
    setSubtitles([]);
    setMapping({});
    setJobId(null);
    setJobStatus(null);
    setJobResult(null);
    setJobEta(null);
    setOutputVideoPath(null);
    setSavedPath(null);
    setAssembleError(null);
    setError(null);
    setStatus('IDLE');
  };

  const identityReady = identity === 'READY';

  return (
    <div className="min-h-screen bg-background text-foreground font-sans p-8 flex flex-col items-center">
      {/* Header */}
      <header className="w-full max-w-6xl flex justify-between items-center mb-10">
        <div className="flex items-center gap-3">
          <Mic2 className="w-8 h-8 text-primary" />
          <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-cyan-500 bg-clip-text text-transparent">
            OmniVoice Studio
          </h1>
        </div>
        <div className="flex items-center gap-2 text-primary bg-primary/10 px-4 py-2 rounded-full border border-primary/20">
          <ShieldCheck className="w-4 h-4" />
          <span className="text-sm font-medium">Zero-Trust Secured</span>
        </div>
      </header>

      {/* Cảnh báo chạy ngoài desktop app (không có ffmpeg cục bộ) */}
      {!isTauri() && (
        <div className="w-full max-w-6xl mb-6 p-4 bg-amber-500/10 border border-amber-500/20 text-amber-400 rounded-xl flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 shrink-0" />
          <span>
            Đang chạy trong trình duyệt web. Việc tách/ghép video cần <b>ứng dụng desktop OmniVoice</b>{' '}
            (có ffmpeg đi kèm) — trình duyệt không truy cập được file gốc của bạn.
          </span>
        </div>
      )}

      {/* Panel định danh: sinh khóa + đăng ký (Turnstile nếu cấu hình) */}
      {!identityReady && (
        <div className="w-full max-w-6xl mb-6 p-5 bg-card border rounded-2xl">
          <div className="flex items-center gap-3 mb-2">
            <KeyRound className="w-5 h-5 text-primary animate-pulse" />
            <h2 className="font-semibold">Thiết lập định danh thiết bị (ECDSA non-extractable)</h2>
          </div>
          {identity === 'GENERATING' && <p className="text-sm text-muted-foreground">Đang sinh cặp khóa cục bộ…</p>}
          {identity === 'REGISTERING' && <p className="text-sm text-muted-foreground">Đang đăng ký với Gateway…</p>}
          {identity === 'ERROR' && (
            <p className="text-sm text-red-400">Không khởi tạo được định danh trên thiết bị này.</p>
          )}
          {identity === 'NEEDS_REGISTRATION' && (
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground">
                Hoàn tất xác minh để đăng ký thiết bị của bạn với Gateway.
              </p>
              {TURNSTILE_SITE_KEY && (
                <TurnstileWidget siteKey={TURNSTILE_SITE_KEY} onToken={(t) => t && register(t)} />
              )}
              {regError && <p className="text-sm text-red-400">{regError}</p>}
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="w-full max-w-6xl mb-6 p-4 bg-red-500/10 border border-red-500/20 text-red-400 rounded-xl flex items-center gap-3">
          <XCircle className="w-5 h-5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <main className="w-full max-w-6xl">
        {/* ---------------- STEP 1: chọn video + tách audio cục bộ ------------- */}
        {step === 1 && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2">
              <button
                onClick={pickVideo}
                disabled={!identityReady}
                className="w-full h-96 border-2 border-dashed border-input rounded-2xl bg-card/50 flex flex-col items-center justify-center transition-all hover:border-primary/50 hover:bg-card disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <div className="w-20 h-20 bg-muted rounded-full flex items-center justify-center mb-6 shadow-lg shadow-black/20">
                  <Film className="w-10 h-10 text-muted-foreground" />
                </div>
                <h3 className="text-xl font-medium mb-2 text-center px-4">
                  {videoPath ? videoPath.split(/[\\/]/).pop() : 'Chọn Video từ máy của bạn'}
                </h3>
                <p className="text-muted-foreground text-sm">MP4, MKV, MOV — video KHÔNG rời khỏi máy bạn</p>
                {audioInfo && (
                  <div className="mt-6 px-4 py-2 bg-primary/20 text-primary rounded-lg text-xs font-mono text-center">
                    <FileAudio className="w-4 h-4 inline mr-1" />
                    audio {Math.round(audioInfo.size_bytes / 1024)} KB · md5 {audioInfo.md5.slice(0, 12)}…
                  </div>
                )}
              </button>
            </div>

            <div className="flex flex-col gap-6">
              <div className="bg-card text-card-foreground border rounded-2xl p-6 shadow-sm">
                <div className="flex items-center gap-2 mb-6 border-b pb-4">
                  <Languages className="w-5 h-5 text-primary" />
                  <h2 className="text-lg font-semibold">Cấu hình Dịch thuật</h2>
                </div>
                <div className="space-y-5">
                  <div>
                    <label className="block text-sm text-muted-foreground mb-2">Ngôn ngữ Gốc (nguồn)</label>
                    <select
                      value={sourceLang}
                      onChange={(e) => setSourceLang(e.target.value)}
                      className="w-full bg-background border border-input rounded-lg p-3 text-sm focus:ring-2 focus:ring-ring focus:border-ring outline-none"
                    >
                      {LANGUAGES.map((l) => (
                        <option key={l.value} value={l.value}>{l.label}</option>
                      ))}
                    </select>
                    <p className="mt-1 text-xs text-muted-foreground">Giúp worker đếm âm tiết đúng để căn khẩu hình (lip-sync).</p>
                  </div>
                  <div>
                    <label className="block text-sm text-muted-foreground mb-2">Ngôn ngữ Đích</label>
                    <select
                      value={targetLang}
                      onChange={(e) => setTargetLang(e.target.value)}
                      className="w-full bg-background border border-input rounded-lg p-3 text-sm focus:ring-2 focus:ring-ring focus:border-ring outline-none"
                    >
                      {LANGUAGES.map((l) => (
                        <option key={l.value} value={l.value}>{l.label}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm text-muted-foreground mb-2">Phong cách (Tone)</label>
                    <select
                      value={style}
                      onChange={(e) => setStyle(e.target.value as 'Formal' | 'Casual' | 'Slang')}
                      className="w-full bg-background border border-input rounded-lg p-3 text-sm focus:ring-2 focus:ring-ring focus:border-ring outline-none"
                    >
                      <option value="Formal">Trang trọng (Tôi/Bạn)</option>
                      <option value="Casual">Tự nhiên (Mình/Cậu)</option>
                      <option value="Slang">GenZ (Lóng/Trend)</option>
                    </select>
                  </div>
                </div>
              </div>

              <button
                onClick={startExtraction}
                disabled={!videoPath || !identityReady || status !== 'IDLE'}
                className="w-full py-4 bg-primary hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl font-bold text-primary-foreground shadow-lg transition-all active:scale-95 flex items-center justify-center gap-2"
              >
                {status === 'EXTRACTING' ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" /> ĐANG TÁCH AUDIO CỤC BỘ…
                  </>
                ) : (
                  <>
                    TÁCH AUDIO CỤC BỘ <ArrowRight className="w-5 h-5" />
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* ---------------- STEP 2: human-in-the-loop (tuỳ chọn) + gửi -------- */}
        {step === 2 && (
          <div className="flex flex-col gap-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="bg-primary/5 border border-primary/20 rounded-2xl p-5 text-sm text-muted-foreground">
              Bạn có thể <b>tự thêm câu thoại</b> để định hướng bản dịch, hoặc để trống — khi đó GPU worker
              sẽ tự bóc băng (Whisper) rồi dịch &amp; lồng tiếng. Không có phụ đề bịa sẵn.
            </div>

            {subtitles.length > 0 ? (
              <>
                <SubtitleEditor subtitles={subtitles} onChange={handleSubtitleChange} onSpeakerChange={handleSpeakerChange} />
                <VoiceMapper speakers={uniqueSpeakers} mapping={mapping} onChange={handleMappingChange} />
              </>
            ) : (
              <div className="text-center text-muted-foreground py-8 border rounded-2xl bg-card/40">
                Chưa có câu thoại nào. Worker sẽ tự bóc băng, hoặc bấm “Thêm câu thoại”.
              </div>
            )}

            <div className="flex justify-between items-center gap-4 mt-2">
              <button
                onClick={addSegment}
                className="px-5 py-3 bg-secondary hover:bg-secondary/80 text-secondary-foreground rounded-xl font-medium transition-colors flex items-center gap-2"
              >
                <Plus className="w-4 h-4" /> Thêm câu thoại
              </button>
              <div className="flex gap-4">
                <button
                  onClick={() => setStep(1)}
                  className="px-6 py-3 bg-secondary hover:bg-secondary/80 text-secondary-foreground rounded-xl font-medium transition-colors"
                >
                  Quay lại
                </button>
                <button
                  onClick={submitJob}
                  disabled={status !== 'IDLE'}
                  className="px-8 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:opacity-50 rounded-xl font-bold text-white shadow-lg shadow-purple-900/30 transition-all active:scale-95 flex items-center gap-2"
                >
                  {status === 'IDLE' ? (
                    '🚀 KÝ ĐIỆN TỬ & LỒNG TIẾNG'
                  ) : (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" /> {status}
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ---------------- STEP 3: trạng thái công việc THẬT (poll) ---------- */}
        {step === 3 && (
          <div className="flex flex-col gap-8 animate-in zoom-in duration-500 w-full max-w-3xl mx-auto">
            <div className="flex flex-col items-center justify-center p-8 bg-card border rounded-2xl shadow-sm text-center">
              {jobStatus === 'DONE' ? (
                <div className="w-20 h-20 bg-primary/20 rounded-full flex items-center justify-center mb-6 border-4 border-primary/30">
                  <CheckCircle2 className="w-10 h-10 text-primary" />
                </div>
              ) : jobStatus === 'POLL_PAUSED' ? (
                // Trạng thái MỀM: hết ngân sách chờ phía client, KHÔNG phải lỗi.
                <div className="w-20 h-20 bg-amber-500/20 rounded-full flex items-center justify-center mb-6 border-4 border-amber-500/30">
                  <PauseCircle className="w-10 h-10 text-amber-400" />
                </div>
              ) : TERMINAL_STATUSES.includes(jobStatus || '') ? (
                <div className="w-20 h-20 bg-red-500/20 rounded-full flex items-center justify-center mb-6 border-4 border-red-500/30">
                  <XCircle className="w-10 h-10 text-red-400" />
                </div>
              ) : (
                <div className="w-20 h-20 bg-muted rounded-full flex items-center justify-center mb-6">
                  <Loader2 className="w-10 h-10 text-primary animate-spin" />
                </div>
              )}

              <h2 className="text-2xl font-bold mb-2">
                {jobStatus === 'DONE'
                  ? 'Lồng tiếng hoàn tất trên máy chủ'
                  : jobStatus === 'REJECTED_FRAUD'
                    ? 'Kết quả bị từ chối (nghi gian lận thời gian)'
                    : jobStatus === 'TERMINATED_TIMEOUT'
                      ? 'Worker bị hủy do quá thời hạn'
                      : jobStatus === 'POLL_PAUSED'
                        ? 'Tạm dừng theo dõi (máy chủ có thể vẫn đang xử lý)'
                        : jobStatus === 'FAILED' || jobStatus === 'ERROR'
                          ? 'Xử lý thất bại'
                          : 'Đang xử lý trên GPU worker…'}
              </h2>

              <p className="text-muted-foreground max-w-md mb-2 font-mono text-sm">
                Job {jobId} · trạng thái: <b>{jobStatus || '...'}</b>
              </p>

              {/* ETA server ước tính (chỉ khi còn đang xử lý). Trung thực: đây là ước
                  lượng, không phải cam kết — hiển thị cả khi đã vượt, không tô hồng. */}
              {jobEta != null &&
              jobStatus !== 'DONE' &&
              jobStatus !== 'POLL_PAUSED' &&
              !TERMINAL_STATUSES.includes(jobStatus || '') ? (
                <p className="text-muted-foreground/80 text-xs mb-6">
                  Ước tính khoảng ~{jobEta}s (tùy độ dài video &amp; tải máy chủ).
                </p>
              ) : (
                <div className="mb-6" />
              )}

              {/* POLL_PAUSED: cho phép TIẾP TỤC kiểm tra thay vì báo hỏng. */}
              {jobStatus === 'POLL_PAUSED' && (
                <div className="w-full max-w-lg mb-8 space-y-3">
                  <p className="text-sm text-muted-foreground">
                    Đã hết thời gian theo dõi phía ứng dụng, nhưng máy chủ có thể vẫn đang render.
                    Bấm để tiếp tục kiểm tra trạng thái — không cần gửi lại từ đầu.
                  </p>
                  <button
                    onClick={() => jobId && deviceId && void pollJob(jobId, deviceId)}
                    disabled={!jobId || !deviceId || status === 'PROCESSING'}
                    className="w-full py-3 bg-amber-500/90 hover:bg-amber-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl font-bold text-black shadow-lg transition-all active:scale-95 flex items-center justify-center gap-2"
                  >
                    {status === 'PROCESSING' ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" /> ĐANG KIỂM TRA LẠI…
                      </>
                    ) : (
                      <>
                        <ArrowRight className="w-4 h-4" /> TIẾP TỤC KIỂM TRA
                      </>
                    )}
                  </button>
                </div>
              )}

              {jobStatus === 'DONE' && (
                <div className="w-full max-w-lg mb-8 text-left space-y-4">
                  {/* Thông điệp TRUNG THỰC từ worker (đã tách nền hay chưa…). */}
                  {jobResult?.message && (
                    <p className="text-sm text-muted-foreground">{jobResult.message}</p>
                  )}

                  {/* Pipeline THỰC SỰ đã chạy + cờ trung thực — không quảng cáo bước chưa chạy. */}
                  {jobResult && (
                    <div className="rounded-xl border bg-background/60 p-4 space-y-3 text-sm">
                      {jobResult.pipeline?.length ? (
                        <div className="flex flex-wrap gap-1.5">
                          {jobResult.pipeline.map((p) => (
                            <span
                              key={p}
                              className="px-2 py-0.5 rounded-md bg-primary/10 text-primary text-xs font-medium"
                            >
                              {p}
                            </span>
                          ))}
                        </div>
                      ) : null}
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <Flag ok={jobResult.separated} label="Tách nhạc nền (Demucs)" />
                        <Flag ok={jobResult.watermarked} label="Watermark (AudioSeal)" />
                        <div className="text-muted-foreground">
                          Số giọng đã dùng:{' '}
                          <b className="text-foreground">{jobResult.distinct_voices ?? 0}</b>
                        </div>
                        {jobResult.device_used && (
                          <div className="text-muted-foreground">
                            Thiết bị: <b className="text-foreground">{jobResult.device_used}</b>
                          </div>
                        )}
                      </div>
                      {/* Cảnh báo TRUNG THỰC (bước tăng cường thiếu, câu TTS lỗi, căn/cắt lip-sync…). */}
                      {jobResult.notes?.length ? (
                        <ul className="list-disc list-inside space-y-1 text-amber-500/90">
                          {jobResult.notes.map((n, i) => (
                            <li key={i}>{n}</li>
                          ))}
                        </ul>
                      ) : null}
                    </div>
                  )}

                  {/* Lắp ráp cục bộ: tải track lồng tiếng + ghép vào VIDEO GỐC (video không rời máy). */}
                  {!jobResult?.artifactReady ? (
                    <p className="text-xs text-amber-500/90">
                      Máy chủ báo chưa có file kết quả để tải (artifact chưa sẵn sàng).
                    </p>
                  ) : !isTauri() ? (
                    <p className="text-xs text-amber-500/90">
                      Việc ghép audio vào video cần <b>ứng dụng desktop OmniVoice</b> (có ffmpeg đi kèm).
                      Trình duyệt web không truy cập được video gốc của bạn.
                    </p>
                  ) : savedPath ? (
                    <div className="flex items-start gap-2 text-primary text-sm">
                      <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
                      <span className="break-all">
                        Đã lưu video lồng tiếng: <b>{savedPath}</b>
                      </span>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <button
                        onClick={
                          outputVideoPath ? () => saveAssembled(outputVideoPath) : downloadAndMux
                        }
                        disabled={assembling || !videoPath}
                        className="w-full py-3 bg-primary hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl font-bold text-primary-foreground shadow-lg transition-all active:scale-95 flex items-center justify-center gap-2"
                      >
                        {assembling ? (
                          <>
                            <Loader2 className="w-4 h-4 animate-spin" /> ĐANG TẢI &amp; GHÉP CỤC BỘ…
                          </>
                        ) : outputVideoPath ? (
                          <>
                            <Save className="w-4 h-4" /> LƯU VIDEO LỒNG TIẾNG
                          </>
                        ) : (
                          <>
                            <Download className="w-4 h-4" /> TẢI TRACK &amp; GHÉP VÀO VIDEO
                          </>
                        )}
                      </button>
                      {assembleError && <p className="text-xs text-red-400">{assembleError}</p>}
                      {!videoPath && (
                        <p className="text-xs text-amber-500/90">
                          Không còn video gốc trong phiên này — hãy chọn lại video ở bước 1 để ghép.
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )}

              <button
                onClick={resetAll}
                className="px-8 py-3 bg-secondary hover:bg-secondary/80 text-secondary-foreground rounded-xl font-medium transition-colors"
              >
                Làm video khác
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
