import os
import json
import re
import math
import logging
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field, ValidationError

from src.timecode import to_seconds

# ZERO-LOGGING (tiêu chí #2): logger ở mức WARNING và TUYỆT ĐỐI không đưa prompt /
# văn bản gốc / bản dịch vào bất kỳ mức log nào. Chỉ log METADATA (đếm segment, số
# lần thử, TÊN loại lỗi). KHÔNG bao giờ log/raise str(exception) của parse: message
# của json.JSONDecodeError / pydantic.ValidationError có thể nhúng nguyên đoạn kịch
# bản mà mô hình sinh ra -> rò rỉ plaintext ra log hoặc response.
logger = logging.getLogger("omnivoice.translation")
logger.setLevel(logging.WARNING)

# 6 nhãn cảm xúc HỢP LỆ (TRANSLATION_RULES.md §2.2). Hạ nguồn (model_manager
# EMOTION_PROSODY) tra cứu bằng CHỮ HOA; ta chuẩn hóa .upper() + whitelist tại đây để
# một nhãn lệch hoa/thường từ mô hình cục bộ không ÂM THẦM rơi về NEUTRAL (fake-success).
VALID_EMOTIONS = frozenset(
    {"NEUTRAL", "HAPPY", "ANGRY", "SAD", "WHISPERING", "SHOUTING"}
)

# Số lần thử tối đa khi mô hình trả JSON/không đúng schema (TRANSLATION_RULES.md §4).
MAX_TRANSLATION_ATTEMPTS = 3

# ── SELF-REVIEW (Qwen TỰ CHẤM ĐIỂM + TỰ CHỈNH SỬA) ──────────────────────────────────
# Sau khi vòng dịch ban đầu cho ra bản HỢP LỆ, Qwen tự soi lại bản dịch của chính mình
# và viết gọn lại những câu LỆCH NHỊP. Cơ chế dùng HAI tín hiệu TÁCH BẠCH (No-Fake-Success):
#   (A) KHÁCH QUAN, deterministic, không-model = _pacing_penalty (đếm âm tiết). Vừa là
#       CỔNG chọn câu cần sửa, vừa là LUẬT CHẤP NHẬN "chỉ nhận nếu ĐO ĐƯỢC là tốt hơn".
#   (B) MODEL SELF-SCORE = Qwen tự chấm 1-5 + đề xuất bản viết lại (qua seam _generate).
#       ADVISORY THUẦN: định hướng bản sửa; KHÔNG bao giờ là cổng chấp nhận, KHÔNG log.
MAX_REVIEW_ROUNDS = 2          # trần cứng số vòng review (ENV QWEN_MAX_REVIEW_ROUNDS; <=0 tắt)
# Bất đối xứng có chủ đích: TRÀN âm tiết (audio tràn ra ngoài cảnh) là lỗi lip-sync NẶNG
# NHẤT (§2.1 "bằng đúng HOẶC nhỏ hơn") nên phạt gấp đôi HỤT (dead air, nhẹ hơn).
PACING_OVERFLOW_WEIGHT = 2
PACING_UNDERFLOW_WEIGHT = 1
PACING_TOLERANCE = 2          # candidate iff _pacing_penalty(current) > 2 (câu khớp nhịp: bỏ qua)
REVIEW_MIN_SYLLABLE_RATIO = 0.5   # sàn chống cắt cụt: từ chối bản có got < 0.5*orig
MIN_REVISED_SYLLABLES = 2     # sàn TUYỆT ĐỐI: khi orig>=2, không nhận bản cụt về 1 từ
                              # (0.5*orig <= 1 với orig<=2 nên ratio một mình KHÔNG cắn)

# Ngôn ngữ mà count_syllables cho tín hiệu THẬT: vi = đếm từ; Latinh = đếm cụm nguyên âm
# (xấp xỉ). CJK/Arab/Cyrillic/Thai... chạy qua nhánh vowel-group -> [aeiouy] không khớp ->
# suy biến về 1 cho MỌI câu. Gate self-review trên tín hiệu suy biến sẽ "đo cải thiện" so
# với baseline GIẢ -> cắt cụt bản dịch (fake-success). Nhận CẢ mã ISO LẪN tên tiếng Anh vì
# target tới dạng TÊN ("Vietnamese") còn source tới dạng MÃ ("vi"/"en"/"ja").
# QUAN TRỌNG: original_syllables được đếm ở phía SOURCE, nên PHẢI kiểm cả source lẫn target
# đo được — nếu chỉ chặn target CJK mà bỏ sót source CJK, baseline penalty là giả -> cổng
# biến thành động cơ cắt cụt bản dịch về 1 âm tiết (lỗ hổng do 2 verifier độc lập phát hiện).
_PACING_MEASURABLE_LANGS = frozenset({
    "vi", "vie", "vietnamese",
    "en", "eng", "english",
    "fr", "fra", "fre", "french",
    "es", "spa", "spanish",
    "de", "deu", "ger", "german",
    "it", "ita", "italian",
    "pt", "por", "portuguese",
    "nl", "nld", "dut", "dutch",
    "ro", "ron", "rum", "romanian",
    "id", "ind", "indonesian",
    "ms", "msa", "may", "malay",
    "ca", "cat", "catalan",
})


class TranslatedSegment(BaseModel):
    id: int | str
    start: float | str
    end: float | str
    duration: float
    original_text: str
    original_syllables: int
    translated_text: str = Field(description="The translated text matching the pacing and style.")
    # translated_syllables/pacing_status ĐÃ BỎ (P4): hạ nguồn không đọc; enum cũ lệch
    # shared-types. LLM vẫn phải tự căn số âm tiết theo prompt LIP-SYNC, chỉ không báo lại.
    emotion: str = Field(description="One of: NEUTRAL, HAPPY, ANGRY, SAD, WHISPERING, SHOUTING")
    speaker_id: str


class TranslationResult(BaseModel):
    segments: List[TranslatedSegment]


class _LlmSegment(BaseModel):
    """Schema TỐI THIỂU mà MÔ HÌNH được phép điền.

    Ta KHÔNG tin mô hình cục bộ echo lại start/end/duration/speaker_id: một mô hình 4B
    dễ renumber id hoặc làm hỏng số thực -> đặt sai vị trí clip lồng tiếng (start/end là
    trọng yếu cho bước mix). Mô hình chỉ điền id (để ghép) + translated_text + emotion;
    mọi mốc thời gian được GHÉP LẠI từ processed_segments đã tin cậy (xem _merge)."""
    id: int | str
    translated_text: str
    emotion: str


class _ReviewItem(BaseModel):
    """Schema TỐI THIỂU mô hình được phép điền ở VÒNG SELF-REVIEW (mirror _LlmSegment).

    KHÔNG khai báo start/end/duration/speaker_id -> Zero-Trust: vòng review CHỈ được đổi
    translated_text + emotion; mọi mốc thời gian ghép từ TranslatedSegment tin cậy, không
    bao giờ từ echo review (pydantic mặc định BỎ QUA field thừa nên echo start/speaker bậy
    bị vứt). Các trường Optional -> row thiếu field degrade thành NO-OP (giữ bản cũ) thay
    vì đánh sập cả vòng."""
    id: int | str
    score: Optional[int] = None
    revised_text: Optional[str] = None
    emotion: Optional[str] = None


def count_syllables(text: str, lang: str = "en") -> int:
    """
    Very basic syllable counter for calculating pacing.
    Vietnamese is typically 1 word = 1 syllable (separated by spaces).
    English uses a rough vowel group regex.
    """
    if not text:
        return 0
    text = text.strip()
    if lang.lower().startswith("vi"):
        # Vietnamese is mostly monosyllabic, counting spaces is usually sufficient
        return len(text.split())
    else:
        # Rough English syllable count (vowel groups)
        text = text.lower()
        # Remove trailing e
        text = re.sub(r'e\b', '', text)
        vowel_groups = re.findall(r'[aeiouy]+', text)
        return max(1, len(vowel_groups))


class TranslationService:
    """Dịch thoại bằng LLM Qwen CHẠY CỤC BỘ, thường trú VRAM — KHÔNG gọi API cloud.

    Vì sao bỏ OpenAI: gửi transcript (original_text từng câu) tới API cloud là hành vi
    RÒ RỈ plaintext (tiêu chí #2 Zero-Logging/Zero-Trust) — đúng lớp rò rỉ dự án đã gắn
    cờ cho edge-tts (RH3). Chạy Qwen cục bộ khiến transcript KHÔNG bao giờ rời máy; và
    không còn `openai` client object có thể nhúng payload request vào exception.

    Thiết kế phản chiếu asr_service (tiêu chí #6 GPU Model Residence):
      - __init__ NHẸ: KHÔNG import transformers, KHÔNG nạp trọng số, KHÔNG chạm CUDA —
        để `from src.translation_service import ...` và việc thu thập test KHÔNG nổ trên
        máy CPU-only thiếu thư viện.
      - load_model(): nạp Qwen vào VRAM MỘT LẦN (gọi từ ModelManager.load_all_models,
        cạnh Whisper), fail-closed nếu thiếu GPU/transformers/trọng số.
      - translate_segments(): fail-closed (raise) khi chưa nạp; retry tối đa 3; ghép kết
        quả lên mốc thời gian tin cậy; TUYỆT ĐỐI không trả kết quả một phần/giả.
    """

    def __init__(self, model_id: Optional[str] = None):
        # Cấu hình đọc lười từ ENV. __init__ PHẢI an toàn để import: không transformers,
        # không nạp trọng số, không CUDA — mọi việc nặng nằm trong load_model().
        self.model_id = model_id or os.environ.get(
            "QWEN_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507"
        )
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        # Observability self-review — ĐẾM-ONLY (không plaintext): {'revised','rounds','skipped'}.
        # import-safe (không đụng transformers/VRAM); giữ nguyên bất biến import của test.
        self.last_review_stats: Optional[Dict[str, Any]] = None

    def load_model(self):
        """Nạp Qwen vào VRAM MỘT LẦN (gọi từ ModelManager.load_all_models).

        Fail-closed đúng như asr_service:
          - thiếu transformers -> RuntimeError (No-Fake-Success, không mock/không giả);
          - KHÔNG có CUDA -> RuntimeError: Qwen 4B trên CPU là bất khả dụng (giây→phút mỗi
            câu) và RAM CPU không phải VRAM; nạp im lặng lên CPU sẽ quảng cáo "thường trú"
            GIẢ, vi phạm tiêu chí #6 + No-Fake-Success.
        """
        if self.is_loaded:
            return

        # Chặn mọi cuộc gọi mạng (kể cả tải metadata model) — suy luận phải HOÀN TOÀN
        # offline: đây là điều kiện tiên quyết của quyền riêng tư (không phone-home).
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        logger.info("Đang nạp mô hình dịch Qwen cục bộ vào VRAM...")
        try:
            import torch
            # import LƯỜI: transformers KHÔNG được import ở top-level/__init__, nếu không
            # `from src.translation_service import ...` sẽ nổ ImportError khi thu thập test
            # trên máy thiếu thư viện (import-safety).
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not torch.cuda.is_available():
                # Qwen phải thường trú VRAM (tiêu chí #6). CPU = fail-closed, không giả.
                raise RuntimeError(
                    "Qwen yêu cầu GPU CUDA để thường trú VRAM; máy này không có CUDA. "
                    "Dịch fail-closed thay vì chạy giả trên CPU."
                )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                device_map="cuda",
            )
            self.is_loaded = True
            logger.info("Nạp Qwen dịch cục bộ thành công (VRAM).")
        except ImportError as e:
            # Fail-closed: thiếu thư viện thì KHÔNG được coi là "đã nạp".
            raise RuntimeError(
                "transformers chưa được cài — không thể nạp mô hình dịch Qwen cục bộ. "
                "Cài extra:  uv sync --extra local-llm"
            ) from e
        except Exception as e:
            # Fail-closed: trọng số/CUDA không sẵn sàng thì để lỗi nổ ra; chỉ log tên loại.
            logger.error(f"Nạp Qwen thất bại: {type(e).__name__}")
            raise RuntimeError("Không thể nạp mô hình dịch Qwen cục bộ.") from e

    def _generate(self, prompt: str) -> str:
        """Seam DUY NHẤT chạm mô hình: chat template (non-thinking) -> generate -> decode.

        Test CPU-only patch chính hàm này (side_effect JSON đóng hộp) — y hệt cách test
        cũ mock `beta.chat.completions.parse` — nên mọi nhánh xây prompt / parse / retry /
        fail-closed đều kiểm thử được KHÔNG cần GPU hay trọng số.
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("Qwen translation model chưa được nạp!")

        import torch

        messages = [{"role": "user", "content": prompt}]
        # enable_thinking=False BẮT BUỘC: chế độ 'thinking' mặc định của Qwen3 phun chuỗi
        # <think> làm tăng độ trễ per-segment và chèn token phi-JSON phá ràng buộc JSON.
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            # Greedy (do_sample=False) cho JSON xác định; max_new_tokens cấu hình qua ENV
            # cho lô nhiều segment. Guided decoding (xgrammar/outlines) là lever CHẤT LƯỢNG
            # ở prod — hardware-blocked ở đây; retry-max-3 mới là bảo chứng đúng đắn.
            generated = self.model.generate(
                **inputs,
                max_new_tokens=int(os.environ.get("QWEN_MAX_NEW_TOKENS", "4096")),
                do_sample=False,
            )
        out_tokens = generated[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(out_tokens, skip_special_tokens=True)

    def translate_segments(self, segments: List[Dict[str, Any]], target_language: str, style: str,
                           source_language: str = "en") -> List[Dict[str, Any]]:
        """
        Dịch các segment bằng Qwen cục bộ, đảm bảo Lip-sync + Emotion tagging.

        source_language: ngôn ngữ GỐC của thoại, dùng để đếm âm tiết đúng cách. Bản cũ
        ghim cứng "en" nên video tiếng Việt (đơn âm tiết) bị đếm sai bằng regex nguyên
        âm tiếng Anh -> ước lượng pacing lệch. Nhánh "vi" trong count_syllables được gọi ở đây.
        """
        # Chuẩn bị dữ liệu đầu vào + tính âm tiết gốc (backend-agnostic; luôn chạy TRƯỚC
        # cả cổng fail-closed để việc căn âm tiết theo source_language không bị bỏ qua).
        processed_segments = []
        for seg in segments:
            orig_text = seg.get("text", seg.get("original_text", ""))
            orig_syllables = count_syllables(orig_text, lang=source_language)

            # Tính duration nếu chưa có. start/end được CHUẨN HÓA về giây qua src.timecode
            # (CC-1): client PHẢI gửi số giây, nhưng nếu lỡ tới ở dạng "HH:MM:SS" thì
            # float() trần sẽ nổ -> rơi fallback -> pacing sai âm thầm. to_seconds xử lý cả hai.
            start_s = to_seconds(seg.get("start", 0))
            end_s = to_seconds(seg.get("end", 0))
            duration = seg.get("duration")
            if duration is None:
                duration = max(0.1, end_s - start_s)

            processed_segments.append({
                "id": seg.get("id", ""),
                "start": start_s,
                "end": end_s,
                "duration": round(duration, 2),
                "original_text": orig_text,
                "original_syllables": orig_syllables,
                "speaker_id": seg.get("speaker", seg.get("speaker_id", "SPEAKER_UNKNOWN")),
            })

        # Fail-closed TRƯỚC khi chạm mô hình: chưa nạp Qwen -> raise (trigger đổi từ "thiếu
        # OPENAI_API_KEY" sang "mô hình chưa nạp"). _generate KHÔNG được gọi ở nhánh này.
        if not self.is_loaded:
            raise RuntimeError(
                "Mô hình dịch Qwen chưa được nạp lên VRAM "
                "(load_model chưa chạy hoặc fail-closed trên máy không GPU)."
            )

        prompt = self._build_prompt(target_language, style, processed_segments)

        # Auto-Retry tối đa 3 (TRANSLATION_RULES.md §4). "Không hợp lệ" = JSON hỏng HOẶC
        # pydantic ValidationError HOẶC sai tập id/số lượng segment. Emotion lệch enum
        # KHÔNG tính là invalid (xem _merge: quy về NEUTRAL + log) vì nó chỉ là gợi ý ngữ
        # điệu thứ yếu mà hạ nguồn đã map an toàn — đánh sập cả lô vì một nhãn lệch sẽ hạ
        # độ tin cậy với mô hình nhỏ mà gần như không thêm tính đúng đắn.
        last_error_name: Optional[str] = None
        for attempt in range(1, MAX_TRANSLATION_ATTEMPTS + 1):
            raw = self._generate(prompt)
            try:
                llm_segments = self._parse_and_validate(raw, processed_segments)
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                # ZERO-LOGGING: chỉ tên loại lỗi + metadata; KHÔNG str(e) (có thể nhúng
                # nguyên đoạn output mô hình sinh ra).
                last_error_name = type(e).__name__
                logger.warning(
                    f"Qwen output không hợp lệ "
                    f"(lần {attempt}/{MAX_TRANSLATION_ATTEMPTS}): {last_error_name}"
                )
                continue

            merged = self._merge(llm_segments, processed_segments)
            logger.info(f"Đã dịch {len(merged)} segment (lần {attempt}).")

            # SELF-REVIEW (best-effort, FAIL-OPEN): Qwen tự chấm điểm + tự chỉnh sửa những
            # câu lệch nhịp. KHÁC vòng dịch ban đầu (fail-CLOSED/raise): một enhancement tùy
            # chọn KHÔNG BAO GIỜ được đánh sập bản dịch ĐÃ hợp lệ -> mọi lỗi bất ngờ đều nuốt,
            # giữ baseline. ZERO-LOGGING: chỉ TÊN loại lỗi (str(e) có thể nhúng plaintext).
            if self._self_review_enabled():
                try:
                    merged = self._self_review(merged, target_language, source_language, style)
                except Exception as e:
                    logger.warning(f"Self-review bỏ qua: {type(e).__name__}")
            return [seg.model_dump() for seg in merged]

        # Hết 3 lần vẫn hỏng -> fail-closed. KHÔNG bịa, KHÔNG trả một phần (No-Fake-Success).
        # Message chỉ chứa TÊN loại lỗi, không nhúng plaintext -> an toàn cho log/response.
        raise RuntimeError(
            f"Dịch Qwen thất bại sau {MAX_TRANSLATION_ATTEMPTS} lần thử "
            f"(lỗi cuối: {last_error_name})."
        )

    def _build_prompt(self, target_language: str, style: str,
                      processed_segments: List[Dict[str, Any]]) -> str:
        """Xây prompt CÓ NHÚNG schema đầu ra + enum cảm xúc.

        Bản OpenAI lấy hình dạng output từ response_format=TranslationResult (server-side).
        Qwen prompt-only KHÔNG biết phải trả trường gì nếu ta không nói -> phải nhúng schema
        rõ ràng, nếu không json.loads có thể chạy nhưng pydantic/khớp id sẽ hỏng (migration-break).
        Ta chỉ yêu cầu schema TỐI THIỂU {id, translated_text, emotion} để giảm token + tránh
        mô hình làm hỏng mốc thời gian; mọi trường mốc thời gian ghép lại từ đầu vào tin cậy.
        """
        # Chỉ đưa các trường mô hình CẦN để dịch (giấu start/end khỏi output của mô hình).
        model_inputs = [
            {
                "id": p["id"],
                "duration": p["duration"],
                "original_syllables": p["original_syllables"],
                "speaker_id": p["speaker_id"],
                "original_text": p["original_text"],
            }
            for p in processed_segments
        ]
        input_json = json.dumps({"input_segments": model_inputs}, ensure_ascii=False, indent=2)

        return (
            "You are a professional dubbing translator for movies and videos.\n"
            f"Target language: {target_language}.\n"
            f"Style/Tone: {style}.\n\n"
            "CRITICAL RULES:\n"
            "1. LIP-SYNC (Pacing): each input segment gives `duration` (seconds) and "
            "`original_syllables`. Keep the translated syllable count strictly close to "
            "`original_syllables` to match pacing. If the text is too long for the "
            "duration, you MUST paraphrase or summarize.\n"
            "2. EMOTION TAGGING: assign exactly one emotion per segment, chosen ONLY from "
            "this enum: NEUTRAL, HAPPY, ANGRY, SAD, WHISPERING, SHOUTING.\n"
            "3. STRICT JSON: respond with ONLY a JSON object of EXACTLY this shape — no "
            "prose, no markdown, no code fences, no commentary:\n"
            '   {"segments": [{"id": <same id as input>, "translated_text": "<translation>", '
            '"emotion": "<ONE OF THE ENUM>"}]}\n'
            "   Return exactly one object per input segment, preserving each `id` EXACTLY. "
            "Do NOT add or drop segments. Do NOT include start/end/duration/original_text "
            "in your output.\n\n"
            f"INPUT:\n{input_json}\n"
        )

    @staticmethod
    def _extract_json(raw: str) -> str:
        """Lột code fence / văn xuôi quanh JSON (mô hình nhỏ hay bọc ```json ... ```).

        Nếu không tìm được khối JSON hợp lệ, trả chuỗi để json.loads NỔ -> kích hoạt retry
        rồi fail-closed. KHÔNG "sửa" theo kiểu che giấu lỗi.
        """
        s = (raw or "").strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[1] if "\n" in s else ""
            fence = s.rfind("```")
            if fence != -1:
                s = s[:fence]
            s = s.strip()
        if not s.startswith("{"):
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                s = s[start:end + 1]
        return s.strip()

    def _parse_and_validate(self, raw: str, processed_segments: List[Dict[str, Any]]) -> List["_LlmSegment"]:
        """Parse output mô hình + kiểm bất biến chịu tải. Ném lỗi (→retry) nếu vi phạm."""
        data = json.loads(self._extract_json(raw))  # json.JSONDecodeError -> retry
        if isinstance(data, dict):
            seg_list = data.get("segments", [])
        elif isinstance(data, list):
            seg_list = data
        else:
            raise ValueError("Qwen JSON không phải object/array.")

        llm_segments = [_LlmSegment(**s) for s in seg_list]  # ValidationError -> retry

        # Bất biến ID-parity: mô hình phải trả ĐÚNG tập id đầu vào (không thêm/bớt/renumber).
        # start/end được ghép từ đầu vào theo id, nên id sai = đặt sai câu -> phải retry.
        in_ids = sorted(str(p["id"]) for p in processed_segments)
        out_ids = sorted(str(s.id) for s in llm_segments)
        if in_ids != out_ids:
            raise ValueError("Qwen trả sai tập id / số lượng segment so với đầu vào.")
        return llm_segments

    def _merge(self, llm_segments: List["_LlmSegment"],
               processed_segments: List[Dict[str, Any]]) -> List[TranslatedSegment]:
        """Ghép translated_text + emotion (từ mô hình) lên mốc thời gian TIN CẬY từ đầu vào.

        Đây là phòng thủ chống mô hình nhỏ làm hỏng start/end/id/speaker_id: hợp đồng 9
        trường trả về (model_dump) giữ nguyên cho process_job/TTS, nhưng mọi mốc thời gian
        lấy từ processed_segments — không bao giờ từ echo của LLM."""
        by_id = {str(s.id): s for s in llm_segments}
        coerced = 0
        merged: List[TranslatedSegment] = []
        for p in processed_segments:
            llm = by_id[str(p["id"])]  # đảm bảo tồn tại bởi kiểm ID-parity ở trên
            emotion = (llm.emotion or "").strip().upper()
            if emotion not in VALID_EMOTIONS:
                # KHÔNG im lặng: đếm + log metadata (không lộ text). Quy về NEUTRAL vì hạ
                # nguồn map unknown->NEUTRAL an toàn; emotion là gợi ý ngữ điệu thứ yếu.
                coerced += 1
                emotion = "NEUTRAL"
            merged.append(TranslatedSegment(
                id=p["id"],
                start=p["start"],
                end=p["end"],
                duration=p["duration"],
                original_text=p["original_text"],
                original_syllables=p["original_syllables"],
                translated_text=llm.translated_text,
                emotion=emotion,
                speaker_id=p["speaker_id"],
            ))
        if coerced:
            logger.warning(f"{coerced} nhãn cảm xúc ngoài enum bị quy về NEUTRAL.")
        return merged

    # ── SELF-REVIEW: Qwen tự chấm điểm + tự chỉnh sửa (best-effort, FAIL-OPEN) ─────────

    def _self_review_enabled(self) -> bool:
        """Kill-switch prod (ENV QWEN_SELF_REVIEW, MẶC ĐỊNH BẬT — user 'cần cơ chế' này).
        '0'/'false'/'no'/'off' -> tắt hoàn toàn (không thêm lần _generate nào)."""
        v = os.environ.get("QWEN_SELF_REVIEW", "1").strip().lower()
        return v in {"1", "true", "yes", "on"}

    @staticmethod
    def _pacing_measurable(language: str) -> bool:
        """count_syllables có cho tín hiệu THẬT với ngôn ngữ này không?

        vi = đếm từ (thật); Latinh = đếm cụm nguyên âm (xấp xỉ thật). CJK/Arab/Cyrillic...
        suy biến về 1 cho mọi câu -> gate self-review trên đó là gate trên tín hiệu GIẢ.
        Chuẩn hóa primary subtag ('en-US'/'pt_BR' -> 'en'/'pt') rồi tra tập mã+tên."""
        key = (language or "").strip().lower()
        key = key.split("-")[0].split("_")[0]
        return key in _PACING_MEASURABLE_LANGS

    def _pacing_penalty(self, translated_text: str, original_syllables: int,
                        target_language: str) -> int:
        """Điểm phạt lệch nhịp >= 0 (0 = khớp hoàn hảo). TRÀN phạt gấp đôi HỤT.

        Hàm THUẦN (str,int,str)->int: gọi trực tiếp trên CPU, KHÔNG cần mock. Đây là tín
        hiệu DUY NHẤT quyết định (a) câu nào là candidate và (b) chấp nhận/không một bản sửa
        -> 'cải thiện' luôn là một phép giảm số nguyên ĐO ĐƯỢC, không phải lời tuyên bố."""
        got = count_syllables(translated_text, target_language)
        diff = got - original_syllables
        if diff > 0:
            return PACING_OVERFLOW_WEIGHT * diff
        return PACING_UNDERFLOW_WEIGHT * (-diff)

    @staticmethod
    def _coerce_emotion(raw: Optional[str], fallback: str) -> str:
        """Chuẩn hóa emotion do review đề xuất. Ngoài enum -> GIỮ emotion HỢP LỆ hiện tại
        (fallback), KHÔNG tụt về NEUTRAL: bản dịch đầu đã chốt emotion hợp lệ, review chỉ
        được NÂNG CẤP khi chắc chắn — không được hạ cấp một nhãn đang đúng."""
        e = (raw or "").strip().upper()
        return e if e in VALID_EMOTIONS else fallback

    def _build_review_prompt(self, target_language: str, style: str,
                             review_items: List[Dict[str, Any]]) -> str:
        """Prompt TỰ-SOI, CHẠY CỤC BỘ (chỉ tới _generate — KHÔNG cloud, KHÔNG log).

        Nhúng bản dịch hiện tại + số liệu pacing để Qwen tự chấm rồi viết gọn lại. Giấu
        start/end khỏi mô hình; review chỉ được đổi text/emotion (mốc thời gian ghép từ đầu
        vào tin cậy). Giữ review trên seam _generate CỤC BỘ là CHỊU-TẢI cho Zero-Egress:
        route review sang cloud sẽ tái mở lỗ rò transcript mà dự án đã bỏ OpenAI để đóng."""
        input_json = json.dumps(
            {"review_segments": review_items}, ensure_ascii=False, indent=2
        )
        return (
            "You are a strict senior dubbing-translation reviewer.\n"
            f"Target language: {target_language}.\n"
            f"Style/Tone: {style}.\n\n"
            "For EACH segment below:\n"
            "1. SELF-SCORE it 1-5 (1=poor, 5=perfect) combining: (a) fidelity of meaning to "
            "`original_text`; (b) lip-sync pacing — compare `current_syllables` to "
            "`original_syllables`; (c) emotion/style fit.\n"
            "2. If pacing is off, REWRITE `revised_text` so its syllable count moves toward "
            "`original_syllables` (PREFER equal-or-fewer when it overflows) WITHOUT losing "
            "meaning or breaking the requested register. If it is already good, repeat the "
            "current text unchanged.\n"
            "3. You MAY set `emotion` from the enum ONLY if the current tag is clearly wrong.\n"
            "EMOTION enum: NEUTRAL, HAPPY, ANGRY, SAD, WHISPERING, SHOUTING.\n"
            "STRICT JSON only — no prose, no markdown, no code fences — EXACTLY this shape:\n"
            '   {"reviews": [{"id": <same id as input>, "score": <1-5>, '
            '"revised_text": "<text>", "emotion": "<ONE OF THE ENUM>"}]}\n'
            "Keep each `id` EXACTLY; do NOT add/drop/renumber; do NOT output "
            "start/end/duration/speaker_id (not yours to set).\n\n"
            f"INPUT:\n{input_json}\n"
        )

    def _parse_review(self, raw: str, candidate_ids: set) -> Dict[str, "_ReviewItem"]:
        """Parse output vòng review -> {id: _ReviewItem} cho các id là candidate.

        ZERO-LOGGING: message của JSONDecodeError/ValidationError có thể nhúng NGUYÊN đoạn
        echo của mô hình -> mọi raise ở đây dùng chuỗi TĨNH (không nhét raw/str(e)); caller
        chỉ log type(e).__name__."""
        data = json.loads(self._extract_json(raw))  # JSONDecodeError -> caller (FAIL-OPEN)
        if isinstance(data, dict):
            rows = data.get("reviews", [])
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError("review JSON không phải object/array.")  # TĨNH: không nhúng raw
        out: Dict[str, "_ReviewItem"] = {}
        for r in rows:
            if not isinstance(r, dict):
                continue  # row không phải object -> no-op (không đánh sập cả vòng)
            item = _ReviewItem(**r)  # ValidationError -> caller (FAIL-OPEN)
            rid = str(item.id)
            if rid in candidate_ids and rid not in out:
                out[rid] = item  # bỏ id lạ (không phải candidate); first-wins nếu trùng
        return out

    def _self_review(self, merged: List[TranslatedSegment], target_language: str,
                     source_language: str, style: str) -> List[TranslatedSegment]:
        """Vòng tự-chấm + tự-sửa BOUNDED. Duyệt TOÀN BỘ danh sách, chỉ thay TẠI CHỖ những
        candidate được cổng khách quan CHẤP NHẬN qua model_copy(update={text,emotion}) ->
        9-field contract + ID-parity + merge-onto-trusted-timings BẤT BIẾN mọi vòng, kể cả
        khi review echo bậy.

        GATE ĐO LƯỜNG (No-Fake-Success): chỉ chạy khi CẢ target LẪN source đo được. Vì sao
        CẢ HAI: original_syllables đếm ở phía SOURCE — với source CJK (native script),
        count_syllables suy biến về 1 cho mọi câu -> baseline penalty GIẢ -> cổng sẽ 'cải
        thiện' bằng cách cắt cụt bản dịch về 1 từ. Bỏ qua khi đầu nào không đo được.

        FAIL-OPEN: mọi lỗi parse/JSON -> break, giữ current (đã hợp lệ), KHÔNG raise."""
        self.last_review_stats = {"revised": 0, "rounds": 0, "skipped": None}

        if not (self._pacing_measurable(target_language)
                and self._pacing_measurable(source_language)):
            self.last_review_stats["skipped"] = "lang_not_measurable"
            return merged

        max_rounds = int(os.environ.get("QWEN_MAX_REVIEW_ROUNDS", str(MAX_REVIEW_ROUNDS)))
        if max_rounds <= 0:
            self.last_review_stats["skipped"] = "rounds_disabled"
            return merged

        current = list(merged)
        total_revised = 0
        rounds_run = 0
        for _ in range(max_rounds):
            # (a) Chọn candidate bằng CỔNG KHÁCH QUAN (pacing penalty > tolerance).
            candidates = []
            for idx, seg in enumerate(current):
                if seg.original_syllables <= 0:
                    continue  # không tính được ratio -> bỏ (an toàn)
                pen = self._pacing_penalty(
                    seg.translated_text, seg.original_syllables, target_language
                )
                if pen > PACING_TOLERANCE:
                    candidates.append((idx, seg, pen))
            if not candidates:
                break  # HỘI TỤ: mọi câu đã trong dung sai nhịp
            rounds_run += 1

            # (b) CHỈ gửi candidate cho Qwen tự chấm + viết lại (câu đạt nhịp: bỏ qua ->
            #     trường hợp tốt = 0 lần _generate thêm). Giấu start/end khỏi mô hình.
            review_items = [
                {
                    "id": seg.id,
                    "duration": seg.duration,
                    "original_syllables": seg.original_syllables,
                    "current_syllables": count_syllables(seg.translated_text, target_language),
                    "current_emotion": seg.emotion,
                    "original_text": seg.original_text,
                    "current_translation": seg.translated_text,
                }
                for _, seg, _ in candidates
            ]
            raw = self._generate(self._build_review_prompt(target_language, style, review_items))
            try:
                reviews = self._parse_review(raw, {str(s.id) for _, s, _ in candidates})
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                # FAIL-OPEN: review hỏng -> giữ current (đã hợp lệ). ZERO-LOGGING: chỉ tên lỗi.
                logger.warning(
                    f"Self-review output không hợp lệ (vòng {rounds_run}): {type(e).__name__}"
                )
                break

            # (c) Áp bản sửa theo LUẬT CHẤP NHẬN khách quan (chỉ nhận nếu tốt hơn NGẶT).
            progress = 0
            for idx, seg, old_pen in candidates:
                rv = reviews.get(str(seg.id))
                if rv is None or not rv.revised_text:
                    continue
                new_text = rv.revised_text.strip()
                if not new_text:
                    continue
                orig = seg.original_syllables
                new_syll = count_syllables(new_text, target_language)
                # Sàn chống cắt cụt: từ chối bản quá ngắn DÙ penalty thấp. Sàn tuyệt đối
                # MIN_REVISED_SYLLABLES chặn 'gutting về 1 từ' khi orig >= 2 (0.5*orig <= 1
                # với orig <= 2 nên ratio một mình KHÔNG cắn).
                floor = math.ceil(REVIEW_MIN_SYLLABLE_RATIO * orig)
                if orig >= MIN_REVISED_SYLLABLES:
                    floor = max(floor, MIN_REVISED_SYLLABLES)
                if new_syll < floor:
                    continue
                # CHỈ nhận khi ĐO ĐƯỢC tốt hơn NGẶT ('<'): strict-better ⊂ not-worse nên luôn
                # thỏa "không tệ hơn"; đồng thời chống churn (dao động cùng-penalty) + không
                # áp một hoán-đổi cùng-penalty không kiểm chứng được (rủi ro hồi quy nghĩa).
                if self._pacing_penalty(new_text, orig, target_language) >= old_pen:
                    continue
                new_emotion = self._coerce_emotion(rv.emotion, seg.emotion)
                current[idx] = seg.model_copy(
                    update={"translated_text": new_text, "emotion": new_emotion}
                )
                progress += 1
                total_revised += 1
            if progress == 0:
                break  # HỘI TỤ: không có cải thiện ngặt nào ở vòng này

        if total_revised:
            # ĐẾM-ONLY (không plaintext): số câu ĐỔI dưới cổng khách quan ngặt.
            logger.info(f"Self-review: tinh chỉnh {total_revised} câu qua {rounds_run} vòng.")
        self.last_review_stats = {"revised": total_revised, "rounds": rounds_run, "skipped": None}
        return current


# Expose a singleton instance. __init__ AN TOÀN để import (không transformers/không VRAM);
# việc nạp trọng số do ModelManager.load_all_models() gọi load_model() một lần.
translation_service = TranslationService()
