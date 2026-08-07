import os
import json
import re
import math
import logging
from typing import List, Dict, Any, Optional, Literal

from pydantic import BaseModel, Field, ValidationError

from src.timecode import to_seconds
from src.translation_quality import (
    estimate_spoken_duration_seconds,
    quality_gate_metadata,
    quality_gate_segments,
    normalize_translation_text,
    score_translation,
    speech_unit_count,
    timing_overflow_penalty,
)
from src.qwen_prompt_profiles import build_qwen_system_prompt

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

# Một request hợp lệ có thể chứa tới hàng nghìn segment, trong khi Qwen chỉ được phép sinh
# QWEN_MAX_NEW_TOKENS (mặc định 4096). Chia lô giữ mỗi lần generate trong biên hữu hạn.
# Giới hạn bytes áp lên TOÀN prompt đã serialize, không chỉ original_text: UTF-8 bytes là
# proxy token bảo thủ hơn len(str) cho CJK/emoji và còn tính cả id/speaker/JSON overhead.
DEFAULT_TRANSLATION_BATCH_MAX_SEGMENTS = 32
DEFAULT_TRANSLATION_BATCH_MAX_INPUT_BYTES = 12 * 1024
DEFAULT_REVIEW_BATCH_MAX_SEGMENTS = 12
DEFAULT_REVIEW_BATCH_MAX_INPUT_BYTES = 48 * 1024
HARD_MAX_REVIEW_BATCH_SEGMENTS = 16
HARD_MAX_REVIEW_BATCH_INPUT_BYTES = 64 * 1024
MAX_QWEN_NEW_TOKENS = 8192

# Quality screening is always computed. ``TRANSLATION_QUALITY_MODE`` is the
# canonical policy (observe/strict/off); the boolean below remains a migration
# alias for deployments that still set TRANSLATION_QUALITY_GATE=1.
TRANSLATION_QUALITY_MODE_ENV = "TRANSLATION_QUALITY_MODE"
TRANSLATION_QUALITY_GATE_ENV = "TRANSLATION_QUALITY_GATE"
TRANSLATION_ENSURE_TERMINAL_ENV = "TRANSLATION_ENSURE_TERMINAL"


def _select_cuda_model_dtype(torch_module):
    """Use BF16 only on devices that actually support it; otherwise use FP16."""

    supports_bf16 = getattr(
        torch_module.cuda, "is_bf16_supported", lambda: False
    )()
    return torch_module.bfloat16 if supports_bf16 else torch_module.float16

# ── SELF-REVIEW (Qwen TỰ CHẤM ĐIỂM + TỰ CHỈNH SỬA) ──────────────────────────────────
# Sau khi vòng dịch ban đầu cho ra bản HỢP LỆ, Qwen tự soi nghĩa/ngữ pháp/ngữ cảnh của
# mọi câu và viết lại câu bị gắn cờ; các vòng sau chỉ tiếp tục xử lý câu LỆCH NHỊP.
# Cơ chế dùng HAI tín hiệu TÁCH BẠCH (No-Fake-Success):
#   (A) KHÁCH QUAN, deterministic, không-model = thời lượng nói dự kiến của ngôn ngữ
#       đích so với duration. Vừa là cổng chọn, vừa là luật strict-better.
#   (B) MODEL SELF-SCORE = Qwen tự chấm 4 chiều 1-5 + mã lỗi + bản viết lại.
#       ADVISORY: có thể kích hoạt bản sửa provisional nhưng KHÔNG tạo semanticState=passed.
MAX_REVIEW_ROUNDS = 2          # trần cứng số vòng review (ENV QWEN_MAX_REVIEW_ROUNDS; <=0 tắt)

# Timing chấm trực tiếp thời lượng nói dự kiến của NGÔN NGỮ ĐÍCH so với số giây segment.
# Sửa nghĩa có thể giữ cùng penalty; sửa câu tràn phải giảm penalty ngặt. Không cho Qwen
# bịa thêm nội dung để lấp khoảng lặng.
TIMING_REVIEW_PENALTY_TOLERANCE = 5.0
REVIEW_MIN_TARGET_UNIT_RATIO = 0.5
MIN_REVISED_SPEECH_UNITS = 2
MIN_ACCEPTABLE_REVIEW_SCORE = 4
ReviewIssueCode = Literal[
    "meaning_unclear",
    "meaning_changed",
    "missing_information",
    "hallucination",
    "unnatural_target",
    "grammar_error",
    "wrong_register",
    "entity_mismatch",
    "number_mismatch",
    "negation_mismatch",
    "context_mismatch",
    "timing_overflow",
]
_SEMANTIC_REVIEW_ISSUES = frozenset({
    "meaning_unclear",
    "meaning_changed",
    "missing_information",
    "hallucination",
    "unnatural_target",
    "grammar_error",
    "wrong_register",
    "entity_mismatch",
    "number_mismatch",
    "negation_mismatch",
    "context_mismatch",
})
_REWRITE_SEMANTIC_RISK_ISSUES = frozenset({
    "empty_translation",
    "target_has_no_words",
    "model_artifact",
    "encoding_artifact",
    "source_script_residue",
    "copied_source",
    "likely_untranslated",
    "number_mismatch",
    "protected_token_mismatch",
    "negation_mismatch",
    "boilerplate_hallucination",
    "repeated_token_loop",
    "extreme_length_ratio",
    "too_short",
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
    # ``score`` remains parse-compatible with older timing-only rows, but cannot
    # trigger a semantic rewrite. New semantic rows must supply all four bounded
    # dimensions so a timing problem cannot masquerade as a meaning verdict.
    score: Optional[int] = Field(default=None, ge=1, le=5)
    meaning_score: Optional[int] = Field(default=None, ge=1, le=5)
    grammar_score: Optional[int] = Field(default=None, ge=1, le=5)
    context_score: Optional[int] = Field(default=None, ge=1, le=5)
    timing_score: Optional[int] = Field(default=None, ge=1, le=5)
    issue_codes: List[ReviewIssueCode] = Field(default_factory=list, max_length=12)
    revised_text: Optional[str] = None
    emotion: Optional[str] = None


class _QwenChatPrompt(str):
    """A user prompt that carries its trusted system message through the legacy seam.

    Keeping this as a ``str`` preserves CPU test fakes that patch ``_generate(prompt)``.
    ``encode`` deliberately accounts for both chat messages in batching budgets.
    """

    def __new__(cls, user_prompt: str, system_prompt: str):
        instance = super().__new__(cls, user_prompt)
        instance.system_prompt = system_prompt
        return instance

    def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": str(self)},
        ]
        serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        return serialized.encode(encoding, errors)


def count_syllables(text: str, lang: str = "en") -> int:
    """
    Legacy source-side diagnostic retained in the nine-field output contract.

    Lip-sync decisions no longer compare this value with target text; they use
    target-language speech units against trusted segment duration.
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
        # Count-only quality metadata. Never store plaintext in this field.
        self.last_quality_stats: Optional[Dict[str, Any]] = None
        # Optional per-segment count-only projection for the encrypted Analyze artifact.
        # Each row contains only the trusted input ID, score, decision, semantic state,
        # and bounded issue codes; source/target text is deliberately absent.
        self.last_quality_reports: Optional[List[Dict[str, Any]]] = None

    @staticmethod
    def _quality_mode(explicit: Optional[str] = None) -> str:
        """Resolve the quality policy without exposing any segment text.

        ``observe`` computes metadata and lets the caller surface review items;
        ``strict`` fails closed before TTS; ``off`` is reserved for diagnostics
        and still keeps normalization enabled. The legacy boolean env remains a
        compatibility alias for strict/observe.
        """

        raw = explicit
        if raw is None:
            raw = os.environ.get(TRANSLATION_QUALITY_MODE_ENV)
        if raw is None:
            legacy = os.environ.get(TRANSLATION_QUALITY_GATE_ENV, "0").strip().lower()
            return "strict" if legacy in {"1", "true", "yes", "on", "strict", "reject"} else "observe"
        mode = raw.strip().lower()
        if mode in {"1", "true", "yes", "on", "strict", "reject"}:
            return "strict"
        if mode in {"0", "false", "no", "off", "observe", "review"}:
            return "observe" if mode in {"observe", "review"} else "off"
        raise RuntimeError("translation_quality_mode_invalid")

    @classmethod
    def _quality_gate_enabled(cls, explicit: Optional[str] = None) -> bool:
        return cls._quality_mode(explicit) == "strict"

    @staticmethod
    def _quality_ensure_terminal() -> bool:
        value = os.environ.get(TRANSLATION_ENSURE_TERMINAL_ENV, "0").strip().lower()
        return value in {"1", "true", "yes", "on"}

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

            # Qwen's config currently prefers bfloat16. Volta/Turing devices satisfy the
            # worker's cc>=7 boundary but cannot execute BF16 kernels, so cast to FP16 unless
            # the actual CUDA runtime explicitly reports BF16 support.
            model_dtype = _select_cuda_model_dtype(torch)

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=model_dtype,
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

        system_prompt = getattr(prompt, "system_prompt", None)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": str(prompt)})
        # enable_thinking=False BẮT BUỘC: chế độ 'thinking' mặc định của Qwen3 phun chuỗi
        # <think> làm tăng độ trễ per-segment và chèn token phi-JSON phá ràng buộc JSON.
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        try:
            configured_max_new_tokens = int(
                os.environ.get("QWEN_MAX_NEW_TOKENS", "4096")
            )
        except ValueError:
            configured_max_new_tokens = 4096
        max_new_tokens = max(
            256, min(configured_max_new_tokens, MAX_QWEN_NEW_TOKENS)
        )
        with torch.no_grad():
            # Greedy (do_sample=False) cho JSON xác định; max_new_tokens cấu hình qua ENV
            # cho lô nhiều segment. Guided decoding (xgrammar/outlines) là lever CHẤT LƯỢNG
            # ở prod — hardware-blocked ở đây; retry-max-3 mới là bảo chứng đúng đắn.
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        out_tokens = generated[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(out_tokens, skip_special_tokens=True)

    def _split_translation_batches(
        self,
        processed_segments: List[Dict[str, Any]],
        target_language: str,
        style: str,
        source_language: str = "en",
        prompt_profile: Optional[Dict[str, Any]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Chia lô xác định theo thứ tự, số segment và kích thước prompt UTF-8.

        Không tách một segment thành nhiều phần vì ID là đơn vị nguyên tử của hợp đồng TTS.
        Nếu ngay cả một singleton vượt budget, fail-closed trước khi chạm GPU thay vì âm
        thầm gửi prompt quá cỡ hoặc cắt mất nội dung.
        """
        max_segments = max(
            1,
            int(os.environ.get(
                "QWEN_BATCH_MAX_SEGMENTS",
                str(DEFAULT_TRANSLATION_BATCH_MAX_SEGMENTS),
            )),
        )
        max_input_bytes = max(
            1,
            int(os.environ.get(
                "QWEN_BATCH_MAX_INPUT_BYTES",
                str(DEFAULT_TRANSLATION_BATCH_MAX_INPUT_BYTES),
            )),
        )

        batches: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []

        for segment in processed_segments:
            candidate = [*current, segment]
            candidate_prompt = self._build_prompt(
                target_language,
                style,
                candidate,
                source_language=source_language,
                prompt_profile=prompt_profile,
            )
            exceeds_count = len(candidate) > max_segments
            exceeds_input = len(candidate_prompt.encode("utf-8")) > max_input_bytes

            if not exceeds_count and not exceeds_input:
                current = candidate
                continue

            if current:
                batches.append(current)
                current = [segment]
            else:
                current = candidate

            singleton_prompt = self._build_prompt(
                target_language,
                style,
                current,
                source_language=source_language,
                prompt_profile=prompt_profile,
            )
            if len(singleton_prompt.encode("utf-8")) > max_input_bytes:
                raise RuntimeError(
                    "Một segment vượt QWEN_BATCH_MAX_INPUT_BYTES; từ chối dịch "
                    "thay vì gửi prompt quá giới hạn."
                )

        if current:
            batches.append(current)
        return batches

    def _translate_batch(
        self,
        processed_segments: List[Dict[str, Any]],
        target_language: str,
        style: str,
        batch_number: int,
        batch_count: int,
        source_language: str = "en",
        prompt_profile: Optional[Dict[str, Any]] = None,
    ) -> List[TranslatedSegment]:
        """Dịch một lô với đúng retry/ID-parity/fail-closed của luồng cũ."""
        prompt = self._build_prompt(
            target_language,
            style,
            processed_segments,
            source_language=source_language,
            prompt_profile=prompt_profile,
        )
        last_error_name: Optional[str] = None
        attempt_prompt = prompt

        for attempt in range(1, MAX_TRANSLATION_ATTEMPTS + 1):
            raw = self._generate(attempt_prompt)
            try:
                llm_segments = self._parse_and_validate(raw, processed_segments)
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                # ZERO-LOGGING: chỉ tên loại lỗi + metadata; KHÔNG str(e) (có thể nhúng
                # nguyên đoạn output mô hình sinh ra).
                last_error_name = type(e).__name__
                logger.warning(
                    f"Qwen output không hợp lệ ở batch {batch_number}/{batch_count} "
                    f"(lần {attempt}/{MAX_TRANSLATION_ATTEMPTS}): {last_error_name}"
                )
                if attempt < MAX_TRANSLATION_ATTEMPTS:
                    attempt_prompt = self._build_retry_prompt(prompt, attempt + 1)
                continue

            merged = self._merge(
                llm_segments,
                processed_segments,
                source_language=source_language,
                target_language=target_language,
            )
            logger.info(
                f"Đã dịch {len(merged)} segment ở batch {batch_number}/{batch_count} "
                f"(lần {attempt})."
            )
            return merged

        raise RuntimeError(
            f"Dịch Qwen thất bại ở batch {batch_number}/{batch_count} sau "
            f"{MAX_TRANSLATION_ATTEMPTS} lần thử (lỗi cuối: {last_error_name})."
        )

    @staticmethod
    def _build_retry_prompt(prompt: str, attempt: int) -> str:
        """Change deterministic greedy input after a schema failure.

        Repeating an identical prompt with ``do_sample=False`` reproduces the same
        invalid output and only burns GPU time. The retry keeps the trusted system
        role and original INPUT, but adds bounded corrective feedback without ever
        embedding the rejected model output or plaintext in logs.
        """

        correction = (
            f"RETRY_CORRECTION_ATTEMPT_{attempt}: The previous response failed JSON, "
            "schema, or ID-parity validation. Re-read the same INPUT and regenerate "
            "only the exact JSON contract.\n"
        )
        user_prompt = str(prompt)
        marker = "INPUT:\n"
        if marker in user_prompt:
            before, input_json = user_prompt.split(marker, 1)
            user_prompt = f"{before}{correction}\n{marker}{input_json}"
        else:
            user_prompt = f"{correction}\n{user_prompt}"
        return _QwenChatPrompt(
            user_prompt,
            getattr(prompt, "system_prompt", ""),
        )

    def translate_segments(self, segments: List[Dict[str, Any]], target_language: str, style: str,
                           source_language: str = "en",
                           prompt_profile: Optional[Dict[str, Any]] = None,
                           quality_mode: Optional[str] = None,
                           semantic_judges: Optional[List[str]] = None,
                           semantic_judge_passed: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Dịch các segment bằng Qwen cục bộ, đảm bảo Lip-sync + Emotion tagging.

        source_language: ngôn ngữ GỐC của thoại, dùng cho metadata nguồn và quality checks.
        """
        # Chuẩn bị dữ liệu đầu vào + tính âm tiết gốc (backend-agnostic; luôn chạy TRƯỚC
        # cả cổng fail-closed để việc căn âm tiết theo source_language không bị bỏ qua).
        resolved_quality_mode = self._quality_mode(quality_mode)
        self.last_quality_stats = None
        self.last_quality_reports = None
        # The old boolean could mark every segment as semantically correct without
        # an independently auditable verdict. Keep accepting ``False`` from legacy
        # callers during rollout, but fail closed on ``True`` and require the
        # per-segment ``semantic_judges`` contract instead.
        if semantic_judge_passed is True and semantic_judges is None:
            raise RuntimeError("semantic_review_contract_required")
        processed_segments = []
        for seg in segments:
            orig_text = seg.get("text", seg.get("original_text", ""))
            orig_syllables = count_syllables(orig_text, lang=source_language)

            # start/end là nguồn thời lượng DUY NHẤT vì audio mix/TTS cũng dùng chính span
            # này. Không tin field ``duration`` tùy ý trong payload: một giá trị phóng đại
            # có thể làm scorer cho qua câu sẽ tràn khung thật.
            start_s = to_seconds(seg.get("start", 0))
            end_s = to_seconds(seg.get("end", 0))
            duration = end_s - start_s
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("segment_end_must_be_after_start")

            processed_segments.append({
                "id": seg.get("id", ""),
                "start": start_s,
                "end": end_s,
                "duration": round(duration, 3),
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

        batches = self._split_translation_batches(
            processed_segments, target_language, style, source_language, prompt_profile
        )

        # Dịch baseline của TẤT CẢ batch trước khi review. Nếu bất kỳ batch nào hết retry,
        # exception thoát ra và caller không thể nhận kết quả một phần của các batch trước.
        translated_batches: List[List[TranslatedSegment]] = []
        for batch_number, batch in enumerate(batches, start=1):
            translated_batches.append(self._translate_batch(
                batch,
                target_language,
                style,
                batch_number,
                len(batches),
                source_language,
                prompt_profile,
            ))

        translated_segments = [
            segment for batch in translated_batches for segment in batch
        ]

        # Review the complete ordered dialogue so adjacent context survives translation-batch
        # boundaries. _self_review applies its own hard prompt chunks; it never rebuilds one
        # unbounded prompt for the whole job.
        if self._self_review_enabled():
            try:
                translated_segments = self._self_review(
                    translated_segments,
                    target_language,
                    source_language,
                    style,
                    prompt_profile,
                )
            except Exception as e:
                logger.warning(
                    f"Self-review job bỏ qua: {type(e).__name__}"
                )

        final_segments = [
            segment.model_dump() for segment in translated_segments
        ]
        spoken_texts = [
            self._spoken_timing_text(
                str(segment.get("translated_text", "")), target_language
            )
            for segment in final_segments
        ]
        semantic_judge_list = (
            list(semantic_judges) if semantic_judges is not None else None
        )
        revised_after_baseline = int(
            (self.last_review_stats or {}).get("revised", 0)
        )
        if semantic_judge_list is not None and revised_after_baseline:
            # A verdict supplied before generation cannot be content-bound to text
            # that this call subsequently changed. Force a fresh review of the exact
            # final candidate instead of carrying stale "passed" states forward.
            semantic_judge_list = None
            logger.warning(
                f"Semantic verdict invalidated after {revised_after_baseline} "
                "self-review revision(s)."
            )
        quality = quality_gate_segments(
            final_segments,
            source_key="original_text",
            target_key="translated_text",
            source_language=source_language,
            target_language=target_language,
            spoken_texts=spoken_texts,
            semantic_judges=semantic_judge_list,
            require_semantic_judge=True,
        )
        self.last_quality_stats = quality_gate_metadata(quality)
        judge_states = (
            [str(state).casefold() for state in semantic_judge_list]
            if semantic_judge_list is not None
            else ["not_run"] * len(final_segments)
        )
        self.last_quality_reports = [
            {
                "id": str(segment.get("id", "")),
                "score": report.score,
                "decision": report.decision,
                "semanticState": judge_states[index],
                "issueCodes": list(report.issues),
            }
            for index, (segment, report) in enumerate(
                zip(final_segments, quality.reports, strict=True)
            )
        ]
        if quality.decision == "reject":
            logger.warning(
                "Translation quality gate rejected output: "
                f"segments={len(final_segments)} rejected={quality.rejected}"
            )
            if resolved_quality_mode == "strict":
                raise RuntimeError("translation_quality_rejected")
        elif resolved_quality_mode == "strict" and quality.decision != "accept":
            logger.warning(
                "Translation quality review is pending before strict render: "
                f"segments={len(final_segments)} review={quality.review}"
            )
            raise RuntimeError("translation_quality_requires_review")
        return final_segments

    def _build_prompt(
        self,
        target_language: str,
        style: str,
        processed_segments: List[Dict[str, Any]],
        *,
        source_language: str = "en",
        prompt_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
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
                "speaker_id": p["speaker_id"],
                "original_text": p["original_text"],
            }
            for p in processed_segments
        ]
        input_json = json.dumps({"input_segments": model_inputs}, ensure_ascii=False, indent=2)

        user_prompt = (
            "You are a professional dubbing translator for movies and videos.\n"
            f"Source language: {source_language}.\n"
            f"Target language: {target_language}.\n"
            f"Style/Tone: {style}.\n\n"
            "CRITICAL RULES:\n"
            "1. MEANING FIRST: translate every proposition faithfully. Preserve names, "
            "numbers, dates, times, units, URLs, email addresses, negation, speaker intent, "
            "and register. Do not summarize, invent, or drop a clause merely to shorten it.\n"
            "2. LIP-SYNC (Pacing): each input segment gives trusted `duration` in seconds. "
            "Keep natural target-language speech concise enough for that duration "
            "while retaining meaning; prefer a natural paraphrase only when the literal form "
            "would overrun the segment.\n"
            "3. EMOTION TAGGING: assign exactly one emotion per segment, chosen ONLY from "
            "this enum: NEUTRAL, HAPPY, ANGRY, SAD, WHISPERING, SHOUTING.\n"
            "4. TARGET-LANGUAGE QUALITY: write idiomatic sentences in the target language, "
            "using its normal spelling, diacritics, word order, and punctuation. Never emit "
            "placeholders, source-script residue, boilerplate, or repeated filler words.\n"
            "5. STRICT JSON: respond with ONLY a JSON object of EXACTLY this shape — no "
            "prose, no markdown, no code fences, no commentary:\n"
            '   {"segments": [{"id": <same id as input>, "translated_text": "<translation>", '
            '"emotion": "<ONE OF THE ENUM>"}]}\n'
            "   Return exactly one object per input segment, preserving each `id` EXACTLY. "
            "Do NOT add or drop segments. Do NOT include start/end/duration/original_text "
            "in your output.\n\n"
            f"INPUT:\n{input_json}\n"
        )
        return _QwenChatPrompt(
            user_prompt,
            build_qwen_system_prompt(target_language, prompt_profile),
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
               processed_segments: List[Dict[str, Any]],
               *, source_language: str = "en",
               target_language: str = "Vietnamese") -> List[TranslatedSegment]:
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
            normalized_text = normalize_translation_text(
                llm.translated_text,
                ensure_terminal=self._quality_ensure_terminal(),
            )
            merged.append(TranslatedSegment(
                id=p["id"],
                start=p["start"],
                end=p["end"],
                duration=p["duration"],
                original_text=p["original_text"],
                original_syllables=p["original_syllables"],
                translated_text=normalized_text,
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
    def _spoken_timing_text(translated_text: str, target_language: str) -> str:
        """Return the same private TN copy that the TTS boundary will synthesize."""

        # Runtime import avoids a module cycle: text_preprocessing deliberately uses
        # translation_quality for subtitle normalization. Production prewarms the NeMo
        # grammars before Qwen; development uses the same explicit fallback as TTS.
        from src.text_preprocessing import prepare_tts_text

        return prepare_tts_text(
            translated_text, target_language, ensure_terminal=True
        )

    @staticmethod
    def _timing_penalty(translated_text: str, duration: float,
                        target_language: str) -> float:
        """Target-side speech overflow measured against trusted segment seconds."""

        spoken = TranslationService._spoken_timing_text(
            translated_text, target_language
        )
        return timing_overflow_penalty(spoken, target_language, duration)

    @staticmethod
    def _coerce_emotion(raw: Optional[str], fallback: str) -> str:
        """Chuẩn hóa emotion do review đề xuất. Ngoài enum -> GIỮ emotion HỢP LỆ hiện tại
        (fallback), KHÔNG tụt về NEUTRAL: bản dịch đầu đã chốt emotion hợp lệ, review chỉ
        được NÂNG CẤP khi chắc chắn — không được hạ cấp một nhãn đang đúng."""
        e = (raw or "").strip().upper()
        return e if e in VALID_EMOTIONS else fallback

    def _build_review_prompt(self, target_language: str, style: str,
                             review_items: List[Dict[str, Any]],
                             prompt_profile: Optional[Dict[str, Any]] = None,
                             source_language: str = "en") -> str:
        """Prompt TỰ-SOI, CHẠY CỤC BỘ (chỉ tới _generate — KHÔNG cloud, KHÔNG log).

        Nhúng bản dịch hiện tại + số liệu pacing để Qwen tự chấm rồi viết gọn lại. Giấu
        start/end khỏi mô hình; review chỉ được đổi text/emotion (mốc thời gian ghép từ đầu
        vào tin cậy). Giữ review trên seam _generate CỤC BỘ là CHỊU-TẢI cho Zero-Egress:
        route review sang cloud sẽ tái mở lỗ rò transcript mà dự án đã bỏ OpenAI để đóng."""
        input_json = json.dumps(
            {"review_segments": review_items}, ensure_ascii=False, indent=2
        )
        user_prompt = (
            "You are a strict senior dubbing-translation reviewer.\n"
            f"Source language: {source_language}.\n"
            f"Target language: {target_language}.\n"
            f"Style/Tone: {style}.\n\n"
            "Review EVERY input segment in array order. Use `context_before` and "
            "`context_after` to resolve pronouns, continuity, register, and dialogue intent.\n"
            "For EACH segment:\n"
            "1. Give four integer scores from 1 to 5: `meaning_score` for faithful and "
            "logically meaningful translation, `grammar_score` for natural target-language "
            "grammar, `context_score` for consistency with adjacent dialogue, and "
            "`timing_score` for speech fitting `available_seconds`.\n"
            "2. Return `issue_codes` using ONLY this enum (use [] when clean): "
            "meaning_unclear, meaning_changed, missing_information, hallucination, "
            "unnatural_target, grammar_error, wrong_register, entity_mismatch, "
            "number_mismatch, negation_mismatch, context_mismatch, timing_overflow.\n"
            "3. If any score is below 4 or any issue exists, rewrite `revised_text` into a "
            "complete, meaningful, natural sentence. Preserve every fact, entity, number, "
            "negation, uncertainty, intent and register, while fitting the available time. "
            "Never invent information and never pad an underfilled line. If clean, repeat "
            "`current_translation` exactly.\n"
            "4. You MAY set `emotion` from the enum ONLY if the current tag is clearly wrong.\n"
            "EMOTION enum: NEUTRAL, HAPPY, ANGRY, SAD, WHISPERING, SHOUTING.\n"
            "This is an advisory self-review by the same model, not an independent semantic "
            "approval. STRICT JSON only: no prose, rationale, markdown, or code fences. "
            "Return EXACTLY this shape:\n"
            '   {"reviews": [{"id": <same id>, "meaning_score": <1-5>, '
            '"grammar_score": <1-5>, "context_score": <1-5>, "timing_score": <1-5>, '
            '"issue_codes": ["<enum>"], "revised_text": "<text>", '
            '"emotion": "<ONE OF THE ENUM>"}]}\n'
            "Keep each `id` EXACTLY; do NOT add/drop/renumber; do NOT output "
            "start/end/duration/speaker_id (not yours to set).\n\n"
            f"INPUT:\n{input_json}\n"
        )
        return _QwenChatPrompt(
            user_prompt,
            build_qwen_system_prompt(target_language, prompt_profile),
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
                raise ValueError("review row must be an object")
            item = _ReviewItem(**r)  # ValidationError -> caller (FAIL-OPEN)
            rid = str(item.id)
            if rid not in candidate_ids or rid in out:
                raise ValueError("review id set mismatch")
            out[rid] = item
        if set(out) != candidate_ids:
            raise ValueError("review id set mismatch")
        return out

    def _split_review_chunks(
        self,
        candidates: List[Any],
        review_items: List[Dict[str, Any]],
        target_language: str,
        source_language: str,
        style: str,
        prompt_profile: Optional[Dict[str, Any]],
    ) -> List[tuple[List[Any], List[Dict[str, Any]]]]:
        """Bound review prompts by count and serialized UTF-8 bytes.

        Adjacent rows are already embedded in each item, so chunking does not lose
        context at a chunk boundary. If a singleton is too large only because of its
        duplicated neighbors, retry it without those optional context copies. A truly
        oversized singleton is skipped fail-open rather than sent to the GPU.
        """

        def bounded_env(name: str, default: int, hard_max: int) -> int:
            try:
                configured = int(os.environ.get(name, str(default)))
            except ValueError:
                configured = default
            return max(1, min(configured, hard_max))

        max_segments = bounded_env(
            "QWEN_REVIEW_MAX_SEGMENTS",
            DEFAULT_REVIEW_BATCH_MAX_SEGMENTS,
            HARD_MAX_REVIEW_BATCH_SEGMENTS,
        )
        max_input_bytes = bounded_env(
            "QWEN_REVIEW_MAX_INPUT_BYTES",
            DEFAULT_REVIEW_BATCH_MAX_INPUT_BYTES,
            HARD_MAX_REVIEW_BATCH_INPUT_BYTES,
        )

        def prompt_bytes(pairs: List[tuple[Any, Dict[str, Any]]]) -> int:
            prompt = self._build_review_prompt(
                target_language,
                style,
                [item for _, item in pairs],
                prompt_profile,
                source_language=source_language,
            )
            return len(prompt.encode("utf-8"))

        chunks: List[List[tuple[Any, Dict[str, Any]]]] = []
        current: List[tuple[Any, Dict[str, Any]]] = []
        oversized = 0

        for candidate, item in zip(candidates, review_items):
            trial = [*current, (candidate, item)]
            if len(trial) <= max_segments and prompt_bytes(trial) <= max_input_bytes:
                current = trial
                continue

            if current:
                chunks.append(current)
                current = []

            singleton = [(candidate, item)]
            if prompt_bytes(singleton) <= max_input_bytes:
                current = singleton
                continue

            compact_item = dict(item)
            compact_item["context_before"] = None
            compact_item["context_after"] = None
            compact_singleton = [(candidate, compact_item)]
            if prompt_bytes(compact_singleton) <= max_input_bytes:
                current = compact_singleton
            else:
                oversized += 1

        if current:
            chunks.append(current)
        if oversized:
            logger.warning(
                f"Self-review skipped {oversized} oversized singleton segment(s)."
            )

        return [
            (
                [candidate for candidate, _ in chunk],
                [item for _, item in chunk],
            )
            for chunk in chunks
        ]

    def _self_review(self, merged: List[TranslatedSegment], target_language: str,
                     source_language: str, style: str,
                     prompt_profile: Optional[Dict[str, Any]] = None) -> List[TranslatedSegment]:
        """Run a bounded same-model semantic/timing review without granting approval.

        Round one evaluates every sentence with adjacent context. A semantic rewrite is
        provisional and is accepted only when deterministic structural risks and timing do
        not worsen. Timing overflow still requires strict measurable improvement. Trusted
        IDs, timestamps and speakers are never taken from model output.

        This model can revise its draft, but cannot set the independent semantic verdict;
        ``last_quality_reports`` therefore remains ``not_run`` unless a separate validated
        reviewer is supplied. Any generation or schema failure is fail-open to the valid
        baseline translation and never logs source or translated text.
        """
        self.last_review_stats = {"revised": 0, "rounds": 0, "skipped": None}

        try:
            configured_rounds = int(
                os.environ.get("QWEN_MAX_REVIEW_ROUNDS", str(MAX_REVIEW_ROUNDS))
            )
        except ValueError:
            configured_rounds = MAX_REVIEW_ROUNDS
        # This is a real hard cap, not merely a default. A bad deployment value
        # must not create an unbounded same-model loop or unexpected GPU spend.
        max_rounds = max(0, min(configured_rounds, MAX_REVIEW_ROUNDS))
        if max_rounds <= 0:
            self.last_review_stats["skipped"] = "rounds_disabled"
            return merged

        current = list(merged)
        total_revised = 0
        rounds_run = 0
        for round_index in range(max_rounds):
            # Round one reviews every sentence for meaning and fluency. Later
            # rounds are reserved for lines that still measurably overflow.
            candidates = []
            for idx, seg in enumerate(current):
                if seg.duration <= 0:
                    continue
                spoken = self._spoken_timing_text(
                    seg.translated_text, target_language
                )
                pen = timing_overflow_penalty(
                    spoken, target_language, seg.duration
                )
                if round_index == 0 or pen > TIMING_REVIEW_PENALTY_TOLERANCE:
                    candidates.append((idx, seg, pen, spoken))
            if not candidates:
                break
            rounds_run += 1

            review_items = []
            for idx, seg, pen, spoken in candidates:
                before = current[idx - 1] if idx > 0 else None
                after = current[idx + 1] if idx + 1 < len(current) else None

                def context_row(value: Optional[TranslatedSegment]) -> Optional[Dict[str, Any]]:
                    if value is None:
                        return None
                    return {
                        "id": value.id,
                        "speaker_id": value.speaker_id,
                        "original_text": value.original_text,
                        "current_translation": value.translated_text,
                    }

                review_items.append({
                    "id": seg.id,
                    "speaker_id": seg.speaker_id,
                    "available_seconds": seg.duration,
                    "current_speech_units": speech_unit_count(
                        spoken, target_language
                    ),
                    "estimated_spoken_seconds": estimate_spoken_duration_seconds(
                        spoken, target_language
                    ),
                    "current_emotion": seg.emotion,
                    "original_text": seg.original_text,
                    "current_translation": seg.translated_text,
                    "deterministic_timing_overflow": (
                        pen > TIMING_REVIEW_PENALTY_TOLERANCE
                    ),
                    "context_before": context_row(before),
                    "context_after": context_row(after),
                })

            review_chunks = self._split_review_chunks(
                candidates,
                review_items,
                target_language,
                source_language,
                style,
                prompt_profile,
            )
            reviews: Dict[str, _ReviewItem] = {}
            for chunk_number, (candidate_chunk, item_chunk) in enumerate(
                review_chunks, start=1
            ):
                try:
                    raw = self._generate(self._build_review_prompt(
                        target_language,
                        style,
                        item_chunk,
                        prompt_profile,
                        source_language=source_language,
                    ))
                    chunk_reviews = self._parse_review(
                        raw,
                        {str(segment.id) for _, segment, _, _ in candidate_chunk},
                    )
                    reviews.update(chunk_reviews)
                except Exception as e:
                    # Keep only stable metadata; parser exceptions may embed model text.
                    logger.warning(
                        f"Self-review output không hợp lệ (vòng {rounds_run}, "
                        f"chunk {chunk_number}/{len(review_chunks)}): {type(e).__name__}"
                    )
                    continue

            progress = 0
            for idx, seg, old_pen, old_spoken in candidates:
                rv = reviews.get(str(seg.id))
                if rv is None or not rv.revised_text:
                    continue

                review_scores = (
                    rv.meaning_score,
                    rv.grammar_score,
                    rv.context_score,
                    rv.timing_score,
                )
                complete_review_scores = all(
                    score is not None for score in review_scores
                )
                semantic_flagged = bool(
                    complete_review_scores
                    and (
                        set(rv.issue_codes).intersection(_SEMANTIC_REVIEW_ISSUES)
                        or any(
                            score < MIN_ACCEPTABLE_REVIEW_SCORE
                            for score in review_scores[:3]
                        )
                    )
                )
                deterministic_timing_overflow = (
                    old_pen > TIMING_REVIEW_PENALTY_TOLERANCE
                )
                # A clean semantic review cannot silently rewrite a sentence.
                # Timing-only legacy rows remain accepted under the old strict gate.
                if not semantic_flagged and not deterministic_timing_overflow:
                    continue

                new_text = normalize_translation_text(
                    rv.revised_text,
                    ensure_terminal=self._quality_ensure_terminal(),
                )
                if not new_text:
                    continue
                old_units = speech_unit_count(seg.translated_text, target_language)
                new_units = speech_unit_count(new_text, target_language)
                # Sàn chống cắt cụt dựa trên chính bản dịch mục tiêu hiện tại, không dựa
                # vào đơn vị của ngôn ngữ nguồn vốn không tương đương.
                floor = math.ceil(REVIEW_MIN_TARGET_UNIT_RATIO * old_units)
                if old_units >= MIN_REVISED_SPEECH_UNITS:
                    floor = max(floor, MIN_REVISED_SPEECH_UNITS)
                if new_units < floor:
                    continue
                new_spoken = self._spoken_timing_text(new_text, target_language)
                new_pen = timing_overflow_penalty(
                    new_spoken, target_language, seg.duration
                )
                if deterministic_timing_overflow and new_pen >= old_pen:
                    continue
                # Meaning/grammar rewrites may keep the same duration, but can
                # never consume more timing budget than the current sentence.
                if not deterministic_timing_overflow and new_pen > old_pen:
                    continue
                old_quality = score_translation(
                    seg.original_text,
                    seg.translated_text,
                    source_language=source_language,
                    target_language=target_language,
                    duration_seconds=seg.duration,
                    spoken_text=old_spoken,
                )
                new_quality = score_translation(
                    seg.original_text,
                    new_text,
                    source_language=source_language,
                    target_language=target_language,
                    duration_seconds=seg.duration,
                    spoken_text=new_spoken,
                )
                newly_introduced_risks = (
                    set(new_quality.issues) - set(old_quality.issues)
                ).intersection(_REWRITE_SEMANTIC_RISK_ISSUES)
                if newly_introduced_risks or new_quality.score < old_quality.score:
                    continue
                new_emotion = self._coerce_emotion(rv.emotion, seg.emotion)
                if (
                    new_text == seg.translated_text
                    and new_emotion == seg.emotion
                ):
                    continue
                current[idx] = seg.model_copy(
                    update={"translated_text": new_text, "emotion": new_emotion}
                )
                progress += 1
                total_revised += 1
            if progress == 0:
                break

        if total_revised:
            # ĐẾM-ONLY (không plaintext): số câu ĐỔI dưới cổng khách quan ngặt.
            logger.info(f"Self-review: tinh chỉnh {total_revised} câu qua {rounds_run} vòng.")
        self.last_review_stats = {"revised": total_revised, "rounds": rounds_run, "skipped": None}
        return current


# Expose a singleton instance. __init__ AN TOÀN để import (không transformers/không VRAM);
# việc nạp trọng số do ModelManager.load_all_models() gọi load_model() một lần.
translation_service = TranslationService()
