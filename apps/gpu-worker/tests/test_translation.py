import json
import logging
import importlib.util
import pytest
from unittest.mock import MagicMock, patch

from src.translation_service import (
    TranslationService,
    count_syllables,
    MAX_TRANSLATION_ATTEMPTS,
)

# Các test này chạy trên máy CPU-only KHÔNG có GPU/transformers/trọng số Qwen: mọi
# nhánh xây prompt / parse / retry / fail-closed đều kiểm thử bằng cách mock seam DUY
# NHẤT `_generate` (trả JSON đóng hộp) — y hệt cách bộ test cũ mock cloud parse. Suy
# luận Qwen THẬT (chất lượng dịch, cư trú VRAM, độ trễ) là residual_hardware, không giả xanh.


def _loaded_service() -> TranslationService:
    """Service coi như đã nạp (is_loaded=True) nhưng KHÔNG trọng số: test thay `_generate`."""
    svc = TranslationService()
    svc.is_loaded = True
    return svc


def _valid_output(ids, emotion: str = "HAPPY") -> str:
    return json.dumps({
        "segments": [
            {"id": i, "translated_text": f"dịch {i}", "emotion": emotion} for i in ids
        ]
    })


# ---- Fail-closed: chưa nạp mô hình (No-Fake-Success) --------------------------------

def test_translation_refuses_when_model_not_loaded():
    """Mô hình Qwen CHƯA nạp (is_loaded=False) -> translate phải TỪ CHỐI (raise), tuyệt
    đối không bịa bản dịch giả rồi báo thành công. Sau khi bỏ đường OpenAI cloud (đóng
    lỗ rò transcript, tiêu chí #2), trigger fail-closed đổi từ 'thiếu OPENAI_API_KEY'
    sang 'mô hình chưa nạp'."""
    svc = TranslationService()
    assert svc.is_loaded is False  # __init__ KHÔNG nạp trọng số (import-safe)
    segments = [{"id": 1, "start": 0.0, "end": 2.5, "text": "Get out!", "speaker": "SPEAKER_01"}]
    with pytest.raises(RuntimeError):
        svc.translate_segments(segments, "Vietnamese", "Slang")


def test_unloaded_service_never_calls_generate():
    """Cổng fail-closed nằm TRƯỚC khi chạm mô hình: `_generate` KHÔNG bao giờ được gọi
    khi chưa nạp (không tốn/không chạm GPU)."""
    svc = TranslationService()
    svc._generate = MagicMock()
    with pytest.raises(RuntimeError):
        svc.translate_segments(
            [{"id": 1, "start": 0, "end": 1, "text": "hi"}], "Vietnamese", "Formal"
        )
    svc._generate.assert_not_called()


def test_service_construction_is_import_safe():
    """__init__ KHÔNG nạp trọng số / KHÔNG import transformers: thu thập test trên máy
    CPU-only thiếu thư viện vẫn phải chạy. Việc module import được + dựng được service
    (is_loaded=False, model=None) chính là bằng chứng import-safety."""
    svc = TranslationService()
    assert svc.is_loaded is False
    assert svc.model is None
    assert svc.tokenizer is None


# ---- source_language threading (giữ ý định test cũ, đổi trigger fail-closed) ---------

def test_translate_threads_source_language_into_syllable_count():
    """source_language phải được truyền xuống count_syllables (không ghim cứng 'en').
    count_syllables chạy TRƯỚC cổng fail-closed nên vẫn bắt được 'lang' dù sau đó raise."""
    svc = TranslationService()  # is_loaded=False -> fail-closed sau khi đếm âm tiết
    segments = [{"id": 1, "start": 0, "end": 1, "text": "xin chào các bạn"}]
    with patch("src.translation_service.count_syllables", return_value=3) as mock_cs:
        with pytest.raises(RuntimeError):
            svc.translate_segments(segments, "English", "Formal", source_language="vi")
    mock_cs.assert_called_once()
    assert mock_cs.call_args.kwargs.get("lang") == "vi"


# ---- count_syllables (backend-agnostic, không đổi) ---------------------------------

def test_count_syllables_english():
    assert count_syllables("Hello world", "en") == 3
    assert count_syllables("Get out of my house", "en") >= 4
    assert count_syllables("I am very happy today", "en") >= 5


def test_count_syllables_vietnamese():
    assert count_syllables("Chào mừng các bạn", "vi") == 4
    assert count_syllables("Cút khỏi nhà tao", "vi") == 4


# ---- Prompt building + parse (mock _generate) --------------------------------------

def test_qwen_builds_prompt_and_parses():
    """Prompt phải nhúng schema đầu ra + enum cảm xúc + input_segments (Qwen prompt-only
    không có response_format); kết quả giữ ĐỦ hợp đồng 9 trường (model_dump)."""
    svc = _loaded_service()
    captured = {}

    def fake_generate(prompt):
        captured["prompt"] = prompt
        return _valid_output([1])

    svc._generate = fake_generate
    segs = [{"id": 1, "start": 0.0, "end": 2.0, "text": "hello", "speaker": "S1"}]
    out = svc.translate_segments(segs, "Vietnamese", "Formal", source_language="en")

    # Prompt hướng dẫn mô hình: schema + enum cảm xúc + pacing + input.
    assert "translated_text" in captured["prompt"]
    assert "SHOUTING" in captured["prompt"]          # enum cảm xúc được liệt kê đủ
    assert '"duration"' in captured["prompt"]         # đưa pacing cho lip-sync
    assert "input_segments" in captured["prompt"]
    # Hợp đồng 9 trường nguyên vẹn cho process_job/TTS hạ nguồn.
    assert set(out[0].keys()) == {
        "id", "start", "end", "duration", "original_text",
        "original_syllables", "translated_text", "emotion", "speaker_id",
    }
    assert out[0]["translated_text"] == "dịch 1"
    assert out[0]["emotion"] == "HAPPY"
    assert out[0]["speaker_id"] == "S1"


def test_duration_computed_from_timecode_strings():
    """CC-1 phòng thủ (đổi seam sang `_generate`): start/end dạng "HH:MM:SS" phải cho
    duration = end-start (giây), KHÔNG rơi fallback 2.0s như bản cũ `float("00:00:15")`.
    Mốc chuẩn hóa về giây phải xuất hiện ở đầu ra (ghép từ đầu vào tin cậy)."""
    svc = _loaded_service()
    svc._generate = MagicMock(return_value=_valid_output([1]))
    segs = [{"id": 1, "start": "00:00:10", "end": "00:00:15", "text": "hello", "speaker": "S1"}]
    out = svc.translate_segments(segs, "Vietnamese", "Formal", source_language="en")

    assert out[0]["duration"] == 5.0   # 15 - 10, KHÔNG phải fallback 2.0
    assert out[0]["start"] == 10.0     # mốc đã chuẩn hóa về giây cho bước mix hạ nguồn
    assert out[0]["end"] == 15.0


# ---- Auto-Retry tối đa 3 + fail-closed (TRANSLATION_RULES §4) -----------------------

def test_qwen_retry_on_invalid_then_success():
    """JSON hỏng 2 lần rồi hợp lệ -> thành công; `_generate` gọi ĐÚNG 3 lần."""
    svc = _loaded_service()
    svc._generate = MagicMock(side_effect=["not json", "{bad", _valid_output([1])])
    segs = [{"id": 1, "start": 0, "end": 1, "text": "hello", "speaker": "S1"}]
    out = svc.translate_segments(segs, "Vietnamese", "Formal")
    assert svc._generate.call_count == 3
    assert out[0]["translated_text"] == "dịch 1"


def test_qwen_fail_closed_after_max_retries():
    """Hết 3 lần vẫn hỏng -> raise (fail-closed); `_generate` gọi đúng 3 lần; KHÔNG bịa."""
    svc = _loaded_service()
    svc._generate = MagicMock(side_effect=["x", "y", "z"])
    segs = [{"id": 1, "start": 0, "end": 1, "text": "hello", "speaker": "S1"}]
    with pytest.raises(RuntimeError):
        svc.translate_segments(segs, "Vietnamese", "Formal")
    assert svc._generate.call_count == MAX_TRANSLATION_ATTEMPTS


def test_qwen_fail_closed_on_id_count_mismatch():
    """Mô hình trả THIẾU segment (sai tập id) = không hợp lệ -> retry -> fail-closed.
    Chống 'đặt sai câu' âm thầm khi mô hình nhỏ renumber/bỏ segment."""
    svc = _loaded_service()
    # Đầu vào 2 id nhưng mô hình chỉ trả 1 -> ID-parity fail mọi lần.
    svc._generate = MagicMock(side_effect=[_valid_output([1])] * MAX_TRANSLATION_ATTEMPTS)
    segs = [
        {"id": 1, "start": 0, "end": 1, "text": "a", "speaker": "S1"},
        {"id": 2, "start": 1, "end": 2, "text": "b", "speaker": "S2"},
    ]
    with pytest.raises(RuntimeError):
        svc.translate_segments(segs, "Vietnamese", "Formal")
    assert svc._generate.call_count == MAX_TRANSLATION_ATTEMPTS


# ---- Ghép mốc thời gian tin cậy + chuẩn hóa cảm xúc ---------------------------------

def test_qwen_reconciles_timings_from_trusted_input():
    """Mô hình echo start/end/speaker BẬY -> đầu ra vẫn mang mốc thời gian + speaker
    TIN CẬY từ đầu vào (không bao giờ tin echo của LLM). emotion lệch hoa/thường được
    chuẩn hóa lên CHỮ HOA."""
    svc = _loaded_service()
    poisoned = json.dumps({"segments": [
        {"id": 1, "translated_text": "ok", "emotion": "happy",
         "start": 999.0, "end": -5.0, "speaker_id": "HACKED"}
    ]})
    svc._generate = MagicMock(return_value=poisoned)
    segs = [{"id": 1, "start": 3.0, "end": 8.0, "text": "hello", "speaker": "S1"}]
    out = svc.translate_segments(segs, "Vietnamese", "Formal")

    assert out[0]["start"] == 3.0          # từ đầu vào, KHÔNG phải 999.0 của LLM
    assert out[0]["end"] == 8.0            # KHÔNG phải -5.0
    assert out[0]["speaker_id"] == "S1"    # KHÔNG phải "HACKED"
    assert out[0]["translated_text"] == "ok"
    assert out[0]["emotion"] == "HAPPY"    # 'happy' -> chuẩn hóa CHỮ HOA


def test_qwen_normalizes_and_coerces_emotion():
    """Emotion lệch hoa/thường -> chuẩn hóa; NGOÀI enum -> quy về NEUTRAL (không im lặng
    đánh sập cả lô: emotion là gợi ý ngữ điệu thứ yếu, hạ nguồn map unknown->NEUTRAL an
    toàn — đánh sập vì một nhãn lệch sẽ hạ độ tin cậy mà gần như không thêm tính đúng)."""
    svc = _loaded_service()
    out_json = json.dumps({"segments": [
        {"id": 1, "translated_text": "a", "emotion": "angry"},     # -> ANGRY
        {"id": 2, "translated_text": "b", "emotion": "EXCITED"},   # ngoài enum -> NEUTRAL
    ]})
    svc._generate = MagicMock(return_value=out_json)
    segs = [
        {"id": 1, "start": 0, "end": 1, "text": "a", "speaker": "S1"},
        {"id": 2, "start": 1, "end": 2, "text": "b", "speaker": "S2"},
    ]
    out = svc.translate_segments(segs, "Vietnamese", "Formal")
    by_id = {o["id"]: o for o in out}
    assert by_id[1]["emotion"] == "ANGRY"
    assert by_id[2]["emotion"] == "NEUTRAL"


# ---- load_model fail-closed (thiếu transformers/GPU) -------------------------------

def test_load_model_fail_closed_without_runtime():
    """load_model phải fail-closed (raise RuntimeError) + KHÔNG đặt is_loaded khi thiếu
    transformers HOẶC thiếu CUDA (mirror asr_service). Trên máy test CPU-only/không thư
    viện, chỉ nhánh fail-closed chạy được; nạp trọng số thật là residual_hardware."""
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    transformers_present = importlib.util.find_spec("transformers") is not None
    if transformers_present and cuda:
        pytest.skip("Máy có GPU + transformers: load_model sẽ nạp trọng số thật (residual_hardware).")

    svc = TranslationService()
    with pytest.raises(RuntimeError):
        svc.load_model()
    assert svc.is_loaded is False


# ════════════════════════════════════════════════════════════════════════════════════
# SELF-REVIEW: Qwen TỰ CHẤM ĐIỂM + TỰ CHỈNH SỬA (bounded, FAIL-OPEN)
# ════════════════════════════════════════════════════════════════════════════════════
# Toàn bộ mock seam `_generate` (CPU-only). Cổng KHÁCH QUAN _pacing_penalty (đếm âm tiết)
# vừa chọn candidate vừa là LUẬT CHẤP NHẬN "chỉ nhận nếu ĐO ĐƯỢC tốt hơn NGẶT" — kiểm thử
# thật, không giả xanh. Self-score của mô hình là advisory (định hướng bản viết lại), KHÔNG
# phải cổng chấp nhận nên không cần model thật. Chất lượng dịch Qwen thật = residual_hardware.


def _w(n: int) -> str:
    """n 'âm tiết' cho ngôn ngữ đếm-từ (vi target/source): count_syllables đếm .split()
    nên kiểm soát pacing CHÍNH XÁC bằng số từ, không phụ thuộc heuristic nguyên âm."""
    return " ".join(["la"] * n)


def _translated_output(items, emotion: str = "HAPPY") -> str:
    """Vòng dịch BAN ĐẦU: items = list[(id, translated_text)] -> {"segments":[...]}.
    Khác _valid_output ở chỗ cho phép ĐẶT translated_text tùy ý (điều khiển pacing)."""
    return json.dumps({
        "segments": [
            {"id": i, "translated_text": t, "emotion": emotion} for i, t in items
        ]
    })


def _review_output(items) -> str:
    """Vòng SELF-REVIEW: items = list[dict] -> {"reviews":[...]}. dict được echo NGUYÊN
    (kể cả field bậy start/speaker_id) để test Zero-Trust bỏ qua chúng."""
    return json.dumps({"reviews": items})


def _seg(sid, text, start=0.0, end=2.0, speaker="S1"):
    return {"id": sid, "start": start, "end": end, "text": text, "speaker": speaker}


# ---- Đơn vị: _pacing_penalty + _pacing_measurable ----------------------------------

def test_pacing_penalty_overflow_weighted_double():
    """TRÀN âm tiết (lip-sync nặng nhất) phạt GẤP ĐÔI HỤT; khớp -> 0."""
    svc = TranslationService()
    assert svc._pacing_penalty("a b c d", 2, "Vietnamese") == 4   # tràn +2 -> 2*2
    assert svc._pacing_penalty("a", 3, "Vietnamese") == 2         # hụt -2 -> 1*2
    assert svc._pacing_penalty("a b c", 3, "Vietnamese") == 0     # khớp -> 0


def test_pacing_measurable_allowlist():
    """Nhận CẢ mã ISO LẪN tên tiếng Anh + chuẩn hóa primary subtag; CJK/None -> False."""
    m = TranslationService._pacing_measurable
    assert m("Vietnamese") and m("English") and m("vi") and m("en")
    assert m("en-US") and m("pt_BR")            # subtag normalize -> en/pt
    assert not m("Japanese") and not m("Korean") and not m("Chinese")
    assert not m("ja") and not m("ko") and not m("zh")
    assert not m("") and not m(None)


# ---- Cổng candidate: bỏ qua khi đã trong dung sai ----------------------------------

def test_self_review_skipped_when_within_tolerance():
    """Câu đã khớp nhịp (penalty<=tolerance) -> KHÔNG có candidate -> KHÔNG thêm lần
    _generate nào (trường hợp tốt = 0 chi phí review)."""
    svc = _loaded_service()
    svc._generate = MagicMock(side_effect=[_valid_output([1])])   # chỉ 1 phần tử: gọi 2 sẽ nổ
    out = svc.translate_segments([_seg(1, "la la")], "Vietnamese", "Formal", source_language="vi")
    assert svc._generate.call_count == 1              # orig=2, "dịch 1"=2 -> penalty 0
    assert out[0]["translated_text"] == "dịch 1"      # baseline nguyên vẹn
    assert svc.last_review_stats["revised"] == 0
    assert svc.last_review_stats["rounds"] == 0


# ---- Luật chấp nhận khách quan ------------------------------------------------------

def test_self_review_accepts_shorter_revision_on_overflow():
    """Bản dịch TRÀN nhịp -> review viết gọn -> chấp nhận (penalty giảm ngặt)."""
    svc = _loaded_service()
    initial = _translated_output([(1, _w(12))])                  # 12 vs orig 6 -> penalty 12
    review = _review_output([{"id": 1, "revised_text": _w(6), "emotion": "SAD"}])
    svc._generate = MagicMock(side_effect=[initial, review])
    out = svc.translate_segments([_seg(1, _w(6))], "Vietnamese", "Formal", source_language="vi")
    assert svc._generate.call_count == 2
    assert out[0]["translated_text"] == _w(6)         # bản gọn được nhận
    assert out[0]["emotion"] == "SAD"                 # emotion hợp lệ được áp
    assert svc.last_review_stats["revised"] == 1


def test_self_review_rejects_worse_pacing_keeps_old():
    """Review đề xuất bản TỆ HƠN (penalty cao hơn) -> TỪ CHỐI, giữ bản cũ."""
    svc = _loaded_service()
    initial = _translated_output([(1, _w(12))])                  # penalty 12
    review = _review_output([{"id": 1, "revised_text": _w(20), "emotion": "SAD"}])  # penalty 28
    svc._generate = MagicMock(side_effect=[initial, review])
    out = svc.translate_segments([_seg(1, _w(6))], "Vietnamese", "Formal", source_language="vi")
    assert out[0]["translated_text"] == _w(12)        # giữ bản cũ
    assert out[0]["emotion"] == "HAPPY"               # emotion cũ giữ nguyên (không áp)
    assert svc.last_review_stats["revised"] == 0


def test_self_review_rejects_equal_pacing_keeps_old():
    """Bản cùng penalty (chỉ đổi từ, không cải thiện ĐO ĐƯỢC) -> TỪ CHỐI ('<' NGẶT):
    chống churn + không áp hoán-đổi cùng-nhịp không kiểm chứng được (rủi ro hồi quy nghĩa)."""
    svc = _loaded_service()
    initial = _translated_output([(1, _w(12))])                  # penalty 12
    review = _review_output([{"id": 1, "revised_text": " ".join(["xx"] * 12)}])  # cũng 12 -> penalty 12
    svc._generate = MagicMock(side_effect=[initial, review])
    out = svc.translate_segments([_seg(1, _w(6))], "Vietnamese", "Formal", source_language="vi")
    assert out[0]["translated_text"] == _w(12)        # cùng penalty -> giữ cũ
    assert svc.last_review_stats["revised"] == 0


def test_self_review_rejects_gutted_revision():
    """Sàn tỷ lệ: bản quá ngắn (got < 0.5*orig) bị từ chối DÙ penalty thấp hơn."""
    svc = _loaded_service()
    initial = _translated_output([(1, _w(14))])                  # orig 6 -> penalty 16
    review = _review_output([{"id": 1, "revised_text": _w(2)}])  # 2 < ceil(0.5*6)=3 -> loại
    svc._generate = MagicMock(side_effect=[initial, review])
    out = svc.translate_segments([_seg(1, _w(6))], "Vietnamese", "Formal", source_language="vi")
    assert out[0]["translated_text"] == _w(14)        # bản cụt bị loại, giữ cũ
    assert svc.last_review_stats["revised"] == 0


def test_self_review_rejects_gutted_small_original():
    """MUST-FIX: sàn TUYỆT ĐỐI MIN_REVISED_SYLLABLES. Với orig=2, 0.5*orig=1 nên sàn tỷ lệ
    KHÔNG cắn -> bản cụt về 1 từ vẫn phải bị chặn bởi sàn tuyệt đối (>=2)."""
    svc = _loaded_service()
    initial = _translated_output([(1, _w(6))])                   # orig 2 (source en "hello") -> penalty (6-2)*2=8
    review = _review_output([{"id": 1, "revised_text": "x"}])    # 1 từ: >= tỷ lệ (floor=1) NHƯNG < sàn tuyệt đối 2
    svc._generate = MagicMock(side_effect=[initial, review])
    out = svc.translate_segments([_seg(1, "hello")], "Vietnamese", "Formal", source_language="en")
    assert out[0]["translated_text"] == _w(6)         # bản 1-từ bị chặn bởi sàn tuyệt đối
    assert svc.last_review_stats["revised"] == 0


# ---- Bất biến hợp đồng: 9 trường + ID-parity + mốc thời gian tin cậy -----------------

def test_self_review_preserves_9field_and_ids():
    """Sau review, hợp đồng 9 trường + tập/thứ tự id nguyên vẹn (chỉ text/emotion đổi)."""
    svc = _loaded_service()
    initial = _translated_output([(1, _w(12)), (2, _w(2))])      # id1 tràn (candidate), id2 khớp
    review = _review_output([{"id": 1, "revised_text": _w(6)}])
    svc._generate = MagicMock(side_effect=[initial, review])
    segs = [_seg(1, _w(6)), _seg(2, _w(2), start=2.0, end=4.0, speaker="S2")]
    out = svc.translate_segments(segs, "Vietnamese", "Formal", source_language="vi")
    for o in out:
        assert set(o.keys()) == {
            "id", "start", "end", "duration", "original_text",
            "original_syllables", "translated_text", "emotion", "speaker_id",
        }
    assert [o["id"] for o in out] == [1, 2]           # tập + thứ tự id giữ nguyên
    assert out[0]["translated_text"] == _w(6)         # id1 được sửa
    assert out[1]["translated_text"] == _w(2)         # id2 (khớp nhịp) không đổi


def test_self_review_preserves_trusted_timings_on_revision():
    """Review echo start/end/speaker_id BẬY -> đầu ra vẫn mang mốc thời gian TIN CẬY từ
    đầu vào (Zero-Trust): _ReviewItem không khai báo các field đó nên pydantic vứt chúng."""
    svc = _loaded_service()
    initial = _translated_output([(1, _w(12))])
    review = _review_output([{
        "id": 1, "revised_text": _w(6), "emotion": "happy",
        "start": 999.0, "end": -5.0, "speaker_id": "HACKED", "duration": 123.0,
    }])
    svc._generate = MagicMock(side_effect=[initial, review])
    out = svc.translate_segments([_seg(1, _w(6), start=3.0, end=8.0)], "Vietnamese", "Formal",
                                 source_language="vi")
    assert out[0]["translated_text"] == _w(6)         # bản sửa được nhận
    assert out[0]["start"] == 3.0                     # KHÔNG phải 999.0 của review
    assert out[0]["end"] == 8.0                       # KHÔNG phải -5.0
    assert out[0]["duration"] == 5.0                  # từ đầu vào, KHÔNG phải 123.0
    assert out[0]["speaker_id"] == "S1"               # KHÔNG phải "HACKED"


# ---- Vòng lặp bounded + hội tụ -------------------------------------------------------

def test_self_review_bounded_rounds(monkeypatch):
    """Trần cứng QWEN_MAX_REVIEW_ROUNDS: dừng ĐÚNG sau cap vòng dù vẫn còn cải thiện được.
    Cấp DƯ side_effect (rev3/rev4) -> nếu chạy quá cap sẽ tiêu thụ chúng và call_count>3."""
    monkeypatch.setenv("QWEN_MAX_REVIEW_ROUNDS", "2")
    svc = _loaded_service()
    initial = _translated_output([(1, _w(20))])                  # orig 2 -> penalty 36
    rev1 = _review_output([{"id": 1, "revised_text": _w(12)}])   # penalty 20 (<36, vẫn >2)
    rev2 = _review_output([{"id": 1, "revised_text": _w(8)}])    # penalty 12 (<20, vẫn >2)
    rev3 = _review_output([{"id": 1, "revised_text": _w(6)}])    # KHÔNG được tiêu thụ (quá cap)
    rev4 = _review_output([{"id": 1, "revised_text": _w(4)}])
    svc._generate = MagicMock(side_effect=[initial, rev1, rev2, rev3, rev4])
    out = svc.translate_segments([_seg(1, "la la")], "Vietnamese", "Formal", source_language="vi")
    assert svc._generate.call_count == 3              # 1 dịch + 2 review (== 1 + cap)
    assert out[0]["translated_text"] == _w(8)         # dừng ở kết quả vòng 2
    assert svc.last_review_stats["rounds"] == 2


def test_self_review_early_exit_no_progress(monkeypatch):
    """Không có cải thiện NGẶT nào ở một vòng -> break NGAY (không đợi hết cap)."""
    monkeypatch.setenv("QWEN_MAX_REVIEW_ROUNDS", "2")
    svc = _loaded_service()
    initial = _translated_output([(1, _w(12))])                  # penalty 12
    worse = _review_output([{"id": 1, "revised_text": _w(20)}])  # penalty 28 -> loại -> progress 0
    extra = _review_output([{"id": 1, "revised_text": _w(6)}])   # KHÔNG được tiêu thụ (đã break)
    svc._generate = MagicMock(side_effect=[initial, worse, extra])
    out = svc.translate_segments([_seg(1, _w(6))], "Vietnamese", "Formal", source_language="vi")
    assert svc._generate.call_count == 2              # break sau vòng 1, KHÔNG chạy vòng 2
    assert out[0]["translated_text"] == _w(12)
    assert svc.last_review_stats["rounds"] == 1


def test_self_review_converges_empty_reviews():
    """Review trả mảng rỗng -> không sửa -> break (hội tụ), giữ baseline."""
    svc = _loaded_service()
    initial = _translated_output([(1, _w(12))])
    svc._generate = MagicMock(side_effect=[initial, json.dumps({"reviews": []})])
    out = svc.translate_segments([_seg(1, _w(6))], "Vietnamese", "Formal", source_language="vi")
    assert svc._generate.call_count == 2
    assert out[0]["translated_text"] == _w(12)
    assert svc.last_review_stats["revised"] == 0


# ---- FAIL-OPEN + Zero-Trust trên output review --------------------------------------

def test_self_review_fail_open_on_bad_json():
    """Review trả JSON HỎNG -> KHÔNG raise (khác vòng dịch fail-CLOSED); giữ baseline
    ĐÃ hợp lệ. Một enhancement tùy chọn không bao giờ được đánh sập bản dịch tốt."""
    svc = _loaded_service()
    initial = _translated_output([(1, _w(12))])
    svc._generate = MagicMock(side_effect=[initial, "totally not json"])
    out = svc.translate_segments([_seg(1, _w(6))], "Vietnamese", "Formal", source_language="vi")
    assert svc._generate.call_count == 2
    assert out[0]["translated_text"] == _w(12)        # baseline giữ nguyên, không nổ
    assert svc.last_review_stats["revised"] == 0
    assert svc.last_review_stats["rounds"] == 1


def test_self_review_ignores_unknown_review_id():
    """Review trả id LẠ (không phải candidate) -> lọc bỏ; câu candidate không có bản sửa
    -> giữ nguyên. Chống mô hình renumber/bịa id trong vòng review."""
    svc = _loaded_service()
    initial = _translated_output([(1, _w(12))])
    review = _review_output([{"id": 999, "revised_text": _w(6)}])
    svc._generate = MagicMock(side_effect=[initial, review])
    out = svc.translate_segments([_seg(1, _w(6))], "Vietnamese", "Formal", source_language="vi")
    assert [o["id"] for o in out] == [1]              # id nguyên vẹn
    assert out[0]["translated_text"] == _w(12)        # id lạ bị bỏ -> giữ baseline
    assert svc.last_review_stats["revised"] == 0


def test_self_review_emotion_coerced_within_enum():
    """emotion review hợp lệ -> áp; NGOÀI enum -> GIỮ emotion hợp lệ hiện tại (KHÔNG tụt
    NEUTRAL): review chỉ được NÂNG CẤP, không hạ cấp một nhãn đang đúng."""
    svc = _loaded_service()
    initial = _translated_output([(1, _w(12)), (2, _w(12))], emotion="HAPPY")
    review = _review_output([
        {"id": 1, "revised_text": _w(6), "emotion": "sad"},       # hợp lệ -> SAD
        {"id": 2, "revised_text": _w(6), "emotion": "EXCITED"},   # ngoài enum -> giữ HAPPY
    ])
    svc._generate = MagicMock(side_effect=[initial, review])
    segs = [_seg(1, _w(6)), _seg(2, _w(6), start=2.0, end=4.0, speaker="S2")]
    out = svc.translate_segments(segs, "Vietnamese", "Formal", source_language="vi")
    by_id = {o["id"]: o for o in out}
    assert by_id[1]["emotion"] == "SAD"
    assert by_id[2]["emotion"] == "HAPPY"             # KHÔNG phải NEUTRAL


# ---- Gate ngôn ngữ không đo được (CJK) — No-Fake-Success -----------------------------

def test_self_review_skipped_for_cjk_target():
    """Target không đo được (Japanese) -> BỎ QUA self-review hoàn toàn (0 lần _generate
    thêm): count_syllables suy biến về 1 cho CJK -> cổng sẽ 'đo cải thiện' so baseline GIẢ."""
    svc = _loaded_service()
    svc._generate = MagicMock(side_effect=[_translated_output([(1, _w(12))])])
    out = svc.translate_segments([_seg(1, "hello")], "Japanese", "Formal", source_language="en")
    assert svc._generate.call_count == 1              # không vòng review
    assert out[0]["translated_text"] == _w(12)        # baseline nguyên vẹn
    assert svc.last_review_stats["skipped"] == "lang_not_measurable"


def test_self_review_skipped_for_cjk_source():
    """MUST-FIX (lỗ hổng 2 verifier phát hiện): original_syllables đếm phía SOURCE. Với
    source CJK (chữ bản địa), count_syllables suy biến về 1 -> baseline penalty GIẢ -> nếu
    không chặn, cổng sẽ 'cải thiện' bằng cách CẮT CỤT bản dịch về 1 từ. Gate yêu cầu CẢ
    source đo được -> bỏ qua an toàn."""
    svc = _loaded_service()
    svc._generate = MagicMock(side_effect=[_translated_output([(1, _w(12))])])
    # source "ja" + chữ Nhật bản địa: count_syllables("こんにちは世界","ja") suy biến -> 1.
    out = svc.translate_segments([_seg(1, "こんにちは世界")], "Vietnamese", "Formal",
                                 source_language="ja")
    assert svc._generate.call_count == 1              # KHÔNG gut về 1 từ; bỏ qua sạch
    assert out[0]["translated_text"] == _w(12)        # baseline nguyên vẹn (không cắt cụt)
    assert svc.last_review_stats["skipped"] == "lang_not_measurable"


# ---- Kill-switch ENV ----------------------------------------------------------------

def test_self_review_disabled_via_env(monkeypatch):
    """QWEN_SELF_REVIEW=0 -> tắt hoàn toàn: KHÔNG lần _generate review nào dù câu lệch nhịp."""
    monkeypatch.setenv("QWEN_SELF_REVIEW", "0")
    svc = _loaded_service()
    svc._generate = MagicMock(side_effect=[_translated_output([(1, _w(12))])])
    out = svc.translate_segments([_seg(1, _w(6))], "Vietnamese", "Formal", source_language="vi")
    assert svc._generate.call_count == 1              # tắt -> chỉ vòng dịch
    assert out[0]["translated_text"] == _w(12)
    assert svc.last_review_stats is None              # _self_review không được gọi


# ---- Zero-Logging: plaintext KHÔNG rò ra log ---------------------------------------

def test_self_review_zero_logging(caplog):
    """Kịch bản/bản dịch chảy qua seam _generate (mock) chứ KHÔNG qua logger. Sentinel
    trong text KHÔNG được xuất hiện ở BẤT KỲ mức log nào; chỉ metadata đếm-only được log."""
    svc = _loaded_service()
    sentinel = "SENTINEL_SECRET_XYZ"
    initial = _translated_output([(1, _w(12))])
    # revised chứa sentinel + đủ dài để được nhận (6 từ, penalty 0 < 12).
    review = _review_output([{"id": 1, "revised_text": sentinel + " " + _w(5)}])
    svc._generate = MagicMock(side_effect=[initial, review])
    src_text = sentinel + " " + _w(5)                 # original_text cũng mang sentinel (6 từ -> orig 6)
    with caplog.at_level(logging.INFO, logger="omnivoice.translation"):
        out = svc.translate_segments([_seg(1, src_text)], "Vietnamese", "Formal",
                                     source_language="vi")
    assert out[0]["translated_text"] == sentinel + " " + _w(5)   # bản sửa được nhận
    assert sentinel not in caplog.text                # KHÔNG rò plaintext ra log
    assert "tinh chỉnh" in caplog.text                # metadata đếm-only CÓ được log
    assert svc.last_review_stats["revised"] == 1
