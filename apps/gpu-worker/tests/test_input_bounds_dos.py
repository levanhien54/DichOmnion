"""Đợt 17 F3/F4 — JobPayload phải BOUND đầu vào fail-closed (chống DoS OOM/treo).

Một thiết bị đã đăng ký nhưng KHÔNG đáng tin (Zero-Trust) KÝ hợp lệ được payload với
`segments` khổng lồ / text cực dài. translate_segments gộp TẤT CẢ segment vào MỘT prompt
Qwen rồi tokenize + generate một lần -> tràn VRAM (OOM) hoặc treo tới khi Gateway timeout
15' -> Trạm 3 cách ly worker 24h -> DoS chéo tenant. Cổng pydantic ở BIÊN từ chối (422)
TRƯỚC semaphore/model, biến "treo cả cụm" thành "một job hỏng sạch" (4xx = Gateway FAILED,
không quarantine).

Các test khóa bất biến: quá SỐ segment / quá dài text MỘT segment / quá dài TỔNG text /
free-text field (target/style/source) quá dài -> ValidationError; payload hợp lệ -> qua.
Kiểm ngay tại tầng model (cơ chế enforce) — không cần GPU/mạng.
"""
import pytest
from pydantic import ValidationError

from src.main import (
    JobPayload,
    _MAX_SEGMENTS,
    _MAX_SEGMENT_TEXT_CHARS,
    _MAX_SEGMENT_META_CHARS,
    _MAX_TOTAL_TEXT_CHARS,
)


def _valid_kwargs(**overrides):
    base = {
        "job_id": "job-1",
        "audio_url": "https://pub-abc.r2.dev/a.wav",
        "audio_md5": "deadbeef",
        "target_language": "Vietnamese",
        "translation_style": "Formal",
        "segments": [{"id": 1, "text": "hello", "start": 0.0, "end": 1.0}],
        "voice_map": {},
        "source_language": "en",
    }
    base.update(overrides)
    return base


def test_valid_payload_passes():
    """Payload bình thường (1 segment ngắn) qua cổng — bound KHÔNG chặn nhầm job thật."""
    p = JobPayload(**_valid_kwargs())
    assert len(p.segments) == 1


def test_rejects_too_many_segments():
    """Số segment vượt trần -> ValidationError (không để prompt phình theo N)."""
    seg = {"id": 1, "text": "x", "start": 0.0, "end": 1.0}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg] * (_MAX_SEGMENTS + 1)))


def test_accepts_exactly_max_segments():
    """Đúng trần (biên) vẫn được nhận — bound là '>' chứ không phải '>='."""
    seg = {"id": 1, "text": "x", "start": 0.0, "end": 1.0}
    p = JobPayload(**_valid_kwargs(segments=[seg] * _MAX_SEGMENTS))
    assert len(p.segments) == _MAX_SEGMENTS


def test_rejects_oversized_single_segment_text():
    """Một segment với text cực dài -> ValidationError (một dòng thoại không thể dài vậy)."""
    huge = {"id": 1, "text": "a" * (_MAX_SEGMENT_TEXT_CHARS + 1), "start": 0.0, "end": 1.0}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[huge]))


def test_rejects_oversized_total_text():
    """Tổng text vượt trần dù MỖI segment vừa phải -> ValidationError (chặn trục thứ 3)."""
    # Mỗi segment = nửa trần-per-segment; số lượng vừa đủ để TỔNG vượt _MAX_TOTAL_TEXT_CHARS
    # nhưng KHÔNG vượt trần _MAX_SEGMENTS (tách bạch với test đếm segment).
    per = _MAX_SEGMENT_TEXT_CHARS
    n = (_MAX_TOTAL_TEXT_CHARS // per) + 2
    assert n <= _MAX_SEGMENTS, "điều kiện test: n phải dưới trần segment để cô lập trục tổng"
    seg = {"id": 1, "text": "a" * per, "start": 0.0, "end": 1.0}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg] * n))


def test_rejects_nonstring_segment_text():
    """text phi-chuỗi (client bị tước/độc hại) -> fail-closed sớm, không đẩy xuống model."""
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[{"id": 1, "text": {"nested": "obj"}}]))


def test_rejects_oversized_target_language():
    """target_language phình (cũng nhúng vào prompt) -> ValidationError."""
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(target_language="V" * 5000))


def test_rejects_oversized_translation_style():
    """translation_style phình -> ValidationError."""
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(translation_style="F" * 5000))


def test_empty_segments_still_valid():
    """segments rỗng vẫn hợp lệ ở tầng model (ASR sẽ điền) — bound chỉ chặn TRẦN TRÊN."""
    p = JobPayload(**_valid_kwargs(segments=[]))
    assert p.segments == []


# --- Đợt 18 F6: `id` và `speaker`/`speaker_id` cũng nhúng vào prompt -> phải bound ---

def test_rejects_oversized_segment_id():
    """`id` chuỗi khổng lồ (text tí hon để lọt F3/F4) -> ValidationError.

    Đây CHÍNH là vector F3/F4 tưởng đã đóng: bound cũ chỉ đo `text`, nên `id` dài vẫn
    lọt vào _build_prompt -> phình prompt -> OOM/treo -> quarantine chéo tenant."""
    seg = {"id": "z" * (_MAX_SEGMENT_META_CHARS + 1), "text": "hi", "start": 0.0, "end": 1.0}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg]))


def test_rejects_oversized_speaker():
    """`speaker` chuỗi khổng lồ -> ValidationError (nhúng làm speaker_id trong prompt)."""
    seg = {"id": 1, "speaker": "S" * (_MAX_SEGMENT_META_CHARS + 1),
           "text": "hi", "start": 0.0, "end": 1.0}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg]))


def test_rejects_oversized_speaker_id_alias():
    """`speaker_id` (alias client dùng khi thiếu `speaker`) cũng bị bound -> ValidationError."""
    seg = {"id": 1, "speaker_id": "S" * (_MAX_SEGMENT_META_CHARS + 1),
           "text": "hi", "start": 0.0, "end": 1.0}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg]))


def test_accepts_meta_at_cap():
    """id/speaker ĐÚNG bằng trần (biên) vẫn nhận — điều kiện chặn là '>' chứ không '>='."""
    seg = {"id": "z" * _MAX_SEGMENT_META_CHARS,
           "speaker": "S" * _MAX_SEGMENT_META_CHARS,
           "text": "hi", "start": 0.0, "end": 1.0}
    p = JobPayload(**_valid_kwargs(segments=[seg]))
    assert len(p.segments) == 1


def test_meta_counts_toward_total_budget():
    """id/speaker vừa-mỗi-trường nhưng GỘP nhiều segment vượt TỔNG -> ValidationError.

    Chặn trục count × field: mỗi id/speaker ≤ trần per-field nhưng 2000 × ~512 vẫn có thể
    thổi prompt qua _MAX_TOTAL_TEXT_CHARS nếu không tính vào tổng."""
    per_meta = _MAX_SEGMENT_META_CHARS
    # Mỗi segment đóng góp 2*per_meta (id + speaker) + 1 (text) vào total.
    n = (_MAX_TOTAL_TEXT_CHARS // (2 * per_meta)) + 2
    assert n <= _MAX_SEGMENTS, "điều kiện test: n phải dưới trần segment để cô lập trục tổng"
    seg = {"id": "z" * per_meta, "speaker": "S" * per_meta,
           "text": "x", "start": 0.0, "end": 1.0}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg] * n))


def test_nonstring_id_does_not_crash_bound():
    """`id` phi-chuỗi (int — như client thật/test cũ gửi) KHÔNG làm cổng bound nổ TypeError.

    len() chỉ chạy khi isinstance str; id số bỏ qua đo (đã bị JSON round-trip thu về)."""
    seg = {"id": 12345, "text": "hi", "start": 0.0, "end": 1.0}
    p = JobPayload(**_valid_kwargs(segments=[seg]))
    assert len(p.segments) == 1


# --- Đợt 19 F8: start/end/duration phải là SỐ HỮU HẠN (chống crash muộn -> 500 -> retry 3×) ---
#
# start/end đi vào int(to_seconds(x)*1000) tại audio_engine.mix_audio; duration đi vào
# round(duration,2) tại translation_service. Chuỗi "1e999"/"inf" -> float() Python = inf ->
# int(inf) OverflowError; "nan" -> int(nan) ValueError; duration phi-số -> round() TypeError.
# Không cổng nào cũ chặn (F3/F4/F6 bound ĐỘ DÀI chuỗi, không phải TÍNH HỢP LỆ số). Nổ MUỘN
# sau cả pipeline -> 500 -> Gateway retry toàn bộ 3 lần. Chặn ở cổng = 422 terminal, không retry.

@pytest.mark.parametrize("bad", ["1e999", "inf", "-inf", "Infinity", "nan", "NaN"])
def test_rejects_nonfinite_start(bad):
    """start chuẩn hóa ra vô cực/NaN (chuỗi float() Python nhận) -> ValidationError.

    Đây là input lọt MỌI cổng cũ (text/id/speaker hợp lệ) nhưng làm int(inf*1000) nổ
    OverflowError tại mix_audio — SAU khi đã tốn tải audio + Qwen + TTS + Demucs."""
    seg = {"id": 1, "text": "hi", "start": bad, "end": 1.0}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg]))


@pytest.mark.parametrize("bad", ["1e999", "nan"])
def test_rejects_nonfinite_end(bad):
    """end vô cực/NaN cũng bị chặn (end đi cùng đường số học start tại mix_audio)."""
    seg = {"id": 1, "text": "hi", "start": 0.0, "end": bad}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg]))


@pytest.mark.parametrize("bad", ["abc", [], {"x": 1}, "5.0"])
def test_rejects_nonnumeric_duration(bad):
    """duration phi-số (chuỗi/list/dict) -> ValidationError.

    duration KHÔNG qua to_seconds — translation_service round(duration,2) tiêu thụ trực
    tiếp nên chuỗi "5.0"/list/dict đều ném TypeError tại round(). Hợp đồng = số."""
    seg = {"id": 1, "text": "hi", "start": 0.0, "end": 1.0, "duration": bad}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg]))


def test_rejects_bool_duration():
    """duration bool bị loại: round(True,2)=1 sẽ nuốt lỗi client âm thầm (fail-closed)."""
    seg = {"id": 1, "text": "hi", "start": 0.0, "end": 1.0, "duration": True}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg]))


def test_accepts_finite_numeric_timecodes():
    """Hợp đồng hợp lệ: start/end SỐ giây + duration số -> qua (bound KHÔNG chặn nhầm)."""
    seg = {"id": 1, "text": "hi", "start": 12.5, "end": 15.0, "duration": 2.5}
    p = JobPayload(**_valid_kwargs(segments=[seg]))
    assert len(p.segments) == 1


def test_accepts_hhmmss_timecode_string():
    """Chuỗi "HH:MM:SS" (đường phòng thủ CC-1 của to_seconds) chuẩn hóa hữu hạn -> qua."""
    seg = {"id": 1, "text": "hi", "start": "00:00:12", "end": "00:00:15"}
    p = JobPayload(**_valid_kwargs(segments=[seg]))
    assert len(p.segments) == 1


def test_accepts_unparseable_start_as_zero():
    """Rác không parse ("abc") -> to_seconds 0.0 (hữu hạn) -> KHÔNG chặn (worker coi như 0,
    không nổ). Chỉ vô cực/NaN mới nguy hiểm; giữ ranh giới ở đúng chỗ, không quá tay."""
    seg = {"id": 1, "text": "hi", "start": "abc", "end": 1.0}
    p = JobPayload(**_valid_kwargs(segments=[seg]))
    assert len(p.segments) == 1


# --- Đợt 24 CC23-01: HOÀN THIỆN F8 — chặn TRÀN-×1000 (start/end hữu-hạn-ở-giây nhưng ×1000 -> inf) ---
#
# F8 chỉ kiểm math.isfinite(to_seconds(x)) (hữu hạn Ở GIÂY). Nhưng SINK thật là
# int(to_seconds(x)*1000) (mix_audio:177) và int((end-start)*1000) (:191). 1e306 hữu hạn ở
# giây nhưng ×1000 = 1e309 = inf -> int(inf) OverflowError -> nổ MUỘN trong mix_audio -> 500 ->
# retry 3×. Cả trục CHÉO: start=-1e305,end=1e305 -> mỗi cái ×1000 hữu hạn nhưng HIỆU ×1000 =
# 2e308 = inf. Đây là lỗ hổng ĐÃ CHỨNG MINH trong phủ của F8 (kiểm giá-trị-trước-nhân, sink
# nhân ×1000), nên vá = HOÀN THIỆN F8, không phải trục mới.

@pytest.mark.parametrize("field", ["start", "end"])
def test_rejects_scale_overflow_timecode(field):
    """start/end = 1e306: hữu hạn Ở GIÂY (F8 cũ cho qua) nhưng ×1000 -> inf -> int(inf)
    OverflowError tại mix_audio. Phải bị chặn 422 (hoàn thiện F8)."""
    seg = {"id": 1, "text": "hi", "start": 0.0, "end": 1.0}
    seg[field] = 1e306
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg]))


@pytest.mark.parametrize("bad", ["1e306", "-1e306"])
def test_rejects_scale_overflow_timecode_string(bad):
    """Chuỗi 1e306 (float() Python nhận, hữu hạn) cũng ×1000 -> inf -> phải chặn."""
    seg = {"id": 1, "text": "hi", "start": bad, "end": 1.0}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg]))


def test_rejects_cross_field_difference_overflow():
    """Trục CHÉO: start=-1e305, end=1e305 — mỗi cái ×1000 HỮU HẠN (1e308), nhưng HIỆU
    (end-start)×1000 = 2e308 = inf -> int(inf) tại int((end-start)*1000). F8 per-field
    KHÔNG bắt được; hoàn thiện F8 phải kiểm cả HIỆU."""
    seg = {"id": 1, "text": "hi", "start": -1e305, "end": 1e305}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg]))


def test_accepts_large_but_safe_timecodes():
    """1e300 ×1000 = 1e303 (vẫn hữu hạn) -> KHÔNG chặn: ranh giới đặt ĐÚNG chỗ tràn thực,
    không quá tay với giá trị lớn-nhưng-an-toàn."""
    seg = {"id": 1, "text": "hi", "start": 0.0, "end": 1e300}
    p = JobPayload(**_valid_kwargs(segments=[seg]))
    assert len(p.segments) == 1


# --- Đợt 20 F9: PHẦN TỬ segment phải là object (chống crash .get muộn -> 500 -> retry 3×) ---
#
# Bound F3/F4/F6/F8 đều soi GIÁ TRỊ TRƯỜNG của một dict; không cổng nào loại phần tử có KIỂU
# không-phải-dict. Bản cũ `if not isinstance(seg,dict): continue` (bỏ qua) đẩy phần tử rác
# xuống translate_segments -> seg.get(...) ném AttributeError (str/int/None/list không có .get)
# -> nổ MUỘN (sau tải audio) -> 500 -> Gateway retry toàn bộ pipeline 3×. Giờ fail-closed 422.

@pytest.mark.parametrize("bad", ["x", 1, None, ["nested"], 3.14, True])
def test_rejects_non_dict_segment_element(bad):
    """Phần tử segment phi-object (chuỗi/số/None/list/bool) -> ValidationError tại parse.

    Đây là input lọt MỌI cổng field cũ (không có trường nào để đo) nhưng làm
    translate_segments `seg.get("text", ...)` nổ AttributeError -> 500 -> retry 3×."""
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[bad]))


def test_rejects_non_dict_element_mixed_with_valid():
    """Một phần tử rác lẫn giữa các dict hợp lệ vẫn bị chặn (không lọt qua khe giữa)."""
    segs = [
        {"id": 1, "text": "ok", "start": 0.0, "end": 1.0},
        "sneaky",
        {"id": 2, "text": "ok2", "start": 1.0, "end": 2.0},
    ]
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=segs))


# --- Đợt 20 F10: voice_map VALUE phải là chuỗi (chống crash _resolve_voice muộn -> 500 -> 3×) --
#
# voice_map (speaker_id -> voice) forward verbatim từ Gateway, worker nhận `voice_map: dict`
# không kiểm value. _resolve_voice: VOICE_ID_GENDER.get(value) ném TypeError nếu value KHÔNG
# hashable (list/dict); value hashable phi-chuỗi (int) -> tts_service voice.startswith(...) ném
# AttributeError. Nổ MUỘN trong vòng TTS (sau tải audio + Qwen) -> 500 -> retry 3×. Ép value là
# chuỗi tại cổng (422). CHỈ kiểm KIỂU — số-lượng-mục là non-bug Đợt-19 đã bác (không phải trục bloat).

@pytest.mark.parametrize("bad_val", [{"x": "y"}, ["a"], 42, True, None, 3.14])
def test_rejects_non_string_voice_map_value(bad_val):
    """voice_map value phi-chuỗi (dict/list/int/bool/None/float) -> ValidationError tại parse."""
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(voice_map={"SPEAKER_00": bad_val}))


def test_rejects_non_string_voice_map_key():
    """Khóa phi-chuỗi cũng bị loại (dù JSON luôn cho khóa chuỗi, chốt fail-closed đối xứng)."""
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(voice_map={5: "nam_tram"}))


def test_accepts_string_voice_map_values():
    """voice_map hợp lệ (speaker_id -> chuỗi voice) -> qua. Không chặn hợp đồng đúng."""
    vm = {"SPEAKER_00": "nam_tram", "SPEAKER_01": "vi-VN-NamMinhNeural"}
    p = JobPayload(**_valid_kwargs(voice_map=vm))
    assert p.voice_map == vm


def test_accepts_empty_voice_map():
    """voice_map rỗng (mặc định — mọi giọng theo ngôn ngữ đích) -> qua."""
    p = JobPayload(**_valid_kwargs(voice_map={}))
    assert p.voice_map == {}


# --- Đợt 21 F11: KIỂU của segment id/speaker (chống crash _merge muộn -> 500 -> retry 3×) ------
#
# id/speaker đi VERBATIM (không chuẩn hóa) vào translation_service._merge dựng TranslatedSegment
# (id: int|str, speaker_id: str). _merge chạy NGOÀI khối try/except retry (translation_service.py
# :317), nên value SAI KIỂU nổ MUỘN thành pydantic ValidationError KHÔNG bắt -> HTTP 500 ->
# Gateway retry TOÀN BỘ pipeline 3×. Cổng cũ (F6) chỉ đo ĐỘ DÀI KHI đã là chuỗi nên value phi-
# chuỗi/phi-int lọt sạch. Ép KIỂU tại parse (422) — cùng trục KIỂU với F9 (phần tử)/F10 (voice_map).

@pytest.mark.parametrize("bad_speaker", [{"x": "y"}, ["SPEAKER_00"], 7, 3.14, True, None])
def test_rejects_non_string_speaker(bad_speaker):
    """speaker phi-chuỗi (dict/list/int/float/bool/None) -> ValidationError tại parse (sink str-only)."""
    seg = {"id": "s1", "text": "hi", "start": 0.0, "end": 1.0, "speaker": bad_speaker}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg]))


def test_rejects_non_string_speaker_id_alias():
    """Nhánh alias `speaker_id` (khi thiếu `speaker`) cũng bị ép kiểu — không có khe lách."""
    seg = {"id": "s1", "text": "hi", "start": 0.0, "end": 1.0, "speaker_id": {"x": "y"}}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg]))


@pytest.mark.parametrize("bad_id", [{"x": "y"}, ["s1"], None, 3.14, True])
def test_rejects_bad_type_segment_id(bad_id):
    """id phi-(str|int) (dict/list/None/float/bool) -> ValidationError (sink TranslatedSegment.id: int|str)."""
    seg = {"id": bad_id, "text": "hi", "start": 0.0, "end": 1.0}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg]))


def test_accepts_string_speaker_and_id():
    """speaker chuỗi + id chuỗi (đúng hợp đồng shared-types) -> qua."""
    seg = {"id": "s1", "text": "hi", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_03"}
    p = JobPayload(**_valid_kwargs(segments=[seg]))
    assert p.segments[0]["speaker"] == "SPEAKER_03"


def test_accepts_int_segment_id():
    """id số nguyên -> qua (sink TranslatedSegment.id là int|str; base fixture vốn dùng id=1)."""
    seg = {"id": 42, "text": "hi", "start": 0.0, "end": 1.0}
    p = JobPayload(**_valid_kwargs(segments=[seg]))
    assert p.segments[0]["id"] == 42


def test_accepts_segment_without_speaker_or_id():
    """Thiếu cả speaker lẫn id -> worker mặc định chuỗi rỗng/SPEAKER_UNKNOWN -> KHÔNG chặn nhầm."""
    seg = {"text": "hi", "start": 0.0, "end": 1.0}
    p = JobPayload(**_valid_kwargs(segments=[seg]))
    assert p.segments[0]["text"] == "hi"


# --- Đợt 22 F12: NULLABILITY của text/original_text (cổng cũ `or ""` che falsy phi-chuỗi) ---
# `text: null` (và các falsy phi-chuỗi 0/False/{}/[]) bị `seg.get(...) or ""` biến thành "" TRƯỚC
# isinstance nên LỌT cổng, rồi consumer translation_service.py:265 (KHÔNG `or ""`) đọc None ->
# _merge dựng TranslatedSegment(original_text=None) [str bắt buộc] NGOÀI try/except -> ValidationError
# -> 500 -> Gateway retry 3×. Bỏ `or ""` -> cổng đọc CÙNG biểu thức consumer -> ép str tại parse (422).

@pytest.mark.parametrize("bad_text", [None, 0, False, {}, []])
def test_rejects_falsy_nonstring_text(bad_text):
    """text falsy-phi-chuỗi (None/0/False/{}/[]) — bản cũ `or ""` che, nay -> ValidationError tại parse."""
    seg = {"id": "s1", "text": bad_text, "start": 0.0, "end": 1.0}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg]))


def test_rejects_null_original_text_when_text_absent():
    """`text` vắng + `original_text: None` -> consumer đọc None -> phải reject tại cổng (đối xứng .get)."""
    seg = {"id": "s1", "original_text": None, "start": 0.0, "end": 1.0}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg]))


def test_rejects_null_text_even_with_valid_original_text():
    """`text: None` HIỆN DIỆN che `original_text` hợp lệ: consumer `.get("text",...)` trả None (key có
    mặt) -> vẫn nổ ở _merge. Cổng phản chiếu: key "text" hiện diện None -> reject (không fallback)."""
    seg = {"id": "s1", "text": None, "original_text": "hi", "start": 0.0, "end": 1.0}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg]))


def test_accepts_empty_string_text():
    """text="" (chuỗi rỗng hợp lệ — segment không thoại) -> qua; bỏ `or ""` KHÔNG chặn nhầm chuỗi rỗng."""
    seg = {"id": "s1", "text": "", "start": 0.0, "end": 1.0}
    p = JobPayload(**_valid_kwargs(segments=[seg]))
    assert p.segments[0]["text"] == ""


def test_accepts_original_text_fallback_when_text_absent():
    """`text` vắng nhưng `original_text` là chuỗi -> consumer fallback dùng nó -> qua (không chặn nhầm)."""
    seg = {"id": "s1", "original_text": "xin chao", "start": 0.0, "end": 1.0}
    p = JobPayload(**_valid_kwargs(segments=[seg]))
    assert p.segments[0]["original_text"] == "xin chao"


def test_accepts_segment_without_any_text():
    """Thiếu CẢ text lẫn original_text -> mặc định "" -> qua (consumer cũng mặc định "")."""
    seg = {"id": "s1", "start": 0.0, "end": 1.0}
    p = JobPayload(**_valid_kwargs(segments=[seg]))
    assert "text" not in p.segments[0]


# --- Đợt 24 F13: LONE SURROGATE trong free-text / segment (crash tokenizer muộn -> 500 -> 3×) ---
# Một lone surrogate ("\ud800", nửa cặp UTF-16) là UTF KHÔNG well-formed: nó lọt mọi cổng KIỂU/
# ĐỘ-DÀI (vẫn là `str` len hợp lệ, và ECDSA verify qua vì TextEncoder thay bằng U+FFFD y hệt hai
# phía) nhưng khi Qwen fast-tokenizer chuyển str -> Rust String (PyO3) ném UnicodeEncodeError
# "surrogates not allowed" trong _generate — NGOÀI khối try/except retry (translation_service.py
# :304 gọi _generate TRƯỚC `try` 305) -> 500 -> Gateway retry TOÀN BỘ pipeline 3× (re-run Whisper
# ASR khi payload không-segments = khuếch đại ~3× GPU). str.encode("utf-8") ném CÙNG lỗi trên CÙNG
# tập -> dùng làm cổng well-formed (422 terminal). Astral char hợp lệ (emoji) PHẢI qua (không false-positive).

_LONE_SURROGATES = ["\ud800", "\udfff", "hi\udc00there", "\ud83d"]  # high, low, giữa-chuỗi, high-lẻ của cặp emoji


@pytest.mark.parametrize("field", ["target_language", "translation_style", "source_language"])
@pytest.mark.parametrize("bad", _LONE_SURROGATES)
def test_rejects_lone_surrogate_freetext(field, bad):
    """Free-text (target/style/source) chứa lone surrogate -> ValidationError tại parse (tokenizer sink)."""
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(**{field: bad}))


@pytest.mark.parametrize("bad", _LONE_SURROGATES)
def test_rejects_lone_surrogate_segment_text(bad):
    """Segment text chứa lone surrogate -> ValidationError (original_text -> _build_prompt -> tokenizer)."""
    seg = {"id": "s1", "text": bad, "start": 0.0, "end": 1.0}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg]))


@pytest.mark.parametrize("bad", _LONE_SURROGATES)
def test_rejects_lone_surrogate_segment_id(bad):
    """Segment id (chuỗi) chứa lone surrogate -> ValidationError (id -> _build_prompt -> tokenizer)."""
    seg = {"id": bad, "text": "hi", "start": 0.0, "end": 1.0}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg]))


@pytest.mark.parametrize("bad", _LONE_SURROGATES)
def test_rejects_lone_surrogate_segment_speaker(bad):
    """Segment speaker chứa lone surrogate -> ValidationError (speaker_id -> _build_prompt -> tokenizer)."""
    seg = {"id": "s1", "text": "hi", "start": 0.0, "end": 1.0, "speaker": bad}
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(segments=[seg]))


def test_accepts_valid_astral_and_nonascii_text():
    """Emoji (cặp surrogate HỢP LỆ / astral U+1F600) + chữ có dấu -> encode UTF-8 được -> QUA.
    Cổng chỉ chặn lone surrogate, KHÔNG chặn Unicode hợp lệ (không false-positive chặn job thật)."""
    seg = {"id": "đoạn-1", "text": "Xin chào 😀 thế giới", "start": 0.0, "end": 1.0, "speaker": "NGƯỜI_01"}
    p = JobPayload(**_valid_kwargs(segments=[seg], target_language="Tiếng Việt 🇻🇳", translation_style="Trang trọng"))
    assert p.segments[0]["text"] == "Xin chào 😀 thế giới"
    assert p.target_language == "Tiếng Việt 🇻🇳"


# --- Đợt 25 AMP-JOBID-SURROGATE-01: LONE SURROGATE trong job_id (crash serializer response -> 500 -> 3×) ---
# job_id là chuỗi client kiểm soát mà F13 BỎ SÓT: `str` trần, KHÔNG validator ở CẢ HAI tầng (Gateway
# chỉ kiểm !jobId/typeof). Lone surrogate qua ECDSA verify + token-binding, chạy hết pipeline, rồi
# `return {"job_id": payload.job_id, ...}` (main.py:362) đi vào Starlette JSONResponse.render =
# json.dumps(..., ensure_ascii=False).encode("utf-8"); .encode ném UnicodeEncodeError LÚC render —
# SAU khi handler return, NGOÀI try/except endpoint -> 500 uncaught -> Gateway retry 5xx 3× toàn
# pipeline (crash SAU render = Denial-of-Wallet tối đa). Sink KHÁC F13 (serializer response, KHÔNG
# phải tokenizer — job_id không vào prompt Qwen). Ép well-formed tại cổng job_id (422 terminal).


@pytest.mark.parametrize("bad", _LONE_SURROGATES)
def test_rejects_lone_surrogate_job_id(bad):
    """job_id chứa lone surrogate -> ValidationError tại parse (chặn crash serializer response
    500 -> 3× retry). Cổng riêng cho job_id vì sink (JSONResponse.encode) khác F13 (tokenizer)."""
    with pytest.raises(ValidationError):
        JobPayload(**_valid_kwargs(job_id=f"job-{bad}"))


def test_accepts_valid_astral_job_id():
    """job_id chứa astral char HỢP LỆ (emoji = cặp surrogate đủ) encode UTF-8 được -> QUA
    (không false-positive chặn job thật)."""
    p = JobPayload(**_valid_kwargs(job_id="job-😀-2026"))
    assert p.job_id == "job-😀-2026"


def test_accepts_none_source_language():
    """source_language=None (mặc định) -> cổng bỏ qua -> QUA (không nổ trên None)."""
    p = JobPayload(**_valid_kwargs(source_language=None))
    assert p.source_language is None
