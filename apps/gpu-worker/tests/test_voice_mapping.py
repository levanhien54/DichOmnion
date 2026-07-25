from src.model_manager import ModelManager


def test_resolve_voice_maps_abstract_client_ids_to_real_edge_voices():
    """voice_id TRỪU TƯỢNG từ UI client (nam_tram, nu_cao...) phải được quy sang giọng
    edge-tts CÓ THẬT theo giới tính + ngôn ngữ đích. Nếu không, edge-tts nhận 'nam_tram'
    sẽ lỗi -> segment câm -> đa giọng giả."""
    mm = ModelManager()
    voice_map = {"SPEAKER_01": "nam_tram", "SPEAKER_02": "nu_cao"}
    # Tiếng Việt: nam -> NamMinh, nữ -> HoaiMy (2 giọng phân biệt thật).
    assert mm._resolve_voice("Vietnamese", "SPEAKER_01", voice_map) == "vi-VN-NamMinhNeural"
    assert mm._resolve_voice("Vietnamese", "SPEAKER_02", voice_map) == "vi-VN-HoaiMyNeural"
    # Đổi ngôn ngữ đích -> giọng thật của ngôn ngữ đó, đúng giới tính.
    assert mm._resolve_voice("english", "SPEAKER_01", {"SPEAKER_01": "nam_tre"}) == "en-US-GuyNeural"
    assert mm._resolve_voice("english", "SPEAKER_02", {"SPEAKER_02": "nu_truyen_cam"}) == "en-US-AriaNeural"


def test_resolve_voice_passes_through_concrete_edge_voice():
    """Nếu ánh xạ đã là voice edge-tts cụ thể thì tôn trọng, không quy đổi lại."""
    mm = ModelManager()
    voice_map = {
        "SPEAKER_01": "vi-VN-NamMinhNeural",
        "SPEAKER_02": "vi-VN-HoaiMyNeural",
    }
    assert mm._resolve_voice("Vietnamese", "SPEAKER_01", voice_map) == "vi-VN-NamMinhNeural"
    assert mm._resolve_voice("Vietnamese", "SPEAKER_02", voice_map) == "vi-VN-HoaiMyNeural"


def test_resolve_voice_falls_back_to_language_default():
    """Không có ánh xạ cho speaker này -> giọng mặc định theo ngôn ngữ đích."""
    mm = ModelManager()
    # Speaker vắng trong map -> mặc định theo ngôn ngữ đích (english).
    assert mm._resolve_voice("english", "SPEAKER_99", {"SPEAKER_01": "x"}) == "en-US-AriaNeural"
    # Không truyền map -> mặc định theo ngôn ngữ.
    assert mm._resolve_voice("japanese", None, None) == "ja-JP-NanamiNeural"
    # Ngôn ngữ lạ + map rỗng -> mặc định tiếng Việt.
    assert mm._resolve_voice("klingon", "S1", {}) == "vi-VN-HoaiMyNeural"


def test_resolve_voice_ignores_empty_mapping_value():
    """Giá trị ánh xạ rỗng không được coi là giọng hợp lệ -> rơi về mặc định."""
    mm = ModelManager()
    assert mm._resolve_voice("english", "SPEAKER_01", {"SPEAKER_01": ""}) == "en-US-AriaNeural"


def test_resolve_voice_covers_every_client_target_language():
    """P6 (No-Fake-Success): mọi ngôn ngữ đích client cho chọn trong dropdown — GỬI Ở
    DẠNG VIẾT HOA đúng như App.tsx (LANGUAGES) — phải quy ra giọng edge-tts CÓ THẬT
    (nam + nữ phân biệt). Nếu worker không có giọng cho một ngôn ngữ đã mở trong UI thì
    đó là tính năng giả. Case-insensitive: _resolve_voice tự .lower()."""
    mm = ModelManager()
    # (giá trị client gửi, giọng nam THẬT, giọng nữ THẬT = mặc định khi không map)
    cases = [
        ("Vietnamese", "vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"),
        ("English", "en-US-GuyNeural", "en-US-AriaNeural"),
        ("Japanese", "ja-JP-KeitaNeural", "ja-JP-NanamiNeural"),
        ("Korean", "ko-KR-InJoonNeural", "ko-KR-SunHiNeural"),
        ("Chinese", "zh-CN-YunxiNeural", "zh-CN-XiaoxiaoNeural"),
        ("French", "fr-FR-HenriNeural", "fr-FR-DeniseNeural"),
        ("Spanish", "es-ES-AlvaroNeural", "es-ES-ElviraNeural"),
        ("German", "de-DE-ConradNeural", "de-DE-KatjaNeural"),
    ]
    for lang, male_voice, female_voice in cases:
        assert mm._resolve_voice(lang, "S1", {"S1": "nam_tram"}) == male_voice, lang
        assert mm._resolve_voice(lang, "S2", {"S2": "nu_cao"}) == female_voice, lang
        assert mm._resolve_voice(lang, None, None) == female_voice, lang  # mặc định (nữ)
