import torch
import logging

# Tuân thủ nguyên tắc Zero-Logging: Tắt log các thông tin nhạy cảm
logging.getLogger("omnivoice.models").setLevel(logging.WARNING)
logger = logging.getLogger("omnivoice.models")

# Bản đồ ngôn ngữ đích -> voice edge-tts mặc định (giọng nữ). Mặc định về giọng Việt.
VOICE_BY_LANG = {
    "vietnamese": "vi-VN-HoaiMyNeural",
    "english": "en-US-AriaNeural",
    "japanese": "ja-JP-NanamiNeural",
    "korean": "ko-KR-SunHiNeural",
    "chinese": "zh-CN-XiaoxiaoNeural",
    "french": "fr-FR-DeniseNeural",
    "spanish": "es-ES-ElviraNeural",
    "german": "de-DE-KatjaNeural",
}

# Giọng THẬT edge-tts hỗ trợ, tách theo giới tính cho từng ngôn ngữ đích. Dùng để
# hiện thực hóa "đa giọng": client gửi voice_id TRỪU TƯỢNG (nam_tram, nu_cao...) — nếu
# đưa thẳng chuỗi đó cho edge-tts sẽ lỗi -> segment câm. Ta ánh xạ theo giới tính sang
# giọng edge-tts CÓ THẬT của đúng ngôn ngữ đích.
VOICES_BY_LANG_GENDER = {
    "vietnamese": {"male": "vi-VN-NamMinhNeural", "female": "vi-VN-HoaiMyNeural"},
    "english": {"male": "en-US-GuyNeural", "female": "en-US-AriaNeural"},
    "japanese": {"male": "ja-JP-KeitaNeural", "female": "ja-JP-NanamiNeural"},
    "korean": {"male": "ko-KR-InJoonNeural", "female": "ko-KR-SunHiNeural"},
    "chinese": {"male": "zh-CN-YunxiNeural", "female": "zh-CN-XiaoxiaoNeural"},
    "french": {"male": "fr-FR-HenriNeural", "female": "fr-FR-DeniseNeural"},
    "spanish": {"male": "es-ES-AlvaroNeural", "female": "es-ES-ElviraNeural"},
    "german": {"male": "de-DE-ConradNeural", "female": "de-DE-KatjaNeural"},
}

# voice_id trừu tượng (từ UI VoiceMapper của client) -> giới tính. LƯU Ý TRUNG THỰC:
# edge-tts chỉ có ~1 nam + 1 nữ mỗi ngôn ngữ, nên 5 "nhân vật" này rút về 2 giọng
# (nam/nữ) mỗi ngôn ngữ. Phân biệt 5 giọng nhân vật riêng biệt (và nhân bản giọng thật)
# cần engine cục bộ GPT-SoVITS — xem residual_hardware.
VOICE_ID_GENDER = {
    "nam_tram": "male",
    "nam_tre": "male",
    "nu_cao": "female",
    "nu_truyen_cam": "female",
    "tre_em": "female",  # edge-tts không có giọng trẻ em -> dùng giọng nữ (gần nhất)
}

# Bản đồ cảm xúc (do LLM gán) -> ngữ điệu edge-tts. Giúp tag EMOTION thực sự
# tác động lên âm thanh thay vì chỉ là metadata trưng bày.
EMOTION_PROSODY = {
    "SHOUTING": {"rate": "+8%", "volume": "+30%", "pitch": "+15Hz"},
    "WHISPERING": {"rate": "-8%", "volume": "-40%", "pitch": "-10Hz"},
    "ANGRY": {"rate": "+10%", "volume": "+15%", "pitch": "+8Hz"},
    "SAD": {"rate": "-12%", "volume": "+0%", "pitch": "-12Hz"},
    "HAPPY": {"rate": "+5%", "volume": "+5%", "pitch": "+12Hz"},
    "NEUTRAL": {"rate": "+0%", "volume": "+0%", "pitch": "+0Hz"},
}


class ModelManager:
    """
    Quản lý toàn bộ vòng đời của các mô hình AI trên VRAM.
    Tuân thủ nguyên tắc "Giữ toàn bộ Model thường trú trên VRAM".

    Lưu ý kiến trúc: mô hình thường trú trong tiến trình này là Whisper (ASR).
    Tách nhạc nền (Demucs) và sinh giọng (edge-tts / GPT-SoVITS) chạy qua tiến
    trình/dịch vụ ngoài, được gọi theo từng job — nên KHÔNG nạp tensor giả ("dummy")
    vào VRAM để "giả vờ" đã sẵn sàng.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.is_loaded = False

    def load_all_models(self):
        """Nạp các mô hình thường trú. Fail-closed nếu trọng số không nạp được."""
        if self.is_loaded:
            return

        logger.info(f"Loading resident models to {self.device} VRAM...")

        # Nạp Whisper (Bóc băng) THẬT và thường trú VRAM.
        # asr_service.load_model đã fail-closed (raise) nếu thiếu thư viện/trọng số.
        from src.asr_service import asr_service
        asr_service.load_model(model_size="base")

        if not asr_service.is_loaded:
            # Không bao giờ đánh dấu "đã nạp" khi mô hình lõi chưa thực sự sẵn sàng.
            raise RuntimeError("Whisper ASR không nạp được — worker không thể phục vụ job.")

        self.models["whisper"] = asr_service

        # Nạp mô hình DỊCH Qwen cục bộ và thường trú VRAM cạnh Whisper (tiêu chí #6 GPU
        # Model Residence). translation_service.load_model() đã fail-closed (raise) khi
        # thiếu transformers/CUDA/trọng số — GIỐNG Whisper, không bao giờ mock/giả. Trên
        # máy KHÔNG có GPU, load_all_models sẽ raise ở đây: đó là hành vi ĐÚNG cho prod
        # (worker không boot nếu không thể dịch cục bộ) và trung thực hơn bản OpenAI cũ
        # (boot "khỏe" nhưng chỉ rò rỉ transcript ra cloud lúc chạy). Chất lượng suy luận,
        # cư trú VRAM, độ trễ là residual_hardware — kiểm chứng trên GPU thật.
        from src.translation_service import translation_service
        translation_service.load_model()
        if not translation_service.is_loaded:
            raise RuntimeError("Qwen dịch không nạp được — worker không thể phục vụ job.")
        self.models["translation"] = translation_service

        self.is_loaded = True
        logger.info("Resident models loaded to VRAM. Worker ready.")

    def _resolve_voice(self, target_language: str, speaker_id: str | None = None,
                       voice_map: dict | None = None) -> str:
        """Chọn giọng edge-tts THẬT cho một segment theo người nói.

        Ưu tiên ánh xạ ĐA GIỌNG người dùng gán ở client (speaker_id -> voice). Giá trị
        ánh xạ có thể là:
          (a) voice_id TRỪU TƯỢNG từ UI (nam_tram, nu_cao...) -> quy theo GIỚI TÍNH sang
              giọng edge-tts có thật của đúng ngôn ngữ đích;
          (b) một voice edge-tts CỤ THỂ (vi-VN-NamMinhNeural) -> dùng thẳng.
        Không có ánh xạ cho speaker này -> giọng mặc định (nữ) theo ngôn ngữ đích.

        Trước đây hàm bỏ qua speaker hoàn toàn -> mọi nhân vật dùng CHUNG một giọng
        (đa giọng chỉ là mô tả, không chạy). Và nếu đưa thẳng voice_id trừu tượng cho
        edge-tts thì segment sẽ câm -> đa giọng giả. Cả hai đều được sửa ở đây."""
        key = (target_language or "").strip().lower()
        lang_voices = VOICES_BY_LANG_GENDER.get(key, VOICES_BY_LANG_GENDER["vietnamese"])

        if voice_map and speaker_id:
            chosen = voice_map.get(speaker_id)
            # F10 defense-in-depth: chỉ nhận chuỗi. Cổng JobPayload._bound_voice_map đã loại
            # value phi-chuỗi (422), nhưng chốt cả ở đây để value bất thường (nếu tới từ đường
            # khác) DEGRADE về giọng mặc định thay vì ném TypeError/AttributeError giữa vòng TTS.
            if isinstance(chosen, str) and chosen:
                gender = VOICE_ID_GENDER.get(chosen)
                if gender:
                    # (a) voice_id trừu tượng -> giọng thật theo giới tính + ngôn ngữ.
                    return lang_voices.get(gender, lang_voices["female"])
                # (b) đã là voice edge-tts cụ thể -> tôn trọng lựa chọn.
                return chosen

        return VOICE_BY_LANG.get(key, VOICE_BY_LANG["vietnamese"])

    @staticmethod
    def _cleanup_temp(paths):
        """Dọn file tạm trung gian của MỘT job (best-effort).

        Chỉ xóa file tạm nội bộ do worker sinh ra (audio tải về, clip TTS, stem
        Demucs, bản mix trước watermark). KHÔNG bao giờ xóa đầu ra cuối
        (dubbed_audio) — caller đã discard nó khỏi tập này. Lỗi xóa được nuốt để
        không ảnh hưởng kết quả job."""
        import os
        for p in paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

    async def process_job(self, audio_url: str, config: dict):
        """
        Pipeline xử lý thật (không sleep/giả lập):
        (tải audio) -> [ASR nếu thiếu segments] -> Dịch -> TTS -> Tách nền -> Mix -> Watermark

        Đầu vào là AUDIO đã tách phía client (quyền riêng tư: worker không chạm video thô).
        Đầu ra là track lồng tiếng đã mix + watermark; client tự ghép (mux) vào video cục bộ.
        """
        if not self.is_loaded:
            raise RuntimeError("Models chưa được nạp lên VRAM!")

        from src.audio_service import audio_service
        from src.audio_engine import audio_engine

        # Ràng buộc TOÀN VẸN NỘI DUNG (fail-closed): object key R2 dùng chung cho mọi job/bản
        # build (presigned PUT chỉ trỏ 1 key), nên một job khác có thể ghi đè object trước khi
        # ta tải -> ta có thể nhận nhầm audio của tenant khác. md5 client KÝ kèm là "vé" ràng
        # buộc: thiếu vé -> từ chối (không xử lý audio không rõ nguồn); download_audio so md5
        # tải-về với vé và fail-closed khi lệch. Biến rò rỉ chéo tenant thành từ chối an toàn.
        expected_md5 = (config.get("audio_md5") or "").strip()
        if not expected_md5:
            raise RuntimeError("Thiếu md5 toàn vẹn audio — từ chối job (fail-closed).")

        import os

        # Đợt 15/LRC1 — dọn temp TRÊN MỌI ĐƯỜNG THOÁT (fail-closed cả khi lỗi): pipeline
        # có nhiều bước có thể NỔ giữa chừng (TTS synthesize, mix_audio fail-closed,
        # watermark...). Bản cũ đặt bước dọn ở CUỐI đường thành công nên khi một bước nổ,
        # cleanup KHÔNG chạy -> stem GIỌNG GỐC (vocals.wav), audio nguồn tải-về và clip TTS
        # rớt lại trong temp: vừa rò rỉ nội dung nhạy cảm (Zero-Logging #2) vừa phình đĩa
        # (caller ở main.py chỉ trả 500, KHÔNG dọn). Ta THEO DÕI mọi temp trung gian ngay
        # khi tạo ra, rồi dọn TẤT CẢ trong finally; chỉ giữ lại đầu ra cuối (dubbed_audio).
        cleanup_paths = set()
        demucs_job_dir = None
        final_audio_path = None
        try:
            # Đợt 17 F1 — dọn CƠ HỘI các file kết quả cuối (*_final.wav/*_wm.wav) đã quá hạn.
            # process_job GIỮ đầu ra cuối cho client tải nhưng KHÔNG có bước nào xóa nó sau đó
            # -> tích tụ VĨNH VIỄN (phình đĩa + audio lồng tiếng nhạy cảm nằm lại). Mỗi job mới
            # thu hồi đĩa từ các final cũ hơn TTL (client đã tải xong). Xem sweep_stale_finals;
            # bản thân sweep đã nuốt hết OSError nên không thể làm hỏng job — chỉ cần chắn lỗi
            # PHÂN TÍCH env (giá trị rác) để bước dọn rác không bao giờ chặn một job thật.
            try:
                _final_ttl_s = float(os.environ.get("WORKER_FINAL_TTL_S", "3600"))
            except (TypeError, ValueError):
                _final_ttl_s = 3600.0
            audio_engine.sweep_stale_finals(_final_ttl_s)

            # 1) Luôn tải audio nguồn: cần cho cả ASR (nếu thiếu segments) LẪN việc
            #    tách nhạc nền khi mix. Đây chính là chỗ bug cũ bỏ sót ở nhánh có segments.
            downloaded_path = audio_service.download_audio(audio_url, expected_md5)
            cleanup_paths.add(downloaded_path)
            local_audio_path = downloaded_path

            # Phòng thủ: nếu vô tình nhận container video, tách audio 16kHz mono.
            if local_audio_path.endswith((".mp4", ".mkv", ".mov")):
                local_audio_path = audio_engine.extract_audio_from_video(local_audio_path)
                cleanup_paths.add(local_audio_path)

            # 2) ASR chỉ khi client chưa gửi segments đã được người dùng duyệt.
            segments = config.get("segments", [])
            asr_ran = False
            if not segments:
                from src.asr_service import asr_service
                segments = asr_service.transcribe(local_audio_path)
                asr_ran = True  # để báo pipeline TRUNG THỰC: chỉ liệt kê Whisper khi thật sự chạy

            if not segments:
                raise RuntimeError("Không có segment nào để xử lý (audio rỗng hoặc ASR trả về trống).")

            # 3) Dịch bằng Qwen cục bộ THƯỜNG TRÚ VRAM (fail-closed nếu chưa nạp). Gọi ĐÚNG
            #    instance đã đăng ký ở load_all_models (self.models["translation"]) — không
            #    re-import singleton để tránh mọi khả năng nạp bản thứ hai / nạp lười lúc chạy.
            #    Truyền ngôn ngữ GỐC để đếm âm tiết đúng (căn lip-sync), không ghim cứng "en".
            translation_service = self.models["translation"]
            target_language = config.get("target_language", "Vietnamese")
            translated_segments = translation_service.translate_segments(
                segments=segments,
                target_language=target_language,
                style=config.get("style", "Formal"),
                source_language=config.get("source_language") or "en",
            )

            # 4) TTS: đọc ĐÚNG trường translated_text (bug cũ đọc "text" -> rỗng -> câm),
            #    chọn voice THEO TỪNG NGƯỜI NÓI (đa giọng thật) và điều biến ngữ điệu theo
            #    cảm xúc. voice_map = ánh xạ speaker_id -> voice do người dùng gán ở client.
            from src.tts_service import tts_service
            voice_map = config.get("voice_map") or {}

            tts_clips = []
            used_voices = set()
            tts_attempted = 0  # số câu CÓ thoại đã thử sinh giọng (để báo trung thực nếu rớt)
            missing_translation = 0  # câu CÓ thoại gốc nhưng bản dịch RỖNG -> không thể lồng tiếng
            for seg in translated_segments:
                spoken_text = seg.get("translated_text", "")
                if not spoken_text:
                    # I6/NFS: phân biệt "khoảng lặng thật" với "translator bỏ sót". Nếu câu có
                    # thoại GỐC (original_text) mà bản dịch rỗng thì đây là câu MẤT tiếng do
                    # thiếu bản dịch — ĐẾM để báo trung thực thay vì lặng lẽ bỏ. Bug cũ chỉ
                    # continue, và tts_attempted không tính các câu này nên note TTS bên dưới
                    # không bao giờ phản ánh chúng (mất tiếng âm thầm = fake success).
                    if (seg.get("original_text") or "").strip():
                        missing_translation += 1
                    continue
                tts_attempted += 1
                voice = self._resolve_voice(target_language, seg.get("speaker_id"), voice_map)
                prosody = EMOTION_PROSODY.get(seg.get("emotion", "NEUTRAL"), EMOTION_PROSODY["NEUTRAL"])
                audio_path = await tts_service.synthesize(
                    spoken_text,
                    voice=voice,
                    rate=prosody["rate"],
                    volume=prosody["volume"],
                    pitch=prosody["pitch"],
                )
                if audio_path:
                    used_voices.add(voice)
                    # Theo dõi để dọn ở finally kể cả khi bước sau nổ giữa chừng.
                    cleanup_paths.add(audio_path)
                    tts_clips.append({
                        "start": seg.get("start", 0),
                        "end": seg.get("end", 0),
                        "audio_path": audio_path,
                    })

            # 5) Mix KHÔNG điều kiện khi có TTS clips (bug cũ dùng 'local_audio_path' in locals()
            #    khiến nhánh có-sẵn-segments bị bỏ mix hoàn toàn -> mất tiếng).
            if not tts_clips:
                raise RuntimeError("TTS không tạo được clip nào — không thể render bản lồng tiếng.")

            instrumental_path, separated = audio_engine.extract_instrumental(local_audio_path)
            cleanup_paths.add(instrumental_path)   # stem no_vocals của Demucs (== local nếu fail-closed)
            if separated:
                # Demucs xuất <temp>/demucs_out/htdemucs/<basename>/{no_vocals,vocals}.wav.
                # instrumental_path là no_vocals.wav; stem vocals.wav (GIỌNG GỐC đã bóc tách —
                # NHẠY CẢM) VÀ cả thư mục job phải bị dọn ở finally, nếu không giọng gốc rò rỉ
                # trong temp. Ghim demucs_job_dir NGAY tại đây để finally rmtree được kể cả khi
                # mix_audio/watermark phía dưới nổ.
                demucs_job_dir = os.path.dirname(instrumental_path)
                cleanup_paths.add(os.path.join(demucs_job_dir, "vocals.wav"))  # stem giọng gốc
            mixed_audio_path = audio_engine.mix_audio(instrumental_path, tts_clips, ducking_db=-5.0)
            cleanup_paths.add(mixed_audio_path)  # bản mix trước watermark (== final nếu không watermark)
            final_audio_path, watermarked = audio_engine.add_watermark(mixed_audio_path)

            # Xây danh sách pipeline TRUNG THỰC theo bước THỰC SỰ chạy — không quảng cáo
            # bước chưa chạy (No-Fake-Success). Whisper CHỈ khi worker tự ASR; nếu client gửi
            # segments đã duyệt thì bước ASR bị bỏ, không được liệt kê "Whisper".
            # Mix luôn chạy (mix_audio đã fail-closed).
            pipeline = []
            if asr_ran:
                pipeline.append("Whisper")
            pipeline.extend(["Translation", "TTS"])
            if separated:
                pipeline.append("Demucs")
            pipeline.append("Mix")
            if watermarked:
                pipeline.append("Watermark")

            # Cảnh báo trung thực khi bước tăng cường không khả dụng.
            notes = []
            if not separated:
                notes.append(
                    "Chưa tách được nhạc nền (Demucs không khả dụng) — giọng gốc có thể "
                    "còn nằm dưới bản lồng tiếng."
                )
            if not watermarked:
                notes.append("Chưa gài được watermark (AudioSeal không khả dụng).")

            # Báo cáo TRUNG THỰC việc căn lip-sync (time-stretch) — chỉ khi thực sự có clip
            # được co/giãn về đúng độ dài đoạn hình.
            mix_stats = getattr(
                audio_engine,
                "last_mix_stats",
                {"clips": 0, "stretched": 0, "truncated": 0, "dropped_oor": 0},
            )
            if mix_stats.get("stretched"):
                notes.append(
                    f"Đã căn lip-sync (time-stretch) {mix_stats['stretched']}/{mix_stats['clips']} "
                    "clip về đúng độ dài đoạn hình."
                )
            # WPC-2/NFS-03: báo trung thực nếu phải cắt bớt clip quá dài (dịch dài hơn thời lượng gốc).
            if mix_stats.get("truncated"):
                notes.append(
                    f"Đã cắt bớt {mix_stats['truncated']}/{mix_stats['clips']} clip lồng tiếng quá dài "
                    "(bản dịch dài hơn thời lượng gốc) để không tràn sang đoạn kế."
                )
            # NFS-MIX-OOR: báo trung thực nếu có clip bị bỏ vì mốc bắt đầu nằm NGOÀI phạm vi
            # nhạc nền (pydub overlay lặng lẽ nuốt) — nếu không, người dùng mất tiếng lồng của
            # các đoạn đó mà pipeline vẫn báo "thành công" (No-Fake-Success).
            if mix_stats.get("dropped_oor"):
                notes.append(
                    f"Có {mix_stats['dropped_oor']}/{mix_stats['clips']} clip lồng tiếng bị bỏ vì "
                    "mốc bắt đầu nằm ngoài độ dài nhạc nền — bản lồng tiếng thiếu các đoạn này."
                )

            # WPC-1/NFS-02: nếu một số câu có thoại nhưng TTS không sinh được (voice server
            # lỗi/rớt), báo trung thực số câu thiếu tiếng thay vì lặng lẽ bỏ qua.
            tts_rendered = len(tts_clips)
            if tts_rendered < tts_attempted:
                notes.append(
                    f"Có {tts_attempted - tts_rendered}/{tts_attempted} câu không sinh được giọng "
                    "(TTS lỗi) — bản lồng tiếng thiếu các câu này."
                )

            # I6/NFS: câu CÓ thoại gốc nhưng bản dịch RỖNG (translator bỏ sót) — báo trung thực
            # số câu mất tiếng vì THIẾU BẢN DỊCH, tách bạch với lỗi TTS ở trên.
            if missing_translation:
                notes.append(
                    f"Có {missing_translation} câu có thoại gốc nhưng KHÔNG có bản dịch "
                    "(bị bỏ sót khi dịch) — bản lồng tiếng thiếu các câu này."
                )

            message = (
                "Đã tách nền, lồng tiếng và mix xong track âm thanh lồng tiếng."
                if separated
                else "Đã lồng tiếng và mix xong track âm thanh (chưa tách nhạc nền)."
            )

            return {
                "status": "success",
                "message": message,
                "device_used": self.device,
                "pipeline": pipeline,
                "separated": separated,
                "watermarked": watermarked,
                # Số giọng THỰC SỰ đã dùng — báo trung thực năng lực đa giọng thay vì
                # quảng cáo "multi-speaker" khi thực tế chỉ render một giọng.
                "distinct_voices": len(used_voices),
                "notes": notes,
                # Zero-Logging: KHÔNG trả translated_segments (chứa kịch bản GỐC + BẢN DỊCH
                # dạng plaintext). Client không dùng tới; nếu trả thì gateway sẽ ghi nó vào KV
                # 24h và lộ qua poll — rò rỉ nội dung nhạy cảm. Worker chỉ trả AUDIO đầu ra.
                "dubbed_audio": final_audio_path,
            }
        finally:
            # WPC-3 + Đợt 15/LRC1: dọn MỌI temp trung gian dù job THÀNH CÔNG hay NỔ giữa
            # chừng. Trên nhánh thành công, đầu ra cuối (final_audio_path) được loại khỏi tập
            # để client tải; trên nhánh lỗi final_audio_path=None nên KHÔNG loại gì -> dọn
            # sạch, không để stem giọng gốc (vocals.wav), audio nguồn hay clip TTS rớt lại
            # (rò rỉ Zero-Logging + phình đĩa). Best-effort: _cleanup_temp nuốt lỗi xóa.
            cleanup_paths.discard(final_audio_path)  # nếu không watermark thì final == mixed -> vẫn giữ đúng đầu ra
            cleanup_paths.discard(None)
            self._cleanup_temp(cleanup_paths)
            # Xóa nốt thư mục job htdemucs (CHỈ khi ĐÃ tách thật -> demucs_job_dir được gán).
            # Nếu fail-closed thì instrumental_path == local_audio_path, demucs_job_dir vẫn
            # None nên TUYỆT ĐỐI không rmtree gốc temp.
            if demucs_job_dir:
                import shutil
                shutil.rmtree(demucs_job_dir, ignore_errors=True)
