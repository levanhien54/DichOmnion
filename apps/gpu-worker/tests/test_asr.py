import pytest
from unittest.mock import patch, MagicMock
from src.asr_service import ASRService
import tempfile
import os

@pytest.fixture
def asr_service():
    service = ASRService()
    return service

def test_asr_service_refuses_when_not_loaded(asr_service):
    """No-Fake-Success: chưa nạp model thì transcribe phải TỪ CHỐI (raise),
    tuyệt đối không trả về dữ liệu bịa (mock) như thể đã bóc băng thành công."""
    with pytest.raises(RuntimeError):
        asr_service.transcribe("dummy.wav")
    
@patch('faster_whisper.WhisperModel')
def test_asr_service_real_transcribe(mock_whisper_class, asr_service):
    """Kiểm tra hàm transcribe với model đã được load"""
    # Mock model
    mock_model = MagicMock()
    
    # Tạo object giả lập Segment
    class FakeSegment:
        def __init__(self, start, end, text):
            self.start = start
            self.end = end
            self.text = text
            
    # Giả lập kết quả trả về của model.transcribe
    fake_segments = [
        FakeSegment(0.0, 2.0, "Hello world"),
        FakeSegment(2.0, 5.5, "This is a test")
    ]
    mock_model.transcribe.return_value = (fake_segments, {"language": "en"})
    
    # Thiết lập cho service
    mock_whisper_class.return_value = mock_model
    asr_service.load_model(model_size="tiny")
    
    assert asr_service.is_loaded == True
    assert asr_service.model == mock_model
    
    # Gọi hàm transcribe với file rác (vì model đã bị mock)
    fd, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, 'w') as f:
        f.write("fake audio data")
        
    try:
        result = asr_service.transcribe(path)
        
        # Verify kết quả
        assert len(result) == 2
        assert result[0]["id"] == "sub-1"
        assert result[0]["text"] == "Hello world"
        assert result[0]["duration"] == 2.0
        
        assert result[1]["id"] == "sub-2"
        assert result[1]["text"] == "This is a test"
        assert result[1]["duration"] == 3.5
    finally:
        os.remove(path)
