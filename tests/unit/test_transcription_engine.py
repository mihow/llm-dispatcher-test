"""Unit tests for TranscriptionEngine."""

import numpy as np
import pytest
from unittest.mock import Mock, patch
from radio_assistant.transcription_engine import (
    TranscriptionEngine,
    TranscriptionResult,
)


class TestTranscriptionEngine:
    """Test suite for TranscriptionEngine class."""

    @patch("radio_assistant.transcription_engine.WhisperModel")
    def test_init_default_params(self, mock_whisper_model: Mock) -> None:
        """Test initialization with default parameters."""
        engine = TranscriptionEngine()

        assert engine.model_size == "base"
        assert engine.device == "cpu"
        assert engine.compute_type == "int8"
        mock_whisper_model.assert_called_once_with("base", device="cpu", compute_type="int8")

    @patch("radio_assistant.transcription_engine.WhisperModel")
    def test_init_custom_params(self, mock_whisper_model: Mock) -> None:
        """Test initialization with custom parameters."""
        engine = TranscriptionEngine(model_size="small", device="cuda", compute_type="float16")

        assert engine.model_size == "small"
        assert engine.device == "cuda"
        assert engine.compute_type == "float16"
        mock_whisper_model.assert_called_once_with("small", device="cuda", compute_type="float16")

    @patch("radio_assistant.transcription_engine.WhisperModel")
    def test_transcribe_basic(self, mock_whisper_model: Mock) -> None:
        """Test basic transcription."""
        # Setup mock
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model

        mock_segment = Mock()
        mock_segment.start = 0.0
        mock_segment.end = 1.5
        mock_segment.text = "Hello world"
        mock_segment.avg_logprob = -0.2

        mock_info = Mock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        # Create engine and transcribe
        engine = TranscriptionEngine()
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = engine.transcribe(audio, sample_rate=16000)

        # Verify result
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.confidence == -0.2
        assert result.duration_ms >= 0  # May be 0 with mocked model
        assert len(result.segments) == 1
        assert result.segments[0]["text"] == "Hello world"
        assert result.segments[0]["start"] == 0.0
        assert result.segments[0]["end"] == 1.5

    @patch("radio_assistant.transcription_engine.WhisperModel")
    def test_transcribe_multiple_segments(self, mock_whisper_model: Mock) -> None:
        """Test transcription with multiple segments."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model

        # Create multiple segments
        segments = []
        for i, text in enumerate(["This is", "WSJJ659", "calling"]):
            seg = Mock()
            seg.start = i * 0.5
            seg.end = (i + 1) * 0.5
            seg.text = text
            seg.avg_logprob = -0.1 * (i + 1)
            segments.append(seg)

        mock_info = Mock()
        mock_model.transcribe.return_value = (segments, mock_info)

        engine = TranscriptionEngine()
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = engine.transcribe(audio)

        assert result.text == "This is WSJJ659 calling"
        assert len(result.segments) == 3
        assert abs(result.confidence - (-0.2)) < 0.01  # Average of -0.1, -0.2, -0.3

    @patch("radio_assistant.transcription_engine.WhisperModel")
    def test_transcribe_empty_audio(self, mock_whisper_model: Mock) -> None:
        """Test transcription of silence/empty audio."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        mock_info = Mock()
        mock_model.transcribe.return_value = ([], mock_info)

        engine = TranscriptionEngine()
        audio = np.zeros(16000, dtype=np.float32)
        result = engine.transcribe(audio)

        assert result.text == ""
        assert result.confidence == 0.0
        assert len(result.segments) == 0

    @patch("radio_assistant.transcription_engine.WhisperModel")
    def test_transcribe_normalizes_audio(self, mock_whisper_model: Mock) -> None:
        """Test that audio is normalized to [-1, 1] range."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model

        mock_segment = Mock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "test"
        mock_segment.avg_logprob = -0.1

        mock_info = Mock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        engine = TranscriptionEngine()

        # Audio that exceeds [-1, 1] range
        audio = np.array([2.0, -3.0, 1.5], dtype=np.float32)
        engine.transcribe(audio)

        # Verify transcribe was called with normalized audio
        called_audio = mock_model.transcribe.call_args[0][0]
        assert np.max(np.abs(called_audio)) <= 1.0

    @patch("radio_assistant.transcription_engine.WhisperModel")
    def test_transcribe_converts_dtype(self, mock_whisper_model: Mock) -> None:
        """Test that audio dtype is converted to float32."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model

        mock_segment = Mock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "test"
        mock_segment.avg_logprob = -0.1

        mock_info = Mock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        engine = TranscriptionEngine()

        # Test with int16 audio
        audio = np.array([100, -200, 150], dtype=np.int16)
        engine.transcribe(audio)

        # Verify transcribe was called with float32
        called_audio = mock_model.transcribe.call_args[0][0]
        assert called_audio.dtype == np.float32

    @patch("radio_assistant.transcription_engine.WhisperModel")
    def test_transcribe_invalid_sample_rate(self, mock_whisper_model: Mock) -> None:
        """Test that invalid sample rate raises error."""
        engine = TranscriptionEngine()
        audio = np.random.randn(8000).astype(np.float32)

        with pytest.raises(ValueError, match="Sample rate must be 16000"):
            engine.transcribe(audio, sample_rate=8000)

    @patch("radio_assistant.transcription_engine.WhisperModel")
    def test_transcribe_timing_recorded(self, mock_whisper_model: Mock) -> None:
        """Test that transcription timing is recorded."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model

        mock_segment = Mock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "test"
        mock_segment.avg_logprob = -0.1

        mock_info = Mock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        engine = TranscriptionEngine()
        audio = np.random.randn(16000).astype(np.float32)
        result = engine.transcribe(audio)

        assert result.duration_ms >= 0  # May be 0 with mocked model
        assert isinstance(result.duration_ms, int)

    @patch("radio_assistant.transcription_engine.WhisperModel")
    def test_transcribe_with_callsign(self, mock_whisper_model: Mock) -> None:
        """Test transcription containing amateur radio callsign."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model

        mock_segment = Mock()
        mock_segment.start = 0.0
        mock_segment.end = 2.0
        mock_segment.text = "This is WSJJ659 calling"
        mock_segment.avg_logprob = -0.15

        mock_info = Mock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        engine = TranscriptionEngine()
        audio = np.random.randn(16000).astype(np.float32)
        result = engine.transcribe(audio)

        assert "WSJJ659" in result.text
        assert result.confidence < 0  # Log prob is negative

    @patch("radio_assistant.transcription_engine.WhisperModel")
    def test_segment_metadata_preserved(self, mock_whisper_model: Mock) -> None:
        """Test that segment metadata is properly preserved."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model

        mock_segment = Mock()
        mock_segment.start = 1.5
        mock_segment.end = 3.2
        mock_segment.text = "test segment"
        mock_segment.avg_logprob = -0.25

        mock_info = Mock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        engine = TranscriptionEngine()
        audio = np.random.randn(16000).astype(np.float32)
        result = engine.transcribe(audio)

        assert len(result.segments) == 1
        seg = result.segments[0]
        assert seg["start"] == 1.5
        assert seg["end"] == 3.2
        assert seg["text"] == "test segment"
        assert seg["confidence"] == -0.25

    @patch("radio_assistant.transcription_engine.WhisperModel")
    def test_model_called_with_correct_params(self, mock_whisper_model: Mock) -> None:
        """Test that Whisper model is called with correct parameters."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model

        mock_info = Mock()
        mock_model.transcribe.return_value = ([], mock_info)

        engine = TranscriptionEngine()
        audio = np.random.randn(16000).astype(np.float32)
        engine.transcribe(audio)

        # Verify model.transcribe was called with correct params
        mock_model.transcribe.assert_called_once()
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["beam_size"] == 5
        assert call_kwargs["language"] == "en"
        assert call_kwargs["task"] == "transcribe"


class TestTranscriptionResult:
    """Test suite for TranscriptionResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a TranscriptionResult."""
        result = TranscriptionResult(
            text="test text",
            confidence=0.95,
            duration_ms=1500,
            segments=[{"start": 0.0, "end": 1.0, "text": "test"}],
        )

        assert result.text == "test text"
        assert result.confidence == 0.95
        assert result.duration_ms == 1500
        assert len(result.segments) == 1

    def test_result_immutable_fields(self) -> None:
        """Test that TranscriptionResult fields can be accessed."""
        result = TranscriptionResult(text="test", confidence=0.9, duration_ms=1000, segments=[])

        assert hasattr(result, "text")
        assert hasattr(result, "confidence")
        assert hasattr(result, "duration_ms")
        assert hasattr(result, "segments")
