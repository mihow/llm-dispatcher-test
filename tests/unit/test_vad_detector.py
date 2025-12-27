"""Unit tests for VADDetector."""

import numpy as np
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from radio_assistant.vad_detector import VADDetector


class TestVADDetector:
    """Test suite for VADDetector class."""

    @patch("radio_assistant.vad_detector.silero_vad")
    def test_init_default_params(self, mock_silero: Mock) -> None:
        """Test initialization with default parameters."""
        mock_model = Mock()
        mock_silero.load_silero_vad.return_value = mock_model

        vad: VADDetector = VADDetector()

        assert vad.threshold == 0.5
        assert vad.min_speech_duration_ms == 250
        assert vad.min_silence_duration_ms == 100
        assert vad.sample_rate == 16000

    @patch("radio_assistant.vad_detector.silero_vad")
    def test_init_custom_params(self, mock_silero: Mock) -> None:
        """Test initialization with custom parameters."""
        mock_model = Mock()
        mock_silero.load_silero_vad.return_value = mock_model

        vad: VADDetector = VADDetector(
            threshold=0.7,
            min_speech_duration_ms=500,
            min_silence_duration_ms=200,
            sample_rate=8000,
        )

        assert vad.threshold == 0.7
        assert vad.min_speech_duration_ms == 500
        assert vad.min_silence_duration_ms == 200
        assert vad.sample_rate == 8000

    @patch("radio_assistant.vad_detector.silero_vad")
    def test_init_invalid_threshold(self, mock_silero: Mock) -> None:
        """Test initialization with invalid threshold raises ValueError."""
        mock_model = Mock()
        # mock_utils removed
        mock_silero.load_silero_vad.return_value = mock_model

        with pytest.raises(ValueError, match="Invalid threshold"):
            VADDetector(threshold=-0.1)

        with pytest.raises(ValueError, match="Invalid threshold"):
            VADDetector(threshold=1.5)

    @patch("radio_assistant.vad_detector.silero_vad")
    def test_init_invalid_durations(self, mock_silero: Mock) -> None:
        """Test initialization with invalid durations raises ValueError."""
        mock_model = Mock()
        # mock_utils removed
        mock_silero.load_silero_vad.return_value = mock_model

        with pytest.raises(ValueError, match="Invalid min_speech_duration_ms"):
            VADDetector(min_speech_duration_ms=-1)

        with pytest.raises(ValueError, match="Invalid min_silence_duration_ms"):
            VADDetector(min_silence_duration_ms=0)

    @patch("radio_assistant.vad_detector.silero_vad")
    def test_init_invalid_sample_rate(self, mock_silero: Mock) -> None:
        """Test initialization with invalid sample rate raises ValueError."""
        mock_model = Mock()
        # mock_utils removed
        mock_silero.load_silero_vad.return_value = mock_model

        with pytest.raises(ValueError, match="Invalid sample_rate"):
            VADDetector(sample_rate=44100)

    @patch("radio_assistant.vad_detector.silero_vad")
    def test_init_model_loading_fails(self, mock_silero: Mock) -> None:
        """Test initialization handles model loading failure."""
        mock_silero.load_silero_vad.side_effect = Exception("Model load failed")

        with pytest.raises(RuntimeError, match="Failed to load VAD model"):
            VADDetector()

    @patch("radio_assistant.vad_detector.silero_vad")
    def test_is_speech_detects_speech(self, mock_silero: Mock) -> None:
        """Test is_speech detects speech above threshold."""
        mock_model = Mock()
        mock_model.return_value = torch.tensor(0.8)  # High probability
        # mock_utils removed
        mock_silero.load_silero_vad.return_value = mock_model

        vad: VADDetector = VADDetector(threshold=0.5)

        audio = np.random.randn(8000).astype(np.float32) * 0.5
        result: bool = vad.is_speech(audio)

        assert result is True
        mock_model.assert_called_once()

    @patch("radio_assistant.vad_detector.silero_vad")
    def test_is_speech_no_speech_detected(self, mock_silero: Mock) -> None:
        """Test is_speech returns False below threshold."""
        mock_model = Mock()
        mock_model.return_value = torch.tensor(0.2)  # Low probability
        # mock_utils removed
        mock_silero.load_silero_vad.return_value = mock_model

        vad: VADDetector = VADDetector(threshold=0.5)

        audio = np.random.randn(8000).astype(np.float32) * 0.5
        result: bool = vad.is_speech(audio)

        assert result is False

    @patch("radio_assistant.vad_detector.silero_vad")
    def test_is_speech_empty_audio(self, mock_silero: Mock) -> None:
        """Test is_speech handles empty audio."""
        mock_model = Mock()
        # mock_utils removed
        mock_silero.load_silero_vad.return_value = mock_model

        vad: VADDetector = VADDetector()

        audio = np.array([], dtype=np.float32)
        result: bool = vad.is_speech(audio)

        assert result is False
        mock_model.assert_not_called()

    @patch("radio_assistant.vad_detector.silero_vad")
    def test_is_speech_invalid_input(self, mock_silero: Mock) -> None:
        """Test is_speech raises ValueError for invalid input."""
        mock_model = Mock()
        # mock_utils removed
        mock_silero.load_silero_vad.return_value = mock_model

        vad: VADDetector = VADDetector()

        with pytest.raises(ValueError, match="must be a numpy array"):
            vad.is_speech([1, 2, 3])  # type: ignore

    @patch("radio_assistant.vad_detector.silero_vad")
    def test_is_speech_handles_exception(self, mock_silero: Mock) -> None:
        """Test is_speech handles model exceptions gracefully."""
        mock_model = Mock()
        mock_model.side_effect = Exception("Model error")
        # mock_utils removed
        mock_silero.load_silero_vad.return_value = mock_model

        vad: VADDetector = VADDetector()

        audio = np.random.randn(8000).astype(np.float32)
        result: bool = vad.is_speech(audio)

        assert result is False

    @patch("radio_assistant.vad_detector.silero_vad")
    def test_get_speech_timestamps(self, mock_silero: Mock) -> None:
        """Test get_speech_timestamps returns correct format."""
        mock_model = Mock()
        # Simulate timestamps in samples
        mock_silero.get_speech_timestamps.return_value = [
            {"start": 8000, "end": 24000},  # 0.5s - 1.5s
            {"start": 32000, "end": 48000},  # 2.0s - 3.0s
        ]
        mock_silero.load_silero_vad.return_value = mock_model

        vad: VADDetector = VADDetector()

        audio = np.random.randn(64000).astype(np.float32) * 0.5  # 4s of audio
        timestamps = vad.get_speech_timestamps(audio)

        assert len(timestamps) == 2
        assert timestamps[0]["start"] == 0.5
        assert timestamps[0]["end"] == 1.5
        assert timestamps[1]["start"] == 2.0
        assert timestamps[1]["end"] == 3.0

    @patch("radio_assistant.vad_detector.silero_vad")
    def test_get_speech_timestamps_no_speech(self, mock_silero: Mock) -> None:
        """Test get_speech_timestamps with no speech."""
        mock_model = Mock()
        mock_silero.get_speech_timestamps.return_value = []
        mock_silero.load_silero_vad.return_value = mock_model

        vad: VADDetector = VADDetector()

        audio = np.random.randn(16000).astype(np.float32) * 0.01  # Very quiet
        timestamps = vad.get_speech_timestamps(audio)

        assert timestamps == []

    @patch("radio_assistant.vad_detector.silero_vad")
    def test_get_speech_timestamps_invalid_input(self, mock_silero: Mock) -> None:
        """Test get_speech_timestamps with invalid input."""
        mock_model = Mock()
        # mock_utils removed
        mock_silero.load_silero_vad.return_value = mock_model

        vad: VADDetector = VADDetector()

        with pytest.raises(ValueError, match="must be a numpy array"):
            vad.get_speech_timestamps([1, 2, 3])  # type: ignore

    @patch("radio_assistant.vad_detector.silero_vad")
    def test_get_speech_timestamps_handles_exception(self, mock_silero: Mock) -> None:
        """Test get_speech_timestamps handles exceptions gracefully."""
        mock_model = Mock()
        mock_silero.get_speech_timestamps.side_effect = Exception("Timestamp error")
        mock_silero.load_silero_vad.return_value = mock_model

        vad: VADDetector = VADDetector()

        audio = np.random.randn(16000).astype(np.float32)
        timestamps = vad.get_speech_timestamps(audio)

        assert timestamps == []

    @patch("radio_assistant.vad_detector.silero_vad")
    def test_prepare_audio_mono(self, mock_silero: Mock) -> None:
        """Test _prepare_audio with mono audio."""
        mock_model = Mock()
        # mock_utils removed
        mock_silero.load_silero_vad.return_value = mock_model

        vad: VADDetector = VADDetector()

        audio = np.random.randn(1000).astype(np.float32) * 0.5
        tensor = vad._prepare_audio(audio)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1000,)
        assert tensor.dtype == torch.float32

    @patch("radio_assistant.vad_detector.silero_vad")
    def test_prepare_audio_stereo_raises_error(self, mock_silero: Mock) -> None:
        """Test _prepare_audio raises error for stereo audio."""
        mock_model = Mock()
        # mock_utils removed
        mock_silero.load_silero_vad.return_value = mock_model

        vad: VADDetector = VADDetector()

        audio = np.random.randn(1000, 2).astype(np.float32)

        with pytest.raises(ValueError, match="Audio must be mono"):
            vad._prepare_audio(audio)

    @patch("radio_assistant.vad_detector.silero_vad")
    def test_prepare_audio_normalizes_high_values(self, mock_silero: Mock) -> None:
        """Test _prepare_audio normalizes values > 1.0."""
        mock_model = Mock()
        # mock_utils removed
        mock_silero.load_silero_vad.return_value = mock_model

        vad: VADDetector = VADDetector()

        # Audio with values > 1.0
        audio = np.random.randn(1000).astype(np.float32) * 5.0
        tensor = vad._prepare_audio(audio)

        assert tensor.abs().max() <= 1.0

    @patch("radio_assistant.vad_detector.silero_vad")
    def test_prepare_audio_converts_dtype(self, mock_silero: Mock) -> None:
        """Test _prepare_audio converts non-float32 arrays."""
        mock_model = Mock()
        # mock_utils removed
        mock_silero.load_silero_vad.return_value = mock_model

        vad: VADDetector = VADDetector()

        audio = np.random.randint(-32768, 32767, 1000, dtype=np.int16)
        tensor = vad._prepare_audio(audio)

        assert tensor.dtype == torch.float32

    @patch("radio_assistant.vad_detector.silero_vad")
    def test_reset(self, mock_silero: Mock) -> None:
        """Test reset method."""
        mock_model = Mock()
        # mock_utils removed
        mock_silero.load_silero_vad.return_value = mock_model

        vad: VADDetector = VADDetector()
        vad.reset()  # Should not raise any exceptions
