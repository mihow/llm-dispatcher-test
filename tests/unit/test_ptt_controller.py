"""Unit tests for PTTController."""

import numpy as np
import pytest
from unittest.mock import Mock, patch
from radio_assistant.ptt_controller import PTTController
from radio_assistant.audio_interface import AudioInterface


class TestPTTController:
    """Test suite for PTTController class."""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        ptt = PTTController()

        assert ptt.method == "vox"
        assert ptt.padding_ms == 300
        assert ptt.audio is None

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        mock_audio = Mock(spec=AudioInterface)
        ptt = PTTController(method="vox", vox_padding_ms=500, audio_interface=mock_audio)

        assert ptt.method == "vox"
        assert ptt.padding_ms == 500
        assert ptt.audio is mock_audio

    def test_init_unsupported_method(self) -> None:
        """Test that unsupported transmission method raises error."""
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            PTTController(method="gpio")

    def test_transmit_without_audio_interface(self) -> None:
        """Test that transmit without audio interface raises error."""
        ptt = PTTController()
        audio = np.random.randn(1000).astype(np.float32)

        with pytest.raises(ValueError, match="AudioInterface not configured"):
            ptt.transmit(audio)

    def test_transmit_with_vox_default_padding(self) -> None:
        """Test VOX transmission with default padding."""
        mock_audio = Mock(spec=AudioInterface)
        ptt = PTTController(vox_padding_ms=300, audio_interface=mock_audio)

        audio = np.random.randn(1000).astype(np.float32)
        ptt.transmit(audio, sample_rate=16000)

        # Verify audio interface play_audio was called
        mock_audio.play_audio.assert_called_once()

        # Get the padded audio that was passed
        padded_audio = mock_audio.play_audio.call_args[0][0]

        # Verify padding was added (300ms = 4800 samples at 16kHz)
        expected_padding_samples = int(300 * 16000 / 1000)
        assert len(padded_audio) == len(audio) + expected_padding_samples

        # Verify padding is silence
        assert np.allclose(padded_audio[:expected_padding_samples], 0.0)

        # Verify original audio follows padding
        assert np.allclose(padded_audio[expected_padding_samples:], audio)

    @pytest.mark.parametrize("padding_ms", [100, 200, 300, 500])
    def test_transmit_various_padding_durations(self, padding_ms: int) -> None:
        """Test VOX transmission with various padding durations."""
        mock_audio = Mock(spec=AudioInterface)
        ptt = PTTController(vox_padding_ms=padding_ms, audio_interface=mock_audio)

        audio = np.random.randn(1000).astype(np.float32)
        sample_rate = 16000

        ptt.transmit(audio, sample_rate=sample_rate)

        # Get padded audio
        padded_audio = mock_audio.play_audio.call_args[0][0]

        # Verify correct padding length
        expected_padding_samples = int(padding_ms * sample_rate / 1000)
        assert len(padded_audio) == len(audio) + expected_padding_samples

    def test_add_vox_padding_calculation(self) -> None:
        """Test VOX padding calculation."""
        ptt = PTTController(vox_padding_ms=300)

        audio = np.ones(1000, dtype=np.float32)
        sample_rate = 16000

        padded = ptt._add_vox_padding(audio, sample_rate)

        # 300ms at 16kHz = 4800 samples
        expected_padding = int(300 * 16000 / 1000)
        assert len(padded) == 1000 + expected_padding
        assert np.allclose(padded[:expected_padding], 0.0)
        assert np.allclose(padded[expected_padding:], 1.0)

    def test_add_vox_padding_preserves_dtype(self) -> None:
        """Test that VOX padding preserves audio dtype."""
        ptt = PTTController(vox_padding_ms=100)

        # Test with float32
        audio_f32 = np.random.randn(100).astype(np.float32)
        padded_f32 = ptt._add_vox_padding(audio_f32, 16000)
        assert padded_f32.dtype == np.float32

        # Test with float64
        audio_f64 = np.random.randn(100).astype(np.float64)
        padded_f64 = ptt._add_vox_padding(audio_f64, 16000)
        assert padded_f64.dtype == np.float64

    def test_add_vox_padding_zero_padding(self) -> None:
        """Test VOX padding with zero padding duration."""
        ptt = PTTController(vox_padding_ms=0)

        audio = np.random.randn(100).astype(np.float32)
        padded = ptt._add_vox_padding(audio, 16000)

        # With zero padding, output should equal input
        assert np.array_equal(padded, audio)

    def test_get_padding_duration(self) -> None:
        """Test getting VOX padding duration."""
        ptt = PTTController(vox_padding_ms=500)
        assert ptt.get_padding_duration_ms() == 500

    def test_set_padding_duration(self) -> None:
        """Test setting VOX padding duration."""
        ptt = PTTController(vox_padding_ms=300)

        ptt.set_padding_duration_ms(500)
        assert ptt.padding_ms == 500
        assert ptt.get_padding_duration_ms() == 500

    def test_set_padding_duration_negative(self) -> None:
        """Test that setting negative padding raises error."""
        ptt = PTTController(vox_padding_ms=300)

        with pytest.raises(ValueError, match="Padding must be non-negative"):
            ptt.set_padding_duration_ms(-100)

    def test_transmit_with_different_sample_rates(self) -> None:
        """Test transmit with various sample rates."""
        mock_audio = Mock(spec=AudioInterface)
        ptt = PTTController(vox_padding_ms=200, audio_interface=mock_audio)

        for sample_rate in [8000, 16000, 44100, 48000]:
            audio = np.random.randn(1000).astype(np.float32)
            ptt.transmit(audio, sample_rate=sample_rate)

            padded = mock_audio.play_audio.call_args[0][0]
            expected_padding = int(200 * sample_rate / 1000)
            assert len(padded) == len(audio) + expected_padding

    def test_transmit_empty_audio(self) -> None:
        """Test transmit with empty audio."""
        mock_audio = Mock(spec=AudioInterface)
        ptt = PTTController(vox_padding_ms=100, audio_interface=mock_audio)

        audio = np.array([], dtype=np.float32)
        ptt.transmit(audio, sample_rate=16000)

        padded = mock_audio.play_audio.call_args[0][0]
        # Should just be padding
        expected_padding = int(100 * 16000 / 1000)
        assert len(padded) == expected_padding

    def test_transmit_multichannel_audio(self) -> None:
        """Test transmit with multichannel audio."""
        mock_audio = Mock(spec=AudioInterface)
        ptt = PTTController(vox_padding_ms=200, audio_interface=mock_audio)

        # Stereo audio
        audio = np.random.randn(1000, 2).astype(np.float32)
        ptt.transmit(audio, sample_rate=16000)

        padded = mock_audio.play_audio.call_args[0][0]
        # Padding should be added to first dimension
        expected_padding = int(200 * 16000 / 1000)
        assert padded.shape[0] == audio.shape[0] + expected_padding
        assert padded.shape[1] == 2

    def test_transmit_preserves_audio_content(self) -> None:
        """Test that transmission preserves original audio content."""
        mock_audio = Mock(spec=AudioInterface)
        ptt = PTTController(vox_padding_ms=100, audio_interface=mock_audio)

        # Create distinctive audio pattern
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        ptt.transmit(audio, sample_rate=16000)

        padded = mock_audio.play_audio.call_args[0][0]
        padding_samples = int(100 * 16000 / 1000)

        # Verify original audio is preserved after padding
        assert np.allclose(padded[padding_samples:], audio)

    def test_multiple_transmissions_independent(self) -> None:
        """Test that multiple transmissions are independent."""
        mock_audio = Mock(spec=AudioInterface)
        ptt = PTTController(vox_padding_ms=200, audio_interface=mock_audio)

        audio1 = np.random.randn(500).astype(np.float32)
        audio2 = np.random.randn(1000).astype(np.float32)

        ptt.transmit(audio1, sample_rate=16000)
        ptt.transmit(audio2, sample_rate=16000)

        # Verify two separate calls
        assert mock_audio.play_audio.call_count == 2

        # Verify each has correct padding
        call1_audio = mock_audio.play_audio.call_args_list[0][0][0]
        call2_audio = mock_audio.play_audio.call_args_list[1][0][0]

        padding_samples = int(200 * 16000 / 1000)
        assert len(call1_audio) == len(audio1) + padding_samples
        assert len(call2_audio) == len(audio2) + padding_samples

    def test_dynamic_padding_change(self) -> None:
        """Test changing padding duration between transmissions."""
        mock_audio = Mock(spec=AudioInterface)
        ptt = PTTController(vox_padding_ms=100, audio_interface=mock_audio)

        audio = np.random.randn(1000).astype(np.float32)

        # First transmission with 100ms padding
        ptt.transmit(audio, sample_rate=16000)
        padded1 = mock_audio.play_audio.call_args[0][0]

        # Change padding
        ptt.set_padding_duration_ms(300)

        # Second transmission with 300ms padding
        ptt.transmit(audio, sample_rate=16000)
        padded2 = mock_audio.play_audio.call_args[0][0]

        # Verify different padding lengths
        assert len(padded1) < len(padded2)
        assert len(padded1) == len(audio) + int(100 * 16000 / 1000)
        assert len(padded2) == len(audio) + int(300 * 16000 / 1000)

    @patch("radio_assistant.ptt_controller.logger")
    def test_logging_on_transmit(self, mock_logger: Mock) -> None:
        """Test that transmit logs appropriate messages."""
        mock_audio = Mock(spec=AudioInterface)
        ptt = PTTController(vox_padding_ms=200, audio_interface=mock_audio)

        audio = np.random.randn(1000).astype(np.float32)
        ptt.transmit(audio, sample_rate=16000)

        # Verify logging occurred
        assert mock_logger.info.called

    @patch("radio_assistant.ptt_controller.logger")
    def test_logging_on_padding_change(self, mock_logger: Mock) -> None:
        """Test that padding change logs message."""
        ptt = PTTController(vox_padding_ms=300)
        ptt.set_padding_duration_ms(500)

        # Verify logging occurred
        assert mock_logger.info.called
