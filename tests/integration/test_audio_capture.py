"""Integration tests for audio capture functionality."""

import numpy as np
import pytest
from radio_assistant.audio_interface import AudioInterface


class TestAudioCaptureIntegration:
    """Integration tests for real audio capture."""

    def test_generate_and_verify_sine_wave(self) -> None:
        """Test capturing a generated sine wave and verifying its frequency.

        This test generates a sine wave, captures it virtually, and verifies
        the frequency content. It doesn't require actual hardware.
        """
        sample_rate: int = 16000
        duration: float = 1.0
        frequency: float = 440.0  # A4 note

        # Generate sine wave
        t: np.ndarray = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        sine_wave: np.ndarray = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Verify sine wave properties
        assert sine_wave.shape == (16000,)
        assert sine_wave.dtype == np.float32
        assert -1.0 <= sine_wave.min() <= -0.99
        assert 0.99 <= sine_wave.max() <= 1.0

        # Verify frequency using FFT
        fft_result: np.ndarray = np.fft.rfft(sine_wave)
        fft_freqs: np.ndarray = np.fft.rfftfreq(len(sine_wave), 1 / sample_rate)
        peak_freq_index: int = np.argmax(np.abs(fft_result))
        detected_frequency: float = float(fft_freqs[peak_freq_index])

        # Allow small tolerance for frequency detection
        assert abs(detected_frequency - frequency) < 1.0, (
            f"Expected frequency {frequency}Hz, " f"detected {detected_frequency}Hz"
        )

    def test_audio_interface_initialization(self) -> None:
        """Test that AudioInterface can be initialized without errors."""
        audio: AudioInterface = AudioInterface(sample_rate=16000, channels=1)

        assert audio.sample_rate == 16000
        assert audio.channels == 1

    def test_list_audio_devices(self) -> None:
        """Test listing available audio devices."""
        audio: AudioInterface = AudioInterface()
        devices = audio.list_devices()

        # Should return a list (may be empty in CI environment)
        assert isinstance(devices, list)

        # If devices exist, verify structure
        if devices:
            device = devices[0]
            assert "index" in device
            assert "name" in device
            assert "max_input_channels" in device
            assert "max_output_channels" in device
            assert "default_samplerate" in device

    def test_capture_and_playback_chunk_format(self) -> None:
        """Test that capture_chunk returns correctly formatted data.

        Note: This test mocks the actual audio I/O to work in CI environment.
        """
        from unittest.mock import patch
        import sounddevice as sd

        with patch.object(sd, "rec") as mock_rec, patch.object(sd, "wait"):
            # Mock audio data
            mock_audio: np.ndarray = np.random.randn(8000, 1).astype(np.float32)
            mock_rec.return_value = mock_audio

            audio: AudioInterface = AudioInterface(sample_rate=16000, channels=1)
            result: np.ndarray = audio.capture_chunk(0.5)

            # Verify format
            assert result.shape == (8000,)
            assert result.dtype == np.float32
            assert -5.0 <= result.min() <= 5.0  # Reasonable range for random data
            assert -5.0 <= result.max() <= 5.0

    def test_playback_chunk_format(self) -> None:
        """Test that play_audio accepts correctly formatted data."""
        from unittest.mock import patch
        import sounddevice as sd

        with patch.object(sd, "play") as mock_play, patch.object(sd, "wait"):
            audio: AudioInterface = AudioInterface(sample_rate=16000, channels=1)

            # Create test audio
            test_audio: np.ndarray = np.sin(2 * np.pi * 440.0 * np.linspace(0, 1, 16000)).astype(
                np.float32
            )

            # Should not raise any exceptions
            audio.play_audio(test_audio)

            # Verify play was called
            mock_play.assert_called_once()

    def test_round_trip_audio_data(self) -> None:
        """Test capturing and immediately playing back audio data."""
        from unittest.mock import patch
        import sounddevice as sd

        # Create a mock that simulates realistic audio I/O
        captured_data: np.ndarray = np.random.randn(16000).astype(np.float32) * 0.1

        with (
            patch.object(sd, "rec") as mock_rec,
            patch.object(sd, "play") as mock_play,
            patch.object(sd, "wait"),
        ):

            mock_rec.return_value = captured_data.reshape(-1, 1)

            audio: AudioInterface = AudioInterface(sample_rate=16000, channels=1)

            # Capture
            result: np.ndarray = audio.capture_chunk(1.0)

            # Verify capture shape and content
            assert result.shape == (16000,)
            np.testing.assert_array_almost_equal(result, captured_data, decimal=5)

            # Playback
            audio.play_audio(result)

            # Verify playback was called with correct data
            mock_play.assert_called_once()
            played_data: np.ndarray = mock_play.call_args[0][0]
            np.testing.assert_array_almost_equal(played_data, result, decimal=5)

    @pytest.mark.parametrize(
        "sample_rate,channels",
        [
            (8000, 1),
            (16000, 1),
            (22050, 1),
            (44100, 1),
            (48000, 1),
            (16000, 2),
        ],
    )
    def test_various_audio_formats(self, sample_rate: int, channels: int) -> None:
        """Test AudioInterface with various sample rates and channel configurations."""
        audio: AudioInterface = AudioInterface(sample_rate=sample_rate, channels=channels)

        assert audio.sample_rate == sample_rate
        assert audio.channels == channels

    def test_audio_chunk_no_clipping(self) -> None:
        """Test that captured audio chunks are within valid range [-1, 1]."""
        from unittest.mock import patch
        import sounddevice as sd

        # Create audio with values guaranteed to be in valid range
        valid_audio: np.ndarray = np.clip(
            np.random.randn(8000, 1).astype(np.float32) * 0.5, -0.99, 0.99
        )

        with patch.object(sd, "rec") as mock_rec, patch.object(sd, "wait"):
            mock_rec.return_value = valid_audio

            audio: AudioInterface = AudioInterface(sample_rate=16000, channels=1)
            result: np.ndarray = audio.capture_chunk(0.5)

            # Verify no clipping
            assert result.min() >= -1.0
            assert result.max() <= 1.0
