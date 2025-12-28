"""Integration tests for PTT controller."""

from pathlib import Path
import pytest
import soundfile as sf
import numpy as np
from radio_assistant.ptt_controller import PTTController
from radio_assistant.audio_interface import AudioInterface


class TestPTTIntegration:
    """Integration tests for PTTController with real audio."""

    @pytest.fixture
    def audio_interface(self) -> AudioInterface:
        """Create audio interface for testing."""
        return AudioInterface(sample_rate=16000, channels=1)

    @pytest.fixture
    def ptt_controller(self, audio_interface: AudioInterface) -> PTTController:
        """Create PTT controller with audio interface."""
        return PTTController(vox_padding_ms=300, audio_interface=audio_interface)

    @pytest.fixture
    def response_audio_dir(self) -> Path:
        """Get path to response audio directory."""
        return Path(__file__).parent.parent / "audio" / "responses"

    def test_ptt_with_real_audio_interface(
        self, ptt_controller: PTTController, response_audio_dir: Path
    ) -> None:
        """Test PTT with real audio interface (mocked playback)."""
        from unittest.mock import patch
        import sounddevice as sd

        audio_path = response_audio_dir / "signal_received.wav"
        audio, sr = sf.read(audio_path, dtype="float32")

        with patch.object(sd, "play"), patch.object(sd, "wait"):
            # Should not raise exceptions
            ptt_controller.transmit(audio, sample_rate=sr)

    def test_ptt_loads_response_audio(self, response_audio_dir: Path) -> None:
        """Test loading response audio files."""
        response_files = ["signal_received.wav", "ready_to_copy.wav"]

        for filename in response_files:
            audio_path = response_audio_dir / filename
            assert audio_path.exists(), f"Missing response file: {filename}"

            audio, sr = sf.read(audio_path, dtype="float32")
            assert len(audio) > 0
            assert sr == 16000

    @pytest.mark.parametrize("padding_ms", [100, 200, 300, 500])
    def test_ptt_various_padding_with_response_audio(
        self, response_audio_dir: Path, padding_ms: int
    ) -> None:
        """Test PTT with various padding durations on response audio."""
        from unittest.mock import Mock

        mock_audio = Mock(spec=AudioInterface)
        ptt = PTTController(vox_padding_ms=padding_ms, audio_interface=mock_audio)

        audio_path = response_audio_dir / "signal_received.wav"
        audio, sr = sf.read(audio_path, dtype="float32")

        ptt.transmit(audio, sample_rate=sr)

        # Verify transmission occurred
        mock_audio.play_audio.assert_called_once()

        # Verify padding was added
        transmitted_audio = mock_audio.play_audio.call_args[0][0]
        expected_padding = int(padding_ms * sr / 1000)
        assert len(transmitted_audio) == len(audio) + expected_padding

    def test_ptt_end_to_end_with_detection(
        self, ptt_controller: PTTController, response_audio_dir: Path
    ) -> None:
        """Test end-to-end: detection triggers response transmission."""
        from unittest.mock import patch
        import sounddevice as sd
        from radio_assistant.callsign_detector import CallsignDetector

        # Simulate callsign detection
        detector = CallsignDetector("WSJJ659")
        detection = detector.detect("WSJJ659 dispatch")

        assert detection.detected is True

        # If detected, transmit response
        if detection.detected:
            audio_path = response_audio_dir / "signal_received.wav"
            audio, sr = sf.read(audio_path, dtype="float32")

            with patch.object(sd, "play"), patch.object(sd, "wait"):
                ptt_controller.transmit(audio, sample_rate=sr)

    def test_generate_vox_test_files(self, response_audio_dir: Path) -> None:
        """Test generating VOX test files with different padding."""
        import tempfile

        audio_path = response_audio_dir / "signal_received.wav"
        audio, sr = sf.read(audio_path, dtype="float32")

        padding_durations = [100, 200, 300, 500]

        with tempfile.TemporaryDirectory() as tmpdir:
            for padding_ms in padding_durations:
                ptt = PTTController(vox_padding_ms=padding_ms)
                padded = ptt._add_vox_padding(audio, sr)

                output_path = Path(tmpdir) / f"vox_{padding_ms}ms.wav"
                sf.write(output_path, padded, sr)

                # Verify file was created
                assert output_path.exists()

                # Verify file contents
                loaded, loaded_sr = sf.read(output_path, dtype="float32")
                assert loaded_sr == sr
                assert len(loaded) == len(audio) + int(padding_ms * sr / 1000)

    def test_ptt_response_timing(self, response_audio_dir: Path) -> None:
        """Test that PTT response timing is reasonable."""
        from unittest.mock import Mock
        import time

        mock_audio = Mock(spec=AudioInterface)
        ptt = PTTController(vox_padding_ms=300, audio_interface=mock_audio)

        audio_path = response_audio_dir / "signal_received.wav"
        audio, sr = sf.read(audio_path, dtype="float32")

        # Measure transmission time
        start = time.perf_counter()
        ptt.transmit(audio, sample_rate=sr)
        duration_ms = (time.perf_counter() - start) * 1000

        # Should complete quickly (just processing, not actual playback)
        assert duration_ms < 100  # Should be <100ms

    def test_ptt_multichannel_response_audio(self, response_audio_dir: Path) -> None:
        """Test PTT with multichannel audio if available."""
        from unittest.mock import Mock

        # Create stereo version of response
        audio_path = response_audio_dir / "signal_received.wav"
        mono_audio, sr = sf.read(audio_path, dtype="float32")

        # Convert to stereo
        stereo_audio = np.stack([mono_audio, mono_audio], axis=1)

        mock_audio = Mock(spec=AudioInterface)
        ptt = PTTController(vox_padding_ms=200, audio_interface=mock_audio)

        ptt.transmit(stereo_audio, sample_rate=sr)

        transmitted = mock_audio.play_audio.call_args[0][0]
        assert transmitted.ndim == 2
        assert transmitted.shape[1] == 2

    def test_ptt_preserves_audio_quality(self, response_audio_dir: Path) -> None:
        """Test that PTT preserves audio quality."""
        from unittest.mock import Mock

        audio_path = response_audio_dir / "signal_received.wav"
        audio, sr = sf.read(audio_path, dtype="float32")

        mock_audio = Mock(spec=AudioInterface)
        ptt = PTTController(vox_padding_ms=300, audio_interface=mock_audio)

        ptt.transmit(audio, sample_rate=sr)

        transmitted = mock_audio.play_audio.call_args[0][0]
        padding_samples = int(300 * sr / 1000)

        # Verify original audio is bit-identical after padding
        assert np.array_equal(transmitted[padding_samples:], audio)

    def test_ptt_batch_transmissions(self, response_audio_dir: Path) -> None:
        """Test multiple transmissions in sequence."""
        from unittest.mock import Mock

        mock_audio = Mock(spec=AudioInterface)
        ptt = PTTController(vox_padding_ms=200, audio_interface=mock_audio)

        # Load multiple response files
        response_files = ["signal_received.wav", "ready_to_copy.wav"]

        for filename in response_files:
            audio_path = response_audio_dir / filename
            audio, sr = sf.read(audio_path, dtype="float32")
            ptt.transmit(audio, sample_rate=sr)

        # Verify multiple transmissions
        assert mock_audio.play_audio.call_count == len(response_files)
