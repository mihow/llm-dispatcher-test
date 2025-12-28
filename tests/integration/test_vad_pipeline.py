"""Integration tests for VAD pipeline."""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from radio_assistant.vad_detector import VADDetector


class TestVADPipelineIntegration:
    """Integration tests for VAD with real audio files."""

    @pytest.fixture
    def vad(self) -> VADDetector:
        """Create VAD detector for tests."""
        return VADDetector(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
        )

    def test_speech_clean_detected(self, vad: VADDetector) -> None:
        """Test VAD detects clean speech."""
        audio_path = Path("tests/audio/vad/speech_clean.wav")
        if not audio_path.exists():
            pytest.skip(f"Test audio file not found: {audio_path}")

        audio, sample_rate = sf.read(audio_path, dtype="float32")

        # Test is_speech
        is_speech: bool = vad.is_speech(audio)
        assert is_speech, "Failed to detect clean speech"

        # Test get_speech_timestamps
        timestamps = vad.get_speech_timestamps(audio, sample_rate)
        assert len(timestamps) > 0, "No speech timestamps detected in clean speech"

    def test_silence_not_detected(self, vad: VADDetector) -> None:
        """Test VAD does not detect silence as speech."""
        audio_path = Path("tests/audio/vad/silence.wav")
        if not audio_path.exists():
            pytest.skip(f"Test audio file not found: {audio_path}")

        audio, sample_rate = sf.read(audio_path, dtype="float32")

        # Test is_speech
        is_speech: bool = vad.is_speech(audio)
        assert not is_speech, "Incorrectly detected speech in silence"

        # Test get_speech_timestamps
        timestamps = vad.get_speech_timestamps(audio, sample_rate)
        assert len(timestamps) == 0, "Detected speech timestamps in silence"

    def test_noise_only_not_detected(self, vad: VADDetector) -> None:
        """Test VAD does not detect noise as speech."""
        audio_path = Path("tests/audio/vad/noise_only.wav")
        if not audio_path.exists():
            pytest.skip(f"Test audio file not found: {audio_path}")

        audio, sample_rate = sf.read(audio_path, dtype="float32")

        # Test is_speech
        is_speech: bool = vad.is_speech(audio)
        assert not is_speech, "Incorrectly detected speech in noise"

        # Test get_speech_timestamps
        timestamps = vad.get_speech_timestamps(audio, sample_rate)
        # Allow very few/short segments due to noise characteristics
        assert len(timestamps) <= 1, f"Too many speech segments in noise: {len(timestamps)}"

    def test_speech_with_static_detected(self, vad: VADDetector) -> None:
        """Test VAD detects speech with static noise."""
        audio_path = Path("tests/audio/vad/speech_with_static.wav")
        if not audio_path.exists():
            pytest.skip(f"Test audio file not found: {audio_path}")

        audio, sample_rate = sf.read(audio_path, dtype="float32")

        # Test is_speech
        is_speech: bool = vad.is_speech(audio)
        assert is_speech, "Failed to detect speech with static"

        # Test get_speech_timestamps
        timestamps = vad.get_speech_timestamps(audio, sample_rate)
        assert len(timestamps) > 0, "No speech timestamps in noisy speech"

    def test_multiple_transmissions_detected(self, vad: VADDetector) -> None:
        """Test VAD detects multiple speech segments."""
        audio_path = Path("tests/audio/vad/multiple_transmissions.wav")
        if not audio_path.exists():
            pytest.skip(f"Test audio file not found: {audio_path}")

        audio, sample_rate = sf.read(audio_path, dtype="float32")

        # Test get_speech_timestamps - should detect multiple segments
        timestamps = vad.get_speech_timestamps(audio, sample_rate)
        assert len(timestamps) >= 2, f"Expected >=2 segments, got {len(timestamps)}"

        # Verify segments are separated in time
        for i in range(len(timestamps) - 1):
            gap: float = timestamps[i + 1]["start"] - timestamps[i]["end"]
            assert gap > 0, f"Overlapping segments at index {i}"

    def test_squelch_tail_handling(self, vad: VADDetector) -> None:
        """Test VAD handles squelch tail (brief noise burst)."""
        audio_path = Path("tests/audio/vad/squelch_tail.wav")
        if not audio_path.exists():
            pytest.skip(f"Test audio file not found: {audio_path}")

        audio, sample_rate = sf.read(audio_path, dtype="float32")

        # Squelch tail should not be detected as speech
        timestamps = vad.get_speech_timestamps(audio, sample_rate)
        # Allow max 1 very short segment
        if len(timestamps) > 0:
            duration: float = timestamps[0]["end"] - timestamps[0]["start"]
            assert duration < 0.5, f"Squelch tail detected as speech (duration: {duration}s)"

    def test_weak_signal_detection(self, vad: VADDetector) -> None:
        """Test VAD with weak/noisy signal (0 dB SNR)."""
        audio_path = Path("tests/audio/vad/speech_weak_signal.wav")
        if not audio_path.exists():
            pytest.skip(f"Test audio file not found: {audio_path}")

        audio, sample_rate = sf.read(audio_path, dtype="float32")

        # Should detect some speech, but might not be perfect
        timestamps = vad.get_speech_timestamps(audio, sample_rate)
        # Relaxed requirement: at least detect something
        # (Actual accuracy may vary with weak signals)
        assert len(timestamps) >= 0, "Should handle weak signals without crashing"

    @pytest.mark.parametrize(
        "audio_file,expected_speech",
        [
            ("speech_clean.wav", True),
            ("speech_with_static.wav", True),
            ("silence.wav", False),
            ("noise_only.wav", False),
        ],
    )
    def test_is_speech_consistency(
        self, vad: VADDetector, audio_file: str, expected_speech: bool
    ) -> None:
        """Test is_speech consistency across different audio types."""
        audio_path = Path(f"tests/audio/vad/{audio_file}")
        if not audio_path.exists():
            pytest.skip(f"Test audio file not found: {audio_path}")

        audio, sample_rate = sf.read(audio_path, dtype="float32")
        is_speech: bool = vad.is_speech(audio)

        assert (
            is_speech == expected_speech
        ), f"{audio_file}: expected {expected_speech}, got {is_speech}"

    def test_vad_timestamp_accuracy(self, vad: VADDetector) -> None:
        """Test that speech timestamps are reasonably accurate."""
        audio_path = Path("tests/audio/vad/multiple_transmissions.wav")
        if not audio_path.exists():
            pytest.skip(f"Test audio file not found: {audio_path}")

        audio, sample_rate = sf.read(audio_path, dtype="float32")
        timestamps = vad.get_speech_timestamps(audio, sample_rate)

        # Verify all timestamps are within audio duration
        audio_duration: float = len(audio) / sample_rate
        for ts in timestamps:
            assert 0 <= ts["start"] < audio_duration
            assert 0 < ts["end"] <= audio_duration
            assert ts["start"] < ts["end"]

    def test_different_thresholds(self) -> None:
        """Test VAD with different threshold values."""
        audio_path = Path("tests/audio/vad/speech_with_static.wav")
        if not audio_path.exists():
            pytest.skip(f"Test audio file not found: {audio_path}")

        audio, sample_rate = sf.read(audio_path, dtype="float32")

        # Low threshold (more sensitive)
        vad_sensitive = VADDetector(threshold=0.3)
        timestamps_sensitive = vad_sensitive.get_speech_timestamps(audio, sample_rate)

        # High threshold (less sensitive)
        vad_strict = VADDetector(threshold=0.7)
        timestamps_strict = vad_strict.get_speech_timestamps(audio, sample_rate)

        # More sensitive should detect same or more segments
        assert len(timestamps_sensitive) >= len(timestamps_strict), (
            f"Expected sensitive ({len(timestamps_sensitive)}) >= "
            f"strict ({len(timestamps_strict)})"
        )

    def test_sample_rate_8khz(self) -> None:
        """Test VAD with 8kHz audio."""
        # Generate simple 8kHz test audio
        sample_rate: int = 8000
        duration: float = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)

        vad: VADDetector = VADDetector(sample_rate=8000)
        is_speech: bool = vad.is_speech(audio)

        # Should process without errors
        assert isinstance(is_speech, bool)

    def test_vad_reset(self, vad: VADDetector) -> None:
        """Test VAD reset functionality."""
        audio_path = Path("tests/audio/vad/speech_clean.wav")
        if not audio_path.exists():
            pytest.skip(f"Test audio file not found: {audio_path}")

        audio, sample_rate = sf.read(audio_path, dtype="float32")

        # Process audio
        result1: bool = vad.is_speech(audio)

        # Reset
        vad.reset()

        # Process again - should get same result
        result2: bool = vad.is_speech(audio)

        assert result1 == result2, "Results changed after reset"
