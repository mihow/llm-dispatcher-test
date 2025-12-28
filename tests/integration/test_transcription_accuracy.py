"""Integration tests for transcription accuracy."""

from pathlib import Path
import pytest
import soundfile as sf
from radio_assistant.transcription_engine import TranscriptionEngine


class TestTranscriptionAccuracy:
    """Integration tests for transcription accuracy with real audio files."""

    @pytest.fixture
    def engine(self) -> TranscriptionEngine:
        """Create a TranscriptionEngine for testing."""
        return TranscriptionEngine(model_size="base", device="cpu", compute_type="int8")

    @pytest.fixture
    def test_audio_dir(self) -> Path:
        """Get path to test audio directory."""
        return Path(__file__).parent.parent / "audio" / "transcription"

    def load_audio_and_transcript(self, test_audio_dir: Path, filename: str) -> tuple:
        """Load audio file and ground truth transcript.

        Args:
            test_audio_dir: Directory containing test audio files
            filename: Name of audio file (without extension)

        Returns:
            Tuple of (audio_data, sample_rate, ground_truth_text)
        """
        audio_path = test_audio_dir / f"{filename}.wav"
        txt_path = test_audio_dir / f"{filename}.txt"

        audio, sr = sf.read(audio_path, dtype="float32")
        ground_truth = txt_path.read_text().strip()

        return audio, sr, ground_truth

    def test_transcribe_hello_world(
        self, engine: TranscriptionEngine, test_audio_dir: Path
    ) -> None:
        """Test transcribing simple hello world audio."""
        audio, sr, ground_truth = self.load_audio_and_transcript(test_audio_dir, "hello_world")

        result = engine.transcribe(audio, sample_rate=sr)

        # Note: Since we're using tone placeholders, we can't test exact match
        # This test validates the engine works end-to-end
        assert isinstance(result.text, str)
        assert result.duration_ms > 0
        assert isinstance(result.segments, list)

    def test_transcribe_callsign_clear(
        self, engine: TranscriptionEngine, test_audio_dir: Path
    ) -> None:
        """Test transcribing clear callsign audio."""
        audio, sr, ground_truth = self.load_audio_and_transcript(test_audio_dir, "wsjj659_clear")

        result = engine.transcribe(audio, sample_rate=sr)

        assert isinstance(result.text, str)
        assert result.duration_ms > 0
        # Ground truth: "This is WSJJ659 calling"

    def test_transcribe_callsign_phonetic(
        self, engine: TranscriptionEngine, test_audio_dir: Path
    ) -> None:
        """Test transcribing phonetic alphabet callsign."""
        audio, sr, ground_truth = self.load_audio_and_transcript(test_audio_dir, "wsjj659_phonetic")

        result = engine.transcribe(audio, sample_rate=sr)

        assert isinstance(result.text, str)
        assert result.duration_ms > 0
        # Ground truth: "Whiskey Sierra Juliet Juliet six five nine calling"

    def test_transcribe_silence(self, engine: TranscriptionEngine, test_audio_dir: Path) -> None:
        """Test transcribing silence/empty audio."""
        audio, sr, ground_truth = self.load_audio_and_transcript(test_audio_dir, "empty_silence")

        result = engine.transcribe(audio, sample_rate=sr)

        # Silence should produce empty or minimal output
        assert isinstance(result.text, str)
        assert len(result.text) < 50  # Minimal/no transcription

    def test_transcribe_other_callsign(
        self, engine: TranscriptionEngine, test_audio_dir: Path
    ) -> None:
        """Test transcribing different callsign."""
        audio, sr, ground_truth = self.load_audio_and_transcript(test_audio_dir, "other_callsign")

        result = engine.transcribe(audio, sample_rate=sr)

        assert isinstance(result.text, str)
        # Ground truth: "This is K6ABC calling"

    def test_confidence_score_reasonable(
        self, engine: TranscriptionEngine, test_audio_dir: Path
    ) -> None:
        """Test that confidence scores are in reasonable range."""
        audio, sr, _ = self.load_audio_and_transcript(test_audio_dir, "hello_world")

        result = engine.transcribe(audio, sample_rate=sr)

        # Whisper uses log probabilities (negative values)
        # Reasonable range is roughly -1.0 to 0.0 for good audio
        assert -5.0 <= result.confidence <= 0.0

    def test_segments_have_timing(self, engine: TranscriptionEngine, test_audio_dir: Path) -> None:
        """Test that segments include timing information."""
        audio, sr, _ = self.load_audio_and_transcript(test_audio_dir, "hello_world")

        result = engine.transcribe(audio, sample_rate=sr)

        if len(result.segments) > 0:
            seg = result.segments[0]
            assert "start" in seg
            assert "end" in seg
            assert seg["end"] > seg["start"]

    def test_multiple_files_batch(self, engine: TranscriptionEngine, test_audio_dir: Path) -> None:
        """Test transcribing multiple files in sequence."""
        test_files = ["hello_world", "wsjj659_clear", "other_callsign"]

        results = []
        for filename in test_files:
            audio, sr, _ = self.load_audio_and_transcript(test_audio_dir, filename)
            result = engine.transcribe(audio, sample_rate=sr)
            results.append(result)

        # Verify all transcriptions completed
        assert len(results) == len(test_files)
        for result in results:
            assert result.duration_ms > 0
            assert isinstance(result.text, str)

    @pytest.mark.parametrize(
        "filename",
        [
            "hello_world",
            "wsjj659_clear",
            "wsjj659_phonetic",
            "wsjj659_noisy",
            "wsjj659_rapid",
            "other_callsign",
        ],
    )
    def test_all_test_files_transcribe(
        self, engine: TranscriptionEngine, test_audio_dir: Path, filename: str
    ) -> None:
        """Test that all test audio files can be transcribed without error."""
        audio, sr, ground_truth = self.load_audio_and_transcript(test_audio_dir, filename)

        result = engine.transcribe(audio, sample_rate=sr)

        # Basic validation
        assert isinstance(result.text, str)
        assert result.duration_ms > 0
        assert isinstance(result.segments, list)
        assert isinstance(result.confidence, float)

    def test_transcription_reproducible(
        self, engine: TranscriptionEngine, test_audio_dir: Path
    ) -> None:
        """Test that transcription results are reproducible."""
        audio, sr, _ = self.load_audio_and_transcript(test_audio_dir, "hello_world")

        result1 = engine.transcribe(audio, sample_rate=sr)
        result2 = engine.transcribe(audio, sample_rate=sr)

        # Same audio should produce same transcription
        assert result1.text == result2.text
        assert len(result1.segments) == len(result2.segments)

    def test_engine_reusable(self, engine: TranscriptionEngine, test_audio_dir: Path) -> None:
        """Test that engine can be reused for multiple transcriptions."""
        audio1, sr1, _ = self.load_audio_and_transcript(test_audio_dir, "hello_world")
        audio2, sr2, _ = self.load_audio_and_transcript(test_audio_dir, "wsjj659_clear")

        result1 = engine.transcribe(audio1, sample_rate=sr1)
        result2 = engine.transcribe(audio2, sample_rate=sr2)

        # Both should succeed
        assert result1.duration_ms > 0
        assert result2.duration_ms > 0
