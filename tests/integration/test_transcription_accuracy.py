"""Integration tests for transcription accuracy."""

import re
from pathlib import Path
import pytest
import soundfile as sf
from jiwer import wer, cer
from radio_assistant.transcription_engine import TranscriptionEngine
from radio_assistant.callsign_detector import CallsignDetector


def normalize_for_comparison(text: str) -> str:
    """Normalize text for WER/CER comparison.

    Args:
        text: Input text

    Returns:
        Normalized text (lowercase, no punctuation, normalized whitespace)
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text


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

        # Normalize for comparison
        predicted = normalize_for_comparison(result.text)
        expected = normalize_for_comparison(ground_truth)

        # Calculate accuracy metrics
        word_error_rate = wer(expected, predicted)
        char_error_rate = cer(expected, predicted)

        # Assert accuracy threshold (TTS audio should be very accurate)
        assert word_error_rate < 0.20, (
            f"WER too high: {word_error_rate:.2%}\n"
            f"Expected: '{expected}'\n"
            f"Got:      '{predicted}'"
        )

        # Keep structure checks
        assert result.duration_ms > 0
        assert isinstance(result.segments, list)

        # Log for visibility
        print(f"✓ WER={word_error_rate:.2%}, CER={char_error_rate:.2%}")

    def test_transcribe_callsign_clear(
        self, engine: TranscriptionEngine, test_audio_dir: Path
    ) -> None:
        """Test transcribing clear callsign audio."""
        audio, sr, ground_truth = self.load_audio_and_transcript(test_audio_dir, "wsjj659_clear")

        result = engine.transcribe(audio, sample_rate=sr)

        # Normalize for comparison
        predicted = normalize_for_comparison(result.text)
        expected = normalize_for_comparison(ground_truth)

        # Calculate accuracy metrics
        word_error_rate = wer(expected, predicted)
        char_error_rate = cer(expected, predicted)

        # Assert accuracy threshold (clear audio should be very accurate)
        assert word_error_rate < 0.15, (
            f"WER too high: {word_error_rate:.2%}\n"
            f"Expected: '{expected}'\n"
            f"Got:      '{predicted}'"
        )

        # Callsign detection validation
        detector = CallsignDetector(callsign="WSJJ659", require_dispatch_keyword=False)
        detection = detector.detect(result.text)

        assert detection.detected, f"Failed to detect WSJJ659 in: '{result.text}'"
        assert (
            detection.confidence > 0.8
        ), f"Detection confidence too low: {detection.confidence:.2%}"

        # Keep structure checks
        assert result.duration_ms > 0

        # Log for visibility
        print(
            f"✓ WER={word_error_rate:.2%}, CER={char_error_rate:.2%}, "
            f"Callsign: {detection.confidence:.0%}"
        )

    def test_transcribe_callsign_phonetic(
        self, engine: TranscriptionEngine, test_audio_dir: Path
    ) -> None:
        """Test transcribing phonetic alphabet callsign."""
        audio, sr, ground_truth = self.load_audio_and_transcript(test_audio_dir, "wsjj659_phonetic")

        result = engine.transcribe(audio, sample_rate=sr)

        # Normalize for comparison
        predicted = normalize_for_comparison(result.text)
        expected = normalize_for_comparison(ground_truth)

        # Calculate accuracy metrics
        word_error_rate = wer(expected, predicted)
        char_error_rate = cer(expected, predicted)

        # Higher threshold for phonetic alphabet (harder to transcribe)
        assert word_error_rate < 0.25, (
            f"WER too high: {word_error_rate:.2%}\n"
            f"Expected: '{expected}'\n"
            f"Got:      '{predicted}'"
        )

        # Keep structure checks
        assert result.duration_ms > 0

        # Log for visibility
        print(f"✓ WER={word_error_rate:.2%}, CER={char_error_rate:.2%}")

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

        # Normalize for comparison
        predicted = normalize_for_comparison(result.text)
        expected = normalize_for_comparison(ground_truth)

        # Calculate accuracy metrics
        word_error_rate = wer(expected, predicted)
        char_error_rate = cer(expected, predicted)

        # Assert accuracy threshold
        assert word_error_rate < 0.20, (
            f"WER too high: {word_error_rate:.2%}\n"
            f"Expected: '{expected}'\n"
            f"Got:      '{predicted}'"
        )

        # Callsign detection validation
        detector = CallsignDetector(callsign="K6ABC", require_dispatch_keyword=False)
        detection = detector.detect(result.text)

        assert detection.detected, f"Failed to detect K6ABC in: '{result.text}'"
        assert (
            detection.confidence > 0.8
        ), f"Detection confidence too low: {detection.confidence:.2%}"

        # Log for visibility
        print(
            f"✓ WER={word_error_rate:.2%}, CER={char_error_rate:.2%}, "
            f"Callsign: {detection.confidence:.0%}"
        )

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
        "filename,wer_threshold",
        [
            ("hello_world", 0.20),
            ("wsjj659_clear", 0.15),
            ("wsjj659_phonetic", 0.25),
            ("wsjj659_noisy", 0.30),
            ("wsjj659_rapid", 0.30),
            ("other_callsign", 0.20),
        ],
    )
    def test_all_test_files_transcribe(
        self, engine: TranscriptionEngine, test_audio_dir: Path, filename: str, wer_threshold: float
    ) -> None:
        """Test that all test audio files can be transcribed with acceptable accuracy."""
        audio, sr, ground_truth = self.load_audio_and_transcript(test_audio_dir, filename)

        result = engine.transcribe(audio, sample_rate=sr)

        # Basic validation
        assert isinstance(result.text, str)
        assert result.duration_ms > 0
        assert isinstance(result.segments, list)
        assert isinstance(result.confidence, float)

        # Accuracy validation
        predicted = normalize_for_comparison(result.text)
        expected = normalize_for_comparison(ground_truth)

        word_error_rate = wer(expected, predicted)
        char_error_rate = cer(expected, predicted)

        assert word_error_rate < wer_threshold, (
            f"{filename}: WER {word_error_rate:.2%} exceeds threshold {wer_threshold:.2%}\n"
            f"Expected: '{expected}'\n"
            f"Got:      '{predicted}'"
        )

        print(f"✓ {filename}: WER={word_error_rate:.2%}, CER={char_error_rate:.2%}")

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
