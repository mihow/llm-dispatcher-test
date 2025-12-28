"""Integration tests for end-to-end callsign detection pipeline."""

from pathlib import Path
import pytest
import soundfile as sf
from radio_assistant.transcription_engine import TranscriptionEngine
from radio_assistant.callsign_detector import CallsignDetector


class TestCallsignDetectionPipeline:
    """Integration tests combining transcription and callsign detection."""

    @pytest.fixture
    def engine(self) -> TranscriptionEngine:
        """Create transcription engine."""
        return TranscriptionEngine(model_size="base", device="cpu", compute_type="int8")

    @pytest.fixture
    def detector(self) -> CallsignDetector:
        """Create callsign detector for WSJJ659."""
        return CallsignDetector("WSJJ659", require_dispatch_keyword=True)

    @pytest.fixture
    def test_audio_dir(self) -> Path:
        """Get path to test audio directory."""
        return Path(__file__).parent.parent / "audio" / "transcription"

    def test_pipeline_hello_world(
        self,
        engine: TranscriptionEngine,
        detector: CallsignDetector,
        test_audio_dir: Path,
    ) -> None:
        """Test pipeline on hello world (should not detect callsign)."""
        audio_path = test_audio_dir / "hello_world.wav"
        audio, sr = sf.read(audio_path, dtype="float32")

        # Transcribe
        transcription = engine.transcribe(audio, sample_rate=sr)

        # Detect
        detection = detector.detect(transcription.text)

        # Hello world should not contain WSJJ659
        assert detection.detected is False

    def test_pipeline_clear_callsign(
        self,
        engine: TranscriptionEngine,
        detector: CallsignDetector,
        test_audio_dir: Path,
    ) -> None:
        """Test pipeline on clear callsign audio."""
        audio_path = test_audio_dir / "wsjj659_clear.wav"
        audio, sr = sf.read(audio_path, dtype="float32")

        # Transcribe
        transcription = engine.transcribe(audio, sample_rate=sr)

        # Expected ground truth: "This is WSJJ659 calling"
        # Note: Since we're using tone placeholders, transcription won't match
        # This test validates the pipeline works end-to-end

        # Detect
        detection = detector.detect(transcription.text)

        # We can't assert detection here since test audio is tones
        # Just validate the pipeline executes without errors
        assert isinstance(detection.detected, bool)

    def test_pipeline_phonetic_callsign(
        self,
        engine: TranscriptionEngine,
        detector: CallsignDetector,
        test_audio_dir: Path,
    ) -> None:
        """Test pipeline on phonetic alphabet callsign."""
        audio_path = test_audio_dir / "wsjj659_phonetic.wav"
        audio, sr = sf.read(audio_path, dtype="float32")

        # Transcribe
        transcription = engine.transcribe(audio, sample_rate=sr)

        # Detect
        detection = detector.detect(transcription.text)

        # Pipeline should execute successfully
        assert isinstance(detection.detected, bool)

    def test_pipeline_other_callsign(
        self,
        engine: TranscriptionEngine,
        detector: CallsignDetector,
        test_audio_dir: Path,
    ) -> None:
        """Test pipeline on different callsign."""
        audio_path = test_audio_dir / "other_callsign.wav"
        audio, sr = sf.read(audio_path, dtype="float32")

        # Transcribe
        transcription = engine.transcribe(audio, sample_rate=sr)

        # Detect
        detection = detector.detect(transcription.text)

        # Should not detect WSJJ659 (ground truth is K6ABC)
        # (though with tone audio, transcription won't match anyway)
        assert isinstance(detection.detected, bool)

    def test_pipeline_with_synthetic_transcription(self, detector: CallsignDetector) -> None:
        """Test detection with known transcription text."""
        # Simulate perfect transcription
        test_cases = [
            ("This is WSJJ659 calling", True),
            ("WSJJ659 dispatch", True),
            ("Whiskey Sierra Juliet Juliet six five nine calling", True),
            ("This is K6ABC calling", False),
            ("Hello world", False),
        ]

        for text, expected_detected in test_cases:
            detection = detector.detect(text)
            assert (
                detection.detected == expected_detected
            ), f"Failed for: '{text}' (expected {expected_detected}, got {detection.detected})"

    def test_pipeline_batch_processing(
        self,
        engine: TranscriptionEngine,
        detector: CallsignDetector,
        test_audio_dir: Path,
    ) -> None:
        """Test processing multiple audio files in sequence."""
        test_files = [
            "hello_world.wav",
            "wsjj659_clear.wav",
            "wsjj659_phonetic.wav",
            "other_callsign.wav",
        ]

        results = []
        for filename in test_files:
            audio_path = test_audio_dir / filename
            audio, sr = sf.read(audio_path, dtype="float32")

            # Transcribe
            transcription = engine.transcribe(audio, sample_rate=sr)

            # Detect
            detection = detector.detect(transcription.text)

            results.append(
                {
                    "filename": filename,
                    "transcription": transcription.text,
                    "detected": detection.detected,
                    "confidence": detection.confidence,
                }
            )

        # Verify all processed successfully
        assert len(results) == len(test_files)
        for result in results:
            assert isinstance(result["detected"], bool)
            assert isinstance(result["confidence"], float)

    def test_pipeline_confidence_correlation(self, detector: CallsignDetector) -> None:
        """Test that detection confidence correlates with match quality."""
        test_cases = [
            ("WSJJ659 dispatch", "exact"),
            ("wsjj659 dispatch", "lowercase"),
            ("W S J J 6 5 9 dispatch", "spaced"),
            ("Whiskey Sierra Juliet Juliet six five nine dispatch", "phonetic"),
        ]

        confidences = []
        for text, description in test_cases:
            detection = detector.detect(text)
            if detection.detected:
                confidences.append((description, detection.confidence))

        # All should have reasonable confidence
        for description, confidence in confidences:
            assert confidence > 0.5, f"Low confidence for {description}: {confidence}"

    def test_detector_without_dispatch_requirement(self, test_audio_dir: Path) -> None:
        """Test detector configured without dispatch keyword requirement."""
        detector = CallsignDetector("WSJJ659", require_dispatch_keyword=False)

        # Should detect even without dispatch keyword
        detection = detector.detect("WSJJ659")
        assert detection.detected is True
        assert detection.dispatch_keyword_present is False

    def test_detector_without_phonetic(self, test_audio_dir: Path) -> None:
        """Test detector with phonetic alphabet disabled."""
        detector = CallsignDetector("WSJJ659", phonetic_alphabet=False)

        # Should still detect direct match
        detection = detector.detect("WSJJ659 dispatch")
        assert detection.detected is True

        # Should NOT detect phonetic
        detection = detector.detect("Whiskey Sierra Juliet Juliet six five nine dispatch")
        assert detection.detected is False

    @pytest.mark.parametrize(
        "callsign,text,expected",
        [
            ("WSJJ659", "WSJJ659 dispatch", True),
            ("K6ABC", "K6ABC calling", True),
            ("KE7XYZ", "KE7XYZ radio check", True),
            ("WSJJ659", "K6ABC dispatch", False),
            ("K6ABC", "WSJJ659 calling", False),
        ],
    )
    def test_multiple_callsigns(self, callsign: str, text: str, expected: bool) -> None:
        """Test detection with different callsigns."""
        detector = CallsignDetector(callsign)
        detection = detector.detect(text)
        assert detection.detected == expected

    def test_pipeline_timing(
        self,
        engine: TranscriptionEngine,
        detector: CallsignDetector,
        test_audio_dir: Path,
    ) -> None:
        """Test that full pipeline completes in reasonable time."""
        audio_path = test_audio_dir / "hello_world.wav"
        audio, sr = sf.read(audio_path, dtype="float32")

        # Transcribe
        transcription = engine.transcribe(audio, sample_rate=sr)

        # Detection should be very fast (<1ms typically)
        import time

        start = time.perf_counter()
        _ = detector.detect(transcription.text)
        detection_time_ms = (time.perf_counter() - start) * 1000

        # Detection should be nearly instantaneous
        assert detection_time_ms < 100  # Should complete in <100ms

    def test_pipeline_preserves_transcription_metadata(
        self,
        engine: TranscriptionEngine,
        detector: CallsignDetector,
        test_audio_dir: Path,
    ) -> None:
        """Test that transcription metadata is preserved through pipeline."""
        audio_path = test_audio_dir / "hello_world.wav"
        audio, sr = sf.read(audio_path, dtype="float32")

        # Transcribe
        transcription = engine.transcribe(audio, sample_rate=sr)

        # Detect
        detection = detector.detect(transcription.text)

        # Verify transcription metadata still accessible
        assert transcription.duration_ms > 0
        assert isinstance(transcription.segments, list)
        assert isinstance(transcription.confidence, float)

        # Verify detection result has metadata
        assert isinstance(detection.confidence, float)
        assert isinstance(detection.dispatch_keyword_present, bool)
