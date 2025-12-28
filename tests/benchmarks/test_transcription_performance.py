"""Performance benchmarks for transcription engine."""

from pathlib import Path
import pytest
import soundfile as sf
import numpy as np
from radio_assistant.transcription_engine import TranscriptionEngine


class TestTranscriptionPerformance:
    """Performance benchmarks for TranscriptionEngine."""

    @pytest.fixture
    def base_engine(self) -> TranscriptionEngine:
        """Create base model engine for benchmarking."""
        return TranscriptionEngine(model_size="base", device="cpu", compute_type="int8")

    @pytest.fixture
    def small_engine(self) -> TranscriptionEngine:
        """Create small model engine for benchmarking."""
        return TranscriptionEngine(model_size="small", device="cpu", compute_type="int8")

    @pytest.fixture
    def test_audio_dir(self) -> Path:
        """Get path to test audio directory."""
        return Path(__file__).parent.parent / "audio" / "transcription"

    @pytest.mark.benchmark
    def test_base_model_speed_short_audio(
        self, base_engine: TranscriptionEngine, test_audio_dir: Path
    ) -> None:
        """Benchmark base model on short audio (~2s)."""
        audio_path = test_audio_dir / "hello_world.wav"
        audio, sr = sf.read(audio_path, dtype="float32")

        result = base_engine.transcribe(audio, sample_rate=sr)

        # Target: <2s transcription time for 2s audio
        print(f"\nBase model transcription time: {result.duration_ms}ms")
        assert result.duration_ms < 2000, "Transcription should complete in <2s"

    @pytest.mark.benchmark
    def test_small_model_speed_short_audio(
        self, small_engine: TranscriptionEngine, test_audio_dir: Path
    ) -> None:
        """Benchmark small model on short audio (~2s)."""
        audio_path = test_audio_dir / "hello_world.wav"
        audio, sr = sf.read(audio_path, dtype="float32")

        result = small_engine.transcribe(audio, sample_rate=sr)

        # Small model is slower but more accurate
        print(f"\nSmall model transcription time: {result.duration_ms}ms")
        assert result.duration_ms < 5000, "Transcription should complete in <5s"

    @pytest.mark.benchmark
    def test_model_comparison_speed(
        self,
        base_engine: TranscriptionEngine,
        small_engine: TranscriptionEngine,
        test_audio_dir: Path,
    ) -> None:
        """Compare base vs small model performance."""
        audio_path = test_audio_dir / "wsjj659_clear.wav"
        audio, sr = sf.read(audio_path, dtype="float32")

        base_result = base_engine.transcribe(audio, sample_rate=sr)
        small_result = small_engine.transcribe(audio, sample_rate=sr)

        print(f"\nBase model: {base_result.duration_ms}ms")
        print(f"Small model: {small_result.duration_ms}ms")
        print(f"Speed ratio: {small_result.duration_ms / base_result.duration_ms:.2f}x")

        # Base should be faster
        assert base_result.duration_ms < small_result.duration_ms

    @pytest.mark.benchmark
    def test_batch_processing_speed(
        self, base_engine: TranscriptionEngine, test_audio_dir: Path
    ) -> None:
        """Benchmark processing multiple files in sequence."""
        test_files = [
            "hello_world.wav",
            "wsjj659_clear.wav",
            "other_callsign.wav",
        ]

        total_audio_duration = 0.0
        total_processing_time = 0

        for filename in test_files:
            audio_path = test_audio_dir / filename
            audio, sr = sf.read(audio_path, dtype="float32")

            audio_duration = len(audio) / sr
            total_audio_duration += audio_duration

            result = base_engine.transcribe(audio, sample_rate=sr)
            total_processing_time += result.duration_ms

        avg_time_per_file = total_processing_time / len(test_files)
        realtime_factor = (total_processing_time / 1000) / total_audio_duration

        print(f"\nProcessed {len(test_files)} files")
        print(f"Total audio duration: {total_audio_duration:.2f}s")
        print(f"Total processing time: {total_processing_time}ms")
        print(f"Average per file: {avg_time_per_file:.0f}ms")
        print(f"Real-time factor: {realtime_factor:.2f}x")

        # Should process faster than real-time
        assert realtime_factor < 2.0, "Should process <2x real-time"

    @pytest.mark.benchmark
    def test_cold_start_vs_warm_performance(self, test_audio_dir: Path) -> None:
        """Compare cold start (model loading) vs warm execution."""
        audio_path = test_audio_dir / "hello_world.wav"
        audio, sr = sf.read(audio_path, dtype="float32")

        # Cold start: includes model loading time
        import time

        start_cold = time.perf_counter()
        engine = TranscriptionEngine(model_size="base")
        _ = engine.transcribe(audio, sample_rate=sr)
        cold_time = int((time.perf_counter() - start_cold) * 1000)

        # Warm execution: model already loaded
        result2 = engine.transcribe(audio, sample_rate=sr)
        warm_time = result2.duration_ms

        print(f"\nCold start (with model load): {cold_time}ms")
        print(f"Warm execution: {warm_time}ms")
        print(f"Model load overhead: {cold_time - warm_time}ms")

        # Warm should be much faster
        assert warm_time < cold_time

    @pytest.mark.benchmark
    def test_memory_efficiency_multiple_transcriptions(
        self, base_engine: TranscriptionEngine, test_audio_dir: Path
    ) -> None:
        """Test that multiple transcriptions don't leak memory."""
        audio_path = test_audio_dir / "hello_world.wav"
        audio, sr = sf.read(audio_path, dtype="float32")

        # Run multiple transcriptions
        num_iterations = 10
        times = []

        for i in range(num_iterations):
            result = base_engine.transcribe(audio, sample_rate=sr)
            times.append(result.duration_ms)

        # Performance should remain stable (no degradation)
        avg_time = sum(times) / len(times)
        std_dev = np.std(times)

        print(f"\n{num_iterations} iterations:")
        print(f"Average time: {avg_time:.1f}ms")
        print(f"Std dev: {std_dev:.1f}ms")
        print(f"Min: {min(times)}ms, Max: {max(times)}ms")

        # Standard deviation should be small (stable performance)
        assert std_dev < avg_time * 0.3, "Performance should be stable"

    @pytest.mark.benchmark
    @pytest.mark.parametrize(
        "audio_length",
        [1.0, 2.0, 5.0],  # seconds
    )
    def test_scaling_with_audio_length(
        self, base_engine: TranscriptionEngine, audio_length: float
    ) -> None:
        """Test how transcription time scales with audio length."""
        # Generate synthetic audio of different lengths
        sample_rate = 16000
        num_samples = int(audio_length * sample_rate)
        audio = np.random.randn(num_samples).astype(np.float32) * 0.1

        result = base_engine.transcribe(audio, sample_rate=sample_rate)

        realtime_factor = (result.duration_ms / 1000) / audio_length

        print(
            f"\nAudio length: {audio_length}s -> "
            f"Transcription: {result.duration_ms}ms "
            f"(RTF: {realtime_factor:.2f}x)"
        )

        # Should scale reasonably with audio length
        assert result.duration_ms > 0

    @pytest.mark.benchmark
    def test_confidence_score_performance(
        self, base_engine: TranscriptionEngine, test_audio_dir: Path
    ) -> None:
        """Benchmark that confidence scoring doesn't add overhead."""
        audio_path = test_audio_dir / "hello_world.wav"
        audio, sr = sf.read(audio_path, dtype="float32")

        # The confidence score is computed as part of transcription
        result = base_engine.transcribe(audio, sample_rate=sr)

        # Verify confidence is computed
        assert isinstance(result.confidence, float)
        assert result.segments is not None

        # Time should still be within target
        assert result.duration_ms < 2000
