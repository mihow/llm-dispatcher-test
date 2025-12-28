"""Speech-to-text transcription engine using Whisper."""

from dataclasses import dataclass
import time
import numpy as np
from faster_whisper import WhisperModel
from loguru import logger


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription."""

    text: str
    confidence: float
    duration_ms: int
    segments: list[dict]


class TranscriptionEngine:
    """Speech-to-text wrapper using faster-whisper."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        """Initialize transcription engine.

        Args:
            model_size: Whisper model size (base, small, medium, large)
            device: Device to run on (cpu, cuda)
            compute_type: Compute type for inference (int8, float16, float32)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

        logger.info(f"Loading Whisper model: {model_size} on {device} with {compute_type}")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("Whisper model loaded successfully")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of audio (must be 16000 for Whisper)

        Returns:
            TranscriptionResult with text, confidence, timing, and segments
        """
        if sample_rate != 16000:
            raise ValueError(f"Sample rate must be 16000 Hz for Whisper, got {sample_rate}")

        start_time = time.perf_counter()

        # faster-whisper expects float32 audio in [-1, 1] range
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        # Transcribe
        segments_iter, info = self.model.transcribe(
            audio, beam_size=5, language="en", task="transcribe"
        )

        # Collect segments
        segments = []
        full_text = []
        total_confidence = 0.0
        segment_count = 0

        for segment in segments_iter:
            segments.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "confidence": segment.avg_logprob,
                }
            )
            full_text.append(segment.text)
            total_confidence += segment.avg_logprob
            segment_count += 1

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        # Calculate average confidence
        avg_confidence = total_confidence / segment_count if segment_count > 0 else 0.0

        # Combine text from all segments
        text = " ".join(full_text).strip()

        logger.debug(f"Transcribed in {duration_ms}ms: '{text}' (confidence: {avg_confidence:.3f})")

        return TranscriptionResult(
            text=text,
            confidence=avg_confidence,
            duration_ms=duration_ms,
            segments=segments,
        )
