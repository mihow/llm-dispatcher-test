"""Voice Activity Detection wrapper using Silero VAD."""

from typing import Dict, List, Optional
import numpy as np
import torch
import silero_vad
from loguru import logger


class VADDetector:
    """Voice Activity Detection wrapper using Silero VAD.

    Detects speech in audio data and provides speech/silence timestamps.
    Optimized for radio-quality audio with noise and static.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        sample_rate: int = 16000,
    ) -> None:
        """Initialize VAD detector.

        Args:
            threshold: Speech probability threshold (0.0-1.0). Lower = more sensitive
            min_speech_duration_ms: Minimum speech duration to detect (ms)
            min_silence_duration_ms: Minimum silence duration between speech (ms)
            sample_rate: Audio sample rate in Hz (default: 16000)

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If VAD model cannot be loaded
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Invalid threshold: {threshold}, must be in [0.0, 1.0]")
        if min_speech_duration_ms <= 0:
            raise ValueError(
                f"Invalid min_speech_duration_ms: {min_speech_duration_ms}, must be positive"
            )
        if min_silence_duration_ms <= 0:
            raise ValueError(
                f"Invalid min_silence_duration_ms: {min_silence_duration_ms}, must be positive"
            )
        if sample_rate not in (8000, 16000):
            raise ValueError(f"Invalid sample_rate: {sample_rate}, must be 8000 or 16000")

        self.threshold: float = threshold
        self.min_speech_duration_ms: int = min_speech_duration_ms
        self.min_silence_duration_ms: int = min_silence_duration_ms
        self.sample_rate: int = sample_rate

        # Load Silero VAD model
        try:
            logger.info("Loading Silero VAD model...")
            self.model = self._load_model()
            logger.info("Silero VAD model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load VAD model: {e}") from e

        logger.info(
            f"VADDetector initialized: threshold={threshold}, "
            f"min_speech={min_speech_duration_ms}ms, "
            f"min_silence={min_silence_duration_ms}ms, "
            f"sample_rate={sample_rate}Hz"
        )

    def _load_model(self):
        """Load Silero VAD model.

        Returns:
            Silero VAD model

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            model = silero_vad.load_silero_vad()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load Silero VAD model: {e}") from e

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Detect if audio chunk contains speech.

        Note: This method uses get_speech_timestamps under the hood.
        For better performance on long audio, use get_speech_timestamps directly.

        Args:
            audio_chunk: Audio data as numpy array (float32, mono)

        Returns:
            True if speech detected, False otherwise

        Raises:
            ValueError: If audio_chunk is invalid
        """
        if not isinstance(audio_chunk, np.ndarray):
            raise ValueError("audio_chunk must be a numpy array")

        if len(audio_chunk) == 0:
            return False

        # Use get_speech_timestamps which handles chunk sizes correctly
        try:
            timestamps = self.get_speech_timestamps(audio_chunk, self.sample_rate)
            is_speech_detected: bool = len(timestamps) > 0

            logger.debug(
                f"Speech detection: {len(timestamps)} segments found, "
                f"detected: {is_speech_detected}"
            )

            return is_speech_detected

        except Exception as e:
            logger.error(f"Error during speech detection: {e}")
            return False

    def get_speech_timestamps(
        self, audio: np.ndarray, sample_rate: Optional[int] = None
    ) -> List[Dict]:
        """Get start/end timestamps of speech segments in audio.

        Args:
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Sample rate (default: use detector's sample_rate)

        Returns:
            List of dicts with 'start' and 'end' timestamps in seconds:
            [{"start": 0.5, "end": 2.3}, ...]

        Raises:
            ValueError: If audio or sample_rate is invalid
        """
        if not isinstance(audio, np.ndarray):
            raise ValueError("audio must be a numpy array")

        if sample_rate is None:
            sample_rate = self.sample_rate

        if sample_rate != self.sample_rate:
            logger.warning(
                f"Sample rate mismatch: provided {sample_rate}Hz, "
                f"detector expects {self.sample_rate}Hz"
            )

        # Convert to torch tensor
        audio_tensor: torch.Tensor = self._prepare_audio(audio)

        # Get speech timestamps from Silero VAD
        try:
            with torch.no_grad():
                speech_timestamps = silero_vad.get_speech_timestamps(
                    audio_tensor,
                    self.model,
                    threshold=self.threshold,
                    sampling_rate=sample_rate,
                    min_speech_duration_ms=self.min_speech_duration_ms,
                    min_silence_duration_ms=self.min_silence_duration_ms,
                    return_seconds=False,  # Get samples first
                )

            # Convert to seconds
            result: List[Dict] = []
            for ts in speech_timestamps:
                start_sec: float = float(ts["start"]) / sample_rate
                end_sec: float = float(ts["end"]) / sample_rate
                result.append({"start": start_sec, "end": end_sec})

            logger.debug(
                f"Detected {len(result)} speech segments in "
                f"{len(audio) / sample_rate:.2f}s of audio"
            )

            return result

        except Exception as e:
            logger.error(f"Error during speech timestamp detection: {e}")
            return []

    def _prepare_audio(self, audio: np.ndarray) -> torch.Tensor:
        """Prepare audio for VAD processing.

        Args:
            audio: Audio data as numpy array

        Returns:
            Audio as torch tensor (float32)

        Raises:
            ValueError: If audio shape is invalid
        """
        # Ensure 1D array
        if len(audio.shape) > 1:
            if audio.shape[1] == 1:
                audio = audio.flatten()
            else:
                raise ValueError(
                    f"Audio must be mono. Got shape: {audio.shape}. "
                    "Use AudioInterface with channels=1"
                )

        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize to [-1, 1] if needed
        max_val: float = float(np.abs(audio).max())
        if max_val > 1.0:
            logger.warning(f"Audio values exceed [-1, 1] range (max: {max_val}), normalizing")
            audio = audio / max_val

        # Convert to torch tensor
        audio_tensor: torch.Tensor = torch.from_numpy(audio)

        return audio_tensor

    def reset(self) -> None:
        """Reset VAD detector state.

        Useful when processing multiple independent audio streams.
        """
        logger.debug("Resetting VAD detector state")
        # Silero VAD model is stateless, so just log
        pass
