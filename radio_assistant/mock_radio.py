"""Mock radio interface for testing."""

import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger


class MockRadioInterface:
    """Mock audio interface that plays test audio files instead of live radio."""

    def __init__(self, test_audio_path: str | Path, sample_rate: int = 16000):
        """Initialize mock radio interface.

        Args:
            test_audio_path: Path to test audio file
            sample_rate: Sample rate to use
        """
        self.test_audio_path = Path(test_audio_path)
        self.sample_rate = sample_rate
        self.position = 0
        self.channels = 1

        # Load test audio
        self.test_audio, loaded_sr = sf.read(self.test_audio_path, dtype="float32")

        # Resample if needed
        if loaded_sr != sample_rate:
            logger.warning(f"Test audio sample rate {loaded_sr} != {sample_rate}, resampling")
            import scipy.signal

            num_samples = int(len(self.test_audio) * sample_rate / loaded_sr)
            self.test_audio = scipy.signal.resample(self.test_audio, num_samples).astype(np.float32)

        # Ensure mono
        if self.test_audio.ndim > 1:
            self.test_audio = self.test_audio.mean(axis=1)

        logger.info(
            f"Loaded mock audio: {self.test_audio_path} "
            f"({len(self.test_audio)} samples @ {sample_rate} Hz)"
        )

    def capture_chunk(self, duration_sec: float) -> np.ndarray:
        """Return chunks from test file.

        Args:
            duration_sec: Duration of chunk in seconds

        Returns:
            Audio chunk
        """
        num_samples = int(duration_sec * self.sample_rate)

        # Get chunk from current position
        chunk = self.test_audio[self.position : self.position + num_samples]
        self.position += num_samples

        # If we've reached the end, pad with silence
        if len(chunk) < num_samples:
            padding = num_samples - len(chunk)
            chunk = np.pad(chunk, (0, padding), mode="constant", constant_values=0)
            logger.debug(f"End of test audio reached, padded {padding} samples")

        return chunk.astype(np.float32)

    def play_audio(self, audio_data: np.ndarray) -> None:
        """Mock play_audio (does nothing for testing).

        Args:
            audio_data: Audio data to play
        """
        logger.debug(f"Mock play_audio: {len(audio_data)} samples")
        # In a real test, you might want to capture this for verification

    def reset(self) -> None:
        """Reset playback position to start."""
        self.position = 0
        logger.debug("Mock radio interface reset")

    def has_more_audio(self) -> bool:
        """Check if there's more audio to play.

        Returns:
            True if position hasn't reached end of test audio
        """
        return self.position < len(self.test_audio)

    def get_duration_seconds(self) -> float:
        """Get total duration of test audio.

        Returns:
            Duration in seconds
        """
        return len(self.test_audio) / self.sample_rate
