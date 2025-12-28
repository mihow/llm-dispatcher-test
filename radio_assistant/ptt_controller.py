"""PTT (Push-to-Talk) control abstraction for radio transmission."""

import numpy as np
from loguru import logger
from radio_assistant.audio_interface import AudioInterface


class PTTController:
    """Transmission control abstraction (VOX now, hardware PTT future)."""

    def __init__(
        self,
        method: str = "vox",
        vox_padding_ms: int = 300,
        audio_interface: AudioInterface | None = None,
    ):
        """Initialize PTT controller.

        Args:
            method: Transmission method ("vox" now, "gpio"/"serial" future)
            vox_padding_ms: VOX pre-trigger padding duration in milliseconds
            audio_interface: AudioInterface instance for playback
        """
        self.method = method
        self.padding_ms = vox_padding_ms
        self.audio = audio_interface

        if method != "vox":
            raise NotImplementedError(
                f"Transmission method '{method}' not yet implemented. "
                "Currently only 'vox' is supported."
            )

        logger.info(f"Initialized PTTController (method={method}, padding={vox_padding_ms}ms)")

    def transmit(self, audio_data: np.ndarray, sample_rate: int = 16000) -> None:
        """Transmit audio using configured method.

        Args:
            audio_data: Audio data to transmit
            sample_rate: Sample rate of audio data (Hz)

        Raises:
            ValueError: If audio_interface not configured
        """
        if self.audio is None:
            raise ValueError("AudioInterface not configured")

        if self.method == "vox":
            # Add VOX padding before actual audio
            padded = self._add_vox_padding(audio_data, sample_rate)

            logger.info(
                f"Transmitting {len(audio_data)} samples "
                f"(+{int(self.padding_ms * sample_rate / 1000)} padding samples)"
            )

            # Transmit via audio interface
            self.audio.play_audio(padded)

        # Future methods (GPIO, serial) would go here
        # elif self.method == "gpio":
        #     self._gpio_transmit(audio_data)
        # elif self.method == "serial":
        #     self._serial_transmit(audio_data)

    def _add_vox_padding(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Prepend silence to trigger VOX before actual audio.

        Args:
            audio: Audio data
            sample_rate: Sample rate (Hz)

        Returns:
            Audio with VOX padding prepended
        """
        # Calculate padding samples
        padding_samples = int(self.padding_ms * sample_rate / 1000)

        # Generate silence with same shape as audio
        if audio.ndim == 1:
            # Mono audio
            silence = np.zeros(padding_samples, dtype=audio.dtype)
        else:
            # Multichannel audio - preserve channel count
            silence = np.zeros((padding_samples, audio.shape[1]), dtype=audio.dtype)

        # Concatenate padding and audio
        padded = np.concatenate([silence, audio])

        logger.debug(f"Added {padding_samples} samples ({self.padding_ms}ms) of VOX padding")

        return padded

    def get_padding_duration_ms(self) -> int:
        """Get configured VOX padding duration.

        Returns:
            Padding duration in milliseconds
        """
        return self.padding_ms

    def set_padding_duration_ms(self, padding_ms: int) -> None:
        """Set VOX padding duration.

        Args:
            padding_ms: Padding duration in milliseconds

        Raises:
            ValueError: If padding is negative
        """
        if padding_ms < 0:
            raise ValueError(f"Padding must be non-negative, got {padding_ms}")

        logger.info(f"VOX padding updated: {self.padding_ms}ms -> {padding_ms}ms")
        self.padding_ms = padding_ms
