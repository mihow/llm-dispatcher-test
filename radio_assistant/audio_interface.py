"""Audio I/O abstraction for platform-agnostic audio capture and playback."""

from typing import Dict, List, Optional
import numpy as np
import sounddevice as sd
from loguru import logger


class AudioInterface:
    """Platform-agnostic audio I/O abstraction.

    Handles audio capture and playback using sounddevice for Linux/RPi.
    Designed for future extension to iOS/Android audio APIs.
    """

    def __init__(
        self,
        input_device: Optional[str] = None,
        output_device: Optional[str] = None,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        """Initialize audio interface.

        Args:
            input_device: Name or index of input device (None = default)
            output_device: Name or index of output device (None = default)
            sample_rate: Sample rate in Hz (default: 16000 for Whisper)
            channels: Number of audio channels (1 = mono, 2 = stereo)

        Raises:
            ValueError: If sample_rate or channels are invalid
            RuntimeError: If audio devices cannot be initialized
        """
        if sample_rate <= 0:
            raise ValueError(f"Invalid sample_rate: {sample_rate}, must be positive")
        if channels not in (1, 2):
            raise ValueError(f"Invalid channels: {channels}, must be 1 or 2")

        self.sample_rate: int = sample_rate
        self.channels: int = channels
        self.input_device: Optional[str] = input_device
        self.output_device: Optional[str] = output_device

        # Resolve device names to indices
        self._input_device_index: Optional[int] = self._resolve_device(
            input_device, kind="input"
        )
        self._output_device_index: Optional[int] = self._resolve_device(
            output_device, kind="output"
        )

        logger.info(
            f"AudioInterface initialized: {sample_rate}Hz, {channels}ch, "
            f"input={input_device}, output={output_device}"
        )

    def _resolve_device(
        self, device: Optional[str], kind: str = "input"
    ) -> Optional[int]:
        """Resolve device name to device index.

        Args:
            device: Device name or index, or None for default
            kind: 'input' or 'output'

        Returns:
            Device index, or None for default device

        Raises:
            ValueError: If device name not found
        """
        if device is None:
            return None

        # If device is already an integer, return it
        if isinstance(device, int):
            return device

        # Try to parse as integer
        try:
            device_index: int = int(device)
            return device_index
        except ValueError:
            pass

        # Search for device by name
        devices: List[Dict] = self.list_devices()
        for dev in devices:
            if dev["name"] == device:
                # Check if device supports the requested kind
                if kind == "input" and dev["max_input_channels"] > 0:
                    return dev["index"]
                elif kind == "output" and dev["max_output_channels"] > 0:
                    return dev["index"]

        raise ValueError(
            f"Device '{device}' not found or doesn't support {kind}. "
            f"Use list_devices() to see available devices."
        )

    def capture_chunk(self, duration_sec: float) -> np.ndarray:
        """Capture audio chunk of specified duration.

        Args:
            duration_sec: Duration to capture in seconds

        Returns:
            Audio data as numpy array with shape (samples,) for mono
            or (samples, channels) for stereo. dtype is float32 in range [-1, 1].

        Raises:
            ValueError: If duration_sec is invalid
            RuntimeError: If audio capture fails
        """
        if duration_sec <= 0:
            raise ValueError(f"Invalid duration: {duration_sec}, must be positive")

        num_samples: int = int(duration_sec * self.sample_rate)

        try:
            logger.debug(
                f"Capturing {duration_sec}s ({num_samples} samples) "
                f"from device {self._input_device_index}"
            )

            audio_data: np.ndarray = sd.rec(
                frames=num_samples,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                device=self._input_device_index,
            )
            sd.wait()  # Wait until recording is finished

            # Flatten to 1D for mono
            if self.channels == 1 and len(audio_data.shape) == 2:
                audio_data = audio_data.flatten()

            logger.debug(
                f"Captured chunk: shape={audio_data.shape}, "
                f"dtype={audio_data.dtype}, "
                f"range=[{audio_data.min():.3f}, {audio_data.max():.3f}]"
            )

            return audio_data

        except Exception as e:
            raise RuntimeError(f"Audio capture failed: {e}") from e

    def play_audio(self, audio_data: np.ndarray) -> None:
        """Play audio through output device.

        Args:
            audio_data: Audio data as numpy array. Expected shape is (samples,)
                       for mono or (samples, channels) for stereo.
                       dtype should be float32 in range [-1, 1].

        Raises:
            ValueError: If audio_data shape/dtype is invalid
            RuntimeError: If audio playback fails
        """
        if not isinstance(audio_data, np.ndarray):
            raise ValueError("audio_data must be a numpy array")

        # Validate shape
        if len(audio_data.shape) == 1:
            # Mono audio
            if self.channels != 1:
                raise ValueError(
                    f"Audio data is mono but interface expects {self.channels} channels"
                )
        elif len(audio_data.shape) == 2:
            # Stereo audio
            if audio_data.shape[1] != self.channels:
                raise ValueError(
                    f"Audio data has {audio_data.shape[1]} channels "
                    f"but interface expects {self.channels}"
                )
        else:
            raise ValueError(
                f"Invalid audio data shape: {audio_data.shape}. "
                "Expected (samples,) or (samples, channels)"
            )

        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            logger.debug(f"Converting audio from {audio_data.dtype} to float32")
            audio_data = audio_data.astype(np.float32)

        try:
            logger.debug(
                f"Playing audio: shape={audio_data.shape}, "
                f"duration={len(audio_data) / self.sample_rate:.2f}s"
            )

            sd.play(
                audio_data,
                samplerate=self.sample_rate,
                device=self._output_device_index,
            )
            sd.wait()  # Wait until playback is finished

            logger.debug("Playback completed")

        except Exception as e:
            raise RuntimeError(f"Audio playback failed: {e}") from e

    def list_devices(self) -> List[Dict]:
        """List available audio devices.

        Returns:
            List of device dictionaries containing:
                - index: Device index
                - name: Device name
                - max_input_channels: Number of input channels
                - max_output_channels: Number of output channels
                - default_samplerate: Default sample rate
        """
        devices: List[Dict] = []

        for i, dev in enumerate(sd.query_devices()):
            device_info: Dict = {
                "index": i,
                "name": dev["name"],
                "max_input_channels": dev["max_input_channels"],
                "max_output_channels": dev["max_output_channels"],
                "default_samplerate": dev["default_samplerate"],
            }
            devices.append(device_info)

        return devices

    def get_device_info(self, device: Optional[str] = None, kind: str = "input") -> Dict:
        """Get information about a specific device.

        Args:
            device: Device name or index (None = current device)
            kind: 'input' or 'output'

        Returns:
            Device information dictionary

        Raises:
            ValueError: If device not found
        """
        if device is None:
            device_index: Optional[int] = (
                self._input_device_index if kind == "input" else self._output_device_index
            )
        else:
            device_index = self._resolve_device(device, kind=kind)

        try:
            info: Dict = sd.query_devices(device_index)  # type: ignore
            return dict(info)  # Convert to regular dict
        except Exception as e:
            raise ValueError(f"Failed to get device info: {e}") from e
