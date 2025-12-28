"""Unit tests for AudioInterface."""

import numpy as np
import pytest
from unittest.mock import patch
from radio_assistant.audio_interface import AudioInterface


class TestAudioInterface:
    """Test suite for AudioInterface class."""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        with patch("radio_assistant.audio_interface.sd"):
            audio: AudioInterface = AudioInterface()

            assert audio.sample_rate == 16000
            assert audio.channels == 1
            assert audio.input_device is None
            assert audio.output_device is None

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        with patch("radio_assistant.audio_interface.sd") as mock_sd:
            # Mock device query to return test devices
            mock_sd.query_devices.return_value = [
                {
                    "name": "test_input",
                    "max_input_channels": 2,
                    "max_output_channels": 0,
                    "default_samplerate": 48000,
                },
                {
                    "name": "test_output",
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                    "default_samplerate": 48000,
                },
            ]

            audio: AudioInterface = AudioInterface(
                input_device="test_input",
                output_device="test_output",
                sample_rate=48000,
                channels=2,
            )

            assert audio.sample_rate == 48000
            assert audio.channels == 2
            assert audio.input_device == "test_input"
            assert audio.output_device == "test_output"

    def test_init_invalid_sample_rate(self) -> None:
        """Test initialization with invalid sample rate raises ValueError."""
        with patch("radio_assistant.audio_interface.sd"):
            with pytest.raises(ValueError, match="Invalid sample_rate"):
                AudioInterface(sample_rate=-1)

            with pytest.raises(ValueError, match="Invalid sample_rate"):
                AudioInterface(sample_rate=0)

    def test_init_invalid_channels(self) -> None:
        """Test initialization with invalid channels raises ValueError."""
        with patch("radio_assistant.audio_interface.sd"):
            with pytest.raises(ValueError, match="Invalid channels"):
                AudioInterface(channels=0)

            with pytest.raises(ValueError, match="Invalid channels"):
                AudioInterface(channels=3)

    def test_capture_chunk_mono(self) -> None:
        """Test capturing audio chunk with mono channel."""
        with patch("radio_assistant.audio_interface.sd") as mock_sd:
            # Setup mock to return 2D array (samples, 1)
            mock_audio_data: np.ndarray = np.random.randn(8000, 1).astype(np.float32)
            mock_sd.rec.return_value = mock_audio_data

            audio: AudioInterface = AudioInterface(sample_rate=16000, channels=1)
            result: np.ndarray = audio.capture_chunk(0.5)

            # Verify sd.rec was called correctly
            mock_sd.rec.assert_called_once()
            call_kwargs = mock_sd.rec.call_args[1]
            assert call_kwargs["frames"] == 8000  # 0.5s * 16000 Hz
            assert call_kwargs["samplerate"] == 16000
            assert call_kwargs["channels"] == 1
            assert call_kwargs["dtype"] == np.float32

            # Verify wait was called
            mock_sd.wait.assert_called_once()

            # Verify result shape (should be flattened to 1D)
            assert result.shape == (8000,)
            assert result.dtype == np.float32

    def test_capture_chunk_stereo(self) -> None:
        """Test capturing audio chunk with stereo channels."""
        with patch("radio_assistant.audio_interface.sd") as mock_sd:
            # Setup mock to return 2D array (samples, 2)
            mock_audio_data: np.ndarray = np.random.randn(8000, 2).astype(np.float32)
            mock_sd.rec.return_value = mock_audio_data

            audio: AudioInterface = AudioInterface(sample_rate=16000, channels=2)
            result: np.ndarray = audio.capture_chunk(0.5)

            # Verify result shape (should stay 2D)
            assert result.shape == (8000, 2)
            assert result.dtype == np.float32

    def test_capture_chunk_invalid_duration(self) -> None:
        """Test capture_chunk with invalid duration raises ValueError."""
        with patch("radio_assistant.audio_interface.sd"):
            audio: AudioInterface = AudioInterface()

            with pytest.raises(ValueError, match="Invalid duration"):
                audio.capture_chunk(-1.0)

            with pytest.raises(ValueError, match="Invalid duration"):
                audio.capture_chunk(0.0)

    def test_capture_chunk_handles_exception(self) -> None:
        """Test capture_chunk handles sounddevice exceptions."""
        with patch("radio_assistant.audio_interface.sd") as mock_sd:
            mock_sd.rec.side_effect = Exception("Test error")

            audio: AudioInterface = AudioInterface()

            with pytest.raises(RuntimeError, match="Audio capture failed"):
                audio.capture_chunk(1.0)

    def test_play_audio_mono(self) -> None:
        """Test playing mono audio."""
        with patch("radio_assistant.audio_interface.sd") as mock_sd:
            audio: AudioInterface = AudioInterface(sample_rate=16000, channels=1)

            audio_data: np.ndarray = np.random.randn(8000).astype(np.float32)
            audio.play_audio(audio_data)

            # Verify sd.play was called correctly
            mock_sd.play.assert_called_once()
            call_args = mock_sd.play.call_args
            np.testing.assert_array_equal(call_args[0][0], audio_data)
            assert call_args[1]["samplerate"] == 16000

            # Verify wait was called
            mock_sd.wait.assert_called_once()

    def test_play_audio_stereo(self) -> None:
        """Test playing stereo audio."""
        with patch("radio_assistant.audio_interface.sd") as mock_sd:
            audio: AudioInterface = AudioInterface(sample_rate=16000, channels=2)

            audio_data: np.ndarray = np.random.randn(8000, 2).astype(np.float32)
            audio.play_audio(audio_data)

            # Verify sd.play was called
            mock_sd.play.assert_called_once()

    def test_play_audio_converts_dtype(self) -> None:
        """Test play_audio converts non-float32 arrays."""
        with patch("radio_assistant.audio_interface.sd") as mock_sd:
            audio: AudioInterface = AudioInterface(sample_rate=16000, channels=1)

            # Provide int16 data
            audio_data: np.ndarray = np.random.randint(-32768, 32767, 8000, dtype=np.int16)
            audio.play_audio(audio_data)

            # Verify play was called with float32
            call_args = mock_sd.play.call_args
            assert call_args[0][0].dtype == np.float32

    def test_play_audio_invalid_type(self) -> None:
        """Test play_audio with invalid data type raises ValueError."""
        with patch("radio_assistant.audio_interface.sd"):
            audio: AudioInterface = AudioInterface()

            with pytest.raises(ValueError, match="must be a numpy array"):
                audio.play_audio([1, 2, 3])  # type: ignore

    def test_play_audio_invalid_shape(self) -> None:
        """Test play_audio with invalid shape raises ValueError."""
        with patch("radio_assistant.audio_interface.sd"):
            audio: AudioInterface = AudioInterface(channels=1)

            # 3D array
            with pytest.raises(ValueError, match="Invalid audio data shape"):
                audio.play_audio(np.random.randn(100, 2, 2).astype(np.float32))

    def test_play_audio_channel_mismatch(self) -> None:
        """Test play_audio with channel mismatch raises ValueError."""
        with patch("radio_assistant.audio_interface.sd"):
            audio: AudioInterface = AudioInterface(channels=1)

            # Stereo data but interface expects mono
            with pytest.raises(ValueError, match="Audio data has 2 channels but interface expects 1"):
                audio.play_audio(np.random.randn(100, 2).astype(np.float32))

    def test_play_audio_handles_exception(self) -> None:
        """Test play_audio handles sounddevice exceptions."""
        with patch("radio_assistant.audio_interface.sd") as mock_sd:
            mock_sd.play.side_effect = Exception("Test error")

            audio: AudioInterface = AudioInterface()

            with pytest.raises(RuntimeError, match="Audio playback failed"):
                audio.play_audio(np.random.randn(100).astype(np.float32))

    def test_list_devices(self) -> None:
        """Test listing audio devices."""
        with patch("radio_assistant.audio_interface.sd") as mock_sd:
            mock_devices = [
                {
                    "name": "Device 1",
                    "max_input_channels": 2,
                    "max_output_channels": 2,
                    "default_samplerate": 48000,
                },
                {
                    "name": "Device 2",
                    "max_input_channels": 1,
                    "max_output_channels": 0,
                    "default_samplerate": 16000,
                },
            ]
            mock_sd.query_devices.return_value = mock_devices

            audio: AudioInterface = AudioInterface()
            devices = audio.list_devices()

            assert len(devices) == 2
            assert devices[0]["index"] == 0
            assert devices[0]["name"] == "Device 1"
            assert devices[1]["index"] == 1
            assert devices[1]["name"] == "Device 2"

    def test_resolve_device_by_name(self) -> None:
        """Test resolving device by name."""
        with patch("radio_assistant.audio_interface.sd") as mock_sd:
            mock_devices = [
                {
                    "name": "USB Audio",
                    "max_input_channels": 1,
                    "max_output_channels": 1,
                    "default_samplerate": 16000,
                },
            ]
            mock_sd.query_devices.return_value = mock_devices

            audio: AudioInterface = AudioInterface(input_device="USB Audio")
            assert audio._input_device_index == 0

    def test_resolve_device_not_found(self) -> None:
        """Test resolving device with invalid name raises ValueError."""
        with patch("radio_assistant.audio_interface.sd") as mock_sd:
            mock_devices = [
                {
                    "name": "Device 1",
                    "max_input_channels": 1,
                    "max_output_channels": 0,
                    "default_samplerate": 16000,
                },
            ]
            mock_sd.query_devices.return_value = mock_devices

            with pytest.raises(ValueError, match="not found"):
                AudioInterface(input_device="Nonexistent Device")

    def test_resolve_device_by_index(self) -> None:
        """Test resolving device by index."""
        with patch("radio_assistant.audio_interface.sd"):
            audio: AudioInterface = AudioInterface(input_device="0")
            assert audio._input_device_index == 0

    def test_get_device_info(self) -> None:
        """Test getting device information."""
        with patch("radio_assistant.audio_interface.sd") as mock_sd:
            mock_info = {
                "name": "Test Device",
                "max_input_channels": 2,
                "max_output_channels": 2,
            }
            mock_sd.query_devices.return_value = mock_info

            audio: AudioInterface = AudioInterface()
            info = audio.get_device_info(device="0", kind="input")

            assert info["name"] == "Test Device"
            mock_sd.query_devices.assert_called_with(0)
