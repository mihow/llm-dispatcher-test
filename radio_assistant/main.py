"""Main RadioAssistant application."""

from pathlib import Path
import numpy as np
from pydantic import BaseModel
from loguru import logger
import soundfile as sf

from radio_assistant.audio_interface import AudioInterface
from radio_assistant.vad_detector import VADDetector
from radio_assistant.transcription_engine import TranscriptionEngine
from radio_assistant.callsign_detector import CallsignDetector
from radio_assistant.ptt_controller import PTTController


class AppConfig(BaseModel):
    """Application configuration with validation."""

    callsign: str = "WSJJ659"
    vox_padding_ms: int = 300
    vad_threshold: float = 0.5
    whisper_model: str = "base"
    chunk_duration_sec: float = 0.5
    ptt_method: str = "vox"
    require_dispatch_keyword: bool = True
    enable_phonetic_detection: bool = True
    log_level: str = "INFO"
    response_audio_path: str = "tests/audio/responses/signal_received.wav"
    sample_rate: int = 16000


class RadioAssistant:
    """Main application coordinating all components."""

    def __init__(self, config: AppConfig, audio_interface: AudioInterface | None = None):
        """Initialize RadioAssistant.

        Args:
            config: Application configuration
            audio_interface: Optional custom audio interface (for testing)
        """
        self.config = config
        self.running = False
        self.buffer: list[np.ndarray] = []

        # Initialize components
        logger.info(f"Initializing RadioAssistant for callsign {config.callsign}")

        if audio_interface is None:
            self.audio = AudioInterface(
                sample_rate=config.sample_rate,
                channels=1,
            )
        else:
            self.audio = audio_interface

        self.vad = VADDetector(threshold=config.vad_threshold)

        self.transcription = TranscriptionEngine(
            model_size=config.whisper_model,
            device="cpu",
            compute_type="int8",
        )

        self.callsign = CallsignDetector(
            callsign=config.callsign,
            require_dispatch_keyword=config.require_dispatch_keyword,
            phonetic_alphabet=config.enable_phonetic_detection,
        )

        self.ptt = PTTController(
            method=config.ptt_method,
            vox_padding_ms=config.vox_padding_ms,
            audio_interface=self.audio,
        )

        # Load response audio
        self.response_audio = self._load_response_audio()

        logger.info("RadioAssistant initialized successfully")

    def _load_response_audio(self) -> np.ndarray:
        """Load response audio file.

        Returns:
            Response audio data
        """
        response_path = Path(self.config.response_audio_path)

        if not response_path.exists():
            logger.warning(f"Response audio not found: {response_path}, generating fallback")
            # Generate simple fallback tone
            t = np.linspace(0, 0.5, int(self.config.sample_rate * 0.5))
            return (0.3 * np.sin(2 * np.pi * 800 * t)).astype(np.float32)

        audio, sr = sf.read(response_path, dtype="float32")

        if sr != self.config.sample_rate:
            logger.warning(f"Resampling response audio from {sr} to {self.config.sample_rate}")
            # Simple resampling (for production, use librosa)
            import scipy.signal

            audio = scipy.signal.resample(
                audio, int(len(audio) * self.config.sample_rate / sr)
            ).astype(np.float32)

        logger.info(f"Loaded response audio: {response_path} ({len(audio)} samples)")
        return audio

    def run(self) -> None:
        """Main event loop."""
        logger.info(f"Starting RadioAssistant for {self.config.callsign}")
        self.running = True

        try:
            while self.running:
                # Capture audio chunk
                chunk = self.audio.capture_chunk(self.config.chunk_duration_sec)

                # Voice activity detection
                if self.vad.is_speech(chunk, self.config.sample_rate):
                    self.buffer.append(chunk)
                    logger.debug("Speech detected, buffering")
                elif self.buffer:
                    # Speech ended, process buffer
                    logger.info("Processing buffered speech")
                    self._process_buffer()

        except KeyboardInterrupt:
            logger.info("Stopping RadioAssistant")
            self.running = False
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            self.running = False
            raise

    def process_audio(self, audio: np.ndarray) -> bool:
        """Process a single audio segment (for testing).

        Args:
            audio: Audio data to process

        Returns:
            True if callsign detected and response sent
        """
        # Transcribe
        result = self.transcription.transcribe(audio, self.config.sample_rate)
        logger.info(f"Transcription: {result.text}")

        # Check for callsign
        detection = self.callsign.detect(result.text)
        if detection.detected:
            logger.info(
                f"Callsign detected: {detection.matched_form} "
                f"(confidence={detection.confidence:.2f})"
            )
            self._respond()
            return True
        else:
            logger.debug("Callsign not detected")
            return False

    def _process_buffer(self) -> None:
        """Process buffered audio."""
        if not self.buffer:
            return

        # Concatenate buffered chunks
        audio_data = np.concatenate(self.buffer)
        logger.debug(f"Processing {len(audio_data)} samples")

        # Process audio
        self.process_audio(audio_data)

        # Clear buffer
        self.buffer.clear()

    def _respond(self) -> None:
        """Send response via PTT."""
        logger.info("Transmitting response")
        self.ptt.transmit(self.response_audio, sample_rate=self.config.sample_rate)
        logger.info("Response transmitted")

    def stop(self) -> None:
        """Stop the assistant."""
        logger.info("Stopping RadioAssistant")
        self.running = False
