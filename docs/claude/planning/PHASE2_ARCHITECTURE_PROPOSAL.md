# Phase 2 Architecture Proposal: Radio Voice Assistant Service

**Created**: 2025-12-28
**Status**: Proposal (awaiting review)
**Based On**: Research of Rhasspy, Home Assistant, and Mycroft architectures
**Target**: Streaming service mode for continuous radio operation

---

## Executive Summary

This proposal outlines the architecture for Phase 2: transforming the current batch-processing radio voice assistant into a **long-running async service** capable of continuous operation on a Raspberry Pi.

**Key Design Principles**:
1. **Single-process async architecture** for resource efficiency (<150MB RAM target)
2. **Event-driven coordination** using in-process async queues (no WebSocket/MQTT overhead)
3. **Template-based callsign detection** (DTW) for custom wake words without neural networks
4. **Proven patterns** from production voice assistants (Rhasspy, Home Assistant, Mycroft)
5. **Offline-first** for field deployment without internet dependency

**Resource Comparison**:
| System | Processes | RAM | Network | Target Device |
|--------|-----------|-----|---------|---------------|
| Mycroft | 5 | ~320MB | WebSocket (port 8181, no auth) | Desktop/Pi 4 |
| Rhasspy 2.x | 5+ | ~200MB | MQTT (optional external) | Pi 3/4 |
| Home Assistant | 1 | Varies | Wyoming protocol | Pi 4 |
| **Radio Assistant** | **1** | **<150MB** | **None** | **Pi 3/4** |

---

## 1. Overall Architecture

### 1.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    RadioAssistantService                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                      EventBus                              │  │
│  │  (asyncio.Queue-based in-process message passing)         │  │
│  └───────────────────────────────────────────────────────────┘  │
│         ↕              ↕              ↕              ↕           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │  Audio   │  │  Callsign│  │Transcript│  │   PTT    │      │
│  │  Manager │  │ Detector │  │  Engine  │  │Controller│      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
│         ↕              ↕              ↕              ↕           │
│  ┌────────────────────────────────────────────────────────┐    │
│  │             Streaming VAD Component                     │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
         ↕                                            ↕
   Radio RX Audio                              Radio TX (PTT)
  (via soundcard)                             (via GPIO/VOX)
```

### 1.2 Component Responsibilities

| Component | Purpose | Events Emitted | Events Handled |
|-----------|---------|----------------|----------------|
| **AudioManager** | Capture radio RX audio | `audio.chunk` | `audio.stop`, `audio.start` |
| **StreamingVAD** | Detect speech boundaries | `vad.speech.start`, `vad.speech.complete` | `audio.chunk` |
| **CallsignDetector** | Identify callsigns (DTW) | `callsign.detected` | `vad.speech.complete` |
| **TranscriptionEngine** | Speech-to-text | `transcription.complete` | `vad.speech.complete` |
| **PTTController** | Manage transmit/receive | `ptt.activate`, `ptt.release`, `ptt.complete` | `radio.respond` |
| **ConversationManager** | Pipeline orchestration | `radio.respond`, `session.end` | `callsign.detected`, `transcription.complete` |
| **EventBus** | Message routing | N/A | All events |

---

## 2. Event Bus Architecture

### 2.1 Design

**Pattern**: Simplified Mycroft message bus (in-process, no WebSocket)

```python
from dataclasses import dataclass
from collections import defaultdict
import asyncio
import typing

@dataclass
class Event:
    type: str
    data: dict
    timestamp: float = field(default_factory=time.time)

class AsyncEventBus:
    \"\"\"Lightweight in-process event bus inspired by Mycroft's MessageBus\"\"\"

    def __init__(self, max_queue_size: int = 100):
        self.handlers: dict[str, list[typing.Callable]] = defaultdict(list)
        self.queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue_size)
        self.running = False

    def on(self, event_type: str, handler: typing.Callable):
        \"\"\"Register event handler (like Mycroft's add_event)\"\"\"
        self.handlers[event_type].append(handler)
        logger.debug(f\"Registered handler for '{event_type}': {handler.__name__}\")

    def off(self, event_type: str, handler: typing.Callable):
        \"\"\"Unregister event handler\"\"\"
        if handler in self.handlers[event_type]:
            self.handlers[event_type].remove(handler)

    async def emit(self, event_type: str, data: dict | None = None):
        \"\"\"Emit event to all handlers (like Mycroft's bus.emit)\"\"\"
        event = Event(type=event_type, data=data or {})

        try:
            self.queue.put_nowait(event)
            logger.debug(f\"Emitted event: {event_type}\")
        except asyncio.QueueFull:
            logger.error(f\"Event queue full, dropping event: {event_type}\")
            # Consider: emit 'system.backpressure' event?

    async def dispatch_loop(self):
        \"\"\"Process events from queue (similar to MessageBus event loop)\"\"\"
        self.running = True
        logger.info(\"Event bus dispatch loop started\")

        while self.running:
            try:
                event = await self.queue.get()

                # Dispatch to all registered handlers
                for handler in self.handlers.get(event.type, []):
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event.data)
                        else:
                            handler(event.data)
                    except Exception as e:
                        logger.error(f\"Handler error for {event.type}: {e}\", exc_info=True)
                        await self.emit('system.error', {
                            'component': handler.__name__,
                            'event_type': event.type,
                            'error': str(e)
                        })

                self.queue.task_done()

            except asyncio.CancelledError:
                logger.info(\"Event bus dispatch loop cancelled\")
                break
            except Exception as e:
                logger.error(f\"Event bus error: {e}\", exc_info=True)

        logger.info(\"Event bus dispatch loop stopped\")

    async def stop(self):
        \"\"\"Stop the dispatch loop\"\"\"
        self.running = False
```

### 2.2 Event Naming Convention

Following Mycroft's semantic naming:

- **Action requests** (verbs): `mic.mute`, `ptt.activate`
- **Pre-action** (present continuous): `ptt.activating`
- **Post-action** (past tense): `ptt.activated`, `transcription.complete`
- **State changes**: `vad.speech.start`, `vad.speech.end`
- **System events**: `system.ready`, `system.stop`, `system.error`

**Radio Assistant Events**:
```python
# Audio pipeline
'audio.chunk'                   # Audio data available
'audio.start', 'audio.stop'     # Audio capture control

# Voice Activity Detection
'vad.speech.start'              # Speech detected
'vad.speech.complete'           # Full utterance captured (data: audio, duration)

# Callsign Detection
'callsign.detected'             # Callsign recognized (data: callsign, confidence)
'callsign.enable', 'callsign.disable'  # Detection control

# Transcription
'transcription.start'           # STT started
'transcription.complete'        # STT finished (data: text, confidence)

# Radio Control
'radio.respond'                 # Generate response (data: text, context)
'ptt.activate', 'ptt.release'   # PTT control
'ptt.complete'                  # Transmission finished

# Session Management
'session.start'                 # Conversation started
'session.continue'              # Multi-turn continuation
'session.end'                   # Conversation ended
'session.timeout'               # Inactivity timeout

# System Lifecycle
'system.ready'                  # All components initialized
'system.stop'                   # Shutdown requested
'system.error'                  # Component error (data: component, error)
```

### 2.3 Benefits Over WebSocket/MQTT

| Aspect | WebSocket (Mycroft) | MQTT (Rhasspy) | **AsyncQueue (Proposed)** |
|--------|---------------------|----------------|---------------------------|
| Latency | 10-50ms | 5-20ms | **<1ms** |
| Overhead | Serialization + network | Broker + network | **Queue ops only** |
| Security | Port 8181, no auth | Port 1883, optional auth | **No network** |
| Resource | WebSocket connections | MQTT broker process | **asyncio primitives** |
| Debugging | Network inspection tools | MQTT clients | **Direct Python debugging** |
| Testing | Mock WebSocket server | Mock MQTT broker | **Direct function calls** |

---

## 3. Audio Streaming Architecture

### 3.1 PyAudio Callback Pattern

Following Rhasspy's thread-safe approach:

```python
import pyaudio
import numpy as np

class AudioManager:
    \"\"\"Manages radio RX audio capture with PyAudio\"\"\"

    # Audio configuration (standard for voice processing)
    SAMPLE_RATE = 16000
    CHANNELS = 1  # Mono
    SAMPLE_WIDTH = 2  # 16-bit
    CHUNK_FRAMES = 480  # 30ms chunks (good for low latency)
    CHUNK_BYTES = CHUNK_FRAMES * SAMPLE_WIDTH

    def __init__(self, event_bus: AsyncEventBus, device_index: int | None = None):
        self.bus = event_bus
        self.device_index = device_index
        self.stream = None
        self.p = None
        self.audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)

        # Register handlers
        self.bus.on('audio.stop', self.on_stop_request)
        self.bus.on('audio.start', self.on_start_request)

    def _callback(self, in_data, frame_count, time_info, status):
        \"\"\"PyAudio callback - queue only, don't process (thread safety)\"\"\"
        if status:
            logger.warning(f\"PyAudio status: {status}\")

        if in_data:
            try:
                # Non-blocking queue put (drop if full = backpressure)
                self.audio_queue.put_nowait(in_data)
            except asyncio.QueueFull:
                # Drop frame rather than block
                pass

        return (in_data, pyaudio.paContinue)

    async def input_loop(self):
        \"\"\"Async loop that emits audio chunks as events\"\"\"
        logger.info(\"Audio input loop started\")

        # Open PyAudio stream
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.CHANNELS,
            rate=self.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.CHUNK_FRAMES,
            input_device_index=self.device_index,
            stream_callback=self._callback,
        )

        self.stream.start_stream()
        await self.bus.emit('system.ready', {'component': 'AudioManager'})

        # Process queued audio
        try:
            while True:
                audio_bytes = await self.audio_queue.get()

                # Convert to numpy array for processing
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

                # Emit event (downstream components process)
                await self.bus.emit('audio.chunk', {
                    'audio': audio_np,
                    'sample_rate': self.SAMPLE_RATE,
                    'timestamp': time.time(),
                })

        except asyncio.CancelledError:
            logger.info(\"Audio input loop cancelled\")
        finally:
            await self.cleanup()

    async def cleanup(self):
        \"\"\"Cleanup audio resources\"\"\"
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        logger.info(\"Audio cleanup complete\")

    async def on_stop_request(self, data):
        \"\"\"Handle audio.stop event\"\"\"
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            logger.info(\"Audio capture stopped\")

    async def on_start_request(self, data):
        \"\"\"Handle audio.start event\"\"\"
        if self.stream and not self.stream.is_active():
            self.stream.start_stream()
            logger.info(\"Audio capture started\")
```

### 3.2 Streaming VAD Component

Following Rhasspy's energy-based approach + Home Assistant's async patterns:

```python
import webrtcvad

class StreamingVAD:
    \"\"\"Voice Activity Detection for continuous audio stream\"\"\"

    def __init__(self, event_bus: AsyncEventBus, aggressiveness: int = 2):
        self.bus = event_bus
        self.vad = webrtcvad.Vad(aggressiveness)  # 0-3 (3 = most aggressive)

        # Speech detection state
        self.is_speech_active = False
        self.speech_buffer = []
        self.silence_chunks = 0

        # Configuration
        self.min_speech_chunks = 10  # Minimum chunks to consider speech (300ms)
        self.max_silence_chunks = 20  # Max silence before ending (600ms)

        # Register handler
        self.bus.on('audio.chunk', self.on_audio_chunk)

    async def on_audio_chunk(self, data):
        \"\"\"Process audio chunk for voice activity\"\"\"
        audio = data['audio']
        sample_rate = data['sample_rate']

        # Convert to bytes for WebRTC VAD
        audio_bytes = audio.tobytes()

        # WebRTC VAD check (requires specific frame lengths: 10/20/30ms)
        # Our 30ms chunks (480 frames @ 16kHz) match this
        is_speech = self.vad.is_speech(audio_bytes, sample_rate)

        if is_speech:
            # Speech detected
            if not self.is_speech_active:
                # Speech start
                self.is_speech_active = True
                self.speech_buffer = []
                self.silence_chunks = 0
                await self.bus.emit('vad.speech.start', {'timestamp': time.time()})
                logger.debug(\"Speech start detected\")

            self.speech_buffer.append(audio)
            self.silence_chunks = 0

        else:
            # Silence detected
            if self.is_speech_active:
                self.silence_chunks += 1

                # Still in speech, buffer silence (might be pause)
                if self.silence_chunks <= self.max_silence_chunks:
                    self.speech_buffer.append(audio)
                else:
                    # Enough silence, speech ended
                    await self._emit_complete_speech()

    async def _emit_complete_speech(self):
        \"\"\"Emit complete speech utterance\"\"\"
        if len(self.speech_buffer) < self.min_speech_chunks:
            # Too short, ignore (likely noise)
            logger.debug(f\"Speech too short ({len(self.speech_buffer)} chunks), ignoring\")
            self.is_speech_active = False
            self.speech_buffer = []
            return

        # Concatenate buffered audio
        complete_audio = np.concatenate(self.speech_buffer)
        duration = len(complete_audio) / AudioManager.SAMPLE_RATE

        await self.bus.emit('vad.speech.complete', {
            'audio': complete_audio,
            'duration': duration,
            'sample_rate': AudioManager.SAMPLE_RATE,
            'timestamp': time.time(),
        })

        logger.info(f\"Speech complete: {duration:.2f}s, {len(complete_audio)} samples\")

        # Reset state
        self.is_speech_active = False
        self.speech_buffer = []
        self.silence_chunks = 0
```

---

## 4. Callsign Detection (Template-Based DTW)

Following Rhasspy Raven's approach:

```python
from python_speech_features import mfcc
from dtw import dtw
import numpy as np

class CallsignDetector:
    \"\"\"Template-based callsign detection using Dynamic Time Warping\"\"\"

    def __init__(self, event_bus: AsyncEventBus, templates_dir: str):
        self.bus = event_bus
        self.templates = self._load_templates(templates_dir)
        self.enabled = True
        self.refractory_until = 0

        # Configuration (Rhasspy Raven defaults)
        self.distance_threshold = 0.22  # Normalized DTW distance
        self.probability_threshold = 0.5  # Detection confidence
        self.minimum_matches = 1  # Templates that must match

        # Register handlers
        self.bus.on('vad.speech.complete', self.on_speech_complete)
        self.bus.on('callsign.enable', self.on_enable)
        self.bus.on('callsign.disable', self.on_disable)

    def _load_templates(self, templates_dir: str) -> dict[str, list[np.ndarray]]:
        \"\"\"Load MFCC templates for each callsign\"\"\"
        templates = {}

        for template_file in Path(templates_dir).glob('*.npy'):
            callsign = template_file.stem.split('_')[0]  # e.g., "WSJJ659_01.npy" -> "WSJJ659"
            mfcc_template = np.load(template_file)

            if callsign not in templates:
                templates[callsign] = []
            templates[callsign].append(mfcc_template)

        logger.info(f\"Loaded templates for {len(templates)} callsigns\")
        return templates

    async def on_speech_complete(self, data):
        \"\"\"Check if speech contains a callsign\"\"\"
        if not self.enabled:
            return

        # Check refractory period (prevent rapid re-triggering)
        if time.time() < self.refractory_until:
            logger.debug(\"In refractory period, skipping detection\")
            return

        audio = data['audio']
        sample_rate = data['sample_rate']

        # Extract MFCC features
        features = mfcc(audio, samplerate=sample_rate)

        # Compare against all callsign templates
        best_match = None
        best_confidence = 0

        for callsign, templates in self.templates.items():
            matches = 0

            for template in templates:
                # Calculate DTW distance
                distance, _ = dtw(features, template)
                normalized_dist = distance / len(features)

                if normalized_dist < self.distance_threshold:
                    matches += 1

            # Check if enough templates matched
            if matches >= self.minimum_matches:
                confidence = matches / len(templates)

                if confidence >= self.probability_threshold:
                    if confidence > best_confidence:
                        best_match = callsign
                        best_confidence = confidence

        if best_match:
            # Set refractory period (3 seconds)
            self.refractory_until = time.time() + 3.0

            await self.bus.emit('callsign.detected', {
                'callsign': best_match,
                'confidence': best_confidence,
                'timestamp': time.time(),
            })

            logger.info(f\"Callsign detected: {best_match} (confidence: {best_confidence:.2f})\")

    async def on_enable(self, data):
        self.enabled = True
        logger.info(\"Callsign detection enabled\")

    async def on_disable(self, data):
        self.enabled = False
        logger.info(\"Callsign detection disabled\")
```

### 4.1 Template Generation Script

```python
import soundfile as sf
from python_speech_features import mfcc
import numpy as np

def create_callsign_template(wav_path: str, output_path: str):
    \"\"\"Create MFCC template from WAV recording\"\"\"
    # Load audio
    audio, sample_rate = sf.read(wav_path)

    # Trim silence from edges
    audio = trim_silence(audio, threshold=0.01)

    # Extract MFCC features
    features = mfcc(audio, samplerate=sample_rate)

    # Save template
    np.save(output_path, features)
    logger.info(f\"Created template: {output_path}\")

def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    \"\"\"Trim silence from start and end of audio\"\"\"
    # Find non-silent samples
    non_silent = np.abs(audio) > threshold
    if not np.any(non_silent):
        return audio

    # Find start and end indices
    start = np.argmax(non_silent)
    end = len(audio) - np.argmax(non_silent[::-1])

    return audio[start:end]

# Usage:
# 1. Record 3+ examples of each callsign
# 2. Generate templates:
create_callsign_template('recordings/WSJJ659_01.wav', 'templates/WSJJ659_01.npy')
create_callsign_template('recordings/WSJJ659_02.wav', 'templates/WSJJ659_02.npy')
create_callsign_template('recordings/WSJJ659_03.wav', 'templates/WSJJ659_03.npy')
```

---

## 5. Conversation Management & Session Lifecycle

Following Rhasspy's dialogue manager pattern:

```python
from enum import Enum
import uuid

class PipelineState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"

class ConversationManager:
    \"\"\"Orchestrates voice pipeline and session lifecycle\"\"\"

    def __init__(self, event_bus: AsyncEventBus):
        self.bus = event_bus
        self.state = PipelineState.IDLE
        self.session: dict | None = None
        self.session_timeout = 30  # seconds

        # Register handlers
        self.bus.on('callsign.detected', self.on_callsign_detected)
        self.bus.on('transcription.complete', self.on_transcription_complete)
        self.bus.on('ptt.complete', self.on_response_complete)

    async def on_callsign_detected(self, data):
        \"\"\"Callsign detected - start conversation session\"\"\"
        if self.state != PipelineState.IDLE:
            logger.warning(f\"Callsign detected but state is {self.state}, ignoring\")
            return

        # Start new session
        self.session = {
            'id': str(uuid.uuid4()),
            'callsign': data['callsign'],
            'confidence': data['confidence'],
            'started': time.time(),
        }

        self.state = PipelineState.LISTENING

        await self.bus.emit('session.start', self.session)

        # Disable callsign detection during session (prevent re-triggering)
        await self.bus.emit('callsign.disable')

        logger.info(f\"Session started: {self.session['id']} for {self.session['callsign']}\")

    async def on_transcription_complete(self, data):
        \"\"\"Speech transcribed - generate response\"\"\"
        if self.state != PipelineState.LISTENING:
            return

        # Check session timeout
        if self.session and (time.time() - self.session['started']) > self.session_timeout:
            logger.warning(\"Session timeout\")
            await self.end_session()
            return

        self.state = PipelineState.PROCESSING

        text = data['text']
        logger.info(f\"Transcription: {text}\")

        # TODO: Intent recognition, response generation
        # For now, simple echo response
        response_text = f\"{self.session['callsign']}, roger. You said: {text}. Over.\"

        await self.bus.emit('radio.respond', {
            'text': response_text,
            'callsign': self.session['callsign'],
        })

        self.state = PipelineState.RESPONDING

    async def on_response_complete(self, data):
        \"\"\"Response transmitted - end session\"\"\"
        if self.state == PipelineState.RESPONDING:
            await self.end_session()

    async def end_session(self):
        \"\"\"End conversation session and return to idle\"\"\"
        if self.session:
            duration = time.time() - self.session['started']
            logger.info(f\"Session ended: {self.session['id']} (duration: {duration:.1f}s)\")

            await self.bus.emit('session.end', {
                'session_id': self.session['id'],
                'duration': duration,
            })

        self.session = None
        self.state = PipelineState.IDLE

        # Re-enable callsign detection
        await self.bus.emit('callsign.enable')

    async def monitor_timeout(self):
        \"\"\"Background task to monitor session timeouts\"\"\"
        while True:
            await asyncio.sleep(5)  # Check every 5 seconds

            if self.session:
                elapsed = time.time() - self.session['started']
                if elapsed > self.session_timeout:
                    logger.warning(f\"Session timeout: {self.session['id']}\")
                    await self.bus.emit('session.timeout', {'session_id': self.session['id']})
                    await self.end_session()
```

---

## 6. PTT Controller (Radio-Specific)

```python
import RPi.GPIO as GPIO

class PTTController:
    \"\"\"Manages Push-To-Talk for radio transmission\"\"\"

    def __init__(self, event_bus: AsyncEventBus, gpio_pin: int, vox_padding_ms: int = 200):
        self.bus = event_bus
        self.gpio_pin = gpio_pin
        self.vox_padding_ms = vox_padding_ms
        self.is_transmitting = False

        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.gpio_pin, GPIO.OUT)
        GPIO.output(self.gpio_pin, GPIO.LOW)  # PTT inactive

        # Register handlers
        self.bus.on('radio.respond', self.on_respond_request)
        self.bus.on('ptt.activate', self.on_activate)
        self.bus.on('ptt.release', self.on_release)

    async def on_respond_request(self, data):
        \"\"\"Generate TTS and transmit over radio\"\"\"
        text = data['text']
        logger.info(f\"Generating response: {text}\")

        # Generate TTS audio
        audio = await self.generate_tts(text)

        # Transmit
        await self.transmit(audio)

    async def transmit(self, audio: np.ndarray):
        \"\"\"Transmit audio over radio\"\"\"
        if self.is_transmitting:
            logger.warning(\"Already transmitting, ignoring request\")
            return

        try:
            # Activate PTT
            await self.bus.emit('ptt.activate')

            # Wait for VOX padding (radio TX ramp-up)
            await asyncio.sleep(self.vox_padding_ms / 1000)

            # Play audio to radio TX
            await self.play_audio(audio)

            # Wait for VOX padding (radio TX tail)
            await asyncio.sleep(self.vox_padding_ms / 1000)

        finally:
            # Release PTT
            await self.bus.emit('ptt.release')

            # Signal completion
            await self.bus.emit('ptt.complete', {'timestamp': time.time()})

    async def on_activate(self, data):
        \"\"\"Activate PTT (GPIO high)\"\"\"
        GPIO.output(self.gpio_pin, GPIO.HIGH)
        self.is_transmitting = True
        logger.info(\"PTT activated\")

    async def on_release(self, data):
        \"\"\"Release PTT (GPIO low)\"\"\"
        GPIO.output(self.gpio_pin, GPIO.LOW)
        self.is_transmitting = False
        logger.info(\"PTT released\")

    async def generate_tts(self, text: str) -> np.ndarray:
        \"\"\"Generate TTS audio (placeholder)\"\"\"
        # TODO: Integrate Piper TTS or similar
        # For now, return silence
        duration_s = len(text) * 0.1  # Rough estimate
        samples = int(duration_s * AudioManager.SAMPLE_RATE)
        return np.zeros(samples, dtype=np.int16)

    async def play_audio(self, audio: np.ndarray):
        \"\"\"Play audio to radio TX (via sound card output)\"\"\"
        # TODO: Implement using PyAudio output stream
        await asyncio.sleep(len(audio) / AudioManager.SAMPLE_RATE)
        logger.info(f\"Played {len(audio)} samples\")

    def cleanup(self):
        \"\"\"Cleanup GPIO\"\"\"
        GPIO.output(self.gpio_pin, GPIO.LOW)
        GPIO.cleanup()
```

---

## 7. Main Service Implementation

```python
class RadioAssistantService:
    \"\"\"Main service orchestrating all components\"\"\"

    def __init__(self, config: dict):
        self.config = config
        self.event_bus = AsyncEventBus()

        # Initialize components
        self.audio = AudioManager(self.event_bus, device_index=config.get('audio_device'))
        self.vad = StreamingVAD(self.event_bus, aggressiveness=config.get('vad_aggressiveness', 2))
        self.callsign = CallsignDetector(self.event_bus, templates_dir=config['templates_dir'])
        self.transcription = TranscriptionEngine(self.event_bus, model_size=config.get('whisper_model', 'base'))
        self.conversation = ConversationManager(self.event_bus)
        self.ptt = PTTController(self.event_bus, gpio_pin=config['ptt_gpio_pin'])

        self.cleanup_callbacks = []

    async def start(self):
        \"\"\"Start the service\"\"\"
        logger.info(\"Starting Radio Voice Assistant Service\")

        # Start event bus dispatch
        bus_task = asyncio.create_task(self.event_bus.dispatch_loop(), name='event_bus')

        # Start all component loops
        tasks = [
            asyncio.create_task(self.audio.input_loop(), name='audio_input'),
            asyncio.create_task(self.conversation.monitor_timeout(), name='session_timeout'),
        ]

        # Register cleanup
        self.cleanup_callbacks = [
            self.audio.cleanup,
            self.ptt.cleanup,
        ]

        try:
            await asyncio.gather(bus_task, *tasks)
        except asyncio.CancelledError:
            logger.info(\"Service cancelled\")
        except Exception as e:
            logger.error(f\"Service error: {e}\", exc_info=True)
        finally:
            await self.stop()

    async def stop(self):
        \"\"\"Stop the service and cleanup\"\"\"
        logger.info(\"Stopping Radio Voice Assistant Service\")

        # Stop event bus
        await self.event_bus.stop()

        # Execute cleanup callbacks
        for cleanup in self.cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(cleanup):
                    await cleanup()
                else:
                    cleanup()
            except Exception as e:
                logger.error(f\"Cleanup error: {e}\", exc_info=True)

        logger.info(\"Service stopped\")

def main():
    \"\"\"Main entry point\"\"\"
    import yaml

    # Load configuration
    with open('/etc/radio/config.yaml') as f:
        config = yaml.safe_load(f)

    # Create service
    service = RadioAssistantService(config)

    # Run service
    try:
        asyncio.run(service.start())
    except KeyboardInterrupt:
        logger.info(\"Interrupted by user\")

if __name__ == '__main__':
    main()
```

---

## 8. Deployment & Service Management

### 8.1 systemd Service

**File**: `/etc/systemd/system/radio-assistant.service`

```ini
[Unit]
Description=Radio Voice Assistant
After=network.target sound.target
Requires=sound.target

[Service]
Type=simple
User=radio
Group=radio
WorkingDirectory=/opt/radio
ExecStart=/usr/local/bin/radio-assistant --config /etc/radio/config.yaml
Restart=on-failure
RestartSec=10
StandardOutput=append:/var/log/radio/assistant.log
StandardError=append:/var/log/radio/assistant.err.log

# Resource limits
MemoryMax=200M
CPUQuota=150%

[Install]
WantedBy=multi-user.target
```

### 8.2 Configuration File

**File**: `/etc/radio/config.yaml`

```yaml
# Audio configuration
audio_device: null  # null = default, or specify device index
sample_rate: 16000
channels: 1

# VAD configuration
vad_aggressiveness: 2  # 0-3 (3 = most aggressive)
min_speech_chunks: 10  # 300ms minimum speech
max_silence_chunks: 20  # 600ms silence ends speech

# Callsign detection
templates_dir: /etc/radio/templates
distance_threshold: 0.22
probability_threshold: 0.5
minimum_matches: 1

# Transcription
whisper_model: base  # tiny, base, small, medium, large

# Session management
session_timeout: 30  # seconds

# PTT configuration
ptt_gpio_pin: 17  # BCM pin number
vox_padding_ms: 200  # Pre/post TX delay

# Logging
log_level: INFO
log_file: /var/log/radio/assistant.log
```

### 8.3 Installation Script

```bash
#!/bin/bash
set -e

echo "Installing Radio Voice Assistant..."

# Create user
sudo useradd -r -s /bin/false radio
sudo usermod -a -G audio,gpio radio

# Create directories
sudo mkdir -p /opt/radio
sudo mkdir -p /etc/radio/templates
sudo mkdir -p /var/log/radio
sudo chown -R radio:radio /opt/radio /etc/radio /var/log/radio

# Install Python dependencies
pip install -r requirements.txt

# Copy files
sudo cp radio_assistant.py /usr/local/bin/radio-assistant
sudo chmod +x /usr/local/bin/radio-assistant
sudo cp config.yaml /etc/radio/config.yaml

# Install systemd service
sudo cp radio-assistant.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable radio-assistant

echo "Installation complete"
echo "Configure /etc/radio/config.yaml and add templates to /etc/radio/templates/"
echo "Start service with: sudo systemctl start radio-assistant"
```

---

## 9. Testing Strategy

### 9.1 Component Testing

```python
import pytest

class TestEventBus:
    @pytest.mark.asyncio
    async def test_emit_and_handle(self):
        bus = AsyncEventBus()
        received = []

        async def handler(data):
            received.append(data)

        bus.on('test.event', handler)
        await bus.emit('test.event', {'value': 42})

        # Process event
        event = await bus.queue.get()
        for h in bus.handlers['test.event']:
            await h(event.data)

        assert received == [{'value': 42}]

class TestStreamingVAD:
    @pytest.mark.asyncio
    async def test_speech_detection(self):
        bus = AsyncEventBus()
        vad = StreamingVAD(bus)

        # Create speech audio (mock)
        speech_audio = np.random.randint(-1000, 1000, size=8000, dtype=np.int16)

        # Feed to VAD
        await vad.on_audio_chunk({
            'audio': speech_audio,
            'sample_rate': 16000,
        })

        # Check event was emitted
        event = await bus.queue.get()
        assert event.type == 'vad.speech.start'
```

### 9.2 Integration Testing

```python
@pytest.mark.asyncio
async def test_full_pipeline():
    \"\"\"Test complete audio → callsign → response pipeline\"\"\"
    config = load_test_config()
    service = RadioAssistantService(config)

    # Start service in background
    service_task = asyncio.create_task(service.start())

    # Inject test audio
    test_audio = load_test_audio('wsjj659_calling.wav')
    await service.audio.audio_queue.put(test_audio.tobytes())

    # Wait for response
    await asyncio.sleep(2)

    # Verify callsign detected
    # Verify PTT activated
    # Verify response generated

    # Cleanup
    service_task.cancel()
    await service.stop()
```

### 9.3 Long-Running Stability Test

```bash
#!/bin/bash
# Run service for 24 hours and monitor

sudo systemctl start radio-assistant

# Monitor memory every minute
for i in {1..1440}; do
    PID=$(pgrep -f radio-assistant)
    MEM=$(ps -p $PID -o rss= | awk '{print $1/1024}')
    echo "$(date) - Memory: ${MEM}MB" >> stability.log
    sleep 60
done

# Check for memory leaks
python analyze_stability.py stability.log
```

---

## 10. Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| **RAM Usage** | <150MB | `ps aux` monitoring |
| **CPU Usage (idle)** | <10% | `top` monitoring |
| **CPU Usage (active)** | <80% | During transcription |
| **Latency (callsign → response start)** | <2 seconds | Event timestamp deltas |
| **Audio dropout rate** | <0.1% | Dropped frames / total |
| **Session timeout accuracy** | ±1 second | Timeout vs. expected |
| **Event queue backpressure** | <1% drops | Queue full events |
| **Uptime** | >7 days | systemd restart counter |

---

## 11. Next Steps

### 11.1 Implementation Plan (2 weeks)

**Week 1: Core Infrastructure**
- Day 1-2: EventBus implementation + tests
- Day 3-4: AudioManager + StreamingVAD
- Day 5: Template generation script + callsign detector stub

**Week 2: Integration**
- Day 6-7: CallsignDetector implementation
- Day 8-9: ConversationManager + PTTController
- Day 10: Integration testing + bug fixes

### 11.2 Open Questions for User

1. **Audio Device**: Which sound card/device for RX audio? (USB sound card, onboard, etc.)
2. **PTT Control**: GPIO pin number for PTT? Or VOX mode?
3. **Callsign Templates**: How many callsigns to support initially? (user will record templates)
4. **TTS Engine**: Piper TTS OK? Or espeak/Festival?
5. **Deployment Target**: Raspberry Pi model? (Pi 3, Pi 4, Pi Zero 2?)

### 11.3 Future Enhancements (Phase 3+)

- **Multi-turn conversations**: Support follow-up questions without repeating callsign
- **Squelch integration**: Coordinate with radio squelch for RX detection
- **CTCSS/DCS filtering**: Decode sub-audible tones for channel selection
- **Advanced intent recognition**: Beyond simple command matching
- **Logging & telemetry**: Event replay for debugging
- **Web dashboard**: Monitor service status, view session history
- **OTA updates**: Remote configuration updates

---

## 12. Conclusion

This architecture proposal provides a **proven, resource-efficient path** to implementing a streaming voice assistant service for radio operation. By adopting patterns from three mature projects (Rhasspy, Home Assistant, Mycroft) while avoiding their complexity overhead, we achieve:

✅ **Single-process async architecture** (vs. 5+ processes in Mycroft/Rhasspy)
✅ **<150MB RAM target** (vs. 320MB in Mycroft)
✅ **No network exposure** (vs. open WebSocket port in Mycroft)
✅ **Template-based wake words** (custom callsigns without neural network training)
✅ **Event-driven coordination** (proven pattern, simplified implementation)
✅ **Offline-first** (field deployment without internet)
✅ **Clear component boundaries** (testable, maintainable)
✅ **systemd integration** (standard Linux service management)

**Estimated Implementation**: 2 weeks (10 working days)
**Resource Footprint**: ~100-150MB RAM, <10% CPU idle, <80% CPU active
**Target Platform**: Raspberry Pi 3/4

---

**Status**: Proposal ready for review
**Next Action**: User approval → begin implementation
**Dependencies**: None (all research complete)
