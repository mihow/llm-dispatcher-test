# Rhasspy Voice Assistant Architecture Research

**Research Date**: 2025-12-28
**Purpose**: Understand Rhasspy's architecture to inform radio voice assistant implementation
**Focus Areas**: Audio streaming, wake word detection, pipeline coordination, service architecture

---

## Executive Summary

Rhasspy is an open-source, fully offline voice assistant platform with two major versions:
- **Rhasspy 2.x**: Uses actor model with MQTT/Hermes protocol (archived October 2025)
- **Rhasspy 3.x**: Uses Wyoming protocol (early developer preview)

Key architectural patterns applicable to radio voice assistant:
1. **Callback-based audio streaming** with actor/async message passing for thread safety
2. **Event-driven pipeline** with clear state transitions and session lifecycle
3. **Modular service architecture** using standardized protocols (MQTT/Hermes or Wyoming)
4. **Template-based wake word detection** with DTW for custom wake words
5. **Supervisord-based process management** for long-running service operation

---

## 1. Audio Streaming Architecture

### 1.1 Callback-Based Approach (Rhasspy 2.x)

**File**: `rhasspy/audio_recorder.py`
**Pattern**: PyAudio stream callback with actor-based message passing

```python
def stream_callback(data, frame_count, time_info, status):
    if data:
        # Send to this actor to avoid threading issues
        self.send(self.myAddress, AudioData(data))
    return (data, pyaudio.paContinue)
```

**Key Design Decisions**:
- **Thread Safety**: Audio callback sends messages to actor rather than processing directly
- **Buffer Management**: Dual-mode operation - streaming (live forwarding) or buffering (accumulation)
- **Buffer Size**: Default 480 frames (30ms chunks) at 16-bit, 16kHz mono
- **State Transitions**: Separate `started` and `recording` states control device lifecycle

**Audio Specifications**:
- Sample rate: 16,000 Hz
- Sample width: 16-bit (2 bytes)
- Channels: 1 (mono)
- Chunk size: 2048 bytes (over MQTT) or 480 frames (in PyAudio callback)

**Repository**: [rhasspy-microphone-pyaudio-hermes](https://github.com/rhasspy/rhasspy-microphone-pyaudio-hermes)

### 1.2 MQTT Audio Streaming (Rhasspy 2.x)

**Protocol**: Hermes protocol over MQTT
**Topic**: `hermes/audioServer/<siteId>/audioFrame`
**Format**: WAV chunks (2048 bytes)

**Optimization for Satellites**:
- **Problem**: Continuous MQTT streaming causes broker congestion
- **Solution**: UDP audio streaming mode
  - Set matching ports: `microphone.<SYSTEM>.udp_audio` and `wake.<SYSTEM>.udp_audio`
  - MQTT streaming disabled until wake word detected
  - Reduces bandwidth consumption on satellites

**Configuration Pattern**:
```yaml
microphone:
  pyaudio:
    udp_audio: 12202  # Same port for both services
wake:
  raven:
    udp_audio: 12202  # Defers MQTT until asr/startListening
```

### 1.3 Wyoming Protocol Audio Streaming (Rhasspy 3.x)

**Protocol**: Wyoming protocol (JSONL + PCM audio)
**Format**: Event-based binary protocol

**Message Structure**:
1. **JSON Header** (UTF-8, newline-terminated)
   - `type`: Event category (required)
   - `data`: Event metadata (optional)
   - `data_length`: Additional JSON bytes (optional)
   - `payload_length`: Binary payload bytes (optional)

2. **Additional Data** (optional UTF-8 JSON)

3. **Payload** (optional, typically PCM audio)

**Audio Events**:
- `audio-start`: Begin audio stream (includes sample rate, width, channels)
- `audio-chunk`: PCM audio data chunks
- `audio-stop`: End audio stream

**Example Flow**:
```
AudioStart (metadata) → AudioChunk (PCM) → AudioChunk (PCM) → ... → AudioStop
```

**Streaming Modes** (Wyoming Satellite):
- **AlwaysStreamingSatellite**: Continuous streaming to server for remote detection
- **VadStreamingSatellite**: Stream only during detected speech (VAD filtering)
- **WakeStreamingSatellite**: Stream only after local wake word detection

**Implementation**:
- Built on `asyncio` for async/await patterns
- Supports TCP, Unix domain sockets, and stdio
- Binary-safe event protocol optimized for streaming

**Repository**: [wyoming](https://github.com/rhasspy/wyoming)

### 1.4 Audio Libraries Used

**Rhasspy 2.x**:
- **PyAudio**: Microphone input with callback streaming
- **python-speech-features**: MFCC extraction for wake word detection
- **rhasspy-silence**: Silence detection (VAD-only, ratio-only, or hybrid)

**Rhasspy 3.x**:
- System commands: `arecord -r 16000 -c 1 -f S16_LE -t raw`
- Wyoming protocol handles audio transport

**Key Takeaway for Radio Assistant**:
- Use PyAudio callbacks for low-latency audio capture
- Implement message passing (actor/async) to avoid threading issues
- Consider UDP or event-based streaming to reduce overhead
- Standard format: 16kHz, 16-bit, mono for voice processing

---

## 2. Wake Word Detection

### 2.1 Raven Wake Word System

**Algorithm**: Dynamic Time Warping (DTW) template matching
**Basis**: Snips Personal Wake Word Detector
**Repository**: [rhasspy-wake-raven](https://github.com/rhasspy/rhasspy-wake-raven)

**Training Process**:
1. Record 3+ WAV templates (16-bit, 16kHz mono)
2. Silence automatically trimmed from edges
3. Convert to MFCC representations
4. Store templates for runtime comparison

**Detection Algorithm**:
1. **Feature Extraction**: Incoming audio → MFCC features
2. **Sliding Window**: Process audio with configurable window shifts (default 0.02s)
3. **Template Matching**: Compare against all templates using DTW
4. **Threshold Check**: Trigger if normalized DTW distance + probability meet thresholds

**Key Parameters**:
```python
distance_threshold: 0.22      # Normalized DTW distance (default)
probability_threshold: 0.5    # Detection confidence (0-1)
minimum_matches: 1            # Templates that must match
dtw_window_size: 5           # Band width for DTW calculation
chunk_size: 1920             # Audio processing chunk size (bytes)
shift_sec: 0.02              # Sliding window shift (seconds)
```

**Libraries**:
- **python-speech-features**: MFCC computation
- **rhasspy-silence**: Silence detection integration
- **LAPACK/ATLAS**: Numerical computations for DTW

**Output**: JSON detection event when thresholds met

### 2.2 False Positive Handling

**Sensitivity Tuning by System**:

**Raven**:
- `probability_threshold < 0.5`: More sensitive (more false positives)
- `probability_threshold > 0.5`: Less sensitive (more false negatives)
- Range: 0.1 to 0.73
- `minimum_matches > 1`: Require multiple template matches (reduces false positives)

**Snowboy**:
- Adjust `sensitivity` parameter
- Adjust `audio_gain` for volume normalization
- Challenge: False positives every 5 minutes with TV/background noise reported
- Supports multiple wake word models with independent sensitivity

**Pocketsphinx**:
- Most flexible but worst performance
- Threshold range: 1e-50 to 1e-5 (smaller = less likely to trigger)
- Keyphrase-based detection

**Porcupine**:
- Excellent out-of-box performance
- Pre-built keyword files available
- Requires 30-day revalidation for custom words

**Mycroft Precise**:
- Fully offline training
- Upload training data, download `.pb` and `.pb.params` files
- Requires upfront model development

**Common Issues**:
- Background noise (TV, music) causes false positives
- Repeated "SSS" sounds can trigger detection
- Trade-off between sensitivity and false positive rate

### 2.3 Integration with Audio Pipeline

**MQTT Messages** (Rhasspy 2.x):
- **Detection**: `hermes/hotword/<wakewordId>/detected`
- **Control**: `hermes/hotword/toggleOff`, `hermes/hotword/toggleOn`
- **Auto-disable**: During voice recording and audio playback (prevents self-triggering)

**Wyoming Events** (Rhasspy 3.x):
- `detect`: Identify spoken wake words
- `detection`: Signal when words recognized
- `voice-started`, `voice-stopped`: Voice activity detection

**Refractory Period**:
- Wake word cannot be detected again for several seconds after detection
- Configurable via `--wake-refractory-seconds` (Wyoming Satellite)
- Prevents rapid re-triggering

**Key Takeaway for Radio Assistant**:
- Template-based DTW is viable for custom wake words
- Requires careful tuning of sensitivity/threshold parameters
- Implement auto-disable during audio playback to prevent self-triggering
- Consider hybrid approach: local wake word + remote processing for better accuracy

---

## 3. Pipeline Architecture

### 3.1 Actor Model (Rhasspy 2.x)

**Pattern**: Stateful actors in separate threads communicating via messages

**Architecture**:
- Each subsystem = stateful actor
- Actors run in separate threads
- Message passing for inter-actor communication
- Central Dialogue Manager coordinates all actors

**Benefits**:
- Clear separation of concerns
- Thread safety through message passing
- Actors handle messages differently based on state
- Independent actor lifecycle management

**Implementation Notes**:
- ActorSystem() provides foundation
- RhasspyCore initializes actor system
- Actors pass messages to each other asynchronously

**Evolution**:
- v2.0: Actor model introduced
- v2.4: FST → language modeling (millions of commands trained in seconds)
- v2.5: Actors → microservices over MQTT

### 3.2 Microservices with MQTT/Hermes (Rhasspy 2.5+)

**Pattern**: Independent services coordinated over MQTT

**Communication Protocol**: Hermes (MQTT-based)
- Message payloads: JSON objects (control) or binary WAV (audio)
- Topic pattern: `hermes/<service>/<siteId>/<action>`

**MQTT Modes**:

**Internal MQTT** (default):
- Automatic `mosquitto` broker on port 12183
- Private network for single-machine deployment
- All services connect to private broker

**External MQTT**:
- Connect to user-provided broker
- Enables distributed server-satellite architecture
- Requires UDP audio streaming configuration for bandwidth optimization

**Service Management**:
- `rhasspy-supervisor` generates supervisord configurations
- Profile directory contains `supervisord.pid`
- SIGHUP signal triggers config reload and process restart
- Docker Compose: `.restart_docker` file triggers restart

**Key Services**:

| Service | Purpose | MQTT Topics |
|---------|---------|-------------|
| Audio Input | Capture microphone | `hermes/audioServer/<siteId>/audioFrame` |
| Wake Word | Hotword detection | `hermes/hotword/<wordId>/detected` |
| Dialogue Manager | Session orchestration | `hermes/dialogueManager/*` |
| ASR | Speech-to-text | `hermes/asr/*` |
| NLU | Intent recognition | `hermes/nlu/*` |
| TTS | Text-to-speech | `hermes/tts/*` |

**Service Discovery**: Each service publishes capabilities via `siteId` property

### 3.3 Domain-Based Architecture (Rhasspy 3.x)

**Pattern**: Domains + Pipelines + Wyoming protocol

**Domains** (functional areas):
- `mic`: Audio input
- `wake`: Wake word detection
- `vad`: Voice activity detection
- `asr`: Speech-to-text
- `intent`: Intent recognition
- `handle`: Response handling
- `tts`: Text-to-speech
- `snd`: Audio output

**Pipelines** (complete workflows):
```
detect → transcribe → recognize → handle → speak
```

**Communication**: Wyoming protocol
- External programs run as servers (resource efficiency)
- HTTP API: `http://localhost:13331/<endpoint>`
- WebSocket API: `ws://localhost:13331/<endpoint>` (streaming)
- Configuration: `configuration.yaml`

**Adapters**: Small scripts in `bin/` to bridge existing programs to Wyoming

### 3.4 Dialogue Manager Implementation

**File**: `rhasspydialogue_hermes/__init__.py`
**Class**: `DialogueHermesMqtt` (extends `HermesClient`)
**Repository**: [rhasspy-dialogue-hermes](https://github.com/rhasspy/rhasspy-dialogue-hermes)

**State Management**:
```python
# Track active sessions
session_by_site: dict[str, SessionInfo]  # One session per site
session_queue_by_site: dict[str, list]   # Queued session requests
all_sessions: dict[str, SessionInfo]     # Lookup by session_id
message_events: dict                     # Coordinate async MQTT handlers
```

**SessionInfo Structure**:
```python
@dataclass
class SessionInfo:
    session_id: str
    site_id: str
    custom_data: str
    intent_filter: list[str]
    audio_capture: bool
    language: str
    wakeword_id: str
    detected: dict  # Wake word detection details
```

**Session Lifecycle**:

1. **Start Session**:
   - `handle_start()`: Create new session with error handling
   - `start_session()`: Process notification vs. action sessions
   - Queue management: One active session per site, others queued

2. **Continue Session**:
   - `handle_continue()`: Advance dialogue step
   - Update intent filters and language
   - Restart ASR for next turn

3. **End Session**:
   - `handle_end()`: Terminate with custom text
   - `end_session()`: Cleanup and process queue
   - Handle timeouts (30s default, configurable)

**MQTT Message Routing** (`on_message()` dispatcher):
```python
# Session control
DialogueStartSession → handle_start()
DialogueContinueSession → handle_continue()
DialogueEndSession → handle_end()

# Speech processing
AsrTextCaptured → route to NLU (confidence filtering)
NluIntent/NluIntentNotRecognized → intent results

# Wake word
HotwordDetected → start session (group-based locking)

# Audio events
TtsSayFinished → audio event tracking
AudioPlayFinished → audio event tracking
```

**Service Coordination**:

**TTS Integration**:
```python
def say(text):
    disable_asr()
    disable_hotword()
    yield TtsSay(text)
    wait_for_completion_event()
    # Timeout estimate: chars * rate
```

**Audio Playback**:
```python
def maybe_play_sound(sound_type):
    disable_asr()
    disable_hotword()
    play_sound_file()  # Randomized from directory
    wait_for_AudioPlayFinished()
```

**ASR/Hotword Toggle**:
- `AsrToggleOff/On` messages manage ASR state
- `HotwordToggleOff/On` messages manage wake word state
- Automatic disable during TTS and playback

**Advanced Features**:
- **Wakeword Group Locking**: Prevent concurrent sessions in same group (asyncio locks)
- **Confidence Filtering**: Auto-reject ASR below threshold (skip NLU)
- **Session Timeout**: Async task monitors inactive sessions (30s default)
- **Audio Format Conversion**: WAV, soundfile, audioread fallbacks

**Async Coordination**:
- Uses `asyncio` event loops
- `asyncio.run_coroutine_threadsafe()` bridges sync/async contexts
- `await asyncio.gather()` for concurrent operations
- Message events coordinate between MQTT handlers

**Key Takeaway for Radio Assistant**:
- Central coordinator (dialogue manager) orchestrates all services
- Clear session lifecycle with state management
- Message-based coordination prevents tight coupling
- Async/await pattern enables efficient I/O handling
- Auto-disable services during audio playback critical for preventing self-triggering

---

## 4. Service Architecture (Long-Running Operation)

### 4.1 Process Management (Rhasspy 2.x)

**Tool**: `supervisord` for persistent process management

**Configuration Generation**:
- `rhasspy-supervisor` tool converts profile → supervisord config
- Can also generate docker-compose.yml

**Lifecycle Management**:
- PID file: `supervisord.pid` in profile directory
- Restart mechanism: SIGHUP signal → reload config → stop/start processes
- Docker: Monitor `.restart_docker` file for restart triggers

**Logging**:
- **Issue**: `stdout_logfile = /dev/stdout` breaks systemd service
- **Solution**: Point to actual log files instead

**Auto-Restart**:
- Docker: `--restart unless-stopped` flag
- Supervisord: Built-in restart policies

### 4.2 Error Recovery

**Microservices Advantages**:
- Independent service failures don't collapse system
- Component isolation enables graceful degradation
- Individual service restart without full system restart

**Error Handling Patterns**:
- MQTT retain messages for state recovery
- Timeout handling in dialogue manager (30s default)
- Error callbacks in Wyoming Satellite (`--error-command`)

**Wyoming Satellite Event Hooks**:
```bash
--startup-command           # Satellite initialization
--connected-command         # Connection established
--disconnected-command      # Connection lost
--detection-command         # Wake word detected
--transcript-command        # STT result
--error-command            # Server error
--timer-finished-command   # Timer completion
```

**Timeout Management**:
- Session timeout: 30s (configurable)
- TTS timeout: Estimated from character count
- Refractory period: Prevents rapid re-triggering

### 4.3 Resource Management

**Audio Device Lifecycle**:
- `keep_device_open` flag in PyAudio recorder
- State transitions: `started` → `recording` → cleanup
- Stream cleanup in callback shutdown

**MQTT Connection Management**:
- Persistent connections with reconnection
- QoS settings for message delivery guarantees
- TLS support for secure communication

**Memory Management**:
- Audio buffers: Named storage with cleanup
- Receiver tracking: Active subscriber management
- Session cleanup: Remove completed sessions

### 4.4 Deployment Patterns

**Single Machine**:
- Internal MQTT broker (port 12183)
- All services on localhost
- Supervisord or systemd management

**Server-Satellite**:
- External MQTT broker
- UDP audio streaming for bandwidth optimization
- Remote wake word detection vs. local detection

**Docker**:
- docker-compose orchestration
- Volume mounts for profile persistence
- `.restart_docker` file for restart coordination

**Debian Package**:
- Systemd service integration
- Standard Linux service management
- Log files instead of stdout

**Key Takeaway for Radio Assistant**:
- Use process manager (supervisord/systemd) for reliability
- Implement lifecycle hooks for connection/error events
- Consider modular architecture for graceful degradation
- Plan for restart without full system reboot

---

## 5. Code Patterns for Radio Voice Assistant

### 5.1 Audio Streaming Pattern

```python
import pyaudio
import asyncio
from dataclasses import dataclass

@dataclass
class AudioData:
    data: bytes
    timestamp: float

class AudioStreamer:
    def __init__(self, sample_rate=16000, chunk_size=480):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = asyncio.Queue()

    def stream_callback(self, data, frame_count, time_info, status):
        """PyAudio callback - avoid processing here, just queue"""
        if data:
            # Use asyncio queue for thread-safe message passing
            asyncio.create_task(
                self.audio_queue.put(AudioData(data, time.time()))
            )
        return (data, pyaudio.paContinue)

    async def process_audio(self):
        """Main processing loop - safe for async operations"""
        while True:
            audio_data = await self.audio_queue.get()
            # Process audio here (wake word, VAD, etc.)
            await self.handle_audio_chunk(audio_data)
```

### 5.2 Wake Word Detection Pattern

```python
import numpy as np
from python_speech_features import mfcc
from dtw import dtw

class TemplateWakeWordDetector:
    def __init__(self, templates, threshold=0.22, probability=0.5):
        self.templates = templates  # List of MFCC template arrays
        self.threshold = threshold
        self.probability = probability
        self.refractory_until = 0

    async def detect(self, audio_chunk):
        """Check if audio chunk contains wake word"""
        # Skip if in refractory period
        if time.time() < self.refractory_until:
            return None

        # Extract MFCC features
        features = mfcc(audio_chunk, samplerate=16000)

        # Compare against all templates
        matches = 0
        for template in self.templates:
            distance, _ = dtw(features, template)
            normalized_dist = distance / len(features)

            if normalized_dist < self.threshold:
                matches += 1

        # Check if enough templates matched
        if matches >= self.minimum_matches:
            confidence = matches / len(self.templates)
            if confidence >= self.probability:
                # Set refractory period (e.g., 3 seconds)
                self.refractory_until = time.time() + 3.0
                return {"confidence": confidence, "timestamp": time.time()}

        return None
```

### 5.3 Event-Based Pipeline Pattern

```python
from enum import Enum
from dataclasses import dataclass

class PipelineState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"

@dataclass
class PipelineEvent:
    type: str
    data: dict
    timestamp: float

class VoicePipeline:
    def __init__(self):
        self.state = PipelineState.IDLE
        self.event_handlers = {}

    def register_handler(self, event_type, handler):
        """Register async handler for event type"""
        self.event_handlers[event_type] = handler

    async def handle_event(self, event: PipelineEvent):
        """Process event based on current state"""
        handler = self.event_handlers.get(event.type)
        if handler:
            await handler(event, self.state)

    async def on_wake_word(self, event, state):
        """Wake word detected - start listening"""
        if state == PipelineState.IDLE:
            self.state = PipelineState.LISTENING
            # Disable wake word detection
            await self.toggle_wake_word(False)
            # Start ASR
            await self.start_asr()

    async def on_speech_end(self, event, state):
        """Speech ended - process transcription"""
        if state == PipelineState.LISTENING:
            self.state = PipelineState.PROCESSING
            transcript = event.data["text"]
            # Process intent
            intent = await self.recognize_intent(transcript)
            await self.handle_intent(intent)

    async def on_response_complete(self, event, state):
        """Response finished - return to idle"""
        if state == PipelineState.RESPONDING:
            self.state = PipelineState.IDLE
            # Re-enable wake word detection
            await self.toggle_wake_word(True)
```

### 5.4 Service Coordination Pattern

```python
class DialogueManager:
    def __init__(self):
        self.active_sessions = {}
        self.session_queue = {}

    async def start_session(self, site_id, session_type="action"):
        """Start new dialogue session"""
        # Check if site already has active session
        if site_id in self.active_sessions:
            # Queue this request
            self.session_queue.setdefault(site_id, []).append(session_type)
            return None

        # Create new session
        session = {
            "id": uuid.uuid4(),
            "site_id": site_id,
            "started": time.time(),
            "type": session_type,
        }
        self.active_sessions[site_id] = session

        # Disable services during session
        await self.toggle_wake_word(site_id, False)

        return session

    async def end_session(self, site_id):
        """End dialogue session and process queue"""
        # Remove active session
        session = self.active_sessions.pop(site_id, None)

        if not session:
            return

        # Re-enable wake word
        await self.toggle_wake_word(site_id, True)

        # Process queued sessions
        if site_id in self.session_queue and self.session_queue[site_id]:
            next_type = self.session_queue[site_id].pop(0)
            await self.start_session(site_id, next_type)

    async def monitor_timeouts(self):
        """Background task to check for timed-out sessions"""
        while True:
            await asyncio.sleep(5)  # Check every 5 seconds

            now = time.time()
            for site_id, session in list(self.active_sessions.items()):
                if now - session["started"] > 30:  # 30 second timeout
                    logger.warning(f"Session timeout for {site_id}")
                    await self.end_session(site_id)
```

### 5.5 Supervisord Configuration Pattern

```ini
[program:radio_voice_assistant]
command=/usr/local/bin/radio-assistant --config /etc/radio/config.yaml
directory=/opt/radio
autostart=true
autorestart=true
startretries=3
stderr_logfile=/var/log/radio/assistant.err.log
stdout_logfile=/var/log/radio/assistant.out.log
user=radio
environment=PYTHONUNBUFFERED="1"

[program:radio_wake_word]
command=/usr/local/bin/radio-wake-word --templates /etc/radio/wake_templates/
directory=/opt/radio
autostart=true
autorestart=true
startretries=3
stderr_logfile=/var/log/radio/wake_word.err.log
stdout_logfile=/var/log/radio/wake_word.out.log
user=radio

[program:radio_audio_input]
command=/usr/local/bin/radio-audio-input --device 0 --rate 16000
directory=/opt/radio
autostart=true
autorestart=true
startretries=3
stderr_logfile=/var/log/radio/audio.err.log
stdout_logfile=/var/log/radio/audio.out.log
user=radio
```

---

## 6. Key Repositories and File References

### 6.1 Rhasspy 2.x (MQTT/Hermes)

**Main Repository**: [rhasspy/rhasspy](https://github.com/rhasspy/rhasspy)
- `rhasspy/audio_recorder.py`: PyAudio callback implementation
- `rhasspy/app.py`: Main application entry point with asyncio
- `docs/services.md`: Service architecture documentation
- `docs/wake-word.md`: Wake word system documentation
- `docs/audio-input.md`: Audio input configuration
- `architecture.sh`: Architecture detection script

**Microphone Service**: [rhasspy-microphone-pyaudio-hermes](https://github.com/rhasspy/rhasspy-microphone-pyaudio-hermes)
- PyAudio → MQTT WAV chunks
- 2048 byte chunks, 16-bit, 16kHz mono
- UDP audio streaming support

**Wake Word (Raven)**: [rhasspy-wake-raven](https://github.com/rhasspy/rhasspy-wake-raven)
- `rhasspywake_raven/`: Core package
- `bin/rhasspy-wake-raven`: CLI entry point
- `etc/okay-rhasspy/`: Example templates
- DTW-based template matching

**Dialogue Manager**: [rhasspy-dialogue-hermes](https://github.com/rhasspy/rhasspy-dialogue-hermes)
- `rhasspydialogue_hermes/__init__.py`: Main DialogueHermesMqtt class
- Session lifecycle management
- MQTT message routing
- Service coordination (ASR, NLU, TTS)

**Supervisor**: [rhasspy-supervisor](https://github.com/rhasspy/rhasspy-supervisor)
- Profile → supervisord config generation
- Docker compose generation

### 6.2 Rhasspy 3.x (Wyoming)

**Main Repository**: [rhasspy/rhasspy3](https://github.com/rhasspy/rhasspy3)
- `docs/wyoming.md`: Wyoming protocol documentation
- `bin/`: Adapter scripts for Wyoming integration
- `configuration.yaml`: Service configuration

**Wyoming Protocol**: [rhasspy/wyoming](https://github.com/rhasspy/wyoming)
- JSONL + PCM audio event protocol
- Async/await implementation
- TCP, Unix sockets, stdio support

**Wyoming Satellite**: [rhasspy/wyoming-satellite](https://github.com/rhasspy/wyoming-satellite)
- `wyoming_satellite/`: Main package
- `wyoming_satellite/example_event_client.py`: Event service example
- Three satellite types: Always/VAD/Wake streaming
- Event hook system for lifecycle management

### 6.3 Related Projects

**Wyoming Piper (TTS)**: [rhasspy/wyoming-piper](https://github.com/rhasspy/wyoming-piper)
- `wyoming_piper/__main__.py`: Wyoming TTS server implementation

**Home Assistant Satellite**: [synesthesiam/homeassistant-satellite](https://github.com/synesthesiam/homeassistant-satellite)
- Streaming audio satellite reference implementation

---

## 7. Adaptations for Radio Voice Assistant

### 7.1 Recommended Architecture

**Audio Input Layer**:
- PyAudio callback-based streaming (16kHz, 16-bit mono)
- Async queue for thread-safe message passing (avoid processing in callback)
- Chunk size: 480 frames (30ms) for low latency

**Wake Word Detection**:
- Template-based DTW (Raven approach) for custom wake words
- MFCC feature extraction using python-speech-features
- Tunable threshold (0.22) and probability (0.5) parameters
- 3-second refractory period to prevent re-triggering
- Auto-disable during audio playback (critical!)

**Pipeline Coordination**:
- Event-based state machine (IDLE → LISTENING → PROCESSING → RESPONDING)
- Dialogue manager coordinates service lifecycle
- Session timeout monitoring (30s default)
- Queue management for concurrent requests

**Service Architecture**:
- Modular services with clear boundaries
- Message-based communication (asyncio queues or MQTT)
- Supervisord for process management
- Event hooks for lifecycle management

**Error Recovery**:
- Individual service restart without system restart
- Connection/disconnection event handlers
- Timeout handling at each pipeline stage
- Graceful degradation if non-critical services fail

### 7.2 Key Learnings

1. **Thread Safety**: Never process audio directly in PyAudio callback - use message passing
2. **Auto-Disable**: Disable wake word during playback to prevent self-triggering
3. **Refractory Period**: Prevent rapid re-detection with cooldown period
4. **State Management**: Clear session lifecycle with timeout monitoring
5. **Async Coordination**: Use asyncio for I/O-bound operations (MQTT, HTTP, audio)
6. **Graceful Degradation**: Modular services allow partial system operation
7. **Sensitivity Tuning**: Provide knobs for threshold/probability adjustment
8. **Resource Cleanup**: Proper device closure and buffer management

### 7.3 Avoiding Common Pitfalls

1. **PyAudio Threading**: Don't do heavy processing in callback thread
2. **MQTT Congestion**: Use UDP or VAD to reduce continuous streaming
3. **Self-Triggering**: Auto-disable wake word during TTS/playback
4. **Session Leaks**: Implement timeout monitoring and cleanup
5. **Logging Issues**: Use file logs, not stdout for systemd services
6. **Restart Complexity**: Plan restart mechanism early (SIGHUP, file-based, etc.)
7. **False Positives**: Tune sensitivity parameters, consider multiple template matches

### 7.4 Technology Stack Recommendations

**Core Audio**:
- PyAudio (callback streaming)
- python-speech-features (MFCC)
- numpy (audio processing)

**Wake Word**:
- DTW algorithm (template matching)
- Or: openWakeWord, Porcupine for pre-trained models

**Communication**:
- Option 1: asyncio queues (single process)
- Option 2: MQTT + Hermes protocol (multi-process)
- Option 3: Wyoming protocol (future-proof, Home Assistant compatible)

**Service Management**:
- supervisord (Linux)
- systemd (production deployments)
- Docker compose (containerized)

**Processing**:
- asyncio event loops
- async/await for I/O operations
- Message passing for thread safety

---

## 8. Sources

### Documentation
- [Rhasspy Documentation](https://rhasspy.readthedocs.io/)
- [Rhasspy Services](https://rhasspy.readthedocs.io/en/latest/services/)
- [Wake Word Detection](https://rhasspy.readthedocs.io/en/latest/wake-word/)
- [Audio Input](https://rhasspy.readthedocs.io/en/latest/audio-input/)
- [About Rhasspy](https://rhasspy.readthedocs.io/en/latest/about/)
- [Wyoming Protocol Documentation](https://github.com/rhasspy/rhasspy3/blob/master/docs/wyoming.md)
- [Wyoming Protocol Spec](https://techjunction.co/encyclopedia/wyoming-protocol/)

### GitHub Repositories
- [rhasspy/rhasspy](https://github.com/rhasspy/rhasspy) - Main Rhasspy 2.x (archived)
- [rhasspy/rhasspy3](https://github.com/rhasspy/rhasspy3) - Rhasspy 3.x (dev preview)
- [rhasspy/wyoming](https://github.com/rhasspy/wyoming) - Wyoming protocol
- [rhasspy/wyoming-satellite](https://github.com/rhasspy/wyoming-satellite) - Satellite implementation
- [rhasspy/rhasspy-microphone-pyaudio-hermes](https://github.com/rhasspy/rhasspy-microphone-pyaudio-hermes)
- [rhasspy/rhasspy-wake-raven](https://github.com/rhasspy/rhasspy-wake-raven)
- [rhasspy/rhasspy-dialogue-hermes](https://github.com/rhasspy/rhasspy-dialogue-hermes)
- [rhasspy/rhasspy-supervisor](https://github.com/rhasspy/rhasspy-supervisor)
- [rhasspy/wyoming-piper](https://github.com/rhasspy/wyoming-piper)

### Source Code Files
- [audio_recorder.py](https://github.com/synesthesiam/rhasspy/blob/master/rhasspy/audio_recorder.py)
- [app.py](https://github.com/synesthesiam/rhasspy/blob/master/app.py)
- [dialogue manager __init__.py](https://github.com/rhasspy/rhasspy-dialogue-hermes/blob/master/rhasspydialogue_hermes/__init__.py)
- [architecture.sh](https://github.com/rhasspy/rhasspy/blob/master/architecture.sh)
- [wake-word.md](https://github.com/rhasspy/rhasspy/blob/master/docs/wake-word.md)
- [services.md](https://github.com/rhasspy/rhasspy/blob/master/docs/services.md)

### Community Discussions
- [Distributed Architecture](https://community.rhasspy.org/t/distributed-architecture/1222)
- [Snowboy False Positives](https://community.rhasspy.org/t/snowboy-custom-wakeword-false-positive/156)
- [Wake Word Detection Rate](https://community.rhasspy.org/t/wake-word-detection-rate/461)
- [Best Wake Word Setup](https://community.rhasspy.org/t/best-wake-word-setup/2670)
- [Dialogue Manager Documentation](https://community.rhasspy.org/t/dialogue-manager-documentation/3658)

### Integration Guides
- [Rhasspy - Home Assistant](https://www.home-assistant.io/integrations/rhasspy/)
- [wyoming PyPI](https://pypi.org/project/wyoming/)

---

**Last Updated**: 2025-12-28
**Research Scope**: Architecture patterns, code examples, and best practices from Rhasspy v2.x and v3.x
**Next Steps**: Apply these patterns to radio voice assistant implementation in Phase 2
