# Voice Assistant Architecture Comparison

**Created**: 2025-12-28
**Purpose**: Compare Rhasspy, Home Assistant, and Mycroft architectures to inform radio voice assistant design
**Research Phase**: Phase 2A - Validation & Research

---

## Executive Summary

Three leading open-source voice assistants were analyzed to extract architectural patterns for building a radio voice assistant. Each project offers distinct approaches to solving common voice processing challenges:

- **Rhasspy**: Offline-first, modular microservices with Wyoming/MQTT protocols
- **Home Assistant**: Async-first pipeline with satellite entity pattern
- **Mycroft**: Message bus coordination with skills framework

**Key Recommendation**: Adopt a **hybrid approach** combining:
- **Rhasspy's** template-based wake word detection and audio streaming patterns
- **Home Assistant's** async/await pipeline architecture and resource cleanup patterns
- **Mycroft's** event-driven message bus concept (simplified to in-process async queues)

Avoid the complexity of multi-process architectures - implement as a **single-process async service** optimized for Raspberry Pi embedded deployment.

---

## 1. Architecture Comparison Matrix

| Aspect | Rhasspy 2.x/3.x | Home Assistant | Mycroft AI | **Radio Assistant** |
|--------|-----------------|----------------|------------|---------------------|
| **Process Model** | Multi-process (MQTT) or Wyoming servers | Single async event loop | Multi-process (messagebus + services) | **Single process, async coroutines** |
| **Communication** | MQTT/Hermes or Wyoming protocol | Wyoming protocol integration | WebSocket messagebus (port 8181) | **In-process async queues** |
| **Pipeline Stages** | mic→wake→asr→nlu→tts→snd | wake→STT→intent→TTS | wake→STT→intent→skills→TTS | **Radio RX→VAD→callsign→STT→respond→PTT** |
| **State Machine** | Session-based dialogue manager | Satellite entity states | Session + conversational context | **Event-driven pipeline states** |
| **Wake Word** | Raven (DTW templates), Porcupine, etc. | Wyoming services, VAD | Precise, PocketSphinx | **Template DTW (custom callsign)** |
| **Audio Streaming** | PyAudio callback→MQTT/UDP | AsyncGenerator streaming | PyAudio callback→messagebus | **PyAudio callback→async queue** |
| **Thread Safety** | Actor model message passing | Asyncio event loop | Actor/messagebus | **Asyncio queues from callback** |
| **Service Management** | supervisord or Docker Compose | Core integration lifecycle | start-mycroft.sh script | **systemd or supervisord** |
| **Error Recovery** | Individual service restart | async_on_unload cleanup | Service independence | **Component isolation + restart** |
| **Deployment** | Raspberry Pi friendly | Runs on Pi 4 | Resource heavy (320MB+ RAM) | **Optimized for Pi (<150MB target)** |
| **Security** | Local only (MQTT can be open) | Local or network via auth | **No security** (port 8181 open) | **No network exposure** |
| **Offline Operation** | Fully offline | Offline capable | Cloud optional | **Fully offline required** |

---

## 2. Audio Streaming Patterns

### 2.1 Rhasspy: Callback + Message Passing

**Implementation** (`rhasspy/audio_recorder.py`):
```python
def stream_callback(data, frame_count, time_info, status):
    # Send to actor to avoid threading issues
    self.send(self.myAddress, AudioData(data))
    return (data, pyaudio.paContinue)
```

**Characteristics**:
- PyAudio callback sends to actor/async queue (never processes directly)
- 16kHz, 16-bit mono, 480 frames (30ms chunks)
- MQTT: 2048-byte WAV chunks over Hermes protocol
- UDP streaming mode for satellites (reduces MQTT congestion)
- Wyoming protocol: JSONL headers + PCM binary payload

**Strengths**:
- Proven thread-safe pattern
- Low latency (30ms chunks)
- Multiple protocol options

**Limitations**:
- MQTT overhead for single-machine deployments
- Complex multi-protocol support

### 2.2 Home Assistant: AsyncGenerator Streaming

**Implementation** (Wyoming TTS):
```python
async def async_get_tts_audio(...) -> AsyncGenerator[bytes]:
    wav_writer = wave.open(wav_io, "wb")
    # ... configure WAV ...

    async for chunk in audio_stream:
        wav_writer.writeframes(chunk)
        yield chunk  # Stream immediately, don't buffer
```

**Characteristics**:
- `AsyncGenerator[bytes]` for streaming audio
- Non-blocking async iteration
- Immediate yield vs. buffering in BytesIO
- Wyoming protocol client with context managers
- VAD recommended client-side to reduce streaming

**Strengths**:
- Natural async/await integration
- Memory-efficient streaming (no full buffering)
- Clean async context managers for resource cleanup

**Limitations**:
- Requires async ecosystem
- Wyoming protocol adds abstraction layer

### 2.3 Mycroft: Callback + MessageBus

**Implementation** (`mycroft/client/speech/mic.py`):
```python
def _record_phrase():
    while True:
        chunk = stream.read(CHUNK_SIZE)
        energy = audioop.rms(chunk)

        if noise_tracker.is_loud(energy):
            speech_buffer.append(chunk)

        if silence_counter > MIN_SILENT_CHUNKS:
            return np.concatenate(speech_buffer)
```

**Characteristics**:
- PyAudio + MutableMicrophone wrapper
- Cyclic buffer for wake word context
- Energy-based VAD with dynamic thresholds
- WebSocket messagebus for coordination

**Strengths**:
- Simple energy-based VAD
- Rolling buffer for wake word detection
- Dynamic threshold adaptation

**Limitations**:
- WebSocket overhead for local processing
- Security vulnerability (port 8181 open, no auth)

### 2.4 Recommendation for Radio Assistant

**Pattern**: Rhasspy callback approach + Home Assistant async patterns

```python
class RadioAudioStreamer:
    def __init__(self):
        self.audio_queue = asyncio.Queue(maxsize=50)

    def stream_callback(self, data, frame_count, time_info, status):
        """PyAudio callback - queue only, don't process"""
        try:
            self.audio_queue.put_nowait(AudioData(data, time.time()))
        except asyncio.QueueFull:
            # Drop frame if queue full (backpressure)
            pass
        return (data, pyaudio.paContinue)

    async def process_stream(self):
        """Async processing loop - safe for I/O operations"""
        while True:
            audio_data = await self.audio_queue.get()
            # Process: VAD, wake word, transcription
            await self.handle_audio_chunk(audio_data)
```

**Key Benefits**:
- Thread-safe callback → async queue pattern (Rhasspy)
- Natural async/await processing (Home Assistant)
- No external protocol overhead (optimized for embedded)
- Backpressure handling via queue limits

---

## 3. Wake Word / Trigger Detection

### 3.1 Rhasspy: Template-Based DTW

**Algorithm**: Raven wake word (Dynamic Time Warping)

**Process**:
1. Record 3+ audio templates of wake word (16kHz mono WAV)
2. Convert to MFCC features
3. Sliding window comparison: incoming audio MFCC vs. templates
4. DTW distance calculation with thresholds

**Configuration**:
```python
distance_threshold: 0.22      # Normalized DTW distance
probability_threshold: 0.5    # Detection confidence
minimum_matches: 1            # Templates required to match
refractory_period: 3.0        # Seconds cooldown
```

**Strengths**:
- Custom wake words without training neural networks
- Works offline with minimal resources
- Tunable sensitivity

**Limitations**:
- False positives in noisy environments
- Requires careful threshold tuning
- Less accurate than neural models (Porcupine, Precise)

### 3.2 Home Assistant: Client-Side VAD + Remote Detection

**Pattern**: VAD filtering before wake word detection

**Approach**:
- **Client VAD**: WebRTC VAD or Pyannote (detects speech presence)
- **Wake word**: Wyoming service (Porcupine, openWakeWord, Rhasspy)
- **Optimization**: Only stream when speech detected (10x bandwidth reduction)

**VAD Parameters**:
- Sensitivity: 0-3 (3 = most sensitive)
- Silence threshold: seconds before stop
- Frame duration: 10/20/30ms

**Strengths**:
- Reduces unnecessary audio streaming
- Privacy-respecting (VAD doesn't transcribe)
- Flexible service architecture

**Limitations**:
- Requires Wyoming service setup
- VAD adds complexity

### 3.3 Mycroft: Multiple Engine Support

**Engines**: Precise (neural), PocketSphinx (phoneme), Wake word services

**Pattern**: Continuous listening with cyclic buffer
- Rolling window maintains audio context
- Periodic wake word checks (every 0.2s)
- Energy threshold + wake word confidence

**Strengths**:
- Multiple engine options
- Proven in production

**Limitations**:
- Resource intensive (neural models)
- Message bus overhead

### 3.4 Recommendation for Radio Assistant

**Pattern**: Rhasspy's template DTW + Home Assistant's VAD filtering

**Rationale**:
- **Custom wake words**: Callsigns are unique identifiers (WSJJ659, KE4GZK, etc.)
- **Template approach**: No need for neural network training per callsign
- **VAD filtering**: Radio silence vs. transmission detection (squelch integration)
- **Offline**: No cloud dependencies for field deployment

**Implementation**:
```python
class CallsignDetector:
    def __init__(self, templates_dir):
        self.templates = self.load_mfcc_templates(templates_dir)
        self.threshold = 0.22
        self.refractory_until = 0

    async def detect(self, audio_chunk):
        if time.time() < self.refractory_until:
            return None

        features = mfcc(audio_chunk, samplerate=16000)

        for callsign, template in self.templates.items():
            distance = dtw(features, template)
            if distance / len(features) < self.threshold:
                self.refractory_until = time.time() + 3.0
                return {"callsign": callsign, "confidence": 1 - distance}

        return None
```

**Key Features**:
- Per-callsign templates (not global wake word)
- Refractory period prevents re-triggering
- Returns detected callsign (not just "detected" flag)
- Tunable threshold for environment adaptation

---

## 4. Pipeline Coordination & State Management

### 4.1 Rhasspy: Dialogue Manager + Session Lifecycle

**State Management**:
```python
session_by_site: dict[str, SessionInfo]      # Active sessions
session_queue_by_site: dict[str, list]       # Queued requests
all_sessions: dict[str, SessionInfo]         # Full lookup
```

**Session Lifecycle**:
1. **Start**: Wake word detected → create session → disable wake word
2. **Continue**: Multi-turn conversation → update intent filters
3. **End**: Response complete → cleanup → re-enable wake word → process queue

**Timeout Handling**:
- Default: 30 seconds inactivity
- Async background task monitors sessions
- Auto-end + cleanup on timeout

**Service Coordination**:
- Auto-disable ASR during TTS playback (prevent self-triggering)
- Audio ducking for simultaneous playback
- Wakeword group locking (prevent concurrent sessions in same group)

**Strengths**:
- Clear session boundaries
- Queue management for concurrent requests
- Proven timeout/cleanup patterns

**Limitations**:
- MQTT message overhead
- Complex multi-service coordination

### 4.2 Home Assistant: Satellite Entity State Machine

**States**:
1. `listening_wake_word`: Awaiting activation
2. `listening_command`: Streaming voice input
3. `processing`: Audio stopped, awaiting results
4. `responding`: Delivering TTS response

**Lifecycle Hooks**:
- `async_added_to_hass()`: Initialize, restore state, subscribe to events
- `async_will_remove_from_hass()`: Cleanup, unsubscribe, disconnect

**Cleanup Pattern** (`async_on_unload`):
```python
async def async_setup_entry(hass, entry):
    # Setup work...
    entry.async_on_unload(lambda: cleanup_resources())
    entry.async_on_unload(entry.add_update_listener(update_listener))
```

**Benefits**:
- Automatic cleanup on errors or unload
- Clear state transitions
- Resource tracking in `runtime_data`

**Strengths**:
- Async-first design
- Explicit cleanup registration
- Entity lifecycle well-defined

**Limitations**:
- Home Assistant framework dependency
- Satellite requires Wyoming protocol

### 4.3 Mycroft: Message Bus + Conversational Context

**Message Bus Pattern**:
```python
Message(
    type='recognizer_loop:wakeword',
    data={'utterance': 'hey mycroft'},
    context={'session_id': '123'}
)
```

**Context System**:
- `set_context('keyword', 'value')`: Add context
- Timeout: 5 minutes default
- Cross-skill sharing
- `converse()` method: Multi-turn conversations within active skill

**Service Coordination**:
- `HotwordToggleOff/On`: Control wake word service
- `AsrToggleOff/On`: Control speech recognition
- `speak` → auto-disable ASR/hotword → wait for `TtsSayFinished`

**Strengths**:
- Simple message-based coordination
- Conversational context for multi-turn
- Event-driven architecture

**Limitations**:
- WebSocket security (port 8181, no auth)
- Message bus overhead
- Resource intensive (5+ Python processes, ~320MB RAM)

### 4.4 Recommendation for Radio Assistant

**Pattern**: Rhasspy's dialogue manager + Home Assistant's async lifecycle + simplified Mycroft-style events

**Architecture**:
```python
class EventBus:
    """In-process async event bus (no WebSocket)"""
    def __init__(self):
        self.handlers: dict[str, list[Callable]] = defaultdict(list)
        self.queue = asyncio.Queue(maxsize=100)

    def on(self, event_type: str, handler):
        self.handlers[event_type].append(handler)

    async def emit(self, event_type: str, data: dict | None = None):
        event = Event(type=event_type, data=data or {})
        await self.queue.put(event)

    async def dispatch_loop(self):
        while True:
            event = await self.queue.get()
            for handler in self.handlers[event.type]:
                try:
                    await handler(event.data)
                except Exception as e:
                    logger.error(f"Handler error: {e}")

class RadioPipeline:
    """Radio voice assistant pipeline"""
    def __init__(self):
        self.state = PipelineState.IDLE
        self.bus = EventBus()
        self.session_timeout = 30  # seconds

        # Register handlers
        self.bus.on('radio.callsign.detected', self.on_callsign)
        self.bus.on('vad.speech.complete', self.on_speech_complete)
        self.bus.on('ptt.complete', self.on_response_complete)

    async def on_callsign(self, data):
        if self.state == PipelineState.IDLE:
            self.state = PipelineState.LISTENING
            self.session_start = time.time()
            await self.bus.emit('wake.disable')
            await self.bus.emit('vad.start')

    async def on_speech_complete(self, data):
        if time.time() - self.session_start > self.session_timeout:
            await self.end_session()
            return

        self.state = PipelineState.PROCESSING
        # Transcribe, detect intent, generate response...

    async def end_session(self):
        self.state = PipelineState.IDLE
        await self.bus.emit('wake.enable')
```

**Key Design Decisions**:
- **In-process event bus** (not WebSocket): Lower latency, no security issues
- **Session timeout** (Rhasspy pattern): Auto-cleanup inactive sessions
- **State machine** (Home Assistant pattern): Clear state transitions
- **Auto-disable wake word** (all three): Prevent self-triggering during response
- **Cleanup callbacks** (Home Assistant pattern): Resource management

---

## 5. Service Architecture & Long-Running Operation

### 5.1 Rhasspy: Multi-Process with supervisord

**Architecture**:
- Multiple Python processes (audio, wake, dialogue, ASR, NLU, TTS)
- supervisord or Docker Compose orchestration
- MQTT broker coordinates services (port 12183 internal, or external)

**Process Management**:
```ini
[program:rhasspy_wake]
command=/usr/bin/rhasspy-wake-raven
autostart=true
autorestart=true
stderr_logfile=/var/log/rhasspy/wake.err.log
```

**Restart Mechanism**:
- SIGHUP signal → reload config → restart processes
- Docker: `.restart_docker` file monitoring
- Individual service restart without full system restart

**Strengths**:
- Service isolation (crash doesn't kill everything)
- Independent restarts
- Graceful degradation

**Limitations**:
- Resource overhead (multiple Python processes)
- supervisord/Docker complexity
- MQTT broker dependency

### 5.2 Home Assistant: Integration Lifecycle

**Pattern**: Config entry lifecycle with cleanup hooks

**Setup**:
```python
async def async_setup_entry(hass, entry):
    # Initialize components
    await setup_services(hass, entry)

    # Register cleanup
    entry.async_on_unload(lambda: cleanup())

    return True
```

**Cleanup**:
```python
async def async_unload_entry(hass, entry):
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        entry.runtime_data.listener()  # Execute cleanup
    return unload_ok
```

**Entity Lifecycle**:
- `async_added_to_hass()`: Entity initialized
- `async_will_remove_from_hass()`: Entity about to be removed
- Properties: Memory-only (no I/O in getters)
- Updates: Polling (`async_update()`) or push-based (`async_schedule_update_ha_state()`)

**Strengths**:
- Automatic cleanup on errors
- Clear lifecycle hooks
- Async-first architecture

**Limitations**:
- Framework-specific patterns
- Not directly portable outside Home Assistant

### 5.3 Mycroft: start-mycroft.sh + MessageBus

**Architecture**:
- 5 separate processes: bus, skills, audio, voice, enclosure
- WebSocket messagebus (port 8181)
- Each service connects to messagebus

**Resource Usage** (Raspberry Pi 4):
```
messagebus:  ~20MB RAM
skills:      ~150MB RAM
audio:       ~50MB RAM
voice (STT): ~100MB RAM
Total:       ~320MB RAM
```

**Restart**:
- `./start-mycroft.sh restart skills` (individual service)
- `./start-mycroft.sh restart all` (all services)
- PID tracking via `/var/run/mycroft/`

**Strengths**:
- Modular service architecture
- Language-agnostic (any client can connect to WebSocket)

**Limitations**:
- **Security**: Port 8181 open, no authentication
- Resource intensive for embedded devices
- Complex multi-process management

### 5.4 Recommendation for Radio Assistant

**Pattern**: Single async process + systemd/supervisord (simplified Rhasspy)

**Rationale**:
- **Embedded target**: Raspberry Pi with limited resources
- **Single purpose**: Radio voice assistant (not general-purpose platform)
- **Security**: No network exposure needed
- **Simplicity**: Easier deployment and debugging

**Architecture**:
```python
class RadioAssistantService:
    def __init__(self, config):
        self.event_bus = EventBus()
        self.audio = AudioManager(self.event_bus)
        self.vad = StreamingVAD(self.event_bus)
        self.callsign = CallsignDetector(self.event_bus)
        self.transcription = TranscriptionEngine(self.event_bus)
        self.ptt = PTTController(self.event_bus)
        self.cleanup_callbacks = []

    async def start(self):
        """Start all components"""
        await asyncio.gather(
            self.event_bus.dispatch_loop(),
            self.audio.input_loop(),
            self.vad.processing_loop(),
            self.ptt.control_loop(),
            self.health_monitor(),
        )

    async def stop(self):
        """Cleanup all components"""
        for callback in self.cleanup_callbacks:
            try:
                await callback()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
```

**systemd Service**:
```ini
[Unit]
Description=Radio Voice Assistant
After=network.target sound.target

[Service]
Type=simple
User=radio
WorkingDirectory=/opt/radio
ExecStart=/usr/local/bin/radio-assistant --config /etc/radio/config.yaml
Restart=on-failure
RestartSec=10
StandardOutput=append:/var/log/radio/assistant.log
StandardError=append:/var/log/radio/assistant.err.log

[Install]
WantedBy=multi-user.target
```

**Benefits**:
- **Single process**: ~100-150MB RAM target (vs. Mycroft's 320MB)
- **systemd integration**: Standard Linux service management
- **No network exposure**: All communication in-process
- **Simple restart**: `systemctl restart radio-assistant`
- **Graceful degradation**: Component errors don't kill entire service

---

## 6. Error Recovery & Reliability

### 6.1 Rhasspy: Service Independence

**Patterns**:
- MQTT retain messages for state recovery
- Individual service restart (supervisord auto-restart)
- Timeout handling in dialogue manager (30s default)
- Event hooks (Wyoming satellite):
  - `--startup-command`
  - `--connected-command`, `--disconnected-command`
  - `--error-command`

**Strengths**:
- Service crashes don't kill system
- Auto-restart policies
- Event-driven lifecycle management

**Limitations**:
- MQTT broker is single point of failure
- Complex multi-service debugging

### 6.2 Home Assistant: Async Cleanup + Reconnection

**Patterns**:
- `async_on_unload()`: Automatic cleanup on errors
- Context managers (`async with`): Resource guarantee
- Reconnection backoff (Wyoming satellites)
- Entity lifecycle hooks

**Known Issues**:
- Wyoming satellites don't reconnect after power cycle (requires HA restart)
- Memory leaks with continuous streaming (importance of cleanup)
- State can get stuck ("Responding" state hangs)

**Strengths**:
- Explicit cleanup registration
- Resource tracking
- Async error handling

**Limitations**:
- Reconnection issues in production
- State management edge cases

### 6.3 Mycroft: MessageBus Recovery

**Patterns**:
- WebSocket reconnection
- Service independence (bus crash doesn't kill clients)
- Session timeout (5 min default)

**Known Issues**:
- **Security**: Port 8181 RCE vulnerabilities
- No authentication/authorization
- Messagebus crash requires restart

**Strengths**:
- Client reconnection support
- Service isolation

**Limitations**:
- Security vulnerabilities
- Single point of failure (messagebus)

### 6.4 Recommendation for Radio Assistant

**Pattern**: Home Assistant cleanup + Rhasspy timeout monitoring + component isolation

**Error Recovery**:
```python
class RadioComponent:
    def __init__(self, event_bus):
        self.bus = event_bus
        self.running = False
        self.cleanup_callbacks = []

    async def start(self):
        try:
            self.running = True
            await self.setup()
            self.cleanup_callbacks.append(self.teardown)
        except Exception as e:
            logger.error(f"Component start failed: {e}")
            await self.cleanup()
            raise

    async def stop(self):
        self.running = False
        await self.cleanup()

    async def cleanup(self):
        for callback in self.cleanup_callbacks:
            try:
                await callback()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
        self.cleanup_callbacks.clear()

class ConnectionManager:
    async def handle_connection_error(self, error):
        \"\"\"Exponential backoff reconnection\"\"\"
        for attempt in range(MAX_RETRIES):
            await asyncio.sleep(BACKOFF_DELAY * (2 ** attempt))
            try:
                await self.reconnect()
                return
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    raise
```

**Health Monitoring**:
```python
async def health_monitor(self):
    \"\"\"Monitor component health and restart if needed\"\"\"
    while True:
        await asyncio.sleep(10)  # Check every 10s

        for component in self.components:
            if not component.is_healthy():
                logger.warning(f"{component.name} unhealthy, restarting...")
                try:
                    await component.stop()
                    await component.start()
                except Exception as e:
                    logger.error(f"Restart failed: {e}")
                    # Could trigger full service restart via systemd
```

**Key Practices**:
- **Cleanup registration** (Home Assistant): Register callbacks early
- **Timeout monitoring** (Rhasspy): Background async task
- **Component health checks**: Periodic status verification
- **Exponential backoff**: Connection retry with increasing delays
- **Graceful degradation**: Continue operating with degraded functionality

---

## 7. Summary & Recommendations

### 7.1 What to Adopt

**From Rhasspy**:
1. **Template-based wake word detection** (DTW/MFCC) - perfect for custom callsigns
2. **PyAudio callback → message queue pattern** - proven thread safety
3. **Session lifecycle management** - clear start/end boundaries
4. **Timeout monitoring** - async background task (30s default)
5. **Auto-disable wake word during playback** - prevents self-triggering
6. **Refractory period** - 3-second cooldown after detection
7. **Audio specifications** - 16kHz, 16-bit mono standard

**From Home Assistant**:
1. **AsyncGenerator streaming** - memory-efficient audio processing
2. **async_on_unload cleanup pattern** - automatic resource management
3. **Lifecycle hooks** - clear component initialization/shutdown
4. **VAD filtering** - reduce processing (only stream during speech)
5. **Context managers** - resource guarantee (`async with`)
6. **Properties = memory only** - no I/O in property getters

**From Mycroft**:
1. **Event-driven architecture concept** - clean component decoupling
2. **Message naming conventions** - semantic event names (verb tenses)
3. **Service coordination pattern** - central dialogue manager
4. **Conversational context** - multi-turn conversation support
5. **Energy-based VAD** - simple but effective approach

### 7.2 What to Simplify/Avoid

**Avoid**:
1. **Multi-process architecture** (Rhasspy/Mycroft) - too heavy for embedded
2. **WebSocket messagebus** (Mycroft) - security issues, overhead
3. **MQTT broker** (Rhasspy) - unnecessary for single machine
4. **Wyoming protocol** (unless integrating with Home Assistant) - adds abstraction
5. **Dynamic skill loading** (Mycroft) - not needed for fixed functionality
6. **Cloud dependencies** (all) - must work offline

**Simplify**:
1. **Event bus**: In-process async queues (not WebSocket/MQTT)
2. **Process model**: Single async process (not 5+ processes)
3. **Service management**: systemd (not supervisord + MQTT + scripts)
4. **Configuration**: Local YAML/JSON (not cloud sync)
5. **Wake word**: Single approach (not 6+ engines to support)

### 7.3 Recommended Architecture

**Core Design**:
- **Single async Python process** with multiple coroutines
- **In-process event bus** using asyncio queues
- **Template-based callsign detection** (DTW)
- **PyAudio callback → async queue** for audio streaming
- **Event-driven pipeline** with clear state machine
- **systemd service** for process management
- **Cleanup callbacks** for resource management
- **Health monitoring** with component restart

**Pipeline Flow**:
```
Radio RX Audio → PyAudio callback → asyncio queue
    ↓
Streaming VAD → detect speech start/end
    ↓
Callsign Detector (DTW) → emit 'callsign.detected'
    ↓
Transcription Engine → emit 'transcription.complete'
    ↓
Intent Detection → emit 'radio.respond'
    ↓
TTS Generation → emit 'audio.ready'
    ↓
PTT Controller → transmit → emit 'ptt.complete'
    ↓
Return to IDLE (re-enable callsign detection)
```

**Event Types** (Mycroft-style naming):
- `audio.chunk` - Audio data available
- `vad.speech.start` - Speech detected
- `vad.speech.complete` - Speech ended (full utterance ready)
- `callsign.detected` - Callsign recognized
- `transcription.complete` - STT finished
- `radio.respond` - Generate response
- `audio.ready` - TTS complete
- `ptt.activate`, `ptt.release` - PTT control
- `ptt.complete` - Transmission finished
- `system.ready`, `system.stop`, `system.error` - Lifecycle events

**Resource Targets**:
- RAM: <150MB (vs. Mycroft 320MB)
- Processes: 1 (vs. Rhasspy 5+, Mycroft 5+)
- Latency: <500ms wake-to-response
- Network: None (fully offline)

### 7.4 Implementation Priority

**Phase 1: Core Event Bus** (1 day)
- Implement AsyncEventBus
- Event registration and dispatch
- Queue management with backpressure

**Phase 2: Audio Pipeline** (2 days)
- PyAudio callback → async queue
- StreamingVAD component
- Event emission on speech boundaries

**Phase 3: Callsign Detection** (2 days)
- Template-based DTW detector
- MFCC feature extraction
- Refractory period + threshold tuning

**Phase 4: Service Integration** (2 days)
- Refactor existing components to event bus
- Session/state management
- Cleanup callbacks and error recovery

**Phase 5: Testing & Validation** (2 days)
- Integration tests for full pipeline
- Long-running stability tests
- Memory leak detection
- Hardware validation (Raspberry Pi)

**Total: ~10 days** (2 work weeks)

---

## 8. Sources

### Research Documents Created
- `docs/claude/planning/RHASSPY_ARCHITECTURE_RESEARCH.md` - Rhasspy patterns
- `docs/claude/planning/HOME_ASSISTANT_VOICE_ARCHITECTURE.md` - Home Assistant patterns
- `docs/claude/planning/MYCROFT_ARCHITECTURE_RESEARCH.md` - Mycroft patterns

### Official Documentation
- [Rhasspy Documentation](https://rhasspy.readthedocs.io/)
- [Home Assistant Voice Pipelines](https://developers.home-assistant.io/docs/voice/pipelines/)
- [Mycroft AI Documentation](https://mycroft-ai.gitbook.io/docs/)

### GitHub Repositories
- [rhasspy/rhasspy](https://github.com/rhasspy/rhasspy)
- [rhasspy/wyoming](https://github.com/rhasspy/wyoming)
- [home-assistant/core](https://github.com/home-assistant/core)
- [MycroftAI/mycroft-core](https://github.com/MycroftAI/mycroft-core)

---

**Last Updated**: 2025-12-28
**Research Status**: Complete
**Next Step**: Phase 2 Architecture Proposal
