# Service Mode Architecture Gaps

**Created**: 2025-12-27
**Critical Issue**: We're testing batch processing, not continuous service operation

## The Real Use Case

```
Radio → Audio Input → [CONTINUOUS LISTENING] → VAD → Transcription → Callsign → PTT → Audio Output → Radio
         ↑                                                                                    ↓
         └────────────────────── Loop Forever ──────────────────────────────────────────────┘
```

**What we're testing:**
```python
# Load audio file
audio = load_wav("test.wav")

# Process once
result = assistant.process_audio(audio)

# Done - exit
```

**What actually needs to happen:**
```python
# Start service
service = RadioService(config)

# Run forever
while True:
    chunk = capture_audio_chunk(duration=0.5)  # Real-time from soundcard
    service.process_chunk(chunk)               # Stream processing
    # Simultaneously: play any pending responses
```

---

## Critical Missing Components

### 1. **Continuous Audio Streaming**

**Current gap:**
- All tests use pre-loaded audio files
- No real-time audio capture
- No streaming buffer management

**What's needed:**

```python
class AudioStreamManager:
    """Manage continuous audio input/output streams."""

    def __init__(self, sample_rate: int = 16000):
        self.input_stream = None
        self.output_stream = None
        self.buffer = RingBuffer(max_size=16000 * 10)  # 10 seconds
        self.running = False

    def start_listening(self):
        """Start continuous audio input."""
        # Open input stream from soundcard
        # Feed into ring buffer
        # Handle overruns

    def play_audio(self, audio: np.ndarray):
        """Play audio through output stream."""
        # Queue audio for playback
        # Handle device busy
        # Wait for completion or timeout
```

**Tests needed:**
- Can we capture audio continuously without dropouts?
- Buffer management under load
- Input/output stream synchronization
- Device connect/disconnect handling

---

### 2. **Streaming VAD (Windowed Detection)**

**Current gap:**
- VAD processes complete audio files
- No sliding window approach
- No state across chunks

**What's needed:**

```python
class StreamingVAD:
    """VAD for continuous audio streams."""

    def __init__(self):
        self.detector = VADDetector()
        self.speech_buffer = []
        self.state = "listening"  # listening, collecting, processing
        self.silence_counter = 0

    def process_chunk(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Process audio chunk, return complete utterance when ready.

        States:
        - listening: waiting for speech to start
        - collecting: gathering speech chunks
        - processing: speech ended, ready to process

        Returns:
            Complete audio segment when speech ends, None otherwise
        """
        is_speech = self.detector.is_speech(chunk)

        if self.state == "listening":
            if is_speech:
                self.state = "collecting"
                self.speech_buffer = [chunk]

        elif self.state == "collecting":
            self.speech_buffer.append(chunk)

            if not is_speech:
                self.silence_counter += 1

                # If silence for N chunks, speech has ended
                if self.silence_counter > 3:  # ~500ms of silence
                    complete_audio = np.concatenate(self.speech_buffer)
                    self.reset()
                    return complete_audio
            else:
                self.silence_counter = 0

        return None

    def reset(self):
        self.speech_buffer = []
        self.state = "listening"
        self.silence_counter = 0
```

**Tests needed:**
- Correctly detects speech start/end in stream
- Handles multiple transmissions back-to-back
- Doesn't miss short utterances
- Doesn't segment mid-sentence
- Timeout handling for very long transmissions

---

### 3. **Asynchronous Processing Pipeline**

**Current gap:**
- Everything is synchronous
- Can't listen while processing previous transmission
- Can't process while playing response

**What's needed:**

```python
import asyncio
from asyncio import Queue

class AsyncRadioService:
    """Asynchronous service with concurrent audio I/O and processing."""

    def __init__(self, config):
        self.audio_queue = Queue()      # Incoming audio chunks
        self.response_queue = Queue()   # Outgoing responses
        self.processing = False

    async def audio_input_loop(self):
        """Continuously capture audio and queue for processing."""
        while self.running:
            chunk = await self.capture_chunk()
            await self.audio_queue.put(chunk)

    async def processing_loop(self):
        """Process audio chunks from queue."""
        streaming_vad = StreamingVAD()

        while self.running:
            chunk = await self.audio_queue.get()

            # Check if speech segment is complete
            complete_audio = streaming_vad.process_chunk(chunk)

            if complete_audio is not None and not self.processing:
                self.processing = True
                response = await self.process_utterance(complete_audio)
                if response:
                    await self.response_queue.put(response)
                self.processing = False

    async def audio_output_loop(self):
        """Play responses from queue."""
        while self.running:
            response = await self.response_queue.get()
            await self.play_response(response)

    async def run(self):
        """Run all loops concurrently."""
        await asyncio.gather(
            self.audio_input_loop(),
            self.processing_loop(),
            self.audio_output_loop(),
        )
```

**Tests needed:**
- Can process while listening?
- Can handle back-to-back transmissions?
- Queue doesn't overflow under load?
- Graceful shutdown?

---

### 4. **State Management Across Transmissions**

**Current gap:**
- No conversation state
- Each transmission processed independently
- No timeout handling

**What's needed:**

```python
class ConversationState:
    """Track conversation state across transmissions."""

    def __init__(self, timeout_seconds: int = 60):
        self.last_transmission_time = None
        self.transmission_count = 0
        self.last_callsign = None
        self.context_window = []  # Recent transmissions
        self.timeout = timeout_seconds

    def add_transmission(self, text: str, callsign: Optional[str]):
        """Add transmission to context."""
        now = time.time()

        # Reset if timeout exceeded
        if self.last_transmission_time:
            if now - self.last_transmission_time > self.timeout:
                self.reset()

        self.context_window.append({
            "text": text,
            "callsign": callsign,
            "timestamp": now
        })

        self.transmission_count += 1
        self.last_transmission_time = now
        self.last_callsign = callsign

        # Keep only recent context (e.g., last 5 transmissions)
        if len(self.context_window) > 5:
            self.context_window.pop(0)

    def should_respond(self, callsign: str) -> bool:
        """Determine if we should respond based on context."""
        # Don't respond to repeated transmissions
        if len(self.context_window) >= 2:
            last_two = self.context_window[-2:]
            if all(t["text"] == last_two[0]["text"] for t in last_two):
                return False  # Same message repeated

        return True

    def reset(self):
        self.context_window = []
        self.transmission_count = 0
```

**Tests needed:**
- Conversation timeout resets state correctly
- Multiple transmissions tracked
- Repeated transmissions detected
- Context maintained across transmissions

---

### 5. **Audio Playback Coordination**

**Current gap:**
- No actual audio output
- No VOX timing coordination
- No handling of "can't transmit right now"

**What's needed:**

```python
class PTTCoordinator:
    """Coordinate PTT activation and audio playback."""

    def __init__(self, vox_padding_ms: int = 300):
        self.transmitting = False
        self.vox_padding_ms = vox_padding_ms
        self.audio_player = AudioPlayer()

    async def play_response(self, audio_data: np.ndarray):
        """Play response audio with proper PTT timing."""

        # 1. Check we're not receiving
        if self.currently_receiving():
            # Wait or abort?
            await self.wait_for_clear_channel(timeout=5.0)

        # 2. Activate PTT (or wait for VOX)
        self.activate_ptt()
        await asyncio.sleep(self.vox_padding_ms / 1000)

        # 3. Play audio
        self.transmitting = True
        try:
            await self.audio_player.play(audio_data)
        finally:
            self.transmitting = False

        # 4. Post-transmission delay
        await asyncio.sleep(self.vox_padding_ms / 1000)

        # 5. Release PTT
        self.release_ptt()

    def currently_receiving(self) -> bool:
        """Check if we're currently receiving audio."""
        # Check VAD state
        # Check if input stream has recent activity
        pass
```

**Tests needed:**
- PTT timing is correct
- Won't transmit while receiving
- VOX padding prevents cutoff
- Queue multiple responses correctly

---

### 6. **Resource Management & Cleanup**

**Current gap:**
- No long-running operation testing
- Unknown memory behavior over time
- No cleanup on shutdown

**What's needed:**

```python
class ResourceManager:
    """Manage resources for long-running service."""

    def __init__(self):
        self.model_cache = {}
        self.audio_buffers = []

    def cleanup_old_buffers(self):
        """Clean up old audio buffers to prevent memory leak."""
        # Remove buffers older than threshold

    def reload_models_if_needed(self):
        """Reload models if memory pressure high."""
        # Check memory usage
        # Unload/reload models

    async def health_check(self) -> dict:
        """Return service health metrics."""
        return {
            "memory_mb": get_memory_usage(),
            "uptime_seconds": get_uptime(),
            "transmissions_processed": self.tx_count,
            "queue_depth": self.audio_queue.qsize(),
            "last_activity": self.last_activity_time,
        }
```

**Tests needed:**
- Memory usage stable over 1000+ transmissions
- No file descriptor leaks
- Graceful shutdown releases all resources
- Can restart without process restart

---

### 7. **Integration Tests for Continuous Operation**

**What we need to test:**

#### Test: Multiple Sequential Transmissions
```python
async def test_sequential_transmissions():
    """Test handling back-to-back transmissions."""
    service = AsyncRadioService(config)

    # Simulate 5 transmissions in sequence
    transmissions = [
        load_audio("wsjj659_clear.wav"),
        load_audio("scenario_noise.wav"),      # Should ignore
        load_audio("wsjj659_noisy.wav"),
        load_audio("other_callsign.wav"),      # Should ignore
        load_audio("wsjj659_clear.wav"),
    ]

    responses = []
    for audio in transmissions:
        response = await service.process_streaming(audio)
        if response:
            responses.append(response)

    # Should respond to 3 valid transmissions, ignore 2
    assert len(responses) == 3
```

#### Test: Interleaved Input/Output
```python
async def test_listen_while_responding():
    """Verify can continue listening during response playback."""

    service = AsyncRadioService(config)

    # Start response playback (takes ~2 seconds)
    response_task = asyncio.create_task(
        service.play_response(response_audio)
    )

    # While responding, receive new transmission
    await asyncio.sleep(0.5)  # Let response start
    new_audio = load_audio("wsjj659_urgent.wav")

    # Should queue new transmission for processing
    await service.audio_queue.put(new_audio)

    await response_task

    # New transmission should be queued, not lost
    assert service.audio_queue.qsize() > 0
```

#### Test: Long-Running Stability
```python
async def test_long_running_stability():
    """Test service stability over extended operation."""

    service = AsyncRadioService(config)

    # Simulate 1 hour of operation
    start_memory = get_memory_usage()

    for i in range(60):  # 60 iterations = 1 hour at 1/min
        # Random mix of speech/silence/noise
        audio = generate_random_scenario()
        await service.process_streaming(audio)

        await asyncio.sleep(1.0)

        # Check health every 10 iterations
        if i % 10 == 0:
            health = await service.health_check()
            assert health["memory_mb"] < start_memory * 1.5  # <50% increase

    end_memory = get_memory_usage()

    # Memory shouldn't grow unbounded
    assert end_memory < start_memory * 2  # <100% increase over 1hr
```

---

### 8. **Edge Cases in Streaming Mode**

**Test cases we need:**

1. **Speech starts during response playback**
   - Buffer it? Queue it? Drop it?
   - Test the chosen behavior

2. **Very long transmission (>30 seconds)**
   - Does it segment properly?
   - Timeout handling?

3. **Audio device disconnects mid-operation**
   - Graceful degradation
   - Recovery when reconnects

4. **Corrupted audio chunks**
   - NaN values, clipping, silence
   - Don't crash, log error

5. **Rapid-fire transmissions (<1s apart)**
   - Can keep up?
   - Queue overflow?

6. **Whisper model timeout/hang**
   - Kill and restart after timeout
   - Don't block entire service

---

## Proposed Service Architecture

```python
class RadioAssistantService:
    """Main service orchestrator."""

    def __init__(self, config: AppConfig):
        # Core components
        self.vad = StreamingVAD()
        self.transcription = TranscriptionEngine()
        self.callsign = CallsignDetector()

        # Service components
        self.audio_stream = AudioStreamManager()
        self.ptt = PTTCoordinator()
        self.conversation = ConversationState()
        self.resource_mgr = ResourceManager()

        # Async queues
        self.audio_queue = asyncio.Queue(maxsize=100)
        self.response_queue = asyncio.Queue(maxsize=10)

        self.running = False

    async def start(self):
        """Start all service loops."""
        self.running = True

        await asyncio.gather(
            self.audio_input_loop(),
            self.vad_loop(),
            self.processing_loop(),
            self.output_loop(),
            self.health_monitor_loop(),
        )

    async def stop(self):
        """Graceful shutdown."""
        self.running = False
        # Wait for queues to drain
        # Close audio streams
        # Save state if needed
```

---

## Implementation Priority

### Phase 1: Basic Streaming (1-2 days)
- [ ] Implement `StreamingVAD` with chunk-by-chunk processing
- [ ] Add ring buffer for audio
- [ ] Test multi-transmission scenarios

### Phase 2: Async Service (1-2 days)
- [ ] Implement async service with queues
- [ ] Add proper audio I/O streams (sounddevice)
- [ ] Test concurrent input/processing/output

### Phase 3: State & Coordination (1 day)
- [ ] Add conversation state management
- [ ] Implement PTT coordination
- [ ] Test interleaved transmit/receive

### Phase 4: Reliability (1 day)
- [ ] Resource management
- [ ] Error recovery
- [ ] Long-running stability tests

---

## Test Strategy for Service Mode

### Unit Tests
- Individual components (VAD, transcription, etc.)
- Already have these ✓

### Integration Tests (Batch)
- Current file-based tests
- Already have these ✓

### **NEW: Streaming Integration Tests**
```python
# tests/streaming/test_continuous_operation.py

async def test_streaming_vad():
    """Test VAD with chunked audio stream."""

async def test_multiple_transmissions_in_sequence():
    """Test processing multiple transmissions."""

async def test_concurrent_input_output():
    """Test listening while transmitting."""

async def test_long_running_stability():
    """Test service runs for extended period."""
```

### **NEW: Service Tests**
```python
# tests/service/test_radio_service.py

async def test_service_lifecycle():
    """Test start/stop/restart."""

async def test_resource_cleanup():
    """Test no resource leaks."""

async def test_error_recovery():
    """Test recovery from failures."""
```

---

## The Gap Summary

**What we're testing now:**
- ✅ Batch processing of audio files
- ✅ Component integration
- ✅ Accuracy with known inputs

**What we're NOT testing:**
- ❌ Continuous audio streaming
- ❌ Real-time chunk-by-chunk processing
- ❌ Concurrent input/output
- ❌ State across transmissions
- ❌ Long-running stability
- ❌ Resource management over time
- ❌ Async coordination

**The real risk:**
Even if all current tests pass, the service could fail in production because we've never tested it as a **continuous streaming service**.

---

## Next Steps

**Before hardware/real audio, we should:**

1. ✅ **Add WER validation** (as planned) - validates correctness
2. ✅ **Implement StreamingVAD** - critical for continuous operation
3. ✅ **Build async service architecture** - required for real deployment
4. ✅ **Add streaming integration tests** - validates service mode works
5. ✅ **Test long-running stability** - ensures no memory leaks/crashes

**Then:**
- Hardware testing will validate audio I/O works
- Real radio audio will validate handles radio artifacts

But if we don't test streaming/service mode, we might find critical architectural issues during hardware testing.
