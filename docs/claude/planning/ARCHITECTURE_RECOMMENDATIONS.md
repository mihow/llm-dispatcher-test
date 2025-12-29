# Architecture Recommendations for Radio Voice Assistant

**Created**: 2025-12-28
**Based on**: Mycroft AI research + current system analysis
**Status**: Ready for implementation

---

## Executive Summary

Based on research into Mycroft AI's message bus architecture and analysis of the current radio assistant implementation, I recommend adopting an **event-driven architecture** with **lightweight async implementation** for the streaming service refactor.

**Key findings:**
- Current batch processing works (WER 0.0-0.125 on test cases)
- Need streaming architecture for continuous operation
- Mycroft's event-driven pattern is proven and suitable
- But Mycroft's WebSocket implementation is too heavy for embedded use
- Hybrid approach: Mycroft patterns + Python async queues

---

## Current System Analysis

### What We Have (Phase 1 Complete)

**Architecture**: Batch processing pipeline

```python
# Current flow (synchronous)
audio = load_wav("transmission.wav")
result = assistant.process_audio(audio)  # All in one call
if result: respond()
```

**Components**: (all working, 225/225 tests passing)
- AudioInterface - loads/plays audio files
- VADDetector - detects speech in complete audio
- TranscriptionEngine - Whisper-based STT
- CallsignDetector - matches callsigns with phonetic support
- PTTController - VOX-based transmission
- RadioAssistant - coordinates batch processing

**Validation results** (from `/home/michael/Projects/Radio/llm-dispatcher-test/validation_report.json`):
- Perfect transcription (WER 0.0) on 5/6 test cases
- One noisy case: WER 0.125 (transcribed "JJ-669" instead of "JJ 659")
- Callsign detection works on clear audio
- Phonetic detection not working yet (line 40-45: WER 0.0 but not detected)

### What We Need (Phase 2B - Streaming Service)

**Architecture**: Continuous streaming service

```python
# Target flow (asynchronous, event-driven)
while running:
    chunk = await capture_chunk(0.5)           # Continuous
    await event_bus.emit('audio.chunk', chunk) # Event-driven
    # Process concurrently with next chunk
```

**Missing components** (from `/home/michael/Projects/Radio/llm-dispatcher-test/docs/claude/planning/SERVICE_MODE_GAPS.md`):
- StreamingVAD (chunk-by-chunk speech detection)
- AsyncEventBus (component coordination)
- ConversationState (multi-turn conversations)
- AudioStreamManager (continuous I/O)
- PTTCoordinator (TX/RX synchronization)
- Resource management (long-running stability)

---

## Mycroft Research Findings

### What Mycroft Does Well

**1. Event-Driven Message Bus**
- All components communicate via typed messages
- Clean separation: audio client, skills, audio service run independently
- Proven pattern for 5+ years of production use

**2. Audio Pipeline Architecture**
- Continuous audio capture with ring buffer
- Energy-based VAD with dynamic thresholds
- Wake word detection every 200ms
- Session management with 5-minute timeout

**3. Component Lifecycle**
- `initialize()` - register handlers, load config
- `converse()` - handle multi-turn conversations
- `stop()` - interrupt current action
- `shutdown()` - clean resource release

**4. Context Management**
- Track conversation state across utterances
- 5-minute timeout (configurable)
- Cross-component context sharing

### What Mycroft Does Poorly (for Radio Use)

**1. Security**
- WebSocket on port 8181 with **zero authentication**
- Known remote code execution vulnerabilities
- Completely open to network (binds 0.0.0.0)

**2. Resource Usage**
- 5+ separate Python processes
- ~320MB RAM total on Raspberry Pi
- WebSocket serialization overhead

**3. Deployment Complexity**
- Multiple services to manage
- Process orchestration required
- Single point of failure (message bus crash)

**4. Cloud Dependencies**
- Backend required for full features
- API key storage on cloud
- Won't work fully offline

---

## Recommended Architecture

### Hybrid Approach: Event Pattern + Lightweight Implementation

**Principle**: Use Mycroft's proven event-driven coordination but implement with Python async primitives instead of WebSocket server.

### Core Design

```python
class AsyncEventBus:
    """Lightweight in-process event bus (no WebSocket)."""

    def __init__(self):
        self.handlers: dict[str, list[Callable]] = defaultdict(list)
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=100)

    def on(self, event_type: str, handler: Callable):
        """Register handler for event type."""
        self.handlers[event_type].append(handler)

    async def emit(self, event_type: str, data: dict):
        """Emit event to all registered handlers."""
        event = Event(type=event_type, data=data)
        await self.queue.put(event)

    async def dispatch_loop(self):
        """Process events from queue."""
        while True:
            event = await self.queue.get()
            for handler in self.handlers[event.type]:
                await handler(event.data)
```

### Service Architecture

```python
class RadioAssistantService:
    """Single-process async service with event-driven coordination."""

    def __init__(self, config: AppConfig):
        # Event bus (in-process, not WebSocket)
        self.event_bus = AsyncEventBus()

        # Components register as event handlers
        self.audio = AudioStreamManager(event_bus=self.event_bus)
        self.vad = StreamingVAD(event_bus=self.event_bus)
        self.transcription = TranscriptionEngine(event_bus=self.event_bus)
        self.callsign = CallsignDetector(event_bus=self.event_bus)
        self.ptt = PTTCoordinator(event_bus=self.event_bus)
        self.conversation = ConversationManager(event_bus=self.event_bus)

    async def start(self):
        """Start all coroutines concurrently."""
        await asyncio.gather(
            self.event_bus.dispatch_loop(),  # Event dispatcher
            self.audio.input_loop(),         # Continuous capture
            self.audio.output_loop(),        # Response playback
            self.conversation.timeout_loop(), # Session management
            self.health_monitor_loop(),      # Resource monitoring
        )
```

### Event Flow (Inspired by Mycroft)

```
Audio Input Loop:
  └─> emit('audio.chunk', {audio: np.ndarray})
        |
        v
StreamingVAD: (listening for 'audio.chunk')
  └─> emit('vad.speech.complete', {audio: np.ndarray, duration: float})
        |
        v
TranscriptionEngine: (listening for 'vad.speech.complete')
  └─> emit('transcription.complete', {text: str, confidence: float})
        |
        v
CallsignDetector: (listening for 'transcription.complete')
  └─> emit('callsign.detected', {callsign: str, matched_form: str})
        |
        v
ConversationManager: (listening for 'callsign.detected')
  └─> emit('radio.respond', {response_audio: np.ndarray})
        |
        v
PTTCoordinator: (listening for 'radio.respond')
  └─> emit('ptt.complete')
```

### Component Implementation Pattern

Each component follows Mycroft's lifecycle pattern:

```python
class StreamingVAD:
    """VAD using event bus coordination."""

    def __init__(self, event_bus: AsyncEventBus):
        self.bus = event_bus
        self.buffer: list[np.ndarray] = []
        self.state = "listening"  # listening | collecting | processing

        # Register as listener (like Mycroft's initialize())
        self.bus.on('audio.chunk', self.on_audio_chunk)
        self.bus.on('system.stop', self.on_stop)

    async def on_audio_chunk(self, data: dict):
        """Handle incoming audio (like Mycroft intent handler)."""
        chunk = data['audio']

        if self.state == "listening":
            if self.is_speech(chunk):
                self.state = "collecting"
                self.buffer = [chunk]
                await self.bus.emit('vad.speech.start')

        elif self.state == "collecting":
            self.buffer.append(chunk)

            if not self.is_speech(chunk):
                self.silence_chunks += 1
                if self.silence_chunks > 3:  # 500ms silence
                    complete = np.concatenate(self.buffer)
                    await self.bus.emit('vad.speech.complete', {
                        'audio': complete,
                        'duration': len(complete) / SAMPLE_RATE
                    })
                    self.reset()

    async def on_stop(self, data: dict):
        """Handle stop event (like Mycroft's stop() method)."""
        self.reset()

    def reset(self):
        """Reset state (like Mycroft's conversation timeout)."""
        self.buffer = []
        self.state = "listening"
        self.silence_chunks = 0
```

---

## Comparison: Current vs Mycroft vs Recommended

| Aspect | Current (Phase 1) | Mycroft | Recommended (Phase 2) |
|--------|-------------------|---------|----------------------|
| **Architecture** | Batch processing | Multi-process, WebSocket bus | Single process, async events |
| **Coordination** | Direct method calls | WebSocket messages | Async event bus |
| **Audio Processing** | Load file, process all | Streaming chunks | Streaming chunks |
| **State Management** | None | ContextManager + Session | ConversationManager |
| **Multi-turn** | Not supported | converse() method | Event-based context |
| **Resource Usage** | ~100MB (batch) | ~320MB (5 processes) | ~150MB (1 process) |
| **Security** | N/A (no network) | Open port 8181 | No network exposure |
| **Deployment** | Run script once | 5 services + orchestration | Single service |
| **Testing** | File-based tests | Message mocking | Event mocking |
| **Suitable for** | Batch validation | Desktop/server | Embedded (Raspberry Pi) |

---

## Implementation Plan

### Phase 1: Event Bus Core (1 day)

**Deliverable**: Working async event bus with tests

```python
# Implement:
- AsyncEventBus class
- Event registration (on)
- Event emission (emit)
- Dispatch loop
- Queue backpressure handling
- Error isolation (handler failure doesn't crash bus)

# Tests:
- test_event_registration()
- test_event_dispatch()
- test_multiple_handlers()
- test_handler_error_isolation()
- test_queue_backpressure()
```

### Phase 2: Refactor Components to Events (2 days)

**Deliverable**: Existing components work with event bus

**Refactor each component:**

```python
# Before (current):
class VADDetector:
    def is_speech(self, audio: np.ndarray) -> bool:
        return self.detector.is_speech(audio)

# After (event-driven):
class StreamingVAD:
    def __init__(self, event_bus: AsyncEventBus):
        self.bus = event_bus
        self.bus.on('audio.chunk', self.on_audio_chunk)

    async def on_audio_chunk(self, data: dict):
        chunk = data['audio']
        complete_utterance = self.process_chunk(chunk)
        if complete_utterance is not None:
            await self.bus.emit('vad.speech.complete', {
                'audio': complete_utterance
            })
```

**Components to refactor:**
1. AudioInterface → AudioStreamManager
   - Emits: `audio.chunk`
   - Listens: `audio.input.start`, `audio.input.stop`

2. VADDetector → StreamingVAD
   - Listens: `audio.chunk`
   - Emits: `vad.speech.start`, `vad.speech.complete`

3. TranscriptionEngine (minor changes)
   - Listens: `vad.speech.complete`
   - Emits: `transcription.complete`

4. CallsignDetector (minor changes)
   - Listens: `transcription.complete`
   - Emits: `callsign.detected`

5. PTTController → PTTCoordinator
   - Listens: `radio.respond`
   - Emits: `ptt.start`, `ptt.complete`

**New component:**
6. ConversationManager
   - Listens: `callsign.detected`, `transcription.complete`
   - Emits: `radio.respond`
   - Manages: session timeout, context tracking

### Phase 3: Add Mycroft-Inspired Features (1 day)

**Session Management:**

```python
class ConversationManager:
    """Manage conversation state (like Mycroft's SessionManager)."""

    def __init__(self, event_bus: AsyncEventBus, timeout: int = 300):
        self.bus = event_bus
        self.timeout = timeout
        self.last_activity = None
        self.active_callsign = None
        self.context: dict = {}

        self.bus.on('callsign.detected', self.on_callsign)
        self.bus.on('transcription.complete', self.on_utterance)

    async def on_callsign(self, data: dict):
        """Handle callsign detection (start/continue session)."""
        self.last_activity = time.time()
        self.active_callsign = data['callsign']
        self.context['last_callsign'] = data['callsign']

        # Decide if should respond
        if self.should_respond():
            await self.bus.emit('radio.respond', {
                'callsign': self.active_callsign,
                'response_type': 'acknowledge'
            })

    async def timeout_loop(self):
        """Monitor session timeout (like Mycroft's session timeout)."""
        while True:
            await asyncio.sleep(10)  # Check every 10s

            if self.last_activity:
                elapsed = time.time() - self.last_activity
                if elapsed > self.timeout:
                    logger.info("Session timeout, resetting context")
                    self.reset()

    def reset(self):
        """Reset conversation state."""
        self.context = {}
        self.active_callsign = None
```

**Context Tracking:**

```python
class ConversationContext:
    """Track conversation context (like Mycroft's ContextManager)."""

    def __init__(self, timeout: int = 300):
        self.context: dict[str, Any] = {}
        self.timestamps: dict[str, float] = {}
        self.timeout = timeout

    def set(self, key: str, value: Any):
        """Set context value (like Mycroft's set_context)."""
        self.context[key] = value
        self.timestamps[key] = time.time()

    def get(self, key: str) -> Any | None:
        """Get context value if not expired."""
        if key not in self.context:
            return None

        # Check timeout
        if time.time() - self.timestamps[key] > self.timeout:
            del self.context[key]
            del self.timestamps[key]
            return None

        return self.context[key]
```

**Lifecycle Events:**

```python
# System lifecycle (like Mycroft's system messages)
await event_bus.emit('system.ready')
await event_bus.emit('system.stop')
await event_bus.emit('component.error', {'component': 'vad', 'error': str(e)})
```

### Phase 4: Streaming Integration Tests (1 day)

**Test continuous operation:**

```python
# tests/streaming/test_event_flow.py

async def test_complete_event_flow():
    """Test full event flow from audio to response."""

    bus = AsyncEventBus()
    service = RadioAssistantService(config, event_bus=bus)

    # Track events received
    events_received = []
    bus.on('ptt.complete', lambda d: events_received.append('ptt.complete'))

    # Simulate audio chunks (like real streaming)
    audio = load_wav("wsjj659_clear.wav")
    chunks = split_into_chunks(audio, chunk_size=8000)  # 0.5s chunks

    for chunk in chunks:
        await bus.emit('audio.chunk', {'audio': chunk})
        await asyncio.sleep(0.01)  # Simulate real-time

    # Wait for processing
    await asyncio.sleep(1.0)

    # Should have completed full flow
    assert 'ptt.complete' in events_received


async def test_multiple_transmissions():
    """Test back-to-back transmissions (like real radio)."""

    service = RadioAssistantService(config)
    responses = []

    service.event_bus.on('radio.respond', lambda d: responses.append(d))

    # Simulate 3 transmissions
    for audio_file in ['wsjj659_clear.wav', 'wsjj659_rapid.wav', 'wsjj659_noisy.wav']:
        audio = load_wav(audio_file)
        await service.process_streaming(audio)

    # Should handle all 3
    assert len(responses) >= 2  # At least 2 should be detected


async def test_session_timeout():
    """Test conversation timeout (like Mycroft's session timeout)."""

    service = RadioAssistantService(config, timeout=2)  # 2s timeout

    # First transmission
    await service.process_audio(load_wav("wsjj659_clear.wav"))
    assert service.conversation.active_callsign == "WSJJ659"

    # Wait for timeout
    await asyncio.sleep(3)

    # Context should be reset
    assert service.conversation.active_callsign is None
```

### Phase 5: Long-Running Stability (1 day)

**Test resource management:**

```python
async def test_memory_stability():
    """Test service runs for extended period without leaks."""

    service = RadioAssistantService(config)
    start_memory = get_memory_usage()

    # Simulate 1 hour of operation
    for i in range(60):
        audio = generate_test_audio()
        await service.process_streaming(audio)
        await asyncio.sleep(1)

        if i % 10 == 0:
            current_memory = get_memory_usage()
            assert current_memory < start_memory * 1.5  # <50% growth

    end_memory = get_memory_usage()
    assert end_memory < start_memory * 2  # <100% growth total
```

---

## Benefits of This Approach

### Vs Current Batch Processing

| Feature | Current | Event-Driven |
|---------|---------|--------------|
| Continuous operation | No | Yes |
| Streaming audio | No | Yes |
| Multi-turn conversations | No | Yes |
| State tracking | No | Yes |
| Concurrent I/O | No | Yes |
| Component isolation | Tight coupling | Loose coupling |
| Testing | File-based | Event mocking |
| Extensibility | Hard | Easy (add listeners) |

### Vs Mycroft's Implementation

| Feature | Mycroft | Our Implementation |
|---------|---------|-------------------|
| Event coordination | WebSocket | Async queues |
| Process model | Multi-process | Single process |
| Memory usage | ~320MB | ~150MB |
| Latency | 10-50ms | <1ms |
| Security | Open port | No network |
| Deployment | Complex | Simple |
| Embedded suitability | Poor | Excellent |

### Key Advantages

1. **Proven Pattern** - Mycroft has thousands of users, pattern is battle-tested
2. **Component Isolation** - Easy to test, modify, extend individual components
3. **Resource Efficient** - Single process, minimal overhead for Raspberry Pi
4. **Testable** - Mock events instead of actual audio/radio hardware
5. **Maintainable** - Clear boundaries, typed events, predictable flow
6. **Extensible** - Add new features by adding event handlers
7. **Secure** - No network exposure, no open ports

---

## Migration Path

### Keep Working During Migration

**Strategy**: Implement new architecture alongside current code, migrate gradually

**Phase 1**: Event bus + tests (no disruption)
**Phase 2**: Refactor one component at a time, keep tests passing
**Phase 3**: Add new features using event pattern
**Phase 4**: Deprecate old batch interface (keep for testing)

**Backward compatibility:**

```python
class RadioAssistant:
    """Facade providing both batch and streaming interfaces."""

    def process_audio(self, audio: np.ndarray) -> bool:
        """Batch interface (keep for testing)."""
        # Still works, internally uses event bus

    async def start_streaming(self):
        """New streaming interface."""
        await self.service.start()
```

---

## Risks and Mitigations

### Risk 1: Complexity Increase

**Risk**: Event-driven architecture more complex than current direct calls

**Mitigation**:
- Clear documentation of event flow
- Visual diagrams of event chains
- Type-safe events (Pydantic models)
- Good logging of event dispatch

### Risk 2: Performance Overhead

**Risk**: Event queue dispatch adds latency

**Mitigation**:
- Benchmark event dispatch (<1ms expected)
- Monitor queue depth in production
- Set queue size limits (backpressure)
- Profile hot paths

### Risk 3: Debugging Difficulty

**Risk**: Harder to trace flow through events vs direct calls

**Mitigation**:
- Structured logging with event IDs
- Event tracing mode for development
- Clear naming conventions
- Test coverage for event flows

### Risk 4: Breaking Existing Tests

**Risk**: Refactoring might break 225 passing tests

**Mitigation**:
- Keep batch interface working during migration
- Migrate tests gradually
- Run full suite after each component refactor
- Feature flags for new/old architecture

---

## Success Metrics

### Phase 2B Success Criteria

After implementing event-driven architecture:

1. **All existing tests still pass** (225/225)
2. **New streaming tests pass** (expect 20+ new tests)
3. **Memory usage** <200MB on Raspberry Pi
4. **Latency** <500ms from audio chunk to response
5. **Stability** runs 24+ hours without crash
6. **Event dispatch** <1ms per event
7. **Queue depth** never exceeds 100 events

### Validation Metrics

Maintain or improve current accuracy:
- WER <0.15 on noisy audio (currently 0.125)
- WER <0.05 on clear audio (currently 0.0)
- Callsign detection >90% on clear audio
- Phonetic detection working (currently broken)

---

## Timeline Estimate

| Phase | Duration | Effort |
|-------|----------|--------|
| 1. Event Bus Core | 1 day | 8 hours |
| 2. Component Refactor | 2 days | 16 hours |
| 3. Mycroft Features | 1 day | 8 hours |
| 4. Streaming Tests | 1 day | 8 hours |
| 5. Stability Testing | 1 day | 8 hours |
| **Total** | **1 week** | **48 hours** |

**Dependencies:**
- WER validation complete (provides baseline)
- Current tests all passing (verified)
- Hardware available for final testing (week 2)

---

## Conclusion

**Recommendation**: Proceed with event-driven architecture refactor using Mycroft's patterns but lightweight async implementation.

**Why this is the right approach:**

1. **Proven** - Mycroft's pattern works in production for similar use case
2. **Necessary** - Need streaming for continuous radio operation
3. **Efficient** - Simpler than Mycroft's implementation, suited for embedded
4. **Testable** - Event mocking much easier than audio hardware mocking
5. **Maintainable** - Clear component boundaries, typed events
6. **Extensible** - Easy to add features (new event handlers)
7. **Low risk** - Can migrate gradually, keep existing tests passing

**Next steps:**

1. Review and approve this architecture
2. Implement Phase 1 (event bus core) - 1 day
3. Refactor components to events - 2 days
4. Add Mycroft-inspired features - 1 day
5. Test streaming operation - 1 day
6. Hardware validation - week 2

**Expected outcome**: Production-ready streaming service for radio voice assistant, based on proven voice assistant architecture patterns, optimized for embedded deployment.

---

## References

- Research document: `/home/michael/Projects/Radio/llm-dispatcher-test/docs/claude/planning/MYCROFT_ARCHITECTURE_RESEARCH.md`
- Current gaps: `/home/michael/Projects/Radio/llm-dispatcher-test/docs/claude/planning/SERVICE_MODE_GAPS.md`
- Phase 2 plan: `/home/michael/Projects/Radio/llm-dispatcher-test/docs/claude/planning/PHASE2_PLANNING_SUMMARY.md`
- Validation results: `/home/michael/Projects/Radio/llm-dispatcher-test/validation_report.json`
- Current implementation: `/home/michael/Projects/Radio/llm-dispatcher-test/radio_assistant/main.py`

---

*Created: 2025-12-28*
*Status: Ready for implementation*
*Phase: 2A - Validation & Research*
