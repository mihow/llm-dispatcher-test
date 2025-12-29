# Voice Assistant Architecture Research Summary

**Created**: 2025-12-28
**Purpose**: Consolidate findings from Mycroft, Home Assistant, and Rhasspy research
**Status**: Research phase complete, ready for implementation decision

---

## Research Documents

1. **MYCROFT_ARCHITECTURE_RESEARCH.md** (32KB) - Mycroft AI message bus research
2. **HOME_ASSISTANT_VOICE_ARCHITECTURE.md** (21KB) - Home Assistant voice pipeline
3. **rhasspy_architecture_research.md** (33KB) - Rhasspy wake word and streaming
4. **ARCHITECTURE_RECOMMENDATIONS.md** (23KB) - Final recommendations

---

## Executive Summary

After researching three major open-source voice assistant architectures, the clear recommendation is:

**Adopt event-driven async architecture inspired by Mycroft, with implementation patterns from Home Assistant's voice pipeline, avoiding the complexity of all three systems.**

**Why:**
- All three systems use event-driven coordination (proven pattern)
- All three struggle with complexity for simple use cases
- Radio assistant is simpler than general voice assistants
- Can adopt patterns without adopting full implementations

---

## Comparative Analysis

### Mycroft AI

**Architecture**: Multi-process WebSocket message bus

**Strengths:**
- ✅ Proven event-driven pattern (5+ years production)
- ✅ Clean component separation
- ✅ Excellent audio pipeline architecture
- ✅ Session/context management built-in
- ✅ Multi-turn conversation support

**Weaknesses:**
- ❌ Zero security (open WebSocket, known RCE vulnerabilities)
- ❌ Heavy resource usage (~320MB, 5+ processes)
- ❌ Complex deployment (multiple services)
- ❌ Cloud backend dependencies
- ❌ Overkill for single-purpose device

**Key Takeaway**: **Pattern is excellent, implementation is too heavy**

---

### Home Assistant Voice Pipeline

**Architecture**: Async event-driven pipeline with HomeAssistantCore bus

**Strengths:**
- ✅ Lightweight async implementation
- ✅ Single-process architecture
- ✅ Configurable pipeline stages
- ✅ Works offline
- ✅ Integration-friendly (REST API, WebSocket, MQTT)

**Weaknesses:**
- ❌ Tightly coupled to Home Assistant ecosystem
- ❌ Requires full HA installation
- ❌ Smart home focused, not general purpose
- ❌ Complex configuration

**Key Takeaway**: **Good async patterns, but ecosystem lock-in**

---

### Rhasspy

**Architecture**: Modular services with MQTT/HTTP coordination

**Strengths:**
- ✅ Offline-first design
- ✅ Modular architecture (swap components)
- ✅ Excellent wake word detection (Porcupine, Precise, Snowboy)
- ✅ Privacy-focused
- ✅ Lightweight per-component

**Weaknesses:**
- ❌ MQTT dependency (another service to manage)
- ❌ Multiple services to coordinate
- ❌ Configuration complexity
- ❌ Project maintenance uncertain

**Key Takeaway**: **Great for privacy, but too many moving parts**

---

## Common Patterns Across All Three

### 1. Event-Driven Coordination

**All three use events/messages for component communication:**

```
Mycroft:        WebSocket MessageBus
Home Assistant: EventBus (in-process)
Rhasspy:        MQTT topics

All follow: Component emits event → Bus dispatches → Listeners react
```

**Lesson**: Event-driven is the proven pattern for voice assistants

---

### 2. Audio Pipeline Structure

**All three have similar pipeline:**

```
Audio Input → Voice Detection → Wake Word → STT → Intent → Action → TTS → Audio Output
```

**Variations:**
- Mycroft: Wake word = callsign analog
- Home Assistant: Configurable stages
- Rhasspy: Modular, swappable components

**Lesson**: Pipeline structure is well-established

---

### 3. Async I/O

**All three support concurrent input/output:**

```
Mycroft:        Multi-process (implicit concurrency)
Home Assistant: asyncio coroutines
Rhasspy:        Separate services (process-level concurrency)
```

**Lesson**: Must support simultaneous listening and responding

---

### 4. State Management

**All three track conversation state:**

```
Mycroft:        ContextManager + SessionManager (5 min timeout)
Home Assistant: conversation.process() with conversation_id
Rhasspy:        Session management per service
```

**Lesson**: Timeout-based sessions are standard (5 minutes typical)

---

## Applicability to Radio Voice Assistant

### What Maps Well from All Three

| Feature | Mycroft | Home Assistant | Rhasspy | Radio Needs |
|---------|---------|---------------|---------|-------------|
| **Event coordination** | WebSocket bus | EventBus | MQTT | **AsyncEventBus** |
| **Audio pipeline** | Proven structure | Configurable | Modular | **Adopt Mycroft's** |
| **Wake word** | Energy + plugin | VAD plugin | Excellent options | **→ Callsign detection** |
| **STT** | Cloud/local | Whisper | Many options | **Whisper (already using)** |
| **Multi-turn** | converse() | conversation_id | Session-based | **Context manager** |
| **Async I/O** | Multi-process | asyncio | Separate services | **asyncio** |
| **Offline** | Partial | Yes | Yes | **Required** |

### What Doesn't Map

| Feature | Why Not Needed |
|---------|----------------|
| Skills marketplace | Fixed radio functionality |
| Smart home integration | Radio-only device |
| Multi-language | English only (HAM radio standard) |
| Cloud backend | Must work offline in field |
| MQTT broker | Unnecessary complexity for embedded |
| REST API | No external clients needed |
| WebSocket server | Security risk, no remote control needed |

---

## Recommended Architecture

### Synthesis of Best Practices

**Take from Mycroft:**
- Event-driven message bus pattern
- Audio pipeline structure (ring buffer, VAD, session timeout)
- Component lifecycle (initialize, handle, stop, shutdown)
- Context/session management

**Take from Home Assistant:**
- Lightweight async implementation (not multi-process)
- In-process event bus (not WebSocket)
- Configurable pipeline stages
- Simple deployment

**Take from Rhasspy:**
- Offline-first philosophy
- No cloud dependencies
- Privacy-conscious design
- Modular thinking (components as plugins)

**Avoid from all three:**
- Multi-process complexity (Mycroft)
- Ecosystem lock-in (Home Assistant)
- MQTT dependency (Rhasspy)
- WebSocket security issues (Mycroft)
- Complex configuration (all three)

---

## Proposed Architecture

### Core Design Principles

1. **Event-driven** (like all three)
2. **Async single-process** (like Home Assistant)
3. **Offline-first** (like Rhasspy)
4. **Secure by default** (unlike Mycroft)
5. **Simple deployment** (unlike all three)
6. **Radio-specific** (unlike all three)

### Implementation

```python
class RadioAssistantService:
    """
    Event-driven async service for radio voice assistant.

    Inspired by:
    - Mycroft: Event coordination pattern
    - Home Assistant: Async implementation
    - Rhasspy: Offline-first philosophy
    """

    def __init__(self, config: AppConfig):
        # Event bus (in-process, not WebSocket like Mycroft)
        self.event_bus = AsyncEventBus()

        # Components (modular like Rhasspy, async like HA)
        self.audio = AudioStreamManager(self.event_bus)
        self.vad = StreamingVAD(self.event_bus)  # Like Mycroft's ResponsiveRecognizer
        self.transcription = WhisperSTT(self.event_bus)
        self.callsign = CallsignDetector(self.event_bus)  # Wake word analog
        self.conversation = ConversationManager(self.event_bus)  # Like Mycroft's context
        self.ptt = PTTCoordinator(self.event_bus)  # Radio-specific

    async def start(self):
        """Start all coroutines (like HA voice pipeline)."""
        await asyncio.gather(
            self.event_bus.dispatch_loop(),
            self.audio.input_loop(),
            self.audio.output_loop(),
            self.conversation.timeout_loop(),  # Like Mycroft's session timeout
        )
```

### Event Bus (Lightweight Mycroft-style)

```python
class AsyncEventBus:
    """
    In-process event bus inspired by Mycroft's MessageBus
    but implemented with async queues like Home Assistant.
    """

    def __init__(self):
        self.handlers: dict[str, list[Callable]] = defaultdict(list)
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=100)

    def on(self, event_type: str, handler: Callable):
        """Register handler (like Mycroft's add_event)."""
        self.handlers[event_type].append(handler)

    async def emit(self, event_type: str, data: dict):
        """Emit event (like Mycroft's bus.emit)."""
        await self.queue.put(Event(type=event_type, data=data))

    async def dispatch_loop(self):
        """Dispatch events to handlers (like HA's event processing)."""
        while True:
            event = await self.queue.get()
            for handler in self.handlers[event.type]:
                await handler(event.data)
```

### Event Flow (Mycroft-inspired)

```
Audio Input:
  emit('audio.chunk')  [Like Mycroft's recognizer_loop:audio_output]
      ↓
StreamingVAD:
  on('audio.chunk') → emit('vad.speech.complete')  [Like recognizer_loop:utterance]
      ↓
WhisperSTT:
  on('vad.speech.complete') → emit('stt.complete')
      ↓
CallsignDetector:
  on('stt.complete') → emit('callsign.detected')  [Like Mycroft's intent match]
      ↓
ConversationManager:
  on('callsign.detected') → emit('radio.respond')  [Like Mycroft's skill response]
      ↓
PTTCoordinator:
  on('radio.respond') → emit('ptt.complete')  [Like Mycroft's speak complete]
```

---

## Key Differences from Existing Systems

### Vs Mycroft

| Aspect | Mycroft | Radio Assistant |
|--------|---------|----------------|
| IPC | WebSocket | In-process queues |
| Processes | 5+ | 1 |
| Memory | ~320MB | ~150MB |
| Security | Open port | No network |
| Deployment | Complex | Single service |
| Wake detection | Generic wake word | Callsign-specific |
| Use case | General assistant | Radio-specific |

### Vs Home Assistant

| Aspect | Home Assistant | Radio Assistant |
|--------|---------------|----------------|
| Ecosystem | Smart home | Radio only |
| Installation | Full HA core | Standalone |
| Configuration | YAML + UI | Python config |
| Integrations | 2000+ | Radio hardware only |
| Use case | Home automation | Radio communications |

### Vs Rhasspy

| Aspect | Rhasspy | Radio Assistant |
|--------|---------|----------------|
| Coordination | MQTT | In-process events |
| Services | Multiple | Single process |
| Configuration | Complex | Simple |
| Modularity | High | Moderate |
| Use case | Privacy-focused assistant | Radio assistant |

---

## Implementation Roadmap

### Phase 1: Event Bus Foundation (1 day)

**Goal**: Working async event bus with tests

```python
- Implement AsyncEventBus
- Event registration and dispatch
- Error isolation (handler failures don't crash bus)
- Queue backpressure handling
- Tests for event flow
```

**Validation**: Event dispatch <1ms, no message loss

---

### Phase 2: Component Refactor (2 days)

**Goal**: Migrate existing components to event-driven

**Refactor components** (inspired by Mycroft's component structure):

1. **AudioStreamManager** (from AudioInterface)
   - Continuous capture (like Mycroft's MutableMicrophone)
   - Ring buffer (like Mycroft's cyclic buffer)
   - Emit `audio.chunk` events

2. **StreamingVAD** (from VADDetector)
   - Chunk-by-chunk processing (like Mycroft's ResponsiveRecognizer)
   - Emit `vad.speech.start` and `vad.speech.complete`
   - State machine: listening → collecting → complete

3. **WhisperSTT** (minor changes to TranscriptionEngine)
   - Listen for `vad.speech.complete`
   - Emit `stt.complete`

4. **CallsignDetector** (minor changes)
   - Listen for `stt.complete`
   - Emit `callsign.detected`

5. **ConversationManager** (new, inspired by Mycroft's SessionManager)
   - Listen for `callsign.detected`
   - Track session state (5 min timeout like Mycroft)
   - Context management (like Mycroft's ContextManager)
   - Emit `radio.respond`

6. **PTTCoordinator** (from PTTController)
   - Listen for `radio.respond`
   - TX/RX coordination (radio-specific, not in any system)
   - Emit `ptt.complete`

**Validation**: All 225 existing tests still pass

---

### Phase 3: Streaming Integration (1 day)

**Goal**: Test continuous operation

```python
- Implement streaming test harness
- Test multiple sequential transmissions
- Test back-to-back processing
- Test session timeout
- Test error recovery
```

**Validation**: 24-hour stability test, <200MB memory

---

### Phase 4: Hardware Validation (1 day)

**Goal**: Validate on Raspberry Pi + radio

```python
- Deploy to hardware
- Test with real radio audio
- Measure latency (<500ms target)
- Document performance characteristics
```

**Validation**: Works on actual radio hardware

---

## Success Metrics

### Technical Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| Memory usage | <200MB | Raspberry Pi constraint |
| Event dispatch latency | <1ms | Mycroft: 10-50ms (WebSocket overhead) |
| End-to-end latency | <500ms | Radio communication standard |
| Uptime | 24+ hours | Continuous operation requirement |
| WER (clear audio) | <0.05 | Current: 0.0 |
| WER (noisy audio) | <0.15 | Current: 0.125 |

### Architecture Metrics

| Metric | Target | Comparison |
|--------|--------|------------|
| Processes | 1 | Mycroft: 5+, Rhasspy: 3+ |
| Network ports | 0 | Mycroft: 8181, HA: varies |
| External services | 0 | Rhasspy: MQTT, Mycroft: backend |
| Configuration files | 1 | Rhasspy: many, HA: many |
| Deployment steps | 1 command | All three: complex |

---

## Risks and Mitigations

### Risk 1: Event-driven complexity

**Risk**: More complex than current direct method calls

**Mitigation**:
- Clear event flow documentation
- Type-safe events (Pydantic models)
- Visual diagrams
- Excellent logging

**Evidence**: All three systems use events successfully

---

### Risk 2: Performance overhead

**Risk**: Event dispatch adds latency

**Mitigation**:
- Target <1ms dispatch (vs Mycroft's 10-50ms)
- In-process queues (no serialization)
- Benchmark and profile
- Monitor queue depth

**Evidence**: Home Assistant proves async is fast enough

---

### Risk 3: Not learning from mistakes

**Risk**: Repeat complexity mistakes of existing systems

**Mitigation**:
- Explicitly document "what NOT to do"
- Keep architecture simple (YAGNI)
- Regular complexity audits
- Resist feature creep

**Evidence**: All three systems grew complex over time

---

## Decision Matrix

### Should we adopt event-driven architecture?

| Factor | Weight | Score | Weighted |
|--------|--------|-------|----------|
| Proven pattern (all three use it) | 10 | 10 | 100 |
| Suitable for streaming | 10 | 10 | 100 |
| Testability improvement | 8 | 9 | 72 |
| Component isolation | 8 | 10 | 80 |
| Complexity increase | -5 | 7 | -35 |
| Performance overhead | -3 | 8 | -24 |
| **Total** | | | **293/400** |

**Recommendation**: **YES** - Benefits clearly outweigh costs

---

### Which implementation approach?

| Approach | Pros | Cons | Score |
|----------|------|------|-------|
| **Mycroft-style (WebSocket)** | Battle-tested, well-documented | Security issues, heavy, complex | 5/10 |
| **Home Assistant-style (async)** | Lightweight, fast, proven | Ecosystem coupling | 8/10 |
| **Rhasspy-style (MQTT)** | Modular, privacy-focused | Extra dependencies, complex | 6/10 |
| **Hybrid (recommended)** | Best of all, radio-optimized | Custom implementation | **9/10** |

**Recommendation**: **Hybrid approach** - Mycroft patterns + Home Assistant async + Rhasspy philosophy

---

## Conclusion

### Research Findings

After comprehensive research into three major voice assistant architectures:

1. **Event-driven coordination is the proven pattern** - All three successful systems use it
2. **Multiple implementation approaches exist** - WebSocket, async queues, MQTT all work
3. **Simplicity is critical for embedded** - All three suffer from complexity
4. **Radio has unique needs** - PTT coordination not addressed by any system
5. **Offline operation is achievable** - Rhasspy and parts of HA prove it works

### Final Recommendation

**Adopt event-driven async architecture with these principles:**

1. **Pattern from Mycroft** - Event bus, component lifecycle, session management
2. **Implementation from Home Assistant** - Async coroutines, single process
3. **Philosophy from Rhasspy** - Offline-first, privacy-conscious, modular
4. **Additions for Radio** - PTT coordination, squelch detection, TX/RX mutual exclusion

**Why this is right:**

- ✅ Proven pattern (used by all three successful systems)
- ✅ Suitable for embedded (unlike Mycroft's multi-process)
- ✅ Testable (event mocking easier than hardware mocking)
- ✅ Maintainable (clear component boundaries)
- ✅ Secure (no open ports unlike Mycroft)
- ✅ Simple (no MQTT unlike Rhasspy)
- ✅ Radio-optimized (PTT coordination, offline-first)

**Estimated effort**: 1 week (5 days)
**Risk**: Low (proven patterns, gradual migration)
**Benefit**: Production-ready streaming architecture

---

## Next Steps

1. **Review this research** - Validate conclusions
2. **Approve architecture** - Go/no-go decision
3. **Implement Phase 1** - Event bus foundation (1 day)
4. **Refactor Phase 2** - Component migration (2 days)
5. **Test Phase 3** - Streaming integration (1 day)
6. **Validate Phase 4** - Hardware testing (1 day)

**Expected outcome**: Radio voice assistant with production-ready streaming architecture, based on proven patterns from three successful voice assistant systems.

---

## Research References

### Primary Research Documents

- **MYCROFT_ARCHITECTURE_RESEARCH.md** - Detailed Mycroft analysis (32KB)
- **HOME_ASSISTANT_VOICE_ARCHITECTURE.md** - Home Assistant voice pipeline (21KB)
- **rhasspy_architecture_research.md** - Rhasspy modular architecture (33KB)
- **ARCHITECTURE_RECOMMENDATIONS.md** - Implementation recommendations (23KB)

### Supporting Documents

- **PHASE2_PLANNING_SUMMARY.md** - Overall Phase 2 plan
- **SERVICE_MODE_GAPS.md** - Current architecture gaps
- **VOICE_ASSISTANT_RESEARCH.md** - Initial research outline

### External References

- [Mycroft AI Documentation](https://mycroft-ai.gitbook.io/docs)
- [Home Assistant Voice](https://www.home-assistant.io/voice_control/)
- [Rhasspy Documentation](https://rhasspy.readthedocs.io/)
- [asyncio Documentation](https://docs.python.org/3/library/asyncio.html)

---

*Research completed: 2025-12-28*
*Status: Ready for implementation decision*
*Phase: 2A - Validation & Research Complete*
