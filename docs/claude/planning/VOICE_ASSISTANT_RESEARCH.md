# Voice Assistant Architecture Research

**Created**: 2025-12-27
**Goal**: Research existing open-source voice assistants to learn from proven architectures before building our streaming service

## Why Research Existing Solutions?

We're about to build:
- Continuous audio streaming
- Async processing pipeline
- Wake word detection (similar to our callsign detection)
- Audio I/O coordination
- Long-running service stability

**Don't reinvent the wheel.** Mature voice assistants have solved these problems.

---

## Projects to Research

### 1. **Rhasspy** (Recommended - Most Relevant)
**URL**: https://github.com/rhasspy/rhasspy

**Why relevant:**
- Open source voice assistant for home automation
- Modular architecture with pluggable components
- Wake word detection → STT → Intent → TTS → Audio output
- **Very similar to our use case:** Listen → Detect trigger → Process → Respond

**What to learn:**
- How they handle continuous audio streaming
- Wake word detection architecture (similar to our callsign detection)
- Async processing pipeline design
- Audio buffer management
- PyAudio/sounddevice integration patterns
- Long-running service stability

**Key files to study:**
- Audio input handling
- Wake word detection service
- Pipeline orchestration
- State management

---

### 2. **Mycroft AI**
**URL**: https://github.com/MycroftAI/mycroft-core

**Why relevant:**
- Complete open-source voice assistant
- Production-ready, deployed on devices
- Message bus architecture for component communication
- Skills system (extensible)

**What to learn:**
- Service architecture (messagebus pattern)
- Audio client design
- State management across intents
- Resource management for long-running operation
- Error recovery patterns

**Key concepts:**
- `mycroft-core` - main service orchestrator
- `mycroft.audio` - audio handling
- Messagebus for inter-component communication
- Skills framework

---

### 3. **Home Assistant Voice Pipeline**
**URL**: https://github.com/home-assistant/core (voice pipeline component)

**Why relevant:**
- Modern async Python architecture (asyncio)
- Integration with various STT/TTS engines
- Pipeline concept: wake → STT → intent → TTS
- Built for reliability and long-running operation

**What to learn:**
- Async/await patterns for audio processing
- Pipeline state machine design
- Integration patterns for external services (Whisper, etc.)
- Configuration and plugin architecture

**Key files:**
```
homeassistant/components/voice_assistant/
homeassistant/components/stt/
homeassistant/components/wake_word/
```

---

### 4. **Kalliope**
**URL**: https://github.com/kalliope-project/kalliope

**Why relevant:**
- Voice-controlled assistant
- Trigger word → STT → Action → TTS
- Simple, focused architecture
- Good for learning fundamentals

**What to learn:**
- Simple trigger word detection
- Order of operations in voice pipeline
- How they coordinate audio I/O

---

### 5. **PyAudio / SoundDevice Examples**
**Not a voice assistant, but critical for audio I/O**

**URLs:**
- https://github.com/spatialaudio/python-sounddevice
- https://people.csail.mit.edu/hubert/pyaudio/

**What to learn:**
- Callback-based audio streaming
- Non-blocking audio I/O
- Buffer management
- Ring buffer patterns
- Handling audio device errors

---

## Specific Questions to Answer

### Audio Streaming
1. **How do they capture continuous audio?**
   - Callback-based or polling?
   - What chunk size?
   - Ring buffer or queue?

2. **How do they handle audio I/O coordination?**
   - Can they listen while speaking?
   - How do they prevent feedback?
   - PTT/VOX coordination patterns?

3. **What buffer sizes work well?**
   - For 16kHz audio
   - Latency vs. reliability tradeoff

### Pipeline Architecture
1. **How is the processing pipeline structured?**
   - Sync or async?
   - Thread-based or async/await?
   - Message passing or direct calls?

2. **How do they handle state?**
   - State machine patterns?
   - Conversation context tracking?
   - Timeout handling?

3. **How do they coordinate components?**
   - Queue-based?
   - Event-driven?
   - Message bus?

### Wake Word / Trigger Detection
1. **How do they detect wake words in stream?**
   - Sliding window?
   - Separate model or keyword spotting?
   - False positive handling?

2. **What happens after trigger detected?**
   - Start recording? Already recording?
   - How to detect end of utterance?
   - Timeout patterns?

### Reliability
1. **How do they handle long-running operation?**
   - Memory management?
   - Resource cleanup?
   - Error recovery?

2. **What error handling patterns?**
   - Audio device disconnect?
   - Model timeout/hang?
   - Queue overflow?

3. **How do they test reliability?**
   - Long-running tests?
   - Stress tests?
   - Memory leak detection?

---

## Research Methodology

### Phase 1: Quick Survey (2-3 hours)
For each project:
1. Read architecture documentation
2. Find main entry point / service orchestrator
3. Trace audio input → processing → output flow
4. Note interesting patterns/decisions
5. Save links to specific files/classes

### Phase 2: Deep Dive (4-6 hours)
Pick 1-2 most relevant projects (recommend Rhasspy + Home Assistant):
1. Clone repository
2. Read and annotate key source files
3. Run locally to understand behavior
4. Extract code patterns we can adapt
5. Document findings

### Phase 3: Synthesis (2-3 hours)
1. Compare approaches across projects
2. Identify best practices
3. Note pitfalls to avoid
4. Create recommended architecture doc
5. List code we can adapt/borrow (with attribution)

**Total: 8-12 hours of research**

---

## Output: Research Findings Document

Create: `docs/claude/research/VOICE_ASSISTANT_FINDINGS.md`

**Structure:**
```markdown
# Voice Assistant Architecture Findings

## Summary
Key learnings and recommendations

## Audio Streaming Patterns
How they handle continuous audio

## Pipeline Architecture Options
Comparison of approaches

## Wake Word Detection
Patterns and best practices

## Recommended Approach for Radio Assistant
Based on research, here's what we should do...

## Code References
Specific files/classes to study or adapt
```

---

## Why This Matters

**Without research:**
- ❌ Might reinvent solved problems
- ❌ Could miss critical edge cases
- ❌ May choose suboptimal architecture
- ❌ Waste time debugging issues others solved

**With research:**
- ✅ Learn from production-tested code
- ✅ Avoid known pitfalls
- ✅ Choose proven patterns
- ✅ Stand on shoulders of giants

**Time investment:**
- Research: 8-12 hours
- Building from scratch without research: 40-60 hours + debugging
- **ROI: Saves 30-50 hours of trial and error**

---

## Research Priority

**Before implementing StreamingVAD and async service:**
1. Research Rhasspy wake word detection (most relevant)
2. Research Home Assistant voice pipeline (async patterns)
3. Research PyAudio/sounddevice streaming examples (technical details)

**Then implement our solution informed by best practices.**

---

## Next Steps

1. **Start research** (recommend Rhasspy first)
2. **Document findings** as we go
3. **Create architecture proposal** based on research
4. **Review with user** before implementing
5. **Implement with confidence** knowing we're following proven patterns

---

## Alternative: Skip Research?

**Could we skip this and just build?**
- Yes, but high risk of architectural mistakes
- Voice pipeline is deceptively complex
- Easy to build something that "works" in simple tests but fails in production
- Research = insurance against costly rewrites

**Recommendation: Do the research.** It's worth 1-2 days to de-risk the next 2-3 weeks of implementation.
