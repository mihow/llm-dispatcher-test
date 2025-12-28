# Phase 2 Planning Summary

**Created**: 2025-12-27
**Status**: Planning / Research Phase
**Current State**: Phase 1 "Marco Polo" complete and merged ✅

---

## Current Status: What We Have

**Phase 1 Complete:**
- ✅ All 6 components implemented (Audio, VAD, Transcription, Callsign, PTT, Pipeline)
- ✅ 225/225 tests passing (100%)
- ✅ TTS-generated test audio with ground truth
- ✅ CI/CD on Python 3.12
- ✅ Batch processing works end-to-end

**What works:**
```python
# Load audio file
audio = load_wav("transmission.wav")

# Process it
result = assistant.process_audio(audio)

# Get response
if result:
    print("Callsign detected, response triggered!")
```

---

## Critical Gaps Identified

### Gap 1: Tests Don't Validate Correctness
**Problem:** Tests check "did it return a string?" not "is the string correct?"

**Evidence:**
```python
# Current test (meaningless)
assert isinstance(result.text, str)

# We have ground truth but ignore it!
ground_truth = "This is WSJJ659 calling"
# ... never compared to result.text
```

**Impact:** We don't know if transcription/detection actually work

**Documented in:** `IMMEDIATE_VALIDATION_PLAN.md`

---

### Gap 2: Not Built for Continuous Operation
**Problem:** Tests batch processing, but deployment needs streaming service

**What we test:**
- Load file → process → done

**What's needed:**
- Continuous audio capture
- Stream processing (chunk by chunk)
- Async I/O (listen while responding)
- State across transmissions
- Long-running stability

**Impact:** Architecture might not work in real deployment

**Documented in:** `SERVICE_MODE_GAPS.md`

---

### Gap 3: Haven't Learned from Existing Solutions
**Problem:** About to build complex streaming architecture without researching proven solutions

**Risk:** Reinventing the wheel, missing critical patterns, costly rewrites

**Documented in:** `VOICE_ASSISTANT_RESEARCH.md`

---

## Recommended Path Forward

### Phase 2A: Validation & Research (1-2 weeks)

**Week 1: Make Tests Actually Test Things**
- [ ] Add WER/CER validation (use ground truth!)
- [ ] Measure actual transcription accuracy
- [ ] Add E2E tests with real assertions
- [ ] Document baseline performance

**Deliverable:** Know exactly how accurate the system is

---

**Week 2: Research Existing Architectures**
- [ ] Study Rhasspy (wake word detection, streaming)
- [ ] Study Home Assistant (async voice pipeline)
- [ ] Study PyAudio patterns (audio I/O)
- [ ] Document findings and recommendations
- [ ] Create architecture proposal

**Deliverable:** Informed architecture design for streaming service

---

### Phase 2B: Streaming Service Implementation (2-3 weeks)

**Based on research findings:**
- [ ] Implement StreamingVAD (chunk-by-chunk detection)
- [ ] Build async service architecture
- [ ] Add continuous audio I/O
- [ ] Implement state management
- [ ] Add streaming integration tests
- [ ] Test long-running stability

**Deliverable:** Production-ready streaming service

---

### Phase 2C: Hardware Validation (1 week)

**Deploy to Raspberry Pi + Baofeng radio:**
- [ ] Test on actual hardware
- [ ] Measure real performance
- [ ] Collect real radio audio samples
- [ ] Test in real-world conditions
- [ ] Document results

**Deliverable:** Hardware-validated proof of concept

---

## Planning Documents Overview

### 1. `IMMEDIATE_VALIDATION_PLAN.md`
**Focus:** Fix current tests to validate correctness

**Key points:**
- Add WER/CER metrics with jiwer
- Use ground truth files we already have
- Validate transcription accuracy
- Add real E2E assertions

**Timeline:** 2-3 days
**Dependencies:** None - can start immediately

---

### 2. `SERVICE_MODE_GAPS.md`
**Focus:** What's needed for continuous service operation

**Key points:**
- StreamingVAD for chunk-by-chunk processing
- Async architecture for concurrent I/O
- State management across transmissions
- PTT coordination
- Long-running stability

**Timeline:** 2-3 weeks
**Dependencies:** Research findings

---

### 3. `VOICE_ASSISTANT_RESEARCH.md`
**Focus:** Learn from existing voice assistants

**Key projects:**
- Rhasspy (most relevant - wake word detection)
- Home Assistant (async patterns)
- Mycroft AI (service architecture)
- PyAudio/sounddevice (audio streaming)

**Timeline:** 1-2 weeks
**Dependencies:** None - can start immediately

---

### 4. `PROOF_OF_CONCEPT_PLAN.md`
**Focus:** Overall vision for what makes a valid PoC

**Key criteria:**
- Real-world validation with actual radio audio
- Performance benchmarks on Raspberry Pi
- Documented failure modes
- Success metrics defined

**Timeline:** Full Phase 2 (4-6 weeks)
**Dependencies:** All of the above

---

## Decision Points

### Question 1: Research First or Implement First?

**Option A: Research → Implement**
- Pros: Informed decisions, avoid costly rewrites
- Cons: Delays implementation by 1-2 weeks
- **Recommendation:** Do the research

**Option B: Implement → Research → Refactor**
- Pros: Faster to "working code"
- Cons: High risk of architectural mistakes, rewrites
- **Not recommended**

---

### Question 2: Validate First or Build Streaming First?

**Option A: Add WER validation → Research → Build streaming**
- Pros: Know what we have works before building more
- Cons: None really
- **Recommendation:** This order

**Option B: Build streaming → Validate later**
- Pros: Faster to "complete" system
- Cons: Building on unvalidated foundation
- **Not recommended**

---

### Question 3: Hardware Testing When?

**Option A: After streaming service works**
- Pros: Test complete system on hardware
- Cons: Delays hardware validation
- **Recommendation:** This approach

**Option B: Hardware test current batch system first**
- Pros: Earlier hardware feedback
- Cons: Will need hardware testing again after streaming refactor
- **Not optimal**

---

## Recommended Timeline

### Immediate (Next Session):
1. ✅ Create planning branch
2. ✅ Add all planning documents
3. ✅ Create PR for planning review
4. ⏸️  Pause for user review and decision

### Week 1: Validation
- Add WER/CER validation
- Measure baseline accuracy
- Document performance

### Week 2: Research
- Study existing voice assistants
- Document findings
- Propose architecture

### Weeks 3-5: Implementation
- Build streaming service (informed by research)
- Test long-running stability
- Document service architecture

### Week 6: Hardware
- Deploy to Raspberry Pi
- Test with real radio
- Validate proof of concept

**Total: 6 weeks to production-ready PoC**

---

## Success Criteria

### Phase 2A Success (Validation & Research)
- ✅ Know exact WER/CER for transcription
- ✅ Have documented architecture proposal
- ✅ Confidence in approach based on research

### Phase 2B Success (Streaming Service)
- ✅ Service runs continuously without crashes
- ✅ Can handle back-to-back transmissions
- ✅ Memory stable over 24+ hours
- ✅ Concurrent I/O works correctly

### Phase 2C Success (Hardware)
- ✅ Works on Raspberry Pi with acceptable performance
- ✅ Integrates with real radio hardware
- ✅ Handles real radio audio conditions
- ✅ Documented limitations and failure modes

---

## Open Questions for Discussion

1. **Research depth:** Quick survey (1 week) or deep dive (2 weeks)?
2. **Which voice assistant to focus on:** Rhasspy, Home Assistant, or both?
3. **Testing strategy:** How much streaming testing before hardware?
4. **Hardware access:** Do we have Raspberry Pi + radio available?
5. **Timeline:** Is 6 weeks realistic given other constraints?

---

## Next Steps

**Immediate:**
1. Review planning documents
2. Discuss approach and timeline
3. Make go/no-go decision on research phase
4. Create PR for planning docs

**If approved:**
1. Start with WER validation (quick win, 1 day)
2. Begin voice assistant research (1-2 weeks)
3. Proceed with streaming implementation

**Output of this phase:**
- Validated current functionality
- Informed architecture design
- Confidence in approach
- Clear path to production-ready PoC

---

*Last updated: 2025-12-27*
*Status: Awaiting review and decision*
