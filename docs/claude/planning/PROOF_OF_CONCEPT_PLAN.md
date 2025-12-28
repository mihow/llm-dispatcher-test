# Proof of Concept Plan: Radio Assistant Validation

**Created**: 2025-12-27
**Status**: Planning
**Goal**: Transform from "tests pass" to "validated proof of concept"

## Current State: Honest Assessment

### What We Have ✅
- All components implemented and integrated
- 225 tests passing
- TTS-generated synthetic speech audio
- Full pipeline runs without crashing

### What We're NOT Validating ❌
1. **Transcription accuracy** - Ground truth files exist but are ignored
2. **Real radio audio** - Using clean TTS, not actual radio transmissions
3. **Performance targets** - No benchmarks for Raspberry Pi deployment
4. **Real-world conditions** - No static, fading, interference, squelch tails
5. **False positive/negative rates** - No metrics on detection reliability
6. **Hardware feasibility** - Never tested on actual Pi or with real radio

### The Core Problem
**We're testing "does it run?" not "does it work?"**

---

## What "Proof of Concept" Should Prove

A valuable PoC for a ham radio voice assistant must demonstrate:

1. **It can reliably detect a specific callsign from radio audio** (not just TTS)
2. **It distinguishes target callsign from other callsigns and noise**
3. **Transcription accuracy is sufficient for the use case** (measured WER)
4. **Performance is acceptable on target hardware** (Raspberry Pi)
5. **End-to-end latency meets requirements** (<3s from speech end to response)
6. **We understand the failure modes** (when does it break and why?)

---

## Critical Path: Making This Real

### Phase A: Validate Core Functionality (CRITICAL)

**Priority 1: Add Transcription Accuracy Testing**

**Current state:**
```python
# What we do now - USELESS
assert isinstance(result.text, str)
assert result.duration_ms > 0
```

**What we should do:**
```python
# Actually validate correctness
from jiwer import wer, cer

result = engine.transcribe(audio)
word_error_rate = wer(ground_truth, result.text)
assert word_error_rate < 0.15, f"WER {word_error_rate:.2%} exceeds 15% threshold"
```

**Implementation:**
- Add `jiwer` dependency for WER/CER calculation
- Update integration tests to compare against ground truth
- Set realistic thresholds based on Whisper base model performance
- Track metrics: WER, CER, callsign detection accuracy

**File**: `tests/integration/test_transcription_accuracy.py`

**Acceptance Criteria:**
- [ ] WER < 15% on clean TTS audio
- [ ] Callsign detection: 100% on clear audio, >95% on noisy audio
- [ ] Tests fail when transcription is wrong (not just when code crashes)

---

**Priority 2: Get Real Radio Audio Samples**

**Why this matters:**
- TTS audio has perfect pronunciation, no static, no fading
- Real radio has: squelch tails, static bursts, heterodynes, fading, compression artifacts
- This is THE critical validation gap

**What we need:**
1. **Clean recordings** (3-5 samples)
   - Clear callsign transmissions
   - Good signal strength
   - Typical radio compression

2. **Noisy recordings** (3-5 samples)
   - Weak signals (S3-S5)
   - Background static
   - Fading/QSB

3. **Edge cases** (2-3 samples)
   - Overlapping transmissions
   - Very weak signals
   - Heavy QRM/QRN

**Sources:**
- Record from actual Baofeng UV-5R
- Find ham radio audio samples online (websdr.org, archive.org)
- Ask ham radio community for test samples
- Use radio recording software (SDR#, GQRX)

**Format requirements:**
- 16kHz mono WAV (match current pipeline)
- Include ground truth transcriptions
- Document conditions (signal strength, noise level)

**Files to create:**
```
tests/audio/real_radio/
├── clear/
│   ├── wsjj659_s9.wav
│   ├── wsjj659_s9.txt
│   └── ...
├── noisy/
│   ├── wsjj659_s5_static.wav
│   ├── wsjj659_s5_static.txt
│   └── ...
└── edge_cases/
    ├── weak_signal_s3.wav
    └── ...
```

**Acceptance Criteria:**
- [ ] 10+ real radio audio samples collected
- [ ] Mix of clear/noisy/edge cases
- [ ] Ground truth transcriptions documented
- [ ] Signal conditions documented (S-meter, noise level)

---

**Priority 3: Benchmark Performance on Raspberry Pi**

**Current gap:**
- Never tested on actual Pi
- No idea if it meets real-time requirements
- Unknown memory/CPU usage

**Target hardware:**
- Raspberry Pi 4 (4GB RAM) - minimum viable
- Raspberry Pi 5 (8GB RAM) - preferred

**Metrics to measure:**
1. **Latency breakdown:**
   - VAD detection time
   - Transcription time (critical bottleneck)
   - Callsign detection time
   - Total end-to-end latency

2. **Resource usage:**
   - CPU utilization (should be <80% sustained)
   - Memory footprint (should be <2GB)
   - Model loading time

3. **Real-time performance:**
   - Can it process audio faster than real-time?
   - What's the queue backlog under load?

**Implementation:**
```python
# Add performance benchmarks
def test_realtime_performance_on_pi():
    """Verify can process audio faster than real-time."""
    audio_duration = 5.0  # seconds

    start = time.time()
    result = engine.transcribe(audio)
    elapsed = time.time() - start

    rtf = elapsed / audio_duration  # Real-Time Factor
    assert rtf < 0.9, f"RTF {rtf:.2f} too slow for real-time (should be <0.9)"
```

**File**: `tests/benchmarks/test_pi_performance.py`

**Acceptance Criteria:**
- [ ] Transcription RTF < 0.9 on Pi 4
- [ ] Total latency < 3s for 2s audio
- [ ] Memory usage < 2GB
- [ ] CPU usage < 80% sustained

---

### Phase B: Real-World Validation

**Priority 4: End-to-End Demo with Real Radio**

**Setup:**
1. Raspberry Pi 4/5
2. Baofeng UV-5R (or similar)
3. Audio cable (3.5mm to Kenwood connector)
4. Test scenario script

**Demo scenario:**
```
1. Record incoming: "This is WSJJ659 calling for a radio check"
2. System detects callsign WSJJ659
3. System transcribes message
4. System triggers VOX/PTT
5. System plays response audio
6. Document: What worked? What failed? Latency?
```

**Validation points:**
- [ ] Callsign detected correctly
- [ ] No false positives from other callsigns
- [ ] No false triggers from noise/static
- [ ] PTT activation timing correct
- [ ] Total response time acceptable
- [ ] System stable over 30+ minute test

**Document in**: `docs/claude/HARDWARE_DEMO_RESULTS.md`

---

**Priority 5: Failure Mode Analysis**

**We need to understand when/why it fails:**

1. **False Positives:**
   - What causes incorrect callsign detection?
   - How often does noise trigger speech detection?
   - Do phonetically similar callsigns confuse it?

2. **False Negatives:**
   - What signal strength is the minimum threshold?
   - Does static prevent callsign detection?
   - Are rapid transmissions missed?

3. **Transcription Errors:**
   - What types of errors occur? (phonetic, similar words)
   - Does callsign detection work even with transcription errors?
   - What's the degradation curve with SNR?

**File**: `docs/claude/FAILURE_ANALYSIS.md`

**Acceptance Criteria:**
- [ ] Documented failure modes with examples
- [ ] Minimum signal strength determined (S-meter)
- [ ] False positive/negative rates measured
- [ ] Mitigation strategies identified

---

### Phase C: Performance Optimization (If Needed)

**If Pi performance is inadequate:**

1. **Model optimization:**
   - Try `tiny` Whisper model (faster but less accurate)
   - Quantize to int8 (already using)
   - Consider distil-whisper models

2. **Pipeline optimization:**
   - Reduce VAD chunk size
   - Parallel processing where possible
   - Cache model in memory

3. **Hardware acceleration:**
   - Test on Pi 5 (better CPU)
   - Consider Coral TPU for inference
   - Optimize audio buffer sizes

---

## Revised Test Strategy

### Unit Tests (Keep but acknowledge limits)
- Purpose: Prevent regressions, smoke tests
- Don't pretend they validate functionality
- Remove overly-mocked tests that test nothing

### Integration Tests (Make them count)
**Before:**
```python
assert isinstance(result.text, str)  # Meaningless
```

**After:**
```python
assert wer(ground_truth, result.text) < threshold  # Validates accuracy
assert "WSJJ659" in result.text.upper()  # Validates callsign detected
```

### Real-World Tests (NEW - Most Important)
- Test with actual radio recordings
- Measure accuracy on representative audio
- Document what works and what doesn't
- This is what proves the concept

---

## Success Criteria for "Valid Proof of Concept"

### Minimum Viable PoC:
1. ✅ Transcription WER < 20% on real radio audio (clear signals)
2. ✅ Callsign detection: >95% accuracy on clear audio, >85% on noisy
3. ✅ False positive rate: <5% on noise/other callsigns
4. ✅ Runs on Raspberry Pi 4 with acceptable latency (<3s)
5. ✅ Successfully completes 10/10 demo scenarios
6. ✅ Failure modes documented and understood

### Stretch Goals:
- WER < 15% on noisy radio audio
- Callsign detection >90% even with weak signals (S3-S5)
- Runs on Pi 4 with <2s latency
- Handles multiple callsigns in conversation
- Graceful degradation with poor signals

---

## What This Proves (and Doesn't)

### If we hit success criteria, we prove:
✅ The approach is technically feasible
✅ Core functionality works with real radio audio
✅ Performance is acceptable on target hardware
✅ We understand the limitations and failure modes

### What this doesn't prove (Phase 2):
❌ Long-term reliability (needs extended testing)
❌ Intelligent response generation (needs LLM integration)
❌ Conversation context handling (needs state management)
❌ Multi-user scenarios (needs better callsign tracking)

---

## Next Actions (Priority Order)

1. **Add WER/CER validation to integration tests** (1-2 hours)
   - Install jiwer
   - Update test assertions
   - Set thresholds

2. **Acquire real radio audio samples** (1-2 days)
   - Record from Baofeng UV-5R
   - Find online samples
   - Create test dataset

3. **Test with real audio, measure accuracy** (2-4 hours)
   - Run pipeline on real samples
   - Calculate metrics
   - Document results

4. **Deploy to Raspberry Pi, benchmark performance** (4-6 hours)
   - Set up Pi environment
   - Run performance tests
   - Optimize if needed

5. **End-to-end hardware demo** (2-4 hours)
   - Set up radio connection
   - Run demo scenarios
   - Record results

6. **Document findings** (2-3 hours)
   - Write up what works
   - Write up what doesn't
   - Recommendations for Phase 2

**Total estimated effort: 2-3 days of focused work**

---

## The Honest Assessment

**Current state**: We have a well-structured codebase with components that integrate properly. Tests pass, but they're mostly validating "it doesn't crash" not "it works correctly."

**What we need**: Real-world validation with actual radio audio and hardware to prove this approach can work.

**The gap**: We're one good weekend of testing away from knowing if this is viable or if we need to pivot the approach.

**Bottom line**: Until we test with real radio audio and real hardware, we have a tech demo, not a proof of concept.
