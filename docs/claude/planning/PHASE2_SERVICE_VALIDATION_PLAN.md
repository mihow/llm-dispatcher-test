# Phase 2: Core Service Validation Plan

**Created:** 2025-12-28
**Scope:** Milestones 1-2 (Streaming Loop + Lifecycle)
**Timeline:** 4-6 days
**Priority:** Prove the `run()` loop and service lifecycle work

---

## Executive Summary

**Problem:** 225 tests passing, but ALL call `process_audio()` once with pre-loaded files. The `run()` loop (main.py:110-136) is NEVER tested.

**Goal:** Prove RadioAssistant works as a continuous service with streaming audio.

**Success:** When the `run()` loop is validated with 10+ streaming scenarios and service lifecycle is tested with 6+ error recovery scenarios.

---

## Current State

**What's Validated (Phase 1):**
- ✅ Component isolation (VAD, transcription, callsign, PTT) - 225 tests
- ✅ Single-shot `process_audio()` pipeline
- ✅ TTS audio transcription (2.08% avg WER)
- ✅ MockRadioInterface exists and can stream from files

**Critical Gaps (Phase 2 Target):**
- ❌ `run()` loop execution - NEVER tested
- ❌ Continuous streaming - chunk-by-chunk processing over time
- ❌ VAD buffering - speech spanning multiple chunks
- ❌ Service lifecycle - startup, shutdown, signal handling
- ❌ Error recovery - component failures during operation

---

## Milestone 1: Streaming Loop Validation

**Goal:** Prove the `run()` loop works with streaming audio chunks
**Effort:** 2-3 days
**Files:** `tests/service/test_streaming_loop.py` (new)

### Infrastructure to Build

**1. StreamSimulator Helper** (`tests/fixtures/stream_simulator.py`)

```python
class StreamSimulator:
    """Simulates streaming operation for testing run() loop."""

    def __init__(self, audio_file: Path, config: AppConfig):
        self.mock_radio = MockRadioInterface(audio_file)
        self.assistant = RadioAssistant(config, audio_interface=self.mock_radio)
        self.results = {'transcriptions': [], 'detections': [], 'responses': []}

    def run_until_complete(self, timeout_sec: float = 10.0) -> dict:
        """Run assistant.run() in thread, collect results, return summary."""
        # Thread management with timeout
        # Patch transcription/detection to capture results
        # Return dict with counts and captured data
```

**Key features:**
- Runs `assistant.run()` in daemon thread
- Timeout safety (pytest-timeout as backup)
- Patches to capture transcriptions/detections without breaking flow
- Returns structured results for assertions

### Test Scenarios

**File:** `tests/service/test_streaming_loop.py`

| Test | Audio | Chunks | Expected | Verifies |
|------|-------|--------|----------|----------|
| test_single_transmission_streaming | 2s callsign | 4+ | 1 detection, 1 response | Basic loop operation |
| test_long_transmission_buffering | 6s speech | 12+ | All buffered, 1 TX | Multi-chunk buffering |
| test_multiple_transmissions_sequential | TX→silence→TX | 8+ | 2 detections, buffer clears | State isolation |
| test_silence_only_no_processing | 10s silence | 20 | 0 detections, clean exit | No false positives |
| test_continuous_speech_stress | 20s continuous | 40 | Document behavior | Stress test/timeout |

**Additional tests:**
- `test_buffer_accumulation_lifecycle` - Instrument buffer size, verify grow→clear pattern
- `test_vad_threshold_behavior` - Verify VAD threshold affects buffering
- `test_chunk_duration_variation` - Test with 0.25s, 0.5s, 1.0s chunks

### Success Criteria

- [ ] All 5+ streaming scenarios pass
- [ ] `run()` loop executes without hanging
- [ ] Buffer state managed correctly (grows during speech, clears after)
- [ ] No memory leaks (buffer doesn't grow unbounded)
- [ ] Thread/timeout infrastructure works reliably

### Critical Files

**To test (not modify):**
- `radio_assistant/main.py:110-136` - `run()` loop
- `radio_assistant/main.py:163-176` - `_process_buffer()`
- `radio_assistant/main.py:137-162` - `process_audio()`

**To reuse:**
- `radio_assistant/mock_radio.py` - MockRadioInterface
- `tests/e2e/test_full_pipeline.py:85-92` - MockRadioInterface usage pattern

---

## Milestone 2: Service Lifecycle & Error Recovery

**Goal:** Prove service starts, stops, and recovers from errors
**Effort:** 2-3 days
**Files:** `tests/service/test_service_lifecycle.py` (new)

### Infrastructure to Build

**1. ServiceRunner Helper** (`tests/fixtures/service_runner.py`)

```python
class ServiceRunner:
    """Manages assistant lifecycle for testing."""

    def __init__(self, config: AppConfig, audio_file: Path):
        self.assistant = None
        self.thread = None
        self.logs = []  # Captured log messages

    def start(self):
        """Start service in background thread with log capture."""

    def stop(self, timeout: float = 2.0):
        """Stop service gracefully."""

    def get_logs(self) -> list:
        """Return captured log messages for assertions."""
```

### Test Scenarios

**File:** `tests/service/test_service_lifecycle.py`

| Test | Scenario | Verification | Verifies |
|------|----------|--------------|----------|
| test_clean_startup_shutdown | Normal start/stop | Logs show init/cleanup | Lifecycle basics |
| test_keyboard_interrupt_graceful | Inject Ctrl-C | Catches exception, logs shutdown | Signal handling |
| test_transcription_failure_recovery | Mock transcribe() raises | Exception caught, logged | Error recovery |
| test_vad_failure_handling | Mock is_speech() raises | Graceful degradation | Component failure |
| test_empty_chunks_no_crash | Zero-length audio | No crashes | Edge cases |
| test_repeated_start_stop_cycles | 3x start→stop | No leaks/degradation | Resource mgmt |

**Additional tests:**
- `test_audio_device_unavailable` - Mock sounddevice unavailable
- `test_buffer_overflow_protection` - Very long continuous speech
- `test_concurrent_processing` - Verify only one utterance processed at a time

### Success Criteria

- [ ] Clean startup/shutdown works
- [ ] KeyboardInterrupt handling works
- [ ] 3+ component failure modes tested and handled
- [ ] No resource leaks across restart cycles
- [ ] All error paths have tests

### Critical Files

**To test:**
- `radio_assistant/main.py:110-136` - `run()` with error handling (try/except)
- `radio_assistant/main.py:184-188` - `stop()` method

**To understand:**
- `radio_assistant/vad_detector.py:84-119` - `is_speech()` behavior
- `radio_assistant/transcription_engine.py:44-100` - `transcribe()` behavior

---

## Execution Plan

### Phase 1: Infrastructure (Day 1)
1. Create `tests/service/` directory
2. Create `tests/fixtures/` directory
3. Implement `StreamSimulator` helper
4. Implement `ServiceRunner` helper
5. Add pytest-timeout to dependencies if not present

### Phase 2: Streaming Tests (Days 2-3)
1. Write `test_streaming_loop.py` with 5 core scenarios
2. Debug threading/timeout issues
3. Add instrumentation tests (buffer lifecycle)
4. Validate all scenarios pass

### Phase 3: Lifecycle Tests (Days 4-5)
1. Write `test_service_lifecycle.py` with 6 core scenarios
2. Test error recovery paths
3. Add edge case tests (empty chunks, concurrent, etc.)
4. Validate all scenarios pass

### Phase 4: Documentation (Day 6)
1. Document any bugs found
2. Create minimal `docs/claude/PHASE2_VALIDATION_RESULTS.md`
3. Update test coverage metrics
4. Identify gaps for future work

---

## Dependencies & Tools

**Required packages:**
- `pytest-timeout` - Prevent hanging tests
- `pytest-mock` - For patching (may already be installed)

**Optional packages:**
- `pytest-rerunfailures` - Retry flaky threading tests

**Infrastructure reuse:**
- `MockRadioInterface` (radio_assistant/mock_radio.py) - ready to use
- Existing test patterns (tests/e2e/test_full_pipeline.py)
- Existing fixtures (AppConfig, audio directories)

---

## Risk Mitigation

### Risk: run() loop blocks pytest thread
**Mitigation:** Use `threading.Thread` with daemon=True, timeout on join()

### Risk: run() loop has bugs
**Mitigation:** This is expected! Fix minimally to unblock tests, document issues

### Risk: Threading makes tests flaky
**Mitigation:** Good logging, use pytest-timeout, add retries if needed

### Risk: Tests take too long
**Mitigation:** Use short audio files (2-6s), aggressive timeouts, skip slow scenarios

---

## Success Criteria: Milestone 1-2 Complete

Phase 2 is COMPLETE when:

### Functional Validation
- [ ] `run()` loop proven to work with streaming audio (5+ scenarios)
- [ ] Service lifecycle tested (start, stop, interrupt)
- [ ] Error recovery tested (3+ failure modes)
- [ ] No hangs or crashes in continuous operation

### Test Coverage
- [ ] Streaming loop: 5-8 tests
- [ ] Lifecycle: 6-9 tests
- [ ] **Total new tests:** 12-17

### Performance
- [ ] Service runs 30+ seconds without crashes
- [ ] No memory leaks in continuous operation
- [ ] Buffer management verified (grow→clear pattern)

### Documentation
- [ ] `PHASE2_VALIDATION_RESULTS.md` created
- [ ] Bugs found documented
- [ ] Test coverage updated
- [ ] Known limitations identified

---

## Out of Scope (Explicitly NOT Doing)

**NOT in this phase:**
- ❌ Real radio audio testing (deferred)
- ❌ Multi-transmission state management (deferred)
- ❌ Adding async/await to RadioAssistant
- ❌ Adding timeout mechanism to run() loop (test current behavior first)
- ❌ Fixing bugs proactively (find via tests, then fix minimally)
- ❌ Performance optimization
- ❌ New features (streaming VAD, event bus, etc.)

**Philosophy:** Test what exists, document gaps, fix blocking bugs only.

---

## Bonus: Simulated Radio Audio (Optional)

If time permits after Milestones 1-2, add basic radio simulation:

**File:** `tests/fixtures/radio_simulator.py`

```python
def apply_radio_effects(audio: np.ndarray, sr: int,
                       snr_db: float = 15) -> np.ndarray:
    """Apply realistic radio DSP effects to clean audio."""
    # Bandpass filter (300-3000 Hz)
    # Add white noise to target SNR
    # Optional: compression, clipping
    return processed_audio
```

**Test:** Create 2-3 "radio-processed" versions of existing TTS audio, verify system still works with degraded quality. Document WER increase.

**Effort:** 0.5-1 day if time available

---

## Next Steps

1. ✅ Review this plan
2. ✅ Approve to proceed
3. → Begin Milestone 1: Implement StreamSimulator
4. → Write streaming loop tests
5. → Begin Milestone 2: Implement ServiceRunner
6. → Write lifecycle tests
7. → Document results

**Ready to execute!**
