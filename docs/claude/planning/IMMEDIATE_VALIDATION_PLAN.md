# Immediate Validation Plan: Make Tests Actually Test Things

**Created**: 2025-12-27
**Goal**: Use existing TTS audio + ground truth to validate the system actually works

## The Key Insight

**We don't need real radio audio to validate the pipeline works.**

We have:
- TTS-generated audio with known content
- Ground truth text files with exact transcriptions
- Known timestamps and expected behaviors
- Full integration from audio → VAD → transcription → callsign detection → PTT

**The problem**: We load ground truth and then ignore it.

---

## Phase 1: Validate Transcription Accuracy (IMMEDIATE)

### Task 1.1: Add WER/CER Metrics

**Install dependency:**
```bash
pip install jiwer
```

**Update pyproject.toml:**
```toml
[project.dependencies]
# ... existing ...
"jiwer>=3.0.0",
```

### Task 1.2: Update Transcription Integration Tests

**File**: `tests/integration/test_transcription_accuracy.py`

**Current (useless):**
```python
def test_transcribe_hello_world(self, engine, test_audio_dir):
    audio, sr, ground_truth = self.load_audio_and_transcript(test_audio_dir, "hello_world")
    result = engine.transcribe(audio, sample_rate=sr)

    # Note: Since we're using tone placeholders, we can't test exact match
    # This test validates the engine works end-to-end
    assert isinstance(result.text, str)  # MEANINGLESS
    assert result.duration_ms > 0
```

**New (validates accuracy):**
```python
from jiwer import wer, cer

def test_transcribe_hello_world(self, engine, test_audio_dir):
    audio, sr, ground_truth = self.load_audio_and_transcript(test_audio_dir, "hello_world")
    result = engine.transcribe(audio, sample_rate=sr)

    # Normalize for comparison
    predicted = result.text.strip().lower()
    expected = ground_truth.strip().lower()

    # Calculate accuracy metrics
    word_error_rate = wer(expected, predicted)
    char_error_rate = cer(expected, predicted)

    # Assert accuracy thresholds
    assert word_error_rate < 0.20, (
        f"WER too high: {word_error_rate:.2%}\n"
        f"Expected: '{expected}'\n"
        f"Got: '{predicted}'"
    )

    # Verify structure
    assert result.duration_ms > 0
    assert isinstance(result.segments, list)

    # Log metrics for analysis
    print(f"WER: {word_error_rate:.2%}, CER: {char_error_rate:.2%}")
```

**Apply to ALL transcription tests:**
- `test_transcribe_hello_world` - WER < 20%
- `test_transcribe_callsign_clear` - WER < 15%, must contain "WSJJ659"
- `test_transcribe_callsign_phonetic` - WER < 25%, must contain "whiskey sierra juliet"
- `test_transcribe_callsign_noisy` - WER < 30% (more tolerant for noisy)
- `test_transcribe_empty_silence` - should return empty or minimal text
- `test_transcribe_other_callsign` - must contain "K6ABC"

**Expected thresholds for TTS audio:**
- Clean audio: WER < 10% (Whisper is very good with clear speech)
- Noisy audio: WER < 20% (TTS with added noise)
- Phonetic spelling: WER < 15% (longer but clear)

### Task 1.3: Add Callsign Detection Validation

**Current:**
```python
result = engine.transcribe(audio, sample_rate=sr)
assert isinstance(result.text, str)  # Useless
```

**New:**
```python
from radio_assistant.callsign_detector import CallsignDetector

result = engine.transcribe(audio, sample_rate=sr)
callsign_detector = CallsignDetector(target_callsign="WSJJ659")

# Test transcription quality
word_error_rate = wer(ground_truth, result.text)
assert word_error_rate < 0.15

# Test callsign detection works
detection = callsign_detector.detect_callsign(result.text)
assert detection.detected, f"Failed to detect WSJJ659 in: '{result.text}'"
assert detection.confidence > 0.8
assert "WSJJ659" in detection.matched_pattern.upper()
```

---

## Phase 2: Validate Full Pipeline End-to-End

### Task 2.1: Add Real E2E Validation

**File**: `tests/e2e/test_full_pipeline.py`

**New test:**
```python
def test_e2e_callsign_detection_and_response(self, config, e2e_audio_dir):
    """Test complete pipeline: audio → VAD → transcription → callsign → PTT."""

    # Load test scenario
    audio_path = e2e_audio_dir / "scenario_wsjj659_clear.wav"
    ground_truth_path = e2e_audio_dir / "scenario_wsjj659_clear.txt"

    audio, sr = sf.read(audio_path, dtype="float32")
    ground_truth = ground_truth_path.read_text().strip()

    # Run full pipeline
    assistant = RadioAssistant(config)

    # Track what happened
    vad_detected = assistant.vad.is_speech(audio)
    transcription = assistant.transcription.transcribe(audio, sr)
    callsign_result = assistant.callsign.detect_callsign(transcription.text)

    # Validate each stage
    assert vad_detected, "VAD failed to detect speech"

    # Transcription accuracy
    wer_score = wer(ground_truth.lower(), transcription.text.lower())
    assert wer_score < 0.15, f"WER {wer_score:.2%} too high: {transcription.text}"

    # Callsign detection
    assert callsign_result.detected, f"Failed to detect callsign in: {transcription.text}"
    assert callsign_result.confidence > 0.8

    # Full pipeline should trigger response
    result = assistant.process_audio(audio)
    assert result is True, "Pipeline should have triggered response for valid callsign"
```

### Task 2.2: Test Negative Cases

**Add tests for what should NOT trigger:**

```python
def test_e2e_other_callsign_no_response(self, config, e2e_audio_dir):
    """Test pipeline correctly rejects other callsigns."""
    audio_path = e2e_audio_dir / "scenario_other_callsign.wav"
    audio, sr = sf.read(audio_path, dtype="float32")

    assistant = RadioAssistant(config)
    result = assistant.process_audio(audio)

    # Should detect speech and transcribe
    vad_detected = assistant.vad.is_speech(audio)
    assert vad_detected, "VAD should detect speech"

    transcription = assistant.transcription.transcribe(audio, sr)
    assert "K6ABC" in transcription.text.upper(), "Should transcribe K6ABC"

    # But should NOT trigger response (wrong callsign)
    assert result is False, "Should not respond to different callsign"

def test_e2e_noise_no_false_positive(self, config, e2e_audio_dir):
    """Test pipeline doesn't false-trigger on noise."""
    audio_path = e2e_audio_dir / "scenario_noise.wav"
    audio, sr = sf.read(audio_path, dtype="float32")

    assistant = RadioAssistant(config)
    result = assistant.process_audio(audio)

    # Should NOT trigger on noise
    assert result is False, "Should not trigger on noise"
```

---

## Phase 3: Performance Benchmarking

### Task 3.1: Measure Real Performance

**File**: `tests/benchmarks/test_realistic_performance.py`

```python
import time
import pytest
from pathlib import Path
import soundfile as sf
from radio_assistant.transcription_engine import TranscriptionEngine
from radio_assistant.vad_detector import VADDetector

class TestRealisticPerformance:
    """Benchmark realistic performance with actual audio."""

    @pytest.fixture
    def test_audio(self):
        """Load realistic test audio."""
        path = Path("tests/audio/transcription/wsjj659_clear.wav")
        audio, sr = sf.read(path, dtype="float32")
        duration = len(audio) / sr
        return audio, sr, duration

    def test_vad_latency(self, test_audio):
        """Measure VAD detection latency."""
        audio, sr, duration = test_audio
        vad = VADDetector()

        start = time.perf_counter()
        result = vad.is_speech(audio)
        elapsed = time.perf_counter() - start

        # VAD should be near-instant
        assert elapsed < 0.1, f"VAD too slow: {elapsed:.3f}s"
        assert result is True

        print(f"VAD latency: {elapsed*1000:.1f}ms for {duration:.1f}s audio")

    def test_transcription_realtime_factor(self, test_audio):
        """Measure transcription real-time factor."""
        audio, sr, duration = test_audio
        engine = TranscriptionEngine(model_size="base", device="cpu")

        start = time.perf_counter()
        result = engine.transcribe(audio, sr)
        elapsed = time.perf_counter() - start

        rtf = elapsed / duration  # Real-Time Factor

        # Should be faster than real-time (RTF < 1.0)
        # Allow some overhead, but <2.0 is minimum acceptable
        assert rtf < 2.0, f"RTF {rtf:.2f} too slow for real-time"

        print(f"Transcription: {duration:.1f}s audio in {elapsed:.2f}s (RTF: {rtf:.2f})")

        # On modern CPU (not Pi), should be quite fast
        if rtf > 1.5:
            pytest.skip("Performance marginal - may not work on Pi")

    def test_end_to_end_latency(self, test_audio):
        """Measure total pipeline latency."""
        audio, sr, duration = test_audio

        from radio_assistant.main import RadioAssistant, AppConfig
        config = AppConfig(callsign="WSJJ659")
        assistant = RadioAssistant(config)

        start = time.perf_counter()
        result = assistant.process_audio(audio)
        elapsed = time.perf_counter() - start

        # Target: <3s for user experience
        # Warn if >5s (likely too slow for real use)
        assert elapsed < 5.0, f"E2E latency {elapsed:.2f}s too slow"

        if elapsed > 3.0:
            pytest.warn(f"E2E latency {elapsed:.2f}s exceeds 3s target")

        print(f"E2E latency: {elapsed:.2f}s for {duration:.1f}s audio")
```

### Task 3.2: Memory Usage Benchmark

```python
import psutil
import os

def test_memory_footprint():
    """Measure memory usage of loaded models."""
    process = psutil.Process(os.getpid())

    # Baseline
    baseline_mb = process.memory_info().rss / 1024 / 1024

    # Load transcription engine
    from radio_assistant.transcription_engine import TranscriptionEngine
    engine = TranscriptionEngine(model_size="base", device="cpu")

    loaded_mb = process.memory_info().rss / 1024 / 1024
    model_size_mb = loaded_mb - baseline_mb

    # Whisper base model should be ~140MB + overhead
    # Total should be <500MB for basic operation
    assert loaded_mb < 500, f"Memory usage {loaded_mb:.0f}MB too high"

    print(f"Memory: baseline={baseline_mb:.0f}MB, loaded={loaded_mb:.0f}MB, model={model_size_mb:.0f}MB")
```

---

## Phase 4: Metrics Dashboard

### Task 4.1: Create Test Report

**File**: `scripts/run_validation_report.py`

```python
#!/usr/bin/env python3
"""Generate comprehensive validation report."""

import json
from pathlib import Path
from jiwer import wer, cer
import soundfile as sf
from radio_assistant.transcription_engine import TranscriptionEngine

def run_validation():
    """Run validation and generate report."""

    engine = TranscriptionEngine(model_size="base", device="cpu")
    test_dir = Path("tests/audio/transcription")

    results = []

    for audio_file in test_dir.glob("*.wav"):
        txt_file = audio_file.with_suffix(".txt")
        if not txt_file.exists():
            continue

        # Load
        audio, sr = sf.read(audio_file, dtype="float32")
        ground_truth = txt_file.read_text().strip()

        # Transcribe
        result = engine.transcribe(audio, sr)

        # Calculate metrics
        wer_score = wer(ground_truth.lower(), result.text.lower())
        cer_score = cer(ground_truth.lower(), result.text.lower())

        results.append({
            "file": audio_file.name,
            "ground_truth": ground_truth,
            "transcription": result.text,
            "wer": wer_score,
            "cer": cer_score,
            "duration_ms": result.duration_ms,
        })

    # Print report
    print("\n" + "="*80)
    print("TRANSCRIPTION VALIDATION REPORT")
    print("="*80)

    for r in results:
        print(f"\n{r['file']}:")
        print(f"  Ground truth: {r['ground_truth']}")
        print(f"  Transcribed:  {r['transcription']}")
        print(f"  WER: {r['wer']:.2%}, CER: {r['cer']:.2%}")
        print(f"  Duration: {r['duration_ms']}ms")

    # Summary
    avg_wer = sum(r['wer'] for r in results) / len(results)
    avg_cer = sum(r['cer'] for r in results) / len(results)

    print("\n" + "="*80)
    print(f"SUMMARY: {len(results)} files")
    print(f"Average WER: {avg_wer:.2%}")
    print(f"Average CER: {avg_cer:.2%}")
    print("="*80)

    # Save JSON
    Path("validation_report.json").write_text(json.dumps(results, indent=2))

    return avg_wer < 0.15  # Success if <15% WER average

if __name__ == "__main__":
    success = run_validation()
    exit(0 if success else 1)
```

---

## Implementation Order

### Day 1 (2-3 hours): Core Validation
1. ✅ Install jiwer: `pip install jiwer`
2. ✅ Update `test_transcription_accuracy.py` with WER/CER assertions
3. ✅ Run tests - see actual accuracy numbers
4. ✅ Adjust thresholds based on real performance

### Day 2 (2-3 hours): E2E Validation
1. ✅ Add E2E tests with full pipeline validation
2. ✅ Test negative cases (wrong callsign, noise)
3. ✅ Verify false positive/negative rates

### Day 3 (2-3 hours): Performance
1. ✅ Add performance benchmarks
2. ✅ Measure RTF, latency, memory
3. ✅ Create validation report script
4. ✅ Document baseline performance

---

## Success Criteria (with TTS Audio)

**After completing this plan, we'll know:**

✅ **Transcription accuracy**: Exact WER/CER on clean and noisy TTS audio
✅ **Callsign detection**: Does it actually detect "WSJJ659" correctly?
✅ **False positives**: Does it ignore other callsigns and noise?
✅ **False negatives**: Does it catch all valid instances?
✅ **Performance**: What's the actual latency and resource usage?
✅ **Reliability**: Does it work consistently across multiple scenarios?

**Expected Results (realistic targets):**
- WER < 10% on clean TTS audio (Whisper is excellent with clear speech)
- WER < 20% on noisy TTS audio
- 100% callsign detection on clear audio
- >95% callsign detection on noisy audio
- 0% false positives on other callsigns
- RTF < 1.5 on modern CPU (will be slower on Pi, but gives baseline)
- E2E latency <3s on modern hardware

---

## What This Validates

**With TTS audio, we can definitively prove:**
1. ✅ Transcription engine works correctly
2. ✅ Callsign detection logic is sound
3. ✅ Pipeline integrates properly
4. ✅ Performance is in the right ballpark

**What we can't prove (needs real radio later):**
- How it handles real radio artifacts (squelch, fading, etc.)
- Actual hardware deployment on Pi
- Real-world interference patterns

**But that's fine!** We validate the core functionality first, then test real-world conditions as Phase 2.

---

## The Bottom Line

**You're right**: TTS audio with known ground truth is perfect for validation.

**The problem**: We're not validating against it.

**The fix**: 6-9 hours of focused work to add real assertions.

**The result**: We'll know exactly how well this works, not just that it runs.

Ready to start with Task 1.1 (add jiwer and update transcription tests)?
