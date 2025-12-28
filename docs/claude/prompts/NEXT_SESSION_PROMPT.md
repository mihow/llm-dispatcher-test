# Next Session: Add Real Validation to Tests

**Created**: 2025-12-27
**Branch**: Create new `feature/add-transcription-validation`
**Priority**: HIGH - Make tests actually validate correctness
**Estimated Time**: 2-3 hours

---

## Current State

**What we have:**
- ✅ Phase 1 complete - all 225 tests passing (100%)
- ✅ TTS-generated audio with ground truth text files
- ✅ Batch processing works end-to-end
- ✅ Planning documents created for Phase 2

**The problem:**
- ❌ Tests validate "did it run?" not "did it work correctly?"
- ❌ Ground truth files exist but are ignored
- ❌ No measurement of transcription accuracy

**Example of current useless test:**
```python
# tests/integration/test_transcription_accuracy.py:44-52
def test_transcribe_hello_world(self, engine, test_audio_dir):
    audio, sr, ground_truth = self.load_audio_and_transcript(test_audio_dir, "hello_world")
    result = engine.transcribe(audio, sample_rate=sr)

    # Note: Since we're using tone placeholders, we can't test exact match
    # This test validates the engine works end-to-end
    assert isinstance(result.text, str)  # USELESS - just checks it's a string
    assert result.duration_ms > 0
    assert isinstance(result.segments, list)
```

**We load `ground_truth` but never use it!**

---

## Goal for This Session

**Add real validation using Word Error Rate (WER) and Character Error Rate (CER)**

After this session:
- ✅ Tests will compare transcriptions against ground truth
- ✅ We'll know exact accuracy of our system
- ✅ Tests will fail if transcription quality degrades
- ✅ We'll have baseline metrics for future improvements

**Then:** Research voice assistant architectures (next session)

---

## Step-by-Step Implementation

### Step 1: Install jiwer (5 minutes)

**Add to pyproject.toml:**
```toml
[project.dependencies]
# ... existing dependencies ...
"jiwer>=3.0.0",
```

**Install:**
```bash
pip install jiwer
```

**Verify:**
```bash
python -c "from jiwer import wer, cer; print('jiwer installed OK')"
```

---

### Step 2: Update Transcription Integration Tests (1-2 hours)

**File to edit:** `tests/integration/test_transcription_accuracy.py`

**Add import at top:**
```python
from jiwer import wer, cer
```

**Update each test function to validate against ground truth.**

**Example transformation:**

**BEFORE:**
```python
def test_transcribe_hello_world(
    self, engine: TranscriptionEngine, test_audio_dir: Path
) -> None:
    """Test transcribing simple hello world audio."""
    audio, sr, ground_truth = self.load_audio_and_transcript(test_audio_dir, "hello_world")
    result = engine.transcribe(audio, sample_rate=sr)

    # Note: Since we're using tone placeholders, we can't test exact match
    assert isinstance(result.text, str)
    assert result.duration_ms > 0
```

**AFTER:**
```python
def test_transcribe_hello_world(
    self, engine: TranscriptionEngine, test_audio_dir: Path
) -> None:
    """Test transcribing simple hello world audio."""
    audio, sr, ground_truth = self.load_audio_and_transcript(test_audio_dir, "hello_world")
    result = engine.transcribe(audio, sample_rate=sr)

    # Normalize for comparison
    predicted = result.text.strip().lower()
    expected = ground_truth.strip().lower()

    # Calculate accuracy metrics
    word_error_rate = wer(expected, predicted)
    char_error_rate = cer(expected, predicted)

    # Assert accuracy threshold (TTS audio should be very accurate)
    assert word_error_rate < 0.20, (
        f"WER too high: {word_error_rate:.2%}\n"
        f"Expected: '{expected}'\n"
        f"Got:      '{predicted}'"
    )

    # Keep structure checks
    assert result.duration_ms > 0
    assert isinstance(result.segments, list)

    # Log for visibility
    print(f"✓ WER={word_error_rate:.2%}, CER={char_error_rate:.2%}")
```

**Apply to all test functions:**
- `test_transcribe_hello_world` - WER < 20%
- `test_transcribe_callsign_clear` - WER < 15%, check "WSJJ659" present
- `test_transcribe_callsign_phonetic` - WER < 25%
- `test_transcribe_callsign_noisy` - WER < 30%
- `test_transcribe_empty_silence` - expect empty or minimal text
- `test_transcribe_other_callsign` - check "K6ABC" present

---

### Step 3: Add Callsign Detection Validation (30 minutes)

**For callsign tests, also verify detection:**

```python
def test_transcribe_callsign_clear(
    self, engine: TranscriptionEngine, test_audio_dir: Path
) -> None:
    """Test transcribing clear callsign audio."""
    from radio_assistant.callsign_detector import CallsignDetector

    audio, sr, ground_truth = self.load_audio_and_transcript(test_audio_dir, "wsjj659_clear")
    result = engine.transcribe(audio, sample_rate=sr)

    # Transcription accuracy
    predicted = result.text.strip().lower()
    expected = ground_truth.strip().lower()
    word_error_rate = wer(expected, predicted)

    assert word_error_rate < 0.15, (
        f"WER {word_error_rate:.2%} too high\n"
        f"Expected: '{expected}'\n"
        f"Got:      '{predicted}'"
    )

    # Callsign detection validation
    detector = CallsignDetector(target_callsign="WSJJ659")
    detection = detector.detect_callsign(result.text)

    assert detection.detected, f"Failed to detect WSJJ659 in: '{result.text}'"
    assert detection.confidence > 0.8
    assert "WSJJ659" in result.text.upper()

    print(f"✓ WER={word_error_rate:.2%}, Callsign: {detection.confidence:.0%}")
```

---

### Step 4: Add Validation Report Script (30 minutes)

**Create:** `scripts/run_validation_report.py`

```python
#!/usr/bin/env python3
"""Generate comprehensive validation report."""

import json
from pathlib import Path
from jiwer import wer, cer
import soundfile as sf
from radio_assistant.transcription_engine import TranscriptionEngine
from radio_assistant.callsign_detector import CallsignDetector


def run_validation():
    """Run validation and generate report."""
    print("=" * 80)
    print("TRANSCRIPTION VALIDATION REPORT")
    print("=" * 80)

    engine = TranscriptionEngine(model_size="base", device="cpu")
    detector = CallsignDetector(target_callsign="WSJJ659")
    test_dir = Path("tests/audio/transcription")

    results = []
    total_wer = 0
    total_cer = 0
    count = 0

    for audio_file in sorted(test_dir.glob("*.wav")):
        txt_file = audio_file.with_suffix(".txt")
        if not txt_file.exists():
            continue

        audio, sr = sf.read(audio_file, dtype="float32")
        ground_truth = txt_file.read_text().strip()

        if not ground_truth:
            continue

        result = engine.transcribe(audio, sr)

        predicted = result.text.strip().lower()
        expected = ground_truth.strip().lower()
        wer_score = wer(expected, predicted)
        cer_score = cer(expected, predicted)

        detection = detector.detect_callsign(result.text)

        results.append({
            "file": audio_file.name,
            "ground_truth": ground_truth,
            "transcription": result.text,
            "wer": wer_score,
            "cer": cer_score,
            "callsign_detected": detection.detected,
            "confidence": detection.confidence if detection.detected else 0.0,
        })

        status = "✓" if wer_score < 0.20 else "✗"
        print(f"\n{status} {audio_file.name}")
        print(f"  Expected:     '{ground_truth}'")
        print(f"  Transcribed:  '{result.text}'")
        print(f"  WER: {wer_score:.2%}, CER: {cer_score:.2%}")

        total_wer += wer_score
        total_cer += cer_score
        count += 1

    avg_wer = total_wer / count if count > 0 else 0
    avg_cer = total_cer / count if count > 0 else 0

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Files tested: {count}")
    print(f"Average WER:  {avg_wer:.2%}")
    print(f"Average CER:  {avg_cer:.2%}")
    print("=" * 80)

    Path("validation_report.json").write_text(json.dumps(results, indent=2))

    success = avg_wer < 0.15
    status_msg = "✓ PASSED" if success else "✗ FAILED"
    print(f"\n{status_msg}: Average WER threshold")

    return success


if __name__ == "__main__":
    import sys
    success = run_validation()
    sys.exit(0 if success else 1)
```

**Make executable and run:**
```bash
chmod +x scripts/run_validation_report.py
python scripts/run_validation_report.py
```

---

### Step 5: Document Baseline Metrics (15 minutes)

**Create:** `docs/claude/BASELINE_METRICS.md`

```markdown
# Baseline Transcription Metrics

**Date**: 2025-12-27
**Model**: Whisper base (CPU)
**Audio**: TTS-generated (gTTS)

## Results

| File | WER | CER | Notes |
|------|-----|-----|-------|
| hello_world.wav | X.XX% | X.XX% | Clean speech |
| wsjj659_clear.wav | X.XX% | X.XX% | Callsign detected |
| wsjj659_phonetic.wav | X.XX% | X.XX% | Phonetic alphabet |
| wsjj659_noisy.wav | X.XX% | X.XX% | With noise |
| other_callsign.wav | X.XX% | X.XX% | K6ABC |

**Summary:**
- Average WER: X.XX%
- Average CER: X.XX%
- Callsign detection: XX% accuracy

[Fill in actual values after running validation report]
```

---

## Quick Commands

```bash
# Setup
git checkout main && git pull
git checkout -b feature/add-transcription-validation

# Install
pip install jiwer

# Test
pytest tests/integration/test_transcription_accuracy.py -v -s

# Report
python scripts/run_validation_report.py

# Commit
git add -A
git commit -m "feat: add WER/CER validation to transcription tests"
git push -u origin feature/add-transcription-validation

# PR
gh pr create --title "Add transcription accuracy validation"
```

---

## Expected Results

With TTS audio + Whisper base:
- Clean audio: WER < 10%
- Noisy audio: WER < 20%
- Callsign detection: >95% accuracy

---

## Success Criteria

- ✅ jiwer installed
- ✅ All tests validate against ground truth
- ✅ WER/CER metrics logged
- ✅ Validation report runs
- ✅ Baseline metrics documented
- ✅ PR created

**Then: Research voice assistant architectures (next session)**

---

## After This Session

**Next:** Research Phase (see `docs/claude/planning/VOICE_ASSISTANT_RESEARCH.md`)
- Study Rhasspy, Home Assistant, Mycroft AI
- Document streaming architecture patterns
- Create informed architecture proposal

**Full roadmap:** `docs/claude/planning/PHASE2_PLANNING_SUMMARY.md`

---

*Status: Ready to start*
*Time: 2-3 hours*
*Then: Research existing voice assistants*
