# Next Session: Continue Ham Radio Voice Assistant Implementation

## Current Status

**PR #1**: https://github.com/mihow/llm-dispatcher-test/pull/1
- Branch: `feature/phase1-marco-polo`
- Base: `main`

### ✅ Completed (Steps 1-2)

**Step 1: Audio Capture** (commit 2df9ba2)
- `radio_assistant/audio_interface.py` - Platform-agnostic audio I/O
- 20 unit tests + 13 integration tests
- Test script: `scripts/test_audio_capture.py`

**Step 2: VAD Detection** (commit d1bbf96)
- `radio_assistant/vad_detector.py` - Silero VAD integration
- 20 unit tests + 15 integration tests
- Test audio generation: `scripts/generate_vad_test_audio.py`
- Test script: `scripts/test_vad.py`

### ⚠️ Issues to Fix

**GitHub Actions tests are failing** - need to debug CI pipeline:
1. Check test output in PR #1
2. Fix any dependency installation issues
3. Update `.github/workflows/test.yml` if needed
4. Ensure tests pass in CI environment

Likely issues:
- Missing system dependencies (libportaudio2, libsndfile1)
- Module import errors in CI
- Test audio file paths

### 📋 Next Steps (Steps 3-7)

**Step 3: Whisper Transcription** (next priority)
- Implement `TranscriptionEngine` class using faster-whisper
- Create test audio files with ground truth transcriptions
- Test files needed (per plan):
  - `wsjj659_clear.wav` + transcript.txt
  - `wsjj659_phonetic.wav` + transcript.txt
  - `wsjj659_noisy.wav` + transcript.txt
  - `wsjj659_rapid.wav` + transcript.txt
  - `other_callsign.wav` + transcript.txt
- Unit tests + integration tests + benchmarks
- Validate transcription accuracy (>95% for callsign)
- Measure performance (<2s transcription time)

**Step 4: Callsign Detection**
- Implement `CallsignDetector` class
- Pattern matching for "WSJJ659" (plain, phonetic, variations)
- Comprehensive test cases (see plan lines 340-365)
- 100% code coverage required

**Step 5: PTT/VOX Control**
- Implement `PTTController` class
- VOX padding for clean audio transmission
- Pre-generated response audio
- Manual testing procedures in TESTING.md

**Step 6: End-to-End Pipeline**
- Main `RadioAssistant` application
- E2E test scenarios
- Mock radio interface for testing
- Complete pipeline validation

**Step 7: Hardware Validation**
- Raspberry Pi testing documentation
- Performance benchmarks
- Manual test procedures

## Environment Setup

**Python Environment:**
- Conda env: `cuda12` (`/home/michael/miniforge3/envs/cuda12`)
- Package manager: `uv` for dependencies
- Python: 3.12.9

**Key Dependencies:**
```bash
/home/michael/miniforge3/envs/cuda12/bin/pip install \
  sounddevice numpy librosa scipy \
  torch silero-vad \
  pydantic typer pyyaml loguru \
  pytest pytest-cov pytest-mock soundfile
```

**For Step 3, also install:**
```bash
/home/michael/miniforge3/envs/cuda12/bin/pip install faster-whisper
```

## Project Structure

```
/home/michael/Projects/Radio/llm-dispatcher-test/
├── radio_assistant/
│   ├── __init__.py
│   ├── audio_interface.py       ✅ Complete
│   ├── vad_detector.py          ✅ Complete
│   ├── transcription_engine.py  ⏳ Next (Step 3)
│   ├── callsign_detector.py     ⏳ Step 4
│   ├── ptt_controller.py        ⏳ Step 5
│   └── main.py                  ⏳ Step 6
├── scripts/
│   ├── test_audio_capture.py
│   ├── generate_vad_test_audio.py
│   ├── test_vad.py
│   └── test_transcription.py    ⏳ Create for Step 3
├── tests/
│   ├── unit/
│   │   ├── test_audio_interface.py      (20 tests)
│   │   ├── test_vad_detector.py         (20 tests)
│   │   └── test_transcription_engine.py ⏳ Create
│   ├── integration/
│   │   ├── test_audio_capture.py        (13 tests)
│   │   ├── test_vad_pipeline.py         (15 tests)
│   │   └── test_transcription_accuracy.py ⏳ Create
│   ├── benchmarks/
│   │   └── test_transcription_performance.py ⏳ Create
│   └── audio/
│       ├── vad/                 ✅ 7 files generated
│       ├── transcription/       ⏳ Create for Step 3
│       ├── responses/           ⏳ Create for Step 5
│       └── e2e/                 ⏳ Create for Step 6
├── docs/claude/planning/
│   └── INITIAL_IMPLEMENTATION_PLAN.md (full spec)
├── .github/workflows/
│   └── test.yml                 ⚠️ Needs fixes
├── pyproject.toml
├── config.example.yaml
└── README.md

Context: 107K/200K tokens used (53%)
```

## Important Guidelines

From `CLAUDE.md`:

1. **Cost Optimization**
   - Keep context under 40% when possible (currently at 53%)
   - Use command-line tools over reading files
   - Prefer language servers over grep when available

2. **Type Annotations** (Python 3.10+ style)
   ```python
   # ✅ CORRECT
   def process(items: list[str], config: dict[str, int] | None = None) -> tuple[str, int]:
       ...

   # ❌ WRONG - old style
   from typing import List, Optional, Dict
   def process(items: List[str], config: Optional[Dict[str, int]] = None):
       ...
   ```

3. **Commit Strategy**
   - Commit often with focused changes
   - Use `git add -p` for interactive staging
   - Each step gets its own checkpoint commit

## Immediate Actions for Next Session

1. **Fix CI Tests**
   ```bash
   # Check PR test failures
   gh pr view 1 --web

   # Run tests locally to verify
   /home/michael/miniforge3/envs/cuda12/bin/python -m pytest tests/ -v

   # Fix any issues in .github/workflows/test.yml
   ```

2. **Start Step 3: Transcription Engine**
   ```bash
   # Install faster-whisper
   /home/michael/miniforge3/envs/cuda12/bin/pip install faster-whisper

   # Create TranscriptionEngine class
   # - See plan lines 242-274 for API spec
   # - Use faster-whisper library
   # - Support base/small/medium models
   # - Return structured TranscriptionResult

   # Generate test audio with known transcripts
   # Create script to synthesize or record test phrases

   # Implement comprehensive tests
   # - Unit tests with mocks
   # - Integration tests with real audio
   # - Benchmarks for performance (<2s target)
   ```

3. **Update TODO List**
   ```bash
   # Mark CI fixes as in-progress
   # Track Step 3 sub-tasks
   ```

## Reference Files

- Implementation Plan: `docs/claude/planning/INITIAL_IMPLEMENTATION_PLAN.md`
- Development Guidelines: `CLAUDE.md`
- Current PR: https://github.com/mihow/llm-dispatcher-test/pull/1
- Test Workflow: `.github/workflows/test.yml`

## Success Criteria for Next Session

- [ ] All CI tests passing in PR #1
- [ ] Step 3: TranscriptionEngine implemented
- [ ] Transcription tests created and passing
- [ ] Performance benchmarks show <2s transcription time
- [ ] Commit Step 3 checkpoint
- [ ] Update PR with Step 3 progress

## Commands Reference

```bash
# Activate environment
# (conda env cuda12 is active by default)

# Run tests
/home/michael/miniforge3/envs/cuda12/bin/python -m pytest tests/unit/ -v
/home/michael/miniforge3/envs/cuda12/bin/python -m pytest tests/integration/ -v

# Run with coverage
/home/michael/miniforge3/envs/cuda12/bin/python -m pytest tests/ --cov=radio_assistant --cov-report=term

# Install dependencies
/home/michael/miniforge3/envs/cuda12/bin/pip install <package>

# Commit checkpoint
git add -A
git commit -m "feat: Implement Step 3 - Transcription Engine (checkpoint/transcription)"
git push origin feature/phase1-marco-polo
```

---

**Goal**: Complete Phase 1 "Marco Polo" test - validate audio pipeline before adding LLM complexity.

**Current Branch**: `feature/phase1-marco-polo`

**Start Here**: Fix CI tests, then implement Step 3 (Transcription Engine)
