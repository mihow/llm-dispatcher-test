# Next Session: Research Voice Assistant Architectures

**Created**: 2025-12-27
**Branch**: `planning/phase2-validation-and-service`
**Priority**: MEDIUM - Research before implementation
**Context**: 14K tokens used

---

## Current State

**What we completed:**
- ✅ Phase 1 complete - all 225 tests passing (100%)
- ✅ WER/CER validation added to transcription tests
- ✅ Baseline metrics documented: 2.08% avg WER, 0.57% avg CER
- ✅ Validation report script created
- ✅ All 16 integration tests passing with real accuracy validation
- ✅ PR #3 updated with validation changes

**Current metrics:**
- Average WER: 2.08% (excellent)
- Average CER: 0.57% (excellent)
- Callsign detection: 95% confidence (100% accuracy)
- Only issue: noisy audio shows 12.50% WER (669 vs 659 transcription error)

---

## Next Steps

### Option 1: Fix Noisy Audio Test (Quick Fix - 30 min)

The `wsjj659_noisy.wav` file shows 12.50% WER because Whisper transcribes "669" instead of "659".

**Investigate and fix:**
1. Check the actual audio file - is it too noisy?
2. Try different Whisper models (small, medium) for better accuracy
3. Consider adjusting WER threshold for noisy audio (currently 30%)
4. Update ground truth if "669" is actually what's in the audio

**Quick check:**
```bash
# Listen to the audio
ffplay tests/audio/transcription/wsjj659_noisy.wav

# Try with larger model
# Edit validation script to use model_size="small"
python scripts/run_validation_report.py
```

### Option 2: Research Voice Assistant Architectures (2-3 hours)

**Goal:** Study existing open-source voice assistants to inform Phase 2 architecture design.

**Research targets:**
1. **Rhasspy** - Offline voice assistant
   - How does it handle streaming audio?
   - State machine architecture
   - Integration patterns

2. **Home Assistant Voice** - Voice control for home automation
   - Wake word detection
   - Intent recognition
   - Response generation

3. **Mycroft AI** - Open source voice assistant
   - Message bus architecture
   - Skill system
   - Audio processing pipeline

**Deliverables:**
- `docs/claude/research/VOICE_ASSISTANT_COMPARISON.md`
  - Architecture diagrams (ASCII)
  - Pros/cons of each approach
  - Lessons learned for our system

- `docs/claude/planning/PHASE2_ARCHITECTURE_PROPOSAL.md`
  - Proposed architecture based on research
  - Component breakdown
  - Integration points with existing code

**See:** `docs/claude/planning/VOICE_ASSISTANT_RESEARCH.md` for research template

---

## Recommended Path

**I recommend Option 2** - Do the research before building more features.

**Why:**
- Tests are already passing with good metrics
- The noisy audio issue is minor (12.50% WER is acceptable)
- Understanding existing patterns will save time in Phase 2
- We need informed architecture decisions before implementing streaming/service

**After research, we can:**
1. Create informed Phase 2 architecture design
2. Implement streaming audio handler
3. Build service layer with proper patterns
4. Add more sophisticated validation

---

## Quick Commands

### If choosing Option 1 (Fix Tests):
```bash
# Check current branch
git status

# Run validation report
conda activate cuda12
python scripts/run_validation_report.py

# Run specific test
pytest tests/integration/test_transcription_accuracy.py::TestTranscriptionAccuracy::test_transcribe_callsign_phonetic -v -s
```

### If choosing Option 2 (Research):
```bash
# Create research document
mkdir -p docs/claude/research

# Start research session - use web search and documentation
# Focus on:
# - Rhasspy architecture
# - Home Assistant Voice
# - Mycroft message bus pattern
```

---

## Context for Next Session

**Key files to understand:**
- `radio_assistant/transcription_engine.py` - Current transcription
- `radio_assistant/callsign_detector.py` - Detection logic
- `tests/integration/test_transcription_accuracy.py` - Integration tests
- `scripts/run_validation_report.py` - Validation script
- `docs/claude/BASELINE_METRICS.md` - Current metrics

**Architecture questions to answer:**
1. How should we handle streaming audio (not batch)?
2. What state machine do we need for voice interaction?
3. How do we integrate VAD + transcription + detection + response?
4. Should we use a message bus pattern like Mycroft?
5. How do we handle concurrent audio streams (multi-channel)?

**Pending issues:**
- Noisy audio transcription (669 vs 659) - minor issue
- Need streaming audio architecture
- Need service/daemon design
- Need proper state management for conversations

---

## Success Criteria

**For Option 1 (Fix Tests):**
- ✅ Noisy audio WER < 10%
- ✅ All tests still passing
- ✅ Metrics updated in BASELINE_METRICS.md

**For Option 2 (Research):**
- ✅ Voice assistant comparison document created
- ✅ Architecture proposal drafted
- ✅ Key patterns identified for our use case
- ✅ Decision on message bus vs direct integration
- ✅ Streaming audio approach defined

---

## Full Roadmap

**Phase 2 Plan** (see `docs/claude/planning/PHASE2_PLANNING_SUMMARY.md`):

1. ✅ Add WER/CER validation (DONE - this session)
2. 🔄 Research voice assistants (NEXT - recommended)
3. 📋 Design Phase 2 architecture
4. 📋 Implement streaming audio handler
5. 📋 Build service layer
6. 📋 Add state management
7. 📋 Integration testing

---

*Status: Ready for research phase*
*Recommend: Option 2 - Research before building*
*Time: 2-3 hours for thorough research*
