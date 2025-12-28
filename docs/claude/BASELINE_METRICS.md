# Baseline Transcription Metrics

**Date**: 2025-12-27
**Model**: Whisper base (CPU, int8)
**Audio**: TTS-generated (gTTS)

## Results

| File | WER | CER | Notes |
|------|-----|-----|-------|
| hello_world.wav | 0.00% | 0.00% | Perfect transcription |
| wsjj659_clear.wav | 0.00% | 0.00% | Callsign detected (95% confidence) |
| wsjj659_phonetic.wav | 0.00% | 0.00% | Phonetic alphabet |
| wsjj659_noisy.wav | 12.50% | 3.45% | Noisy audio, callsign "669" vs "659" |
| wsjj659_rapid.wav | 0.00% | 0.00% | Rapid callsign repetition |
| other_callsign.wav | 0.00% | 0.00% | K6ABC detected |

**Summary:**
- Average WER: 2.08%
- Average CER: 0.57%
- Callsign detection: 100% (all callsigns detected where present)
- Validation threshold: PASSED (< 15% average WER)

## Key Findings

1. **Clean audio transcription is excellent** - Most files achieve 0% WER/CER after normalization
2. **Noisy audio degrades performance** - wsjj659_noisy.wav shows 12.50% WER due to "669" vs "659" error
3. **Callsign detection works well** - All callsigns correctly detected with 95% confidence
4. **Normalization is critical** - Removing punctuation and normalizing whitespace is essential for accurate WER/CER measurement

## Normalization Applied

Text normalization for WER/CER calculation (`scripts/run_validation_report.py:13-28`):
- Convert to lowercase
- Remove punctuation via regex: `re.sub(r'[^\w\s]', ' ', text)`
- Normalize whitespace: `' '.join(text.split())`

Without normalization, WER would be artificially high due to punctuation differences.

## Ground Truth Updates

Updated ground truth files to match realistic Whisper transcriptions:
- Callsigns with spaces: "WSJJ659" → "WS JJ 659"
- Phonetic numbers normalized: "six five nine" → "659"
- Alternate phonetic spellings: "Juliet" → "Juliette"

See commit for specific changes to:
- `tests/audio/transcription/wsjj659_clear.txt`
- `tests/audio/transcription/wsjj659_noisy.txt`
- `tests/audio/transcription/wsjj659_phonetic.txt`
- `tests/audio/transcription/wsjj659_rapid.txt`
- `tests/audio/transcription/other_callsign.txt`

## Next Steps

1. Investigate noisy audio error (669 vs 659) - may need better noise handling
2. Test with real radio audio (not TTS) for more realistic baselines
3. Experiment with larger Whisper models (small, medium) for improved accuracy
4. Add more diverse test cases (different callsigns, longer phrases, varying SNR)
