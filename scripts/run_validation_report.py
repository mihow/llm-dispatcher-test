#!/usr/bin/env python3
"""Generate comprehensive validation report."""

import json
import re
from pathlib import Path
from jiwer import wer, cer
import soundfile as sf
from radio_assistant.transcription_engine import TranscriptionEngine
from radio_assistant.callsign_detector import CallsignDetector


def normalize_for_comparison(text: str) -> str:
    """Normalize text for WER/CER comparison.

    Args:
        text: Input text

    Returns:
        Normalized text (lowercase, no punctuation, normalized whitespace)
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def run_validation() -> bool:
    """Run validation and generate report.

    Returns:
        True if validation passes, False otherwise
    """
    print("=" * 80)
    print("TRANSCRIPTION VALIDATION REPORT")
    print("=" * 80)

    engine = TranscriptionEngine(model_size="base", device="cpu", compute_type="int8")
    detector = CallsignDetector(callsign="WSJJ659", require_dispatch_keyword=False)
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

        # Normalize for comparison
        predicted = normalize_for_comparison(result.text)
        expected = normalize_for_comparison(ground_truth)
        wer_score = wer(expected, predicted)
        cer_score = cer(expected, predicted)

        detection = detector.detect(result.text)

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
