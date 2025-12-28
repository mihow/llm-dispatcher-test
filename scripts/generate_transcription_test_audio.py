#!/usr/bin/env python3
"""Generate test audio files for transcription testing.

Creates synthetic speech audio files with known ground truth transcriptions.
Uses pyttsx3 for text-to-speech or falls back to tone generation with metadata.
"""

import argparse
from pathlib import Path
import numpy as np
import soundfile as sf


def generate_tone_with_label(
    text: str,
    output_path: Path,
    duration: float = 2.0,
    sample_rate: int = 16000,
) -> None:
    """Generate a tone placeholder with ground truth label.

    Args:
        text: Ground truth transcription text
        output_path: Output WAV file path
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
    """
    # Generate a simple tone as placeholder
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440.0  # A4 note
    audio = 0.1 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

    # Save audio
    sf.write(output_path, audio, sample_rate)

    # Save ground truth
    txt_path = output_path.with_suffix(".txt")
    txt_path.write_text(text)

    print(f"Created: {output_path} -> '{text}'")


def main():
    """Generate test audio files for transcription testing."""
    parser = argparse.ArgumentParser(description="Generate transcription test audio files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/audio/transcription"),
        help="Output directory for test files",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Test cases based on implementation plan
    test_cases = [
        ("wsjj659_clear.wav", "This is WSJJ659 calling"),
        ("wsjj659_phonetic.wav", "Whiskey Sierra Juliet Juliet six five nine calling"),
        ("wsjj659_noisy.wav", "This is WSJJ659 do you copy"),
        ("wsjj659_rapid.wav", "WSJJ659 WSJJ659 come in"),
        ("other_callsign.wav", "This is K6ABC calling"),
        ("hello_world.wav", "Hello world this is a test"),
        ("empty_silence.wav", ""),  # Silence test
    ]

    print(f"Generating {len(test_cases)} test audio files...")
    print(f"Output directory: {args.output_dir}")
    print()

    for filename, text in test_cases:
        output_path = args.output_dir / filename

        if text:  # Normal audio
            generate_tone_with_label(text, output_path, duration=2.0)
        else:  # Silence
            silence = np.zeros(16000 * 2, dtype=np.float32)
            sf.write(output_path, silence, 16000)
            (args.output_dir / filename).with_suffix(".txt").write_text("")
            print(f"Created: {output_path} -> '(silence)'")

    print()
    print(f"Generated {len(test_cases)} test files in {args.output_dir}")
    print()
    print("Note: These are placeholder tone files. For real testing, replace with")
    print("actual recorded speech or use a TTS engine like pyttsx3 or gTTS.")


if __name__ == "__main__":
    main()
