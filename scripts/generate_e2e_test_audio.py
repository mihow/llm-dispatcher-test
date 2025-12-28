#!/usr/bin/env python3
"""Generate end-to-end test audio scenarios."""

import argparse
from pathlib import Path
import numpy as np
import soundfile as sf


def generate_silence(duration: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate silence."""
    return np.zeros(int(duration * sample_rate), dtype=np.float32)


def generate_noise(duration: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate white noise."""
    return (np.random.randn(int(duration * sample_rate)) * 0.1).astype(np.float32)


def generate_tone_sequence(
    frequencies: list[float], duration: float = 0.3, sample_rate: int = 16000
) -> np.ndarray:
    """Generate tone sequence with speech-like envelope."""
    segments = []
    for freq in frequencies:
        t = np.linspace(0, duration, int(sample_rate * duration))
        tone = 0.2 * np.sin(2 * np.pi * freq * t)
        # Add simple envelope
        envelope = np.exp(-5 * np.abs(t - duration / 2))
        tone *= envelope
        segments.append(tone)

    return np.concatenate(segments).astype(np.float32)


def main():
    """Generate E2E test scenarios."""
    parser = argparse.ArgumentParser(description="Generate E2E test audio")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/audio/e2e"),
        help="Output directory",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000

    scenarios = {
        "scenario_silence.wav": generate_silence(2.0, sample_rate),
        "scenario_noise.wav": generate_noise(2.0, sample_rate),
        # These are tone placeholders - for real testing, use actual recordings
        "scenario_other_callsign.wav": generate_tone_sequence([400, 500, 600], 0.3, sample_rate),
        "scenario_wsjj659_clear.wav": generate_tone_sequence(
            [600, 700, 800, 900], 0.3, sample_rate
        ),
        "scenario_wsjj659_noisy.wav": np.concatenate(
            [
                generate_tone_sequence([600, 700, 800], 0.3, sample_rate),
                generate_noise(0.5, sample_rate) * 0.3,
            ]
        ),
    }

    print(f"Generating {len(scenarios)} E2E test scenarios...")
    print(f"Output directory: {args.output_dir}\n")

    for filename, audio in scenarios.items():
        output_path = args.output_dir / filename
        sf.write(output_path, audio, sample_rate)
        duration = len(audio) / sample_rate
        print(f"Created: {output_path} ({duration:.2f}s)")

        # Create expected outcome file
        expected_path = output_path.with_suffix(".txt")
        if "silence" in filename or "noise" in filename:
            expected_path.write_text("no_action")
        elif "other_callsign" in filename:
            expected_path.write_text("transcribe_only")
        elif "wsjj659" in filename:
            expected_path.write_text("response_triggered")

    print(f"\nGenerated {len(scenarios)} scenarios in {args.output_dir}")
    print("\nNote: These are placeholder audio files. For real testing,")
    print("replace with actual recorded speech or TTS-generated audio.")


if __name__ == "__main__":
    main()
