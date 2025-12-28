#!/usr/bin/env python3
"""Generate response audio files for testing."""

import argparse
from pathlib import Path
import numpy as np
import soundfile as sf


def generate_tone_sequence(
    frequencies: list[float], duration: float = 0.5, sample_rate: int = 16000
) -> np.ndarray:
    """Generate a sequence of tones.

    Args:
        frequencies: List of frequencies in Hz
        duration: Duration of each tone in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Audio data as numpy array
    """
    audio_segments = []

    for freq in frequencies:
        t = np.linspace(0, duration, int(sample_rate * duration))
        tone = 0.3 * np.sin(2 * np.pi * freq * t)
        audio_segments.append(tone)

    return np.concatenate(audio_segments).astype(np.float32)


def generate_signal_received(sample_rate: int = 16000) -> np.ndarray:
    """Generate 'signal received' acknowledgment tone.

    Uses a two-tone sequence to simulate a simple acknowledgment beep.

    Args:
        sample_rate: Sample rate in Hz

    Returns:
        Audio data
    """
    # Two-tone sequence: high-low beep
    frequencies = [800.0, 600.0]
    return generate_tone_sequence(frequencies, duration=0.3, sample_rate=sample_rate)


def generate_ready_to_copy(sample_rate: int = 16000) -> np.ndarray:
    """Generate 'ready to copy' acknowledgment tone.

    Args:
        sample_rate: Sample rate in Hz

    Returns:
        Audio data
    """
    # Three ascending tones
    frequencies = [600.0, 700.0, 800.0]
    return generate_tone_sequence(frequencies, duration=0.2, sample_rate=sample_rate)


def main():
    """Generate response audio files."""
    parser = argparse.ArgumentParser(description="Generate response audio files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/audio/responses"),
        help="Output directory",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 16000

    # Generate response files
    responses = {
        "signal_received.wav": generate_signal_received(sample_rate),
        "ready_to_copy.wav": generate_ready_to_copy(sample_rate),
    }

    print(f"Generating {len(responses)} response audio files...")
    print(f"Output directory: {args.output_dir}\n")

    for filename, audio in responses.items():
        output_path = args.output_dir / filename
        sf.write(output_path, audio, sample_rate)
        duration = len(audio) / sample_rate
        print(f"Created: {output_path} ({duration:.2f}s, {len(audio)} samples)")

    print(f"\nGenerated {len(responses)} response files in {args.output_dir}")
    print("\nNote: These are placeholder tone sequences. For production use,")
    print("replace with actual TTS-generated responses or recorded audio.")


if __name__ == "__main__":
    main()
