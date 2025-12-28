#!/usr/bin/env python3
"""Test script for VOX trigger with various padding durations."""

import argparse
from pathlib import Path
import soundfile as sf
from radio_assistant.ptt_controller import PTTController


def main():
    """Generate VOX test files with various padding durations."""
    parser = argparse.ArgumentParser(description="Generate VOX trigger test files")
    parser.add_argument(
        "--padding",
        type=int,
        default=300,
        help="VOX padding duration in milliseconds",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tests/audio/responses/signal_received.wav"),
        help="Input audio file to transmit",
    )
    args = parser.parse_args()

    # Load input audio
    print(f"Loading audio: {args.input}")
    audio, sample_rate = sf.read(args.input, dtype="float32")
    print(f"  Duration: {len(audio)/sample_rate:.2f}s")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Channels: {audio.ndim}")

    # Create PTT controller
    print(f"\nAdding VOX padding: {args.padding}ms")
    ptt = PTTController(vox_padding_ms=args.padding)

    # Add padding
    padded_audio = ptt._add_vox_padding(audio, sample_rate)

    padding_samples = int(args.padding * sample_rate / 1000)
    total_duration = len(padded_audio) / sample_rate

    print(f"  Padding: {padding_samples} samples ({args.padding}ms)")
    print(f"  Total duration: {total_duration:.2f}s")

    # Output file path
    if args.output is None:
        args.output = Path(f"test_vox_{args.padding}ms.wav")

    # Save output
    sf.write(args.output, padded_audio, sample_rate)
    print(f"\nSaved to: {args.output}")

    print("\nManual testing instructions:")
    print("1. Set radio to VOX mode (Menu 4, level 3-5)")
    print("2. Connect computer audio output to radio mic input")
    print("3. Play the generated file through your radio")
    print("4. Verify:")
    print("   - Radio transmits cleanly (TX LED lights)")
    print("   - Audio is not clipped at the beginning")
    print("   - No false triggers from silence")
    print("\nIf audio is clipped, try increasing --padding value")


if __name__ == "__main__":
    main()
