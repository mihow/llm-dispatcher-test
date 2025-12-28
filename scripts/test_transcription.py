#!/usr/bin/env python3
"""Interactive test script for transcription engine."""

import argparse
from pathlib import Path
import soundfile as sf
from radio_assistant.transcription_engine import TranscriptionEngine


def main():
    """Test transcription engine on audio files."""
    parser = argparse.ArgumentParser(description="Test transcription engine")
    parser.add_argument(
        "--audio",
        type=Path,
        help="Audio file to transcribe (WAV format, 16kHz)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        help="Test all files in directory",
    )
    args = parser.parse_args()

    print(f"Loading Whisper model: {args.model} on {args.device}")
    engine = TranscriptionEngine(
        model_size=args.model,
        device=args.device,
        compute_type="int8" if args.device == "cpu" else "float16",
    )

    if args.test_dir:
        # Test all files in directory
        test_files = sorted(args.test_dir.glob("*.wav"))
        print(f"\nTesting {len(test_files)} files from {args.test_dir}\n")

        for audio_path in test_files:
            print(f"File: {audio_path.name}")

            # Load audio
            audio, sr = sf.read(audio_path, dtype="float32")
            print(f"  Duration: {len(audio)/sr:.2f}s")

            # Load ground truth if available
            txt_path = audio_path.with_suffix(".txt")
            if txt_path.exists():
                ground_truth = txt_path.read_text().strip()
                print(f"  Expected: '{ground_truth}'")

            # Transcribe
            result = engine.transcribe(audio, sample_rate=sr)

            print(f"  Result:   '{result.text}'")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Time: {result.duration_ms}ms")
            print()

    elif args.audio:
        # Test single file
        print(f"\nTranscribing: {args.audio}")

        audio, sr = sf.read(args.audio, dtype="float32")
        print(f"Duration: {len(audio)/sr:.2f}s")

        result = engine.transcribe(audio, sample_rate=sr)

        print(f"\nTranscription: '{result.text}'")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Time: {result.duration_ms}ms")
        print(f"\nSegments: {len(result.segments)}")
        for i, seg in enumerate(result.segments):
            print(f"  [{i}] {seg['start']:.2f}s - {seg['end']:.2f}s: '{seg['text']}'")

    else:
        parser.print_help()
        print("\nExample usage:")
        print(
            "  python scripts/test_transcription.py --audio tests/audio/transcription/hello_world.wav"
        )
        print("  python scripts/test_transcription.py --test-dir tests/audio/transcription")


if __name__ == "__main__":
    main()
