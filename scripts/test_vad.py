#!/usr/bin/env python3
"""Test script for VAD functionality."""

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from radio_assistant.vad_detector import VADDetector


def main() -> None:
    """Test VAD on audio files."""
    parser = argparse.ArgumentParser(description="Test VAD functionality")
    parser.add_argument(
        "--input",
        type=str,
        default="tests/audio/vad/",
        help="Input directory or file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vad_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="VAD threshold (0.0-1.0)",
    )
    args = parser.parse_args()

    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Create VAD detector
    vad: VADDetector = VADDetector(
        threshold=args.threshold,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
    )

    # Get input files
    input_path = Path(args.input)
    if input_path.is_dir():
        audio_files = sorted(input_path.glob("*.wav"))
    else:
        audio_files = [input_path]

    if not audio_files:
        logger.error(f"No audio files found in {input_path}")
        sys.exit(1)

    logger.info(f"Processing {len(audio_files)} audio files...")

    results = {}

    for audio_file in audio_files:
        logger.info(f"\nProcessing: {audio_file.name}")

        # Load audio
        try:
            audio_data, sample_rate = sf.read(audio_file, dtype="float32")
        except Exception as e:
            logger.error(f"Failed to load {audio_file}: {e}")
            continue

        logger.info(f"  Duration: {len(audio_data) / sample_rate:.2f}s")
        logger.info(f"  Sample rate: {sample_rate}Hz")

        # Get speech timestamps
        timestamps = vad.get_speech_timestamps(audio_data, sample_rate)

        logger.info(f"  Speech segments: {len(timestamps)}")
        for i, ts in enumerate(timestamps):
            logger.info(f"    Segment {i+1}: {ts['start']:.3f}s - {ts['end']:.3f}s")

        # Quick check with is_speech
        is_speech_detected: bool = vad.is_speech(audio_data)
        logger.info(f"  is_speech(): {is_speech_detected}")

        # Store results
        results[audio_file.name] = {
            "duration": len(audio_data) / sample_rate,
            "sample_rate": sample_rate,
            "speech_segments": timestamps,
            "is_speech": is_speech_detected,
        }

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.success(f"Results saved to: {output_path.absolute()}")

    # Print summary
    logger.info("\nSummary:")
    for filename, result in results.items():
        logger.info(
            f"  {filename}: {len(result['speech_segments'])} segments, "
            f"is_speech={result['is_speech']}"
        )


if __name__ == "__main__":
    main()
