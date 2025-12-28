#!/usr/bin/env python3
"""Test script for audio capture validation."""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from radio_assistant.audio_interface import AudioInterface


def main() -> None:
    """Run audio capture test."""
    parser = argparse.ArgumentParser(description="Test audio capture functionality")
    parser.add_argument(
        "--duration", type=float, default=5.0, help="Duration to capture in seconds"
    )
    parser.add_argument(
        "--output", type=str, default="test_capture.wav", help="Output WAV file path"
    )
    parser.add_argument("--input-device", type=str, default=None, help="Input device name or index")
    parser.add_argument(
        "--list-devices", action="store_true", help="List available devices and exit"
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate in Hz")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels")
    args = parser.parse_args()

    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Create audio interface
    audio: AudioInterface = AudioInterface(
        input_device=args.input_device,
        sample_rate=args.sample_rate,
        channels=args.channels,
    )

    # List devices if requested
    if args.list_devices:
        logger.info("Available audio devices:")
        devices = audio.list_devices()
        for dev in devices:
            logger.info(
                f"  [{dev['index']}] {dev['name']}: "
                f"in={dev['max_input_channels']}, "
                f"out={dev['max_output_channels']}, "
                f"rate={dev['default_samplerate']}"
            )
        return

    # Capture audio
    logger.info(f"Capturing {args.duration}s of audio...")
    try:
        audio_data: np.ndarray = audio.capture_chunk(args.duration)

        # Display statistics
        logger.info("Capture successful:")
        logger.info(f"  Shape: {audio_data.shape}")
        logger.info(f"  dtype: {audio_data.dtype}")
        logger.info(f"  Sample rate: {args.sample_rate} Hz")
        logger.info(f"  Duration: {len(audio_data) / args.sample_rate:.2f}s")
        logger.info(f"  Range: [{audio_data.min():.6f}, {audio_data.max():.6f}]")
        logger.info(f"  Mean: {audio_data.mean():.6f}")
        logger.info(f"  Std: {audio_data.std():.6f}")

        # Check for clipping
        clipped_samples: int = np.sum(np.abs(audio_data) > 0.99)
        if clipped_samples > 0:
            logger.warning(
                f"  WARNING: {clipped_samples} samples appear clipped "
                f"({100 * clipped_samples / len(audio_data):.2f}%)"
            )
        else:
            logger.info("  No clipping detected")

        # Save to WAV file
        output_path = Path(args.output)
        sf.write(output_path, audio_data, args.sample_rate)
        logger.info(f"Saved to: {output_path.absolute()}")

        # Verify playback
        logger.info("Playing back captured audio...")
        audio.play_audio(audio_data)
        logger.info("Playback completed")

        logger.success("Audio capture test PASSED")

    except Exception as e:
        logger.error(f"Audio capture test FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
