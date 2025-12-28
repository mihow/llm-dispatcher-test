#!/usr/bin/env python3
"""Generate test audio files for VAD testing."""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_sine_wave(frequency: float, duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate a sine wave."""
    t: np.ndarray = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine: np.ndarray = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return sine


def generate_speech_like_signal(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate a speech-like signal using multiple sine waves."""
    # Fundamental frequency + harmonics
    frequencies = [200, 400, 600, 800, 1200]
    weights = [1.0, 0.5, 0.3, 0.2, 0.15]

    signal: np.ndarray = np.zeros(int(sample_rate * duration), dtype=np.float32)
    for freq, weight in zip(frequencies, weights):
        signal += weight * generate_sine_wave(freq, duration, sample_rate)

    # Normalize
    signal = signal / np.abs(signal).max() * 0.8

    # Add amplitude modulation (speech-like envelope)
    t: np.ndarray = np.linspace(0, duration, len(signal))
    envelope: np.ndarray = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)  # 4 Hz modulation
    signal = signal * envelope

    return signal.astype(np.float32)


def generate_noise(
    duration: float, sample_rate: int = 16000, noise_level: float = 0.1
) -> np.ndarray:
    """Generate white noise."""
    noise: np.ndarray = np.random.randn(int(sample_rate * duration)).astype(np.float32)
    noise = noise / np.abs(noise).max() * noise_level
    return noise


def add_noise(signal: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    """Add noise to signal at specified SNR."""
    signal_power: float = float(np.mean(signal**2))
    noise_power: float = signal_power / (10 ** (snr_db / 10))
    noise: np.ndarray = np.random.randn(len(signal)).astype(np.float32)
    noise = noise / np.sqrt(np.mean(noise**2)) * np.sqrt(noise_power)
    return (signal + noise).astype(np.float32)


def main() -> None:
    """Generate all VAD test audio files."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    output_dir = Path("tests/audio/vad")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate: int = 16000

    # 1. Clean speech
    logger.info("Generating speech_clean.wav...")
    speech = generate_speech_like_signal(2.0, sample_rate)
    sf.write(output_dir / "speech_clean.wav", speech, sample_rate)

    # 2. Speech with static
    logger.info("Generating speech_with_static.wav...")
    speech = generate_speech_like_signal(2.0, sample_rate)
    speech_noisy = add_noise(speech, snr_db=10)
    sf.write(output_dir / "speech_with_static.wav", speech_noisy, sample_rate)

    # 3. Weak signal (very noisy speech)
    logger.info("Generating speech_weak_signal.wav...")
    speech = generate_speech_like_signal(2.0, sample_rate)
    speech_weak = add_noise(speech, snr_db=0)  # 0 dB SNR
    sf.write(output_dir / "speech_weak_signal.wav", speech_weak, sample_rate)

    # 4. Silence
    logger.info("Generating silence.wav...")
    silence = np.zeros(int(2.0 * sample_rate), dtype=np.float32)
    sf.write(output_dir / "silence.wav", silence, sample_rate)

    # 5. Noise only
    logger.info("Generating noise_only.wav...")
    noise = generate_noise(2.0, sample_rate, noise_level=0.1)
    sf.write(output_dir / "noise_only.wav", noise, sample_rate)

    # 6. Squelch tail (short burst then silence)
    logger.info("Generating squelch_tail.wav...")
    squelch = np.concatenate(
        [
            generate_noise(0.2, sample_rate, noise_level=0.3),  # Brief noise burst
            np.zeros(int(1.8 * sample_rate), dtype=np.float32),  # Silence
        ]
    )
    sf.write(output_dir / "squelch_tail.wav", squelch, sample_rate)

    # 7. Multiple transmissions
    logger.info("Generating multiple_transmissions.wav...")
    speech1 = generate_speech_like_signal(1.0, sample_rate)
    silence1 = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
    speech2 = generate_speech_like_signal(0.8, sample_rate)
    silence2 = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
    speech3 = generate_speech_like_signal(1.2, sample_rate)

    multiple = np.concatenate([speech1, silence1, speech2, silence2, speech3])
    sf.write(output_dir / "multiple_transmissions.wav", multiple, sample_rate)

    logger.success(f"Generated {7} VAD test audio files in {output_dir}")


if __name__ == "__main__":
    main()
