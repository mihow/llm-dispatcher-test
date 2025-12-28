#!/usr/bin/env python3
"""Generate TTS audio files to replace placeholder tone files."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from gtts import gTTS
from loguru import logger
from pydub import AudioSegment
from pydub.effects import normalize

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def add_noise(signal: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    """Add white noise to signal at specified SNR."""
    signal_power: float = float(np.mean(signal**2))
    noise_power: float = signal_power / (10 ** (snr_db / 10))
    noise: np.ndarray = np.random.randn(len(signal)).astype(np.float32)
    noise = noise / np.sqrt(np.mean(noise**2)) * np.sqrt(noise_power)
    return (signal + noise).astype(np.float32)


def generate_radio_static(duration_ms: int, level: float = 0.05) -> AudioSegment:
    """Generate radio static noise."""
    sample_rate = 16000
    samples = int(duration_ms * sample_rate / 1000)
    noise = np.random.randn(samples).astype(np.float32) * level
    # Convert to AudioSegment
    audio_segment = AudioSegment(
        noise.tobytes(), frame_rate=sample_rate, sample_width=4, channels=1
    )
    return audio_segment


def text_to_speech_file(
    text: str,
    output_path: Path,
    add_static: bool = False,
    snr_db: float = 10.0,
    sample_rate: int = 16000,
) -> None:
    """Generate TTS audio and save to file.

    Args:
        text: Text to convert to speech
        output_path: Output file path
        add_static: Whether to add radio static noise
        snr_db: Signal-to-noise ratio in dB (only used if add_static=True)
        sample_rate: Target sample rate (default 16000 Hz)
    """
    try:
        # Generate TTS using gTTS (Google TTS)
        tts = gTTS(text=text, lang="en", slow=False)

        # Save to temporary mp3 file
        temp_mp3 = output_path.with_suffix(".mp3")
        tts.save(str(temp_mp3))

        # Convert mp3 to wav with desired sample rate
        audio = AudioSegment.from_mp3(str(temp_mp3))

        # Convert to mono if not already
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Normalize audio to peak level (prevents clipping)
        audio = normalize(audio, headroom=0.1)

        # Add static if requested
        if add_static:
            # Convert to numpy for SNR-based noise addition
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            # Normalize to [-1, 1] range
            max_val = np.abs(samples).max()
            if max_val > 0:
                samples = samples / max_val
            else:
                samples = samples * 0.0

            # Add noise at specified SNR
            noisy_samples = add_noise(samples, snr_db=snr_db)

            # Clip to prevent overflow and scale to int16 range
            noisy_samples = np.clip(noisy_samples, -1.0, 1.0)
            noisy_samples = (noisy_samples * 32767 * 0.9).astype(np.int16)  # 0.9 for headroom

            audio = AudioSegment(
                noisy_samples.tobytes(),
                frame_rate=audio.frame_rate,
                sample_width=2,
                channels=1,
            )

        # Resample to target sample rate
        if audio.frame_rate != sample_rate:
            audio = audio.set_frame_rate(sample_rate)

        # Export as WAV
        audio.export(str(output_path), format="wav")

        # Clean up temp file
        temp_mp3.unlink(missing_ok=True)

        logger.info(f"Generated: {output_path.name} ({len(audio)}ms)")

    except Exception as e:
        logger.error(f"Failed to generate {output_path.name}: {e}")
        raise


def generate_multiple_transmissions(output_path: Path, sample_rate: int = 16000) -> None:
    """Generate audio with multiple speech transmissions separated by silence."""
    # Generate three short transmissions
    segments = []

    # First transmission
    text1 = "WSJJ659 calling"
    tts1 = gTTS(text=text1, lang="en", slow=False)
    temp1 = output_path.parent / "temp1.mp3"
    tts1.save(str(temp1))
    audio1 = AudioSegment.from_mp3(str(temp1))
    segments.append(audio1)

    # Silence (500ms)
    segments.append(AudioSegment.silent(duration=500))

    # Second transmission
    text2 = "Do you copy?"
    tts2 = gTTS(text=text2, lang="en", slow=False)
    temp2 = output_path.parent / "temp2.mp3"
    tts2.save(str(temp2))
    audio2 = AudioSegment.from_mp3(str(temp2))
    segments.append(audio2)

    # Silence (500ms)
    segments.append(AudioSegment.silent(duration=500))

    # Third transmission
    text3 = "Over"
    tts3 = gTTS(text=text3, lang="en", slow=False)
    temp3 = output_path.parent / "temp3.mp3"
    tts3.save(str(temp3))
    audio3 = AudioSegment.from_mp3(str(temp3))
    segments.append(audio3)

    # Combine all segments
    combined = segments[0]
    for segment in segments[1:]:
        combined += segment

    # Convert to mono and normalize to peak level
    if combined.channels > 1:
        combined = combined.set_channels(1)
    combined = normalize(combined, headroom=0.1)

    # Resample to target sample rate
    if combined.frame_rate != sample_rate:
        combined = combined.set_frame_rate(sample_rate)

    # Export
    combined.export(str(output_path), format="wav")

    # Clean up temp files
    temp1.unlink(missing_ok=True)
    temp2.unlink(missing_ok=True)
    temp3.unlink(missing_ok=True)

    logger.info(f"Generated: {output_path.name} (multiple transmissions, {len(combined)}ms)")


def main() -> None:
    """Generate all TTS audio files."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    sample_rate = 16000

    logger.info("Generating TTS audio files...")

    # =========================================================================
    # VAD Test Audio
    # =========================================================================
    logger.info("\n=== VAD Test Audio ===")
    vad_dir = Path("tests/audio/vad")
    vad_dir.mkdir(parents=True, exist_ok=True)

    # Clean speech
    text_to_speech_file(
        "This is WSJJ659 calling, do you copy?",
        vad_dir / "speech_clean.wav",
        add_static=False,
        sample_rate=sample_rate,
    )

    # Speech with static (10 dB SNR)
    text_to_speech_file(
        "This is WSJJ659 calling, do you copy?",
        vad_dir / "speech_with_static.wav",
        add_static=True,
        snr_db=10,
        sample_rate=sample_rate,
    )

    # Weak signal (0 dB SNR - very noisy)
    text_to_speech_file(
        "This is WSJJ659 calling",
        vad_dir / "speech_weak_signal.wav",
        add_static=True,
        snr_db=0,
        sample_rate=sample_rate,
    )

    # Multiple transmissions
    generate_multiple_transmissions(
        vad_dir / "multiple_transmissions.wav", sample_rate=sample_rate
    )

    # Silence (keep existing - just zeros)
    silence = np.zeros(int(2.0 * sample_rate), dtype=np.float32)
    sf.write(vad_dir / "silence.wav", silence, sample_rate)
    logger.info(f"Generated: silence.wav")

    # Noise only (keep existing - white noise)
    noise = np.random.randn(int(2.0 * sample_rate)).astype(np.float32) * 0.1
    sf.write(vad_dir / "noise_only.wav", noise, sample_rate)
    logger.info(f"Generated: noise_only.wav")

    # Squelch tail (short burst then silence)
    squelch_audio = AudioSegment.silent(duration=100)  # Brief static
    static = generate_radio_static(100, level=0.05)
    squelch_audio = static + AudioSegment.silent(duration=1900)
    squelch_audio = squelch_audio.set_frame_rate(sample_rate)
    squelch_audio.export(str(vad_dir / "squelch_tail.wav"), format="wav")
    logger.info(f"Generated: squelch_tail.wav")

    # =========================================================================
    # Transcription Test Audio
    # =========================================================================
    logger.info("\n=== Transcription Test Audio ===")
    trans_dir = Path("tests/audio/transcription")
    trans_dir.mkdir(parents=True, exist_ok=True)

    # Clear callsign
    text_to_speech_file(
        "This is WSJJ659 calling",
        trans_dir / "wsjj659_clear.wav",
        add_static=False,
        sample_rate=sample_rate,
    )

    # Phonetic alphabet
    text_to_speech_file(
        "Whiskey Sierra Juliet Juliet six five nine calling",
        trans_dir / "wsjj659_phonetic.wav",
        add_static=False,
        sample_rate=sample_rate,
    )

    # Noisy
    text_to_speech_file(
        "This is WSJJ659 do you copy",
        trans_dir / "wsjj659_noisy.wav",
        add_static=True,
        snr_db=5,
        sample_rate=sample_rate,
    )

    # Rapid
    text_to_speech_file(
        "WSJJ659 WSJJ659 come in",
        trans_dir / "wsjj659_rapid.wav",
        add_static=False,
        sample_rate=sample_rate,
    )

    # Other callsign
    text_to_speech_file(
        "This is K6ABC calling",
        trans_dir / "other_callsign.wav",
        add_static=False,
        sample_rate=sample_rate,
    )

    # Hello world
    text_to_speech_file(
        "Hello world this is a test",
        trans_dir / "hello_world.wav",
        add_static=False,
        sample_rate=sample_rate,
    )

    # Empty silence (keep as silence)
    silence = np.zeros(int(1.0 * sample_rate), dtype=np.float32)
    sf.write(trans_dir / "empty_silence.wav", silence, sample_rate)
    logger.info(f"Generated: empty_silence.wav")

    # =========================================================================
    # E2E Test Audio
    # =========================================================================
    logger.info("\n=== E2E Test Audio ===")
    e2e_dir = Path("tests/audio/e2e")
    e2e_dir.mkdir(parents=True, exist_ok=True)

    # Clear WSJJ659 scenario
    text_to_speech_file(
        "This is WSJJ659 calling, do you copy?",
        e2e_dir / "scenario_wsjj659_clear.wav",
        add_static=False,
        sample_rate=sample_rate,
    )

    # Noisy WSJJ659 scenario
    text_to_speech_file(
        "WSJJ659 calling, radio check",
        e2e_dir / "scenario_wsjj659_noisy.wav",
        add_static=True,
        snr_db=8,
        sample_rate=sample_rate,
    )

    # Other callsign scenario
    text_to_speech_file(
        "This is K6ABC calling for a radio check",
        e2e_dir / "scenario_other_callsign.wav",
        add_static=False,
        sample_rate=sample_rate,
    )

    # Silence scenario
    silence = np.zeros(int(2.0 * sample_rate), dtype=np.float32)
    sf.write(e2e_dir / "scenario_silence.wav", silence, sample_rate)
    logger.info(f"Generated: scenario_silence.wav")

    # Noise only scenario
    noise = np.random.randn(int(2.0 * sample_rate)).astype(np.float32) * 0.1
    sf.write(e2e_dir / "scenario_noise.wav", noise, sample_rate)
    logger.info(f"Generated: scenario_noise.wav")

    # =========================================================================
    # Response Audio (keep existing if satisfactory)
    # =========================================================================
    logger.info("\n=== Response Audio ===")
    response_dir = Path("tests/audio/responses")
    response_dir.mkdir(parents=True, exist_ok=True)

    # Signal received
    text_to_speech_file(
        "Signal received, standing by",
        response_dir / "signal_received.wav",
        add_static=False,
        sample_rate=sample_rate,
    )

    # Ready to copy
    text_to_speech_file(
        "Ready to copy, go ahead",
        response_dir / "ready_to_copy.wav",
        add_static=False,
        sample_rate=sample_rate,
    )

    logger.info("\n✅ All TTS audio files generated successfully!")


if __name__ == "__main__":
    main()
