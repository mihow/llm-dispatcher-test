# Ham Radio Voice Assistant - Implementation Prompt

## Executive Summary

Build a local-first voice assistant for ham radio that validates the audio processing pipeline before adding LLM capabilities. This Phase 1 "Marco Polo" implementation proves the core concept: listen to radio traffic, detect callsign, respond with confirmation.

## Final Vision (Context for Architecture Decisions)

**Phase 2+ Goals:**
- Continuous monitoring of ham radio traffic
- LLM-based query detection (operator asks callsign + "dispatch" + question)
- Intelligent response generation using local LLMs (Llama 3.2 3B, Phi-3, Gemma 2)
- Action capabilities: web search, file reading, complex reasoning with instructor library
- Local TTS (Piper/Coqui) for voice responses
- Edge deployment: RPi with accelerators, iOS/Android apps
- Hardware PTT control (GPIO, USB serial) alongside VOX support

**Why Phase 1 Matters:**
Validates the entire audio I/O pipeline, timing constraints, VOX triggering, and transcription accuracy before introducing LLM complexity and costs. Each component must work reliably on constrained hardware.

## Phase 1: Marco Polo Test - Detailed Scope

**What It Does:**
1. Captures audio from radio via USB sound card
2. Detects speech using VAD (silero-vad)
3. Transcribes speech using faster-whisper (local, no API)
4. Detects callsign "WSJJ659" in transcription
5. Responds with pre-recorded "signal received" audio via VOX

**What It Validates:**
- Audio I/O pipeline works on target hardware (RPi)
- VAD handles radio-quality audio with noise/static
- Whisper transcription is fast enough (<2-3s) and accurate enough
- VOX triggering works without clipping audio
- Full pipeline completes in acceptable time (<5s end-to-end)

## High-Risk Components (Validate First)

### 1. Audio I/O Timing & Buffering (HIGHEST RISK)
- **Risk:** VAD might miss speech boundaries, buffer management issues
- **Validation:** Step 1-2 checkpoints, real-time performance tests

### 2. VOX Triggering Reliability
- **Risk:** Wrong padding duration, inconsistent triggering, false triggers
- **Validation:** Step 5 checkpoint, manual radio testing

### 3. Whisper Performance on RPi
- **Risk:** Too slow, too much RAM, poor accuracy on radio audio
- **Validation:** Step 3 checkpoint, benchmark on actual hardware

### 4. Callsign Detection Accuracy
- **Risk:** Transcription errors, phonetic alphabet issues, false positives
- **Validation:** Step 4 checkpoint, comprehensive test cases

## Iterative Development Plan

### Step 1: Audio Capture Validation (Checkpoint: `audio-capture`)
**Goal:** Prove USB sound card audio capture works correctly

**Deliverables:**
````
scripts/test_audio_capture.py  # Captures audio, saves WAV, displays stats
radio_assistant/audio_interface.py  # AudioInterface abstraction
tests/unit/test_audio_interface.py  # Unit tests with mocks
tests/integration/test_audio_capture.py  # Integration tests
````

**Implementation Requirements:**
````python
class AudioInterface:
    """Platform-agnostic audio I/O abstraction"""
    def __init__(self, 
                 input_device: Optional[str] = None,
                 output_device: Optional[str] = None,
                 sample_rate: int = 16000,
                 channels: int = 1):
        # Use sounddevice for Linux/RPi
        # Design for future: iOS/Android audio APIs
        ...
    
    def capture_chunk(self, duration_sec: float) -> np.ndarray:
        """Capture audio chunk of specified duration"""
        ...
    
    def play_audio(self, audio_data: np.ndarray):
        """Play audio through output device"""
        ...
    
    def list_devices(self) -> List[Dict]:
        """List available audio devices"""
        ...
````

**Tests:**
- Unit: Mock sounddevice, verify chunk format (shape, dtype, range)
- Integration: Capture from generated sine wave, verify frequency
- Manual: Capture 5 seconds from radio, verify playback

**GitHub Action:**
````yaml
name: Audio Capture Validation
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install system dependencies
        run: sudo apt-get install -y libportaudio2
      - name: Install Python dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest tests/unit/test_audio_interface.py -v
      - name: Run integration tests
        run: pytest tests/integration/test_audio_capture.py -v
      - name: Generate test capture
        run: python scripts/test_audio_capture.py --duration 5 --output test_capture.wav
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: test-audio-captures
          path: test_capture.wav
````

**Success Criteria:**
- All tests pass
- Captured audio has correct sample rate (16kHz)
- No clipping or distortion in test capture
- Audio playback works

---

### Step 2: VAD Integration (Checkpoint: `vad-detection`)
**Goal:** Prove VAD reliably detects speech in radio-quality audio

**Deliverables:**
````
scripts/test_vad.py  # Process test files, log speech segments
radio_assistant/vad_detector.py  # VAD abstraction
tests/audio/vad/  # Test audio files
tests/unit/test_vad_detector.py
tests/integration/test_vad_pipeline.py
````

**Test Audio Files to Create:**
````
tests/audio/vad/
├── speech_clean.wav           # Clear voice, no noise
├── speech_with_static.wav     # Voice + radio static
├── speech_weak_signal.wav     # Faint voice with noise
├── silence.wav                # Pure silence
├── noise_only.wav             # Static/white noise, no speech
├── squelch_tail.wav           # Radio squelch opening/closing
└── multiple_transmissions.wav # Several utterances with gaps
````

**Implementation Requirements:**
````python
class VADDetector:
    """Voice Activity Detection wrapper"""
    def __init__(self, 
                 threshold: float = 0.5,
                 min_speech_duration_ms: int = 250,
                 min_silence_duration_ms: int = 100):
        # Use silero-vad
        ...
    
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Detect if audio chunk contains speech"""
        ...
    
    def get_speech_timestamps(self, 
                             audio: np.ndarray,
                             sample_rate: int) -> List[Dict]:
        """Get start/end timestamps of speech segments"""
        # Returns: [{"start": 0.5, "end": 2.3}, ...]
        ...
````

**Tests:**
- Unit: Mock VAD model, verify logic
- Integration: Process each test file, verify speech/silence classification
- Assert: 
  - `speech_clean.wav` → speech detected
  - `silence.wav` → no speech
  - `noise_only.wav` → no speech
  - `multiple_transmissions.wav` → correct segment count

**GitHub Action:**
````yaml
- name: Run VAD tests
  run: pytest tests/integration/test_vad_pipeline.py -v --json-report
- name: Process test audio files
  run: python scripts/test_vad.py --input tests/audio/vad/ --output vad_results.json
- name: Validate VAD results
  run: python scripts/validate_vad_results.py vad_results.json
- name: Upload results
  uses: actions/upload-artifact@v3
  with:
    name: vad-test-results
    path: vad_results.json
````

**Success Criteria:**
- 100% accuracy on clean test files
- >90% accuracy on noisy test files
- No false positives on silence/noise
- Speech segment timing within ±100ms

---

### Step 3: Whisper Transcription (Checkpoint: `transcription`)
**Goal:** Prove Whisper transcribes radio audio fast and accurately enough

**Deliverables:**
````
scripts/test_transcription.py  # Transcribe files, measure accuracy/latency
radio_assistant/transcription_engine.py
tests/audio/transcription/  # Labeled test audio
tests/unit/test_transcription_engine.py
tests/integration/test_transcription_accuracy.py
tests/benchmarks/test_transcription_performance.py
````

**Critical Test Files (with ground truth transcriptions):**
````
tests/audio/transcription/
├── wsjj659_clear.wav
│   └── transcript.txt: "WSJJ659 dispatch, radio check"
├── wsjj659_phonetic.wav
│   └── transcript.txt: "Whiskey Sierra Juliet Juliet 659 dispatch"
├── wsjj659_noisy.wav
│   └── transcript.txt: "WSJJ659 dispatch, how copy?"
├── wsjj659_rapid.wav
│   └── transcript.txt: "WSJJ659 dispatch quick question"
├── other_callsign.wav
│   └── transcript.txt: "KE7CDB calling CQ on 146.52"
└── radio_quality_actual.wav
    └── transcript.txt: (actual radio recording transcription)
````

**Implementation Requirements:**
````python
class TranscriptionEngine:
    """Speech-to-text wrapper with model abstraction"""
    def __init__(self, 
                 model_size: str = "base",  # base, small, medium
                 device: str = "cpu",
                 compute_type: str = "int8"):
        # Use faster-whisper
        # Future: whisper.cpp, Distil-Whisper
        ...
    
    def transcribe(self, 
                   audio: np.ndarray,
                   sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe audio to text"""
        # Returns: TranscriptionResult with text, confidence, timing
        ...
    
    @dataclass
    class TranscriptionResult:
        text: str
        confidence: float
        duration_ms: int
        segments: List[Dict]  # Word-level timing if needed
````

**Tests:**
- Unit: Mock Whisper, test result parsing
- Integration: Transcribe each test file, compare to ground truth
- Performance: Measure transcription time on each file
- Accuracy: Calculate WER (Word Error Rate) for callsign detection
- Benchmark: Test `base` vs `small` models

**GitHub Action:**
````yaml
- name: Run transcription tests
  run: pytest tests/integration/test_transcription_accuracy.py -v
- name: Benchmark transcription
  run: pytest tests/benchmarks/test_transcription_performance.py --benchmark-only
- name: Transcribe all test files
  run: python scripts/test_transcription.py --input tests/audio/transcription/ --output transcription_results.json
- name: Calculate accuracy metrics
  run: python scripts/calculate_wer.py transcription_results.json
- name: Upload results
  uses: actions/upload-artifact@v3
  with:
    name: transcription-results
    path: |
      transcription_results.json
      wer_report.txt
````

**Success Criteria:**
- Callsign "WSJJ659" correctly transcribed in >95% of test cases
- Transcription latency <2 seconds for `base` model
- Memory usage <1GB on CPU
- Clear degradation path if accuracy insufficient (try `small` model)

---

### Step 4: Callsign Detection Logic (Checkpoint: `callsign-matching`)
**Goal:** Prove robust detection of WSJJ659 in various forms

**Deliverables:**
````
radio_assistant/callsign_detector.py
tests/unit/test_callsign_detector.py  # Comprehensive test cases
````

**Implementation Requirements:**
````python
class CallsignDetector:
    """Detect callsign in transcribed text"""
    def __init__(self, 
                 callsign: str,
                 require_dispatch_keyword: bool = True,
                 phonetic_alphabet: bool = True):
        self.callsign = callsign.upper()
        self.require_dispatch = require_dispatch_keyword
        self.phonetic = phonetic_alphabet
        # Build phonetic mapping if enabled
        ...
    
    def detect(self, transcription: str) -> DetectionResult:
        """Check if callsign present in transcription"""
        # Handle: plain, phonetic, case-insensitive, spacing variations
        ...
    
    @dataclass
    class DetectionResult:
        detected: bool
        confidence: float
        matched_form: Optional[str]  # e.g., "WSJJ659" or "Whiskey Sierra..."
        dispatch_keyword_present: bool
````

**Comprehensive Test Cases:**
````python
test_cases = [
    # Positive cases
    ("WSJJ659 dispatch", True, "exact match"),
    ("wsjj659 dispatch", True, "lowercase"),
    ("W S J J 6 5 9 dispatch", True, "spaced"),
    ("W-S-J-J-6-5-9 dispatch", True, "dashed"),
    ("Whiskey Sierra Juliet Juliet 659 dispatch", True, "phonetic"),
    ("Whiskey Sierra Juliet Juliet six five nine dispatch", True, "phonetic numbers"),
    ("This is WSJJ659 dispatch calling", True, "embedded"),
    ("WSJJ659 dispatch, radio check", True, "with punctuation"),
    
    # Negative cases
    ("WSJJ659", False, "no dispatch keyword"),
    ("KE7XYZ dispatch", False, "different callsign"),
    ("WSJ659 dispatch", False, "incomplete callsign"),
    ("659 dispatch", False, "partial callsign"),
    ("dispatch WSJJ659", False, "wrong order if strict"),
    ("WSJJ658 dispatch", False, "similar but wrong"),
    
    # Edge cases
    ("  WSJJ659   dispatch  ", True, "extra whitespace"),
    ("WSJJ659, dispatch.", True, "punctuation"),
    ("wsjj 659 dispatch", True, "space in callsign"),
]
````

**Tests:**
- Unit: Parametrized tests for all cases
- Unit: Phonetic alphabet mapping accuracy
- Unit: Edge cases (empty string, gibberish, etc.)
- 100% test coverage required

**GitHub Action:**
````yaml
- name: Run callsign detection tests
  run: pytest tests/unit/test_callsign_detector.py -v --cov=radio_assistant.callsign_detector --cov-report=term --cov-report=html
- name: Coverage gate
  run: |
    coverage report --fail-under=100
````

**Success Criteria:**
- All positive cases detected
- Zero false positives
- Phonetic alphabet support working
- 100% code coverage

---

### Step 5: VOX Response Playback (Checkpoint: `vox-playback`)
**Goal:** Prove VOX triggering works without audio clipping

**Deliverables:**
````
scripts/test_vox_trigger.py  # Test various padding durations
radio_assistant/ptt_controller.py  # PTT abstraction
tests/unit/test_ptt_controller.py
tests/audio/responses/signal_received.wav  # Pre-generated TTS response
TESTING.md  # Manual VOX testing procedures
````

**Implementation Requirements:**
````python
class PTTController:
    """Transmission control abstraction (VOX now, hardware PTT future)"""
    def __init__(self, 
                 method: str = "vox",  # Future: gpio, serial
                 vox_padding_ms: int = 300,
                 audio_interface: AudioInterface = None):
        self.method = method
        self.padding_ms = vox_padding_ms
        self.audio = audio_interface
        ...
    
    def transmit(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """Transmit audio using configured method"""
        if self.method == "vox":
            padded = self._add_vox_padding(audio_data, sample_rate)
            self.audio.play_audio(padded)
        # Future: elif self.method == "gpio": ...
        ...
    
    def _add_vox_padding(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Prepend silence to trigger VOX before actual audio"""
        padding_samples = int(self.padding_ms * sample_rate / 1000)
        silence = np.zeros(padding_samples, dtype=audio.dtype)
        return np.concatenate([silence, audio])
````

**Tests:**
- Unit: Mock audio interface, verify padding calculation
- Unit: Test padding with various durations (100ms, 200ms, 300ms, 500ms)
- Integration: Generate padded audio files for manual testing
- Manual: Actual radio VOX test (documented in TESTING.md)

**GitHub Action:**
````yaml
- name: Run PTT controller tests
  run: pytest tests/unit/test_ptt_controller.py -v
- name: Generate test response files
  run: |
    python scripts/test_vox_trigger.py --padding 100 --output test_vox_100ms.wav
    python scripts/test_vox_trigger.py --padding 200 --output test_vox_200ms.wav
    python scripts/test_vox_trigger.py --padding 300 --output test_vox_300ms.wav
    python scripts/test_vox_trigger.py --padding 500 --output test_vox_500ms.wav
- name: Upload test files for manual validation
  uses: actions/upload-artifact@v3
  with:
    name: vox-test-files
    path: test_vox_*.wav
````

**Manual Testing Procedure (TESTING.md):**
````markdown
## VOX Trigger Validation

1. Set radio to VOX mode (Menu 4, level 3-5)
2. Connect computer audio output to radio mic
3. Download test files from GitHub Actions artifacts
4. Play each test file (100ms, 200ms, 300ms, 500ms padding)
5. Verify:
   - Radio transmits cleanly (LED lights)
   - "Signal received" audio is not clipped
   - No false triggers from silence
6. Document minimum working padding duration
7. Update config.yaml with validated padding value
````

**Success Criteria:**
- Unit tests pass
- Padding calculation verified
- Manual test confirms one padding duration works cleanly
- No audio clipping reported

---

### Step 6: End-to-End Pipeline (Checkpoint: `e2e-pipeline`)
**Goal:** Full pipeline integration with comprehensive test coverage

**Deliverables:**
````
radio_assistant/main.py  # Main application
radio_assistant/mock_radio.py  # MockRadioInterface for testing
tests/e2e/test_full_pipeline.py  # E2E scenarios
tests/audio/e2e/  # Complete test scenarios
config.example.yaml  # Example configuration
````

**Complete E2E Test Scenarios:**
````
tests/audio/e2e/
├── scenario_silence.wav
│   └── expected: no_action
├── scenario_noise.wav
│   └── expected: no_action
├── scenario_other_callsign.wav
│   └── expected: transcribe_only (no response)
├── scenario_data_transmission.wav
│   └── expected: ignore
├── scenario_wsjj659_clear.wav
│   └── expected: response_triggered
├── scenario_wsjj659_noisy.wav
│   └── expected: response_triggered
├── scenario_wsjj659_phonetic.wav
│   └── expected: response_triggered
└── scenario_rapid_transmissions.wav
    └── expected: multiple_responses
````

**Implementation Requirements:**
````python
# Configuration
class AppConfig(BaseModel):
    """Application configuration with validation"""
    callsign: str = "WSJJ659"
    vox_padding_ms: int = 300
    vad_threshold: float = 0.5
    whisper_model: str = "base"  # base, small
    chunk_duration_sec: float = 0.5
    ptt_method: str = "vox"
    require_dispatch_keyword: bool = True
    enable_phonetic_detection: bool = True
    log_level: str = "INFO"

# Main application
class RadioAssistant:
    """Main application coordinating all components"""
    def __init__(self, config: AppConfig):
        self.config = config
        self.audio = AudioInterface(...)
        self.vad = VADDetector(...)
        self.transcription = TranscriptionEngine(...)
        self.callsign = CallsignDetector(...)
        self.ptt = PTTController(...)
        self.buffer = []
        ...
    
    def run(self):
        """Main event loop"""
        logger.info(f"Starting RadioAssistant for {self.config.callsign}")
        
        while self.running:
            # Capture audio chunk
            chunk = self.audio.capture_chunk(self.config.chunk_duration_sec)
            
            # Voice activity detection
            if self.vad.is_speech(chunk):
                self.buffer.append(chunk)
                logger.debug("Speech detected, buffering")
            elif self.buffer:
                # Speech ended, process buffer
                logger.info("Processing buffered speech")
                audio_data = np.concatenate(self.buffer)
                
                # Transcribe
                result = self.transcription.transcribe(audio_data)
                logger.info(f"Transcription: {result.text}")
                
                # Check for callsign
                detection = self.callsign.detect(result.text)
                if detection.detected:
                    logger.info(f"Callsign detected: {detection.matched_form}")
                    self._respond()
                else:
                    logger.debug("Callsign not detected")
                
                self.buffer.clear()
    
    def _respond(self):
        """Send response via PTT"""
        response_audio = self._load_response_audio()
        self.ptt.transmit(response_audio)
        logger.info("Response transmitted")

# Mock for testing
class MockRadioInterface(AudioInterface):
    """Plays test audio files instead of live radio"""
    def __init__(self, test_audio_path: str, sample_rate: int = 16000):
        self.test_audio, _ = librosa.load(test_audio_path, sr=sample_rate, mono=True)
        self.position = 0
        self.sample_rate = sample_rate
    
    def capture_chunk(self, duration_sec: float) -> np.ndarray:
        """Return chunks from test file"""
        num_samples = int(duration_sec * self.sample_rate)
        chunk = self.test_audio[self.position:self.position + num_samples]
        self.position += num_samples
        
        if len(chunk) < num_samples:
            # Loop or pad if needed
            chunk = np.pad(chunk, (0, num_samples - len(chunk)))
        
        return chunk
````

**E2E Tests:**
````python
@pytest.mark.parametrize("scenario,expected", [
    ("scenario_silence.wav", "no_action"),
    ("scenario_noise.wav", "no_action"),
    ("scenario_other_callsign.wav", "transcribe_only"),
    ("scenario_wsjj659_clear.wav", "response_triggered"),
    ("scenario_wsjj659_noisy.wav", "response_triggered"),
])
def test_e2e_scenario(scenario, expected):
    """Test complete pipeline with various scenarios"""
    config = AppConfig(callsign="WSJJ659")
    
    # Use mock radio interface
    mock_audio = MockRadioInterface(f"tests/audio/e2e/{scenario}")
    assistant = RadioAssistant(config)
    assistant.audio = mock_audio
    
    # Run pipeline
    result = assistant.process_test_scenario()
    
    # Assert expected behavior
    assert result.action == expected
````

**GitHub Action:**
````yaml
- name: Run E2E tests
  run: pytest tests/e2e/test_full_pipeline.py -v --json-report --html=report.html
- name: Test all scenarios
  run: |
    python scripts/run_all_scenarios.py --output e2e_results.json
- name: Validate results
  run: python scripts/validate_e2e_results.py e2e_results.json
- name: Performance benchmark
  run: pytest tests/benchmarks/test_e2e_performance.py --benchmark-only
- name: Upload reports
  uses: actions/upload-artifact@v3
  with:
    name: e2e-test-results
    path: |
      e2e_results.json
      report.html
      benchmark_results.json
````

**Success Criteria:**
- All E2E scenarios pass
- Full pipeline completes in <5 seconds
- No false positives or false negatives
- Memory usage acceptable for RPi
- Clean separation for future LLM integration

---

### Step 7: Hardware Validation (Checkpoint: `hardware-validation`)
**Goal:** Verify performance on actual Raspberry Pi with real radio

**Deliverables:**
````
TESTING.md  # Complete manual testing procedures
HARDWARE.md  # Hardware setup guide
benchmarks/rpi_performance.json  # Performance data
````

**Manual Testing Procedures (TESTING.md):**
````markdown
## Hardware Setup

### Raspberry Pi Configuration
- Model: [Document actual model used]
- OS: Raspberry Pi OS (Debian-based)
- Python: 3.11+
- RAM: [Document available]
- USB Sound Card: [Document model]

### Radio Configuration
- Radio: Baofeng UV-5R mini
- Connector: Kenwood 2-pin
- VOX Setting: Menu 4, Level [document tested level]
- Frequency: [test frequency]

### Audio Connections
1. Radio speaker (3.5mm) → USB sound card input
2. USB sound card output → Radio mic (2.5mm)
3. PTT pin floating (VOX mode)

## Performance Benchmarks

Run on actual hardware:
```bash
# Install dependencies
pip install -r requirements.txt

# Run performance tests
python scripts/benchmark_hardware.py --output benchmarks/rpi_performance.json

# Test scenarios
python scripts/test_hardware.py --all-scenarios
```

Expected metrics:
- Audio capture latency: <50ms
- VAD processing: <100ms per chunk
- Whisper transcription (base): <2 seconds
- Full pipeline: <5 seconds
- Memory usage: <1GB
- CPU usage: <80% sustained

## Manual Test Cases

1. **Silence Test**
   - Leave radio on squelch
   - Verify: No false triggers for 5 minutes

2. **Other Traffic Test**
   - Monitor active frequency
   - Verify: Other callsigns ignored

3. **Signal Recognition Test**
   - Transmit: "WSJJ659 dispatch, radio check"
   - Verify: Response received within 5 seconds
   - Verify: Response audio not clipped

4. **Noisy Conditions Test**
   - Reduce signal strength or add noise
   - Verify: Still detects callsign

5. **Rapid Transmission Test**
   - Send multiple transmissions quickly
   - Verify: Each transmission gets response

6. **Phonetic Test**
   - Transmit: "Whiskey Sierra Juliet Juliet 659 dispatch"
   - Verify: Detected and responded

## Edge Cases

- Very weak signal (S1-S2)
- Very strong signal (S9+)
- Fast speech
- Slow speech
- Background QRM/QRN
- Simultaneous transmissions (should handle gracefully)

## Documentation Requirements

For each test:
- Record timestamp
- Signal strength
- Response latency
- Any failures or anomalies
- Audio quality assessment
````

**No GitHub Action** - this is manual validation only

**Deliverables:**
- Completed test report with measurements
- Performance benchmark data
- Known limitations documented
- Recommendations for production deployment

---

## Architecture Details

### Project Structure
````
radio-assistant/
├── radio_assistant/
│   ├── __init__.py
│   ├── main.py                    # Main application
│   ├── config.py                  # Configuration with Pydantic
│   ├── audio_interface.py         # Audio I/O abstraction
│   ├── vad_detector.py            # VAD wrapper
│   ├── transcription_engine.py    # Whisper wrapper
│   ├── callsign_detector.py       # Callsign matching logic
│   ├── ptt_controller.py          # PTT abstraction (VOX + future)
│   └── mock_radio.py              # Testing utilities
├── scripts/
│   ├── test_audio_capture.py
│   ├── test_vad.py
│   ├── test_transcription.py
│   ├── test_vox_trigger.py
│   ├── run_all_scenarios.py
│   ├── benchmark_hardware.py
│   └── validate_*.py              # Validation scripts
├── tests/
│   ├── unit/
│   │   ├── test_audio_interface.py
│   │   ├── test_vad_detector.py
│   │   ├── test_transcription_engine.py
│   │   ├── test_callsign_detector.py
│   │   └── test_ptt_controller.py
│   ├── integration/
│   │   ├── test_audio_capture.py
│   │   ├── test_vad_pipeline.py
│   │   └── test_transcription_accuracy.py
│   ├── e2e/
│   │   └── test_full_pipeline.py
│   ├── benchmarks/
│   │   ├── test_transcription_performance.py
│   │   └── test_e2e_performance.py
│   └── audio/
│       ├── vad/                   # VAD test files
│       ├── transcription/         # Transcription test files
│       ├── responses/             # Pre-generated responses
│       └── e2e/                   # Full scenario files
├── config.example.yaml
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── TESTING.md
├── HARDWARE.md
├── ARCHITECTURE.md
└── CLAUDE.md                      # Development learnings
````

### Technology Stack

**Core Dependencies:**
````
# Audio processing
sounddevice>=0.4.6          # Cross-platform audio I/O
numpy>=1.24.0               # Audio array processing
librosa>=0.10.0             # Audio utilities

# Speech processing
faster-whisper>=0.10.0      # Local transcription
silero-vad>=4.0.0          # Voice activity detection

# Configuration & CLI
pydantic>=2.5.0            # Config validation
typer>=0.9.0               # CLI interface
pyyaml>=6.0                # Config files

# Logging
loguru>=0.7.0              # Better logging

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-benchmark>=4.0.0
pytest-html>=4.0.0
````

**Development Dependencies:**
````
black>=23.0.0              # Code formatting
mypy>=1.7.0               # Type checking
ruff>=0.1.0               # Linting
````

### CLI Interface
````bash
# Basic usage
radio-assistant --callsign WSJJ659

# With options
radio-assistant \
  --callsign WSJJ659 \
  --input-device "USB Audio" \
  --output-device "USB Audio" \
  --vox-padding 300 \
  --whisper-model base \
  --config config.yaml \
  --log-level DEBUG

# List audio devices
radio-assistant --list-devices

# Run with test scenario
radio-assistant --test-mode --test-file tests/audio/e2e/scenario_wsjj659_clear.wav

# Benchmark mode
radio-assistant --benchmark
````

### Configuration File (config.yaml)
````yaml
# Operator configuration
callsign: WSJJ659
require_dispatch_keyword: true
enable_phonetic_detection: true

# Audio configuration
audio:
  input_device: null  # null = default
  output_device: null
  sample_rate: 16000
  channels: 1

# VAD configuration
vad:
  threshold: 0.5
  min_speech_duration_ms: 250
  min_silence_duration_ms: 100

# Transcription configuration
transcription:
  model_size: base  # base, small, medium
  device: cpu       # cpu, cuda
  compute_type: int8

# PTT configuration
ptt:
  method: vox              # vox, gpio, serial
  vox_padding_ms: 300
  # Future GPIO config
  # gpio_pin: 17
  # gpio_active_low: true

# Application configuration
app:
  chunk_duration_sec: 0.5
  response_audio: tests/audio/responses/signal_received.wav
  log_level: INFO
  log_file: radio_assistant.log
````

### Code Style Requirements

**Python Standards:**
````python
# Type annotations required
def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
    """Transcribe audio to text"""
    result: str = self._process(audio)
    return TranscriptionResult(text=result)

# Namespace imports preferred
import radio_assistant.models as models
config = models.AppConfig()

# NOT: from radio_assistant.models import AppConfig

# Pydantic for configuration
class AppConfig(BaseModel):
    callsign: str
    vox_padding_ms: int = 300
    
    @field_validator('callsign')
    def validate_callsign(cls, v):
        if not v:
            raise ValueError("Callsign required")
        return v.upper()

# Dataclasses for simple structures
@dataclass
class TranscriptionResult:
    text: str
    confidence: float
    duration_ms: int

# Logging, not print
logger.info(f"Transcription: {result.text}")
logger.debug(f"Confidence: {result.confidence}")

# Variable assignment before return
def detect(self, text: str) -> bool:
    normalized: str = text.upper().strip()
    found: bool = self.callsign in normalized
    return found

# NOT: return self.callsign in text.upper().strip()
````

**Error Handling:**
````python
try:
    result = self.transcription.transcribe(audio)
except Exception as e:
    logger.error(f"Transcription failed: {e}")
    # Graceful degradation
    return None
````

**Documentation:**
````python
def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
    """
    Transcribe audio to text using Whisper.
    
    Args:
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz (default: 16000)
    
    Returns:
        TranscriptionResult containing text and metadata
    
    Raises:
        ValueError: If audio is empty or invalid
        RuntimeError: If Whisper model not loaded
    """
    ...
````

## Testing Strategy

### Test Pyramid
````
     E2E (few)
       /\
      /  \
   Integration (some)
    /    \
   /      \
  Unit (many)
````

### Test Coverage Requirements
- Unit tests: >90% coverage
- Integration tests: Critical paths covered
- E2E tests: All user scenarios covered
- Benchmarks: Performance regression detection

### CI/CD Pipeline

**Per-PR Checks:**
1. Code formatting (black)
2. Type checking (mypy)
3. Linting (ruff)
4. Unit tests (pytest)
5. Integration tests
6. Coverage report
7. Benchmark comparison

**Per-Checkpoint Additional Checks:**
- Step 1: Audio capture validation
- Step 2: VAD accuracy validation
- Step 3: Transcription accuracy + performance
- Step 4: 100% callsign detection coverage
- Step 5: VOX test file generation
- Step 6: Full E2E scenarios

### Automated Test Execution
````yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run ruff
        run: ruff check .
      - name: Run mypy
        run: mypy radio_assistant/

  test:
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install system dependencies
        run: sudo apt-get install -y libportaudio2
      - name: Install Python dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt
      - name: Run unit tests
        run: pytest tests/unit/ -v --cov --cov-report=xml
      - name: Run integration tests
        run: pytest tests/integration/ -v
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmarks
        run: pytest tests/benchmarks/ --benchmark-only --benchmark-json=benchmark.json
      - name: Store benchmark
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: benchmark.json
````

## Success Criteria Summary

### Functional Requirements
- ✅ Captures audio from USB sound card reliably
- ✅ Detects speech vs silence/noise accurately
- ✅ Transcribes radio-quality audio with acceptable accuracy
- ✅ Detects WSJJ659 in multiple forms (plain, phonetic, variants)
- ✅ Responds via VOX without audio clipping
- ✅ Full pipeline completes in <5 seconds

### Technical Requirements
- ✅ All automated tests pass without hardware
- ✅ 90%+ code coverage
- ✅ Runs on Raspberry Pi with acceptable performance
- ✅ Modular architecture supports future enhancements
- ✅ Clear abstraction for PTT method swapping
- ✅ Clean separation points for LLM integration

### Development Requirements
- ✅ AI coding agent can verify functionality via tests
- ✅ GitHub Actions validate each checkpoint
- ✅ No manual intervention needed for automated validation
- ✅ Clear documentation for manual hardware testing
- ✅ Performance benchmarks establish baselines

### Documentation Requirements
- ✅ README.md: Setup and usage
- ✅ TESTING.md: Verification procedures
- ✅ HARDWARE.md: Hardware setup guide
- ✅ ARCHITECTURE.md: Design decisions and future roadmap
- ✅ CLAUDE.md: Development learnings and notes

## Future Expansion Path (Phase 2+)

### LLM Integration Points
````python
# Replace CallsignDetector with LLM decision
class LLMDispatchDetector:
    """Use LLM to determine if transmission requires response"""
    def detect(self, transcription: str) -> DispatchDecision:
        # Use instructor for structured output
        decision = self.llm.query(
            f"Is this directed at {self.callsign} dispatch? {transcription}"
        )
        return decision

@dataclass
class DispatchDecision:
    is_for_me: bool
    confidence: float
    should_respond: bool
    query_type: Optional[str]  # weather, location, info, etc.
````

### Response Generation
````python
# Replace pre-recorded audio with LLM + TTS
class LLMResponseGenerator:
    """Generate intelligent responses using local LLM"""
    def generate(self, query: str, context: Dict) -> str:
        # Use instructor for structured response
        response = self.llm.generate(
            query=query,
            context=context,
            tools=[web_search, file_reader]
        )
        return response.text

# Add TTS
class TTSEngine:
    """Text-to-speech using Piper/Coqui"""
    def synthesize(self, text: str) -> np.ndarray:
        # Generate audio from text
        ...
````

### Hardware PTT Control
````python
# GPIO method for RPi
class GPIOPTTController(PTTController):
    """Control PTT via GPIO pin"""
    def __init__(self, gpio_pin: int = 17, active_low: bool = True):
        import RPi.GPIO as GPIO
        self.pin = gpio_pin
        GPIO.setup(self.pin, GPIO.OUT)
        ...
    
    def transmit(self, audio_data: np.ndarray):
        # Key PTT
        GPIO.output(self.pin, GPIO.LOW if self.active_low else GPIO.HIGH)
        time.sleep(0.1)  # Key delay
        
        # Play audio
        self.audio.play_audio(audio_data)
        
        # Wait for audio to finish
        time.sleep(len(audio_data) / self.sample_rate)
        
        # Unkey PTT
        GPIO.output(self.pin, GPIO.HIGH if self.active_low else GPIO.LOW)
````

## Implementation Approach

### Start with Step 1
Begin with audio capture validation (the most fundamental component). Each subsequent step builds on validated functionality.

### Iterative Development
- Complete each checkpoint fully before moving to next
- Each checkpoint has its own PR
- All tests must pass before merging
- Document learnings in CLAUDE.md

### Parallel Work Possible
- Test audio file creation can happen alongside code development
- Documentation can be written incrementally
- CI/CD can be set up early

### Risk Mitigation
- Validate high-risk components first (audio I/O, Whisper performance)
- Have fallback options (model size, padding duration)
- Document all hardware-specific issues found
- Maintain clear rollback points (checkpoints)

## Questions for Clarification

Before starting implementation:

1. **Raspberry Pi specifics:**
   - Which model? (affects Whisper model size choice)
   - Available RAM?
   - Accelerator hardware present?

2. **Audio preferences:**
   - Sample rate preference? (8kHz radio-appropriate vs 16kHz Whisper-preferred)
   - Mono vs stereo?

3. **Whisper model:**
   - Start with `base` or `small`?
   - Acceptable accuracy threshold?
   - Acceptable latency threshold?

4. **VOX settings:**
   - Current VOX level on radio?
   - Timeout duration?

5. **Testing:**
   - Should phonetic detection be Phase 1 or Phase 2?
   - Desired verbosity for logs during development vs production?

## Getting Started

1. **Clone and setup:**
````bash
   git clone <repo>
   cd radio-assistant
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt -r requirements-dev.txt
````

2. **Start with Step 1 checkpoint:**
````bash
   git checkout -b checkpoint/audio-capture
   # Implement audio capture
   pytest tests/unit/test_audio_interface.py
   git commit -m "Implement audio capture"
   git push origin checkpoint/audio-capture
   # Open PR for review
````

3. **Iterate through checkpoints** following the validation sequence

This is a complete, AI-agent-verifiable implementation plan. Ready to begin with Step 1?
