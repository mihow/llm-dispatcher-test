# Home Assistant Voice Pipeline Architecture Research

**Date:** 2025-12-28
**Focus:** Async patterns, pipeline design, and continuous operation reliability
**Source Repository:** https://github.com/home-assistant/core

---

## Executive Summary

Home Assistant's voice pipeline architecture provides a well-structured approach to async audio processing with a clear state machine, event-driven design, and integration patterns suitable for long-running operations. The architecture is built on Python's asyncio with careful attention to resource management, though some reliability issues remain in edge cases (reconnection, state management).

**Key Takeaways:**
1. Event-driven pipeline with distinct stages (wake word → STT → intent → TTS)
2. Wyoming protocol provides async peer-to-peer communication for voice services
3. Satellite entity pattern enables standardized device integration
4. Resource cleanup uses `async_on_unload` pattern for automatic cleanup registration
5. Entity lifecycle hooks (`async_added_to_hass`, `async_will_remove_from_hass`) manage continuous operation

---

## 1. Async Architecture

### 1.1 Event Loop Integration

Home Assistant uses a **single-threaded asyncio event loop** that manages a queue of tasks executed sequentially. Tasks are implemented as coroutines using async/await syntax.

**Key Pattern:** `hass.async_create_task()` instead of `asyncio.create_task()`
- Uses a lower-level interface allowing tasks to be scheduled immediately
- Provides better control over task scheduling within the Home Assistant framework
- Reference: [Home Assistant: Concurrency Model](https://www.thecandidstartup.org/2025/10/20/home-assistant-concurrency-model.html)

**Critical Design Principle:**
> "Once an automation starts running it will keep running until it waits, or performs external IO."

This means synchronous operations (like state reads/writes) won't suspend coroutines, but I/O operations (network calls, database queries) will.

### 1.2 Wyoming Protocol Async Patterns

The Wyoming protocol implements peer-to-peer communication for voice assistants using **JSONL format + PCM audio**.

**AsyncTcpClient Implementation** (`wyoming/client.py`):
```python
# Connection establishment
self._reader, self._writer = await asyncio.open_connection(host=self.host, port=self.port)

# Event communication
async_read_event(self._reader)
async_write_event(event, self._writer)

# Cleanup
writer.close()
await writer.wait_closed()
```

**Context Manager Pattern:**
- Implements `__aenter__` and `__aexit__` for structured resource management
- Ensures `connect()` on entry and `disconnect()` on exit
- Nullifies reader/writer references before closing to prevent reuse

**Audio Streaming:**
- TTS: Uses `AsyncGenerator[bytes]` for streaming audio chunks
- STT: Processes audio via `async_process_audio_stream()` method
- Two modes: streaming (yields chunks immediately) vs non-streaming (buffers in BytesIO)

Reference: [Wyoming Protocol GitHub](https://github.com/rhasspy/wyoming)

### 1.3 Home Assistant TTS Async Patterns

**Wyoming TTS Component** (`homeassistant/components/wyoming/tts.py`):

**Memory Management:**
- Non-streaming: Accumulates chunks in `io.BytesIO()` buffer
- Streaming: Sends WAV header with zero frames, then yields audio chunks directly
- Avoids full-message buffering through immediate yield operations

**Error Handling:**
```python
try:
    # Network operations
except (OSError, WyomingError):
    return (None, None)
```

**Cleanup:**
- Explicit `wav_writer.close()` to finalize WAV files
- Context managers (`async with`, `with`) ensure resource release
- `finally` blocks guarantee `await client.disconnect()` execution

Reference: [Wyoming TTS Source](https://github.com/home-assistant/core/blob/dev/homeassistant/components/wyoming/tts.py)

---

## 2. Voice Pipeline State Machine

### 2.1 Pipeline Stages

The Assist pipeline executes **four sequential stages**:

1. **Wake word detection** - Identifies activation phrase
2. **Speech to text (STT)** - Converts audio to text
3. **Intent recognition** - Processes user intent
4. **Text to speech (TTS)** - Generates spoken response

Users can customize execution via `start_stage` and `end_stage` parameters for partial pipeline runs.

Reference: [Assist Pipelines Developer Docs](https://developers.home-assistant.io/docs/voice/pipelines/)

### 2.2 Pipeline Events

**Event Types:**
- `run-start` / `run-end` - Pipeline lifecycle boundaries
- `wake_word-start` / `wake_word-end` - Wake word detection events
- `stt-start` / `vad-start` / `vad-end` / `stt-end` - Speech recognition phases
- `intent-start` / `intent-progress` / `intent-end` - Intent processing updates
- `tts-start` / `tts-end` - Audio generation events
- `error` - Failure notifications with error codes

**Event-Driven Architecture:**
- Pipeline emits events over WebSocket connection
- Enables real-time communication between clients and pipeline system
- `intent-progress` event provides intermediate updates during processing

### 2.3 Satellite Entity State Machine

The `AssistSatelliteEntity` defines **four operational states**:

1. `listening_wake_word` - Awaiting wake phrase activation
2. `listening_command` - Actively streaming voice input
3. `processing` - Audio stream stopped, awaiting pipeline results
4. `responding` - Delivering synthesized speech response

**Pipeline Integration:**
- `async_accept_pipeline_from_satellite()` - Forwards audio to assist pipeline and manages entity state
- Abstract `on_pipeline_event()` - Must be overridden to handle pipeline events (stt start, stt end, etc.)
- `tts_response_finished()` - Called upon TTS playback completion

Reference: [Satellite Entity Architecture](https://github.com/home-assistant/architecture/discussions/1114)

### 2.4 Error Handling

**13 Specific Error Codes:**
- Provider availability errors (e.g., `stt-provider-missing`)
- Stream failures
- Timeouts (e.g., `wake-word-timeout`)
- Unsupported configurations
- `tts-failed`

**Known Issues:**
- Pipeline can hang after certain commands (e.g., "nevermind")
- State can get stuck in "Responding" state
- Timeout errors if calls not answered within 30 seconds
- End-of-speech detection failures causing commands to timeout

References:
- [Voice Assistant Pipeline Hung Issue](https://github.com/home-assistant/core/issues/105872)
- [Assist Satellite Stuck in Responding State](https://github.com/home-assistant/core/issues/142363)
- [End of Speech Detection Issue](https://github.com/home-assistant/core/issues/122177)

---

## 3. Integration Patterns

### 3.1 Wyoming Satellite Implementation

**Main Event Loop Structure** (`wyoming_satellite/__main__.py`):
```python
async def main() -> None:
    satellite_task = asyncio.create_task(satellite.run(), name='satellite run')
    await server.run(partial(SatelliteEventHandler, wyoming_info, satellite, args))
```

**Three Satellite Implementations:**
- `WakeStreamingSatellite` - Buffers audio until wake word detected
- `VadStreamingSatellite` - Streams after voice activity detected
- `AlwaysStreamingSatellite` - Continuous audio streaming

**Error Recovery:**
- `try/finally` blocks ensure satellite stops cleanly: `finally: await satellite.stop()`
- KeyboardInterrupt handling for safe termination
- Dependency validation at startup
- File existence validation for WAV files
- Logging for troubleshooting

Reference: [Wyoming Satellite GitHub](https://github.com/rhasspy/wyoming-satellite)

### 3.2 Voice Activity Detection (VAD)

**Implementation Approaches:**
- **WebRTC VAD**: Used in GLaSSIST project, tested by Google
- **Pyannote VAD**: Alternative for voice activity detection
- **ESPHome**: Provides `on_stt_vad_start` and `on_stt_vad_end` automation hooks

**VAD Parameters (WebRTC):**
- Sensitivity: 0-3 (3 = most sensitive)
- Silence threshold: seconds
- Sample rate
- Frame duration: 10/20/30ms

**Privacy Consideration:**
> "VAD only detects the presence of human speech without transcribing or understanding the words."

**Client-Side VAD:**
> "Clients should avoid unnecessary audio streaming by using a local voice activity detector (VAD) to only start streaming when human speech is detected."

References:
- [Assist Pipelines - VAD](https://developers.home-assistant.io/docs/voice/pipelines/)
- [GLaSSIST GitHub](https://github.com/SmolinskiP/GLaSSIST)
- [ESPHome Voice Assistant](https://esphome.io/components/voice_assistant/)

### 3.3 Service Abstraction

**Speech-to-Text Entity** (`homeassistant/components/stt/__init__.py`):
- Core requirement: **"Only streaming content is allowed!"**
- Must implement `async_process_audio_stream()` method
- Declares support for: languages, audio formats (WAV/OGG), codecs (PCM/Opus), bit rates, sample rates, channels

**Wyoming Integration Pattern:**
- External services connect via Wyoming protocol
- URI-based connection scheme: `tcp://127.0.0.1:10023`
- Supports TCP, Unix domain sockets, and Standard I/O
- JSON events + optional binary payload

Reference: [Speech-to-Text Entity Docs](https://developers.home-assistant.io/docs/core/entity/stt/)

### 3.4 Whisper Integration

**Wyoming Whisper Implementation:**
- Processing via `async_process_audio_stream` method
- Receives async stream of audio bytes
- Accumulates audio data in memory
- Enforces 25MB limit for cloud APIs
- Uses `asyncio.to_thread()` to avoid blocking event loop

**Performance Characteristics:**
- Raspberry Pi 4: ~8 seconds per command
- Wyoming allows remote installation for better performance
- TensorRT optimization: 0.07s vs 0.40s for 20-second clip

References:
- [Whisper Integration Docs](https://www.home-assistant.io/integrations/whisper/)
- [Faster Whisper Custom Component](https://github.com/AlexxIT/FasterWhisper)
- [TensorRT Acceleration Blog](https://jonahmay.net/accelerating-speech-to-text-stt-for-home-assistant-with-tensorrt/)

---

## 4. Long-Running Reliability

### 4.1 Resource Cleanup Patterns

**Config Entry Cleanup:**
```python
async def async_unload_entry(hass: HomeAssistant, entry: MyConfigEntry) -> bool:
    if (unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS)):
        entry.runtime_data.listener()  # Finish cleanup
    return unload_ok
```

**Automatic Cleanup Registration:**
- `entry.async_on_unload(callback)` - Registers callbacks called during unload or setup failure
- Callbacks execute if `async_setup_entry` raises ConfigEntryError, ConfigEntryAuthFailed, or ConfigEntryNotReady
- Example: `entry.async_on_unload(entry.add_update_listener(update_listener))`

**Purpose:**
> "Clean up all entities, unsubscribe any event listener and close all connections."

Reference: [Config Entry Unloading](https://developers.home-assistant.io/docs/core/integration-quality-scale/rules/config-entry-unloading/)

### 4.2 Entity Lifecycle Hooks

**`async_added_to_hass()`:**
- Called when entity has entity_id and hass object assigned
- Before first write to state machine
- Use cases: restoring state, subscribing to updates, registering callbacks

**`async_will_remove_from_hass()`:**
- Called when entity is about to be removed
- Use cases: disconnecting from servers, unsubscribing, releasing resources

**Update Patterns:**
- **Polling:** Implement `update()` or `async_update()`, set `should_poll = True`
- **Push-based:** Set `should_poll = False`, call `async_schedule_update_ha_state()` manually

**Critical Design Rule:**
> "Properties should always only return information from memory and not do I/O (like network requests)."

Reference: [Entity Lifecycle Docs](https://developers.home-assistant.io/docs/core/entity/)

### 4.3 Memory Management

**Known Issues:**
- Memory leaks reported with satellites continuing to stream audio without processing
- Out-of-memory issues have occurred in production
- Importance of cleanup callbacks to avoid leaks

**Best Practices:**
- Store listeners in `runtime_data` for tracking
- Clean up subscriptions in `async_unload_entry`
- Use `async_on_unload` for automatic cleanup
- Avoid buffering entire audio streams when possible

References:
- [Memory Leak Discussion](https://community.home-assistant.io/t/memory-leak-with-my-configuration-any-tip-to-find-the-culprit-module/584309)
- [Out of Memory Issue](https://github.com/home-assistant/core/issues/69695)

### 4.4 Reliability Challenges

**Wyoming Satellite Reconnection Issues:**
- After power cycle, satellite detects wake words but HA doesn't recognize connection
- Wyoming Protocol integration holds open connection that doesn't drop on restart
- Workaround: Reboot HA core or disable/re-enable Wyoming integration
- Some users report disconnect errors every 90 seconds
- Pi Zero 2 installations become less stable over weeks

**Performance Optimization:**
- On-device wake word processing (microWakeWord) for faster responses
- Streaming TTS responses reduce latency by 10x
- Minimize exposed entities to Assist (performance penalty with more entity names)

**Continuous Operation Mode:**
- Multiple conversation turns supported
- Similar to analog phone interaction
- Refractory period prevents rapid re-triggering

References:
- [Wyoming Satellite Reconnection Issue](https://github.com/home-assistant/core/issues/108001)
- [Wyoming Satellites Won't Reconnect](https://community.home-assistant.io/t/wyoming-satellites-wont-reconnect-with-home-assistant/777236)
- [Best Practices for Assist](https://www.home-assistant.io/voice_control/best_practices/)
- [Voice Chapter 10 Blog](https://www.home-assistant.io/blog/2025/06/25/voice-chapter-10/)

---

## 5. Key Architectural Patterns for llm-dispatcher

### 5.1 Async Patterns to Adopt

1. **Use `asyncio.create_task()` for pipeline execution**
   - Don't block the event loop with long-running operations
   - Use proper task naming for debugging

2. **Implement streaming audio processing**
   - Use `AsyncGenerator[bytes]` for audio chunks
   - Avoid buffering entire audio in memory
   - Process chunks as they arrive

3. **Event-driven state machine**
   - Emit events at each pipeline stage
   - Allow clients to track progress in real-time
   - Provide intermediate updates during processing

4. **Context managers for resource cleanup**
   - Use `async with` for network connections
   - Implement `__aenter__` and `__aexit__` for custom clients
   - Nullify references before closing

### 5.2 Pipeline Design Patterns

1. **Clear stage separation**
   - Wake word → STT → Intent → TTS
   - Allow partial pipeline execution (start_stage, end_stage)
   - Each stage emits start/end events

2. **Configurable pipeline components**
   - Plugin-based architecture for STT/TTS services
   - Service abstraction layer
   - URI-based connection configuration

3. **Comprehensive error handling**
   - Define specific error codes for each failure type
   - Emit error events with context
   - Implement timeout handling

### 5.3 Reliability Patterns

1. **Automatic cleanup registration**
   - Use callback registration pattern (`async_on_unload`)
   - Clean up on both normal shutdown and errors
   - Track resources in structured data (like `runtime_data`)

2. **Lifecycle hooks**
   - `async_added_to_hass` equivalent for initialization
   - `async_will_remove_from_hass` equivalent for cleanup
   - Separate I/O from property access

3. **Reconnection handling**
   - Detect stale connections
   - Implement automatic reconnection with backoff
   - Provide manual retry mechanism

4. **VAD for efficiency**
   - Client-side VAD to reduce unnecessary streaming
   - Configurable sensitivity and silence thresholds
   - Privacy-respecting (no transcription)

### 5.4 Patterns to Avoid

1. **Don't hold persistent connections without timeout**
   - Home Assistant has issues with satellites not reconnecting
   - Implement connection health checks

2. **Don't buffer entire audio streams**
   - Use streaming/chunking approach
   - Set size limits (e.g., 25MB)

3. **Don't do I/O in property getters**
   - Keep properties synchronous and memory-based
   - Use dedicated update methods for I/O

4. **Don't skip cleanup on errors**
   - Always use try/finally
   - Register cleanup callbacks early

---

## 6. Implementation Recommendations

### 6.1 For llm-dispatcher Service Mode

**Audio Processing:**
```python
async def process_audio_stream(
    audio_stream: AsyncGenerator[bytes, None]
) -> AsyncGenerator[TranscriptionEvent, None]:
    """Process audio stream with event emission."""
    try:
        yield TranscriptionEvent(type="start")

        # Process chunks as they arrive
        async for chunk in audio_stream:
            # Process chunk without buffering entire stream
            result = await stt_service.process_chunk(chunk)
            if result:
                yield TranscriptionEvent(type="progress", text=result)

        yield TranscriptionEvent(type="end")
    except Exception as e:
        yield TranscriptionEvent(type="error", error_code="stt-failed", message=str(e))
    finally:
        await cleanup_resources()
```

**Service Lifecycle:**
```python
class DispatcherService:
    async def start(self):
        """Initialize service with cleanup registration."""
        await self.stt_client.connect()
        self.cleanup_callbacks.append(self.stt_client.disconnect)

    async def stop(self):
        """Execute all cleanup callbacks."""
        for callback in self.cleanup_callbacks:
            try:
                await callback()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
```

**Error Recovery:**
```python
async def handle_connection_error(self, error: Exception):
    """Recover from connection errors with backoff."""
    for attempt in range(MAX_RETRIES):
        await asyncio.sleep(BACKOFF_DELAY * (2 ** attempt))
        try:
            await self.reconnect()
            return
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
```

### 6.2 Testing Considerations

**Continuous Operation Tests:**
- Run for extended periods (hours/days) to detect memory leaks
- Monitor resource usage over time
- Test reconnection after network disruptions
- Verify cleanup callbacks are called

**Async Pattern Tests:**
- Test that long operations don't block event loop
- Verify streaming works with slow clients
- Test error handling doesn't leak resources
- Validate state transitions

---

## 7. Sources

### Official Documentation
- [Assist Pipelines Developer Docs](https://developers.home-assistant.io/docs/voice/pipelines/)
- [Speech-to-Text Entity Docs](https://developers.home-assistant.io/docs/core/entity/stt/)
- [Entity Lifecycle Docs](https://developers.home-assistant.io/docs/core/entity/)
- [Config Entry Unloading](https://developers.home-assistant.io/docs/core/integration-quality-scale/rules/config-entry-unloading/)
- [Whisper Integration](https://www.home-assistant.io/integrations/whisper/)
- [Wyoming Protocol Integration](https://www.home-assistant.io/integrations/wyoming/)
- [Voice Assistant Integration](https://www.home-assistant.io/integrations/voice_assistant/)
- [Best Practices for Assist](https://www.home-assistant.io/voice_control/best_practices/)
- [Troubleshooting Assist](https://www.home-assistant.io/voice_control/troubleshooting/)

### Architecture & Design
- [Home Assistant: Concurrency Model](https://www.thecandidstartup.org/2025/10/20/home-assistant-concurrency-model.html)
- [Satellite Entity Architecture](https://github.com/home-assistant/architecture/discussions/1114)

### GitHub Repositories
- [Wyoming Protocol](https://github.com/rhasspy/wyoming)
- [Wyoming Satellite](https://github.com/rhasspy/wyoming-satellite)
- [Faster Whisper Custom Component](https://github.com/AlexxIT/FasterWhisper)
- [StreamAssist Custom Component](https://github.com/AlexxIT/StreamAssist)
- [GLaSSIST Desktop Voice Assistant](https://github.com/SmolinskiP/GLaSSIST)

### Source Code References
- [Wyoming TTS Component](https://github.com/home-assistant/core/blob/dev/homeassistant/components/wyoming/tts.py)
- [Entity Platform](https://github.com/home-assistant/core/blob/dev/homeassistant/helpers/entity_platform.py)

### Blog Posts & Community
- [Voice Chapter 10 - Assist Today](https://www.home-assistant.io/blog/2025/06/25/voice-chapter-10/)
- [Voice Chapter 11 - Multilingual Assistants](https://www.home-assistant.io/blog/2025/10/22/voice-chapter-11)
- [Voice Chapter 8 - Assist in the Home](https://www.home-assistant.io/blog/2024/12/19/voice-chapter-8-assist-in-the-home/)
- [TensorRT STT Acceleration](https://jonahmay.net/accelerating-speech-to-text-stt-for-home-assistant-with-tensorrt/)

### Issues & Discussions
- [Voice Assistant Pipeline Hung](https://github.com/home-assistant/core/issues/105872)
- [Assist Satellite Stuck in Responding State](https://github.com/home-assistant/core/issues/142363)
- [End of Speech Detection Issue](https://github.com/home-assistant/core/issues/122177)
- [Wyoming Satellite Reconnection Issue](https://github.com/home-assistant/core/issues/108001)
- [Wyoming Satellites Won't Reconnect](https://community.home-assistant.io/t/wyoming-satellites-wont-reconnect-with-home-assistant/777236)
- [Creating Persistent Async Tasks](https://community.home-assistant.io/t/creating-persistent-async-tasks/180257)

---

**End of Document**
