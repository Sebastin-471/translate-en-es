# ADR-002: Async Pipeline with Typed Queues

## Status
Accepted

## Context
Real-time audio translation requires concurrent processing: audio capture (blocking I/O), VAD (fast ML), ASR (heavy ML), MT (heavy ML), UI rendering (GUI thread). Sequential processing would introduce unacceptable latency. Thread-based concurrency is complex and error-prone.

## Decision
Use `asyncio` with typed `asyncio.Queue[T]` for inter-stage communication:

```
AudioSource → Queue[AudioChunk] → VAD → Queue[VADSegment] → ASR → Queue[TranscriptResult] → MT → Queue[TranslationResult] → UI
```

- Each stage runs as an independent `asyncio.Task`
- `StageRunner` generic class handles queue consumption, processing, and metric collection
- Heavy ML inference offloaded via `asyncio.to_thread()` to keep event loop free
- Partial results (`is_partial=True`) flow through same queues for real-time UX
- `PipelineShutdown` sentinel propagates graceful shutdown

## Consequences

### Positive
- **True Concurrency**: All stages process simultaneously; no blocking
- **Backpressure**: Bounded queues prevent memory exhaustion
- **Observability**: Per-stage latency metrics via queue timestamps
- **Testability**: Mock engines + in-memory queues = fast deterministic tests
- **Type Safety**: Generic queues enforce payload types at compile time

### Negative
- **Complexity**: Async/await throughout; requires understanding of event loop
- **Thread Offloading**: `asyncio.to_thread()` adds overhead; not true parallelism
- **Debugging**: Stack traces across async boundaries can be confusing

### Risks
- Event loop blocking if synchronous code accidentally called (mitigated by code review)
- Queue deadlocks if stages don't propagate shutdown sentinel