# ADR-005: Thread-Safe Hotkeys via Async Queue

## Status
Accepted

## Context
Global hotkeys (`pynput`) and system tray (`pystray`) run on separate OS threads. They must communicate with the asyncio event loop running the pipeline. Direct `asyncio.run_coroutine_threadsafe()` calls create race conditions and are hard to test.

## Decision
Use thread-safe `asyncio.Queue` with dedicated dispatch tasks:

- Hotkey/Tray threads: `queue.put_nowait(coro)` → non-blocking
- Event loop: `await queue.get()` → executes coroutine on loop thread
- Single consumer task per component (`_dispatch_loop`) serializes callbacks
- Eliminates `run_coroutine_threadsafe` and cross-thread async issues

Applied to both `HotkeyManager` and `SystemTray`.

## Consequences

### Positive
- **Thread Safety**: All async code runs on single event loop thread
- **Deterministic Ordering**: Callbacks processed FIFO
- **Testability**: Can inject mock queue in tests
- **Backpressure**: Queue full → log warning, drop event (non-blocking)

### Negative
- **Latency**: Extra queue hop adds microseconds
- **Complexity**: More code than direct `call_soon_threadsafe`

### Risks
- Queue overflow if callbacks slower than events (mitigated by logging + bounded queue)