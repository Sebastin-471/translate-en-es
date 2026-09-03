# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the translate-en-es project.
Each ADR documents a significant architectural decision, its context, and consequences.

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [001](001-hexagonal-architecture.md) | Hexagonal Architecture (Ports & Adapters) | Accepted | 2026-08-28 |
| [002](002-async-pipeline-queues.md) | Async Pipeline with Typed Queues | Accepted | 2026-08-28 |
| [003](003-gpu-model-manager.md) | Centralized GPU Model Manager | Accepted | 2026-08-28 |
| [004](004-configuration-management.md) | Environment-Specific Configuration with Hot-Reload | Accepted | 2026-08-28 |
| [005](005-thread-safe-hotkeys.md) | Thread-Safe Hotkeys via Async Queue | Accepted | 2026-08-28 |
| [006](006-audio-device-manager.md) | Audio Device Manager for Enumeration and Fallback | Accepted | 2026-08-28 |
| [007](007-model-preparation-pipeline.md) | Offline Model Preparation Pipeline | Accepted | 2026-08-28 |
| [008](008-observability.md) | OpenTelemetry Tracing and Prometheus Metrics | Accepted | 2026-08-28 |
| [009](009-engine-plugin-registry.md) | Plugin Registry for Engine Implementations | Accepted | 2026-08-28 |

## Template

Use this template for new ADRs:

```markdown
# ADR-NNN: Title

## Status
Proposed | Accepted | Rejected | Deprecated | Superseded

## Context
What is the issue that we're seeing that is motivating this decision or change?

## Decision
What is the change that we're proposing and/or doing?

## Consequences
What becomes easier or more difficult to do because of this change?
- Positive consequences
- Negative consequences
- Risks
```