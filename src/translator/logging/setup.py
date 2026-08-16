"""Structured logging setup using structlog.

Configures structlog with:
  - Automatic timestamps (ISO 8601)
  - Log level filtering
  - Stage name injection
  - Console rendering (colorized) for development
  - JSON rendering for production / log aggregation
  - Optional file output

Call `setup_logging()` once at application startup (in the composition root)
before any other module logs.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from translator.core.config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    """Configure structlog and stdlib logging.

    Args:
        config: Logging configuration from AppConfig.
    """
    log_level = getattr(logging, config.level.upper(), logging.INFO)

    # Configure stdlib logging (structlog wraps it)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Shared processors for all log entries
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if config.format == "json":
        # JSON output for production
        renderer = structlog.processors.JSONRenderer()
    else:
        # Pretty console output for development
        renderer = structlog.dev.ConsoleRenderer(
            colors=sys.stderr.isatty(),
        )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure the formatter for stdlib handlers
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Apply formatter to all handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

    # Add file handler if configured
    if config.file_path:
        file_handler = logging.FileHandler(config.file_path, encoding="utf-8")
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),  # Always JSON for files
                ],
            )
        )
        root_logger.addHandler(file_handler)

    # Log the configuration itself
    log = structlog.get_logger("translator.logging")
    log.info(
        "logging_configured",
        level=config.level,
        format=config.format,
        file_path=config.file_path or "(stdout only)",
    )
