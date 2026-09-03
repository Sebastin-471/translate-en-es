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
    from translator.core.config import AppConfig, LoggingConfig


def setup_logging(config: AppConfig | LoggingConfig) -> None:
    """Configure structlog and stdlib logging.

    Args:
        config: Logging configuration from AppConfig or LoggingConfig.
    """
    # Handle both AppConfig and LoggingConfig
    if hasattr(config, "logging"):
        logging_config = config.logging
        app_config = config
    else:
        logging_config = config
        app_config = None

    log_level = getattr(logging, logging_config.level.upper(), logging.INFO)

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

    if logging_config.format == "json":
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
    if logging_config.file_path:
        file_handler = logging.FileHandler(logging_config.file_path, encoding="utf-8")
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),  # Always JSON for files
                ],
            )
        )
        root_logger.addHandler(file_handler)

    # Initialize observability if AppConfig provided
    if app_config:
        _init_observability(app_config)

    # Log the configuration itself
    log = structlog.get_logger("translator.logging")
    log.info(
        "logging_configured",
        level=logging_config.level,
        format=logging_config.format,
        file_path=logging_config.file_path or "(stdout only)",
    )


def _init_observability(config: AppConfig) -> None:
    """Initialize tracing and metrics from config."""
    # Initialize tracing
    try:
        from translator.observability.tracing import init_tracing
        otlp_endpoint = None  # Could be added to config later
        init_tracing(
            service_name="translate-en-es",
            otlp_endpoint=otlp_endpoint,
            enabled=config.logging.level != "DEBUG",  # Enable in non-debug
        )
    except ImportError:
        pass

    # Initialize metrics
    try:
        from translator.observability.metrics import init_metrics
        init_metrics(enabled=True)
    except ImportError:
        pass
