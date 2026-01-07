"""
Logging configuration for the rank-1 analysis pipeline.

Provides structured logging with:
- Console output with rich formatting
- Optional file logging
- Context-aware log prefixes
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from rich.logging import RichHandler
from rich.console import Console

# Global console for rich output
console = Console()

# Logger registry
_loggers: dict[str, logging.Logger] = {}


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    name: str = "rank1"
) -> logging.Logger:
    """
    Set up logging with rich console output and optional file logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        name: Logger name

    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    # Rich console handler
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def get_logger(name: str = "rank1") -> logging.Logger:
    """Get a logger instance, creating it if necessary."""
    if name not in _loggers:
        return setup_logging(name=name)
    return _loggers[name]


@contextmanager
def log_context(context: str, logger: Optional[logging.Logger] = None):
    """Context manager for adding context to log messages."""
    if logger is None:
        logger = get_logger()

    logger.info(f"[bold blue]>>> {context}[/bold blue]", extra={"markup": True})
    try:
        yield logger
    except Exception as e:
        logger.error(f"[bold red]!!! {context} failed: {e}[/bold red]", extra={"markup": True})
        raise
    else:
        logger.info(f"[bold green]<<< {context} complete[/bold green]", extra={"markup": True})


class ProgressLogger:
    """Helper for logging progress of long-running operations."""

    def __init__(self, total: int, description: str, logger: Optional[logging.Logger] = None):
        self.total = total
        self.description = description
        self.logger = logger or get_logger()
        self.current = 0

    def update(self, n: int = 1, message: str = "") -> None:
        """Update progress counter."""
        self.current += n
        pct = 100 * self.current / self.total if self.total > 0 else 0

        if message:
            self.logger.debug(f"{self.description}: {self.current}/{self.total} ({pct:.1f}%) - {message}")
        else:
            self.logger.debug(f"{self.description}: {self.current}/{self.total} ({pct:.1f}%)")

    def done(self) -> None:
        """Mark operation as complete."""
        self.logger.info(f"{self.description}: completed {self.total} items")
