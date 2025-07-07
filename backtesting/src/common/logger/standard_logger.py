import logging
import sys
from typing import Any, Optional, Dict
from datetime import datetime
import json

from .logger_interface import LoggerInterface, LogLevel


class ColorFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
        "BOLD": "\033[1m",  # Bold
        "DIM": "\033[2m",  # Dim
    }

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: bool = True,
    ):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            # Add colors to log level
            level_color = self.COLORS.get(record.levelname, "")
            reset = self.COLORS["RESET"]
            bold = self.COLORS["BOLD"]
            dim = self.COLORS["DIM"]

            # Format timestamp
            timestamp = datetime.fromtimestamp(record.created).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3]

            # Format message with colors
            formatted_msg = (
                f"{dim}{timestamp}{reset} "
                f"{level_color}{bold}[{record.levelname:8}]{reset} "
                f"{bold}{record.name}{reset}: "
                f"{record.getMessage()}"
            )

            # Add context if available
            if hasattr(record, "context") and record.context:
                context_str = json.dumps(
                    record.context, indent=None, separators=(",", ":")
                )
                formatted_msg += f" {dim}| Context: {context_str}{reset}"

            return formatted_msg
        else:
            return super().format(record)


class LevelFilter(logging.Filter):
    """Filter to control log levels for specific handlers"""

    def __init__(self, min_level: int):
        super().__init__()
        self.min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.min_level


class StandardLogger(LoggerInterface):
    """Standard logger implementation using Python's logging library"""

    def __init__(
        self,
        name: str = "restful-api-testing",
        level: LogLevel = LogLevel.INFO,
        console_level: Optional[LogLevel] = None,
        file_level: Optional[LogLevel] = None,
        use_colors: bool = True,
        log_file: Optional[str] = None,
    ):
        self.name = name
        self.logger = logging.getLogger(name)
        self.context: Dict[str, Any] = {}
        self.use_colors = use_colors
        self.log_file = log_file

        # Set default levels
        self._console_level = console_level or level
        self._file_level = file_level or level if log_file else None

        # Clear any existing handlers
        self.logger.handlers.clear()
        self.logger.propagate = False

        # Set logger to DEBUG to allow all messages through
        # Individual handlers will filter based on their levels
        self.logger.setLevel(logging.DEBUG)

        # Setup console handler
        self.console_handler = self._setup_console_handler()

        # Setup file handler if specified
        self.file_handler = None
        if log_file:
            self.file_handler = self._setup_file_handler(log_file)

    def _setup_console_handler(self) -> logging.StreamHandler:
        """Setup colored console handler with level filtering"""
        console_handler = logging.StreamHandler(sys.stdout)

        # Add level filter for console
        level_mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }

        console_filter = LevelFilter(level_mapping[self._console_level])
        console_handler.addFilter(console_filter)

        # Use colored formatter for console
        console_formatter = ColorFormatter(use_colors=self.use_colors)
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(console_handler)
        return console_handler

    def _setup_file_handler(self, log_file: str) -> logging.FileHandler:
        """Setup file handler without colors and with level filtering"""
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")

        # Add level filter for file if file_level is set
        if self._file_level:
            level_mapping = {
                LogLevel.DEBUG: logging.DEBUG,
                LogLevel.INFO: logging.INFO,
                LogLevel.WARNING: logging.WARNING,
                LogLevel.ERROR: logging.ERROR,
                LogLevel.CRITICAL: logging.CRITICAL,
            }

            file_filter = LevelFilter(level_mapping[self._file_level])
            file_handler.addFilter(file_filter)

        # Use plain formatter for file
        file_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)

        self.logger.addHandler(file_handler)
        return file_handler

    def _log_with_context(self, level: int, message: str, *args, **kwargs) -> None:
        """Internal method to log with context"""
        extra = kwargs.get("extra", {})
        if self.context:
            extra["context"] = self.context
        kwargs["extra"] = extra

        self.logger.log(level, message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message"""
        self._log_with_context(logging.DEBUG, message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message"""
        self._log_with_context(logging.INFO, message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message"""
        self._log_with_context(logging.WARNING, message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message"""
        self._log_with_context(logging.ERROR, message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        """Log critical message"""
        self._log_with_context(logging.CRITICAL, message, *args, **kwargs)

    def log(self, level: LogLevel, message: str, *args, **kwargs) -> None:
        """Log message with specified level"""
        level_mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }

        self._log_with_context(level_mapping[level], message, *args, **kwargs)

    def set_level(self, level: LogLevel) -> None:
        """Set minimum log level (backward compatibility - affects both console and file)"""
        self.set_console_level(level)
        if self.file_handler:
            self.set_file_level(level)

    def set_console_level(self, level: LogLevel) -> None:
        """Set minimum log level for console output"""
        self._console_level = level

        # Update console handler filter
        if self.console_handler:
            # Remove old filters
            self.console_handler.filters.clear()

            # Add new filter
            level_mapping = {
                LogLevel.DEBUG: logging.DEBUG,
                LogLevel.INFO: logging.INFO,
                LogLevel.WARNING: logging.WARNING,
                LogLevel.ERROR: logging.ERROR,
                LogLevel.CRITICAL: logging.CRITICAL,
            }

            console_filter = LevelFilter(level_mapping[level])
            self.console_handler.addFilter(console_filter)

    def set_file_level(self, level: LogLevel) -> None:
        """Set minimum log level for file output"""
        if not self.file_handler:
            return

        self._file_level = level

        # Update file handler filter
        self.file_handler.filters.clear()

        level_mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }

        file_filter = LevelFilter(level_mapping[level])
        self.file_handler.addFilter(file_filter)

    def get_console_level(self) -> LogLevel:
        """Get current console log level"""
        return self._console_level

    def get_file_level(self) -> Optional[LogLevel]:
        """Get current file log level (None if no file logging)"""
        return self._file_level

    def add_context(self, **context: Any) -> None:
        """Add contextual information to logs"""
        self.context.update(context)

    def clear_context(self) -> None:
        """Clear contextual information"""
        self.context.clear()

    def add_handler(self, handler: logging.Handler) -> None:
        """Add custom handler"""
        self.logger.addHandler(handler)

    def remove_handler(self, handler: logging.Handler) -> None:
        """Remove handler"""
        self.logger.removeHandler(handler)
