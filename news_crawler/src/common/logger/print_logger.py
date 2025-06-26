import sys
from typing import Any, Dict, Optional
from datetime import datetime

from .logger_interface import LoggerInterface, LogLevel


class PrintLogger(LoggerInterface):
    """Simple print-based logger implementation"""

    # ANSI color codes (same as StandardLogger)
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

    LEVEL_ORDER = {
        LogLevel.DEBUG: 0,
        LogLevel.INFO: 1,
        LogLevel.WARNING: 2,
        LogLevel.ERROR: 3,
        LogLevel.CRITICAL: 4,
    }

    def __init__(
        self,
        name: str = "print-logger",
        level: LogLevel = LogLevel.INFO,
        use_colors: bool = True,
    ):
        self.name = name
        self.level = level
        self.context: Dict[str, Any] = {}
        self.use_colors = use_colors and sys.stdout.isatty()

    def _should_log(self, level: LogLevel) -> bool:
        """Check if message should be logged based on current level"""
        return self.LEVEL_ORDER[level] >= self.LEVEL_ORDER[self.level]

    def _format_message(self, level: LogLevel, message: str) -> str:
        """Format message with colors and timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        if self.use_colors:
            level_color = self.COLORS.get(level.value, "")
            reset = self.COLORS["RESET"]
            bold = self.COLORS["BOLD"]
            dim = self.COLORS["DIM"]

            formatted_msg = (
                f"{dim}{timestamp}{reset} "
                f"{level_color}{bold}[{level.value:8}]{reset} "
                f"{bold}{self.name}{reset}: "
                f"{message}"
            )

            # Add context if available
            if self.context:
                context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
                formatted_msg += f" {dim}| Context: {context_str}{reset}"

            return formatted_msg
        else:
            formatted_msg = f"{timestamp} [{level.value:8}] {self.name}: {message}"
            if self.context:
                context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
                formatted_msg += f" | Context: {context_str}"
            return formatted_msg

    def _print_message(self, level: LogLevel, message: str) -> None:
        """Print formatted message to appropriate stream"""
        if not self._should_log(level):
            return

        formatted_msg = self._format_message(level, message)

        # Use stderr for errors and critical
        if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            print(formatted_msg, file=sys.stderr)
        else:
            print(formatted_msg)

    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message"""
        if args:
            message = message % args
        self._print_message(LogLevel.DEBUG, message)

    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message"""
        if args:
            message = message % args
        self._print_message(LogLevel.INFO, message)

    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message"""
        if args:
            message = message % args
        self._print_message(LogLevel.WARNING, message)

    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message"""
        if args:
            message = message % args
        self._print_message(LogLevel.ERROR, message)

    def critical(self, message: str, *args, **kwargs) -> None:
        """Log critical message"""
        if args:
            message = message % args
        self._print_message(LogLevel.CRITICAL, message)

    def log(self, level: LogLevel, message: str, *args, **kwargs) -> None:
        """Log message with specified level"""
        if args:
            message = message % args
        self._print_message(level, message)

    def set_level(self, level: LogLevel) -> None:
        """Set minimum log level"""
        self.level = level

    def set_console_level(self, level: LogLevel) -> None:
        """Set minimum log level for console output (same as set_level for PrintLogger)"""
        self.set_level(level)

    def set_file_level(self, level: LogLevel) -> None:
        """Set minimum log level for file output (no-op for PrintLogger)"""
        pass  # PrintLogger doesn't support file output

    def get_console_level(self) -> LogLevel:
        """Get current console log level"""
        return self.level

    def get_file_level(self) -> Optional[LogLevel]:
        """Get current file log level (None for PrintLogger)"""
        return None

    def add_context(self, **context: Any) -> None:
        """Add contextual information to logs"""
        self.context.update(context)

    def clear_context(self) -> None:
        """Clear contextual information"""
        self.context.clear()
