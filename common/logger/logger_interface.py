from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional


class LogLevel(Enum):
    """Log level enumeration"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggerInterface(ABC):
    """Abstract base interface for all logger implementations"""

    @abstractmethod
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message"""
        pass

    @abstractmethod
    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message"""
        pass

    @abstractmethod
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message"""
        pass

    @abstractmethod
    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message"""
        pass

    @abstractmethod
    def critical(self, message: str, *args, **kwargs) -> None:
        """Log critical message"""
        pass

    @abstractmethod
    def log(self, level: LogLevel, message: str, *args, **kwargs) -> None:
        """Log message with specified level"""
        pass

    @abstractmethod
    def set_level(self, level: LogLevel) -> None:
        """Set minimum log level"""
        pass

    @abstractmethod
    def add_context(self, **context: Any) -> None:
        """Add contextual information to logs"""
        pass

    @abstractmethod
    def clear_context(self) -> None:
        """Clear contextual information"""
        pass

    @abstractmethod
    def set_console_level(self, level: LogLevel) -> None:
        """Set minimum log level for console output"""
        pass

    @abstractmethod
    def set_file_level(self, level: LogLevel) -> None:
        """Set minimum log level for file output"""
        pass

    @abstractmethod
    def get_console_level(self) -> LogLevel:
        """Get current console log level"""
        pass

    @abstractmethod
    def get_file_level(self) -> Optional[LogLevel]:
        """Get current file log level (None if no file logging)"""
        pass
