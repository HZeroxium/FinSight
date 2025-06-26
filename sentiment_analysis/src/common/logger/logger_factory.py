from typing import Optional, Dict, Any
from enum import Enum

from .logger_interface import LoggerInterface, LogLevel
from .standard_logger import StandardLogger
from .print_logger import PrintLogger


class LoggerType(Enum):
    """Available logger types"""

    STANDARD = "standard"
    PRINT = "print"


class LoggerFactory:
    """Factory for creating logger instances"""

    _instances: Dict[str, LoggerInterface] = {}

    @classmethod
    def get_logger(
        cls,
        name: str = "restful-api-testing",
        logger_type: LoggerType = LoggerType.STANDARD,
        level: LogLevel = LogLevel.INFO,
        console_level: Optional[LogLevel] = None,
        file_level: Optional[LogLevel] = None,
        use_colors: bool = True,
        log_file: Optional[str] = None,
        **kwargs: Any,
    ) -> LoggerInterface:
        """
        Get or create a logger instance

        Args:
            name: Logger name
            logger_type: Type of logger to create
            level: Default log level (backward compatibility)
            console_level: Log level for console output (overrides level)
            file_level: Log level for file output (overrides level)
            use_colors: Whether to use colored output
            log_file: Optional log file path (for StandardLogger)
            **kwargs: Additional arguments for logger construction

        Returns:
            Logger instance
        """
        cache_key = f"{name}_{logger_type.value}"

        if cache_key not in cls._instances:
            if logger_type == LoggerType.STANDARD:
                logger = StandardLogger(
                    name=name,
                    level=level,
                    console_level=console_level,
                    file_level=file_level,
                    use_colors=use_colors,
                    log_file=log_file,
                    **kwargs,
                )
            elif logger_type == LoggerType.PRINT:
                logger = PrintLogger(
                    name=name,
                    level=console_level or level,
                    use_colors=use_colors,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown logger type: {logger_type}")

            cls._instances[cache_key] = logger

        return cls._instances[cache_key]

    @classmethod
    def create_logger(
        cls,
        name: str,
        logger_type: LoggerType = LoggerType.STANDARD,
        level: LogLevel = LogLevel.INFO,
        console_level: Optional[LogLevel] = None,
        file_level: Optional[LogLevel] = None,
        use_colors: bool = True,
        log_file: Optional[str] = None,
        **kwargs: Any,
    ) -> LoggerInterface:
        """
        Create a new logger instance (not cached)

        Args:
            name: Logger name
            logger_type: Type of logger to create
            level: Default log level (backward compatibility)
            console_level: Log level for console output (overrides level)
            file_level: Log level for file output (overrides level)
            use_colors: Whether to use colored output
            log_file: Optional log file path (for StandardLogger)
            **kwargs: Additional arguments for logger construction

        Returns:
            New logger instance
        """
        if logger_type == LoggerType.STANDARD:
            return StandardLogger(
                name=name,
                level=level,
                console_level=console_level,
                file_level=file_level,
                use_colors=use_colors,
                log_file=log_file,
                **kwargs,
            )
        elif logger_type == LoggerType.PRINT:
            return PrintLogger(
                name=name, level=console_level or level, use_colors=use_colors, **kwargs
            )
        else:
            raise ValueError(f"Unknown logger type: {logger_type}")

    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached logger instances"""
        cls._instances.clear()
