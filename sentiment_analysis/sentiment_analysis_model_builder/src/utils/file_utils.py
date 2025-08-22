# utils/file_utils.py

"""File utility functions for sentiment analysis."""

import json
from pathlib import Path
from typing import Any

from loguru import logger


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, create if necessary.

    Args:
        path: Directory path to ensure

    Returns:
        Path object for the directory
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(
    data: Any, file_path: Path, indent: int = 2, ensure_ascii: bool = False
) -> None:
    """Save data to JSON file.

    Args:
        data: Data to save
        file_path: Path to save the file
        indent: JSON indentation
        ensure_ascii: Whether to ensure ASCII output

    Raises:
        OSError: If file cannot be written
    """
    try:
        ensure_directory(file_path.parent)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)

        logger.debug(f"Data saved to {file_path}")

    except Exception as e:
        logger.error(f"Failed to save data to {file_path}: {e}")
        raise OSError(f"Cannot save data to {file_path}: {e}")


def load_json(file_path: Path) -> Any:
    """Load data from JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.debug(f"Data loaded from {file_path}")
        return data

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in MB

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        size_bytes = file_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        return round(size_mb, 2)

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to get file size for {file_path}: {e}")
        raise


def list_files_with_extension(directory: Path, extension: str) -> list[Path]:
    """List all files with specific extension in directory.

    Args:
        directory: Directory to search
        extension: File extension to filter (e.g., '.json')

    Returns:
        List of file paths
    """
    if not directory.exists() or not directory.is_dir():
        return []

    files = [f for f in directory.iterdir() if f.is_file() and f.suffix == extension]
    return sorted(files)


def backup_file(file_path: Path, backup_suffix: str = ".backup") -> Path:
    """Create a backup of a file.

    Args:
        file_path: Path to the file to backup
        backup_suffix: Suffix for backup file

    Returns:
        Path to backup file

    Raises:
        FileNotFoundError: If original file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)

    try:
        import shutil

        shutil.copy2(file_path, backup_path)
        logger.info(f"Backup created: {backup_path}")
        return backup_path

    except Exception as e:
        logger.error(f"Failed to create backup of {file_path}: {e}")
        raise
