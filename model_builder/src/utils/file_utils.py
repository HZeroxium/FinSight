# utils/file_utils.py

import json
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import yaml

import torch
from pydantic import BaseModel

from common.logger.logger_factory import LoggerFactory


class FileUtils:
    """Utility class for file operations with enhanced safety and functionality"""

    _logger = LoggerFactory.get_logger(__name__)

    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, create if not

        Args:
            path: Directory path

        Returns:
            Path: Path object
        """
        path = Path(path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            FileUtils._logger.debug(f"Directory ensured: {path}")
            return path
        except Exception as e:
            FileUtils._logger.error(f"Failed to create directory {path}: {str(e)}")
            raise

    @staticmethod
    def remove_dir(path: Union[str, Path], force: bool = False) -> None:
        """
        Remove directory and all its contents

        Args:
            path: Directory path to remove
            force: If True, ignore errors
        """
        path = Path(path)
        try:
            if path.exists():
                shutil.rmtree(path, ignore_errors=force)
                FileUtils._logger.info(f"Directory removed: {path}")
        except Exception as e:
            if not force:
                FileUtils._logger.error(f"Failed to remove directory {path}: {str(e)}")
                raise

    @staticmethod
    def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
        """
        Copy file from source to destination

        Args:
            src: Source file path
            dst: Destination file path
        """
        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_path}")

        FileUtils.ensure_dir(dst_path.parent)
        shutil.copy2(src_path, dst_path)
        FileUtils._logger.info(f"File copied from {src_path} to {dst_path}")

    @staticmethod
    def get_file_size(path: Union[str, Path]) -> int:
        """
        Get file size in bytes

        Args:
            path: File path

        Returns:
            int: File size in bytes
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path.stat().st_size

    @staticmethod
    def list_files(
        directory: Union[str, Path], pattern: str = "*", recursive: bool = False
    ) -> List[Path]:
        """
        List files in directory matching pattern

        Args:
            directory: Directory to search
            pattern: File pattern (glob style)
            recursive: Whether to search recursively

        Returns:
            List[Path]: List of matching file paths
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if recursive:
            return list(directory.rglob(pattern))
        else:
            return list(directory.glob(pattern))

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        atomic: bool = True,
    ) -> None:
        """
        Save PyTorch model with metadata using atomic operations

        Args:
            model: PyTorch model to save
            path: Save path
            metadata: Optional metadata to save alongside model
            atomic: Whether to use atomic save (write to temp file first)
        """
        path = Path(path)
        FileUtils.ensure_dir(path.parent)

        def _save_model():
            torch.save(model.state_dict(), path)
            FileUtils._logger.info(f"Model saved to {path}")

        if atomic:
            # Use temporary file for atomic save
            with tempfile.NamedTemporaryFile(
                dir=path.parent, suffix=path.suffix, delete=False
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)

            try:
                torch.save(model.state_dict(), tmp_path)
                tmp_path.replace(path)  # Atomic move
                FileUtils._logger.info(f"Model saved atomically to {path}")
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise
        else:
            _save_model()

        # Save metadata if provided
        if metadata:
            metadata_path = path.with_suffix(".json")
            FileUtils.save_json(metadata, metadata_path, atomic=atomic)

    @staticmethod
    def load_model(
        model: torch.nn.Module,
        path: Union[str, Path],
        device: torch.device,
        strict: bool = True,
    ) -> torch.nn.Module:
        """
        Load PyTorch model state dict

        Args:
            model: PyTorch model instance
            path: Model file path
            device: Device to load model on
            strict: Whether to strictly enforce state dict keys match

        Returns:
            torch.nn.Module: Loaded model
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict, strict=strict)
            model.to(device)
            FileUtils._logger.info(f"Model loaded from {path}")
            return model
        except Exception as e:
            FileUtils._logger.error(f"Failed to load model from {path}: {str(e)}")
            raise

    @staticmethod
    def load_metadata(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Load model metadata

        Args:
            path: Model file path (will look for corresponding .json file)

        Returns:
            dict: Metadata or None if not found
        """
        metadata_path = Path(path).with_suffix(".json")
        if metadata_path.exists():
            return FileUtils.load_json(metadata_path)
        return None

    @staticmethod
    def save_json(
        data: Union[Dict[str, Any], BaseModel],
        path: Union[str, Path],
        indent: int = 2,
        atomic: bool = True,
    ) -> None:
        """
        Save data as JSON file

        Args:
            data: Data to save (dict or Pydantic model)
            path: Save path
            indent: JSON indentation
            atomic: Whether to use atomic save
        """
        path = Path(path)
        FileUtils.ensure_dir(path.parent)

        # Convert Pydantic model to dict if needed
        if isinstance(data, BaseModel):
            json_data = data.model_dump()
        else:
            json_data = data

        def _save_json():
            with open(path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=indent, ensure_ascii=False)

        if atomic:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=path.parent,
                suffix=".json",
                delete=False,
                encoding="utf-8",
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                json.dump(json_data, tmp_file, indent=indent, ensure_ascii=False)

            try:
                tmp_path.replace(path)
                FileUtils._logger.info(f"JSON saved atomically to {path}")
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise
        else:
            _save_json()
            FileUtils._logger.info(f"JSON saved to {path}")

    @staticmethod
    def load_json(path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load JSON file

        Args:
            path: JSON file path

        Returns:
            dict: Loaded data
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            FileUtils._logger.debug(f"JSON loaded from {path}")
            return data
        except json.JSONDecodeError as e:
            FileUtils._logger.error(f"Invalid JSON in {path}: {str(e)}")
            raise
        except Exception as e:
            FileUtils._logger.error(f"Failed to load JSON from {path}: {str(e)}")
            raise

    @staticmethod
    def save_yaml(
        data: Union[Dict[str, Any], BaseModel], path: Union[str, Path]
    ) -> None:
        """
        Save data as YAML file

        Args:
            data: Data to save
            path: Save path
        """
        path = Path(path)
        FileUtils.ensure_dir(path.parent)

        if isinstance(data, BaseModel):
            yaml_data = data.model_dump()
        else:
            yaml_data = data

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
        FileUtils._logger.info(f"YAML saved to {path}")

    @staticmethod
    def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load YAML file

        Args:
            path: YAML file path

        Returns:
            dict: Loaded data
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"YAML file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        FileUtils._logger.info(f"YAML loaded from {path}")
        return data

    @staticmethod
    def save_object(
        obj: Any, path: Union[str, Path], protocol: int = pickle.HIGHEST_PROTOCOL
    ) -> None:
        """
        Save Python object using pickle

        Args:
            obj: Object to save
            path: Save path
            protocol: Pickle protocol version
        """
        path = Path(path)
        FileUtils.ensure_dir(path.parent)

        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=protocol)
        FileUtils._logger.info(f"Object saved to {path}")

    @staticmethod
    def load_object(path: Union[str, Path]) -> Any:
        """
        Load Python object using pickle

        Args:
            path: Object file path

        Returns:
            Any: Loaded object
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Object file not found: {path}")

        with open(path, "rb") as f:
            obj = pickle.load(f)
        FileUtils._logger.info(f"Object loaded from {path}")
        return obj
