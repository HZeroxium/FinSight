# adapters/__init__.py

"""
Model serving adapters package.

This package provides different adapters for model serving:
- SimpleServingAdapter: In-memory serving for development
- TritonServingAdapter: NVIDIA Triton Inference Server integration
- TorchServeAdapter: PyTorch TorchServe integration
"""

from .simple_serving import SimpleServingAdapter
from .triton_serving import TritonServingAdapter
from .torchserve_serving import TorchServeAdapter
from .adapter_factory import ServingAdapterFactory

__all__ = [
    "SimpleServingAdapter",
    "TritonServingAdapter",
    "TorchServeAdapter",
    "ServingAdapterFactory",
]
