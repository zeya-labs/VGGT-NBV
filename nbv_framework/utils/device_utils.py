"""Helpers for normalizing device and dtype specifications."""

from __future__ import annotations

from typing import Dict, Optional

import torch

_DTYPE_ALIASES: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float": torch.float32,
    "fp32": torch.float32,
    "single": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float64": torch.float64,
    "double": torch.float64,
}

_DTYPE_DEFAULT_NAMES: Dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float64: "float64",
}


def resolve_device(requested: Optional[str], local_rank: int) -> torch.device:
    if isinstance(requested, str) and requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def coerce_device(value: Optional[str]) -> torch.device:
    if isinstance(value, str) and value:
        return torch.device(value)
    raise ValueError("Device specification must be provided as a non-empty string.")


def resolve_dtype(value: Union[str, torch.dtype, None]) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str) and value:
        key = value.lower()
        if key.startswith("torch."):
            key = key.split(".", 1)[1]
        dtype = _DTYPE_ALIASES.get(key)
        if dtype is not None:
            return dtype
        raise ValueError(f"Unsupported tensor dtype '{value}'.")
    return torch.float32


def dtype_to_string(dtype: torch.dtype) -> str:
    return _DTYPE_DEFAULT_NAMES.get(dtype, str(dtype).split(".")[-1])


__all__ = [
    "resolve_device",
    "coerce_device",
    "resolve_dtype",
    "dtype_to_string",
]
