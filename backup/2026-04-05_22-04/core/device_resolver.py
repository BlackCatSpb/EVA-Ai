from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal, Optional

import torch

Precision = Literal["fp32", "fp16", "bf16"]


@dataclass(frozen=True)
class DeviceConfig:
    use_gpu_if_available: bool = True
    prefer_precision: Precision = "fp16"  # used on CUDA; CPU falls back to fp32
    allow_bfloat16: bool = False
    max_vram_gb: Optional[float] = None  # hint for upper-bound decisions


def resolve_device(cfg: DeviceConfig | None = None) -> torch.device:
    cfg = cfg or DeviceConfig()
    if cfg.use_gpu_if_available and torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


def select_precision(device: torch.device, cfg: DeviceConfig | None = None) -> Precision:
    cfg = cfg or DeviceConfig()
    if device.type == "cuda":
        if cfg.allow_bfloat16 and torch.cuda.is_bf16_supported():
            return "bf16"
        # MX550-class GPUs: prefer fp16
        if cfg.prefer_precision in ("fp16", "bf16"):
            return "fp16"
    return "fp32"


@contextmanager
def autocast_context(device: torch.device, precision: Precision) -> Iterator[None]:
    if device.type == "cuda" and precision in ("fp16", "bf16"):
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        with torch.cuda.amp.autocast(dtype=dtype):
            with torch.inference_mode():
                yield
    else:
        with torch.inference_mode():
            yield


def memory_info() -> dict:
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return {
            "device": torch.cuda.get_device_name(0),
            "total_gb": round(total / 1e9, 2),
            "free_gb": round(free / 1e9, 2),
        }
    return {"device": "cpu", "total_gb": None, "free_gb": None}


def should_pin_memory(device: torch.device) -> bool:
    return device.type == "cuda"
