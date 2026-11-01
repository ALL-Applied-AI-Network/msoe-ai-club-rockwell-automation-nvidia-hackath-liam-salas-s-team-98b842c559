from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class DeviceSelection:
    requested: str
    selected: str
    reason: str


def select_device(requested: str) -> DeviceSelection:
    """Select GPU only after an operator explicitly marks it free."""
    if requested not in {"cpu", "gpu", "auto"}:
        raise ValueError("device must be cpu, gpu, or auto")
    explicitly_free = os.environ.get("FLUIDMIX_GPU_FREE") == "1"
    if requested == "cpu":
        return DeviceSelection(requested, "cpu", "cpu_requested")
    if not explicitly_free:
        return DeviceSelection(requested, "cpu", "gpu_not_explicitly_free")
    try:
        import torch

        available = bool(torch.cuda.is_available())
    except (ImportError, RuntimeError):
        available = False
    if not available:
        return DeviceSelection(requested, "cpu", "gpu_unavailable")
    return DeviceSelection(requested, "cuda", "gpu_explicitly_free")
