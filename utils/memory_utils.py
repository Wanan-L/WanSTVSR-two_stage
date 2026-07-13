import gc
from typing import Any, Dict, Union

import torch
from accelerate.logging import get_logger


def get_memory_statistics(
    device: torch.device | int | None = None,
    precision: int = 3,
) -> Dict[str, Any]:

    if not torch.cuda.is_available():
        return {}

    if device is None:
        device = torch.cuda.current_device()

    memory_allocated = torch.cuda.memory_allocated(device)  # 当前已分配显存
    memory_reserved = torch.cuda.memory_reserved(device)    # 当前保留显存 (含未使用的显存)
    max_memory_allocated = torch.cuda.max_memory_allocated(device) # 最大已分配显存
    max_memory_reserved = torch.cuda.max_memory_reserved(device) # 最大保留显存

    return {
        "memory_allocated": round(bytes_to_gigabytes(memory_allocated), precision),
        "memory_reserved": round(bytes_to_gigabytes(memory_reserved), precision),
        "max_memory_allocated": round(bytes_to_gigabytes(max_memory_allocated), precision),
        "max_memory_reserved": round(bytes_to_gigabytes(max_memory_reserved), precision),
    }


def bytes_to_gigabytes(x: int) -> float:
    if x is not None:
        return x / 1024**3


def reset_peak_memory_stats(device: torch.device | int | None = None) -> None:
    if not torch.cuda.is_available():
        return

    if device is None:
        device = torch.cuda.current_device()

    torch.cuda.reset_peak_memory_stats(device)


def free_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    # TODO(aryan): handle non-cuda devices


def unload_model(model):
    model.to("cpu")


def make_contiguous(
    x: Union[torch.Tensor, Dict[str, torch.Tensor]],
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(x, torch.Tensor):
        return x.contiguous()
    elif isinstance(x, dict):
        return {k: make_contiguous(v) for k, v in x.items()}
    else:
        return x
