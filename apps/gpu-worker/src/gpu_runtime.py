"""Production GPU admission checks shared by startup and live acceptance.

The worker keeps several models resident at once. Starting model downloads or allocations on
an undersized GPU wastes paid Pod time and can leave a misleading half-ready process, so the
hardware boundary is checked before any heavyweight model is touched.
"""

from dataclasses import dataclass
from typing import Any


MIN_GPU_COMPUTE_MAJOR = 7
MIN_GPU_MEMORY_BYTES = 24_000_000_000


@dataclass(frozen=True)
class GpuRuntimeInfo:
    device: str
    compute_major: int
    compute_minor: int
    total_memory_bytes: int


def validate_production_gpu(torch_module: Any | None = None) -> GpuRuntimeInfo:
    """Return sanitized GPU facts or fail before model loading."""

    if torch_module is None:
        try:
            import torch as torch_module
        except Exception as exc:
            raise RuntimeError(
                "Production GPU preflight failed: PyTorch is unavailable."
            ) from exc

    try:
        if not bool(torch_module.cuda.is_available()):
            raise RuntimeError(
                "Production GPU preflight failed: CUDA is unavailable."
            )

        compute_major, compute_minor = torch_module.cuda.get_device_capability(0)
        compute_major = int(compute_major)
        compute_minor = int(compute_minor)
        if compute_major < MIN_GPU_COMPUTE_MAJOR:
            raise RuntimeError(
                "Production GPU preflight failed: compute capability must be >= 7.0."
            )

        total_memory_bytes = int(
            torch_module.cuda.get_device_properties(0).total_memory
        )
        if total_memory_bytes < MIN_GPU_MEMORY_BYTES:
            raise RuntimeError(
                "Production GPU preflight failed: GPU memory must be at least 24 GB."
            )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "Production GPU preflight failed: GPU capabilities could not be read."
        ) from exc

    return GpuRuntimeInfo(
        device="cuda",
        compute_major=compute_major,
        compute_minor=compute_minor,
        total_memory_bytes=total_memory_bytes,
    )
