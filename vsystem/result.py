from dataclasses import dataclass, field
from time import time
from typing import Any

import numpy as np

from .config import VSystemConfig


@dataclass
class GenerationResult:
    """
    Structured result for one V-System generation run.

    Attributes:
        volume (np.ndarray): Rasterized binary vessel mask.
        turtle_program (str): Generated grammar string.
        coordinates (list[Any]): Turtle coordinates before interpolation.
        interpolated (np.ndarray): Interpolated coordinate array for voxelization.
        config (VSystemConfig): Resolved config used for the run.
        seed (int | None): Applied random seed.
        warnings (list[str]): Non-fatal warnings emitted during the run.
        runtime_stats (dict): Timing and shape metadata for the run.
        metadata (dict): Version or provenance metadata for downstream consumers.
    """

    volume: np.ndarray
    turtle_program: str
    coordinates: list[Any]
    interpolated: np.ndarray
    config: VSystemConfig
    seed: int | None
    warnings: list[str] = field(default_factory=list)
    runtime_stats: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def build_runtime_stats(start_time: float, volume: np.ndarray, interpolated: np.ndarray) -> dict[str, Any]:
    """
    Builds a minimal runtime summary for a generation run.

    Args:
        start_time (float): Start timestamp from `time()`.
        volume (np.ndarray): Output mask.
        interpolated (np.ndarray): Interpolated coordinate array.

    Returns:
        dict[str, Any]: Runtime and size statistics.
    """
    return {
        "elapsed_seconds": time() - start_time,
        "volume_shape": tuple(volume.shape),
        "interpolated_columns": int(interpolated.shape[1]) if interpolated.ndim == 2 else 0,
        "filled_voxels": int(volume.sum()),
    }
