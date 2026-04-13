from .config import (
    DEFAULT_COLLISION_RETRY_COUNT,
    DEFAULT_TORTUOSITY_ANGLE_BASE,
    DEFAULT_TORTUOSITY_ANGLE_CAP,
    DEFAULT_TORTUOSITY_ANGLE_SLOPE,
    DEFAULT_TORTUOSITY_ENABLED,
    DEFAULT_TORTUOSITY_MIN_DEPTH,
    DEFAULT_TORTUOSITY_PROB_BASE,
    DEFAULT_TORTUOSITY_PROB_SLOPE,
    DEFAULT_VOXEL_OVERLAP_TOLERANCE,
    CollisionConfig,
    GenerationConfig,
    OutputConfig,
    ReproducibilityConfig,
    TortuosityConfig,
    VSystemConfig,
    config_to_properties,
    resolve_config,
    validate_config,
)
from .result import GenerationResult


def generate_network(*args, **kwargs):
    """
    Lazily imports and runs the pipeline entrypoint.
    """
    from .pipeline import generate_network as _generate_network

    return _generate_network(*args, **kwargs)


def generate_batch(*args, **kwargs):
    """
    Lazily imports and runs the batch pipeline entrypoint.
    """
    from .pipeline import generate_batch as _generate_batch

    return _generate_batch(*args, **kwargs)


__all__ = [
    "CollisionConfig",
    "DEFAULT_COLLISION_RETRY_COUNT",
    "DEFAULT_TORTUOSITY_ANGLE_BASE",
    "DEFAULT_TORTUOSITY_ANGLE_CAP",
    "DEFAULT_TORTUOSITY_ANGLE_SLOPE",
    "DEFAULT_TORTUOSITY_ENABLED",
    "DEFAULT_TORTUOSITY_MIN_DEPTH",
    "DEFAULT_TORTUOSITY_PROB_BASE",
    "DEFAULT_TORTUOSITY_PROB_SLOPE",
    "DEFAULT_VOXEL_OVERLAP_TOLERANCE",
    "GenerationConfig",
    "GenerationResult",
    "OutputConfig",
    "ReproducibilityConfig",
    "TortuosityConfig",
    "VSystemConfig",
    "config_to_properties",
    "generate_batch",
    "generate_network",
    "resolve_config",
    "validate_config",
]
