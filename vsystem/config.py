from dataclasses import asdict, dataclass

DEFAULT_COLLISION_RETRY_COUNT = -1
DEFAULT_VOXEL_OVERLAP_TOLERANCE = -1
DEFAULT_TORTUOSITY_ENABLED = False
DEFAULT_TORTUOSITY_MIN_DEPTH = 2
DEFAULT_TORTUOSITY_PROB_BASE = 0.0
DEFAULT_TORTUOSITY_PROB_SLOPE = 0.0
DEFAULT_TORTUOSITY_ANGLE_BASE = 0.0
DEFAULT_TORTUOSITY_ANGLE_SLOPE = 0.0
DEFAULT_TORTUOSITY_ANGLE_CAP = 30.0


@dataclass(frozen=True)
class GenerationConfig:
    k: int = 3
    epsilon: float = 10.0
    randmarg: float = 3.0
    sigma: float = 5.0
    stochparams: bool = True


@dataclass(frozen=True)
class CollisionConfig:
    clearance: float = 0.0
    retry_count: int = DEFAULT_COLLISION_RETRY_COUNT
    voxel_overlap_tolerance: int = DEFAULT_VOXEL_OVERLAP_TOLERANCE


@dataclass(frozen=True)
class TortuosityConfig:
    enabled: bool = DEFAULT_TORTUOSITY_ENABLED
    min_depth: int = DEFAULT_TORTUOSITY_MIN_DEPTH
    prob_base: float = DEFAULT_TORTUOSITY_PROB_BASE
    prob_slope: float = DEFAULT_TORTUOSITY_PROB_SLOPE
    angle_base: float = DEFAULT_TORTUOSITY_ANGLE_BASE
    angle_slope: float = DEFAULT_TORTUOSITY_ANGLE_SLOPE
    angle_cap: float = DEFAULT_TORTUOSITY_ANGLE_CAP


@dataclass(frozen=True)
class OutputConfig:
    tissue_volume: tuple[int, int, int] = (500, 500, 150)
    outpath: str = "./output_syn/"
    save_nifti: bool = False
    image_dtype: str = "int8"


@dataclass(frozen=True)
class ReproducibilityConfig:
    seed: int | None = None


@dataclass(frozen=True)
class VSystemConfig:
    generation: GenerationConfig = GenerationConfig()
    collision: CollisionConfig = CollisionConfig()
    tortuosity: TortuosityConfig = TortuosityConfig()
    output: OutputConfig = OutputConfig()
    reproducibility: ReproducibilityConfig = ReproducibilityConfig()


def validate_config(config: VSystemConfig) -> VSystemConfig:
    """
    Validates a typed config and returns it unchanged on success.

    Args:
        config (VSystemConfig): Candidate config.

    Returns:
        VSystemConfig: The same config when validation succeeds.

    Raises:
        ValueError: If any field falls outside supported ranges.
        TypeError: If the config object type is unsupported.
    """
    if not isinstance(config, VSystemConfig):
        raise TypeError("config must be a VSystemConfig instance")

    generation = config.generation
    collision = config.collision
    tortuosity = config.tortuosity
    output = config.output

    if generation.k <= 0:
        raise ValueError("generation.k must be > 0")
    if generation.epsilon <= 0:
        raise ValueError("generation.epsilon must be > 0")
    if generation.randmarg < 0:
        raise ValueError("generation.randmarg must be >= 0")
    if generation.sigma <= 0:
        raise ValueError("generation.sigma must be > 0")

    if collision.clearance < 0:
        raise ValueError("collision.clearance must be >= 0")
    if collision.retry_count < -1:
        raise ValueError("collision.retry_count must be >= -1")
    if collision.voxel_overlap_tolerance < -1:
        raise ValueError("collision.voxel_overlap_tolerance must be >= -1")

    if tortuosity.min_depth < 0:
        raise ValueError("tortuosity.min_depth must be >= 0")
    if not 0.0 <= tortuosity.prob_base <= 1.0:
        raise ValueError("tortuosity.prob_base must be in [0, 1]")
    if tortuosity.prob_slope < 0:
        raise ValueError("tortuosity.prob_slope must be >= 0")
    if tortuosity.angle_base < 0:
        raise ValueError("tortuosity.angle_base must be >= 0")
    if tortuosity.angle_slope < 0:
        raise ValueError("tortuosity.angle_slope must be >= 0")
    if tortuosity.angle_cap < 0:
        raise ValueError("tortuosity.angle_cap must be >= 0")
    if tortuosity.angle_base > tortuosity.angle_cap:
        raise ValueError("tortuosity.angle_base must be <= tortuosity.angle_cap")

    if len(output.tissue_volume) != 3 or any(axis <= 0 for axis in output.tissue_volume):
        raise ValueError("output.tissue_volume must contain three positive integers")

    return config


def config_to_properties(config: VSystemConfig) -> dict:
    """
    Flattens a typed config into the property dictionary used by the core modules.

    Args:
        config (VSystemConfig): Typed config object.

    Returns:
        dict: Flat property mapping.
    """
    config = validate_config(config)
    return {
        "k": config.generation.k,
        "epsilon": config.generation.epsilon,
        "randmarg": config.generation.randmarg,
        "sigma": config.generation.sigma,
        "stochparams": config.generation.stochparams,
        "collision_clearance": config.collision.clearance,
        "collision_retry_count": config.collision.retry_count,
        "voxel_overlap_tolerance": config.collision.voxel_overlap_tolerance,
        "tortuosity_enabled": config.tortuosity.enabled,
        "tortuosity_min_depth": config.tortuosity.min_depth,
        "tortuosity_prob_base": config.tortuosity.prob_base,
        "tortuosity_prob_slope": config.tortuosity.prob_slope,
        "tortuosity_angle_base": config.tortuosity.angle_base,
        "tortuosity_angle_slope": config.tortuosity.angle_slope,
        "tortuosity_angle_cap": config.tortuosity.angle_cap,
    }


def resolve_config(properties=None) -> VSystemConfig:
    """
    Builds a typed config from a property dictionary or config object.

    Args:
        properties (dict | VSystemConfig | None): Flat mapping or config object.

    Returns:
        VSystemConfig: Validated typed config.
    """
    if properties is None:
        return validate_config(VSystemConfig())
    if isinstance(properties, VSystemConfig):
        return validate_config(properties)
    if not isinstance(properties, dict):
        raise TypeError("properties must be a dict, VSystemConfig, or None")

    base = asdict(VSystemConfig())
    if "generation" in properties or "collision" in properties or "tortuosity" in properties:
        for namespace in ("generation", "collision", "tortuosity", "output", "reproducibility"):
            if namespace in properties:
                base[namespace].update(properties[namespace])
        return validate_config(
            VSystemConfig(
                generation=GenerationConfig(**base["generation"]),
                collision=CollisionConfig(**base["collision"]),
                tortuosity=TortuosityConfig(**base["tortuosity"]),
                output=OutputConfig(**base["output"]),
                reproducibility=ReproducibilityConfig(**base["reproducibility"]),
            )
        )

    base["generation"].update({
        "k": properties.get("k", base["generation"]["k"]),
        "epsilon": properties.get("epsilon", base["generation"]["epsilon"]),
        "randmarg": properties.get("randmarg", base["generation"]["randmarg"]),
        "sigma": properties.get("sigma", base["generation"]["sigma"]),
        "stochparams": properties.get("stochparams", base["generation"]["stochparams"]),
    })
    base["collision"].update({
        "clearance": properties.get("collision_clearance", base["collision"]["clearance"]),
        "retry_count": properties.get("collision_retry_count", base["collision"]["retry_count"]),
        "voxel_overlap_tolerance": properties.get(
            "voxel_overlap_tolerance",
            base["collision"]["voxel_overlap_tolerance"],
        ),
    })
    base["tortuosity"].update({
        "enabled": properties.get("tortuosity_enabled", base["tortuosity"]["enabled"]),
        "min_depth": properties.get("tortuosity_min_depth", base["tortuosity"]["min_depth"]),
        "prob_base": properties.get("tortuosity_prob_base", base["tortuosity"]["prob_base"]),
        "prob_slope": properties.get("tortuosity_prob_slope", base["tortuosity"]["prob_slope"]),
        "angle_base": properties.get("tortuosity_angle_base", base["tortuosity"]["angle_base"]),
        "angle_slope": properties.get("tortuosity_angle_slope", base["tortuosity"]["angle_slope"]),
        "angle_cap": properties.get("tortuosity_angle_cap", base["tortuosity"]["angle_cap"]),
    })
    if "tissue_volume" in properties:
        base["output"]["tissue_volume"] = tuple(properties["tissue_volume"])
    if "outpath" in properties:
        base["output"]["outpath"] = properties["outpath"]
    if "seed" in properties:
        base["reproducibility"]["seed"] = properties["seed"]

    return validate_config(
        VSystemConfig(
            generation=GenerationConfig(**base["generation"]),
            collision=CollisionConfig(**base["collision"]),
            tortuosity=TortuosityConfig(**base["tortuosity"]),
            output=OutputConfig(**base["output"]),
            reproducibility=ReproducibilityConfig(**base["reproducibility"]),
        )
    )
