import random
from time import time

import numpy as np

from analyseGrammar import branching_turtle_to_coords
from computeVoxel import process_network
from libGenerator import setProperties
from utils import bezier_interpolation
from vSystem import F

from .config import VSystemConfig, resolve_config
from .result import GenerationResult, build_runtime_stats


def _apply_seed(seed):
    """
    Seeds Python and NumPy RNGs when a seed is provided.

    Args:
        seed (int | None): Seed value.

    Returns:
        None
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def generate_network(config=None, n_iter=1, d0=25.0, return_intermediates=True):
    """
    Runs the grammar -> coordinates -> voxel pipeline as a library call.

    Args:
        config (dict | VSystemConfig | None): Typed config or property mapping.
        n_iter (int, optional): Grammar recursion depth. Defaults to 1.
        d0 (float, optional): Root vessel diameter. Defaults to 25.0.
        return_intermediates (bool, optional): Keeps coordinates and interpolated
            data in the result. Defaults to True.

    Returns:
        GenerationResult: Structured generation output.
    """
    resolved_config = resolve_config(config)
    seed = resolved_config.reproducibility.seed
    _apply_seed(seed)
    setProperties(resolved_config)

    start_time = time()
    turtle_program = F(n_iter, d0)
    coordinates = list(branching_turtle_to_coords(turtle_program, d0))
    interpolated = bezier_interpolation(coordinates)
    volume = process_network(interpolated, tVol=resolved_config.output.tissue_volume).astype(
        resolved_config.output.image_dtype
    )

    if not return_intermediates:
        coordinates = []
        interpolated = np.empty((4, 0), dtype=float)

    return GenerationResult(
        volume=volume,
        turtle_program=turtle_program,
        coordinates=coordinates,
        interpolated=interpolated,
        config=resolved_config,
        seed=seed,
        runtime_stats=build_runtime_stats(start_time, volume, interpolated),
        metadata={"pipeline": "grammar->coords->voxel", "config_version": 1},
    )


def generate_batch(config=None, seeds=None, n_iter=1, d0=25.0, return_intermediates=False):
    """
    Runs multiple generation calls with explicit seeds.

    Args:
        config (dict | VSystemConfig | None): Base config.
        seeds (list[int] | None): Explicit seeds for each run.
        n_iter (int, optional): Grammar recursion depth. Defaults to 1.
        d0 (float, optional): Root vessel diameter. Defaults to 25.0.
        return_intermediates (bool, optional): Whether to keep coordinate
            intermediates. Defaults to False.

    Returns:
        list[GenerationResult]: Batch outputs in seed order.
    """
    seeds = [] if seeds is None else list(seeds)
    base_config = resolve_config(config)
    results = []
    for seed in seeds:
        batch_config = VSystemConfig(
            generation=base_config.generation,
            collision=base_config.collision,
            tortuosity=base_config.tortuosity,
            output=base_config.output,
            reproducibility=type(base_config.reproducibility)(seed=seed),
        )
        results.append(
            generate_network(
                config=batch_config,
                n_iter=n_iter,
                d0=d0,
                return_intermediates=return_intermediates,
            )
        )
    return results
