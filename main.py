import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from vsystem import VSystemConfig, generate_network, resolve_config


def build_parser():
    """
    Builds the command line interface for batch generation.

    Returns:
        argparse.ArgumentParser: Configured CLI parser.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic vascular volumes with V-System.")
    parser.add_argument("--count", type=int, default=6, help="Number of networks to generate.")
    parser.add_argument("--min-iter", type=int, default=2, help="Minimum grammar iterations.")
    parser.add_argument("--max-iter", type=int, default=12, help="Maximum grammar iterations.")
    parser.add_argument("--d0-mean", type=float, default=25.0, help="Mean root vessel diameter.")
    parser.add_argument("--d0-std", type=float, default=10.0, help="Root vessel diameter standard deviation.")
    parser.add_argument(
        "--tissue-volume",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(500, 500, 150),
        help="Output volume dimensions.",
    )
    parser.add_argument("--outpath", default="./output_syn/", help="Directory for generated NIfTI volumes.")
    parser.add_argument("--seed", type=int, default=None, help="Optional base random seed.")
    parser.add_argument("--save-intermediates", action="store_true", help="Keep intermediates in the returned result.")
    return parser


def build_default_config():
    """
    Builds the canonical default config for CLI execution.

    Returns:
        VSystemConfig: Default generation config.
    """
    return VSystemConfig()


def save_volume(result, output_path):
    """
    Saves a generated mask as a NIfTI file.

    Args:
        result: Generation result object.
        output_path (Path): Destination file path.

    Returns:
        None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(result.volume, affine=np.eye(4)), str(output_path))


def run_cli(args=None):
    """
    Runs the command line batch generator.

    Args:
        args (list[str] | None): Optional argument vector.

    Returns:
        None
    """
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    iterations = np.linspace(parsed.min_iter, parsed.max_iter, parsed.count).astype(int)
    base_seed = parsed.seed
    config = build_default_config()

    for index, n_iter in enumerate(iterations):
        run_seed = None if base_seed is None else base_seed + index
        if run_seed is not None:
            np.random.seed(run_seed)
        d0 = np.random.normal(parsed.d0_mean, parsed.d0_std)

        result = generate_network(
            config=resolve_config({
                "generation": config.generation.__dict__,
                "collision": config.collision.__dict__,
                "tortuosity": config.tortuosity.__dict__,
                "output": {
                    "tissue_volume": tuple(parsed.tissue_volume),
                    "outpath": parsed.outpath,
                    "save_nifti": True,
                    "image_dtype": "int8",
                },
                "reproducibility": {"seed": run_seed},
            }),
            n_iter=int(n_iter),
            d0=float(d0),
            return_intermediates=parsed.save_intermediates,
        )

        suffix = "_tort" if result.config.tortuosity.enabled else ""
        output_path = Path(parsed.outpath) / f"Lnet_i{int(n_iter)}_{index}{suffix}.nii.gz"
        print(f"Creating image ... {index} with {int(n_iter)} iterations")
        save_volume(result, output_path)


if __name__ == "__main__":
    run_cli()
