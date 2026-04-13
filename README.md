# V-System

V-System is a Python toolkit for generating synthetic 3D vascular trees with a stochastic Lindenmayer-system pipeline. It produces branching vessel grammars, converts them into 3D centerline coordinates, and voxelizes the result into binary mask volumes that can be used for simulation, benchmarking, and synthetic data generation.

This repository refactored [original V-System work](https://github.com/psweens/V-System)
![Example generation](./Lnet_Generations.jpg)

## Installation

Install dependencies:

```bash
pip install -r REQUIREMENTS.txt
```

If you only want the core runtime used by the generation pipeline, the main Python dependencies are `numpy`, `nibabel`, and `scikit-image`. Some optional utilities in the repository may require additional packages listed in `REQUIREMENTS.txt`.

## CLI Usage

The command line entrypoint is [`main.py`](./main.py). By default it generates a batch of NIfTI vessel masks in `./output_syn/`.

Basic usage:

```bash
python main.py
```

Generate a smaller deterministic batch:

```bash
python main.py --count 2 --min-iter 4 --max-iter 5 --seed 7 --outpath ./output_syn/
```

Set a custom output volume:

```bash
python main.py --tissue-volume 256 256 128
```

Available options:

- `--count`: number of networks to generate
- `--min-iter`: minimum grammar recursion depth
- `--max-iter`: maximum grammar recursion depth
- `--d0-mean`: mean root diameter
- `--d0-std`: root diameter standard deviation
- `--tissue-volume X Y Z`: output volume size
- `--outpath`: output directory
- `--seed`: optional deterministic base seed
- `--save-intermediates`: keep intermediates in memory during CLI generation

## Python API

The cleaned public API lives under [`vsystem/`](./vsystem).

Basic example:

```python
from vsystem import VSystemConfig, generate_network

config = VSystemConfig()
result = generate_network(config=config, n_iter=4, d0=18.0)

print(result.volume.shape)
print(result.turtle_program[:80])
```

You can also override selected namespaces with a plain mapping:

```python
from vsystem import generate_network

result = generate_network(
    config={
        "output": {"tissue_volume": (128, 128, 96)},
        "reproducibility": {"seed": 42},
    },
    n_iter=3,
    d0=14.0,
)
```

## Output Structure

`generate_network(...)` returns a [`GenerationResult`](./vsystem/result.py) object with:

- `volume`: voxelized binary mask as a NumPy array
- `turtle_program`: generated grammar string
- `coordinates`: turtle states before interpolation
- `interpolated`: interpolated centerline array used for voxelization
- `config`: resolved `VSystemConfig`
- `seed`: applied seed, if any
- `warnings`: non-fatal warnings container
- `runtime_stats`: simple runtime and size statistics
- `metadata`: pipeline metadata

CLI runs save output volumes as:

```text
output_syn/
  Lnet_i<iterations>_<index>.nii.gz
```

If tortuosity is enabled in the active config, the filename suffix becomes `_tort.nii.gz`.

## Repository Structure

```text
V-System/
  main.py                  CLI entrypoint
  vSystem.py               grammar generation rules
  analyseGrammar.py        turtle interpretation and geometry checks
  computeVoxel.py          voxel traversal and mask creation
  libGenerator.py          shared parameter state for core modules
  utils.py                 geometry and interpolation helpers
  visuals.py               optional plotting utilities
  vsystem/
    __init__.py            public API exports
    config.py              typed config models and validation
    pipeline.py            library orchestration entrypoints
    result.py              structured result model
  tests/                   regression and feature tests
  README.md
  REQUIREMENTS.txt
```

## Notes

- The runtime pipeline order is unchanged: `vSystem.py -> analyseGrammar.py -> computeVoxel.py`.
- The refactor makes the project import-safe and library-first without rewriting the underlying generation algorithm.
- `libGenerator.py` remains because the grammar and turtle modules still share active generation parameters through it.

## Citation

**Version 1.0**  
> [Quantification of vascular networks in photoacoustic mesoscopy](https://www.sciencedirect.com/science/article/pii/S221359792200026X)  
> Emma L. Brown, Thierry L. Lefebvre, Paul W. Sweeney et al.

**Version 2.0**  
> [Unsupervised Segmentation of 3D Microvascular Photoacoustic Images Using Deep Generative Learning](https://doi.org/10.1002/advs.202402195)  
> Paul W. Sweeney et al., *Advanced Science*, 2024
