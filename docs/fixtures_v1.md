# The vsystem_laplacian_50um_v1 fixtures through `main.py`

One plain `main.py` command per fixture of `presets/fixtures/vsystem_laplacian_50um_v1.json`.
Each uses the fixture's root diameter, seed, persistence, generations, daughter spread and
construction, and accepts the first candidate whose branch and cycle counts fall in the
manifest's ranges and whose references pass their checks. The other settings are the export
mode's defaults, which are the fixtures' common settings (50 µm voxels, `--d-min 50`,
`--epsilon 6 6`, `--randmarg 0.15 0.15`, `--subdivisions 4`, the walk with collision avoidance
at a 100 µm margin, no anomalies, buried stubs pruned).

These make **new networks of the same kind**: `main.py` draws its parameters before growing,
so a tree does not repeat the saved one even at the same seed (the explicit straight shapes
F01 and F02 do). To rebuild the published fixtures exactly, with their candidate selection and
full QA, run `python fixtures.py --out DIR` or `python main.py --fixture F06 --out DIR`.

```bash
cd V-System

# F01_straight_oblique_tube
python main.py --count 1 --seed 16001 --d0 200 0 --construction straight_tube --out out/F01_straight_oblique_tube

# F02_straight_y
python main.py --count 1 --seed 16002 --d0 250 0 --construction straight_y --out out/F02_straight_y

# F03_tortuous_single_vessel
python main.py --count 1 --seed 16003 --d0 200 0 --construction walk_program --persistence 12 --require-branches 1 1 --require-cycles 0 0 --candidates 32 --out out/F03_tortuous_single_vessel

# F04_tortuous_y
python main.py --count 1 --seed 16004 --d0 250 0 --construction walk_program --persistence 12 --daughter-diameter 215 --require-branches 3 3 --require-cycles 0 0 --candidates 32 --out out/F04_tortuous_y

# F05_small_asymmetric_tree
python main.py --count 1 --seed 16005 --d0 300 0 --iterations 3 3 --persistence 12 --sigma 3 --require-branches 5 11 --require-cycles 0 0 --candidates 32 --out out/F05_small_asymmetric_tree

# F06_moderate_tree
python main.py --count 1 --seed 16006 --d0 350 0 --iterations 5 5 --persistence 12 --require-branches 15 39 --require-cycles 0 0 --candidates 32 --out out/F06_moderate_tree

# F07_moderate_tree_stronger_curvature
python main.py --count 1 --seed 16006 --d0 350 0 --iterations 5 5 --persistence 6 --require-branches 15 39 --require-cycles 0 0 --candidates 32 --out out/F07_moderate_tree_stronger_curvature

# F08_thin_short_daughters
python main.py --count 1 --seed 16008 --d0 500 0 --iterations 4 4 --persistence 8 --thin-daughters 5 --require-branches 15 59 --require-cycles 0 0 --candidates 32 --out out/F08_thin_short_daughters

# F09_dense_tortuous_tree
python main.py --count 1 --seed 16009 --d0 450 0 --iterations 7 7 --persistence 8 --sigma 3 --require-branches 61 149 --require-cycles 0 0 --candidates 32 --out out/F09_dense_tortuous_tree

# F10_close_non_touching_branches
python main.py --count 1 --seed 16010 --d0 400 0 --iterations 6 6 --persistence 8 --sigma 3 --companion --require-branches 31 99 --require-cycles 0 0 --candidates 32 --out out/F10_close_non_touching_branches

# F11_tortuous_single_cycle
python main.py --count 1 --seed 16011 --d0 350 0 --iterations 5 5 --persistence 8 --sigma 3 --anastomosis-fraction 0.5 --keep-bridges 1 1 --write-unbridged --require-branches 15 59 --require-cycles 1 1 --candidates 32 --out out/F11_tortuous_single_cycle

# F12_dense_sparse_cycles
python main.py --count 1 --seed 16012 --d0 500 0 --iterations 7 7 --persistence 6 --sigma 3 --anastomosis-fraction 0.5 --keep-bridges 2 3 --write-unbridged --require-branches 61 149 --require-cycles 2 3 --candidates 32 --out out/F12_dense_sparse_cycles
```

Every command above was run against this version: all twelve were accepted, with branch and
cycle counts inside the manifest ranges (F01–F04 on the same grids as the published fixtures).
F11 and F12 also write the tree before its bridges, on the same grid, as `<stem>_unbridged_*`.
