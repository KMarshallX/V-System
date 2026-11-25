#!/usr/bin/env python3
"""
fixtures.py -- build a fixture set of synthetic vessel networks with known references.

Runs every fixture of a manifest (by default the vsystem_laplacian_50um_v1 set in
presets/fixtures/) through recipes.py: candidate selection, geometry and digital-topology
QA, rendering at 50 um isotropic voxels with V-System's own rasteriser (connect=True).
Each accepted fixture gets a NIfTI vessel mask, the 26-connected and foreground-centred
26-connected references (reference.py), a SkelHub-style GraphML reference graph, the
V-System centreline archive, a graph JSON, metadata and a PNG preview.

    python fixtures.py --manifest presets/fixtures/vsystem_laplacian_50um_v1.json --out DIR

Nothing here judges a fixture by how a skeletonization algorithm performs on it.
"""
import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np

from recipes import (ADJUSTMENTS, BORDER_VOX, BRANCH_CONVENTION, COMMON, D_FLOOR_UM, D_ROOT_MAX_UM,
                     DENSE_STEP_UM, OPENING_MIN_UM, ORIGINS, PLANS, ROUNDOFF_UM, SPACING_MM,
                     TOPOLOGY_CONVENTION, TORTUOSITY_TARGETS, V, VOXEL_UM, adapter_self_tests, border_report,
                     check, fixture_specs, graph_json, read_nifti_check, run_fixture, run_pair, save_archive,
                     sha256_file, verify_archive_roundtrip, write_json, write_nifti, write_preview)
from reference import FIXTURE_NAMES, export_references


def git(path, *args):
    try:
        return subprocess.run(["git", "-C", path, *args], capture_output=True, text=True, check=True).stdout
    except Exception as exc:                # noqa: BLE001 - recorded, not fatal
        return f"unavailable: {exc}"


def provenance(vsystem_path, argv):
    import matplotlib
    import nibabel
    import scipy
    import skimage
    import tifffile
    status = git(vsystem_path, "status", "--porcelain")
    tracked_diff = git(vsystem_path, "diff", "HEAD")
    return {
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": [sys.executable, *argv],
        "python": sys.version, "platform": platform.platform(),
        "dependencies": {"numpy": np.__version__, "scipy": scipy.__version__, "nibabel": nibabel.__version__,
                         "matplotlib": matplotlib.__version__, "scikit-image": skimage.__version__,
                         "tifffile": tifffile.__version__},
        "vsystem": {"path": os.path.abspath(vsystem_path), "commit": git(vsystem_path, "rev-parse", "HEAD").strip(),
                    "describe": git(vsystem_path, "log", "-1", "--format=%h %s").strip(),
                    "tracked_files_modified": bool(tracked_diff.strip()),
                    "tracked_diff_sha256": hashlib.sha256(tracked_diff.encode()).hexdigest(),
                    "status_porcelain": status.splitlines(),
                    "dirty_note": "untracked files only" if status.strip() and not tracked_diff.strip() else
                                  ("clean" if not status.strip() else "tracked files modified")},
    }




def source_points(nodes, grid):
    """Archive micrometres -> voxel xyz by q = T @ [p, 1] (NaN separators stay NaN)."""
    T = grid["T"]
    return (nodes[:3].T @ T[:3, :3].T + T[:3, 3]).T


def centred_volume(report, shape):
    volume = np.zeros(tuple(int(v) for v in shape), dtype=np.uint8)
    for branch in report["branches"]:
        volume[tuple(np.asarray(branch["voxel_path_xyz"], dtype=int).T)] = 1
    return volume


def export_fixture_references(directory, fixture_id, nodes, mask, grid, reference6=None):
    """
    Writes vessel_mask.nii.gz, reference_skeleton_26_con.nii.gz,
    reference_skeleton_26_con_centered.nii.gz and reference_graph.graphml next to the
    archive and graph JSON (already written), plus reference_skeleton_6_con.nii.gz when
    `reference6` is given. Returns the reference export report.
    """
    edges = V.graph.build(nodes)["edges"]
    attributes = {"fixture_id": fixture_id, "dataset_id": DATASET_ID,
                  "source_reference_graph_sha256": sha256_file(os.path.join(directory, "reference_graph.json")),
                  "source_reference_network_sha256": sha256_file(os.path.join(directory, "reference_network.npz"))}
    report = export_references(directory, FIXTURE_NAMES, nodes, edges, mask, source_points(nodes, grid),
                               grid["affine"], grid["T"], attributes)
    if reference6 is not None:
        write_nifti(os.path.join(directory, "reference_skeleton_6_con.nii.gz"), reference6, grid["affine"])
    return report


def reference_summary(report):
    return {"method": report["method"], "topology_components_tunnels_cavities":
            report["topology_components_tunnels_cavities"], "voxels_26_con": report["old_voxels"],
            "voxels_26_con_centered": report["new_voxels"], "changed_voxels": report["changed_voxels"],
            "branches_changed": sum(1 for b in report["branches"] if b["changed"]),
            "proposals_rejected": sum(1 for b in report["branches"] if not b["proposal_accepted"]),
            "graphml": report["graphml"], "outputs": report["outputs"], "limitations": report["limitations"]}


def write_fixture(out_dir, spec, accepted, prov, log, write_6con=False):
    """Writes every artifact of one accepted fixture and returns its summary."""
    candidate, m, rendered, records = accepted["candidate"], accepted["m"], accepted["rendered"], accepted["records"]
    fid = spec["id"]
    fixture_dir = os.path.join(out_dir, f"{fid}_{spec['name']}")
    os.makedirs(fixture_dir, exist_ok=False)
    graph_info = m["graph"]
    grid = rendered["grid"]
    nodes = candidate["nodes"]
    shape = grid["shape"]
    archive_record = {
        "fixture_id": fid, "dataset_id": DATASET_ID, "units": "um", "axis_order": "xyz",
        "fit": "voxel_size", "voxel_size": VOXEL_UM, "connect": True, "clip_axes": [], "volume": shape.tolist(),
        "grammar_um_to_voxel_xyz": grid["T"].tolist(), "seed": accepted["seed"],
        "construction": candidate["construction"], "events": candidate["events"],
        "origin_codes": {str(i): n for i, n in enumerate(ORIGINS)},
        "polyline_origins": [ORIGINS[int(candidate["origin"][c[0]])] for c in V.graph.polylines(nodes)],
        "note": "re-render with computeVoxel.process_network(nodes, volume, fit='voxel_size', voxel_size=50)",
    }
    save_archive(os.path.join(fixture_dir, "reference_network.npz"), nodes, candidate["program"], archive_record,
                 graph_info, candidate["tree"])
    write_json(os.path.join(fixture_dir, "reference_graph.json"), graph_json(spec, graph_info))
    references = export_fixture_references(fixture_dir, fid, nodes, rendered["mask"], grid,
                                           rendered["reference"] if write_6con else None)
    if fid in ("F06", "F07"):
        with open(os.path.join(fixture_dir, "branching_specification.txt"), "w") as handle:
            handle.write(candidate["program"] + "\n")
    write_preview(os.path.join(fixture_dir, "preview.png"), spec, rendered, candidate, graph_info, m)
    unbridged_summary = None
    if "unbridged" in rendered:
        u = rendered["unbridged"]
        u_dir = os.path.join(fixture_dir, "unbridged")
        os.makedirs(u_dir)
        u_record = dict(archive_record)
        u_record.update({"fixture_id": fid + "_unbridged", "polyline_origins": [
            ORIGINS[int(u["origin"][c[0]])] for c in V.graph.polylines(u["nodes"])],
            "note": "pre-bridge counterpart on the bridged fixture's grid. Re-render by mapping with "
                    "grammar_um_to_voxel_xyz (process_network on these nodes alone would re-centre them)"})
        save_archive(os.path.join(u_dir, "reference_network.npz"), u["nodes"], candidate["program"], u_record,
                     u["graph"], u["tree"])
        write_json(os.path.join(u_dir, "reference_graph.json"),
                   graph_json(spec, u["graph"], {"fixture_id": fid + "_unbridged",
                                                 "counterpart_note": "supporting pre-bridge counterpart, "
                                                                     "not a primary fixture"}))
        u_references = export_fixture_references(u_dir, fid + "_unbridged", u["nodes"], u["mask"], grid,
                                                 u["reference"] if write_6con else None)
        unbridged_summary = {"graph_betti": u["graph"]["betti"], "mask_topology": u["mask_topology"],
                             "reference_topology": u["reference_topology"],
                             "branches": len(u["graph"]["branches"]),
                             "references": reference_summary(u_references)}
    # file-level verification
    nifti_mask = read_nifti_check(os.path.join(fixture_dir, "vessel_mask.nii.gz"), rendered["mask"], grid["affine"])
    nifti_ref = read_nifti_check(os.path.join(fixture_dir, FIXTURE_NAMES["26_con_centered"]),
                                 centred_volume(references, shape), grid["affine"])
    pair_ok = (nifti_mask["affine"] == nifti_ref["affine"] and nifti_mask["zooms_mm"] == nifti_ref["zooms_mm"])
    spacing_ok = all(abs(z - SPACING_MM) < 1e-6 for z in nifti_mask["zooms_mm"] + nifti_ref["zooms_mm"])
    files_ok = all(x["shape_ok"] and x["values_ok"] and x["dtype"] == "uint8" and x["spatial_unit"] == "mm"
                   and x["affine_ok"] and x["qform_equals_sform"] for x in (nifti_mask, nifti_ref))
    records.append(check("nifti_spacing_pair", bool(spacing_ok and pair_ok and files_ok),
                         {"mask": nifti_mask, "reference": nifti_ref},
                         "0.05 mm spacing on each axis; mask and centred reference share shape, affine and "
                         "units; read-back equals the rendered mask and the union of the centred branch paths"))
    roundtrip = verify_archive_roundtrip(fixture_dir, nodes, graph_info, candidate["tree"], candidate["program"],
                                         grid, unbridged=rendered.get("unbridged"), reference=rendered["reference"])
    records.append(check("archive_roundtrip_and_rerender", all(roundtrip.values()), roundtrip,
                         "save/reload equality; exact re-render of the mask from the archive", kind="verification"))
    records.append(check("preview_png", os.path.getsize(os.path.join(fixture_dir, "preview.png")) > 0,
                         {"file": "preview.png"}, "inspectable PNG preview"))
    hard = [r for r in records if r["kind"] in ("hard", "verification")]
    desired = [r for r in records if r["kind"] == "desired"]
    warnings = [f"{r['check']}: {r['passed']}" for r in desired if r["passed"] is not True]
    if spec["id"] in ("F09", "F10", "F11", "F12") and not m.get("near_floor_terminals"):
        warnings.append("no near-floor (50-75 um) terminal branch in this fixture")
    selection = accepted.get("selection") or {}
    if selection.get("preference_met") is False:
        warnings.append(f"selection preference unmet within the budget: {selection['preference']}")
    artifacts = {}
    for root, _, files in os.walk(fixture_dir):
        for name in sorted(files):
            path = os.path.join(root, name)
            artifacts[os.path.relpath(path, fixture_dir)] = sha256_file(path)
    tort = m["tortuosity"]
    metadata = {
        "fixture_id": fid, "name": spec["name"], "level": spec["level"], "purpose": spec["purpose"],
        "recipe": spec["recipe"], "construction": candidate["construction"],
        "label": "SYNTHETIC GEOMETRY -- generating-centreline reference, not a measured anatomy",
        "root_diameter_um": {"specified": spec["root_diameter_um"], "actual": m["diameters"]["root_um"]},
        "diameter_range_um": m["diameters"],
        "bifurcation_asymmetry": m["bifurcation_asymmetry"],
        "grid": {"shape_xyz": shape.tolist(), "voxel_spacing_um": [VOXEL_UM] * 3,
                 "field_of_view_um": (shape * VOXEL_UM).tolist(), "border_voxels_required": BORDER_VOX,
                 "mask_border": border_report(rendered["mask"]), "geometry_bounds_vox": grid["geometry_bounds_vox"]},
        "coordinates": {"archive_units": "um", "array_axes": ["x", "y", "z"],
                        "grammar_um_to_voxel_xyz": grid["T"].tolist(),
                        "nifti_affine_voxel_to_mm": grid["affine"].tolist(), "nifti_spatial_unit": "mm",
                        "voxel_centre_convention": "voxel i covers [i - 0.5, i + 0.5) in voxel coordinates",
                        "relation": "p_mm = affine @ q = p_um / 1000; q = T @ p_um"},
        "generator": {"effective_parameters": candidate["params"], "common": COMMON, "plan": accepted["plan"],
                      "connect": True, "fit": "voxel_size", "clip_axes": [], "grow_in_volume": False,
                      "renderer": "computeVoxel.fit_to_volume + rasterise_segments(connect=True)",
                      "reference": "26-connected references from the generating centrelines (reference.py): "
                                   "a shortest ordered 26-path per branch through densely sampled source voxels, "
                                   "then re-routed towards higher foreground clearance while keeping per-branch "
                                   "voxel counts, endpoints and (26,6) topology; QA traces a 6-connected path "
                                   "with computeVoxel.rasterise_line (the renderer's own chain traversal)"},
        "seed": {"base_seed": spec["base_seed"], "candidate_index": accepted["index"], "seed": accepted["seed"],
                 "rule": "base_seed + 1000 * candidate_index"},
        "selection": accepted.get("selection"),
        "provenance": {"vsystem_commit": prov["vsystem"]["commit"], "vsystem_dirty": prov["vsystem"]["dirty_note"],
                       "python": prov["python"].split()[0],
                       "dependencies": prov["dependencies"],
                       "polyline_origins": archive_record["polyline_origins"]},
        "topology": {"intended": {"components": spec["required_graph_components"],
                                  "independent_cycles": spec["required_independent_cycle_range"],
                                  "branch_count_range": spec["desired_branch_count_range"]},
                     "graph": {**graph_info["betti"], "branches": len(graph_info["branches"]),
                               "tips": graph_info["tips"], "junctions": graph_info["junctions"],
                               "degree_histogram": graph_info["degree_histogram"]},
                     "rendered_mask": m["rendered_topology"], "reference_skeleton": m["reference_topology"],
                     "digital_topology_convention": TOPOLOGY_CONVENTION, "unbridged": unbridged_summary},
        "branch_count_convention": BRANCH_CONVENTION,
        "tortuosity": {k: v for k, v in tort.items() if k != "per_branch"},
        "tortuosity_per_branch": tort.get("per_branch"),
        "describe": m["describe"],
        "max_centreline_sample_spacing_um": m["max_sample_spacing_um"],
        "clearance": m["clearance"],
        "close_approach": m.get("close_approach"), "loop_openings": m.get("loop_openings"),
        "near_floor_terminals": m.get("near_floor_terminals"),
        "references": reference_summary(references),
        "events": {"vsystem": candidate["events"], "adapter": {k: v for k, v in candidate["extras"].items()
                                                                 if k not in ("bridges_assessed",)}},
        "bridges_assessed": candidate["extras"].get("bridges_assessed"),
        "interpretation": INTERPRETATION,
        "qa": {"passed": all(r["passed"] is True for r in hard), "hard_checks": hard, "desired_checks": desired,
               "warnings": warnings},
        "artifacts_sha256": artifacts,
    }
    write_json(os.path.join(fixture_dir, "metadata.json"), metadata)
    artifacts["metadata.json"] = sha256_file(os.path.join(fixture_dir, "metadata.json"))
    log(f"  wrote {fixture_dir} (QA {'pass' if metadata['qa']['passed'] else 'FAIL'})")
    return {"id": fid, "dir": fixture_dir, "metadata": metadata, "artifacts": artifacts, "records": records,
            "summary": {"id": fid, "name": spec["name"], "level": spec["level"],
                        "root_diameter_um": spec["root_diameter_um"], "min_diameter_um": m["diameters"]["min_um"],
                        "shape": shape.tolist(), "branches": len(graph_info["branches"]),
                        "tips": graph_info["tips"], "junctions": graph_info["junctions"],
                        "graph_cycles": graph_info["betti"]["cycles"],
                        "mask_b0_b1_b2": [m["rendered_topology"][k] for k in ("components", "cycles", "cavities")],
                        "arc_chord_length_weighted": tort.get("arc_chord_length_weighted"),
                        "seed": accepted["seed"], "candidate_index": accepted["index"],
                        "construction": candidate["construction"], "qa_passed": metadata["qa"]["passed"],
                        "warnings": warnings}}


DATASET_ID = "vsystem_laplacian_50um_v1"      # replaced by the manifest's dataset_id at run time
INTERPRETATION = [
    "At 50 um spacing a 50 um vessel is one voxel across; its sampling limits are part of the dataset.",
    "V-System connect=True draws every segment's face-connected voxel chain, forcing sub-voxel tubes into "
    "connected paths; accidental contacts are checked per branch under 26-connectivity.",
    "Saved paths are generating-centreline references, not exact medial axes of the voxelised vessel union.",
    "Curvature settings are synthetic test conditions, not validated anatomical reconstructions.",
    "Difficulty levels describe increasing intended challenges, not a guaranteed ordering of algorithm scores.",
]


def dataset_checks(manifest, written, pair_record):
    records = []
    ids = sorted(w["id"] for w in written)
    expected = [f["id"] for f in manifest["fixtures"]]
    records.append(check("twelve_primary_fixtures", ids == sorted(expected) and all(
        w["metadata"]["qa"]["passed"] for w in written), {"accepted": ids}, "exactly F01-F12 accepted with QA pass"))
    mins = [w["summary"]["min_diameter_um"] for w in written]
    roots = [w["summary"]["root_diameter_um"] for w in written]
    records.append(check("dataset_reaches_floor", bool(mins) and D_FLOOR_UM - ROUNDOFF_UM <= min(mins) <= 55.0,
                         {"smallest_diameter_um": min(mins) if mins else None},
                         "smallest vessel diameter approximately 50 um (50-55 um)"))
    records.append(check("root_diameters", bool(roots) and max(roots) <= D_ROOT_MAX_UM and len(set(roots)) > 1,
                         {"roots_um": dict(zip(ids, roots))}, "no root above 500 um; roots vary"))
    straight = [w["id"] for w in written if w["id"] in ("F01", "F02")]
    loops = [w["id"] for w in written if w["summary"]["graph_cycles"] > 0]
    records.append(check("dataset_composition", len(straight) == 2 and loops == ["F11", "F12"],
                         {"straight_controls": straight, "loop_fixtures": loops},
                         "two straight controls, eight tortuous loop-free, two tortuous with loops"))
    levels = {}
    for w in written:
        levels.setdefault(w["summary"]["level"], []).append(w["id"])
    records.append(check("levels", sorted(levels) == [0, 1, 2, 3], levels,
                         "simple (0-1), moderate (2) and complex (3) levels present", kind="desired"))
    return records


CONSTRUCTION_LABELS = {
    "straight_tube": "explicit straight tube", "straight_y": "explicit straight Y",
    "walk_program": "custom walk program", "tree": "grammar tree", "tree_thin": "grammar tree + thin daughters",
    "tree_companion": "grammar tree + companion", "tree_bridges": "grammar tree + bridges",
}

README_TEMPLATE = """# {dataset_id}

Synthetic vessel fixtures for regression-testing Laplacian (and other) 3D vessel
skeletonization. **Synthetic geometry**: generated by V-System (fixtures.py), not
measured anatomy.

## Contents

- `F01_*` ... `F12_*`: one directory per primary fixture
  - `vessel_mask.nii.gz`: binary uint8 0/1 vessel mask (the skeletonizer input)
  - Reference centrelines on the same grid (neither is a skeletonization of the mask):
    - `reference_skeleton_26_con_centered.nii.gz`: foreground-centred 26-connected
      reference; **the version to use for quantitative evaluation**. Each branch is
      re-routed, within the `_26_con` candidate voxels, towards higher foreground
      distance-transform clearance, keeping per-branch voxel counts, endpoints,
      junction anchors and (26,6) topology
    - `reference_skeleton_26_con.nii.gz`: the generating centrelines re-traced as a
      shortest ordered 26-connected path per branch, keeping the rounded branch endpoints
    - `reference_skeleton_6_con.nii.gz` (only with `--write-6con`): the generating
      centrelines traced with the renderer's own voxel chain
  - `reference_network.npz`: V-System centreline archive (`nodes` (4, N) x, y, z,
    diameter in um with NaN separators; `edges`, `node_kind`, `tree`, `program`,
    `metadata`)
  - `reference_graph.json`: topology nodes (tips, junctions) with stable IDs, and
    branches (maximal paths between them) with ordered XYZ and diameter samples in um
  - `reference_graph.graphml`: the same graph in SkelHub GraphML form, one node per tip
    or junction and one edge per branch. Edges carry the smooth generating centreline
    (`centerline_world_points` in mm, `centerline_voxel_points`) and the matching
    `_26_con_centered` branch path (`centerline_voxels`). No diameters
  - `metadata.json`: parameters, seed, provenance, topology, tortuosity, clearance,
    QA results and SHA-256 checksums
  - `preview.png`: three maximum-intensity projections with the generating centreline
  - F06/F07 `branching_specification.txt`: the shared pre-avoidance grammar string
  - F11/F12 `unbridged/`: the pre-bridge tree, rendered on the loop fixture's grid, with
    the same mask, references and GraphML
- `manifest.requested.json`: the handoff manifest as received
- `manifest.resolved.json`: actual parameters, adjustments, selection history and
  unmet requirements
- `validation_report.json`: every check with measured values and pass/fail
- `dataset_index.csv`: one row per fixture
- `generation.log`: the console log of the generating run

## Conventions

- Voxel spacing 50 um isotropic (NIfTI 0.05 mm, units mm), arrays indexed (x, y, z).
- `q = T p`: archive micrometres `p` to voxel coordinates `q`, with `T = diag(1/50) + t`
  (`grammar_um_to_voxel_xyz` in metadata). NIfTI affine `diag(0.05) - 0.05 t` maps `q`
  back to `p / 1000` mm. qform and sform match.
- Voxel `i` covers `[i - 0.5, i + 0.5)`. At least {border} empty voxels on every side.
- Rendering: `computeVoxel.fit_to_volume(fit="voxel_size", voxel_size=50)` then
  `rasterise_segments(connect=True)`. Both 26-connected references are checked to lie
  inside the mask and to keep the source graph's (26,6) digital topology.
- Digital topology: {topology}
- Branch count: {branches}

## Fixtures

{table}

Grammar trees are vSystem.F trees grown by `main.grow_network` with the persistent
random walk and collision avoidance. F08 and F10 add controlled side branches (thin
50-70 um daughters; one companion routed alongside a nonincident branch). F11 and F12
keep a seeded subset of the bridges made by `anastomosis.anastomose`.

## Decisions and limitations

- **Buried stubs pruned.** V-System ends a colliding daughter a step or two after its
  junction, leaving a tip inside or against another vessel. Terminal branches whose tip
  lies less than 100 um (2 voxels) outside every other tube are removed after growth,
  before anything else. Each one is listed in `metadata.json` (`events.adapter.pruned_stubs`).
- **Clearance.** Tubes may overlap only inside V-System's junction neighbourhoods at the
  100 um generation margin. Growth checks points, not tubes, so a few densified pairs sit
  3-8 um under 100 um without touching. Both figures are in `metadata.json`.
- **Selection.** Candidates are tried in seed order. F09-F12 prefer a candidate with a
  near-floor (50-75 um) terminal; F06/F07 prefer one where both realise the same topology.
  Rejected candidates and their reasons are in `manifest.resolved.json`.
- **F06/F07** share seed, grammar string and diameters. F07 only lowers the walk
  persistence (6 vs 12 diameters).
- **Side branches** in F08 and F10 keep a constant calibre. The host keeps its calibre too:
  the Murray-law change is under 1% and is ignored.
- **Arc/chord on loop fixtures.** A bridge joins two old terminal branches into one long
  branch between junctions that can be close together, so F11/F12 arc/chord means are high.
- **Unbridged counterparts** use the loop fixture's grid. Re-render them with the stored
  transform; `process_network` on those nodes alone would re-centre them.
- No TIFFs, 12.5 um renders, noise or algorithm runs (disabled by default in the manifest).

## Interpretation

{interpretation}

## Reproduce

```bash
cd {vsystem}
{command} --out <new empty directory>
```

V-System commit `{commit}` ({dirty}); Python {python}. fixtures.py refuses an existing
nonempty output directory.

## Status

{status}
"""


def write_dataset_files(out_dir, manifest, manifest_path, prov, written, histories, adapter_tests,
                        dataset_records, unmet, log, argv=None, wanted=None):
    with open(manifest_path) as source, open(os.path.join(out_dir, "manifest.requested.json"), "w") as target:
        target.write(source.read())
    by_id = {w["id"]: w for w in written}
    # dataset index
    with open(os.path.join(out_dir, "dataset_index.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fixture_id", "name", "level", "root_diameter_um", "min_diameter_um", "shape_x", "shape_y",
                         "shape_z", "branches", "tips", "junctions", "graph_cycles", "mask_b0", "mask_b1", "mask_b2",
                         "arc_chord_length_weighted", "seed", "candidate_index", "qa_passed", "directory"])
        for f in manifest["fixtures"]:
            w = by_id.get(f["id"])
            if w is None:
                writer.writerow([f["id"], f["name"], f["level"], f["root_diameter_um"]] + [""] * 15 + ["NOT ACCEPTED"])
                continue
            s = w["summary"]
            writer.writerow([s["id"], s["name"], s["level"], s["root_diameter_um"], round(s["min_diameter_um"], 3),
                             *s["shape"], s["branches"], s["tips"], s["junctions"], s["graph_cycles"],
                             *s["mask_b0_b1_b2"],
                             "" if s["arc_chord_length_weighted"] is None else round(s["arc_chord_length_weighted"], 4),
                             s["seed"], s["candidate_index"], s["qa_passed"], os.path.basename(w["dir"])])
    expected = len(manifest["fixtures"]) if wanted is None else len(set(wanted))
    complete = (len(written) == expected and not unmet
                and all(r["passed"] is True for r in dataset_records if r["kind"] == "hard"))
    report = {
        "dataset_id": DATASET_ID, "output_directory": os.path.abspath(out_dir),
        "status": ("complete" if wanted is None else "partial") if complete else "incomplete",
        "summary": {"accepted_fixtures": sorted(by_id), "all_hard_checks_passed": complete,
                    "unmet_requirements": unmet},
        "adapter_verification": adapter_tests, "dataset_checks": dataset_records,
        "fixtures": {fid: {"qa_passed": w["metadata"]["qa"]["passed"], "checks": w["records"],
                           "warnings": w["metadata"]["qa"]["warnings"]} for fid, w in by_id.items()},
        "selection_history": histories,
        "provenance": prov,
    }
    write_json(os.path.join(out_dir, "validation_report.json"), report)
    resolved = dict(manifest)
    resolved["resolution"] = {
        "status": report["status"], "resolved_at": prov["generated_at"],
        "output_directory": os.path.abspath(out_dir),
        "adjustments": ADJUSTMENTS,
        "unmet_requirements": unmet,
        "common_effective": {**COMMON, "voxel_spacing_um": [VOXEL_UM] * 3, "fit": "voxel_size", "connect": True,
                             "clip_axes": [], "grow_in_volume": False, "border_voxels": BORDER_VOX,
                             "dense_clearance_step_um": DENSE_STEP_UM, "loop_opening_min_um": OPENING_MIN_UM,
                             "tortuosity_targets": TORTUOSITY_TARGETS, "candidate_budget": histories.get("_budget"),
                             "digital_topology_convention": TOPOLOGY_CONVENTION,
                             "branch_count_convention": BRANCH_CONVENTION},
        "fixtures": {},
        "provenance": prov,
        "not_produced": {"tiff": "optional; not written (NIfTI xyz 0/1 is the primary format)",
                         "optional_disabled_by_default": manifest["outputs"]["optional_disabled_by_default"]},
    }
    for f in manifest["fixtures"]:
        w = by_id.get(f["id"])
        entry = {"accepted": w is not None, "selection_history": histories.get(f["id"], [])}
        if w is not None:
            md = w["metadata"]
            entry.update({"seed": md["seed"], "construction": md["construction"],
                          "effective_parameters": md["generator"]["effective_parameters"],
                          "plan": md["generator"]["plan"], "grid": md["grid"],
                          "topology": md["topology"], "tortuosity": md["tortuosity"],
                          "qa_passed": md["qa"]["passed"], "warnings": md["qa"]["warnings"],
                          "artifacts_sha256": w["artifacts"]})
        resolved["resolution"]["fixtures"][f["id"]] = entry
    write_json(os.path.join(out_dir, "manifest.resolved.json"), resolved)
    rows = ["| ID | Level | Name | Root (um) | Min d (um) | Branches | Cycles | Shape | Construction |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for f in manifest["fixtures"]:
        w = by_id.get(f["id"])
        if w is None:
            rows.append(f"| {f['id']} | {f['level']} | {f['name']} | {f['root_diameter_um']:.0f} | | | | | NOT ACCEPTED |")
            continue
        s = w["summary"]
        rows.append(f"| {s['id']} | {s['level']} | {s['name']} | {s['root_diameter_um']:.0f} | "
                    f"{s['min_diameter_um']:.1f} | {s['branches']} | {s['graph_cycles']} | "
                    f"{'x'.join(map(str, s['shape']))} | {CONSTRUCTION_LABELS[PLANS[s['id']]['builder']]} |")
    status = ((f"All {len(manifest['fixtures'])} fixtures accepted" if wanted is None else
               f"Requested fixtures {', '.join(sorted(by_id))} accepted")
              + "; every hard check passed (see validation_report.json)." if complete else "INCOMPLETE. Unmet requirements:\n\n" + "\n".join(f"- {u}" for u in unmet))
    with open(os.path.join(out_dir, "README.md"), "w") as handle:
        handle.write(README_TEMPLATE.format(
            dataset_id=DATASET_ID, border=BORDER_VOX, topology=TOPOLOGY_CONVENTION, branches=BRANCH_CONVENTION,
            table="\n".join(rows), interpretation="\n".join(f"- {x}" for x in INTERPRETATION),
            vsystem=prov["vsystem"]["path"], commit=prov["vsystem"]["commit"][:7],
            dirty=prov["vsystem"]["dirty_note"], python=prov["python"].split()[0], status=status,
            command=" ".join(argv or ["python fixtures.py", "--manifest", "manifest.requested.json"])))
    log(f"dataset status: {report['status']}")
    return report


DEFAULT_MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets", "fixtures",
                                "vsystem_laplacian_50um_v1.json")


def main(argv=None):
    global DATASET_ID
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                        help="fixture manifest (default presets/fixtures/vsystem_laplacian_50um_v1.json)")
    parser.add_argument("--out", required=True, help="new or empty output directory")
    parser.add_argument("--only", default=None, help="comma-separated fixture IDs (development runs)")
    parser.add_argument("--budget", type=int, default=None, help="candidate budget (default: manifest)")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="added to every fixture's base seed, for new networks from the same recipes "
                             "(default 0: the manifest's seeds)")
    parser.add_argument("--write-6con", action="store_true",
                        help="also write reference_skeleton_6_con.nii.gz, the renderer-chain trace QA uses")
    args = parser.parse_args(argv)
    vsystem_path = os.path.dirname(os.path.abspath(__file__))
    with open(args.manifest) as handle:
        manifest = json.load(handle)
    DATASET_ID = manifest.get("dataset_id", DATASET_ID)
    budget = args.budget or int(manifest["common"]["reproducibility"]["candidate_budget_initial"])
    out_dir = os.path.abspath(args.out)
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise SystemExit(f"refusing to write into the existing nonempty directory {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "generation.log")

    def log(message):
        print(message, flush=True)
        with open(log_path, "a") as handle:
            handle.write(message + "\n")

    prov = provenance(vsystem_path, sys.argv if argv is None else ["fixtures.py", *argv])
    log(f"{DATASET_ID}: V-System {prov['vsystem']['commit'][:7]} ({prov['vsystem']['dirty_note']}), "
        f"budget {budget}, seed offset {args.seed_offset}")
    adapter_tests = adapter_self_tests()
    for t in adapter_tests:
        log(f"self-test {t['check']}: {'pass' if t['passed'] else 'FAIL'}")
    specs = {s["id"]: s for s in fixture_specs(manifest)}
    for spec in specs.values():
        spec["base_seed"] = int(spec["base_seed"]) + args.seed_offset
    wanted = [s for s in specs] if not args.only else args.only.split(",")
    histories = {"_budget": budget}
    written, unmet = [], []
    unknown = [fid for fid in wanted if fid not in specs]
    if unknown:
        raise SystemExit(f"unknown fixture IDs {unknown}; the manifest has {sorted(specs)}")
    for fid in wanted:
        if fid == "F07" and "F06" in wanted:
            continue                        # built with F06, which shares its seed and grammar string
        spec = specs[fid]
        log(f"{fid} {spec['name']}")
        if fid in ("F06", "F07"):
            spec = specs["F06"]
            pair, history = run_pair(spec, specs["F07"], budget, log)
            histories.update(history)
            if pair is None:
                unmet.append(f"F06/F07: no candidate index within the budget of {budget} passed both fixtures "
                             "and the paired tortuosity check")
                continue
            for pid in ("F06", "F07"):
                if args.only and pid not in wanted:
                    continue
                written.append(write_fixture(out_dir, specs[pid], pair[pid], prov, log, args.write_6con))
            continue
        accepted, history = run_fixture(spec, budget, log)
        histories[fid] = history
        if accepted is None:
            reasons = sorted({f["check"] for h in history for f in h["failed"]})
            unmet.append(f"{fid}: candidate budget of {budget} exhausted; failing checks: {', '.join(reasons)}")
            continue
        written.append(write_fixture(out_dir, spec, accepted, prov, log, args.write_6con))
    dataset_records = dataset_checks(manifest, written, None) if not args.only else []
    command = ["python fixtures.py", "--manifest", os.path.relpath(os.path.abspath(args.manifest), vsystem_path)]
    if args.seed_offset:
        command += ["--seed-offset", str(args.seed_offset)]
    report = write_dataset_files(out_dir, manifest, args.manifest, prov, written, histories,
                                 adapter_tests, dataset_records, unmet, log, command,
                                 wanted if args.only else None)
    return 0 if report["status"] in ("complete", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())
