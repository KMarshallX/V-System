"""Optional NIfTI masks and source-derived 26-connected centreline references.

The path objective and sequential topology gate reproduce the 50 um fixture
procedure. References are constrained digital paths, not exact medial axes.
"""
import hashlib
import importlib
from itertools import product
from pathlib import Path
import shutil
import tempfile

import numpy as np

import graph
from computeVoxel import fit_to_volume


SAMPLE_STEP_VOX = 0.05
CLEARANCE_TOL_MM = 1e-12
UNIT_TO_MM = {"um": 0.001, "µm": 0.001, "μm": 0.001, "mm": 1.0, "m": 1000.0}
SUFFIXES = {
    "mask": "_mask.nii.gz",
    "uncentered": "_reference_26_con.nii.gz",
    "centered": "_reference_26_con_centered.nii.gz",
}


def dependencies():
    """Load the optional export packages, or explain how to install them."""
    try:
        return importlib.import_module("nibabel"), importlib.import_module("scipy.ndimage")
    except ImportError as error:
        raise ValueError('--nifti needs optional packages; install with pip install ".[nifti]"') from error


def check_options(fit, voxel_size, units):
    """Validate export settings before generation; return millimetres per unit."""
    if fit != "voxel_size":
        raise ValueError("--nifti requires --fit voxel_size")
    if voxel_size is None or not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("--nifti requires a positive, finite --voxel-size")
    factor = UNIT_TO_MM.get(units.strip().lower())
    if factor is None:
        raise ValueError("--nifti requires physical --units um, mm or m")
    dependencies()
    return factor


def rendering_grid(nodes, shape, voxel_size, units):
    """Return rendered XYZ points, source-to-voxel transform and mm affine.

    Uses the renderer's fit_to_volume mapping on the complete network, so
    branches are never fitted independently. Does not modify the input nodes.
    """
    factor = check_options("voxel_size", voxel_size, units)
    points, _ = fit_to_volume(nodes, shape, fit="voxel_size", voxel_size=voxel_size)
    lo = np.nanmin(nodes[:3], axis=1)
    extent = np.nanmax(nodes[:3], axis=1) - lo
    scale = 1.0 / voxel_size
    offset = (np.asarray(shape) - scale * extent) / 2.0
    transform = np.eye(4)
    transform[:3, :3] /= voxel_size
    transform[:3, 3] = offset - lo * scale
    affine = np.eye(4)
    affine[:3, :3] *= voxel_size * factor
    affine[:3, 3] = (lo - offset * voxel_size) * factor
    return points, transform, affine


def thin_path(points, foreground_distance=None):
    """Select a shortest ordered 26-path, optionally favouring physical EDT.

    Densify source segments to at most SAMPLE_STEP_VOX per coordinate, round
    with floor(q + .5), and retain candidate visits in source order. Forward
    shortcuts stop at the first candidate outside the current 26-neighbourhood.
    Costs are ordered by step count, negative summed clearance, then sampled
    geometry error. Endpoints and first-encountered exact ties are retained.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Each branch needs at least two finite XYZ samples")
    dense = []
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / SAMPLE_STEP_VOX)))
        dense.append(a + np.arange(n)[:, None] / n * (b - a))
    samples = np.concatenate([*dense, points[-1:]])
    indices = np.floor(samples + 0.5).astype(int)
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(indices, axis=0), axis=1)) + 1]
    candidates = indices[starts]
    errors = np.minimum.reduceat(np.sum((indices - samples) ** 2, axis=1), starts)
    clearance = np.zeros(len(candidates))
    if foreground_distance is not None:
        foreground_distance = np.asarray(foreground_distance)
        if (foreground_distance.ndim != 3 or np.any(candidates < 0)
                or np.any(candidates >= foreground_distance.shape)):
            raise ValueError("Candidate voxels must be inside the distance grid")
        clearance = foreground_distance[tuple(candidates.T)]
        if not np.all(np.isfinite(clearance) & (clearance > 0)):
            raise ValueError("Every candidate must have positive finite foreground clearance")
    costs = [(0, 0., 0.)] + [(len(candidates) + 1, np.inf, np.inf)] * (len(candidates) - 1)
    parent = np.full(len(candidates), -1, dtype=int)
    for i, point in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            if np.max(np.abs(candidates[j] - point)) > 1:
                break
            if np.array_equal(candidates[j], point):
                continue
            cost = (costs[i][0] + 1, costs[i][1] - float(clearance[j]),
                    costs[i][2] + float(errors[j]))
            if cost < costs[j]:
                costs[j] = cost
                parent[j] = i
    selected = [len(candidates) - 1]
    while selected[-1] != 0:
        previous = int(parent[selected[-1]])
        if previous < 0:
            raise ValueError("Source samples do not admit an ordered 26-connected path")
        selected.append(previous)
    return candidates[selected[::-1]]


def topology(mask):
    """Return (components, tunnels, cavities) for foreground 26/background 6.

    Count the closed-cube cells exactly as in the fixture procedure. Requires
    scipy for labelling, but no additional topology package.
    """
    _, ndimage = dependencies()
    mask = np.asarray(mask, dtype=bool)
    components = ndimage.label(mask, ndimage.generate_binary_structure(3, 3))[1]
    cavities = ndimage.label(~np.pad(mask, 1), ndimage.generate_binary_structure(3, 1))[1] - 1
    coordinates = np.argwhere(mask)

    def count(offsets):
        """Count distinct incident cells for the given corner offsets."""
        return len(np.unique(np.concatenate([coordinates + p for p in offsets]), axis=0))

    vertices = count(list(product((0, 1), repeat=3)))
    edges = faces = 0
    for axis in range(3):
        edges += count([p for p in product((0, 1), repeat=3) if p[axis] == 0])
        offset = np.zeros(3, dtype=int)
        offset[axis] = 1
        faces += count([np.zeros(3, dtype=int), offset])
    euler = vertices - edges + faces - len(coordinates)
    return int(components), int(components + cavities - euler), int(cavities)


def accept_centered_path(counts, old, new, expected):
    """Update shared voxel counts only when a proposal retains union topology."""
    if np.array_equal(old, new):
        return True
    np.add.at(counts, tuple(old.T), -1)
    np.add.at(counts, tuple(new.T), 1)
    if topology(counts > 0) == tuple(expected):
        return True
    np.add.at(counts, tuple(new.T), -1)
    np.add.at(counts, tuple(old.T), 1)
    return False


def geometry_errors(path, source_points, affine):
    """Measure voxel-centre distances to the source piecewise-linear axis in mm."""
    nib, _ = dependencies()
    path = nib.affines.apply_affine(affine, path)
    points = nib.affines.apply_affine(affine, source_points)
    a, delta = points[:-1], np.diff(points, axis=0)
    squared = np.sum(delta ** 2, axis=1)
    valid = squared > 0
    a, delta, squared = a[valid], delta[valid], squared[valid]
    if not len(a):
        distance = np.linalg.norm(path - points[0], axis=1)
    else:
        distance = []
        for chunk in np.array_split(path, max(1, int(np.ceil(len(path) / 128)))):
            relative = chunk[:, None, :] - a
            t = np.clip(np.sum(relative * delta, axis=2) / squared, 0, 1)
            distance.extend(np.linalg.norm(relative - t[:, :, None] * delta, axis=2).min(axis=1))
        distance = np.array(distance)
    return {"mean_mm": float(distance.mean()), "max_mm": float(distance.max())}


def _check_path(path, mask):
    """Check array bounds, foreground membership and nonzero 26-neighbour steps."""
    if np.any(path < 0) or np.any(path >= mask.shape):
        raise ValueError("Reference path leaves the rendering grid")
    if not mask[tuple(path.T)].all():
        raise ValueError("Reference path leaves the vessel foreground")
    if not np.all(np.max(np.abs(np.diff(path, axis=0)), axis=1) == 1):
        raise ValueError("Reference path has invalid 26-connected steps")


def build_references(nodes, edges, mask, points, affine):
    """Build both references on an existing grid and return arrays and provenance.

    points is the complete network's (3, N) rendered coordinate array. Accepting
    it explicitly also permits replaying a control on its parent's fixed grid.
    No source graph JSON is needed. Inputs are not modified.
    """
    _, ndimage = dependencies()
    nodes = np.asarray(nodes, dtype=float)
    points = np.asarray(points, dtype=float)
    mask = np.asarray(mask)
    affine = np.asarray(affine, dtype=float)
    if mask.ndim != 3 or not np.isin(mask, [0, 1]).all() or not mask.any():
        raise ValueError("Reference export needs a nonempty binary 3D vessel mask")
    mask = mask.astype(bool)
    if nodes.ndim != 2 or nodes.shape[0] != 4 or points.shape != (3, nodes.shape[1]):
        raise ValueError("Source and rendered coordinate arrays must have matching columns")
    finite = np.isfinite(nodes[:3]).all(axis=0)
    if not finite.any() or not np.isfinite(points[:, finite]).all():
        raise ValueError("Source geometry must contain finite rendered coordinates")
    if np.any(points[:, finite] < -0.5) or np.any(points[:, finite] >= np.array(mask.shape)[:, None] - 0.5):
        raise ValueError("Source centreline leaves the rendering field of view; enlarge --volume or use --grow-in-volume")
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1])):
        raise ValueError("Reference export needs a finite spatial affine")
    axes = affine[:3, :3]
    spacing = np.linalg.norm(axes, axis=0)
    if np.any(spacing <= 0) or not np.allclose(axes.T @ axes, np.diag(spacing ** 2), atol=1e-12):
        raise ValueError("Reference export needs an orthogonal grid with positive spacing")
    canonical = graph.canonical_columns(nodes)
    expected_graph = graph.betti(edges, canonical)
    expected = (expected_graph["components"], expected_graph["cycles"], 0)
    paths = graph.segments(nodes, edges, canonical)
    if not paths or np.any(graph.degree(edges, nodes.shape[1])[canonical[finite]] == 0):
        raise ValueError("Reference export needs branches without isolated source vertices")
    baseline = np.zeros(mask.shape, dtype=np.uint8)
    counts = np.zeros(mask.shape, dtype=np.int32)
    old_paths = []
    for indices in paths:
        old = thin_path(points[:, indices].T)
        _check_path(old, mask)
        old_paths.append(old)
        baseline[tuple(old.T)] = 1
        np.add.at(counts, tuple(old.T), 1)
    measured = topology(baseline)
    if measured != expected:
        raise ValueError(f"Uncentred reference topology {measured} differs from source expectation {expected}")
    distance = ndimage.distance_transform_edt(np.pad(mask, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    centered = np.zeros(mask.shape, dtype=np.uint8)
    branches = []
    for bi, (indices, old) in enumerate(zip(paths, old_paths)):
        source = points[:, indices].T
        new = thin_path(source, foreground_distance=distance)
        _check_path(new, mask)
        if len(old) != len(new) or not np.array_equal(old[[0, -1]], new[[0, -1]]):
            raise ValueError("Centering changed a branch endpoint or path-entry count")
        before, after = distance[tuple(old.T)], distance[tuple(new.T)]
        if after.sum() < before.sum() - CLEARANCE_TOL_MM:
            raise ValueError("Centering decreased summed branch clearance")
        proposal_mean = float(after.mean())
        accepted = accept_centered_path(counts, old, new, expected)
        if not accepted:
            new, after = old, before
        centered[tuple(new.T)] = 1
        branches.append({
            "id": f"B{bi:04d}", "source_vertex_columns": indices.tolist(),
            "endpoint_columns": indices[[0, -1]].tolist(),
            "uncentered_voxel_path_xyz": old.tolist(), "voxel_path_xyz": new.tolist(),
            "voxel_count": len(new), "changed": not np.array_equal(old, new),
            "proposal_accepted": accepted, "proposal_mean_clearance_mm": proposal_mean,
            "rejection_reason": None if accepted else "Would change full-reference digital topology",
            "mean_clearance_before_mm": float(before.mean()), "mean_clearance_after_mm": float(after.mean()),
            "min_clearance_before_mm": float(before.min()), "min_clearance_after_mm": float(after.min()),
            "geometry_error_before": geometry_errors(old, source, affine),
            "geometry_error_after": geometry_errors(new, source, affine),
        })
    if not np.array_equal(counts > 0, centered != 0) or topology(centered) != expected:
        raise ValueError("Centered branch union failed its topology or reconstruction check")
    report = {
        "status": "ok", "method_version": 1,
        "method": "Minimum steps, maximum summed physical EDT, then minimum sampled geometry error; sequential topology acceptance in source branch order",
        "sample_step_per_axis_voxels": SAMPLE_STEP_VOX,
        "rounding": "floor(q + 0.5)", "clearance_sum_tolerance_mm": CLEARANCE_TOL_MM,
        "axis_order": "xyz", "spatial_units": "mm", "affine_voxel_to_mm": affine.tolist(),
        "shape_xyz": list(mask.shape), "spacing_mm": spacing.tolist(),
        "topology_convention": "foreground 26 / background 6; closed-cube Euler characteristic",
        "topology_components_tunnels_cavities": list(expected),
        "old_voxels": int(baseline.sum()), "new_voxels": int(centered.sum()),
        "changed_voxels": int(np.count_nonzero(centered != baseline)), "branches": branches,
        "validation": "Passed containment, endpoints, path counts, 26-steps, clearance, branch union and digital topology checks",
        "limitations": "Source-constrained digital reference, not an exact medial axis. Fixed anchors; order-dependent proposals; equal path counts need not mean equal physical lengths. EDT is distance to background voxel centres.",
    }
    return {"mask": mask.astype(np.uint8), "uncentered": baseline, "centered": centered}, report


def _save_checked(path, data, affine):
    """Save and reload a uint8 NIfTI with matching qform/sform and mm units."""
    nib, _ = dependencies()
    image = nib.Nifti1Image(data, affine)
    image.set_data_dtype(np.uint8)
    image.header.set_xyzt_units("mm")
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, path)
    loaded = nib.load(path)
    if loaded.shape != data.shape or loaded.get_data_dtype() != np.dtype("uint8"):
        raise ValueError("NIfTI round-trip changed shape or datatype")
    if not np.array_equal(np.asarray(loaded.dataobj), data):
        raise ValueError("NIfTI round-trip changed voxel values")
    # NIfTI-1 stores spatial transforms in float32.
    for value in (loaded.affine, loaded.get_qform(), loaded.get_sform()):
        np.testing.assert_allclose(value, affine, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(loaded.header.get_zooms(), np.linalg.norm(affine[:3, :3], axis=0), rtol=1e-6)
    if loaded.header.get_xyzt_units()[0] != "mm" or int(loaded.header["qform_code"]) != 1 or int(loaded.header["sform_code"]) != 1:
        raise ValueError("NIfTI round-trip changed spatial units or transform codes")


def check_destinations(directory, stem):
    """Return triplet paths, refusing collisions before normal outputs change."""
    targets = {kind: Path(directory) / (stem + suffix) for kind, suffix in SUFFIXES.items()}
    for target in targets.values():
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite NIfTI: {target}")
    return targets


def export_nifti(directory, stem, nodes, edges, mask, voxel_size, units):
    """Validate, stage and publish a NIfTI triplet; return its metadata.

    Refuse existing destinations. If publishing fails, remove only files created
    by this call. Temporary files are cleaned on success and failure alike.
    """
    points, transform, affine = rendering_grid(nodes, mask.shape, voxel_size, units)
    # Use the actual NIfTI-1 grid for EDT and diagnostics as well as the files.
    affine = affine.astype(np.float32).astype(float)
    arrays, report = build_references(nodes, edges, mask, points, affine)
    report["source_to_voxel_xyz"] = transform.tolist()
    report["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    directory = Path(directory)
    targets = check_destinations(directory, stem)
    created = []
    with tempfile.TemporaryDirectory(prefix=".nifti-", dir=directory) as temporary:
        staged = {kind: Path(temporary) / target.name for kind, target in targets.items()}
        for kind, path in staged.items():
            _save_checked(path, arrays[kind], affine)
        report["outputs"] = {kind: {"file": targets[kind].name,
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                             for kind, path in staged.items()}
        try:
            for kind, target in targets.items():
                with target.open("xb") as stream:
                    created.append(target)
                    with staged[kind].open("rb") as source:
                        shutil.copyfileobj(source, stream)
                if hashlib.sha256(target.read_bytes()).hexdigest() != report["outputs"][kind]["sha256"]:
                    raise ValueError(f"Published NIfTI checksum mismatch: {target}")
        except BaseException:
            for target in reversed(created):
                target.unlink()
            raise
    return report
