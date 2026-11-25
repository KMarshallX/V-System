"""NIfTI vessel masks, source-derived 26-connected centreline references and GraphML.

The path objective and sequential topology gate reproduce the 50 um fixture
procedure. References are constrained digital paths, not exact medial axes.
"""
import hashlib
import importlib
import json
from itertools import product
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np

import graph
from computeVoxel import fit_to_volume


SAMPLE_STEP_VOX = 0.05
CLEARANCE_TOL_MM = 1e-12
MERGE_TOL = graph.DEFAULT_TOL        # archive columns this close are one vertex
CHAIN_REACH_VOX = math.sqrt(3.0) / 2.0   # a connect-chain voxel centre lies within this of the centreline
UNIT_TO_MM = {"um": 0.001, "µm": 0.001, "μm": 0.001, "mm": 1.0, "m": 1000.0}
KINDS = ("mask", "26_con", "26_con_centered", "graphml")
# file names of one fixture directory (the vsystem_laplacian_50um_v1 layout)
FIXTURE_NAMES = {
    "mask": "vessel_mask.nii.gz",
    "26_con": "reference_skeleton_26_con.nii.gz",
    "26_con_centered": "reference_skeleton_26_con_centered.nii.gz",
    "graphml": "reference_graph.graphml",
}


def stem_names(stem):
    """File names of one main.py network, all prefixed by its stem."""
    return {
        "mask": stem + "_vessel_mask.nii.gz",
        "26_con": stem + "_reference_26_con.nii.gz",
        "26_con_centered": stem + "_reference_26_con_centered.nii.gz",
        "graphml": stem + "_reference_graph.graphml",
    }


def dependencies():
    """Load the export packages, or explain how to install them."""
    try:
        return importlib.import_module("nibabel"), importlib.import_module("scipy.ndimage")
    except ImportError as error:
        raise ValueError("the reference export needs nibabel and scipy; install V-System with pip install -e .") \
            from error


def _igraph():
    try:
        return importlib.import_module("igraph")
    except ImportError as error:
        raise ValueError("the GraphML export needs python-igraph; install V-System with pip install -e .") \
            from error


def check_options(fit, voxel_size, units):
    """Validate export settings before generation; return millimetres per unit."""
    if fit != "voxel_size":
        raise ValueError("the reference export requires --fit voxel_size (use --original_export for the "
                         "isotropic and stretch fits)")
    if voxel_size is None or not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("the reference export requires a positive, finite --voxel-size")
    factor = UNIT_TO_MM.get(units.strip().lower())
    if factor is None:
        raise ValueError("the reference export requires physical --units um, mm or m")
    dependencies()
    _igraph()
    return factor


def choose_grid(nodes, voxel_size, units, border=4):
    """Volume shape, transform and affine that hold the whole network at a fixed voxel size.

    The shape covers the geometry bounds including radii (and the connect-chain reach of
    sqrt(3)/2 voxel for thin vessels) with `border` empty voxels on every side, under
    computeVoxel.fit_to_volume(fit="voxel_size"), which centres the centreline extent.
    Returns (shape, transform, affine): transform maps source units to voxel xyz and
    affine maps voxel xyz to millimetres.
    """
    factor = check_options("voxel_size", voxel_size, units)
    scale = 1.0 / float(voxel_size)
    xyz = nodes[:3]
    finite = np.isfinite(xyz).all(axis=0)
    if not finite.any():
        raise ValueError("the network has no points")
    lo = np.nanmin(xyz, axis=1)
    extent = np.nanmax(xyz, axis=1) - lo
    rel = (xyz[:, finite] - lo[:, None]) * scale
    r_eff = np.maximum(nodes[3, finite] / 2.0 * scale, CHAIN_REACH_VOX)
    low = (rel - r_eff).min(axis=1)
    high = (rel + r_eff).max(axis=1)
    shape = []
    for axis in range(3):
        n = int(math.ceil(high[axis] - low[axis] + 2 * border))
        while True:
            offset = (n - scale * extent[axis]) / 2.0
            if offset + low[axis] >= border - 0.5 and offset + high[axis] <= n - border - 0.5:
                break
            n += 1
        shape.append(n)
    shape = np.array(shape, dtype=int)
    offset = (shape - scale * extent) / 2.0
    t_vox = offset - lo * scale
    transform = np.eye(4)
    transform[:3, :3] *= scale
    transform[:3, 3] = t_vox
    affine = np.eye(4)
    affine[:3, :3] *= voxel_size * factor
    affine[:3, 3] = -voxel_size * factor * t_vox
    return tuple(int(v) for v in shape), transform, affine


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


def check_destinations(directory, names):
    """Return output paths by kind, refusing collisions before anything is written."""
    if set(names) != set(KINDS):
        raise ValueError(f"reference export names must cover {KINDS}")
    targets = {kind: Path(directory) / names[kind] for kind in KINDS}
    for target in targets.values():
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite: {target}")
    return targets


def graph_record(nodes, edges):
    """Topology nodes (tips, junctions) and branches with stable IDs.

    Branches are graph.segments in order, so branch i matches build_references'
    branch i. Node IDs N0000... number the vertices of degree other than two by
    column, then any ring anchor; branch IDs are B0000...
    """
    nodes = np.asarray(nodes, dtype=float)
    n = nodes.shape[1]
    canonical = graph.canonical_columns(nodes)
    deg = graph.degree(edges, n)
    vertices = np.flatnonzero(canonical == np.arange(n))
    paths = graph.segments(nodes, edges, canonical)
    node_id = {int(v): f"N{i:04d}" for i, v in enumerate(int(v) for v in vertices if deg[v] != 2)}
    branches = []
    for bi, path in enumerate(paths):
        path = [int(v) for v in path]
        for v in (path[0], path[-1]):
            if v not in node_id:                # anchor of a pure cycle
                node_id[v] = f"N{len(node_id):04d}"
        branches.append({"id": f"B{bi:04d}", "endpoints": [node_id[path[0]], node_id[path[-1]]],
                         "vertex_columns": path, "closed": path[0] == path[-1]})
    out = []
    for v, name in sorted(node_id.items(), key=lambda item: item[1]):
        d = int(deg[v])
        kind = "isolated" if d == 0 else "tip" if d == 1 else "junction" if d >= 3 else "ring_anchor"
        out.append({"id": name, "column": v, "degree": d, "kind": kind})
    return {"nodes": out, "branches": branches, "betti": graph.betti(edges, canonical)}


def _json(value):
    return json.dumps(value, separators=(",", ":"))


def write_graphml(path, nodes, edges, report, points, affine, centered, attributes=None):
    """Write the reference graph as SkelHub-style GraphML and check it; return a summary.

    One node per tip or junction (name, kind, X/Y/Z in mm, voxel_pos, voxels) and one
    edge per branch: branch_id, centerline_world_points / centerline_voxel_points (the
    smooth generating centreline, in mm and voxel xyz), centerline_voxels (the branch's
    foreground-centred 26-connected path from `report`) and num_centerline_voxels.
    Every path runs from the edge's source node to its target node. points is the
    (3, N) voxel coordinate array the references were traced from. No diameters.
    """
    ig = _igraph()
    nodes = np.asarray(nodes, dtype=float)
    points = np.asarray(points, dtype=float)
    affine = np.asarray(affine, dtype=float)
    centered = np.asarray(centered) != 0
    record = graph_record(nodes, edges)
    branches = record["branches"]
    voxel_paths = [np.asarray(b["voxel_path_xyz"], dtype=int) for b in report["branches"]]
    if len(voxel_paths) != len(branches):
        raise ValueError(f"{len(voxel_paths)} centred paths but {len(branches)} branches")
    for branch, entry in zip(branches, report["branches"]):
        if branch["vertex_columns"] != list(entry["source_vertex_columns"]):
            raise ValueError(f"{branch['id']}: graph branch and reference branch differ")

    def world(q):
        return q @ affine[:3, :3].T + affine[:3, 3]

    node_index = {node["id"]: i for i, node in enumerate(record["nodes"])}
    columns = [node["column"] for node in record["nodes"]]
    node_voxel = points[:, columns].T
    node_world = world(node_voxel)
    node_rounded = np.floor(node_voxel + 0.5).astype(int)

    g = ig.Graph(directed=False)
    g.add_vertices(len(record["nodes"]))
    for i, node in enumerate(record["nodes"]):
        vertex = g.vs[i]
        vertex["name"] = node["id"]
        vertex["kind"] = node["kind"]
        vertex["X"], vertex["Y"], vertex["Z"] = (float(v) for v in node_world[i])
        vertex["voxel_pos"] = _json([float(v) for v in node_voxel[i]])
        vertex["voxels"] = _json([[int(v) for v in node_rounded[i]]])

    pairs = [(node_index[b["endpoints"][0]], node_index[b["endpoints"][1]]) for b in branches]
    g.add_edges(pairs)
    union = np.zeros(centered.shape, dtype=bool)
    for edge, branch, voxel_path, (start, end) in zip(g.es, branches, voxel_paths, pairs):
        cols = branch["vertex_columns"]
        source = nodes[:3, cols].T
        keep = np.r_[True, np.linalg.norm(np.diff(source, axis=0), axis=1) > MERGE_TOL]
        sample_voxel = points[:, cols].T[keep]
        if not ((voxel_path[0] == node_rounded[start]).all() and (voxel_path[-1] == node_rounded[end]).all()):
            raise ValueError(f"{branch['id']}: centred path endpoints differ from the rounded nodes")
        steps = np.abs(np.diff(voxel_path, axis=0))
        if len(voxel_path) > 1 and (steps.max() > 1 or (steps.sum(axis=1) == 0).any()):
            raise ValueError(f"{branch['id']}: centred voxel path is not a 26-connected path without repeats")
        if (edge.source, edge.target) == (end, start) and start != end:
            sample_voxel, voxel_path = sample_voxel[::-1], voxel_path[::-1]
        elif (edge.source, edge.target) != (start, end):
            raise ValueError(f"{branch['id']}: GraphML edge endpoints do not match the branch")
        edge["branch_id"] = branch["id"]
        edge["centerline_voxels"] = _json(voxel_path.tolist())
        edge["centerline_voxel_points"] = _json(sample_voxel.tolist())
        edge["centerline_world_points"] = _json(world(sample_voxel).tolist())
        edge["num_centerline_voxels"] = int(len(voxel_path))
        union[tuple(voxel_path.T)] = True
    if not np.array_equal(union, centered):
        raise ValueError("union of the GraphML centred paths differs from the centred reference")
    components = len(g.connected_components())
    cycles = g.ecount() - g.vcount() + components
    if (components, cycles) != (record["betti"]["components"], record["betti"]["cycles"]):
        raise ValueError(f"GraphML topology ({components}, {cycles}) differs from the source graph")

    attributes = dict(attributes or {})
    g["description"] = attributes.pop("description", (
        "Reference centreline graph. Nodes are tips and junctions; each edge is one branch. "
        "centerline_world_points / centerline_voxel_points: smooth generating centreline; "
        "centerline_voxels: foreground-centred 26-connected path matching the centred reference. "
        "Diameters not exported."))
    g["world_units"] = "mm"
    g["voxel_axis_order"] = "xyz"
    g["affine"] = _json(affine.tolist())
    g["volume_shape"] = _json([int(v) for v in centered.shape])
    for key, value in attributes.items():
        g[key] = value
    g["generator"] = "V-System reference.py"
    g.write_graphml(str(path))

    loaded = ig.Graph.Read_GraphML(str(path))
    if (loaded.vcount(), loaded.ecount()) != (g.vcount(), g.ecount()):
        raise ValueError("GraphML read-back has different node or edge counts")
    for name in ("centerline_voxels", "centerline_world_points"):
        if loaded.es[name] != g.es[name]:
            raise ValueError(f"GraphML read-back changed edge attribute {name}")
    return {"nodes": g.vcount(), "edges": g.ecount(), "components": components, "cycles": cycles,
            "tips": sum(1 for node in record["nodes"] if node["kind"] == "tip"),
            "junctions": sum(1 for node in record["nodes"] if node["kind"] == "junction")}


def export_references(directory, names, nodes, edges, mask, points, affine, transform=None, attributes=None):
    """Validate, stage and publish the mask, both references and the GraphML; return metadata.

    names maps each of KINDS to a file name in `directory` (stem_names or FIXTURE_NAMES).
    points is the (3, N) voxel coordinate array of every archive column on the mask's grid.
    Refuses existing destinations. If publishing fails, removes only files created by
    this call. Temporary files are cleaned on success and failure alike.
    """
    # Use the actual NIfTI-1 grid for EDT and diagnostics as well as the files.
    affine = np.asarray(affine, dtype=float).astype(np.float32).astype(float)
    arrays, report = build_references(nodes, edges, mask, points, affine)
    if transform is not None:
        report["source_to_voxel_xyz"] = np.asarray(transform).tolist()
    report["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    directory = Path(directory)
    targets = check_destinations(directory, names)
    volumes = {"mask": arrays["mask"], "26_con": arrays["uncentered"], "26_con_centered": arrays["centered"]}
    created = []
    with tempfile.TemporaryDirectory(prefix=".reference-", dir=directory) as temporary:
        staged = {kind: Path(temporary) / target.name for kind, target in targets.items()}
        for kind, data in volumes.items():
            _save_checked(staged[kind], data, affine)
        digest = {kind: hashlib.sha256(staged[kind].read_bytes()).hexdigest() for kind in volumes}
        graph_attributes = {"centred_reference_sha256": digest["26_con_centered"]}
        graph_attributes.update(attributes or {})
        report["graphml"] = write_graphml(staged["graphml"], nodes, edges, report, points, affine,
                                          volumes["26_con_centered"], graph_attributes)
        digest["graphml"] = hashlib.sha256(staged["graphml"].read_bytes()).hexdigest()
        report["outputs"] = {kind: {"file": targets[kind].name, "sha256": digest[kind]} for kind in KINDS}
        try:
            for kind, target in targets.items():
                with target.open("xb") as stream:
                    created.append(target)
                    with staged[kind].open("rb") as source:
                        shutil.copyfileobj(source, stream)
                if hashlib.sha256(target.read_bytes()).hexdigest() != digest[kind]:
                    raise ValueError(f"Published file checksum mismatch: {target}")
        except BaseException:
            for target in reversed(created):
                target.unlink()
            raise
    return report


def export_nifti(directory, stem, nodes, edges, mask, voxel_size, units):
    """Export on main.py's fitted grid (--fit voxel_size with an explicit --volume); return metadata."""
    points, transform, affine = rendering_grid(nodes, mask.shape, voxel_size, units)
    return export_references(directory, stem_names(stem), nodes, edges, mask, points, affine, transform)
