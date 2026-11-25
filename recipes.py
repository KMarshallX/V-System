"""
Fixture recipes for synthetic vessel networks with known reference centrelines.

The building blocks behind fixtures.py (the vsystem_laplacian_50um_v1 fixture set) and
main.py's export options: explicit straight tubes and Ys, custom walk programs, grammar
trees grown by main.grow_network, buried-stub pruning, controlled thin daughters, a
companion branch routed alongside a nonincident branch, a seeded subset of anastomosis
bridges, graph analysis with stable IDs, geometry and digital-topology QA, and the
candidate selection loop.

Shape origins are labelled per polyline:

    explicit_polyline        straight polylines written here (F01, F02)
    custom_walk_program      a custom turtle program interpreted by V-System's turtle with
                             its persistent random walk and collision avoidance (F03, F04)
    grammar_walk_tree        vSystem.F grammar via main.grow_network with the walk and
                             collision avoidance (F05-F12)
    controlled_thin_daughter thin terminal side branches added here (F08)
    controlled_companion     one side branch routed alongside a nonincident branch (F10)
    anastomosis_bridge       bridges made by anastomosis.anastomose, of which a seeded
                             subset is kept (F11, F12)

Nothing here judges a fixture by how a skeletonization algorithm performs on it.
"""
import hashlib
import json
import math
import os
import random
import time
from types import SimpleNamespace

import numpy as np

import analyseGrammar
import anastomosis
import collisions
import computeVoxel
import describe
import graph
import libGenerator
import main
import spatial
import tortuosity
import utils
import vSystem

V = SimpleNamespace(main=main, graph=graph, computeVoxel=computeVoxel, describe=describe, anastomosis=anastomosis,
                    tortuosity=tortuosity, libGenerator=libGenerator, collisions=collisions,
                    analyseGrammar=analyseGrammar, utils=utils, spatial=spatial, vSystem=vSystem)

# ----------------------------------------------------------------------------------------------
# Fixed conventions (user requirements and output contract from the manifest)
# ----------------------------------------------------------------------------------------------
VOXEL_UM = 50.0
SCALE = 1.0 / VOXEL_UM                      # voxels per micrometre
SPACING_MM = VOXEL_UM / 1000.0
BORDER_VOX = 4                              # empty voxels required on every side
D_FLOOR_UM = 50.0
D_ROOT_MAX_UM = 500.0
NEAR_FLOOR_BAND_UM = (50.0, 75.0)
ROUNDOFF_UM = 1e-9
CHAIN_REACH_VOX = math.sqrt(3.0) / 2.0      # a connect-chain voxel centre lies within this of the centreline
DIGITAL_SAFE_CLEARANCE_UM = math.sqrt(3.0) * VOXEL_UM

# ----------------------------------------------------------------------------------------------
# Engineering defaults of this adapter (not biological measurements)
# ----------------------------------------------------------------------------------------------
DENSE_STEP_UM = 5.0         # sample spacing for continuous-tube clearance; sampled minima overstate
                            # the continuous minimum by at most this much
EXPLICIT_STEP_UM = 25.0     # centreline sample spacing of explicit straight shapes (half a voxel)
INSERT_STEP_UM = 12.5       # sample spacing of controlled insertions (a quarter voxel)
CLEARANCE_SEARCH_UM = 400.0 # nonincident clearance is exhaustive below this surface gap
OPENING_MIN_UM = 150.0      # loop openings: at least three voxels of background
TORTUOSITY_SAGITTA_VOX = 1.0
STUB_EXPOSURE_MIN_UM = 100.0  # a terminal tip must lie >= 2 voxels outside every other tube
RUN_SMOOTH_UM = 100.0     # smoothing of the target centreline the F10 companion run follows
MIN_SIDE_ANGLE_DEG = 75.0   # side branches leave at least this far from the host tangent, so that
                            # collisions.KIN_REACH (1.7) covers their junction overlap
ORIGINS = ("grammar_walk_tree", "explicit_polyline", "anastomosis_bridge",
           "controlled_thin_daughter", "controlled_companion", "custom_walk_program")
ORIGIN = {name: code for code, name in enumerate(ORIGINS)}
RNG_TAGS = {"thin_daughters": 101, "companion": 102, "bridge_selection": 103}

COMMON = {
    "subdivisions": 4, "epsilon": 6.0, "randmarg": 0.15, "k": 3.0, "roll_angle": 70.0,
    "sigma": 5.0, "collision_margin_um": 100.0, "collision_attempts": 20,
    "aneurysm_prob": 0.0, "stenosis_prob": 0.0, "d_min_um": D_FLOOR_UM,
}

# Per-fixture construction plans. Values not fixed by the manifest are this adapter's
# effective choices; they are recorded in metadata and in the resolved manifest.
PLANS = {
    "F01": {"builder": "straight_tube", "length_um": 3000.0, "direction": (1.0, 0.62, 0.38)},
    "F02": {"builder": "straight_y", "root_length_um": 2000.0, "daughter1_diameter_um": 215.0,
            "daughter_lengths_um": (1750.0, 1600.0), "root_direction": (0.55, 0.75, 0.37),
            "plane_hint": (0.3, -0.2, 0.9)},
    "F03": {"builder": "walk_program", "shape": "single", "lengths_um": (5000.0,), "moves": (10,)},
    "F04": {"builder": "walk_program", "shape": "y", "lengths_um": (2500.0, 2000.0, 1800.0),
            "moves": (5, 4, 4), "daughter1_diameter_um": 215.0},
    "F05": {"builder": "tree", "niter": 3, "sigma": 3.0},
    "F06": {"builder": "tree", "niter": 5},
    "F07": {"builder": "tree", "niter": 5},
    "F08": {"builder": "tree_thin", "niter": 4, "thin_count": 5, "thin_diameter_um": (50.0, 70.0),
            "thin_protrusion_um": (150.0, 300.0), "host_min_diameter_um": 300.0, "max_attempts": 400},
    "F09": {"builder": "tree", "niter": 7, "sigma": 3.0},
    "F10": {"builder": "tree_companion", "niter": 6, "sigma": 3.0, "run_length_um": 500.0,
            "target_clearance_um": 125.0, "approach_band_um": (100.0, 150.0),
            "companion_diameter_um": (100.0, 150.0), "target_min_diameter_um": 100.0,
            "leave_angle_deg": 80.0, "host_chord_um": (500.0, 2500.0), "max_targets": 30,
            "offsets_per_target": 6, "hosts_per_offset": 6},
    "F11": {"builder": "tree_bridges", "niter": 5, "sigma": 3.0, "bridges": (1, 1), "anastomosis_fraction": 0.5,
            "anastomosis_radius": 25.0, "anastomosis_min_separation": 3},
    "F12": {"builder": "tree_bridges", "niter": 7, "sigma": 3.0, "bridges": (2, 3),
            "anastomosis_fraction": 0.5, "anastomosis_radius": 25.0, "anastomosis_min_separation": 3},
}

# Every departure from a value the manifest proposed, with its reason. Filled during tuning.
ADJUSTMENTS = [
    {"item": "manifest-proposed numeric values", "manifest_value": "subdivisions 4, epsilon 6, randmarg 0.15, "
     "Murray exponent 3, roll 70 deg, collision margin 100 um, 20 attempts, persistences as listed, desired "
     "branch-count ranges as listed, candidate budget 32", "effective_value": "unchanged",
     "reason": "the proposed values produced valid fixtures; no tuning of them was needed"},
    {"item": "sigma (V-System daughter-diameter spread; not in the manifest)", "manifest_value": None,
     "effective_value": "3 for F05, F09, F10, F11, F12; V-System default 5 elsewhere",
     "reason": "a smaller sigma makes bifurcations more asymmetric: F05's purpose is asymmetric daughters, and "
               "F09-F12 are asked to seek near-floor branches without excessive branching"},
    {"item": "drawn generations niter (not in the manifest)", "manifest_value": None,
     "effective_value": {fid: plan["niter"] for fid, plan in PLANS.items() if "niter" in plan},
     "reason": "chosen so that the realised branch counts fall in the desired ranges after collision "
               "terminations and stub pruning"},
    {"item": "buried collision stubs", "manifest_value": "not addressed",
     "effective_value": "pruned: terminal branches whose tip lies < 100 um outside every other tube are removed "
                        "after growth, before insertions, bridges and rendering; every final tip is checked",
     "reason": "V-System ends a colliding daughter a step or two after its junction, leaving a tip inside or "
               "against another vessel in almost every candidate; such a branch cannot be recovered from the "
               "mask and at one junction it also added a spurious tunnel to the reference. Decision taken by "
               "the user on 2026-10-01 when asked."},
    {"item": "continuous clearance criterion", "manifest_value": "no collisions outside documented junction "
     "neighbourhoods", "effective_value": "junction neighbourhoods = V-System's rule at the 100 um generation "
     "margin; failure = any overlap (< 0 um) outside them, or any overlap between nonincident branches",
     "reason": "growth checks centreline points only, so densified (5 um) pairs sit 3-8 um below the 100 um "
               "target in places without touching; exemption windows sized at margin 0 flag sibling overlaps "
               "just beyond the smaller windows. Both figures are reported."},
    {"item": "selection preferences", "manifest_value": "seek near-floor branches in F09-F12; F07 same "
     "topology specification as F06", "effective_value": "within the budget, prefer near-floor terminals "
     "(F09-F12) and same realised topology (F06/F07); fall back to the first passing candidate",
     "reason": "turns the manifest's 'seek'/'retain' wording into a predefined, reproducible selection rule"},
    {"item": "F10 companion run", "manifest_value": "controlled close approach", "effective_value":
     "offset by r_target + r_companion + 125 um from a Gaussian-smoothed (sigma 100 um) copy of the target",
     "reason": "the walk turns at every step, so an offset of the raw centreline folds; the true clearance is "
               "measured on the unsmoothed geometry"},
    {"item": "F08 thin daughter length", "manifest_value": "3-6 voxel lengths",
     "effective_value": "3-6 voxels (150-300 um) beyond the host surface; total length = host radius + that",
     "reason": "a daughter shorter than the host radius would not leave the host tube at all"},
]

TORTUOSITY_TARGETS = {
    # fixture: (min length-weighted arc/chord, min fraction of branch length with sagitta >= 1 voxel)
    "F03": {"arc_chord_min": 1.10, "sagitta_min_vox": 3.0},
    "F04": {"arc_chord_min": 1.03, "per_branch_sagitta_min_vox": 1.0},
    "tree": {"arc_chord_min": 1.03, "bent_length_fraction_min": 0.5},
}



# ==============================================================================================
# Small geometry helpers
# ==============================================================================================
def unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def perpendicular_basis(tangent):
    """Two unit vectors completing `tangent` to an orthonormal frame."""
    tangent = unit(tangent)
    trial = np.array([1.0, 0.0, 0.0]) if abs(tangent[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = unit(trial - (trial @ tangent) * tangent)
    return e1, np.cross(tangent, e1)


def murray_daughter(d0, d1, k=3.0):
    """The second daughter diameter that satisfies Murray's law d0^k = d1^k + d2^k."""
    return float((d0 ** k - d1 ** k) ** (1.0 / k))


def zamir_angles(d1, d2):
    """
    Bifurcation angles (degrees) of daughters d1 and d2 from the parent direction,
    by the minimum-lumen-volume expressions of libGenerator.calBifurcation.
    """
    alpha = d2 / d1
    xtmp = (1 + alpha ** 3) ** (4.0 / 3) + 1 - alpha ** 4
    xtmpb = 2 * ((1 + alpha ** 3) ** (2.0 / 3))
    a1 = math.acos(xtmp / xtmpb)
    xtmp = (1 + alpha ** 3) ** (4.0 / 3) + (alpha ** 4) - 1
    xtmpb = 2 * alpha * alpha * ((1 + alpha ** 3) ** (2.0 / 3))
    a2 = math.acos(xtmp / xtmpb)
    return math.degrees(a1), math.degrees(a2)


def straight_polyline(p0, p1, diameter, max_step):
    """(m, 4) rows from p0 to p1 at spacing <= max_step; the end rows are exact copies."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    n = max(1, int(math.ceil(np.linalg.norm(p1 - p0) / max_step)))
    t = np.arange(n + 1)[:, None] / n
    rows = np.empty((n + 1, 4))
    rows[:, :3] = p0 + t * (p1 - p0)
    rows[0, :3] = p0
    rows[-1, :3] = p1
    rows[:, 3] = diameter
    return rows


def assemble(polylines):
    """(4, N) archive columns from a list of (m, 4) rows, NaN-separated."""
    pieces = []
    for i, rows in enumerate(polylines):
        if i:
            pieces.append(np.full((4, 1), np.nan))
        pieces.append(np.asarray(rows, dtype=float).T)
    return np.concatenate(pieces, axis=1)


def append_polyline(nodes, labels, rows, label_values):
    """Appends rows after a separator; label_values maps label array name -> value."""
    nodes = np.concatenate([nodes, np.full((4, 1), np.nan), np.asarray(rows, dtype=float).T], axis=1)
    for name, value in label_values.items():
        labels[name] = np.concatenate([labels[name], [-1], np.full(len(rows), value)]).astype(np.int8)
    return nodes, labels


def densify(rows, max_step):
    """
    Inserts linearly interpolated rows so that consecutive points are at most `max_step`
    apart. Original rows are kept bitwise. Returns (dense rows, arc positions).
    """
    rows = np.asarray(rows, dtype=float)
    if len(rows) < 2:
        return rows.copy(), np.zeros(len(rows))
    seg = rows[1:] - rows[:-1]
    length = np.linalg.norm(seg[:, :3], axis=1)
    n = np.maximum(1, np.ceil(length / max_step)).astype(np.int64)
    idx = np.repeat(np.arange(len(seg)), n)
    k = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    t = (k / n[idx])[:, None]
    dense = rows[idx] + t * seg[idx]
    start = k == 0
    dense[start] = rows[idx[start]]
    dense = np.vstack([dense, rows[-1:]])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense[:, :3], axis=0), axis=1))])
    return dense, arc


def polyline_rows(nodes, columns):
    return nodes[:, columns].T


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_ready(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(json_ready(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")


def check(name, passed, measured, requirement, kind="hard", note=None):
    """One validation record. passed is True, False or None (not measured)."""
    record = {"check": name, "kind": kind, "passed": passed, "requirement": requirement,
              "measured": measured}
    if note:
        record["note"] = note
    return record


# ==============================================================================================
# Clearance index used while placing controlled insertions
# ==============================================================================================
class NetworkClearance:
    """
    Densely sampled polylines of a network in a k-d tree, for checking a proposed new
    polyline that leaves an existing column H of a host polyline.

    A stored sample q and a proposed sample p collide when |p - q| - r_p - r_q < margin,
    unless q lies on the host polyline and arc(q, H) + arc(p, H) < KIN_REACH (r_p + r_q +
    margin): the junction rule of V-System's collisions.py and describe.py.
    """

    def __init__(self, nodes):
        from scipy.spatial import cKDTree
        points, radii, pid, arc = [], [], [], []
        self.column_arc = {}
        self.column_polyline = {}
        for k, columns in enumerate(V.graph.polylines(nodes)):
            rows = polyline_rows(nodes, columns)
            dense, darc = densify(rows, DENSE_STEP_UM)
            col_arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(rows[:, :3], axis=0), axis=1))])
            for c, a in zip(columns.tolist(), col_arc.tolist()):
                self.column_arc[c] = a
                self.column_polyline[c] = k
            points.append(dense[:, :3])
            radii.append(dense[:, 3] / 2.0)
            pid.append(np.full(len(dense), k))
            arc.append(darc)
        self.points = np.vstack(points)
        self.radii = np.concatenate(radii)
        self.pid = np.concatenate(pid)
        self.arc = np.concatenate(arc)
        self.rmax = float(self.radii.max())
        self.kd = cKDTree(self.points)

    def check(self, rows, host_column, margin, own_curve=True):
        """
        Returns (clear, minimum clearance) for the proposed rows (m, 4) whose first row is
        the host column. The minimum is over pairs not excused by the junction rule.
        """
        dense, arcs = densify(rows, DENSE_STEP_UM)
        q = dense[:, :3]
        rq = dense[:, 3] / 2.0
        host_pid = self.column_polyline[host_column]
        host_arc = self.column_arc[host_column]
        lists = self.kd.query_ball_point(q, rq + self.rmax + margin + 1e-9)
        sizes = np.array([len(item) for item in lists])
        best = math.inf
        if sizes.sum():
            qi = np.repeat(np.arange(len(q)), sizes)
            si = np.fromiter((s for item in lists for s in item), dtype=np.int64, count=int(sizes.sum()))
            gap = np.linalg.norm(self.points[si] - q[qi], axis=1) - self.radii[si] - rq[qi]
            excused = (self.pid[si] == host_pid) & (
                np.abs(self.arc[si] - host_arc) + arcs[qi] < V.collisions.KIN_REACH * (self.radii[si] + rq[qi] + margin))
            gap = gap[~excused]
            if gap.size:
                best = float(gap.min())
        if own_curve and len(q) > 2:
            # rule (a) of describe.clearance on the proposed polyline itself
            dist = np.linalg.norm(q[:, None, :] - q[None, :, :], axis=2)
            sep = np.abs(arcs[:, None] - arcs[None, :])
            thr = rq[:, None] + rq[None, :]
            far = sep > 2.0 * (thr + margin)
            if far.any():
                best = min(best, float((dist - thr)[far].min()))
        return best >= margin, best


# ==============================================================================================
# Builders
# ==============================================================================================
def properties_for(plan):
    if "properties" in plan:                # main.py passes its own drawn properties
        return dict(plan["properties"])
    return {"k": COMMON["k"], "epsilon": plan.get("epsilon", COMMON["epsilon"]),
            "randmarg": plan.get("randmarg", COMMON["randmarg"]), "sigma": plan.get("sigma", COMMON["sigma"]),
            "stochparams": True, "roll_angle": COMMON["roll_angle"],
            "aneurysm_prob": COMMON["aneurysm_prob"], "stenosis_prob": COMMON["stenosis_prob"]}


def built(nodes, origin, construction, params, tree=None, program=None, events=None, extras=None):
    finite = np.isfinite(nodes[0])
    if tree is None:
        tree = np.where(finite, 0, -1).astype(np.int8)
    if np.isscalar(origin):
        origin = np.where(finite, origin, -1).astype(np.int8)
    return {"nodes": nodes, "tree": np.asarray(tree, dtype=np.int8), "origin": np.asarray(origin, dtype=np.int8),
            "construction": construction, "params": params, "program": program,
            "events": dict(events or {}), "extras": dict(extras or {})}


def build_straight_tube(spec, plan, seed):
    d0 = spec["root_diameter_um"]
    u = unit(plan["direction"])
    rows = straight_polyline(np.zeros(3), plan["length_um"] * u, d0, EXPLICIT_STEP_UM)
    params = {"length_um": plan["length_um"], "direction_unit": u.tolist(), "diameter_um": d0,
              "sample_step_um_max": EXPLICIT_STEP_UM}
    return built(assemble([rows]), ORIGIN["explicit_polyline"],
                 "explicit straight polyline written by the adapter; rendered by V-System computeVoxel", params)


def build_straight_y(spec, plan, seed):
    common = plan.get("common", COMMON)
    d0 = spec["root_diameter_um"]
    d1 = plan["daughter1_diameter_um"]
    d2 = murray_daughter(d0, d1, common["k"])
    th1, th2 = zamir_angles(d1, d2)
    u = unit(plan["root_direction"])
    normal = unit(np.cross(u, plan["plane_hint"]))
    dir1 = V.utils.rotate_about(u, normal, th1)
    dir2 = V.utils.rotate_about(u, normal, -th2)
    junction = plan["root_length_um"] * u
    l1, l2 = plan["daughter_lengths_um"]
    root = straight_polyline(np.zeros(3), junction, d0, EXPLICIT_STEP_UM)
    a = straight_polyline(junction, junction + l1 * dir1, d1, EXPLICIT_STEP_UM)
    b = straight_polyline(junction, junction + l2 * dir2, d2, EXPLICIT_STEP_UM)
    params = {"root_length_um": plan["root_length_um"], "daughter_lengths_um": [l1, l2],
              "diameters_um": [d0, d1, d2], "murray_exponent": common["k"],
              "zamir_angles_deg": [th1, th2], "root_direction_unit": u.tolist(),
              "bifurcation_plane_normal": normal.tolist(), "sample_step_um_max": EXPLICIT_STEP_UM}
    return built(assemble([root, a, b]), ORIGIN["explicit_polyline"],
                 "explicit piecewise-straight Y written by the adapter (Murray diameters, Zamir angles); "
                 "rendered by V-System computeVoxel", params)


def _num(value):
    return repr(float(value))


def _stem(diameter, length, moves):
    step = length / moves
    return "{" + "".join("f(" + _num(step) + "," + _num(diameter) + ")" for _ in range(moves)) + "}"


START_DIRECTION = unit((0.3, 0.9, 0.3))
START_PERPENDICULAR = perpendicular_basis(START_DIRECTION)[0]


def interpret_walk_program(program, d0, seed, persistence, plan):
    """Runs a custom turtle program through V-System's walk and collision avoidance."""
    common = plan.get("common", COMMON)
    V.libGenerator.setProperties(properties_for(plan))
    margin = common["collision_margin_um"]
    walk = V.analyseGrammar.WalkSettings(persistence, np.random.default_rng([int(seed), V.main.RNG_STREAMS["walk"]]))
    avoider = V.collisions.CollisionAvoider(
        V.spatial.make_index("grid", cell_size=d0 * V.libGenerator.aneurysm_factor + margin),
        margin=margin, attempts=common["collision_attempts"])
    events = {key: 0 for key in V.main.EVENT_KEYS}
    rows = V.analyseGrammar.branching_turtle_to_coords(
        program, d0, position=(0.0, 0.0, 0.0), direction=START_DIRECTION, perpendicular=START_PERPENDICULAR,
        walk=walk, avoid=avoider, events=events, subdivisions=common["subdivisions"])
    nodes = V.utils.interpolate_segments(rows, subdivisions=common["subdivisions"])
    return nodes, events


def build_walk_program(spec, plan, seed):
    common = plan.get("common", COMMON)
    d0 = spec["root_diameter_um"]
    persistence = spec["persistence"]
    lengths, moves = plan["lengths_um"], plan["moves"]
    params = {"persistence_diameters": persistence, "lengths_um": list(lengths), "moves": list(moves),
              "start_direction": START_DIRECTION.tolist(), "start_perpendicular": START_PERPENDICULAR.tolist(),
              "subdivisions": common["subdivisions"], "collision_margin_um": common["collision_margin_um"],
              "collision_attempts": common["collision_attempts"]}
    if plan["shape"] == "single":
        program = _stem(d0, lengths[0], moves[0])
        params["diameters_um"] = [d0]
    else:
        d1 = plan["daughter1_diameter_um"]
        d2 = murray_daughter(d0, d1, common["k"])
        th1, th2 = zamir_angles(d1, d2)
        roll = _num(common["roll_angle"])
        program = (_stem(d0, lengths[0], moves[0])
                   + "[+(" + _num(th1) + ")/(" + roll + ")" + _stem(d1, lengths[1], moves[1]) + "]"
                   + "[-(" + _num(th2) + ")/(" + roll + ")" + _stem(d2, lengths[2], moves[2]) + "]")
        params.update({"diameters_um": [d0, d1, d2], "zamir_angles_deg": [th1, th2],
                       "roll_angle_deg": common["roll_angle"]})
    nodes, events = interpret_walk_program(program, d0, seed, persistence, plan)
    return built(nodes, ORIGIN["custom_walk_program"],
                 "custom turtle program interpreted by V-System analyseGrammar with the persistent random walk "
                 "(tortuosity.py) and collision avoidance (collisions.py)", params, program=program, events=events)


def grow_tree(spec, plan, seed):
    """main.grow_network with the walk and avoidance; the grammar draws from the seeded global state."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    grown = V.main.grow_network(
        plan["niter"], spec["root_diameter_um"], properties_for(plan), (16, 16, 16), fit="voxel_size",
        clip_axes=(), voxel_size=VOXEL_UM, subdivisions=COMMON["subdivisions"], d_min=COMMON["d_min_um"],
        grow_in_volume=False, tortuosity="walk", persistence=spec["persistence"], avoid_collisions=True,
        collision_margin=COMMON["collision_margin_um"], collision_attempts=COMMON["collision_attempts"],
        collision_index="grid", anastomose=False, seed=seed)
    labels = {"tree": grown["tree"],
              "origin": np.where(np.isfinite(grown["nodes"][0]), ORIGIN["grammar_walk_tree"], -1).astype(np.int8)}
    nodes, labels, pruned = prune_buried_stubs(grown["nodes"], labels)
    grown["nodes"], grown["tree"], grown["origin"] = nodes, labels["tree"], labels["origin"]
    grown["events"] = dict(grown["events"])
    grown["events"]["adapter_buried_stubs_pruned"] = len(pruned)
    grown["pruned_stubs"] = pruned
    params = {"niter": plan["niter"], "d0_um": spec["root_diameter_um"], "properties": properties_for(plan),
              "d_min_um": COMMON["d_min_um"], "tortuosity": "walk", "persistence_diameters": spec["persistence"],
              "subdivisions": COMMON["subdivisions"], "avoid_collisions": True,
              "collision_margin_um": COMMON["collision_margin_um"],
              "collision_attempts": COMMON["collision_attempts"], "collision_index": "grid",
              "grow_in_volume": False, "direction": [0.0, 1.0, 0.0], "perpendicular": [0.0, 0.0, 1.0],
              "rng_streams": {name: [seed, tag] for name, tag in V.main.RNG_STREAMS.items()},
              "grammar_rng": "random.seed(seed); numpy.random.seed(seed) before vSystem.F",
              "stub_pruning": {"exposure_min_um": STUB_EXPOSURE_MIN_UM,
                               "rule": "terminal branches whose tip lies < 100 um outside every other tube are "
                                       "removed, repeatedly, before any insertion, bridge or rendering"}}
    return grown, params


def build_tree(spec, plan, seed):
    grown, params = grow_tree(spec, plan, seed)
    return built(grown["nodes"], grown["origin"],
                 "vSystem.F grammar via main.grow_network (walk tortuosity, collision avoidance, d_min); "
                 "buried collision stubs pruned", params, tree=grown["tree"], program=grown["program"],
                 events=grown["events"], extras={"pruned_stubs": grown["pruned_stubs"]})


def _host_tangent(nodes, column):
    return unit(nodes[:3, column + 1] - nodes[:3, column - 1])


def host_sites(nodes, graph_info, min_diameter):
    """
    Candidate attachment points for a side branch: interior vertices of open graph branches
    (one stem between two junctions or tips) whose calibre is at least `min_diameter`, with
    their arc position along the branch. Both archive neighbours of a site lie on its polyline,
    so the host tangent is a central difference.
    """
    polyline_of = V.graph.polyline_of_column(nodes)
    n = nodes.shape[1]
    sites = []
    for bi, b in enumerate(graph_info["branches"]):
        if b["closed"]:
            continue
        d = np.asarray(b["diameter_um"])
        if d.min() < min_diameter:
            continue
        cols = b["vertex_columns"]
        xyz = np.asarray(b["xyz_um"])
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
        interior = []
        for k in range(1, len(cols) - 1):
            c = cols[k]
            if not 0 < c < n - 1 or not (np.isfinite(nodes[0, c - 1]) and np.isfinite(nodes[0, c + 1])):
                continue
            if polyline_of[c - 1] != polyline_of[c] or polyline_of[c + 1] != polyline_of[c]:
                continue
            interior.append((k, c))
        if interior:
            sites.append({"branch": bi, "arc": arc, "length": float(arc[-1]), "diameter": float(d.min()),
                          "interior": interior})
    return sites


def tip_exposures(graph_info, root_column=0):
    """
    For every terminal branch (one tip end that is not the root), the distance from the tip
    point to the nearest surface of any other branch's tube, on densified centrelines.
    Negative when the tip lies inside another vessel.
    """
    from scipy.spatial import cKDTree
    branches = graph_info["branches"]
    tips = {n["column"] for n in graph_info["nodes"] if n["kind"] == "tip"}
    points, radii, owner = dense_branch_samples(branches)
    kd = cKDTree(points)
    rmax = float(radii.max())
    out = []
    for bi, b in enumerate(branches):
        if len(branches) == 1:
            break
        for end, col in enumerate(b["endpoint_columns"]):
            if col not in tips or col == root_column or b["endpoint_columns"][1 - end] in tips:
                continue
            tip = np.asarray(b["xyz_um"][-1 if end else 0])
            near = np.asarray(kd.query_ball_point(tip, rmax + 4.0 * STUB_EXPOSURE_MIN_UM), dtype=np.int64)
            near = near[owner[near] != bi] if near.size else near
            exposure = (float(np.min(np.linalg.norm(points[near] - tip, axis=1) - radii[near]))
                        if near.size else math.inf)
            out.append({"branch": bi, "id": b["id"], "tip_column": col, "exposure_um": exposure,
                        "length_um": b["arc_length_um"], "diameter_um": float(np.min(b["diameter_um"])),
                        "junction_column": b["endpoint_columns"][1 - end]})
    return out


def tidy_archive(nodes, labels):
    """Drops repeated, leading and trailing separators and redundant single-point polylines."""
    finite = np.isfinite(nodes[0])
    keep = finite.copy()
    for i in np.flatnonzero(~finite):
        keep[i] = i > 0 and finite[i - 1] and i < len(finite) - 1 and finite[i + 1:].any()
    nodes = nodes[:, keep]
    labels = {k: v[keep] for k, v in labels.items()}
    canonical = V.graph.canonical_columns(nodes)
    counts = np.bincount(canonical[canonical >= 0], minlength=nodes.shape[1])
    drop = np.zeros(nodes.shape[1], dtype=bool)
    for columns in V.graph.polylines(nodes):
        if len(columns) == 1 and counts[canonical[columns[0]]] > 1:
            drop[columns[0]] = True
    if drop.any():
        nodes = nodes[:, ~drop]
        labels = {k: v[~drop] for k, v in labels.items()}
        return tidy_archive(nodes, labels)
    return nodes, labels


def prune_buried_stubs(nodes, labels):
    """
    Removes terminal branches whose tip lies less than STUB_EXPOSURE_MIN_UM outside every
    other tube (V-System collision terminations a step or two after a junction), repeating
    until none is left. The junction vertex is kept; it becomes a degree-two vertex.
    Returns (nodes, labels, pruned records).
    """
    pruned = []
    for round_index in range(100):
        graph_info = analyse_graph(nodes, labels["origin"])
        buried = [e for e in tip_exposures(graph_info) if e["exposure_um"] < STUB_EXPOSURE_MIN_UM]
        if not buried:
            return nodes, labels, pruned
        canonical = graph_info["canonical"]
        delete = np.zeros(nodes.shape[1], dtype=bool)
        for item in buried:
            b = graph_info["branches"][item["branch"]]
            for c in b["columns"]:
                if canonical[c] != item["junction_column"]:
                    delete[c] = True
            pruned.append({k: item[k] for k in ("exposure_um", "length_um", "diameter_um")}
                          | {"round": round_index, "tip_xyz_um": nodes[:3, item["tip_column"]].tolist(),
                             "origin": b["origin"]})
        nodes = nodes[:, ~delete]
        labels = {k: v[~delete] for k, v in labels.items()}
        nodes, labels = tidy_archive(nodes, labels)
    raise RuntimeError("stub pruning did not converge")


def build_tree_thin(spec, plan, seed):
    """F08: a grown tree plus controlled thin, short terminal side daughters on thick stems."""
    grown, params = grow_tree(spec, plan, seed)
    return add_thin_daughters(grown, params, plan, seed)


def add_thin_daughters(grown, params, plan, seed, common=COMMON):
    """
    Adds plan["thin_count"] controlled thin, short terminal side daughters to a grown tree
    (a dict with nodes, tree, origin, program, events and pruned_stubs). Returns
    (built candidate, None) or (None, failure).
    """
    nodes = grown["nodes"]
    labels = {"tree": grown["tree"], "origin": grown["origin"]}
    rng = np.random.default_rng([int(seed), RNG_TAGS["thin_daughters"]])
    margin = common["collision_margin_um"]
    graph_info = analyse_graph(nodes, labels["origin"])
    hosts = host_sites(nodes, graph_info, plan["host_min_diameter_um"])
    inserted, attempts_log = [], []
    if not hosts:
        return None, {"reason": "no host stem reaches the host diameter"}
    weights = np.array([h["length"] for h in hosts])
    weights = weights / weights.sum()
    net = NetworkClearance(nodes)
    attempts = 0
    while len(inserted) < plan["thin_count"] and attempts < plan["max_attempts"]:
        attempts += 1
        site = hosts[int(rng.choice(len(hosts), p=weights))]
        d_host = site["diameter"]
        d_thin = float(rng.uniform(*plan["thin_diameter_um"]))
        protrusion = float(rng.uniform(*plan["thin_protrusion_um"]))
        phi = float(rng.uniform(0.0, 2.0 * math.pi))
        a_target = float(rng.uniform(0.0, 1.0))
        r_host = d_host / 2.0
        # stay out of the junction overlap zones at both ends of the host branch
        reach = V.collisions.KIN_REACH * (r_host + d_thin / 2.0 + margin) + r_host
        lo, hi = reach, site["length"] - reach
        if hi <= lo:
            attempts_log.append({"attempt": attempts, "result": "host too short"})
            continue
        a = lo + a_target * (hi - lo)
        k, col = min(site["interior"], key=lambda item: abs(site["arc"][item[0]] - a))
        a_k = float(site["arc"][k])
        if not lo <= a_k <= hi:
            attempts_log.append({"attempt": attempts, "result": "no host vertex inside the allowed span"})
            continue
        if any(item["host_branch"] == site["branch"]
               and abs(item["host_arc_um"] - a_k) < max(3.0 * d_host, 2.0 * reach) for item in inserted):
            attempts_log.append({"attempt": attempts, "result": "too close to an earlier daughter on this host"})
            continue
        tangent = _host_tangent(nodes, col)
        continuing = murray_daughter(d_host, d_thin, common["k"])
        _, theta = zamir_angles(continuing, d_thin)
        theta = max(theta, MIN_SIDE_ANGLE_DEG)
        e1, e2 = perpendicular_basis(tangent)
        axis = math.cos(phi) * e1 + math.sin(phi) * e2
        direction = V.utils.rotate_about(tangent, axis, theta)
        start = nodes[:3, col].copy()
        rows = straight_polyline(start, start + (r_host + protrusion) * direction, d_thin, INSERT_STEP_UM)
        rows[0, :3] = nodes[:3, col]
        clear, gap = net.check(rows, col, margin, own_curve=False)
        if not clear:
            attempts_log.append({"attempt": attempts, "result": "collision", "min_clearance_um": gap})
            continue
        nodes, labels = append_polyline(nodes, labels, rows, {"tree": 0, "origin": ORIGIN["controlled_thin_daughter"]})
        net = NetworkClearance(nodes)
        inserted.append({"host_branch": site["branch"], "host_column": int(col), "host_diameter_um": d_host,
                         "host_arc_um": a_k, "host_branch_length_um": site["length"], "diameter_um": d_thin,
                         "protrusion_um": protrusion, "total_length_um": r_host + protrusion,
                         "angle_from_host_tangent_deg": theta, "roll_rad": phi,
                         "min_clearance_um": gap, "attempt": attempts})
        attempts_log.append({"attempt": attempts, "result": "placed"})
    params.update({"thin_daughters": {k: plan[k] for k in ("thin_count", "thin_diameter_um", "thin_protrusion_um",
                                                             "host_min_diameter_um", "max_attempts")},
                   "rng_thin_daughters": [seed, RNG_TAGS["thin_daughters"]]})
    extras = {"pruned_stubs": grown["pruned_stubs"], "thin_daughters": inserted, "thin_attempts": attempts,
              "thin_failed_attempts": [x for x in attempts_log if x["result"] != "placed"]}
    if len(inserted) < plan["thin_count"]:
        return None, {"reason": f"placed {len(inserted)} of {plan['thin_count']} thin daughters in "
                                f"{attempts} attempts", "extras": extras}
    out = built(nodes, labels["origin"], "vSystem.F grammar via main.grow_network, plus controlled thin terminal "
                "side daughters added by the adapter (straight, constant 50-70 um calibre, Zamir side angle)",
                params, tree=labels["tree"], program=grown["program"], events=grown["events"], extras=extras)
    return out, None


def _resample(xyz, arc, targets):
    return np.column_stack([np.interp(targets, arc, xyz[:, k]) for k in range(3)])


def build_tree_companion(spec, plan, seed):
    """F10: a grown tree plus one side branch routed alongside a nonincident branch."""
    grown, params = grow_tree(spec, plan, seed)
    return add_companion(grown, params, plan, seed, spec["persistence"])


def add_companion(grown, params, plan, seed, persistence, common=COMMON):
    """
    Adds one companion side branch routed alongside a nonincident branch of a grown tree.
    Returns (built candidate, None) or (None, failure).
    """
    nodes = grown["nodes"]
    labels = {"tree": grown["tree"], "origin": grown["origin"]}
    rng = np.random.default_rng([int(seed), RNG_TAGS["companion"]])
    margin = common["collision_margin_um"]
    graph_info = analyse_graph(nodes, labels["origin"])
    branches = graph_info["branches"]
    net = NetworkClearance(nodes)
    sites = host_sites(nodes, graph_info, plan["target_min_diameter_um"])
    # columns of each graph branch, for nonincidence of hosts
    endpoints = [set(b["endpoint_columns"]) for b in branches]
    column_branch = {}
    for bi, b in enumerate(branches):
        for c in b["columns"]:
            column_branch.setdefault(c, set()).add(bi)
    run_length = plan["run_length_um"]
    targets = []
    for bi, b in enumerate(branches):
        if b["closed"]:
            continue
        xyz = np.asarray(b["xyz_um"])
        diam = np.asarray(b["diameter_um"])
        buffer = max(3.0 * float(diam.max()), 300.0)
        if diam.min() >= plan["target_min_diameter_um"] and b["arc_length_um"] >= run_length + 2 * buffer:
            targets.append((bi, buffer))
    order = rng.permutation(len(targets))
    tried = []
    for ti in order[:plan["max_targets"]]:
        bi, buffer = targets[int(ti)]
        b = branches[bi]
        xyz = np.asarray(b["xyz_um"])
        diam = np.asarray(b["diameter_um"])
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
        for _ in range(plan["offsets_per_target"]):
            a0 = float(rng.uniform(buffer, arc[-1] - buffer - run_length))
            dc = float(rng.uniform(*plan["companion_diameter_um"]))
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            s = np.arange(0.0, run_length + 1e-9, INSERT_STEP_UM)
            # the walk turns a little at every step, so its centreline is jagged at the sample
            # scale; the run follows a Gaussian-smoothed copy of the target (sigma RUN_SMOOTH_UM)
            # so that the offset curve cannot fold, and the true clearance is measured afterwards
            pad = np.arange(-4.0 * RUN_SMOOTH_UM, run_length + 4.0 * RUN_SMOOTH_UM + 1e-9, INSERT_STEP_UM)
            wide = _resample(xyz, arc, np.clip(a0 + pad, 0.0, arc[-1]))
            from scipy.ndimage import gaussian_filter1d
            wide = gaussian_filter1d(wide, RUN_SMOOTH_UM / INSERT_STEP_UM, axis=0, mode="nearest")
            first = int(round(4.0 * RUN_SMOOTH_UM / INSERT_STEP_UM))
            centre = wide[first:first + len(s)]
            r_target = float(np.interp(a0 + s, arc, diam).max()) / 2.0
            tangent = np.gradient(centre, axis=0)
            tangent /= np.linalg.norm(tangent, axis=1)[:, None]
            e1, e2 = perpendicular_basis(tangent[0])
            normal = [math.cos(angle) * e1 + math.sin(angle) * e2]
            for t in tangent[1:]:
                n = normal[-1] - (normal[-1] @ t) * t
                normal.append(n / np.linalg.norm(n))
            normal = np.asarray(normal)
            offset = r_target + dc / 2.0 + plan["target_clearance_um"]
            run = centre + offset * normal
            if np.any(np.einsum("ij,ij->i", np.diff(run, axis=0), tangent[:-1]) <= 0.0):
                tried.append({"target": b["id"], "result": "offset run folds"})
                continue
            start, t0 = run[0], tangent[0]
            # host vertices: on branches nonincident with the target, thick enough, inside a chord
            # window, outside the junction zones at the host branch's ends
            candidates = []
            for site in sites:
                hb = site["branch"]
                if hb == bi or endpoints[hb] & endpoints[bi] or site["diameter"] < dc / 0.8:
                    continue
                reach = V.collisions.KIN_REACH * (site["diameter"] / 2.0 + dc / 2.0 + margin) + site["diameter"] / 2.0
                for k, c in site["interior"]:
                    if not reach <= site["arc"][k] <= site["length"] - reach:
                        continue
                    delta = start - nodes[:3, c]
                    chord = float(np.linalg.norm(delta))
                    if not plan["host_chord_um"][0] <= chord <= plan["host_chord_um"][1]:
                        continue
                    if delta @ t0 < 0.3 * chord:
                        continue                # approach the run from behind its start
                    candidates.append((chord, c, hb))
            candidates.sort()
            for chord, c, hb in candidates[:plan["hosts_per_offset"]]:
                host_t = _host_tangent(nodes, c)
                delta = start - nodes[:3, c]
                sign = 1.0 if delta @ host_t >= 0.0 else -1.0
                lateral = delta - (delta @ host_t) * host_t
                if np.linalg.norm(lateral) < 1e-6:
                    continue
                lateral = unit(lateral)
                theta = math.radians(plan["leave_angle_deg"])
                leave = math.cos(theta) * sign * host_t + math.sin(theta) * lateral
                connector = V.tortuosity.bridge_path(nodes[:3, c], start, INSERT_STEP_UM, start_tangent=leave,
                                                     rng=rng, persistence=persistence, diameter=dc,
                                                     end_tangent=t0)
                xyz_c = np.vstack([connector, run[1:]])
                rows = np.column_stack([xyz_c, np.full(len(xyz_c), dc)])
                rows[0, :3] = nodes[:3, c]
                clear, gap = net.check(rows, c, margin, own_curve=True)
                if not clear:
                    tried.append({"target": b["id"], "host_column": int(c), "result": "collision",
                                  "min_clearance_um": gap})
                    continue
                nodes_new, labels_new = append_polyline(nodes, dict(labels), rows,
                                                        {"tree": 0, "origin": ORIGIN["controlled_companion"]})
                info = {"target_branch_columns": b["columns"], "target_endpoint_columns": b["endpoint_columns"],
                        "host_column": int(c), "host_diameter_um": float(nodes[3, c]), "diameter_um": dc,
                        "run_length_um": run_length, "run_start_arc_um": a0,
                        "design_clearance_um": plan["target_clearance_um"], "offset_um": offset,
                        "leave_angle_deg": plan["leave_angle_deg"], "connector_points": int(len(connector)),
                        "min_clearance_at_placement_um": gap, "attempts": len(tried) + 1}
                params.update({"companion": {k: plan[k] for k in (
                    "run_length_um", "target_clearance_um", "approach_band_um", "companion_diameter_um",
                    "target_min_diameter_um", "leave_angle_deg", "host_chord_um")},
                    "rng_companion": [seed, RNG_TAGS["companion"]]})
                out = built(nodes_new, labels_new["origin"],
                            "vSystem.F grammar via main.grow_network, plus one controlled companion side branch "
                            "(tortuosity.bridge_path connector, then a run offset from a nonincident branch)",
                            params, tree=labels_new["tree"], program=grown["program"], events=grown["events"],
                            extras={"pruned_stubs": grown["pruned_stubs"], "companion": info,
                                    "companion_failed_attempts": tried})
                return out, None
    return None, {"reason": "no clear companion placement", "extras": {"companion_failed_attempts": tried[-50:]}}


def _tree_path(nodes, start, end):
    """Vertex path (canonical columns) between two vertices of a tree, by breadth-first search."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order
    built_graph = V.graph.build(nodes)
    edges = built_graph["edges"]
    n = nodes.shape[1]
    matrix = coo_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(n, n))
    _, predecessors = breadth_first_order(matrix, start, directed=False, return_predecessors=True)
    path = [end]
    while path[-1] != start:
        previous = predecessors[path[-1]]
        if previous < 0:
            return None
        path.append(int(previous))
    return path[::-1], built_graph["canonical"]


def loop_opening(nodes, tip, partner, bridge_rows):
    """
    Opening of the loop a bridge closes: the smallest surface clearance between two points
    of the cycle whose separation along the cycle is at least a quarter of its length.
    """
    found = _tree_path(nodes, tip, partner)
    if found is None:
        return None
    path, canonical = found
    vdiam = V.graph.vertex_diameter(nodes, canonical)
    tree_rows = np.column_stack([nodes[:3, path].T, vdiam[path]])
    cycle = np.vstack([tree_rows, bridge_rows[::-1][1:-1]])
    closed = np.vstack([cycle, cycle[:1]])
    dense, arc = densify(closed, 2.0 * DENSE_STEP_UM)
    dense, arc = dense[:-1], arc[:-1]
    total = arc[-1] + np.linalg.norm(dense[-1, :3] - dense[0, :3])
    sep = np.abs(arc[:, None] - arc[None, :])
    sep = np.minimum(sep, total - sep)
    gap = (np.linalg.norm(dense[:, None, :3] - dense[None, :, :3], axis=2)
           - dense[:, None, 3] / 2.0 - dense[None, :, 3] / 2.0)
    far = sep >= total / 4.0
    return {"opening_min_clearance_um": float(gap[far].min()), "cycle_length_um": float(total),
            "tree_path_vertices": len(path)}


def build_tree_bridges(spec, plan, seed):
    """F11/F12: a grown tree, anastomosis.anastomose in mode any, then a seeded subset of valid bridges."""
    grown, params = grow_tree(spec, plan, seed)
    return add_bridges(grown, params, plan, seed, spec["persistence"])


def add_bridges(grown, params, plan, seed, persistence, common=COMMON):
    """
    Runs anastomosis.anastomose (mode any) on a grown tree and keeps a seeded subset of
    plan["bridges"] = (min, max) bridges whose loop opening is at least OPENING_MIN_UM.
    Returns (built candidate, None) or (None, failure).
    """
    tree_nodes = grown["nodes"]
    n0 = tree_nodes.shape[1]
    events = dict(grown["events"])
    rng = np.random.default_rng([int(seed), V.main.RNG_STREAMS["anastomosis"]])
    out_nodes, out_tree, bridges = V.anastomosis.anastomose(
        tree_nodes, rng, plan["anastomosis_fraction"], plan["anastomosis_radius"], mode="any",
        tree=grown["tree"], persistence=persistence, collision_margin=common["collision_margin_um"],
        events=events, attempts=common["collision_attempts"], index_kind="grid",
        min_separation=plan["anastomosis_min_separation"])
    if not np.array_equal(out_nodes[:, :n0], tree_nodes, equal_nan=True):
        raise RuntimeError("anastomose changed the tree columns")
    appended, column = [], n0
    for b in bridges:
        column += 1
        appended.append(out_nodes[:, column:column + b["points"]].T)
        column += b["points"]
    if column != out_nodes.shape[1]:
        raise RuntimeError("could not split the appended bridge polylines")
    assessed = []
    for i, b in enumerate(bridges):
        record = {"index": i, "tip": b["tip"], "partner": b["partner"], "partner_kind": b["partner_kind"],
                  "chord_um": b["chord_um"], "arc_um": b["arc_um"], "diameter_um": b["diameter_um"]}
        if b["partner"] >= n0 or b["tip"] >= n0:
            record["eligible"] = False
            record["reason"] = "joins an earlier bridge"
        else:
            opening = loop_opening(tree_nodes, b["tip"], b["partner"], appended[i])
            record.update(opening or {})
            record["eligible"] = bool(opening and opening["opening_min_clearance_um"] >= OPENING_MIN_UM)
            if not record["eligible"]:
                record["reason"] = "loop opening below three voxels"
        assessed.append(record)
    eligible = [r["index"] for r in assessed if r["eligible"]]
    lo, hi = plan["bridges"]
    params.update({"anastomosis": {"mode": "any", "fraction": plan["anastomosis_fraction"],
                                   "radius_tip_diameters": plan["anastomosis_radius"],
                                   "min_separation_segments": plan["anastomosis_min_separation"],
                                   "collision_margin_um": common["collision_margin_um"],
                                   "attempts": common["collision_attempts"], "persistence": persistence,
                                   "rng": [seed, V.main.RNG_STREAMS["anastomosis"]]},
                   "bridge_selection": {"count_range": [lo, hi], "opening_min_um": OPENING_MIN_UM,
                                        "rng": [seed, RNG_TAGS["bridge_selection"]],
                                        "rule": "seeded permutation of eligible bridges, first k kept, k = "
                                                "min(upper, eligible); eligible = both ends on tree columns "
                                                "and loop opening >= 150 um"}})
    extras = {"pruned_stubs": grown["pruned_stubs"], "bridges_made": len(bridges), "bridges_assessed": assessed}
    if len(eligible) < lo:
        return None, {"reason": f"{len(eligible)} eligible bridges of {len(bridges)} made; need {lo}",
                      "extras": extras}
    selector = np.random.default_rng([int(seed), RNG_TAGS["bridge_selection"]])
    k = min(hi, len(eligible))
    chosen = sorted(int(eligible[j]) for j in selector.permutation(len(eligible))[:k])
    nodes = tree_nodes
    labels = {"tree": grown["tree"], "origin": grown["origin"]}
    for i in chosen:
        nodes, labels = append_polyline(nodes, labels, appended[i], {"tree": V.anastomosis.BRIDGE,
                                                                      "origin": ORIGIN["anastomosis_bridge"]})
    extras["bridges_selected"] = chosen
    extras["unbridged_columns"] = n0
    out = built(nodes, labels["origin"], "vSystem.F grammar via main.grow_network, then anastomosis.anastomose "
                "(mode any); a seeded subset of its valid bridges is kept", params, tree=labels["tree"],
                program=grown["program"], events=events, extras=extras)
    return out, None


BUILDERS = {"straight_tube": build_straight_tube, "straight_y": build_straight_y,
            "walk_program": build_walk_program, "tree": build_tree, "tree_thin": build_tree_thin,
            "tree_companion": build_tree_companion, "tree_bridges": build_tree_bridges}


def build(spec, plan, seed):
    builder = BUILDERS[plan["builder"]]
    result = builder(spec, plan, seed)
    if isinstance(result, tuple):
        return result
    return result, None


# ==============================================================================================
# Graph analysis
# ==============================================================================================
def analyse_graph(nodes, origin=None):
    """
    Graph of an archive (V-System graph.py) with stable topology-node and branch IDs.

    Branches are graph.segments: maximal paths between vertices of degree other than two,
    after contracting degree-two sampling vertices; a pure cycle is one closed branch.
    """
    built_graph = V.graph.build(nodes)
    canonical, edges, deg = built_graph["canonical"], built_graph["edges"], built_graph["degree"]
    n = nodes.shape[1]
    betti = V.graph.betti(edges, canonical)
    vertices = np.flatnonzero(canonical == np.arange(n))
    vdiam = V.graph.vertex_diameter(nodes, canonical)
    paths = V.graph.segments(nodes, edges, canonical)

    finite = canonical >= 0
    idx = np.flatnonzero(finite[:-1] & finite[1:])
    pair_of = {}
    for c, a, b in zip(idx.tolist(), canonical[idx].tolist(), canonical[idx + 1].tolist()):
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        pair_of.setdefault(key, []).append((c, c + 1) if a < b else (c + 1, c))

    def own_column(v, w):
        key = (v, w) if v < w else (w, v)
        cols = pair_of[key][0]
        return cols[0] if key[0] == v else cols[1]

    topo = [int(v) for v in vertices if deg[v] != 2]
    node_id = {v: f"N{i:04d}" for i, v in enumerate(topo)}
    branches, edge_branch, vertex_branch = [], {}, {}
    for bi, path in enumerate(paths):
        path = [int(v) for v in path]
        closed = path[0] == path[-1]
        for v in (path[0], path[-1]):
            if v not in node_id:                # anchor of a pure cycle
                node_id[v] = f"N{len(node_id):04d}"
        xyz = nodes[:3, path].T
        diam = vdiam[path].copy()
        diam[0] = nodes[3, own_column(path[0], path[1])]
        diam[-1] = nodes[3, own_column(path[-1], path[-2])]
        keys = [((a, b) if a < b else (b, a)) for a, b in zip(path[:-1], path[1:])]
        columns = set()
        for key in keys:
            edge_branch[key] = bi
            for pair in pair_of[key]:
                columns.update(pair)
        for v in path:
            vertex_branch.setdefault(v, bi)
        arc = float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
        chord = float(np.linalg.norm(xyz[-1] - xyz[0]))
        names = []
        if origin is not None:
            codes = sorted({int(origin[c]) for c in columns if origin[c] >= 0})
            names = [ORIGINS[c] for c in codes]
        branches.append({"id": f"B{bi:04d}", "endpoints": [node_id[path[0]], node_id[path[-1]]],
                         "endpoint_columns": [path[0], path[-1]], "closed": closed,
                         "vertex_columns": path, "columns": sorted(columns), "xyz_um": xyz.tolist(),
                         "diameter_um": diam.tolist(), "arc_length_um": arc, "chord_um": chord,
                         "arc_chord": (arc / chord) if (not closed and chord > 0) else None,
                         "origin": names})
    nodes_out = []
    for v, name in sorted(node_id.items(), key=lambda item: item[1]):
        d = int(deg[v])
        kind = "isolated" if d == 0 else "tip" if d == 1 else "junction" if d >= 3 else "ring_anchor"
        nodes_out.append({"id": name, "column": v, "xyz_um": nodes[:3, v].tolist(), "degree": d,
                          "kind": kind, "diameter_um": float(vdiam[v])})
    # every consecutive column pair belongs to one branch (zero-length pairs to a branch at that vertex)
    pair_branch = np.full(max(n - 1, 0), -1, dtype=np.int64)
    for c in idx.tolist():
        a, b = int(canonical[c]), int(canonical[c + 1])
        if a != b:
            pair_branch[c] = edge_branch[(a, b) if a < b else (b, a)]
        else:
            pair_branch[c] = vertex_branch.get(a, -1)
    nb = len(branches)
    incident = np.zeros((nb, nb), dtype=bool)
    ends = [set(b["endpoint_columns"]) for b in branches]
    for i in range(nb):
        for j in range(nb):
            incident[i, j] = bool(ends[i] & ends[j])
    degree_hist = {}
    for v in vertices:
        degree_hist[str(int(deg[v]))] = degree_hist.get(str(int(deg[v])), 0) + 1
    return {"canonical": canonical, "edges": edges, "node_kind": built_graph["node_kind"], "betti": betti,
            "branches": branches, "nodes": nodes_out, "pair_branch": pair_branch, "incident": incident,
            "degree_histogram": degree_hist, "tips": int(sum(1 for v in vertices if deg[v] == 1)),
            "junctions": int(sum(1 for v in vertices if deg[v] >= 3))}


# ==============================================================================================
# Geometry measurements
# ==============================================================================================
def tortuosity_stats(branches):
    """Arc/chord and sagitta of open branches; closed branches are reported separately."""
    per = []
    for b in branches:
        if b["closed"] or b["chord_um"] <= 0:
            continue
        xyz = np.asarray(b["xyz_um"])
        a, c = xyz[0], xyz[-1]
        axis = (c - a) / b["chord_um"]
        rel = xyz - a
        t = np.clip(rel @ axis, 0.0, b["chord_um"])
        sagitta = float(np.max(np.linalg.norm(rel - t[:, None] * axis, axis=1)))
        per.append({"id": b["id"], "arc_um": b["arc_length_um"], "chord_um": b["chord_um"],
                    "arc_chord": b["arc_chord"], "sagitta_vox": sagitta * SCALE})
    if not per:
        return {"open_branches": 0, "closed_branches": sum(1 for b in branches if b["closed"])}
    arcs = np.array([p["arc_um"] for p in per])
    ratios = np.array([p["arc_chord"] for p in per])
    sag = np.array([p["sagitta_vox"] for p in per])
    return {"open_branches": len(per), "closed_branches": sum(1 for b in branches if b["closed"]),
            "arc_chord_mean": float(ratios.mean()), "arc_chord_median": float(np.median(ratios)),
            "arc_chord_length_weighted": float((ratios * arcs).sum() / arcs.sum()),
            "arc_chord_max": float(ratios.max()), "sagitta_max_vox": float(sag.max()),
            "bent_length_fraction": float(arcs[sag >= TORTUOSITY_SAGITTA_VOX].sum() / arcs.sum()),
            "per_branch": per}


def dense_branch_samples(branches):
    points, radii, owner = [], [], []
    for bi, b in enumerate(branches):
        rows = np.column_stack([np.asarray(b["xyz_um"]), np.asarray(b["diameter_um"])])
        dense, _ = densify(rows, DENSE_STEP_UM)
        points.append(dense[:, :3])
        radii.append(dense[:, 3] / 2.0)
        owner.append(np.full(len(dense), bi))
    return np.vstack(points), np.concatenate(radii), np.concatenate(owner)


def nonincident_clearance(graph_info, margin, reach=CLEARANCE_SEARCH_UM):
    """
    Minimum surface clearance between every pair of branches that share no endpoint, on
    centrelines densified to DENSE_STEP_UM, with no exemptions.
    """
    from scipy.spatial import cKDTree
    branches = graph_info["branches"]
    incident = graph_info["incident"]
    points, radii, owner = dense_branch_samples(branches)
    kd = cKDTree(points)
    rmax = float(radii.max())
    pair_min = {}
    for bi in range(len(branches)):
        sel = np.flatnonzero(owner == bi)
        lists = kd.query_ball_point(points[sel], radii[sel] + rmax + reach)
        sizes = np.array([len(x) for x in lists])
        if not sizes.sum():
            continue
        qi = np.repeat(sel, sizes)
        si = np.fromiter((s for x in lists for s in x), dtype=np.int64, count=int(sizes.sum()))
        keep = (owner[si] > bi) & ~incident[bi, owner[si]]
        qi, si = qi[keep], si[keep]
        if not qi.size:
            continue
        gap = np.linalg.norm(points[qi] - points[si], axis=1) - radii[qi] - radii[si]
        other = owner[si]
        for o in np.unique(other):
            value = float(gap[other == o].min())
            if value <= reach:
                pair_min[(bi, int(o))] = value
    values = sorted(pair_min.items(), key=lambda item: item[1])
    result = {"search_reach_um": reach, "sample_step_um": DENSE_STEP_UM,
              "pairs_within_reach": len(values), "min_um": values[0][1] if values else None,
              "min_pair": [branches[values[0][0][0]]["id"], branches[values[0][0][1]]["id"]] if values else None,
              "min_is_lower_bound": not values,
              "pairs_below_zero": sum(1 for _, v in values if v < 0.0),
              "pairs_below_margin": sum(1 for _, v in values if v < margin),
              "pairs_below_digital_safe": sum(1 for _, v in values if v < DIGITAL_SAFE_CLEARANCE_UM),
              "closest_pairs": [{"branches": [branches[a]["id"], branches[b]["id"]], "clearance_um": v}
                                for (a, b), v in values[:10]]}
    return result, pair_min


def describe_clearance(nodes, margin):
    """V-System describe.clearance on polylines densified to DENSE_STEP_UM."""
    pieces = []
    for columns in V.graph.polylines(nodes):
        dense, _ = densify(polyline_rows(nodes, columns), DENSE_STEP_UM)
        pieces.append(dense)
    dense_nodes = assemble(pieces)
    result = V.describe.clearance(dense_nodes, margin=margin)
    result["sample_step_um"] = DENSE_STEP_UM
    return result


def max_sample_spacing(nodes):
    worst = 0.0
    for columns in V.graph.polylines(nodes):
        if len(columns) > 1:
            worst = max(worst, float(np.max(np.linalg.norm(np.diff(nodes[:3, columns], axis=1), axis=0))))
    return worst


def near_floor_terminals(graph_info, root_column):
    """Terminal branches (one tip end, not the root) with nonzero length inside the near-floor band."""
    tips = {n["column"] for n in graph_info["nodes"] if n["kind"] == "tip"}
    lo, hi = NEAR_FLOOR_BAND_UM
    found = []
    for b in graph_info["branches"]:
        ends = b["endpoint_columns"]
        terminal = (ends[0] in tips) != (ends[1] in tips) and root_column not in ends
        if not terminal:
            continue
        xyz = np.asarray(b["xyz_um"])
        d = np.asarray(b["diameter_um"])
        inside = (d[:-1] >= lo - ROUNDOFF_UM) & (d[:-1] <= hi) & (d[1:] >= lo - ROUNDOFF_UM) & (d[1:] <= hi)
        length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1)[inside].sum())
        if length > 0.0:
            found.append({"branch": b["id"], "band_length_um": length, "diameter_min_um": float(d.min()),
                          "diameter_max_um": float(d.max()), "origin": b["origin"]})
    return found


def bifurcation_asymmetry(graph_info):
    """
    At every degree-3 junction, the two thinner incident branches (taken as the daughters, by
    median diameter) and their ratio d_small / d_large. Median, because a walked daughter's
    first archive column still carries the parent's diameter.
    """
    ratios = []
    for n in graph_info["nodes"]:
        if n["degree"] != 3:
            continue
        ends = []
        for b in graph_info["branches"]:
            if n["column"] in b["endpoint_columns"]:
                ends.append(float(np.median(b["diameter_um"])))
        if len(ends) == 3:
            d = sorted(ends)
            ratios.append(d[0] / d[1])
    if not ratios:
        return {"junctions": 0}
    return {"junctions": len(ratios), "daughter_ratio_min": float(min(ratios)),
            "daughter_ratio_median": float(np.median(ratios)), "daughter_ratio_max": float(max(ratios)),
            "definition": "at each degree-3 junction, smaller / larger of the median diameters of the two "
                          "thinner incident branches"}


def diameter_summary(nodes, graph_info):
    d = nodes[3][np.isfinite(nodes[0])]
    lo, hi = NEAR_FLOOR_BAND_UM
    band_length = 0.0
    for b in graph_info["branches"]:
        xyz = np.asarray(b["xyz_um"])
        dd = np.asarray(b["diameter_um"])
        inside = (dd[:-1] >= lo - ROUNDOFF_UM) & (dd[:-1] <= hi) & (dd[1:] >= lo - ROUNDOFF_UM) & (dd[1:] <= hi)
        band_length += float(np.linalg.norm(np.diff(xyz, axis=0), axis=1)[inside].sum())
    return {"min_um": float(d.min()), "max_um": float(d.max()), "root_um": float(nodes[3, 0]),
            "all_finite": bool(np.all(np.isfinite(d))), "near_floor_band_um": list(NEAR_FLOOR_BAND_UM),
            "near_floor_length_um": band_length}


# ==============================================================================================
# Grid, rendering and digital topology
# ==============================================================================================
def choose_grid(nodes):
    """
    Volume shape from the geometry bounds including radii (and the connect-chain reach of
    sqrt(3)/2 voxel for thin vessels), with BORDER_VOX empty voxels on every side, under
    computeVoxel.fit_to_volume(fit="voxel_size"), which centres the centreline extent.
    """
    xyz = nodes[:3]
    finite = np.isfinite(xyz).all(axis=0)
    lo = np.nanmin(xyz, axis=1)
    hi = np.nanmax(xyz, axis=1)
    extent = hi - lo
    rel = (xyz[:, finite] - lo[:, None]) * SCALE
    r_geom = nodes[3, finite] / 2.0 * SCALE
    r_eff = np.maximum(r_geom, CHAIN_REACH_VOX)
    low = (rel - r_eff).min(axis=1)
    high = (rel + r_eff).max(axis=1)
    shape = []
    for axis in range(3):
        n = int(math.ceil(high[axis] - low[axis] + 2 * BORDER_VOX))
        while True:
            offset = (n - SCALE * extent[axis]) / 2.0
            if offset + low[axis] >= BORDER_VOX - 0.5 and offset + high[axis] <= n - BORDER_VOX - 0.5:
                break
            n += 1
        shape.append(n)
    shape = np.array(shape, dtype=int)
    offset = (shape - SCALE * extent) / 2.0
    t_vox = offset - lo * SCALE
    T = np.eye(4)
    T[:3, :3] *= SCALE
    T[:3, 3] = t_vox
    affine = np.eye(4)
    affine[:3, :3] *= SPACING_MM
    affine[:3, 3] = -SPACING_MM * t_vox
    geom_low = offset + (rel - r_geom).min(axis=1)
    geom_high = offset + (rel + r_geom).max(axis=1)
    return {"shape": shape, "lo_um": lo, "offset_vox": offset, "t_vox": t_vox, "T": T, "affine": affine,
            "geometry_bounds_vox": {"tube_min": geom_low.tolist(), "tube_max": geom_high.tolist(),
                                    "with_chain_reach_min": (offset + low).tolist(),
                                    "with_chain_reach_max": (offset + high).tolist()}}


def map_points(nodes, grid):
    """Grammar um -> voxel coordinates, by the expression computeVoxel.fit_to_volume uses."""
    scale = np.full(3, SCALE)
    return (nodes[:3] - grid["lo_um"][:, None]) * scale[:, None] + grid["offset_vox"][:, None]


def render(nodes, grid, points=None, radii=None):
    shape = tuple(int(v) for v in grid["shape"])
    if points is None:
        points, radii = V.computeVoxel.fit_to_volume(nodes, shape, fit="voxel_size", voxel_size=VOXEL_UM,
                                                     clip_axes=())
    mask = V.computeVoxel.rasterise_segments(points, radii, shape, connect=True)
    reference = np.zeros(shape, dtype=np.uint8)
    previous = None
    for i in range(points.shape[1]):       # the same traversal and skip rule as rasterise_segments
        p = points[:, i]
        if np.isnan(p[0]):
            previous = None
            continue
        if previous is not None and not np.array_equal(p, previous):
            V.computeVoxel.rasterise_line(reference, previous, p)
        previous = p
    return mask, reference, points, radii


def branch_voxel_sets(points, radii, shape, pair_branch, n_branches, capsules):
    """Per-branch linear voxel indices, drawn with the renderer's own calls into a scratch volume."""
    scratch = np.zeros(tuple(shape), dtype=np.uint8)
    groups = {}
    for c in np.flatnonzero(pair_branch >= 0).tolist():
        p, q = points[:, c], points[:, c + 1]
        if np.isnan(p[0]) or np.isnan(q[0]) or np.array_equal(p, q):
            continue
        groups.setdefault(int(pair_branch[c]), []).append(c)
    sets = [np.zeros(0, dtype=np.int64) for _ in range(n_branches)]
    shape = np.asarray(shape)
    for b, cols in groups.items():
        for c in cols:
            if capsules:
                V.computeVoxel.rasterise_capsule(scratch, points[:, c], points[:, c + 1], radii[c], radii[c + 1])
            V.computeVoxel.rasterise_line(scratch, points[:, c], points[:, c + 1])
        cols_all = np.array(cols + [c + 1 for c in cols])
        pad = (float(radii[cols_all].max()) if capsules else 0.0) + 2.0
        lo = np.maximum(np.floor(points[:, cols_all].min(axis=1) - pad).astype(int), 0)
        hi = np.minimum(np.ceil(points[:, cols_all].max(axis=1) + pad).astype(int) + 1, shape)
        box = scratch[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        local = np.array(np.nonzero(box)) + lo[:, None]
        sets[b] = np.ravel_multi_index(local, tuple(shape))
        box[...] = 0
    return sets


def branch_contacts(sets, shape, incident):
    """Pairs of branches whose voxel sets overlap or are 26-adjacent."""
    from scipy import ndimage
    all_idx = np.concatenate([s for s in sets]) if sets else np.zeros(0, dtype=np.int64)
    all_owner = np.concatenate([np.full(len(s), b) for b, s in enumerate(sets)]) if sets else np.zeros(0, dtype=np.int64)
    order = np.argsort(all_idx, kind="stable")
    sidx, sowner = all_idx[order], all_owner[order]
    shape = tuple(int(v) for v in shape)
    contacts = set()
    for b, lin in enumerate(sets):
        if lin.size == 0:
            continue
        coords = np.array(np.unravel_index(lin, shape))
        lo = np.maximum(coords.min(axis=1) - 1, 0)
        hi = np.minimum(coords.max(axis=1) + 2, shape)
        local = np.zeros(hi - lo, dtype=bool)
        local[tuple(coords - lo[:, None])] = True
        grown = ndimage.binary_dilation(local, structure=np.ones((3, 3, 3), dtype=bool))
        near = np.ravel_multi_index(np.array(np.nonzero(grown)) + lo[:, None], shape)
        left = np.searchsorted(sidx, near, "left")
        right = np.searchsorted(sidx, near, "right")
        counts = right - left
        if counts.sum() == 0:
            continue
        starts = np.repeat(left, counts) + (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts))
        for o in np.unique(sowner[starts]).tolist():
            if o != b:
                contacts.add((min(b, o), max(b, o)))
    nonincident = sorted((a, b) for a, b in contacts if not incident[a, b])
    return sorted(contacts), nonincident


def euler_closed_cubes(padded):
    """Euler characteristic of the union of closed unit cubes at the foreground voxels."""
    P = padded
    vertices = (P[:-1, :-1, :-1] | P[1:, :-1, :-1] | P[:-1, 1:, :-1] | P[:-1, :-1, 1:]
                | P[1:, 1:, :-1] | P[1:, :-1, 1:] | P[:-1, 1:, 1:] | P[1:, 1:, 1:]).sum()
    edges = ((P[:, :-1, :-1] | P[:, 1:, :-1] | P[:, :-1, 1:] | P[:, 1:, 1:]).sum()
             + (P[:-1, :, :-1] | P[1:, :, :-1] | P[:-1, :, 1:] | P[1:, :, 1:]).sum()
             + (P[:-1, :-1, :] | P[1:, :-1, :] | P[:-1, 1:, :] | P[1:, 1:, :]).sum())
    faces = (P[:-1, :, :] | P[1:, :, :]).sum() + (P[:, :-1, :] | P[:, 1:, :]).sum() + (P[:, :, :-1] | P[:, :, 1:]).sum()
    cubes = P.sum()
    return int(vertices) - int(edges) + int(faces) - int(cubes)


JUNCTION_RULE = ("Junction neighbourhoods (V-System collisions.py / describe.clearance at the 100 um generation "
                 "margin m): two points of polylines sharing a vertex J are exempt when arc(P, J) + arc(Q, J) < "
                 "1.7 (r_P + r_Q + m); two points of one polyline are exempt when their arc separation is at most "
                 "2 (r_P + r_Q + m). Branches that share no endpoint get no exemption.")


TOPOLOGY_CONVENTION = ("foreground 26-connected, background 6-connected (the (26,6) pair). Components b0 by "
                       "26-connected labelling; cavities b2 = (6-connected background components of the "
                       "one-voxel-padded volume) - 1; Euler characteristic chi of the union of closed unit cubes "
                       "at foreground voxels (V - E + F - C), cross-checked with skimage.measure.euler_number "
                       "(connectivity=3); cycles b1 = b0 + b2 - chi.")


def digital_topology(volume):
    from scipy import ndimage
    from skimage.measure import euler_number
    fg = volume.astype(bool)
    padded = np.pad(fg, 1)
    _, b0 = ndimage.label(fg, structure=np.ones((3, 3, 3), dtype=bool))
    _, nbg = ndimage.label(~padded, structure=ndimage.generate_binary_structure(3, 1))
    chi = euler_closed_cubes(padded)
    chi_sk = int(euler_number(fg, connectivity=3))
    _, b0_6 = ndimage.label(fg, structure=ndimage.generate_binary_structure(3, 1))
    b2 = int(nbg) - 1
    return {"components": int(b0), "cavities": b2, "euler": chi, "euler_skimage": chi_sk,
            "cycles": int(b0) + b2 - chi, "components_6conn": int(b0_6), "voxels": int(fg.sum())}


def border_report(volume):
    coords = np.array(np.nonzero(volume))
    shape = np.array(volume.shape)
    if coords.size == 0:
        return {"empty": True}
    low = coords.min(axis=1)
    high = shape - 1 - coords.max(axis=1)
    return {"empty": False, "empty_voxels_low": low.tolist(), "empty_voxels_high": high.tolist(),
            "min_empty_voxels": int(min(low.min(), high.min()))}


def voxel_gap(sets, a, b, shape):
    """Smallest centre distance (voxels) between two branches' voxel sets."""
    from scipy.spatial import cKDTree
    if sets[a].size == 0 or sets[b].size == 0:
        return None
    pa = np.array(np.unravel_index(sets[a], tuple(shape))).T
    pb = np.array(np.unravel_index(sets[b], tuple(shape))).T
    distance, _ = cKDTree(pb).query(pa, k=1)
    return float(distance.min())


# ==============================================================================================
# Evaluation of a candidate
# ==============================================================================================
def geometry_checks(spec, plan, candidate):
    """Checks that need no rendering. Returns (records, measurements)."""
    nodes = candidate["nodes"]
    fid = spec["id"]
    graph_info = analyse_graph(nodes, candidate["origin"])
    betti = graph_info["betti"]
    branches = graph_info["branches"]
    margin = COMMON["collision_margin_um"]
    records, m = [], {"graph": graph_info}

    d = diameter_summary(nodes, graph_info)
    m["diameters"] = d
    m["bifurcation_asymmetry"] = bifurcation_asymmetry(graph_info)
    finite_cols = np.isfinite(nodes[0])
    finite_ok = bool(np.all(np.isfinite(nodes[:, finite_cols])) and np.all(np.isnan(nodes[:, ~finite_cols])))
    records.append(check("root_diameter", abs(d["root_um"] - spec["root_diameter_um"]) <= ROUNDOFF_UM
                         and d["max_um"] <= spec["root_diameter_um"] + ROUNDOFF_UM,
                         {"root_um": d["root_um"], "max_um": d["max_um"]},
                         f"root = {spec['root_diameter_um']} um within roundoff and no larger vessel"))
    records.append(check("diameter_bounds", finite_ok and d["min_um"] >= D_FLOOR_UM - ROUNDOFF_UM
                         and d["max_um"] <= D_ROOT_MAX_UM + ROUNDOFF_UM,
                         {"min_um": d["min_um"], "max_um": d["max_um"], "finite_samples_valid": finite_ok},
                         "all finite samples valid, 50 <= d <= 500 um, anomalies disabled"))
    cyc_lo, cyc_hi = spec["required_independent_cycle_range"]
    records.append(check("graph_topology", betti["components"] == spec["required_graph_components"]
                         and cyc_lo <= betti["cycles"] <= cyc_hi, betti,
                         f"components = {spec['required_graph_components']}, cycles in [{cyc_lo}, {cyc_hi}]"))
    n_branches = len(branches)
    lo, hi = spec["desired_branch_count_range"]
    records.append(check("branch_count", lo <= n_branches <= hi, n_branches, f"[{lo}, {hi}] branches",
                         kind="desired"))
    m["max_sample_spacing_um"] = max_sample_spacing(nodes)

    # continuous-tube clearance, two independent ways
    desc0 = describe_clearance(nodes, 0.0)
    desc_m = describe_clearance(nodes, margin)
    nonincident, pair_min = nonincident_clearance(graph_info, margin)
    m["clearance"] = {"describe_margin_0": desc0, "describe_margin_generation": desc_m,
                      "nonincident_branches": nonincident}
    nonincident_ok = nonincident["pairs_below_zero"] == 0
    records.append(check("continuous_tube_clearance",
                         (desc_m["min"] is None or desc_m["min"] >= 0.0) and nonincident_ok,
                         {"describe_pairs_below_generation_margin": desc_m["violations"],
                          "describe_min_um_at_generation_margin": desc_m["min"],
                          "nonincident_min_um": nonincident["min_um"],
                          "nonincident_pairs_below_zero": nonincident["pairs_below_zero"],
                          "info_describe_margin_0": {"violations": desc0["violations"], "min_um": desc0["min"]}},
                         "no two tubes overlap outside the documented junction neighbourhoods; nonincident "
                         "tubes never overlap",
                         note=JUNCTION_RULE + f" Centrelines densified to {DENSE_STEP_UM} um; a sampled minimum "
                              f"overstates the continuous one by at most {DENSE_STEP_UM} um. Growth checks centreline points only, so "
                              "densified pairs can sit slightly below the 100 um target margin; their count and "
                              "minimum are reported, and only an overlap (< 0 um) fails. The margin-0 figures "
                              "use exemption windows sized without the margin; pairs they flag share a junction "
                              "(the nonincident check has none) and are sibling or parent overlaps just beyond "
                              "those smaller windows."))
    exposures = tip_exposures(graph_info)
    m["tip_exposures"] = exposures
    worst = min((e["exposure_um"] for e in exposures), default=None)
    records.append(check("terminal_tips_exposed", worst is None or worst >= STUB_EXPOSURE_MIN_UM,
                         {"terminal_branches": len(exposures), "min_exposure_um": worst},
                         f"every terminal tip lies >= {STUB_EXPOSURE_MIN_UM} um (2 voxels) outside every other tube",
                         kind="desired", note="adapter-added quality check (user decision 2026-10-01): a tip "
                                              "buried in another vessel cannot be recovered from the mask"))
    tort = tortuosity_stats(branches)
    m["tortuosity"] = tort
    described = V.describe.describe(nodes, edges=graph_info["edges"], margin=0.0)
    m["describe"] = {k: described[k] for k in ("arc_chord", "curvature_per_um", "tips", "degree_histogram",
                                                "total_length_mm", "diameter_um", "orientation")}
    tortuous = fid not in ("F01", "F02")
    if fid in ("F01", "F02"):
        records.append(check("straight_control", tort["arc_chord_max"] <= 1.0 + 1e-9,
                             {"arc_chord_max": tort["arc_chord_max"]}, "every branch exactly straight",
                             kind="hard"))
    elif fid == "F03":
        t = TORTUOSITY_TARGETS["F03"]
        records.append(check("tortuosity_visible", tort["arc_chord_max"] >= t["arc_chord_min"]
                             and tort["sagitta_max_vox"] >= t["sagitta_min_vox"],
                             {"arc_chord": tort["arc_chord_max"], "sagitta_vox": tort["sagitta_max_vox"]},
                             f"arc/chord >= {t['arc_chord_min']} and sagitta >= {t['sagitta_min_vox']} voxels",
                             kind="desired"))
    elif fid == "F04":
        t = TORTUOSITY_TARGETS["F04"]
        sag = min(p["sagitta_vox"] for p in tort["per_branch"])
        records.append(check("tortuosity_visible", tort["arc_chord_length_weighted"] >= t["arc_chord_min"]
                             and sag >= t["per_branch_sagitta_min_vox"],
                             {"arc_chord_length_weighted": tort["arc_chord_length_weighted"],
                              "min_branch_sagitta_vox": sag},
                             f"length-weighted arc/chord >= {t['arc_chord_min']}, every branch sagitta >= "
                             f"{t['per_branch_sagitta_min_vox']} voxel", kind="desired"))
    if tortuous and fid not in ("F03", "F04"):
        t = TORTUOSITY_TARGETS["tree"]
        records.append(check("tortuosity_visible", tort["arc_chord_length_weighted"] >= t["arc_chord_min"]
                             and tort["bent_length_fraction"] >= t["bent_length_fraction_min"],
                             {"arc_chord_length_weighted": tort["arc_chord_length_weighted"],
                              "bent_length_fraction": tort["bent_length_fraction"]},
                             f"length-weighted arc/chord >= {t['arc_chord_min']} and >= "
                             f"{t['bent_length_fraction_min']:.0%} of open-branch length in branches with sagitta "
                             f">= {TORTUOSITY_SAGITTA_VOX} voxel", kind="desired"))
    if fid == "F08":
        found = near_floor_terminals(graph_info, root_column=0)
        m["near_floor_terminals"] = found
        records.append(check("thin_terminal_branches", len(found) >= 3, {"count": len(found), "branches": found},
                             ">= 3 distinct terminal branches with a 50-75 um calibre segment of nonzero length"))
    elif fid in ("F09", "F10", "F11", "F12"):
        found = near_floor_terminals(graph_info, root_column=0)
        m["near_floor_terminals"] = found
        records.append(check("near_floor_terminals_sought", None if not found else True,
                             {"count": len(found)}, "additional near-floor terminal branches where feasible",
                             kind="desired", note="informative; not a selection criterion"))
    if fid == "F10":
        records.extend(approach_checks(plan, candidate, graph_info, pair_min, m))
    if fid in ("F11", "F12"):
        records.extend(opening_checks(candidate, graph_info, m))
    return records, m


def _branch_of_columns(graph_info, columns):
    wanted = set(columns)
    best, score = None, 0
    for bi, b in enumerate(graph_info["branches"]):
        overlap = len(wanted & set(b["columns"]))
        if overlap > score:
            best, score = bi, overlap
    return best


def approach_checks(plan, candidate, graph_info, pair_min, m):
    info = candidate["extras"].get("companion")
    records = []
    if not info:
        return [check("close_approach", False, None, "a controlled companion exists", kind="desired")]
    companion_cols = np.flatnonzero(candidate["origin"] == ORIGIN["controlled_companion"]).tolist()
    cb = _branch_of_columns(graph_info, companion_cols)
    tb = _branch_of_columns(graph_info, info["target_branch_columns"])
    branches = graph_info["branches"]
    incident = bool(graph_info["incident"][cb, tb])
    target = branches[tb]
    comp = branches[cb]
    t_rows = np.column_stack([np.asarray(target["xyz_um"]), np.asarray(target["diameter_um"])])
    c_rows = np.column_stack([np.asarray(comp["xyz_um"]), np.asarray(comp["diameter_um"])])
    t_dense, t_arc = densify(t_rows, DENSE_STEP_UM)
    c_dense, _ = densify(c_rows, DENSE_STEP_UM)
    from scipy.spatial import cKDTree
    kd = cKDTree(c_dense[:, :3])
    dist, idx = kd.query(t_dense[:, :3], k=1)
    gap = dist - t_dense[:, 3] / 2.0 - c_dense[idx, 3] / 2.0
    # the approach: target arc where the surface gap is within the band's upper value
    band_lo, band_hi = plan["approach_band_um"]
    within = gap <= band_hi
    step = np.diff(t_arc)
    approach_len = float(step[within[:-1] & within[1:]].sum())
    min_gap = float(gap.min())
    measured = {"companion_branch": comp["id"], "target_branch": target["id"], "nonincident": not incident,
                "min_surface_clearance_um": min_gap, "approach_length_um": approach_len,
                "approach_length_vox": approach_len * SCALE, "band_um": [band_lo, band_hi],
                "companion_diameter_um": info["diameter_um"], "target_diameter_um": float(np.min(target["diameter_um"]))}
    m["close_approach"] = measured
    records.append(check("close_approach_geometry",
                         (not incident) and band_lo <= min_gap <= band_hi and approach_len >= 3 * VOXEL_UM,
                         measured, "nonincident pair, minimum surface clearance in [100, 150] um over an "
                                   "approach of >= 3 voxels (150 um) along the target", kind="desired"))
    m["close_approach_branches"] = (cb, tb)
    return records


def opening_checks(candidate, graph_info, m):
    nodes = candidate["nodes"]
    extras = candidate["extras"]
    n0 = extras["unbridged_columns"]
    tree_nodes = nodes[:, :n0]
    selected = [extras["bridges_assessed"][i] for i in extras["bridges_selected"]]
    openings = []
    for polyline in V.graph.polylines(nodes):
        if polyline[0] < n0:
            continue
        rows = polyline_rows(nodes, polyline)
        tip = int(V.graph.canonical_columns(tree_nodes)[np.flatnonzero(
            np.all(tree_nodes[:3].T == rows[0, :3], axis=1))[0]])
        partner = int(V.graph.canonical_columns(tree_nodes)[np.flatnonzero(
            np.all(tree_nodes[:3].T == rows[-1, :3], axis=1))[0]])
        opening = loop_opening(tree_nodes, tip, partner, rows)
        openings.append({"tip": tip, "partner": partner, **(opening or {})})
    m["loop_openings"] = openings
    values = [o.get("opening_min_clearance_um") for o in openings]
    ok = bool(values) and all(v is not None and v >= OPENING_MIN_UM for v in values)
    return [check("loop_openings_resolved", ok, {"openings_um": values, "bridges": selected},
                  f">= {OPENING_MIN_UM} um (three voxels) of background across every loop opening",
                  kind="desired",
                  note="opening = smallest surface clearance between cycle points at least a quarter of the "
                       "cycle length apart along it")]


def render_checks(spec, candidate, m):
    """Rendering, containment, border, digital topology and contacts."""
    nodes = candidate["nodes"]
    graph_info = m["graph"]
    grid = choose_grid(nodes)
    shape = grid["shape"]
    mask, reference, points, radii = render(nodes, grid)
    records = []
    mapped = map_points(nodes, grid)
    linear = (grid["T"][:3, :3] @ np.nan_to_num(nodes[:3]) + grid["T"][:3, 3:4])
    finite = np.isfinite(nodes[0])
    records.append(check("coordinate_transform", bool(np.array_equal(mapped, points, equal_nan=True))
                         and float(np.max(np.abs(linear[:, finite] - points[:, finite]))) < 1e-9,
                         {"max_abs_deviation_vox": float(np.max(np.abs(linear[:, finite] - points[:, finite])))},
                         "fit_to_volume(voxel_size=50) equals q = T p within roundoff"))
    cyc_lo, cyc_hi = spec["required_independent_cycle_range"]
    graph_cycles = graph_info["betti"]["cycles"]
    border = border_report(mask)
    gb = grid["geometry_bounds_vox"]
    inside = (min(gb["with_chain_reach_min"]) >= BORDER_VOX - 0.5
              and all(h <= n - BORDER_VOX - 0.5 for h, n in zip(gb["with_chain_reach_max"], shape)))
    records.append(check("volume_nonempty_binary", bool(mask.any()) and bool(reference.any())
                         and mask.dtype == np.uint8 and set(np.unique(mask)) <= {0, 1}
                         and set(np.unique(reference)) <= {0, 1},
                         {"mask_voxels": int(mask.sum()), "reference_voxels": int(reference.sum())},
                         "nonempty, binary uint8"))
    records.append(check("reference_in_mask", bool(np.all(mask[reference.astype(bool)] == 1)),
                         {"reference_outside_mask": int(np.sum(reference.astype(bool) & ~mask.astype(bool)))},
                         "reference_skeleton contained in vessel_mask"))
    records.append(check("no_boundary_contact", (not border["empty"]) and border["min_empty_voxels"] >= BORDER_VOX
                         and inside, {"mask": border, "geometry_bounds_vox": gb, "shape": shape.tolist()},
                         f">= {BORDER_VOX} empty voxels on every side; tubes, caps and chain voxels inside"))
    mask_topo = digital_topology(mask)
    ref_topo = digital_topology(reference)
    m["rendered_topology"] = mask_topo
    m["reference_topology"] = ref_topo
    intended_ok = lambda t: (t["components"] == 1 and t["cavities"] == 0 and t["euler"] == t["euler_skimage"]
                             and cyc_lo <= t["cycles"] <= cyc_hi and t["cycles"] == graph_cycles)
    records.append(check("mask_digital_topology", intended_ok(mask_topo), mask_topo,
                         f"b0 = 1, b1 = graph cycles ({graph_cycles}) in [{cyc_lo}, {cyc_hi}], b2 = 0",
                         note=TOPOLOGY_CONVENTION))
    records.append(check("reference_digital_topology", intended_ok(ref_topo) and ref_topo["components_6conn"] == 1,
                         ref_topo, f"b0 = 1 (also under 6-connectivity), b1 = {graph_cycles}, b2 = 0",
                         note="reference declared 26-connected foreground / 6-connected background; it is "
                              "built from face-connected chains, so it is also one 6-connected component"))
    nb = len(graph_info["branches"])
    sets_mask = branch_voxel_sets(points, radii, shape, graph_info["pair_branch"], nb, capsules=True)
    sets_ref = branch_voxel_sets(points, radii, shape, graph_info["pair_branch"], nb, capsules=False)
    union = np.zeros(int(np.prod(shape)), dtype=bool)
    for s in sets_mask:
        union[s] = True
    union_ok = bool(np.array_equal(union.reshape(tuple(shape)), mask.astype(bool)))
    _, bad_mask = branch_contacts(sets_mask, shape, graph_info["incident"])
    _, bad_ref = branch_contacts(sets_ref, shape, graph_info["incident"])
    ids = [b["id"] for b in graph_info["branches"]]
    records.append(check("mask_no_unintended_contact", union_ok and not bad_mask,
                         {"nonincident_contacts": [[ids[a], ids[b]] for a, b in bad_mask],
                          "per_branch_union_equals_mask": union_ok},
                         "no two branches without a shared endpoint overlap or touch under 26-connectivity"))
    records.append(check("reference_no_unintended_contact", not bad_ref,
                         {"nonincident_contacts": [[ids[a], ids[b]] for a, b in bad_ref]},
                         "no two reference paths without a shared endpoint touch under 26-connectivity"))
    if spec["id"] == "F10" and "close_approach_branches" in m:
        cb, tb = m["close_approach_branches"]
        gap = voxel_gap(sets_mask, cb, tb, shape)
        m["close_approach"]["min_voxel_centre_distance_vox"] = gap
        records.append(check("close_approach_digital_separation", gap is not None and gap > math.sqrt(3.0),
                             {"min_voxel_centre_distance_vox": gap},
                             "companion and target voxel sets are not 26-adjacent (centre distance > sqrt(3))",
                             kind="desired"))
    rendered = {"grid": grid, "mask": mask, "reference": reference, "points": points, "radii": radii}
    if spec["id"] in ("F11", "F12"):
        n0 = candidate["extras"]["unbridged_columns"]
        u_nodes = nodes[:, :n0]
        u_mask, u_ref, _, _ = render(u_nodes, grid, points=points[:, :n0], radii=radii[:n0])
        u_graph = analyse_graph(u_nodes, candidate["origin"][:n0])
        u_mask_topo = digital_topology(u_mask)
        u_ref_topo = digital_topology(u_ref)
        ok = (u_graph["betti"]["components"] == 1 and u_graph["betti"]["cycles"] == 0
              and u_mask_topo["components"] == 1 and u_mask_topo["cycles"] == 0 and u_mask_topo["cavities"] == 0
              and u_ref_topo["components"] == 1 and u_ref_topo["cycles"] == 0 and u_ref_topo["cavities"] == 0
              and border_report(u_mask)["min_empty_voxels"] >= BORDER_VOX)
        records.append(check("unbridged_counterpart", ok,
                             {"graph": u_graph["betti"], "mask": u_mask_topo, "reference": u_ref_topo},
                             "pre-bridge tree on the same grid: one component, no cycles, no cavities"))
        rendered["unbridged"] = {"nodes": u_nodes, "mask": u_mask, "reference": u_ref, "graph": u_graph,
                                 "mask_topology": u_mask_topo, "reference_topology": u_ref_topo,
                                 "tree": candidate["tree"][:n0], "origin": candidate["origin"][:n0]}
    return records, rendered


def failed(records, kinds=("hard", "desired")):
    return [r for r in records if r["kind"] in kinds and r["passed"] is False]


def evaluate(spec, plan, candidate):
    records, m = geometry_checks(spec, plan, candidate)
    selection_failures = [r for r in failed(records) if r["check"] != "near_floor_terminals_sought"]
    if selection_failures:
        return False, records, m, None
    render_records, rendered = render_checks(spec, candidate, m)
    records.extend(render_records)
    selection_failures = [r for r in failed(records) if r["check"] != "near_floor_terminals_sought"]
    return not selection_failures, records, m, rendered


# ==============================================================================================
# Writers
# ==============================================================================================
def write_nifti(path, data, affine):
    import nibabel as nib
    img = nib.Nifti1Image(np.ascontiguousarray(data, dtype=np.uint8), affine)
    img.header.set_data_dtype(np.uint8)
    img.header.set_xyzt_units(xyz="mm")
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    img.header["descrip"] = b"synthetic V-System fixture, 0/1 uint8, xyz"
    nib.save(img, path)


def read_nifti_check(path, expected, affine):
    import nibabel as nib
    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    header = img.header
    units = header.get_xyzt_units()[0]
    return {"shape_ok": data.shape == expected.shape, "dtype": str(data.dtype),
            "values_ok": bool(np.array_equal(data, expected)),
            "zooms_mm": [float(z) for z in header.get_zooms()[:3]],
            "spatial_unit": units, "affine_ok": bool(np.allclose(img.affine, affine, atol=1e-7)),
            "qform_equals_sform": bool(np.allclose(img.get_qform(), img.get_sform(), atol=1e-7)),
            "qform_code": int(header["qform_code"]), "sform_code": int(header["sform_code"]),
            "affine": img.affine.tolist()}


def graph_json(spec, graph_info, extra=None):
    branches = []
    for b in graph_info["branches"]:
        branches.append({k: b[k] for k in ("id", "endpoints", "closed", "xyz_um", "diameter_um", "arc_length_um",
                                           "chord_um", "arc_chord", "origin", "vertex_columns")})
    payload = {
        "fixture_id": spec["id"], "units": "um",
        "coordinates": "archive (grammar) coordinates in micrometres; voxel q = T @ [p, 1] with T in metadata.json",
        "branch_count_convention": BRANCH_CONVENTION,
        "node_note": "nodes are topology vertices (degree != 2) with stable IDs; branch samples are sampling "
                     "vertices and carry no IDs. 'column' and 'vertex_columns' index reference_network.npz nodes.",
        "diameter_note": "branch end diameters are the branch's own column diameters at that vertex; interior "
                         "sample diameters are graph.vertex_diameter (largest over coincident columns). V-System "
                         "writes a walked daughter's first column before its own diameter is read, so a "
                         "grammar branch's first sample carries its parent's diameter and the rendered tube "
                         "tapers to the daughter calibre over one walk step",
        "counts": {"nodes": len(graph_info["nodes"]), "branches": len(branches),
                   "tips": graph_info["tips"], "junctions": graph_info["junctions"],
                   "isolated_nodes": sum(1 for n in graph_info["nodes"] if n["kind"] == "isolated"),
                   "closed_branches": sum(1 for b in branches if b["closed"])},
        "betti": graph_info["betti"], "degree_histogram": graph_info["degree_histogram"],
        "nodes": graph_info["nodes"], "branches": branches,
    }
    if extra:
        payload.update(extra)
    return payload


BRANCH_CONVENTION = ("a branch is a maximal path between vertices of degree other than two (tips, junctions), "
                     "after contracting degree-two sampling vertices (V-System graph.segments); a component made "
                     "only of degree-two vertices is one closed branch. Vertices are archive columns merged within "
                     "graph.DEFAULT_TOL (1e-6 um); the root's free end is a tip.")


def write_preview(path, spec, rendered, candidate, graph_info, m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Patch
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    mask = rendered["mask"]
    points = rendered["points"]
    shape = mask.shape
    origin = candidate["origin"]
    colours = {"grammar_walk_tree": "#c0392b", "explicit_polyline": "#c0392b", "custom_walk_program": "#c0392b",
               "anastomosis_bridge": "#1f77b4", "controlled_thin_daughter": "#e67e22",
               "controlled_companion": "#8e44ad"}
    labels = {"grammar_walk_tree": "generating centreline", "explicit_polyline": "generating centreline",
              "custom_walk_program": "generating centreline", "anastomosis_bridge": "anastomosis bridge (loop)",
              "controlled_thin_daughter": "controlled thin daughter (50-70 um)",
              "controlled_companion": "controlled companion branch"}
    highlight = None
    if "close_approach_branches" in m:
        highlight = graph_info["branches"][m["close_approach_branches"][1]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), constrained_layout=True)
    views = [("XY (max over z)", 2, 0, 1), ("XZ (max over y)", 1, 0, 2), ("YZ (max over x)", 0, 1, 2)]
    used = set()
    for ax, (title, axis, a, b) in zip(axes, views):
        mip = mask.max(axis=axis).T
        ax.imshow(mip, origin="lower", cmap="Greys", vmin=0, vmax=2.2, interpolation="nearest",
                  extent=(-0.5 * SPACING_MM, (shape[a] - 0.5) * SPACING_MM, -0.5 * SPACING_MM,
                          (shape[b] - 0.5) * SPACING_MM))
        for columns in V.graph.polylines(candidate["nodes"]):
            name = ORIGINS[int(origin[columns[0]])] if origin[columns[0]] >= 0 else "grammar_walk_tree"
            used.add(name)
            ax.plot(points[a, columns] * SPACING_MM, points[b, columns] * SPACING_MM, color=colours[name],
                    lw=0.9 if name in ("grammar_walk_tree", "explicit_polyline", "custom_walk_program") else 1.4)
        if highlight is not None:
            xyz = np.asarray(highlight["xyz_um"]).T
            q = grid_map(xyz, rendered["grid"])
            ax.plot(q[a] * SPACING_MM, q[b] * SPACING_MM, color="#16a085", lw=1.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("xyz"[a] + " (mm, volume frame)")
        ax.set_ylabel("xyz"[b] + " (mm, volume frame)")
        ax.set_aspect("equal")
        bar = AnchoredSizeBar(ax.transData, 1.0, "1 mm", "lower right", pad=0.4, sep=3, frameon=True,
                              size_vertical=0.04, fontproperties=FontProperties(size=9))
        bar.patch.set_alpha(0.85)
        ax.add_artist(bar)
    handles = [Patch(facecolor="#9a9a9a", edgecolor="none", label="vessel mask (max projection)")]
    for name in ("grammar_walk_tree", "explicit_polyline", "custom_walk_program"):
        if name in used:
            handles.append(Line2D([], [], color=colours[name], lw=1.2, label=labels[name]))
            break
    for name in ("controlled_thin_daughter", "controlled_companion", "anastomosis_bridge"):
        if name in used:
            handles.append(Line2D([], [], color=colours[name], lw=1.6, label=labels[name]))
    if highlight is not None:
        handles.append(Line2D([], [], color="#16a085", lw=1.6, label="approached (target) branch"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    topo = m.get("rendered_topology", {})
    fig.suptitle(f"{spec['id']} {spec['name']}  |  level {spec['level']}  |  SYNTHETIC GEOMETRY (V-System)\n"
                 f"root {spec['root_diameter_um']:.0f} um, d {m['diameters']['min_um']:.0f}-"
                 f"{m['diameters']['max_um']:.0f} um, {len(graph_info['branches'])} branches, "
                 f"graph cycles {graph_info['betti']['cycles']}, mask b0/b1/b2 "
                 f"{topo.get('components')}/{topo.get('cycles')}/{topo.get('cavities')}, "
                 f"voxel 50 um, shape {list(shape)}", fontsize=11)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def grid_map(xyz, grid):
    return (xyz - grid["lo_um"][:, None]) * SCALE + grid["offset_vox"][:, None]


def save_archive(path, nodes, program, record, graph_info, tree):
    V.main.save_network(path, nodes, program=program, metadata=record, edges=graph_info["edges"],
                        node_kind=graph_info["node_kind"], tree=tree)


# ==============================================================================================
# Adapter verification
# ==============================================================================================
def adapter_self_tests():
    """Small known cases for the conversion, graph, topology and loop-count machinery."""
    results = []
    # straight line: coordinates, spacing, graph
    u = unit((1.0, 0.62, 0.38))
    line = assemble([straight_polyline(np.zeros(3), 1000.0 * u, 200.0, EXPLICIT_STEP_UM)])
    grid = choose_grid(line)
    mask, ref, points, radii = render(line, grid)
    p_end = 1000.0 * u
    q_end = grid["T"] @ np.append(p_end, 1.0)
    back_mm = grid["affine"] @ q_end
    g = analyse_graph(line)
    ok = (np.allclose(points[:, -1], q_end[:3], atol=1e-9) and np.allclose(back_mm[:3], p_end / 1000.0, atol=1e-12)
          and g["betti"]["components"] == 1 and g["betti"]["cycles"] == 0
          and g["tips"] == 2 and len(g["branches"]) == 1
          and ref[tuple(np.floor(q_end[:3] + 0.5).astype(int))] == 1
          and np.allclose(np.diag(grid["affine"])[:3], SPACING_MM))
    results.append(check("known_straight_line", bool(ok), {"tips": g["tips"], "branches": len(g["branches"]),
                                                            "end_voxel": np.floor(q_end[:3] + 0.5).tolist()},
                         "q = p/50 + t, affine maps q back to p/1000 mm, one branch, two tips, 0.05 mm spacing",
                         kind="verification"))
    # Y: junction, three tips, Murray/Zamir helper against libGenerator
    spec = {"root_diameter_um": 250.0}
    y = build_straight_y(spec, PLANS["F02"], 0)
    gy = analyse_graph(y["nodes"], y["origin"])
    V.libGenerator.setProperties({"stochparams": False})
    ref_bif = V.libGenerator.calBifurcation(250.0)
    V.libGenerator.setProperties(None)
    th = zamir_angles(ref_bif["d1"], ref_bif["d2"])
    gridy = choose_grid(y["nodes"])
    masky, refy, _, _ = render(y["nodes"], gridy)
    ok = (gy["tips"] == 3 and gy["junctions"] == 1 and len(gy["branches"]) == 3 and gy["betti"]["cycles"] == 0
          and gy["betti"]["components"] == 1 and abs(th[0] - ref_bif["th1"]) < 1e-9 and abs(th[1] - ref_bif["th2"]) < 1e-9
          and digital_topology(masky)["cycles"] == 0 and digital_topology(refy)["components"] == 1)
    results.append(check("known_y", bool(ok), {"tips": gy["tips"], "junctions": gy["junctions"],
                                                "branches": len(gy["branches"]),
                                                "zamir_vs_libGenerator_deg": [th[0] - ref_bif["th1"], th[1] - ref_bif["th2"]]},
                         "one degree-3 junction, three tips, three branches; Zamir angles equal libGenerator's",
                         kind="verification"))
    # loop count on known open and closed paths, graph and digital
    square = np.array([[0, 0, 0], [1000, 0, 0], [1000, 1000, 0], [0, 1000, 0]], dtype=float)
    open_rows = np.column_stack([square, np.full(4, 100.0)])
    closed_rows = np.vstack([open_rows, open_rows[:1]])
    g_open = V.graph.betti(*_edges_canonical(assemble([open_rows])))
    g_closed = V.graph.betti(*_edges_canonical(assemble([closed_rows])))
    ring = np.zeros((9, 9, 5), dtype=np.uint8)
    ring[2:7, 2:7, 2] = 1
    ring[3:6, 3:6, 2] = 0
    open_path = ring.copy()
    open_path[2, 4, 2] = 0
    block = np.zeros((6, 6, 6), dtype=np.uint8)
    block[2:4, 2:4, 2:4] = 1
    shell = np.zeros((7, 7, 7), dtype=np.uint8)
    shell[1:6, 1:6, 1:6] = 1
    shell[2:5, 2:5, 2:5] = 0
    diag = np.zeros((6, 6, 6), dtype=np.uint8)
    diag[1, 1, 1] = diag[2, 2, 2] = diag[3, 3, 3] = 1
    t_ring, t_open, t_block, t_shell, t_diag = (digital_topology(v) for v in (ring, open_path, block, shell, diag))
    ok = (g_open["cycles"] == 0 and g_closed["cycles"] == 1 and t_ring["cycles"] == 1 and t_open["cycles"] == 0
          and t_block["cycles"] == 0 and t_shell["cavities"] == 1 and t_shell["cycles"] == 0
          and t_diag["components"] == 1 and t_diag["cycles"] == 0
          and all(t["euler"] == t["euler_skimage"] for t in (t_ring, t_open, t_block, t_shell, t_diag)))
    results.append(check("known_loop_counts", bool(ok),
                         {"graph_open_cycles": g_open["cycles"], "graph_closed_cycles": g_closed["cycles"],
                          "voxel_ring": t_ring, "voxel_open_path": t_open, "voxel_block": t_block,
                          "voxel_shell": t_shell, "voxel_diagonal_chain": t_diag},
                         "open path b1 = 0, closed path b1 = 1 (graph and voxels); 2x2x2 block b1 = 0; hollow "
                         "shell b2 = 1; corner-touching chain is one 26-component; chi equals skimage",
                         kind="verification"))
    return results


def _edges_canonical(nodes):
    built_graph = V.graph.build(nodes)
    return built_graph["edges"], built_graph["canonical"]


def verify_archive_roundtrip(fixture_dir, nodes, graph_info, tree, program, grid, unbridged=None, reference=None):
    """
    Reload the archive and the NIfTI mask and check exact equality and exact re-rendering.
    The 6-connected reference is read from reference_skeleton_6_con.nii.gz when it was
    written, otherwise `reference` (the in-memory trace) is used.
    """
    import nibabel as nib
    archive = V.main.load_network(os.path.join(fixture_dir, "reference_network.npz"))
    equal = (np.array_equal(archive["nodes"], nodes, equal_nan=True)
             and np.array_equal(archive["edges"], graph_info["edges"])
             and np.array_equal(archive["node_kind"], graph_info["node_kind"])
             and np.array_equal(archive["tree"], tree) and archive["program"] == (program if program else None))
    mask_disk = np.asanyarray(nib.load(os.path.join(fixture_dir, "vessel_mask.nii.gz")).dataobj)
    six = os.path.join(fixture_dir, "reference_skeleton_6_con.nii.gz")
    ref_disk = np.asanyarray(nib.load(six).dataobj) if os.path.exists(six) else reference
    shape = tuple(int(v) for v in grid["shape"])
    rerender = V.computeVoxel.process_network(archive["nodes"], shape, fit="voxel_size", voxel_size=VOXEL_UM,
                                              connect=True)
    _, ref_again, _, _ = render(archive["nodes"], grid)
    result = {"archive_arrays_equal": bool(equal),
              "process_network_rerender_equals_mask": bool(np.array_equal(rerender, mask_disk)),
              "reference_retrace_equals_reference": bool(np.array_equal(ref_again, ref_disk))}
    if unbridged is not None:
        u_dir = os.path.join(fixture_dir, "unbridged")
        u_archive = V.main.load_network(os.path.join(u_dir, "reference_network.npz"))
        points = map_points(u_archive["nodes"], grid)
        radii = u_archive["nodes"][3] / 2.0 * SCALE
        u_mask, _, _, _ = render(u_archive["nodes"], grid, points=points, radii=radii)
        u_disk = np.asanyarray(nib.load(os.path.join(u_dir, "vessel_mask.nii.gz")).dataobj)
        result["unbridged_rerender_with_fixture_grid_equals_mask"] = bool(np.array_equal(u_mask, u_disk))
    return result



# ==============================================================================================
# Fixture loop
# ==============================================================================================
def fixture_specs(manifest):
    specs = []
    for item in manifest["fixtures"]:
        spec = dict(item)
        spec["persistence"] = item.get("tortuosity_persistence_initial")
        specs.append(spec)
    return specs


PREFERENCES = {
    fid: ("at least one near-floor (50-75 um) terminal branch",
          lambda m: bool(m.get("near_floor_terminals"))) for fid in ("F09", "F10", "F11", "F12")
}
SELECTION_RULE = ("candidates are tried in index order (seed = base_seed + 1000 * index); the first that passes every "
                  "hard and selection check and also meets the fixture's preference (if any) is selected; if no "
                  "candidate within the budget meets the preference, the first passing candidate is selected and "
                  "the unmet preference is reported")


def _summary(m):
    return {"branches": len(m["graph"]["branches"]), "tips": m["graph"]["tips"],
            "junctions": m["graph"]["junctions"], "cycles": m["graph"]["betti"]["cycles"]}


def run_fixture(spec, budget, log):
    """Tries candidates in index order; returns (selected dict or None, history)."""
    plan = PLANS[spec["id"]]
    preference = PREFERENCES.get(spec["id"])
    history = []
    fallback = None
    explicit = plan["builder"] in ("straight_tube", "straight_y")
    for index in range(1 if explicit else budget):
        seed = int(spec["base_seed"]) + 1000 * index
        start = time.time()
        candidate, failure = build(spec, plan, seed)
        if candidate is None:
            history.append({"candidate_index": index, "seed": seed, "passed": False, "selected": False,
                            "failed": [{"check": "construction", "measured": failure["reason"]}]})
            log(f"  {spec['id']} candidate {index} (seed {seed}): construction failed: {failure['reason']}")
            continue
        ok, records, m, rendered = evaluate(spec, plan, candidate)
        bad = [{"check": r["check"], "measured": r["measured"]} for r in failed(records)
               if r["check"] != "near_floor_terminals_sought"]
        entry = {"candidate_index": index, "seed": seed, "passed": ok, "selected": False, "failed": bad,
                 **_summary(m), "seconds": round(time.time() - start, 1)}
        history.append(entry)
        preferred = ok and (preference is None or preference[1](m))
        if preference is not None and ok:
            entry["preference"] = preference[0]
            entry["preference_met"] = bool(preferred)
        log(f"  {spec['id']} candidate {index} (seed {seed}): {'pass' if ok else 'reject'} "
            f"branches={len(m['graph']['branches'])} cycles={m['graph']['betti']['cycles']}"
            f"{'' if ok else ' ' + str([b['check'] for b in bad])}"
            f"{' preference unmet' if ok and not preferred else ''} ({time.time() - start:.1f}s)")
        if not ok:
            continue
        chosen = {"candidate": candidate, "records": records, "m": m, "rendered": rendered, "seed": seed,
                  "index": index, "plan": plan, "history_entry": entry}
        if preferred:
            entry["selected"] = True
            chosen["selection"] = {"rule": SELECTION_RULE, "preference": preference and preference[0],
                                   "preference_met": True if preference else None}
            return chosen, history
        if fallback is None:
            fallback = chosen
    if fallback is not None:
        fallback["history_entry"]["selected"] = True
        fallback["selection"] = {"rule": SELECTION_RULE, "preference": preference[0], "preference_met": False}
        log(f"  {spec['id']}: preference unmet within the budget; selecting candidate {fallback['index']}")
        return fallback, history
    return None, history


def run_pair(spec6, spec7, budget, log):
    """
    F06 and F07 share the candidate index, seed and grammar string; F07 must be more tortuous.
    Preference: both realise the same topology (branch, tip and junction counts).
    """
    history = {"F06": [], "F07": []}
    fallback = None
    for index in range(budget):
        out = {}
        for spec in (spec6, spec7):
            plan = PLANS[spec["id"]]
            seed = int(spec["base_seed"]) + 1000 * index
            start = time.time()
            candidate, failure = build(spec, plan, seed)
            ok, records, m, rendered = evaluate(spec, plan, candidate)
            bad = [{"check": r["check"], "measured": r["measured"]} for r in failed(records)]
            entry = {"candidate_index": index, "seed": seed, "passed": ok, "selected": False, "failed": bad,
                     **_summary(m), "seconds": round(time.time() - start, 1)}
            history[spec["id"]].append(entry)
            log(f"  {spec['id']} candidate {index} (seed {seed}): {'pass' if ok else 'reject'} "
                f"branches={len(m['graph']['branches'])}{'' if ok else ' ' + str([b['check'] for b in bad])}")
            out[spec["id"]] = {"candidate": candidate, "records": records, "m": m, "rendered": rendered,
                               "seed": seed, "index": index, "plan": plan, "ok": ok, "history_entry": entry}
            if not ok:
                break
        if len(out) < 2 or not (out["F06"]["ok"] and out["F07"]["ok"]):
            continue
        t6 = out["F06"]["m"]["tortuosity"]
        t7 = out["F07"]["m"]["tortuosity"]
        same_program = out["F06"]["candidate"]["program"] == out["F07"]["candidate"]["program"]
        more = (t7["arc_chord_length_weighted"] > t6["arc_chord_length_weighted"]
                and t7["arc_chord_median"] > t6["arc_chord_median"])
        topo6, topo7 = _summary(out["F06"]["m"]), _summary(out["F07"]["m"])
        same_topology = topo6 == topo7
        pair = {"same_candidate_index": index, "same_branching_specification": same_program,
                "F06_arc_chord_length_weighted": t6["arc_chord_length_weighted"],
                "F07_arc_chord_length_weighted": t7["arc_chord_length_weighted"],
                "F06_arc_chord_median": t6["arc_chord_median"], "F07_arc_chord_median": t7["arc_chord_median"],
                "F06_curvature_per_um": out["F06"]["m"]["describe"]["curvature_per_um"]["mean_abs"],
                "F07_curvature_per_um": out["F07"]["m"]["describe"]["curvature_per_um"]["mean_abs"],
                "F06_topology": topo6, "F07_topology": topo7, "same_realised_topology": same_topology,
                "F06_events": out["F06"]["candidate"]["events"], "F07_events": out["F07"]["candidate"]["events"]}
        rec = check("F07_more_tortuous_than_F06", more and same_program, pair,
                    "same candidate index and branching specification; F07 length-weighted and median arc/chord "
                    "exceed F06's", kind="desired",
                    note=None if same_topology else "collision avoidance and stub pruning changed the realised "
                                                    "topology between the two; the comparison is approximate")
        log(f"  F06/F07 candidate {index}: pair {'pass' if rec['passed'] else 'fail'} (arc/chord "
            f"{t6['arc_chord_length_weighted']:.3f} vs {t7['arc_chord_length_weighted']:.3f}; topology "
            f"{'same' if same_topology else 'differs'})")
        for fid in ("F06", "F07"):
            out[fid]["history_entry"]["pair_check"] = rec
        if not rec["passed"]:
            continue
        for fid in ("F06", "F07"):
            out[fid]["records"].append(rec)
        if same_topology:
            for fid in ("F06", "F07"):
                out[fid]["history_entry"]["selected"] = True
                out[fid]["selection"] = {"rule": SELECTION_RULE, "preference": "same realised topology",
                                         "preference_met": True}
            return out, history
        if fallback is None:
            fallback = out
    if fallback is not None:
        for fid in ("F06", "F07"):
            fallback[fid]["history_entry"]["selected"] = True
            fallback[fid]["selection"] = {"rule": SELECTION_RULE, "preference": "same realised topology",
                                          "preference_met": False}
        log(f"  F06/F07: same realised topology not found within the budget; selecting candidate "
            f"{fallback['F06']['index']}")
        return fallback, history
    return None, history
