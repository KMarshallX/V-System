"""Tests for the NIfTI reference and GraphML export; all files use temporary directories."""
import builtins
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import graph
import main
import reference
from computeVoxel import rasterise_segments


HAVE_EXTRAS = all(importlib.util.find_spec(name) is not None for name in ("nibabel", "scipy", "igraph"))


def line_nodes():
    """Return an oblique tube axis in micrometres with nonzero world origin."""
    return np.array([[100, 500], [200, 450], [-300, -150], [150, 150]], dtype=float)


def rendered_line():
    """Render a small, contained tube through the production coordinate mapping."""
    nodes = line_nodes()
    points, transform, affine = reference.rendering_grid(nodes, (20, 22, 24), 50, "um")
    mask = rasterise_segments(points, nodes[3] / 100, (20, 22, 24))
    return nodes, points, transform, affine, mask


class OptionalDependencyTests(unittest.TestCase):
    def test_missing_packages_explain_install(self):
        with mock.patch.object(reference.importlib, "import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(ValueError, "pip install"):
                reference.check_options("voxel_size", 50, "um")

    def test_prerequisites_fail_before_generation(self):
        variants = [["--voxel-size", "nan"], ["--voxel-size", "-1"], ["--units", "unknown"],
                    ["--fit", "isotropic"], ["--fit", "stretch"], ["--clip-axes", "z"],
                    ["--construction", "walk_program", "--persistence", "8", "--thin-daughters", "2"],
                    ["--write-unbridged"], ["--keep-bridges", "3", "1"], ["--candidates", "0"],
                    ["--lengths", "100"], ["--original_export", "--thin-daughters", "2"],
                    ["--original_export", "--require-branches", "1", "3"], ["--fixture", "F01", "--d0", "300", "0"]]
        for args in variants:
            with self.subTest(args=args), tempfile.TemporaryDirectory() as directory:
                with mock.patch.object(main, "grow_network") as grow:
                    with self.assertRaises(SystemExit):
                        main.main(["--out", directory] + args)
                    grow.assert_not_called()
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_original_export_does_not_import_exporter_or_extras(self):
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("reference", "nibabel", "scipy"):
                raise AssertionError(f"Unexpected optional import: {name}")
            return original(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("builtins.__import__", side_effect=guarded):
                self.assertEqual(main.main(["--original_export", "--count", "0", "--out", directory]), 0)


@unittest.skipUnless(HAVE_EXTRAS, "install nibabel, scipy and python-igraph to test reference export")
class ReferenceTests(unittest.TestCase):
    def test_oblique_path_preserves_endpoints_and_uses_diagonals(self):
        points = np.array([[1., 1., 1.], [9., 7., 5.]])
        path = reference.thin_path(points)
        np.testing.assert_array_equal(path[[0, -1]], points.astype(int))
        self.assertEqual(len(path), 9)
        self.assertTrue(np.any(np.abs(np.diff(path, axis=0)).sum(axis=1) > 1))

    def test_edt_selects_deeper_equal_count_route(self):
        from scipy import ndimage
        points = np.array([[2, 2, 3], [3, 2, 3], [3, 3, 3],
                           [4, 3, 3], [4, 4, 3], [5, 4, 3]], dtype=float)
        mask = np.zeros((9, 9, 7), dtype=bool)
        mask[1:8, 2:7, 1:6] = True
        distance = ndimage.distance_transform_edt(mask, sampling=(.05, .05, .05))
        old = reference.thin_path(points)
        new = reference.thin_path(points, distance)
        np.testing.assert_array_equal(new, [[2, 2, 3], [3, 3, 3], [4, 4, 3], [5, 4, 3]])
        self.assertEqual(len(old), len(new))
        self.assertGreater(distance[tuple(new.T)].sum(), distance[tuple(old.T)].sum())

    def test_topology_rejection_restores_shared_voxel_counts(self):
        lower = np.array([[x, 1, 1] for x in range(1, 8)])
        old = np.array([[1, 1, 1], [1, 2, 1], [1, 3, 1], [2, 4, 1],
                        [3, 4, 1], [4, 4, 1], [5, 4, 1], [6, 4, 1],
                        [7, 3, 1], [7, 2, 1], [7, 1, 1]])
        new = old.copy()
        new[4:7, 1] = [3, 2, 3]
        counts = np.zeros((10, 7, 3), dtype=np.int32)
        for path in (lower, old):
            np.add.at(counts, tuple(path.T), 1)
        before = counts.copy()
        self.assertFalse(reference.accept_centered_path(counts, old, new, (1, 1, 0)))
        np.testing.assert_array_equal(counts, before)

    def test_grid_preserves_source_physical_coordinates_and_unit_conversion(self):
        import nibabel as nib
        nodes, points, transform, affine, _ = rendered_line()
        np.testing.assert_allclose(nib.affines.apply_affine(affine, points.T), nodes[:3].T / 1000,
                                   rtol=0, atol=1e-15)
        np.testing.assert_allclose(nib.affines.apply_affine(transform, nodes[:3].T), points.T)
        mm_nodes = nodes / 1000
        mm_points, _, mm_affine = reference.rendering_grid(mm_nodes, (20, 22, 24), .05, "mm")
        np.testing.assert_allclose(mm_points, points)
        np.testing.assert_allclose(mm_affine, affine)

    def test_branch_paths_reconstruct_both_arrays(self):
        nodes, points, _, affine, mask = rendered_line()
        arrays, report = reference.build_references(nodes, graph.build(nodes)["edges"], mask, points, affine)
        for kind, field in [("uncentered", "uncentered_voxel_path_xyz"), ("centered", "voxel_path_xyz")]:
            reconstructed = np.zeros(mask.shape, dtype=np.uint8)
            for branch in report["branches"]:
                path = np.array(branch[field])
                reconstructed[tuple(path.T)] = 1
            np.testing.assert_array_equal(arrays[kind], reconstructed)
            self.assertEqual(reference.topology(arrays[kind]), (1, 0, 0))

    def test_y_and_closed_cycle_have_expected_topology(self):
        y = np.array([[5, 10, 10, np.nan, 10, 15],
                      [5, 10, 15, np.nan, 10, 5],
                      [10, 10, 10, np.nan, 10, 10],
                      [2, 2, 2, np.nan, 2, 2]], dtype=float)
        t = np.linspace(0, 2 * np.pi, 100)
        ring = np.vstack((12 + 7 * np.cos(t), 12 + 7 * np.sin(t), 0 * t + 10, 0 * t + 2))
        ring[:, -1] = ring[:, 0]
        for nodes, expected, branch_count in [(y, (1, 0, 0), 3), (ring, (1, 1, 0), 1)]:
            with self.subTest(expected=expected):
                points = nodes[:3]
                mask = rasterise_segments(points, nodes[3] / 2, (26, 26, 22))
                arrays, report = reference.build_references(nodes, graph.build(nodes)["edges"], mask,
                                                            points, np.diag([.05, .05, .05, 1]))
                self.assertEqual(reference.topology(arrays["centered"]), expected)
                self.assertEqual(len(report["branches"]), branch_count)

    def test_clipped_geometry_and_missing_foreground_fail(self):
        nodes, points, _, affine, mask = rendered_line()
        edges = graph.build(nodes)["edges"]
        outside = points.copy()
        outside[0, -1] = mask.shape[0]
        with self.assertRaisesRegex(ValueError, "field of view"):
            reference.build_references(nodes, edges, mask, outside, affine)
        holes = mask.copy()
        holes[tuple(reference.thin_path(points.T)[1])] = 0
        with self.assertRaisesRegex(ValueError, "foreground"):
            reference.build_references(nodes, edges, holes, points, affine)

    def test_baseline_with_wrong_topology_fails(self):
        nodes = np.array([[5, 15, np.nan, 5, 15], [5, 15, np.nan, 15, 5],
                          [10, 10, np.nan, 10, 10], [2, 2, np.nan, 2, 2]], dtype=float)
        mask = rasterise_segments(nodes[:3], nodes[3] / 2, (22, 22, 22))
        with self.assertRaisesRegex(ValueError, "topology"):
            reference.build_references(nodes, graph.build(nodes)["edges"], mask, nodes[:3], np.eye(4))

    def test_outputs_share_grid_values_and_verified_hashes(self):
        import nibabel as nib
        nodes, _, _, affine, mask = rendered_line()
        with tempfile.TemporaryDirectory() as directory:
            report = reference.export_nifti(directory, "case", nodes, graph.build(nodes)["edges"], mask, 50, "um")
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                             sorted(reference.stem_names("case").values()))
            images = []
            for kind, output in report["outputs"].items():
                path = Path(directory) / output["file"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), output["sha256"])
                if kind == "graphml":
                    continue
                image = nib.load(path)
                images.append(image)
                np.testing.assert_array_equal(image.affine, report["affine_voxel_to_mm"])
                self.assertEqual(image.header.get_xyzt_units()[0], "mm")
                self.assertEqual(image.get_data_dtype(), np.dtype("uint8"))
                np.testing.assert_allclose(image.affine, affine, rtol=1e-6, atol=1e-7)
                if kind == "mask":
                    np.testing.assert_array_equal(np.asarray(image.dataobj), mask)
            for image in images[1:]:
                np.testing.assert_array_equal(image.affine, images[0].affine)
                np.testing.assert_array_equal(image.get_qform(), images[0].get_qform())
                np.testing.assert_array_equal(image.get_sform(), images[0].get_sform())
            before = {p.name: p.read_bytes() for p in Path(directory).iterdir()}
            with self.assertRaises(FileExistsError):
                reference.export_nifti(directory, "case", nodes, graph.build(nodes)["edges"], mask, 50, "um")
            self.assertEqual(before, {p.name: p.read_bytes() for p in Path(directory).iterdir()})

    def test_roundtrip_graphml_or_publication_failure_leaves_no_outputs(self):
        nodes, _, _, _, mask = rendered_line()
        for stage in ("roundtrip", "graphml", "publish"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                if stage == "roundtrip":
                    target = mock.patch.object(reference, "_save_checked", side_effect=ValueError("bad header"))
                elif stage == "graphml":
                    target = mock.patch.object(reference, "write_graphml", side_effect=ValueError("bad graph"))
                else:
                    original = reference.shutil.copyfileobj
                    calls = []

                    def fail_second(source, destination):
                        calls.append(1)
                        if len(calls) == 2:
                            raise OSError("write failed")
                        original(source, destination)

                    target = mock.patch.object(reference.shutil, "copyfileobj", side_effect=fail_second)
                with target, self.assertRaises((ValueError, OSError)):
                    reference.export_nifti(directory, "case", nodes, graph.build(nodes)["edges"], mask, 50, "um")
                self.assertEqual(list(Path(directory).iterdir()), [])


@unittest.skipUnless(HAVE_EXTRAS, "install nibabel, scipy and python-igraph to test GraphML export")
class GraphmlTests(unittest.TestCase):
    def export(self, nodes, shape, directory):
        points, radii = nodes[:3], nodes[3] / 2
        mask = rasterise_segments(points, radii, shape)
        return reference.export_references(directory, reference.stem_names("g"), nodes, graph.build(nodes)["edges"],
                                           mask, points, np.diag([.05, .05, .05, 1]), attributes={"network_id": "g"})

    def read(self, directory):
        import igraph
        return igraph.Graph.Read_GraphML(str(Path(directory) / "g_reference_graph.graphml"))

    def test_y_graph_nodes_edges_paths_and_world_coordinates(self):
        import nibabel as nib
        y = np.array([[5, 10, 10, np.nan, 10, 15],
                      [5, 10, 15, np.nan, 10, 5],
                      [10, 10, 10, np.nan, 10, 10],
                      [2, 2, 2, np.nan, 2, 2]], dtype=float)
        with tempfile.TemporaryDirectory() as directory:
            report = self.export(y, (26, 26, 22), directory)
            g = self.read(directory)
            self.assertEqual((g.vcount(), g.ecount()), (4, 3))
            self.assertEqual(sorted(g.vs["kind"]), ["junction", "tip", "tip", "tip"])
            self.assertEqual(g.es["branch_id"], ["B0000", "B0001", "B0002"])
            self.assertEqual(report["graphml"]["cycles"], 0)
            centred = np.asarray(nib.load(Path(directory) / "g_reference_26_con_centered.nii.gz").dataobj)
            union = np.zeros_like(centred)
            for edge in g.es:
                path = np.array(json.loads(edge["centerline_voxels"]))
                self.assertEqual(len(path), edge["num_centerline_voxels"])
                # every path runs from the edge's source node to its target node
                for end, vertex in ((path[0], g.vs[edge.source]), (path[-1], g.vs[edge.target])):
                    np.testing.assert_array_equal(end, json.loads(vertex["voxels"])[0])
                union[tuple(path.T)] = 1
                world = np.array(json.loads(edge["centerline_world_points"]))
                voxel = np.array(json.loads(edge["centerline_voxel_points"]))
                np.testing.assert_allclose(world, voxel * 0.05)
            np.testing.assert_array_equal(union, centred)
            self.assertEqual(g["network_id"], "g")
            self.assertEqual(json.loads(g["volume_shape"]), [26, 26, 22])
            self.assertEqual(g["centred_reference_sha256"], report["outputs"]["26_con_centered"]["sha256"])

    def test_ring_is_one_closed_branch_on_its_anchor(self):
        t = np.linspace(0, 2 * np.pi, 100)
        ring = np.vstack((12 + 7 * np.cos(t), 12 + 7 * np.sin(t), 0 * t + 10, 0 * t + 2))
        ring[:, -1] = ring[:, 0]
        with tempfile.TemporaryDirectory() as directory:
            report = self.export(ring, (26, 26, 22), directory)
            g = self.read(directory)
            self.assertEqual((g.vcount(), g.ecount()), (1, 1))
            self.assertEqual(g.vs["kind"], ["ring_anchor"])
            self.assertTrue(g.es[0].is_loop())
            self.assertEqual(report["graphml"]["cycles"], 1)

    def test_grid_holds_network_with_border_and_mm_affine(self):
        nodes = np.array([[0, 2000, np.nan, 2000, 2500], [0, 1500, np.nan, 1500, 3000],
                          [0, -500, np.nan, -500, -400], [300, 200, np.nan, 200, 60]], dtype=float)
        shape, transform, affine = reference.choose_grid(nodes, 50, "um")
        points, radii = main.fit_to_volume(nodes, shape, fit="voxel_size", voxel_size=50)
        finite = np.isfinite(points[0])
        np.testing.assert_allclose((transform[:3, :3] @ nodes[:3, finite] + transform[:3, 3:]), points[:, finite])
        np.testing.assert_allclose(affine[:3, :3] @ points[:, finite] + affine[:3, 3:], nodes[:3, finite] / 1000)
        mask = rasterise_segments(points, radii, shape)
        occupied = np.argwhere(mask)
        self.assertGreaterEqual(occupied.min(), 4)
        self.assertTrue(np.all(np.array(shape) - 1 - occupied.max(axis=0) >= 4))


@unittest.skipUnless(HAVE_EXTRAS, "install nibabel, scipy and python-igraph to test CLI export")
class ExportCliTests(unittest.TestCase):
    # options shared by both modes, so that the two draw the same geometry
    ARGS = ["--count", "1", "--seed", "7", "--iterations", "1", "1", "--d0", "150", "0",
            "--epsilon", "4", "4", "--randmarg", "0.2", "0.2", "--fit", "voxel_size", "--voxel-size", "50",
            "--volume", "64", "64", "64", "--aneurysm-prob", "0", "--stenosis-prob", "0",
            "--tortuosity", "stems", "--subdivisions", "3"]

    def run_main(self, argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main.main(argv)

    def test_default_export_matches_original_geometry_and_mask(self):
        import nibabel as nib
        import tifffile
        with tempfile.TemporaryDirectory() as directory:
            base, export = Path(directory) / "original", Path(directory) / "export"
            self.assertEqual(self.run_main(self.ARGS + ["--original_export", "--out", str(base)]), 0)
            self.assertEqual(self.run_main(self.ARGS + ["--out", str(export)]), 0)
            stem = "Lnet_i1_s7"
            self.assertEqual(sorted(p.name for p in export.iterdir()),
                             sorted([stem + ".json"] + list(reference.stem_names(stem).values())))
            self.assertEqual(sorted(p.suffix for p in base.iterdir()), [".json", ".npz", ".tiff"])
            original = main.load_network(str(base / (stem + ".npz")))
            mask = np.asarray(nib.load(export / (stem + "_vessel_mask.nii.gz")).dataobj)
            np.testing.assert_array_equal(mask, (tifffile.imread(base / (stem + ".tiff")) != 0).transpose(2, 1, 0))
            sidecar = json.loads((export / (stem + ".json")).read_text())
            self.assertEqual(sidecar["reference_export"]["status"], "ok")
            self.assertEqual(sidecar["export"], "reference")
            self.assertEqual(sidecar["axis_order"], "xyz")
            self.assertEqual(sidecar["volume"], [64, 64, 64])
            for key in ("d0", "iterations", "properties"):
                self.assertEqual(sidecar[key], original["metadata"][key])

    def test_auto_grid_uses_v1_tree_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.run_main(["--count", "1", "--seed", "8", "--iterations", "3", "3",
                                            "--out", directory]), 0)
            sidecar = json.loads(next(Path(directory).glob("*.json")).read_text())
            self.assertEqual(sidecar["fit"], "voxel_size")
            self.assertEqual(sidecar["voxel_size"], main.EXPORT_VOXEL_SIZE)
            self.assertEqual(sidecar["clip_axes"], [])
            self.assertEqual(sidecar["d_min"], main.EXPORT_DEFAULTS["d_min"])
            self.assertEqual(sidecar["tortuosity"], "walk")
            self.assertEqual(sidecar["persistence"], main.EXPORT_DEFAULTS["persistence"])
            self.assertTrue(sidecar["avoid_collisions"])
            self.assertEqual(sidecar["subdivisions"], main.EXPORT_DEFAULTS["subdivisions"])
            self.assertTrue(sidecar["prune_stubs"])
            self.assertEqual(sidecar["volume"], sidecar["reference_export"]["shape_xyz"])

    def test_failed_candidates_write_only_a_failure_record(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(reference, "export_references", side_effect=ValueError("invalid topology")):
                status = self.run_main(self.ARGS + ["--out", directory, "--candidates", "2"])
            self.assertEqual(status, 1)
            files = list(Path(directory).iterdir())
            self.assertEqual([p.name for p in files], ["Lnet_failed_s7.json"])
            record = json.loads(files[0].read_text())
            self.assertEqual([c["seed"] for c in record["candidates"]], [7, 1007])
            self.assertTrue(all("invalid topology" in c["reason"] for c in record["candidates"]))

    def test_existing_output_refuses_without_retrying(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Lnet_i1_s7_vessel_mask.nii.gz").write_bytes(b"preserve these bytes")
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            with self.assertRaisesRegex(SystemExit, "Refusing to overwrite"):
                self.run_main(self.ARGS + ["--out", directory])
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})


if __name__ == "__main__":
    unittest.main()
