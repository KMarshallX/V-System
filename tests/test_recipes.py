"""Tests for the fixture recipes, fixtures.py and main.py's construction options (temporary directories)."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import main


HAVE_EXTRAS = all(importlib.util.find_spec(name) is not None
                  for name in ("nibabel", "scipy", "igraph", "skimage", "matplotlib"))

# vsystem_laplacian_50um_v1's published grid shapes and foreground voxel counts
V1 = {"F01": ((62, 44, 32), 801, 1), "F02": ((55, 71, 55), 1647, 3)}


def quiet(function, *args):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return function(*args)


@unittest.skipUnless(HAVE_EXTRAS, "install V-System with the fixtures extra to test fixture recipes")
class FixtureTests(unittest.TestCase):
    def test_straight_controls_match_v1_and_carry_every_output(self):
        import fixtures
        import igraph
        import nibabel as nib
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(quiet(fixtures.main, ["--only", "F01,F02", "--out", directory]), 0)
            report = json.loads((Path(directory) / "validation_report.json").read_text())
            self.assertEqual(report["status"], "partial")
            for fid, (shape, voxels, branches) in V1.items():
                with self.subTest(fixture=fid):
                    case = next(Path(directory).glob(fid + "_*"))
                    names = {p.name for p in case.iterdir()}
                    self.assertTrue({"vessel_mask.nii.gz", "reference_skeleton_26_con.nii.gz",
                                     "reference_skeleton_26_con_centered.nii.gz", "reference_graph.graphml",
                                     "reference_graph.json", "reference_network.npz", "metadata.json",
                                     "preview.png"} <= names)
                    self.assertNotIn("reference_skeleton_6_con.nii.gz", names)
                    mask = np.asarray(nib.load(case / "vessel_mask.nii.gz").dataobj)
                    self.assertEqual(mask.shape, shape)
                    self.assertEqual(int(mask.sum()), voxels)
                    g = igraph.Graph.Read_GraphML(str(case / "reference_graph.graphml"))
                    self.assertEqual(g.ecount(), branches)
                    self.assertEqual(g["fixture_id"], fid)
                    metadata = json.loads((case / "metadata.json").read_text())
                    self.assertTrue(metadata["qa"]["passed"])
                    self.assertEqual(metadata["references"]["graphml"]["edges"], branches)

    def test_main_fixture_option_delegates_and_refuses_other_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(quiet(main.main, ["--fixture", "F01", "--out", directory]), 0)
            self.assertTrue((Path(directory) / "F01_straight_oblique_tube" / "reference_graph.graphml").exists())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SystemExit, "--d0"):
                main.main(["--fixture", "F01", "--d0", "300", "0", "--out", directory])
            with self.assertRaisesRegex(SystemExit, "unknown fixture"):
                quiet(main.main, ["--fixture", "F99", "--out", directory])

    def test_main_straight_tube_construction_draws_the_f01_object(self):
        import nibabel as nib
        with tempfile.TemporaryDirectory() as directory:
            argv = ["--count", "1", "--seed", "16001", "--d0", "200", "0", "--construction", "straight_tube",
                    "--out", directory]
            self.assertEqual(quiet(main.main, argv), 0)
            mask = np.asarray(nib.load(Path(directory) / "Lnet_straight_tube_s16001_vessel_mask.nii.gz").dataobj)
            shape, voxels, _ = V1["F01"]
            self.assertEqual(mask.shape, shape)
            self.assertEqual(int(mask.sum()), voxels)

    def test_split_add_ons_equal_their_fixture_builders(self):
        import recipes
        spec = {"root_diameter_um": 500.0, "persistence": 8}
        plan = recipes.PLANS["F08"]
        built, _ = recipes.build_tree_thin(spec, plan, 16008)
        grown, params = recipes.grow_tree(spec, plan, 16008)
        again, _ = recipes.add_thin_daughters(grown, params, plan, 16008)
        np.testing.assert_array_equal(built["nodes"], again["nodes"])
        self.assertEqual(len(again["extras"]["thin_daughters"]), plan["thin_count"])


if __name__ == "__main__":
    unittest.main()
