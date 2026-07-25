import ast
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[2]
SOURCE_PATH = MODULE_DIR / "MorphoWeaveAtlasBuilder.py"


def _safe_name_function():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    logic_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MorphoWeaveAtlasBuilderLogic"
    )
    function = next(
        node for node in logic_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "isSafeLibraryName"
    )
    function.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace["isSafeLibraryName"]


def _widget_method(name):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    widget_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MorphoWeaveAtlasBuilderWidget"
    )
    function = next(
        node for node in widget_class.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"os": os, "datetime": datetime}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace[name]


def _sampling_functions():
    names = {
        "voxel_representative_indices",
        "farthest_point_subset_indices",
        "sample_indices_voxel_target_points",
    }
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return tuple(namespace[name] for name in (
        "voxel_representative_indices",
        "farthest_point_subset_indices",
        "sample_indices_voxel_target_points",
    ))


class AtlasBuilderWorkflowUnitTest(unittest.TestCase):
    def test_library_names_are_portable_and_path_safe(self):
        is_safe = _safe_name_function()
        self.assertTrue(is_safe("mouse_atlas"))
        self.assertTrue(is_safe("Mouse Atlas 2026"))
        for invalid in ("", ".", "..", "../atlas", "atlas/model", "atlas\\model", "CON", "name."):
            self.assertFalse(is_safe(invalid), invalid)

    def test_overwrite_gate_precedes_output_creation(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        on_run = source.split("  def _onRun(self):", 1)[1].split(
            "  def _saveSsmToLibrary", 1
        )[0]
        self.assertLess(
            on_run.index("confirmOkCancelDisplay"),
            on_run.index("F = self._outFolders"),
        )
        self.assertIn("if dense_ok and saveToLibrary:", on_run)

    def test_configured_library_path_supports_current_default_and_legacy_keys(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        self.assertIn('"MorphoWeaveModelLibrary/databasePath"', source)
        self.assertIn('"DATABASE/databasePath"', source)
        self.assertIn('"MorphoWeaveModels"', source)

    def test_aligned_outputs_can_be_routed_outside_final_output(self):
        out_folders = _widget_method("_outFolders")
        with tempfile.TemporaryDirectory() as output_root, tempfile.TemporaryDirectory() as aligned_root:
            folders = out_folders(object(), output_root, aligned_root)
            self.assertEqual(Path(folders["alignedModels"]).parent, Path(aligned_root))
            self.assertEqual(Path(folders["alignedLMs"]).parent, Path(aligned_root))
            self.assertEqual(Path(folders["atlas"]).parent, Path(folders["output"]))
            self.assertFalse((Path(folders["output"]) / "alignedModels").exists())

    def test_aligned_output_retention_is_default_and_temp_workspace_is_cleaned(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        self.assertIn('self.keepAlignedOutputs.setChecked(True)', source)
        self.assertIn('tempfile.TemporaryDirectory(prefix="MorphoWeave-aligned-")', source)
        self.assertIn("if alignedWorkspace is not None:", source)
        self.assertIn("alignedWorkspace.cleanup()", source)
        self.assertIn("if keepAlignedOutputs:", source)

    def test_voxel_target_sampling_is_exact_deterministic_and_index_stable(self):
        _, _, sample = _sampling_functions()
        rng = np.random.default_rng(23)
        points = rng.normal(size=(5000, 3)) * np.array([3.0, 1.2, 0.35])
        points[:, 2] += 0.08 * points[:, 0] ** 2

        first = sample(points, target_count=300)
        second = sample(points, target_count=300)

        self.assertEqual(len(first), 300)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, np.sort(first))
        self.assertEqual(len(np.unique(first)), len(first))
        self.assertGreaterEqual(first.min(), 0)
        self.assertLess(first.max(), len(points))

    def test_voxel_target_sampling_uses_all_available_points_for_small_meshes(self):
        _, _, sample = _sampling_functions()
        points = np.arange(90, dtype=float).reshape(30, 3)
        np.testing.assert_array_equal(sample(points, target_count=100), np.arange(30))

    def test_voxel_target_sampling_rejects_invalid_inputs(self):
        _, _, sample = _sampling_functions()
        with self.assertRaises(ValueError):
            sample(np.zeros((10, 3)), target_count=0)
        with self.assertRaises(ValueError):
            sample(np.zeros((10, 2)), target_count=5)
        points = np.zeros((10, 3)); points[0, 0] = np.nan
        with self.assertRaises(ValueError):
            sample(points, target_count=5)

    def test_atlas_builder_ui_and_export_use_target_count_sampling(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        self.assertIn("self.targetPointCount.value = 2000", source)
        self.assertIn('advLay.addRow("Target dense points:", self.targetPointCount)', source)
        self.assertIn("self.logic.previewCountForTarget(pd, target)", source)
        self.assertIn("self.sample_indices_voxel_target(", source)
        self.assertIn("def _bbox_diag(self, pd):", source)
        self.assertNotIn("sample_indices_poisson", source)
        self.assertNotIn("Sampling radius (% of diag)", source)


if __name__ == "__main__":
    unittest.main()
