"""Production batch orchestration with real VTK/TPS and MRML/I/O boundary doubles."""
import json
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from LogicTestSupport import Node, load_logic, polydata, slicer_boundary


class BatchResultTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.targets = self.root / "targets"; self.targets.mkdir()
        self.output = self.root / "output"
        self.mesh_output = self.root / "meshes"
        self.slicer = slicer_boundary()
        self.logic, _ = load_logic(self.slicer)
        self.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        self.faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        self.pd = polydata(self.vertices, self.faces)
        self.bad_loads = set()
        self.save_mode = "success"
        self.status = []
        self.progress = []
        self.cancel = False
        self.cancel_stage = None
        self.cancel_after_first = False
        self.slicer.util.loadModel = self.load_model
        self.slicer.util.saveNode = self.save_node
        self.logic._get_ssm_base = lambda table: {}
        cloud = SimpleNamespace(points=self.vertices.copy())
        self.logic.runSubsample = lambda *args: (None, None, cloud, cloud, None, None, .1)
        self.logic.estimateTransform = lambda *args: np.eye(4)
        self.logic.runDeformable = lambda **kwargs: kwargs["sourceLM"].copy()
        self.logic.propagateLandmarkTypes = lambda *args: None
        tiny3d = SimpleNamespace(utility=SimpleNamespace(random=SimpleNamespace(seed=lambda seed: None)))
        self.modules = patch.dict("sys.modules", {"tiny3d": tiny3d})
        self.modules.start(); self.addCleanup(self.modules.stop)

    def load_model(self, path):
        if Path(path).name in self.bad_loads:
            raise RuntimeError("Unreadable fixture")
        node = self.slicer.mrmlScene.AddNewNodeByClass("model", "target")
        node.SetAndObservePolyData(polydata(self.vertices, self.faces))
        return node

    def save_node(self, node, path):
        if self.save_mode == "false":
            return False
        if self.save_mode == "true_without_file":
            return True
        Path(path).write_text(json.dumps({"points": node.points.tolist()}))
        return True

    def on_status(self, message):
        self.status.append(message)
        if self.cancel_stage and self.cancel_stage in message:
            self.cancel = True

    def on_progress(self, done, total, label):
        self.progress.append((done, total, label))
        if self.cancel_after_first and done == 1:
            self.cancel = True

    def run_batch(self, filenames=("one.ply",), **kwargs):
        for name in filenames:
            (self.targets / name).write_text("fixture")
        parameters = {"skipOptimization": True}
        parameters.update(kwargs.pop("parameters", {}))
        result = self.logic.runLandmarkBatch(
            Node(self.pd), Node(points=self.vertices), Node(points=self.vertices[:3]),
            str(self.targets), str(self.output), True, kwargs.pop("projectionFactor", 0),
            parameters, object(), meshOutputDir=str(self.mesh_output),
            status_callback=self.on_status, progress_callback=self.on_progress,
            cancel_callback=lambda: self.cancel, **kwargs)
        self.assertEqual(self.slicer.mrmlScene.nodes, [])
        self.assertEqual(self.slicer.mrmlScene.states, [])
        self.assertFalse(self.slicer.app.paused)
        self.assertEqual(result["attempted"], result["succeeded"] + result["failed"] + result["cancelled_count"])
        self.assertEqual(result["total"], result["attempted"] + result["unprocessed"])
        return result

    def test_all_fail_is_reported_without_success_message(self):
        self.bad_loads = {"one.ply", "two.ply"}
        result = self.run_batch(("one.ply", "two.ply"))
        self.assertEqual(result["succeeded"], 0)
        self.assertEqual(result["failed"], 2)
        self.assertEqual(len(result["failures"]), 2)
        self.assertEqual(result["outputs"], [])
        self.assertIn("0 succeeded, 2 failed", self.status[-1])
        self.assertNotIn("Batch complete.", self.status[-1])

    def test_mixed_results_keep_successful_outputs_and_report_failures(self):
        self.bad_loads = {"bad.ply"}
        result = self.run_batch(("bad.ply", "good.ply"))
        self.assertEqual((result["succeeded"], result["failed"]), (1, 1))
        self.assertEqual(result["failures"][0]["target"], "bad.ply")
        self.assertEqual(len(result["outputs"]), 1)
        saved = json.loads(Path(result["outputs"][0]).read_text())
        np.testing.assert_allclose(saved["points"], self.vertices[:3], atol=1e-12)

    def test_false_save_and_missing_output_are_failures(self):
        for mode in ("false", "true_without_file"):
            with self.subTest(mode=mode):
                self.save_mode = mode
                result = self.run_batch()
                self.assertEqual((result["succeeded"], result["failed"]), (0, 1))
                self.assertIn("Failed to save landmarks", result["failures"][0]["error"])
                self.assertEqual(result["outputs"], [])

    def test_mesh_write_failure_is_not_counted_as_specimen_success(self):
        self.logic.savePolyDataVTP = lambda *args: (_ for _ in ()).throw(RuntimeError("writer failure"))
        result = self.run_batch(saveWarpedMeshes=True)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["outputs"], [])
        self.assertIn("writer failure", result["failures"][0]["error"])

    def test_mesh_output_is_kept_when_later_landmark_save_fails(self):
        self.save_mode = "false"
        result = self.run_batch(saveWarpedMeshes=True)
        self.assertEqual((result["succeeded"], result["failed"]), (0, 1))
        self.assertEqual(len(result["outputs"]), 1)
        self.assertTrue(Path(result["outputs"][0]).is_file())
        self.assertEqual(result["failures"][0]["outputs"], result["outputs"])

    def test_cancellation_before_or_between_targets(self):
        self.cancel = True
        result = self.run_batch(("a.ply", "b.ply"))
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["attempted"], 0)
        self.cancel = False; self.cancel_after_first = True
        result = self.run_batch(("a.ply", "b.ply"))
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["succeeded"], 1)
        self.assertEqual(result["attempted"], 1)
        self.assertEqual(result["unprocessed"], 1)

    def test_cancellation_during_deformable_stage(self):
        self.cancel_stage = "PCA-CPD deformable"
        result = self.run_batch()
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["cancelled_count"], 1)
        self.assertEqual(result["succeeded"], 0)
        self.assertEqual(result["outputs"], [])
        self.assertIn("cancelled", self.status[-1])

    def test_cancellation_in_optimizer_status_callback_prevents_optimizer_entry(self):
        calls = []
        def unexpected_optimizer(*args, **kwargs):
            calls.append(True)
            raise AssertionError("Optimizer was entered after cancellation")
        self.logic.initialize_template = unexpected_optimizer
        self.logic.initialize_template_pose_em = unexpected_optimizer
        for backend, stage in (("ransac", "Optimize (SSM+RANSAC)"),
                               ("pose_em", "Pose EM template-shape optimization")):
            with self.subTest(backend=backend):
                self.cancel = False; self.cancel_stage = stage
                result = self.run_batch(parameters={"skipOptimization": False,
                                                    "optimizationBackend": backend})
                self.assertEqual(calls, [])
                self.assertTrue(result["cancelled"])
                self.assertEqual(result["cancelled_count"], 1)
                self.assertEqual(result["failed"], 0)
                self.assertEqual(result["outputs"], [])

    def test_cancellation_before_writing_does_not_emit_that_output(self):
        for stage, save_mesh in (("Save landmarks", False), ("Save warped mesh", True)):
            with self.subTest(stage=stage):
                self.cancel = False; self.cancel_stage = stage
                result = self.run_batch(saveWarpedMeshes=save_mesh)
                self.assertTrue(result["cancelled"])
                self.assertEqual(result["cancelled_count"], 1)
                self.assertEqual(result["outputs"], [])

    def test_setup_failure_restores_scene_and_render_state(self):
        original = self.slicer.mrmlScene.AddNewNodeByClass
        def fail_second(*args):
            if self.slicer.mrmlScene.nodes:
                raise RuntimeError("node creation failure")
            return original(*args)
        self.slicer.mrmlScene.AddNewNodeByClass = fail_second
        result = self.run_batch()
        self.assertIn("node creation failure", result["error"])
        self.assertEqual(result["attempted"], 0)
        self.assertIn("Batch stopped", self.status[-1])

    def test_projection_count_reaches_saved_landmarks(self):
        result = self.run_batch(projectionFactor=.01)
        self.assertEqual(result["succeeded"], 1)
        saved = json.loads(Path(result["outputs"][0]).read_text())
        np.testing.assert_allclose(saved["points"], self.vertices[:3], atol=1e-7)

    def test_empty_batch_returns_result_without_scene_changes(self):
        result = self.run_batch(())
        self.assertEqual(result["total"], 0)
        self.assertIn("No target meshes", self.status[-1])

    def test_duplicate_output_stems_are_rejected_before_processing(self):
        with self.assertRaisesRegex(ValueError, "unique output stems"):
            self.run_batch(("same.ply", "SAME.stl"))
        self.assertEqual(self.slicer.mrmlScene.nodes, [])


if __name__ == "__main__":
    unittest.main()
