"""Adapter contract tests with fake MRML/Qt and a deterministic fake fitter.

These do not claim to validate native rustcpd accuracy or the live Slicer GUI.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import numpy.random
import numpy.testing

MODULE_DIR = Path(__file__).resolve().parents[2]


class Node:
    def __init__(self, node_id, points=None, singleton=False):
        self.node_id = node_id
        self.points = np.ones((3, 3)) if points is None else np.asarray(points, dtype=float)
        self.mtime = 1
        self.singleton = singleton
        self.parent = None

    def GetID(self):
        return self.node_id

    def GetMTime(self):
        return self.mtime

    def GetSingletonTag(self):
        return "singleton" if self.singleton else None

    def GetPolyData(self):
        return self

    def GetNumberOfPoints(self):
        return len(self.points)

    def GetParentTransformNode(self):
        return self.parent


class Hierarchy:
    def __init__(self):
        self.folders = {1}
        self.next_id = 2

    def create(self):
        value = self.next_id
        self.next_id += 1
        self.folders.add(value)
        return value

    def RemoveItem(self, value):
        self.folders.discard(value)


class Scene:
    def __init__(self):
        self.nodes = []
        self.hierarchy = Hierarchy()
        self.next_id = 1

    def add(self, points=None, singleton=False):
        node = Node(str(self.next_id), points, singleton)
        self.next_id += 1
        self.nodes.append(node)
        return node

    def GetNumberOfNodes(self):
        return len(self.nodes)

    def GetNthNode(self, index):
        return self.nodes[index]

    def GetNodeByID(self, node_id):
        return next((node for node in self.nodes if node.GetID() == node_id), None)

    def RemoveNode(self, node):
        self.nodes.remove(node)

    def GetFirstNodeByClass(self, class_name):
        return self.hierarchy if class_name == "vtkMRMLSubjectHierarchyNode" else None


class Selector:
    def __init__(self, node):
        self.node = node

    def currentNode(self):
        return self.node

    def setCurrentNode(self, node):
        self.node = node


class OutputPath:
    def __init__(self, path):
        self.currentPath = str(path)

    def setCurrentPath(self, path):
        self.currentPath = str(path)


class BaseLogic:
    def __init__(self):
        self.ssm_calls = 0
        self.fail_export = False
        self.fail_create = False

    def ssm_from_table(self, node):
        self.ssm_calls += 1
        return np.ones((3, 3)), np.ones((3, 3, 1)), np.ones(1)

    def create_output_folder(self, name):
        return self.scene.hierarchy.create()

    def create_completion_outputs(self):
        folder = self.create_output_folder("output")
        model = self.scene.add()
        self.scene.add()  # display node
        if self.fail_create:
            raise RuntimeError("failed during output creation")
        return {"folder_item": folder, "model": model}

    def save_completion_outputs(self, outputs, diagnostics, directory):
        self.scene.add()  # storage node created on export
        (Path(directory) / "result.json").write_text(json.dumps(diagnostics, sort_keys=True))
        if self.fail_export:
            raise IOError("failed during export")


class BaseWidget:
    def _read_settings(self):
        return self.settings

    def _preview_landmark_match_count(self):
        node = self.target_landmark_selector.currentNode()
        return 0 if node is None else len(node.points)

    def _run_completion_impl(self):
        self.run_count += 1
        self.seen_landmarks.append(self.target_landmark_selector.currentNode())
        if self.add_unrelated_node:
            self.unrelated = self.logic.scene.add()
        arrays = self.logic.ssm_from_table(self.ssm_table_selector.currentNode())
        rng = np.random.default_rng(self.settings.seed)
        points = arrays[0] + self.target_model_selector.currentNode().points + rng.normal(size=(3, 3))
        outputs = self.logic.create_completion_outputs()
        self.logic.save_completion_outputs(
            outputs, {"points": points.tolist(), "settings": vars(self.settings)},
            self.output_directory.currentPath)
        self._run_folder_item = outputs["folder_item"]
        if self.edit_template_during_fit:
            self.template_dense_selector.currentNode().mtime += 1


def fake_run_completion():
    pass


class BatchSlicerAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scene = Scene()
        fake_slicer = types.ModuleType("slicer")
        fake_slicer.mrmlScene = self.scene
        fake_slicer.util = types.SimpleNamespace(loadModel=self.load, loadMarkups=self.load)
        scripted = types.ModuleType("slicer.ScriptedLoadableModule")
        scripted.ScriptedLoadableModule = object
        base_module = types.ModuleType("MorphoWeaveShapeCompletion")
        base_module.MorphoWeaveShapeCompletionLogic = BaseLogic
        base_module.MorphoWeaveShapeCompletionWidget = BaseWidget
        base_module.run_completion = fake_run_completion
        runner_spec = importlib.util.spec_from_file_location(
            "Resources.Python.MorphoWeaveCompletionBatch",
            MODULE_DIR / "Resources/Python/MorphoWeaveCompletionBatch.py")
        self.batch = importlib.util.module_from_spec(runner_spec)
        fake_vtk = types.ModuleType("vtk")
        fake_vtk_util = types.ModuleType("vtk.util")
        fake_vtk_numpy = types.ModuleType("vtk.util.numpy_support")
        fake_vtk_numpy.vtk_to_numpy = np.asarray
        fake_vtk.util = fake_vtk_util
        fake_vtk_util.numpy_support = fake_vtk_numpy
        patches = {
            "vtk": fake_vtk, "vtk.util": fake_vtk_util,
            "vtk.util.numpy_support": fake_vtk_numpy,
            "ctk": types.ModuleType("ctk"), "qt": types.ModuleType("qt"),
            "slicer": fake_slicer, "slicer.ScriptedLoadableModule": scripted,
            "MorphoWeaveShapeCompletion": base_module,
            "Resources": types.ModuleType("Resources"),
            "Resources.Python": types.ModuleType("Resources.Python"),
            runner_spec.name: self.batch,
        }
        patcher = patch.dict(sys.modules, patches)
        patcher.start()
        self.addCleanup(patcher.stop)
        runner_spec.loader.exec_module(self.batch)
        spec = importlib.util.spec_from_file_location(
            "batch_adapter_under_test", MODULE_DIR / "MorphoWeaveBatchShapeCompletion.py")
        self.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.adapter)
        self.widget = self.make_widget()
        self.original_ids = [node.GetID() for node in self.scene.nodes]
        self.mesh = self.root / "bone.ply"
        self.mesh.write_text(json.dumps([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        self.markups = self.root / "bone.mrk.json"
        self.markups.write_text(json.dumps([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        self.output = self.root / "out"
        self.output.mkdir()

    def load(self, path):
        # Simulate data, display, and storage nodes, including partial load failure.
        node = self.scene.add()
        self.scene.add()
        self.scene.add()
        data = json.loads(Path(path).read_text())
        if data == "fail":
            raise ValueError("failed during loading")
        node.points = np.asarray(data, dtype=float)
        return node

    def make_widget(self):
        widget = BaseWidget()
        for name in ("template_model_selector", "template_dense_selector",
                     "template_sparse_selector", "ssm_table_selector",
                     "target_model_selector", "target_landmark_selector"):
            setattr(widget, name, Selector(self.scene.add()))
        widget.output_directory = OutputPath(self.root / "original-output")
        widget._run_folder_item = 1
        widget.settings = types.SimpleNamespace(use_landmarks=False, seed=17, coverage=0.75)
        widget.run_count = 0
        widget.seen_landmarks = []
        widget.add_unrelated_node = False
        widget.edit_template_during_fit = False
        widget.logic = self.adapter._BatchCompletionLogic(self.scene)
        widget._batch_input_token = self.adapter._shared_input_token(widget)
        return widget

    def process(self, landmarks=None):
        specimen = self.batch.BatchSpecimen("bone", self.mesh, landmarks)
        self.adapter._process_specimen(self.widget, specimen, self.output)

    def assert_scene_restored(self):
        self.assertEqual([node.GetID() for node in self.scene.nodes], self.original_ids)
        self.assertEqual(self.scene.hierarchy.folders, {1})
        self.assertEqual(self.widget._run_folder_item, 1)
        self.assertEqual(self.widget.output_directory.currentPath, str(self.root / "original-output"))

    def test_batch_widget_inherits_single_specimen_implementation(self):
        self.assertIs(self.adapter.MorphoWeaveBatchShapeCompletionWidget._run_completion_impl,
                      BaseWidget._run_completion_impl)

    def test_success_removes_temporary_nodes_and_restores_selectors(self):
        target = self.widget.target_model_selector.currentNode()
        landmarks = self.widget.target_landmark_selector.currentNode()
        self.process()
        self.assertIs(self.widget.target_model_selector.currentNode(), target)
        self.assertIs(self.widget.target_landmark_selector.currentNode(), landmarks)
        self.assertEqual(self.widget.run_count, 1)
        self.assert_scene_restored()

    def test_unassisted_fit_does_not_reuse_original_landmark_selector(self):
        self.process()
        self.assertEqual(self.widget.seen_landmarks, [None])

    def test_matching_landmarks_are_loaded_and_cleaned(self):
        self.widget.settings.use_landmarks = True
        self.process(self.markups)
        self.assertIsNotNone(self.widget.seen_landmarks[0])
        self.assert_scene_restored()

    def test_missing_landmarks_raise_and_restore_scene(self):
        self.widget.settings.use_landmarks = True
        with self.assertRaisesRegex(ValueError, "matching landmark"):
            self.process()
        self.assertEqual(self.widget.run_count, 0)
        self.assert_scene_restored()

    def test_insufficient_landmarks_raise_before_fitting(self):
        self.widget.settings.use_landmarks = True
        self.markups.write_text(json.dumps([[1, 2, 3], [4, 5, 6]]))
        with self.assertRaisesRegex(ValueError, "three matched"):
            self.process(self.markups)
        self.assertEqual(self.widget.run_count, 0)
        self.assert_scene_restored()

    def test_failed_load_removes_partially_created_nodes(self):
        self.mesh.write_text(json.dumps("fail"))
        with self.assertRaisesRegex(ValueError, "loading"):
            self.process()
        self.assert_scene_restored()

    def test_failed_output_creation_removes_nodes_and_folder(self):
        self.widget.logic.fail_create = True
        with self.assertRaisesRegex(RuntimeError, "output creation"):
            self.process()
        self.assert_scene_restored()

    def test_failed_export_removes_storage_and_output_nodes(self):
        self.widget.logic.fail_export = True
        with self.assertRaisesRegex(IOError, "export"):
            self.process()
        self.assert_scene_restored()

    def test_nodes_added_outside_our_io_are_not_deleted(self):
        self.widget.add_unrelated_node = True
        self.process()
        self.assertIsNotNone(self.scene.GetNodeByID(self.widget.unrelated.GetID()))
        self.assertEqual(len(self.scene.nodes), len(self.original_ids) + 1)

    def test_ssm_is_parsed_once_and_arrays_are_isolated(self):
        node = self.widget.ssm_table_selector.currentNode()
        first = self.widget.logic.ssm_from_table(node)
        first[0][:] = 99
        second = self.widget.logic.ssm_from_table(node)
        self.assertEqual(self.widget.logic.ssm_calls, 1)
        np.testing.assert_array_equal(second[0], np.ones((3, 3)))

    def test_repeated_specimens_do_not_grow_scene(self):
        for _ in range(10):
            self.process()
            self.assert_scene_restored()
        self.assertEqual(self.widget.logic.ssm_calls, 1)

    def test_same_seed_matches_direct_single_specimen_fake_fitter(self):
        # This validates adapter forwarding and state isolation, not rustcpd math.
        self.process()
        batched = json.loads((self.output / "result.json").read_text())
        direct = self.root / "direct"
        direct.mkdir()
        self.widget.output_directory.setCurrentPath(str(direct))
        self.widget.target_model_selector.setCurrentNode(self.load(str(self.mesh)))
        self.widget.target_landmark_selector.setCurrentNode(None)
        self.widget._run_completion_impl()
        single = json.loads((direct / "result.json").read_text())
        self.assertEqual(batched, single)
        np.testing.assert_array_equal(batched["points"], single["points"])

    def test_shared_input_edits_between_specimens_are_rejected(self):
        self.widget.template_dense_selector.currentNode().mtime += 1
        with self.assertRaisesRegex(RuntimeError, "changed during the batch"):
            self.process()
        self.assertEqual(self.widget.run_count, 0)
        self.assert_scene_restored()

    def test_shared_input_edits_during_fit_are_rejected(self):
        self.widget.edit_template_during_fit = True
        with self.assertRaisesRegex(RuntimeError, "changed during fitting"):
            self.process()
        self.assert_scene_restored()

    def test_singleton_application_nodes_are_not_owned_by_batch(self):
        singleton = self.widget.logic.tracked_call(self.scene.add, singleton=True)
        self.widget.logic.cleanup_specimen()
        self.assertIsNotNone(self.scene.GetNodeByID(singleton.GetID()))

    def test_array_hash_handles_empty_and_noncontiguous_arrays(self):
        for shape in ((0,), (0, 3)):
            self.assertEqual(len(self.adapter._array_hash(np.empty(shape))), 64)
        array = np.arange(12).reshape(3, 4).T
        self.assertEqual(self.adapter._array_hash(array), self.adapter._array_hash(array.copy()))
        self.assertNotEqual(self.adapter._array_hash(array), self.adapter._array_hash(array + 1))


if __name__ == "__main__":
    unittest.main()
