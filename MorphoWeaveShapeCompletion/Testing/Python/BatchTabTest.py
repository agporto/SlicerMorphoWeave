"""Integrated-tab/adapter tests with fake Qt, MRML and a deterministic fitter.

This tests UI wiring, workflow isolation and exports, not live Slicer or rustcpd.
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
import numpy.testing

MODULE_DIR = Path(__file__).resolve().parents[2]


class Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, *args):
        for callback in list(self.callbacks):
            callback(*args)


class Control:
    def __init__(self, text=""):
        self.text, self.value, self.checked = text, 0, False
        self.currentPath, self.node, self.enabled = "", None, True
        for name in ("currentNodeChanged", "currentPathChanged", "valueChanged", "toggled", "clicked"):
            setattr(self, name, Signal())

    def setEnabled(self, value):
        self.enabled = bool(value)

    def currentNode(self):
        return self.node

    def setCurrentNode(self, value):
        if self.node is not value:
            self.node = value
            self.currentNodeChanged.emit(value)

    def setValue(self, value):
        if self.value != value:
            self.value = value
            self.valueChanged.emit(value)

    def setChecked(self, value):
        if self.checked != value:
            self.checked = bool(value)
            self.toggled.emit(bool(value))

    def setCurrentPath(self, value):
        if self.currentPath != str(value):
            self.currentPath = str(value)
            self.currentPathChanged.emit(str(value))

    def setText(self, value):
        self.text = str(value)

    def setPlainText(self, value):
        self.text = str(value)

    def appendPlainText(self, value):
        self.text += "\n" + str(value)

    def toPlainText(self):
        return self.text

    def clear(self):
        self.text = ""

    def setRange(self, *args):
        pass

    def setWordWrap(self, *args):
        pass

    def setReadOnly(self, *args):
        pass

    def setMinimumHeight(self, *args):
        pass


class Tabs:
    def __init__(self):
        self.entries = []
        self.current = None

    def addTab(self, widget, name):
        self.entries.append((widget, name))
        if self.current is None:
            self.current = widget

    def insertTab(self, index, widget, name):
        self.entries.insert(index, (widget, name))


class Form:
    def __init__(self, *args):
        self.rows = []

    def addRow(self, *args):
        self.rows.append(args)


class Node:
    def __init__(self, node_id, points=None, singleton=False):
        self.node_id = node_id
        self.points = np.ones((3, 3)) if points is None else np.asarray(points, dtype=float)
        self.mtime, self.singleton, self.parent = 1, singleton, None

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

    def GetAttribute(self, name):
        return None

    def GetPoints(self):
        return types.SimpleNamespace(GetData=lambda: self.points)

    def GetPolys(self):
        return types.SimpleNamespace(GetData=lambda: np.empty(0, dtype=np.int64))

    GetStrips = GetLines = GetVerts = GetPolys


class Hierarchy:
    def __init__(self):
        self.folders, self.next_id = {1}, 2

    def create(self):
        value = self.next_id
        self.next_id += 1
        self.folders.add(value)
        return value

    def RemoveItem(self, value):
        self.folders.discard(value)


class Scene:
    def __init__(self):
        self.nodes, self.hierarchy, self.next_id = [], Hierarchy(), 1

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
        return next((n for n in self.nodes if n.GetID() == node_id), None)

    def RemoveNode(self, node):
        self.nodes.remove(node)

    def GetFirstNodeByClass(self, class_name):
        return self.hierarchy if class_name == "vtkMRMLSubjectHierarchyNode" else None


class BaseLogic:
    def __init__(self):
        self.scene = sys.modules["slicer"].mrmlScene
        self.ssm_calls, self.fail_export, self.fail_create = 0, False, False

    def ssm_from_table(self, node):
        self.ssm_calls += 1
        return np.ones((3, 3)), np.ones((3, 3, 1)), np.ones(1)

    def model_polydata_world(self, node):
        return node

    def markups_labels_points_world(self, node):
        return [str(i) for i in range(len(node.points))], node.points

    def create_output_folder(self, name):
        return self.scene.hierarchy.create()

    def create_completion_outputs(self):
        folder, model = self.create_output_folder("output"), self.scene.add()
        self.scene.add()
        if self.fail_create:
            raise RuntimeError("failed during output creation")
        return {"folder_item": folder, "model": model}

    def save_completion_outputs(self, outputs, diagnostics, directory):
        self.scene.add()
        (Path(directory) / "result.json").write_text(json.dumps(diagnostics, sort_keys=True))
        if self.fail_export:
            raise IOError("failed during export")


class BaseModule:
    def __init__(self, parent):
        self.parent = parent
        parent.title, parent.helpText = "Shape Completion", "Existing completion workflow."


class BaseWidget:
    def __init__(self, parent=None):
        self._ui_ready, self._profile, self._run_folder_item = False, None, 1
        self.run_count, self.calibration_count, self.clear_count = 0, 0, 0
        self.seen_landmarks = []
        self.add_unrelated_node = self.edit_template_during_fit = False
        self.edit_settings_during_fit = False
        self.dependencies_available, self.seed, self.hook = True, 17, None

    def setup(self):
        self.logic = BaseLogic()
        self._build_ui()
        self._connect_ui()
        self._ui_ready = True

    def _make_selector(self, *args, **kwargs):
        return Control()

    def _double(self, minimum, maximum, value, *args):
        control = Control()
        control.setValue(value)
        return control

    def _build_ui(self):
        self.tabs = Tabs()
        for name, label in (("complete_tab", "Complete Shape"), ("calibration_tab", "Calibration"),
                            ("advanced_tab", "Advanced")):
            setattr(self, name, Control())
            self.tabs.addTab(getattr(self, name), label)
        for name in ("template_model_selector", "template_dense_selector", "template_sparse_selector",
                     "ssm_table_selector", "target_model_selector", "target_landmark_selector",
                     "output_directory", "coverage_spin", "use_landmarks_check", "order_fallback_check",
                     "profile_path", "interpolation_neighbors", "interpolation_sharpness",
                     "interpolation_chunk_size", "component_interpolation_check",
                     "full_resolution_samples_check", "run_button", "calibration_run_button",
                     "run_progress", "diagnostics_text", "complete_status"):
            setattr(self, name, Control())
        self.coverage_spin.value = 0.75
        self.interpolation_neighbors.value = 8
        self.interpolation_sharpness.value = 2.0
        self.interpolation_chunk_size.value = 200000
        self.component_interpolation_check.checked = True

    def _connect_ui(self):
        for name in ("template_model_selector", "template_dense_selector", "ssm_table_selector",
                     "target_model_selector", "target_landmark_selector"):
            getattr(self, name).currentNodeChanged.connect(self._validate_complete_inputs)
        self.profile_path.currentPathChanged.connect(self._on_profile_path_changed)

    def _on_profile_path_changed(self, path):
        try:
            self._profile = json.loads(Path(path).read_text()) if path else None
        except (ValueError, OSError):
            self._profile = None

    def _validate_complete_inputs(self, *args):
        self.run_button.enabled = self.target_model_selector.currentNode() is not None

    def _validate_calibration_inputs(self, *args):
        self.calibration_run_button.enabled = True

    def _read_settings(self):
        return types.SimpleNamespace(use_landmarks=self.use_landmarks_check.checked,
                                     seed=self.seed, coverage=self.coverage_spin.value)

    def _settings_snapshot(self, settings):
        return dict(vars(settings), allow_order_fallback=self.order_fallback_check.checked)

    def _ensure_dependencies(self):
        return self.dependencies_available

    def _preview_landmark_match_count(self):
        node = self.target_landmark_selector.currentNode()
        return 0 if node is None else len(node.points)

    def _run_completion_impl(self):
        self.run_count += 1
        self.seen_landmarks.append(self.target_landmark_selector.currentNode())
        if self.add_unrelated_node:
            self.unrelated = self.logic.scene.add()
        settings = self._read_settings()
        arrays = self.logic.ssm_from_table(self.ssm_table_selector.currentNode())
        rng = np.random.default_rng(settings.seed)
        points = arrays[0] + self.target_model_selector.currentNode().points + rng.normal(size=(3, 3))
        outputs = self.logic.create_completion_outputs()
        self.logic.save_completion_outputs(outputs, {"points": points.tolist(), "settings": vars(settings)},
                                           self.output_directory.currentPath)
        self._run_folder_item = outputs["folder_item"]
        self.run_progress.setValue(100)
        self.diagnostics_text.setPlainText("temporary batch diagnostics")
        self.complete_status.setText("temporary batch status")
        if self.edit_template_during_fit:
            self.template_dense_selector.currentNode().mtime += 1
        if self.edit_settings_during_fit:
            self.coverage_spin.setValue(0.5)

    def on_run_completion(self):
        if self.hook:
            self.hook()

    def on_run_calibration(self):
        self.calibration_count += 1
        if self.hook:
            self.hook()

    def on_clear_outputs(self):
        self.clear_count += 1

    def enter(self):
        pass

    def cleanup(self):
        self._cancel_calibration = True


def fake_run_completion():
    pass


class BatchTabTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scene = Scene()
        self.events, self.errors = None, []
        fake_slicer = types.ModuleType("slicer")
        fake_slicer.mrmlScene = self.scene
        fake_slicer.util = types.SimpleNamespace(loadModel=self.load, loadMarkups=self.load,
                                                 errorDisplay=self.errors.append)
        fake_slicer.app = types.SimpleNamespace(processEvents=self.process_events)
        fake_slicer.vtkMRMLSubjectHierarchyNode = types.SimpleNamespace(
            GetSubjectHierarchyNode=lambda scene: scene.hierarchy)
        base_name = "Resources.Python.MorphoWeaveShapeCompletionBase"
        base = types.ModuleType(base_name)
        base.MorphoWeaveShapeCompletion = BaseModule
        base.MorphoWeaveShapeCompletionLogic = BaseLogic
        base.MorphoWeaveShapeCompletionWidget = BaseWidget
        base.MorphoWeaveShapeCompletionTest = object
        base.run_completion = fake_run_completion
        base.public_helper = "preserved"
        fake_vtk, fake_vtk_util = types.ModuleType("vtk"), types.ModuleType("vtk.util")
        fake_vtk_numpy = types.ModuleType("vtk.util.numpy_support")
        fake_vtk_numpy.vtk_to_numpy = np.asarray
        fake_vtk.util, fake_vtk_util.numpy_support = fake_vtk_util, fake_vtk_numpy
        fake_qt, fake_ctk = types.ModuleType("qt"), types.ModuleType("ctk")
        for name in ("QWidget", "QLabel", "QCheckBox", "QPushButton", "QProgressBar", "QPlainTextEdit"):
            setattr(fake_qt, name, Control)
        fake_qt.QFormLayout = Form
        fake_ctk.ctkCollapsibleButton = fake_ctk.ctkPathLineEdit = Control
        Control.Dirs, Control.Files = 1, 2
        backend = types.ModuleType("rustcpd")
        backend.__file__, backend.__version__ = __file__, "test-backend"
        patches = {
            "vtk": fake_vtk, "vtk.util": fake_vtk_util, "vtk.util.numpy_support": fake_vtk_numpy,
            "ctk": fake_ctk, "qt": fake_qt, "slicer": fake_slicer, "rustcpd": backend,
            "Resources": types.ModuleType("Resources"),
            "Resources.Python": types.ModuleType("Resources.Python"), base_name: base,
        }
        patcher = patch.dict(sys.modules, patches)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.batch = self.import_module("Resources.Python.MorphoWeaveCompletionBatch",
                                       "Resources/Python/MorphoWeaveCompletionBatch.py")
        self.adapter = self.import_module("Resources.Python.MorphoWeaveShapeCompletionBatch",
                                         "Resources/Python/MorphoWeaveShapeCompletionBatch.py")
        self.main = self.import_module("MorphoWeaveShapeCompletion", "MorphoWeaveShapeCompletion.py")
        log_patch = patch.object(self.adapter.logging, "exception")
        log_patch.start()
        self.addCleanup(log_patch.stop)
        self.widget = self.main.MorphoWeaveShapeCompletionWidget()
        self.widget.setup()
        for name in ("template_model_selector", "template_dense_selector", "template_sparse_selector",
                     "ssm_table_selector", "target_model_selector", "target_landmark_selector"):
            getattr(self.widget, name).setCurrentNode(self.scene.add())
        self.widget.output_directory.setCurrentPath(str(self.root / "original-output"))
        self.widget.run_progress.setValue(73)
        self.widget.diagnostics_text.setPlainText("original diagnostics")
        self.widget.complete_status.setText("original status")
        self.original_ids = [node.GetID() for node in self.scene.nodes]
        inputs, marks = self.root / "inputs", self.root / "marks"
        inputs.mkdir()
        marks.mkdir()
        self.mesh, self.markups = inputs / "bone.ply", marks / "bone.mrk.json"
        self.mesh.write_text(json.dumps([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        self.markups.write_text(self.mesh.read_text())
        self.output = self.root / "direct-adapter-output"
        self.output.mkdir()
        self.widget.batch_input.setCurrentPath(str(inputs))
        self.widget.batch_landmarks.setCurrentPath(str(marks))
        self.widget.batch_output.setCurrentPath(str(self.root / "batch-output"))
        self.widget._batch_input_token = self.adapter._batch_state_token(self.widget)

    def import_module(self, name, relative):
        spec = importlib.util.spec_from_file_location(name, MODULE_DIR / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def process_events(self):
        if self.events:
            self.events()

    def load(self, path):
        node = self.scene.add()
        self.scene.add()
        self.scene.add()
        data = json.loads(Path(path).read_text())
        if data == "fail":
            raise ValueError("failed during loading")
        node.points = np.asarray(data, dtype=float)
        return node

    def process(self, landmarks=None):
        if not isinstance(self.widget.logic, self.adapter._BatchCompletionLogic):
            self.widget.logic = self.adapter._BatchCompletionLogic(self.scene)
        specimen = self.batch.BatchSpecimen("bone", self.mesh, landmarks)
        self.adapter._process_specimen(self.widget, specimen, self.output)

    def use_landmarks(self):
        self.widget.batch_use_landmarks.setChecked(True)
        self.widget._batch_input_token = self.adapter._batch_state_token(self.widget)

    def assert_scene_restored(self):
        self.assertEqual([node.GetID() for node in self.scene.nodes], self.original_ids)
        self.assertEqual(self.scene.hierarchy.folders, {1})
        self.assertEqual(self.widget._run_folder_item, 1)
        self.assertEqual(self.widget.output_directory.currentPath, str(self.root / "original-output"))

    def test_one_module_contains_batch_tab_and_keeps_default_tab(self):
        self.assertEqual([name for _, name in self.widget.tabs.entries],
                         ["Complete Shape", "Batch", "Calibration", "Advanced"])
        self.assertIs(self.widget.tabs.current, self.widget.complete_tab)
        parent = types.SimpleNamespace()
        self.main.MorphoWeaveShapeCompletion(parent)
        self.assertEqual(parent.title, "Shape Completion")
        self.assertFalse(hasattr(self.main, "MorphoWeaveBatchShapeCompletion"))
        self.assertEqual(self.main.public_helper, "preserved")

    def test_same_single_specimen_method_is_inherited(self):
        self.assertIs(type(self.widget)._run_completion_impl, BaseWidget._run_completion_impl)

    def test_model_selectors_synchronize_both_directions(self):
        for left, right in (("template_model_selector", "batch_template_model"),
                            ("template_dense_selector", "batch_template_dense"),
                            ("template_sparse_selector", "batch_template_sparse"),
                            ("ssm_table_selector", "batch_ssm_table")):
            a, b = getattr(self.widget, left), getattr(self.widget, right)
            self.assertIs(a.currentNode(), b.currentNode())
            node = self.scene.add()
            b.setCurrentNode(node)
            self.assertIs(a.currentNode(), node)
            a.setCurrentNode(None)
            self.assertIsNone(b.currentNode())

    def test_coverage_and_landmark_options_synchronize(self):
        self.widget.batch_coverage.setValue(0.6)
        self.assertEqual(self.widget._read_settings().coverage, 0.6)
        self.widget.coverage_spin.setValue(0.8)
        self.assertEqual(self.widget.batch_coverage.value, 0.8)
        self.widget.batch_use_landmarks.setChecked(True)
        self.assertTrue(self.widget.use_landmarks_check.checked)
        self.widget.use_landmarks_check.setChecked(False)
        self.assertFalse(self.widget.batch_use_landmarks.checked)
        self.widget.batch_order_fallback.setChecked(True)
        self.assertTrue(self.widget.order_fallback_check.checked)
        self.widget.order_fallback_check.setChecked(False)
        self.assertFalse(self.widget.batch_order_fallback.checked)

    def test_profile_selection_is_shared_and_loaded_once(self):
        profile = self.root / "profile.json"
        profile.write_text('{"entries": []}')
        self.widget.batch_profile.setCurrentPath(str(profile))
        self.assertEqual(self.widget.profile_path.currentPath, str(profile))
        self.assertEqual(self.widget._profile, {"entries": []})
        self.widget.profile_path.setCurrentPath("")
        self.assertEqual(self.widget.batch_profile.currentPath, "")
        self.assertIsNone(self.widget._profile)

    def test_batch_needs_no_individually_loaded_target(self):
        self.widget.target_model_selector.setCurrentNode(None)
        self.widget.target_landmark_selector.setCurrentNode(None)
        self.widget._validate_batch_inputs()
        self.assertTrue(self.widget.batch_run.enabled)
        self.assertFalse(self.widget.run_button.enabled)

    def test_landmark_mode_requires_paired_folder_and_template_landmarks(self):
        self.widget.batch_landmarks.setCurrentPath("")
        self.widget.batch_use_landmarks.setChecked(True)
        self.assertFalse(self.widget.batch_run.enabled)
        self.widget.batch_landmarks.setCurrentPath(str(self.markups.parent))
        self.assertTrue(self.widget.batch_run.enabled)
        self.widget.batch_template_sparse.setCurrentNode(None)
        self.assertFalse(self.widget.batch_run.enabled)

    def test_input_folder_cannot_be_the_output_folder(self):
        self.widget.batch_output.setCurrentPath(str(self.mesh.parent))
        self.assertFalse(self.widget.batch_run.enabled)

    def test_single_run_blocks_reentrant_batch_and_calibration(self):
        def hook():
            self.assertFalse(self.widget.batch_run.enabled)
            self.widget.on_run_batch()
            self.widget.on_run_calibration()
            self.widget.on_clear_outputs()
        self.widget.hook = hook
        self.widget.on_run_completion()
        self.assertEqual(self.widget.calibration_count, 0)
        self.assertEqual(self.widget.clear_count, 0)
        self.assertFalse(Path(self.widget.batch_output.currentPath).exists())
        self.assertIsNone(self.widget._workflow_busy)
        self.assertTrue(self.widget.batch_run.enabled)

    def test_calibration_keeps_cancel_accessible_but_blocks_batch(self):
        def hook():
            self.assertTrue(self.widget.calibration_tab.enabled)
            self.assertFalse(self.widget.batch_run.enabled)
            self.widget.on_run_batch()
        self.widget.hook = hook
        self.widget.on_run_calibration()
        self.assertEqual(self.widget.calibration_count, 1)
        self.assertFalse(Path(self.widget.batch_output.currentPath).exists())
        self.assertIsNone(self.widget._workflow_busy)

    def test_single_exception_restores_controls(self):
        self.widget.hook = lambda: (_ for _ in ()).throw(RuntimeError("fit failed"))
        with self.assertRaisesRegex(RuntimeError, "fit failed"):
            self.widget.on_run_completion()
        self.assertIsNone(self.widget._workflow_busy)
        self.assertTrue(self.widget.complete_tab.enabled)
        self.assertTrue(self.widget.batch_inputs_box.enabled)

    def test_busy_validation_never_reenables_run_buttons(self):
        self.widget._set_operation("batch")
        self.widget._validate_complete_inputs()
        self.widget._validate_calibration_inputs()
        self.widget._validate_batch_inputs()
        self.assertFalse(self.widget.run_button.enabled)
        self.assertFalse(self.widget.calibration_run_button.enabled)
        self.assertFalse(self.widget.batch_run.enabled)
        self.assertTrue(self.widget.batch_cancel.enabled)
        self.widget._set_operation(None)

    def test_batch_success_restores_single_state_and_scene(self):
        logic = self.widget.logic
        target = self.widget.target_model_selector.currentNode()
        self.widget.on_run_batch()
        self.assertEqual(self.errors, [])
        self.assertIn("1 success", self.widget.batch_status.text)
        self.assertIs(self.widget.logic, logic)
        self.assertIs(self.widget.target_model_selector.currentNode(), target)
        self.assertEqual(self.widget.run_progress.value, 73)
        self.assertEqual(self.widget.diagnostics_text.toPlainText(), "original diagnostics")
        self.assertEqual(self.widget.complete_status.text, "original status")
        self.assert_scene_restored()
        self.assertTrue((Path(self.widget.batch_output.currentPath) / "bone/result.json").is_file())
        self.assertTrue(self.widget.batch_run.enabled)
        self.assertFalse(self.widget.batch_cancel.enabled)

    def test_batch_failure_isolation(self):
        (self.mesh.parent / "a.ply").write_text(json.dumps("fail"))
        self.widget.on_run_batch()
        self.assertIn("1 success", self.widget.batch_status.text)
        self.assertIn("1 failed", self.widget.batch_status.text)
        self.assert_scene_restored()

    def test_batch_resume_does_not_rerun_fitter(self):
        self.widget.on_run_batch()
        self.assertEqual(self.widget.run_count, 1)
        self.widget.on_run_batch()
        self.assertEqual(self.widget.run_count, 1)
        self.assertIn("1 skipped", self.widget.batch_status.text)
        self.assert_scene_restored()

    def test_batch_cancel_before_fitting_restores_state(self):
        self.events = self.widget.on_cancel_batch
        self.widget.on_run_batch()
        self.assertEqual(self.widget.run_count, 0)
        self.assertIn("1 cancelled", self.widget.batch_status.text)
        self.assertIsNone(self.widget._workflow_busy)
        self.assert_scene_restored()

    def test_declined_dependencies_restore_state(self):
        self.widget.dependencies_available = False
        self.widget.on_run_batch()
        self.assertIn("dependencies", self.widget.batch_status.text)
        self.assertIsNone(self.widget._workflow_busy)
        self.assertTrue(self.widget.batch_run.enabled)
        self.assertEqual(self.widget.diagnostics_text.toPlainText(), "original diagnostics")

    def test_invalid_profile_does_not_silently_run_uncalibrated(self):
        self.widget.batch_profile.setCurrentPath(str(self.root / "missing.json"))
        self.widget.on_run_batch()
        self.assertEqual(self.widget.run_count, 0)
        self.assertIn("profile could not be loaded", self.errors[0])
        self.assertIsNone(self.widget._workflow_busy)

    def test_success_removes_temporary_nodes_and_restores_selectors(self):
        target, marks = self.widget.target_model_selector.currentNode(), self.widget.target_landmark_selector.currentNode()
        self.process()
        self.assertIs(self.widget.target_model_selector.currentNode(), target)
        self.assertIs(self.widget.target_landmark_selector.currentNode(), marks)
        self.assert_scene_restored()

    def test_unassisted_fit_does_not_reuse_single_landmarks(self):
        self.process()
        self.assertEqual(self.widget.seen_landmarks, [None])

    def test_matched_landmarks_are_loaded_and_cleaned(self):
        self.use_landmarks()
        self.process(self.markups)
        self.assertIsNotNone(self.widget.seen_landmarks[0])
        self.assert_scene_restored()

    def test_missing_landmarks_fail_before_fit_and_restore_scene(self):
        self.use_landmarks()
        with self.assertRaisesRegex(ValueError, "matching landmark"):
            self.process()
        self.assertEqual(self.widget.run_count, 0)
        self.assert_scene_restored()

    def test_insufficient_landmarks_fail_before_fit(self):
        self.use_landmarks()
        self.markups.write_text(json.dumps([[1, 2, 3], [4, 5, 6]]))
        with self.assertRaisesRegex(ValueError, "three matched"):
            self.process(self.markups)
        self.assertEqual(self.widget.run_count, 0)
        self.assert_scene_restored()

    def test_partial_load_failure_cleans_up(self):
        self.mesh.write_text(json.dumps("fail"))
        with self.assertRaisesRegex(ValueError, "loading"):
            self.process()
        self.assert_scene_restored()

    def test_failed_output_creation_cleans_up(self):
        self.widget.logic = self.adapter._BatchCompletionLogic(self.scene)
        self.widget.logic.fail_create = True
        with self.assertRaisesRegex(RuntimeError, "output creation"):
            self.process()
        self.assert_scene_restored()

    def test_failed_export_cleans_up(self):
        self.widget.logic = self.adapter._BatchCompletionLogic(self.scene)
        self.widget.logic.fail_export = True
        with self.assertRaisesRegex(IOError, "export"):
            self.process()
        self.assert_scene_restored()

    def test_unrelated_nodes_are_preserved(self):
        self.widget.add_unrelated_node = True
        self.process()
        self.assertIsNotNone(self.scene.GetNodeByID(self.widget.unrelated.GetID()))
        self.assertEqual(len(self.scene.nodes), len(self.original_ids) + 1)

    def test_ssm_arrays_are_isolated_and_parsed_once(self):
        logic = self.adapter._BatchCompletionLogic(self.scene)
        node = self.widget.ssm_table_selector.currentNode()
        arrays = logic.ssm_from_table(node)
        arrays[0][:] = 99
        np.testing.assert_array_equal(logic.ssm_from_table(node)[0], np.ones((3, 3)))
        self.assertEqual(logic.ssm_calls, 1)

    def test_ten_specimens_do_not_accumulate_nodes(self):
        for _ in range(10):
            self.process()
            self.assert_scene_restored()
        self.assertEqual(self.widget.logic.ssm_calls, 1)

    def test_same_seed_matches_direct_single_specimen_fake_fitter(self):
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

    def test_changed_shared_input_before_fit_is_rejected(self):
        self.widget.template_dense_selector.currentNode().mtime += 1
        with self.assertRaisesRegex(RuntimeError, "changed during the batch"):
            self.process()
        self.assertEqual(self.widget.run_count, 0)
        self.assert_scene_restored()

    def test_changed_shared_input_during_fit_is_rejected(self):
        self.widget.edit_template_during_fit = True
        with self.assertRaisesRegex(RuntimeError, "changed during fitting"):
            self.process()
        self.assert_scene_restored()

    def test_changed_settings_during_fit_are_rejected(self):
        self.widget.edit_settings_during_fit = True
        with self.assertRaisesRegex(RuntimeError, "changed during fitting"):
            self.process()
        self.assert_scene_restored()

    def test_singletons_are_not_owned(self):
        logic = self.adapter._BatchCompletionLogic(self.scene)
        node = logic.tracked_call(self.scene.add, singleton=True)
        logic.cleanup_specimen()
        self.assertIsNotNone(self.scene.GetNodeByID(node.GetID()))

    def test_cleanup_requests_cancellation(self):
        self.widget.cleanup()
        self.assertTrue(self.widget._cancel_batch)
        self.assertTrue(self.widget._cancel_calibration)

    def test_hashes_handle_empty_and_noncontiguous_arrays(self):
        for shape in ((0,), (0, 3)):
            self.assertEqual(len(self.adapter._array_hash(np.empty(shape))), 64)
        array = np.arange(12).reshape(3, 4).T
        self.assertEqual(self.adapter._array_hash(array), self.adapter._array_hash(array.copy()))
        self.assertNotEqual(self.adapter._array_hash(array), self.adapter._array_hash(array + 1))


if __name__ == "__main__":
    unittest.main()
