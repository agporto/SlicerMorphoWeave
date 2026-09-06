"""Workflow regressions using real VTK exports and a deterministic test backend.

Slicer/Qt are doubled; this is not a live Slicer or native hdmseg benchmark.
"""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import vtk
# Import native SciPy modules before sys.modules is temporarily patched.
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

MODULE_DIR = Path(__file__).resolve().parents[2]
CONTROL_NAMES = (
    "meshDir", "mrkDir", "referencePath", "outDir", "selection", "fixedK",
    "maxK", "neighbors", "components", "bootstraps", "seed", "parallel",
    "smoothing", "writeVtp", "writePly", "previewN", "installButton", "runButton",
)


class Base:
    def __init__(self, parent=None):
        pass


class Control:
    def __init__(self):
        self.currentPath = ""
        self.value = 3
        self.currentIndex = 0
        self.checked = False
        self.enabled = True
        self.text = ""

    def setTextFormat(self, value):
        pass

    def setText(self, value):
        self.text = value

    def setAccessibleName(self, value):
        pass


@contextmanager
def errors_propagate(*args, **kwargs):
    yield


def make_polydata():
    # Three disjoint triangles, with one region per triangle in the first run.
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                       [3, 0, 0], [4, 0, 0], [3, 1, 0],
                       [6, 0, 0], [7, 0, 0], [6, 1, 0]], dtype=float)
    data = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(points, deep=True))
    data.SetPoints(pts)
    cells = vtk.vtkCellArray()
    for start in (0, 3, 6):
        cells.InsertNextCell(3)
        for index in range(start, start + 3):
            cells.InsertCellPoint(index)
    data.SetPolys(cells)
    return points, data


def backend_segment(X, **kwargs):
    k = kwargs["k"] or 2
    labels = np.repeat([0, 1, 2] if k == 3 else [0, 0, 1], 3)
    return types.SimpleNamespace(
        labels=labels, embedding=np.arange(18, dtype=float).reshape(9, 2),
        eigenvalues=np.array([1.0, 0.5]), k=k, modularity=0.25, stability=None)


class SurfaceSegmentationWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.meshes = self.root / "meshes"
        self.markups = self.root / "markups"
        self.output = self.root / "output"
        for directory in (self.meshes, self.markups, self.output):
            directory.mkdir()
        self.slicer = types.ModuleType("slicer")
        self.slicer.util = types.SimpleNamespace(
            showStatusMessage=Mock(), tryWithErrorDisplay=errors_propagate)
        self.slicer.app = types.SimpleNamespace(processEvents=Mock())
        scripted = types.ModuleType("slicer.ScriptedLoadableModule")
        for name in ("ScriptedLoadableModule", "ScriptedLoadableModuleLogic",
                     "ScriptedLoadableModuleWidget", "ScriptedLoadableModuleTest"):
            setattr(scripted, name, Base)
        qt = types.ModuleType("qt")
        qt.Qt = types.SimpleNamespace(RichText=1)
        patcher = patch.dict(sys.modules, {
            "slicer": self.slicer, "slicer.ScriptedLoadableModule": scripted,
            "qt": qt, "ctk": types.ModuleType("ctk"),
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        spec = importlib.util.spec_from_file_location(
            "surface_module_under_test", MODULE_DIR / "MorphoWeaveSurfaceSegmentation.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.points, self.polydata = make_polydata()
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(self.meshes / "bone.vtp"))
        writer.SetInputData(self.polydata)
        self.assertEqual(writer.Write(), 1)
        (self.markups / "bone.mrk.json").write_text(json.dumps({
            "markups": [{"coordinateSystem": "RAS", "controlPoints": [
                {"position": point.tolist(), "positionStatus": "defined"}
                for point in self.points]}]}))
        self.backend = types.SimpleNamespace(__version__="test-backend", segment=Mock(side_effect=backend_segment))
        self.logic = self.module.MorphoWeaveSurfaceSegmentationLogic()
        self.logic._clearPreviews = Mock()
        self.logic._colorTable = Mock()
        self.logic._loadMesh = self.load_mesh
        logger = patch.object(self.module.logging, "exception")
        logger.start()
        self.addCleanup(logger.stop)

    def load_mesh(self, path, needFaces=True):
        # The real projection/smoothing/export methods below are not mocked.
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(path))
        reader.Update()
        data = vtk.vtkPolyData()
        data.DeepCopy(reader.GetOutput())
        vertices = vtk_to_numpy(data.GetPoints().GetData()).copy()
        return vertices, self.logic._faces(data) if needFaces else None, data

    def run_logic(self, k=3, **kwargs):
        return self.logic.run(str(self.meshes), str(self.markups), str(self.output),
                              self.backend, fixedK=k, neighbors=8, components=2,
                              smoothing=0, previewN=0, **kwargs)

    def make_widget(self):
        widget = self.module.MorphoWeaveSurfaceSegmentationWidget()
        for name in CONTROL_NAMES:
            setattr(widget, name, Control())
        widget.status = Control()
        widget.meshDir.currentPath = str(self.meshes)
        widget.mrkDir.currentPath = str(self.markups)
        widget.outDir.currentPath = str(self.output)
        widget.logic = Mock()
        widget.logic.pairFiles.return_value = [("bone.vtp", "bone.mrk.json")]
        widget.logic.run.return_value = {
            "specimens": 1, "k": 3, "modularity": 0.25, "stability": None,
            "skipped": 0, "output_directory": str(self.output / "run-actual"),
        }
        widget._ensureBackend = Mock(return_value=self.backend)
        return widget

    def assert_frozen(self, widget):
        self.assertTrue(widget._running)
        for name in CONTROL_NAMES:
            self.assertFalse(getattr(widget, name).enabled, name)

    def assert_restored(self, widget):
        self.assertFalse(widget._running)
        self.assertTrue(widget.runButton.enabled)
        self.assertTrue(widget.installButton.enabled)
        self.assertTrue(widget.meshDir.enabled)
        self.assertTrue(widget.fixedK.enabled)
        self.assertFalse(widget.maxK.enabled)
        self.assertFalse(widget.bootstraps.enabled)

    def test_rerun_with_fewer_regions_is_isolated_and_preserves_first_run(self):
        first = self.run_logic(3)
        directory1 = Path(first["output_directory"])
        before = {path.name: path.read_bytes() for path in directory1.iterdir()}
        self.assertEqual(len(list(directory1.glob("bone_seg_*.ply"))), 3)
        second = self.run_logic(2)
        directory2 = Path(second["output_directory"])
        self.assertNotEqual(directory1, directory2)
        self.assertEqual(directory1.parent, self.output)
        self.assertEqual(directory2.parent, self.output)
        self.assertEqual({p.name: p.read_bytes() for p in directory1.iterdir()}, before)
        self.assertEqual(sorted(p.name for p in directory2.glob("bone_seg_*.ply")),
                         ["bone_seg_00.ply", "bone_seg_01.ply"])
        cells = 0
        for path in directory2.glob("*.ply"):
            reader = vtk.vtkPLYReader()
            reader.SetFileName(str(path))
            reader.Update()
            self.assertGreater(reader.GetOutput().GetNumberOfCells(), 0)
            cells += reader.GetOutput().GetNumberOfCells()
        self.assertEqual(cells, 3)

    def test_same_timestamp_runs_get_distinct_directories(self):
        with patch.object(self.module.time, "strftime", return_value="20260905_120000"):
            first = self.run_logic(3)
            second = self.run_logic(3)
        self.assertNotEqual(first["output_directory"], second["output_directory"])

    def test_legacy_and_unrelated_files_in_output_root_are_untouched(self):
        for name in ("bone_seg_02.ply", "notes.txt", "MorphoWeaveSurfaceSegmentation_summary.json"):
            (self.output / name).write_bytes(b"preserve existing content")
        self.run_logic(2)
        for path in self.output.iterdir():
            if path.is_file():
                self.assertEqual(path.read_bytes(), b"preserve existing content")

    def test_summary_and_locus_outputs_belong_to_the_returned_run_directory(self):
        result = self.run_logic(2)
        directory = Path(result["output_directory"])
        summary = json.loads(Path(result["summary"]).read_text())
        self.assertEqual(Path(result["summary"]).parent, directory)
        self.assertEqual(summary["k"], 2)
        self.assertEqual(sorted(summary["outputs"]["region_ply_files"]),
                         sorted(p.name for p in directory.glob("*.ply")))
        labels = np.load(directory / "MorphoWeaveSurfaceSegmentation_locus_labels.npy")
        np.testing.assert_array_equal(labels, np.repeat([0, 0, 1], 3))
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(directory / "bone_segmented.vtp"))
        reader.Update()
        np.testing.assert_array_equal(
            vtk_to_numpy(reader.GetOutput().GetPointData().GetArray("SegID")), labels)

    def test_backend_failure_does_not_create_a_run_directory(self):
        self.backend.segment.side_effect = RuntimeError("fit failed")
        with self.assertRaisesRegex(RuntimeError, "fit failed"):
            self.run_logic()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_export_failure_is_recorded_without_modifying_previous_run(self):
        first = self.run_logic()
        directory = Path(first["output_directory"])
        before = {p.name: p.read_bytes() for p in directory.iterdir()}
        with patch.object(self.logic, "_writePlys", side_effect=IOError("disk write failed")):
            second = self.run_logic(2)
        self.assertEqual(second["skipped"], 1)
        self.assertEqual({p.name: p.read_bytes() for p in directory.iterdir()}, before)
        summary = json.loads(Path(second["summary"]).read_text())
        self.assertIn("disk write failed", summary["skipped_meshes"][0]["error"])

    def test_export_options_remain_effective(self):
        result = self.run_logic(writePly=False, writeVtp=False)
        directory = Path(result["output_directory"])
        self.assertFalse(list(directory.glob("*.ply")))
        self.assertFalse(list(directory.glob("*.vtp")))
        self.assertTrue((directory / "MorphoWeaveSurfaceSegmentation_locus_labels.npy").is_file())

    def test_validate_inputs_does_not_reenable_run_while_busy(self):
        widget = self.make_widget()
        widget._setRunning(True)
        widget.logic.pairFiles.reset_mock()
        widget._validateInputs()
        widget.logic.pairFiles.assert_not_called()
        self.assert_frozen(widget)

    def test_selection_callbacks_cannot_unfreeze_parameter_controls(self):
        widget = self.make_widget()
        widget._setRunning(True)
        for mode in range(4):
            widget.selection.currentIndex = mode
            widget._selectionChanged()
            self.assert_frozen(widget)

    def test_run_guard_is_active_before_dependency_installation(self):
        widget = self.make_widget()
        def ensure():
            self.assert_frozen(widget)
            widget._validateInputs()
            widget.onRun()
            widget._installBackend()
            return self.backend
        widget._ensureBackend.side_effect = ensure
        widget.onRun()
        widget._ensureBackend.assert_called_once()
        widget.logic.run.assert_called_once()
        self.assert_restored(widget)

    def test_run_guard_is_active_during_logic_and_event_callbacks(self):
        widget = self.make_widget()
        result = widget.logic.run.return_value
        def run(**kwargs):
            self.assert_frozen(widget)
            widget._validateInputs()
            widget._selectionChanged()
            widget.onRun()
            widget._installBackend()
            return result
        widget.logic.run.side_effect = run
        widget.onRun()
        widget.logic.run.assert_called_once()
        widget._ensureBackend.assert_called_once()
        self.assert_restored(widget)
        self.assertIn(result["output_directory"], widget.status.text)

    def test_declined_dependencies_restore_controls_without_running(self):
        widget = self.make_widget()
        widget._ensureBackend.return_value = None
        widget.onRun()
        widget.logic.run.assert_not_called()
        self.assert_restored(widget)

    def test_dependency_exception_restores_controls(self):
        widget = self.make_widget()
        widget._ensureBackend.side_effect = RuntimeError("install error")
        with self.assertRaisesRegex(RuntimeError, "install error"):
            widget.onRun()
        self.assert_restored(widget)

    def test_fit_exception_restores_controls(self):
        widget = self.make_widget()
        widget.logic.run.side_effect = RuntimeError("fit error")
        with self.assertRaisesRegex(RuntimeError, "fit error"):
            widget.onRun()
        self.assert_restored(widget)

    def test_invalid_inputs_after_fit_do_not_unconditionally_enable_run(self):
        widget = self.make_widget()
        result = widget.logic.run.return_value
        def run(**kwargs):
            widget.meshDir.currentPath = str(self.root / "missing")
            return result
        widget.logic.run.side_effect = run
        widget.onRun()
        self.assertFalse(widget._running)
        self.assertFalse(widget.runButton.enabled)

    def test_install_button_also_guards_against_nested_runs(self):
        widget = self.make_widget()
        def ensure():
            self.assert_frozen(widget)
            widget.onRun()
            widget._installBackend()
            return self.backend
        widget._ensureBackend.side_effect = ensure
        widget._installBackend()
        widget._ensureBackend.assert_called_once()
        widget.logic.run.assert_not_called()
        self.assert_restored(widget)

    def test_install_exception_releases_guard(self):
        widget = self.make_widget()
        widget._ensureBackend.side_effect = RuntimeError("install error")
        with self.assertRaisesRegex(RuntimeError, "install error"):
            widget._installBackend()
        self.assert_restored(widget)

    def test_mode_specific_control_states_are_restored(self):
        widget = self.make_widget()
        for mode in range(4):
            widget.selection.currentIndex = mode
            widget._setRunning(True)
            widget._setRunning(False)
            self.assertEqual(widget.fixedK.enabled, mode == 0)
            self.assertEqual(widget.maxK.enabled, mode != 0)
            self.assertEqual(widget.bootstraps.enabled, mode == 1)


if __name__ == "__main__":
    unittest.main()
