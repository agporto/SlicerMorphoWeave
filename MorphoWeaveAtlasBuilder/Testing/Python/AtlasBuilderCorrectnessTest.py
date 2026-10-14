"""Actual-method regressions with real NumPy/SciPy/VTK, without a Slicer process.

AST loading skips Slicer/Qt module setup, not method bodies. MRML scene, markup
conversion, writer return values, and widget controls are explicit boundary
doubles. These tests do not replace a live Slicer writer/UI integration run.
"""
import ast
from contextlib import nullcontext
from datetime import datetime
import os
import sys
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest import mock

import numpy as np
import vtk
from vtk.util import numpy_support as vtk_np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags, eye as speye
from scipy.sparse.linalg import splu

SOURCE_PATH = Path(__file__).resolve().parents[2] / "MorphoWeaveAtlasBuilder.py"


def load_classes(slicer):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) or
             isinstance(node, ast.ClassDef) and node.name in {
                 "MorphoWeaveAtlasBuilderLogic", "MorphoWeaveAtlasBuilderWidget"}]
    namespace = dict(np=np, vtk=vtk, vtk_np=vtk_np, cKDTree=cKDTree,
                     coo_matrix=coo_matrix, diags=diags, speye=speye, splu=splu,
                     os=os, Path=Path, tempfile=tempfile, datetime=datetime, time=time,
                     slicer=slicer, ScriptedLoadableModuleLogic=object,
                     ScriptedLoadableModuleWidget=object,
                     qt=types.SimpleNamespace(QApplication=types.SimpleNamespace(processEvents=lambda: None)))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace


def polydata(vertices, faces=()):
    points = vtk.vtkPoints()
    points.SetData(vtk_np.numpy_to_vtk(np.asarray(vertices, dtype=float), deep=True))
    result = vtk.vtkPolyData(); result.SetPoints(points)
    cells = vtk.vtkCellArray()
    for face in faces:
        cells.InsertNextCell(len(face))
        for i in face:
            cells.InsertCellPoint(int(i))
    result.SetPolys(cells)
    return result


def points(vertices):
    return polydata(vertices).GetPoints()


def xyz(poly):
    return vtk_np.vtk_to_numpy(poly.GetPoints().GetData()).copy()


def group(items):
    result = vtk.vtkMultiBlockDataSet()
    for i, item in enumerate(items):
        result.SetBlock(i, item)
    return result


class Node:
    def __init__(self, poly=None, array=None):
        self.poly = poly
        self.array = np.asarray(array if array is not None else [], dtype=float).reshape(-1, 3)

    def GetPolyData(self): return self.poly
    def SetAndObservePolyData(self, poly): self.poly = poly
    def GetParentTransformNode(self): return None
    def SetHideFromEditors(self, value): pass
    def GetDisplayNode(self): return None
    def CreateDefaultDisplayNodes(self): pass
    def GetNumberOfControlPoints(self): return len(self.array)


class SlicerBoundary:
    def __init__(self):
        self.removed = []
        self.models = {}
        self.markups = {}
        self.writes = []
        self.saved_arrays = {}
        self.fail_write = None
        self.render_paused = False
        self.util = types.SimpleNamespace(
            NodeModify=lambda node: nullcontext(), saveNode=self.save,
            loadModel=lambda path: Node(poly=self.models[path]),
            loadMarkups=lambda path: Node(array=self.markups[path].copy()),
            arrayFromMarkupsControlPoints=lambda node: node.array,
            updateMarkupsControlPointsFromArray=lambda node, array: setattr(node, "array", np.array(array, copy=True)),
            warningDisplay=lambda *args, **kwargs: None,
            confirmOkCancelDisplay=lambda *args: True,
        )
        self.mrmlScene = types.SimpleNamespace(
            AddNewNodeByClass=lambda *args: Node(), RemoveNode=self.removed.append)
        self.app = types.SimpleNamespace(setRenderPaused=lambda value: setattr(self, "render_paused", value))

    def save(self, node, path):
        self.writes.append(path)
        if self.fail_write == len(self.writes):
            return False
        Path(path).write_bytes(b"boundary writer output\n")
        self.saved_arrays[path] = node.array.copy()
        return True


class AtlasBuilderCorrectnessTest(unittest.TestCase):
    def setUp(self):
        self.slicer = SlicerBoundary()
        self.namespace = load_classes(self.slicer)
        self.logic = self.namespace["MorphoWeaveAtlasBuilderLogic"]()
        self.vertices = np.array([[0., 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 3], [1, 1, 1]])
        # First cell deliberately does not start at point 0: cleaning changes IDs.
        self.faces = np.array([[1, 2, 4], [1, 4, 3], [2, 3, 4], [0, 2, 1], [0, 1, 3], [0, 3, 2]])
        self.mesh = polydata(self.vertices, self.faces)

    def test_rigid_and_similarity_atlas_frames_follow_input_unit_conversion(self):
        tetra = self.vertices[:4] - self.vertices[:4].mean(axis=0)
        faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
        for unit in (0.001, 1., 1000.):
            inputs = [tetra * scale * unit for scale in (10, 20)]
            for normalize in (False, True):
                with self.subTest(unit=unit, normalize=normalize):
                    landmarks = group([polydata(v) for v in inputs])
                    meshes = group([polydata(v, faces) for v in inputs])
                    dense, base, mean = self.logic.denseCorrespondence(landmarks, meshes, useSimilarity=normalize)
                    atlas = self.logic.computeAverageModelFromGroup(dense, base)
                    mean_xyz = vtk_np.vtk_to_numpy(mean.GetData())
                    expected_size = 1. if normalize else np.linalg.norm(tetra) * 15 * unit
                    self.assertAlmostEqual(np.linalg.norm(mean_xyz - mean_xyz.mean(axis=0)) / expected_size, 1., places=7)
                    np.testing.assert_allclose(xyz(atlas), mean_xyz, rtol=2e-6, atol=expected_size*1e-7)
                    aligned_meshes, aligned_landmarks = [], []
                    for v in inputs:
                        transform = vtk.vtkLandmarkTransform()
                        transform.SetModeToSimilarity() if normalize else transform.SetModeToRigidBody()
                        transform.SetSourceLandmarks(points(v)); transform.SetTargetLandmarks(mean); transform.Update()
                        filt = vtk.vtkTransformPolyDataFilter(); filt.SetInputData(polydata(v, faces)); filt.SetTransform(transform); filt.Update()
                        aligned = vtk.vtkPolyData(); aligned.DeepCopy(filt.GetOutput())
                        aligned_meshes.append(aligned); aligned_landmarks.append(polydata(xyz(aligned)))
                    population = self.logic.sparseCorrespondenceBaseMesh(group(aligned_landmarks), group(aligned_meshes), atlas, mean, np.arange(4))
                    np.testing.assert_allclose(np.mean(population, axis=0), xyz(atlas), rtol=2e-6, atol=expected_size*1e-7)

    def test_default_mean_stays_similarity_normalized(self):
        landmarks = group([polydata(self.vertices * 10), polydata(self.vertices * 20)])
        meshes = group([polydata(self.vertices * 10, self.faces), polydata(self.vertices * 20, self.faces)])
        _, _, mean = self.logic.denseCorrespondence(landmarks, meshes)
        v = vtk_np.vtk_to_numpy(mean.GetData())
        self.assertAlmostEqual(np.linalg.norm(v - v.mean(axis=0)), 1.)

    def test_run_mean_and_closest_specimen_propagate_normalization(self):
        landmarks = group([polydata(self.vertices * scale) for scale in (10, 20)])
        meshes = group([polydata(self.vertices * scale, self.faces) for scale in (10, 20)])
        self.logic.importLandmarks = lambda path: (["a", "b"], landmarks)
        self.logic.importPaired = lambda *args, **kwargs: (["a", "b"], meshes, ["a", "b"], landmarks)
        actual = self.logic.procrustesImposition
        seen = []
        def observed(lms, rigidOnly):
            seen.append(rigidOnly)
            return actual(lms, rigidOnly)
        self.logic.procrustesImposition = observed
        for normalize in (False, True):
            self.logic.getClosestToMeanPath("unused", useSimilarity=normalize)
            model, lm = self.logic.runMean("unused", "unused", useSimilarity=normalize)
            self.assertEqual(seen[-2:], [not normalize, not normalize])
            self.assertTrue(np.isfinite(xyz(model.poly)).all())
            self.assertEqual(len(lm.array), 5)

    def test_widget_generation_propagates_normalization(self):
        widget = self.namespace["MorphoWeaveAtlasBuilderWidget"]()
        widget._status = lambda message: None
        widget.overrideCoordCheck = types.SimpleNamespace(isChecked=lambda: False)
        self.slicer.markups["lm"] = self.vertices
        self.slicer.models["mesh"] = self.mesh
        for normalize in (False, True):
            closest = mock.Mock(return_value="sample")
            mean = mock.Mock(return_value=(Node(self.mesh), Node(array=self.vertices)))
            widget.logic = types.SimpleNamespace(getClosestToMeanPath=closest,
                _list_keys=lambda path, exts: {"sample": "lm" if path == "lms" else "mesh"},
                runAlign=mock.Mock(), runMean=mean)
            folders = dict(originalLMs="lms", originalModels="meshes", alignedLMs="aligned-lms", alignedModels="aligned-meshes")
            widget._generateAtlas(folders, normalize, "tps", True)
            self.assertEqual(closest.call_args.kwargs["useSimilarity"], normalize)
            self.assertEqual(mean.call_args.kwargs["useSimilarity"], normalize)

    def test_biharmonic_off_vertex_identity_preserves_coordinates_and_ids(self):
        controls = self.vertices[:4] + [0.01, 0.01, 0.01]
        for fallback in (False, True):
            result = self.logic.biharmonicWarpPolyData(self.mesh, points(controls), points(controls), allow_fallback=fallback)
            np.testing.assert_array_equal(xyz(result), self.vertices)
            np.testing.assert_array_equal(vtk_np.vtk_to_numpy(result.GetPolys().GetData()), vtk_np.vtk_to_numpy(self.mesh.GetPolys().GetData()))

    def test_biharmonic_preserves_exact_rigid_motion(self):
        angle = .43
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        controls = self.vertices[:4] + .01
        target = controls @ rotation.T + [2, -3, 1]
        result = self.logic.biharmonicWarpPolyData(self.mesh, points(controls), points(target), allow_fallback=False)
        np.testing.assert_allclose(xyz(result), self.vertices @ rotation.T + [2, -3, 1], atol=2e-14)

    def test_biharmonic_all_constrained_applies_nonrigid_displacements(self):
        target = self.vertices + np.array([[.1, 0, 0], [0, .2, 0], [0, 0, .3], [.4, 0, 0], [0, .5, 0]])
        result = self.logic.biharmonicWarpPolyData(self.mesh, points(self.vertices), points(target), allow_fallback=False)
        np.testing.assert_allclose(xyz(result), target, atol=1e-14)

    def test_biharmonic_averages_displacements_for_duplicate_nearest_controls(self):
        controls = np.vstack([self.vertices[:4], self.vertices[0] + [.01, 0, 0]])
        delta = np.array([[.1, 0, 0], [0, .2, 0], [0, 0, .3], [0, 0, .4], [.3, 0, 0]])
        result = self.logic.biharmonicWarpPolyData(self.mesh, controls, controls + delta, allow_fallback=False)
        np.testing.assert_allclose(xyz(result)[0], self.vertices[0] + [.2, 0, 0], atol=1e-14)

    def test_biharmonic_nonrigid_result_is_vertex_and_cell_order_invariant(self):
        controls = self.vertices[:4]
        target = controls.copy(); target[3, 2] += .2
        expected = xyz(self.logic.biharmonicWarpPolyData(self.mesh, points(controls), points(target), allow_fallback=False))
        permutation = np.array([4, 2, 0, 3, 1])
        inverse = np.argsort(permutation)
        permuted_mesh = polydata(self.vertices[permutation], inverse[self.faces[::-1, ::-1]])
        actual = xyz(self.logic.biharmonicWarpPolyData(permuted_mesh, points(controls), points(target), allow_fallback=False))
        np.testing.assert_allclose(actual[inverse], expected, atol=1e-12)

    def test_biharmonic_keeps_unused_and_duplicate_vertex_ids(self):
        vertices = np.vstack([self.vertices, [8, 8, 8], self.vertices[0]])
        mesh = polydata(vertices, self.faces)
        controls = self.vertices[:4]
        target = controls.copy(); target[3, 2] += .2
        result = self.logic.biharmonicWarpPolyData(mesh, controls, target, allow_fallback=False)
        self.assertEqual(result.GetNumberOfPoints(), len(vertices))
        np.testing.assert_allclose(xyz(result)[:4], target, atol=1e-14)
        np.testing.assert_array_equal(xyz(result)[5:], vertices[5:])

    def test_biharmonic_unconstrained_component_stays_at_rest(self):
        vertices = np.vstack([self.vertices, self.vertices + 20])
        mesh = polydata(vertices, np.vstack([self.faces, self.faces + 5]))
        target = self.vertices[:4].copy(); target[3, 2] += .2
        result = self.logic.biharmonicWarpPolyData(mesh, self.vertices[:4], target, allow_fallback=False)
        np.testing.assert_array_equal(xyz(result)[5:], vertices[5:])

    def test_biharmonic_solver_failure_raises_or_uses_checked_tps_fallback(self):
        controls = self.vertices[:4]
        target = controls.copy(); target[3, 2] += .2
        with mock.patch.dict(self.namespace, splu=mock.Mock(side_effect=RuntimeError("injected factorization failure"))):
            with self.assertRaisesRegex(RuntimeError, "fallback disabled.*injected factorization failure"):
                self.logic.biharmonicWarpPolyData(self.mesh, controls, target, allow_fallback=False)
            actual = self.logic.biharmonicWarpPolyData(self.mesh, controls, target, allow_fallback=True)
        expected = self.logic._warp_polydata_tps(self.mesh, points(controls), points(target))
        np.testing.assert_allclose(xyz(actual), xyz(expected), atol=1e-12)

    def test_biharmonic_rejects_invalid_inputs_instead_of_returning_unchanged_mesh(self):
        invalid = self.vertices[:4].copy(); invalid[0, 0] = np.nan
        for fallback in (False, True):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                self.logic.biharmonicWarpPolyData(self.mesh, invalid, self.vertices[:4], allow_fallback=fallback)
            with self.assertRaisesRegex(ValueError, "matching"):
                self.logic.biharmonicWarpPolyData(self.mesh, self.vertices[:2], self.vertices[:3], allow_fallback=fallback)

    def test_biharmonic_and_tps_identity_export_preserve_stored_template_order(self):
        # Include an unused off-surface vertex and a duplicate vertex. Only surface
        # IDs are sampled; neither cleaning nor deformation may reinterpret them.
        vertices = np.vstack([self.vertices, [8, 8, 8], self.vertices[0]])
        faces = self.faces.copy(); faces[-1, 0] = 6
        permutations = [np.arange(7), np.array([6, 5, 4, 2, 0, 3, 1])]
        for permutation in permutations:
            inverse = np.argsort(permutation)
            mesh = polydata(vertices[permutation], inverse[faces[::-1]])
            keep = self.logic.sample_indices_voxel_target(mesh, 100)
            self.assertNotIn(int(inverse[5]), keep)
            for backend in ("tps", "biharmonic"):
                with self.subTest(permutation=permutation.tolist(), backend=backend):
                    result = self.logic.sparseCorrespondenceBaseMesh(group([polydata(self.vertices[:4])]), group([mesh]), mesh, points(self.vertices[:4]), keep, warp_method=backend)[0]
                    np.testing.assert_allclose(result, xyz(mesh)[keep], atol=1e-6)

    def test_surface_sampling_still_rejects_nonfinite_unused_vertices(self):
        mesh = polydata(np.vstack([self.vertices, [np.nan, 0, 0]]), self.faces)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.logic.sample_indices_voxel_target(mesh, 100)

    def prepare_export(self, root):
        for name in ("models", "lms", "dense", "atlas"):
            (root / name).mkdir()
        (root / "models/sample.ply").write_bytes(b"input")
        (root / "lms/sample.mrk.json").write_bytes(b"input")
        self.logic.importPaired = lambda *args, **kwargs: (["sample"], group([self.mesh]), ["sample"], group([polydata(self.vertices[:4])]))
        return (Node(self.mesh), Node(array=self.vertices[:4]), str(root / "models"), str(root / "lms"), str(root / "dense"), str(root / "atlas"), 100)

    def test_dense_writer_failure_is_reported_at_each_output(self):
        for failed in (1, 2):
            with tempfile.TemporaryDirectory() as directory:
                args = self.prepare_export(Path(directory))
                self.slicer.writes.clear(); self.slicer.fail_write = failed
                messages = []
                with self.assertRaisesRegex(IOError, "Failed to save output"):
                    self.logic.exportDenseLMs(*args, progress=messages.append)
                self.assertEqual(len(self.slicer.writes), failed)
                self.assertEqual(len([text for text in messages if text.startswith("Saved")]), failed - 1)
                self.assertFalse(self.slicer.render_paused)

    def test_dense_success_writes_matching_template_and_population(self):
        vertices = np.vstack([self.vertices, [8, 8, 8], self.vertices[0]])
        faces = self.faces.copy(); faces[-1, 0] = 6
        permutation = np.array([6, 5, 4, 2, 0, 3, 1])
        self.mesh = polydata(vertices[permutation], np.argsort(permutation)[faces[::-1]])
        for backend in ("biharmonic", "tps"):
            with tempfile.TemporaryDirectory() as directory:
                self.slicer.writes.clear()
                args = self.prepare_export(Path(directory))
                self.logic.exportDenseLMs(*args, warp_method=backend, expected_specimens=["sample"])
                self.assertEqual(len(self.slicer.writes), 2)
                population = self.slicer.saved_arrays[self.slicer.writes[0]]
                template = self.slicer.saved_arrays[self.slicer.writes[1]]
                self.assertEqual(len(template), 6)  # Unused off-surface point excluded.
                np.testing.assert_allclose(population, template, atol=1e-6)

    def test_dense_export_rejects_incomplete_cohort_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.prepare_export(Path(directory))
            with self.assertRaisesRegex(ValueError, "specimen set"):
                self.logic.exportDenseLMs(*args, expected_specimens=["sample", "missing"])
            self.assertEqual(self.slicer.writes, [])
            with mock.patch.object(self.logic, "sparseCorrespondenceBaseMesh", return_value=[]):
                with self.assertRaisesRegex(ValueError, "missing specimens"):
                    self.logic.exportDenseLMs(*args)
            self.assertEqual(self.slicer.writes, [])

    def test_atlas_writer_failure_is_reported_at_each_output(self):
        for failed in (1, 2):
            with tempfile.TemporaryDirectory() as directory:
                self.slicer.writes.clear(); self.slicer.fail_write = failed
                with self.assertRaisesRegex(IOError, "Failed to save output"):
                    self.logic.saveAtlasOnly(Node(self.mesh), Node(array=self.vertices[:4]), directory)
                self.assertEqual(len(self.slicer.writes), failed)

    def test_checked_writer_rejects_missing_and_empty_success_results(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "output.ply"
            self.slicer.util.saveNode = lambda node, path: True
            with self.assertRaisesRegex(IOError, "missing or empty"):
                self.logic._save_node_checked(Node(), str(target))
            target.touch()
            with self.assertRaisesRegex(IOError, "missing or empty"):
                self.logic._save_node_checked(Node(), str(target))

    def test_alignment_writer_failures_cover_model_landmarks_and_semilandmarks(self):
        for failed in (1, 2, 3):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in ("models", "lms", "slms", "out-models", "out-lms", "out-slms"):
                    (root / name).mkdir()
                model_path, lm_path, slm_path = root / "models/sample.ply", root / "lms/sample.mrk.json", root / "slms/sample.mrk.json"
                for path in (model_path, lm_path, slm_path): path.write_bytes(b"input")
                self.slicer.models[str(model_path)] = self.mesh
                self.slicer.markups[str(lm_path)] = self.vertices[:4]
                self.slicer.markups[str(slm_path)] = self.vertices[:3]
                self.slicer.writes.clear(); self.slicer.fail_write = failed
                progress = []
                with self.assertRaisesRegex(IOError, "Failed to save output"):
                    self.logic.runAlign(Node(self.mesh), Node(array=self.vertices[:4]), str(root / "models"), str(root / "lms"), str(root / "out-models"), str(root / "out-lms"), False,
                        slmDir=str(root / "slms"), outSLMDir=str(root / "out-slms"), progress=lambda *args: progress.append(args))
                self.assertEqual(progress, [])
                self.assertEqual(len(self.slicer.writes), failed)
                self.assertFalse(self.slicer.render_paused)

    def test_invalid_library_name_stops_before_output_or_prompt(self):
        widget = self.namespace["MorphoWeaveAtlasBuilderWidget"]()
        widget.saveToLibraryCheck = types.SimpleNamespace(isChecked=lambda: True)
        widget._refreshLibraryPath = lambda: None
        widget.logic = self.logic
        widget._libraryTargetPath = mock.Mock(side_effect=AssertionError("target must not be used"))
        messages = []; widget._status = messages.append
        for name in ("", ".", "..", "../other", "/absolute", "a\\b"):
            widget._libraryName = lambda name=name: name
            widget._onRun()
            self.assertIn("safe, nonempty", messages[-1])
        widget._libraryTargetPath.assert_not_called()

    def test_library_handoff_passes_explicit_root(self):
        with tempfile.TemporaryDirectory() as directory:
            widget = self.namespace["MorphoWeaveAtlasBuilderWidget"]()
            widget.logic = self.logic; widget.libraryRootPath = directory
            widget._libraryName = lambda: "entry"
            widget._status = lambda text: None
            ingest = mock.Mock(return_value=(True, "saved"))
            module = types.SimpleNamespace(MorphoWeaveModelLibraryLogic=lambda: types.SimpleNamespace(ingestSSMDatabase=ingest))
            with mock.patch.dict(sys.modules, {"MorphoWeaveModelLibrary": module}):
                self.assertTrue(widget._saveSsmToLibrary({"atlas": directory, "population_correspondences": directory}))
            self.assertEqual(ingest.call_args.kwargs["libraryRootPath"], directory)
            self.assertEqual(ingest.call_args.kwargs["dbPath"], os.path.join(directory, "entry"))

    def test_widget_failures_never_ingest_or_claim_completion(self):
        for failing_step in ("runAlign", "saveAtlasOnly", "exportDenseLMs"):
            with tempfile.TemporaryDirectory() as directory:
                widget = self.namespace["MorphoWeaveAtlasBuilderWidget"]()
                toggle = lambda value: types.SimpleNamespace(isChecked=lambda: value)
                widget.saveToLibraryCheck = toggle(True); widget.keepAlignedOutputs = toggle(True)
                widget.loadAtlasRadio = toggle(True); widget.useSimilarity = toggle(False)
                widget.autoFallback = toggle(True); widget.overrideCoordCheck = toggle(False)
                widget.warpMethod = types.SimpleNamespace(currentIndex=0)
                widget.targetPointCount = types.SimpleNamespace(value=5)
                widget.outDir = widget.modelDir = widget.lmDir = types.SimpleNamespace(currentPath=directory)
                widget._refreshLibraryPath = lambda: None
                widget._libraryName = lambda: "entry"
                widget._libraryTargetPath = lambda: os.path.join(directory, "not-created")
                widget._loadAtlas = lambda: (Node(self.mesh), Node(array=self.vertices[:4]))
                widget._saveSsmToLibrary = mock.Mock(return_value=True)
                widget.logic = types.SimpleNamespace(isSafeLibraryName=self.logic.isSafeLibraryName, runAlign=mock.Mock(return_value=["sample"]), saveAtlasOnly=mock.Mock(), exportDenseLMs=mock.Mock())
                getattr(widget.logic, failing_step).side_effect = IOError("injected writer failure at /output/file")
                messages = []; widget._status = messages.append
                widget._onRun()
                widget._saveSsmToLibrary.assert_not_called()
                self.assertFalse(any(text.startswith("Completed") for text in messages))
                self.assertIn("/output/file", messages[-1])


if __name__ == "__main__":
    unittest.main()
