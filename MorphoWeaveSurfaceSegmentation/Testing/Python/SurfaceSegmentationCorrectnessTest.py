"""Run actual module methods with real NumPy, SciPy and VTK, without Slicer.

Only Slicer scene/file-loading and Qt UI boundaries use doubles. Numerical
methods, feature extraction, sparse clustering, VTK filters and mesh writers
are the production implementations, loaded unchanged from the module AST.
"""
import ast
import contextlib
import html
import json
import logging
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import vtk
import vtk.util.numpy_support as vtk_np
from scipy.cluster.vq import kmeans2
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

MODULE_DIR = Path(__file__).resolve().parents[2]
SOURCE_PATH = MODULE_DIR / "MorphoWeaveSurfaceSegmentation.py"


def load_production_classes():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    names = {"setWorkflowStatus", "MorphoWeaveSurfaceSegmentationLogic",
             "MorphoWeaveSurfaceSegmentationWidget"}
    body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in names]
    namespace = dict(np=np, vtk=vtk, vtk_np=vtk_np, os=os, tempfile=tempfile,
                     json=json, logging=logging, html=html, cKDTree=cKDTree,
                     coo_matrix=coo_matrix, diags=diags, eigsh=eigsh, kmeans2=kmeans2,
                     ScriptedLoadableModuleLogic=object, ScriptedLoadableModuleWidget=object,
                     qt=types.SimpleNamespace(Qt=types.SimpleNamespace(RichText=1)),
                     slicer=types.SimpleNamespace())
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace["MorphoWeaveSurfaceSegmentationLogic"], namespace["MorphoWeaveSurfaceSegmentationWidget"], namespace


Logic, Widget, MODULE_GLOBALS = load_production_classes()


def polydata(vertices, faces):
    pd = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(vtk_np.numpy_to_vtk(np.asarray(vertices, dtype=float), deep=True))
    pd.SetPoints(points)
    cells = vtk.vtkCellArray()
    for face in faces:
        cells.InsertNextCell(len(face))
        for index in face:
            cells.InsertCellPoint(int(index))
    pd.SetPolys(cells)
    return pd


def square():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return vertices, faces, polydata(vertices, faces)


def octahedron():
    vertices = np.array([[1, 0, 0], [0, 1.3, 0], [-1, 0, 0], [0, -1.3, 0],
                         [0, 0, 1.7], [0, 0, -1.7]], dtype=float)
    faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
                      [1, 0, 5], [2, 1, 5], [3, 2, 5], [0, 3, 5]], dtype=np.int32)
    return vertices, faces, polydata(vertices, faces)


class SurfaceSegmentationCorrectnessTest(unittest.TestCase):
    def setUp(self):
        self.logic = Logic()

    def test_shape_index_analytic_and_flat_convention(self):
        # Sphere, cup, convex/concave cylinder, saddle, and flat point.
        k1 = np.array([2., -2., 2., -2., 2., 0.])
        k2 = np.array([2., -2., 0., 0., -2., 0.])
        shape, curvedness = self.logic._shape_index_curvedness(k1, k2)
        np.testing.assert_allclose(shape, [1, -1, .5, -.5, 0, 0], atol=1e-14)
        np.testing.assert_allclose(curvedness, [2, 2, np.sqrt(2), np.sqrt(2), 2, 0])
        swapped, _ = self.logic._shape_index_curvedness(k2, k1)
        np.testing.assert_allclose(swapped, shape)

    def test_negative_mean_curvature_range_and_saddle_continuity(self):
        h = np.r_[-1.5, np.linspace(-1e-6, 1e-6, 101)]
        gaussian = np.r_[2., np.full(101, -1.)]
        first, second = self.logic._principal_from_HK(h, gaussian)
        shape, _ = self.logic._shape_index_curvedness(first, second)
        self.assertTrue(np.all((-1 <= shape) & (shape <= 1)))
        self.assertAlmostEqual(shape[0], 2 / np.pi * np.arctan2(-3., 1.))
        self.assertLess(np.max(np.abs(np.diff(shape[1:]))), 2e-8)
        # Existing standalone principal features retain magnitude ordering.
        self.assertTrue(np.all(np.abs(first) >= np.abs(second)))

    def test_shape_index_normal_inversion_changes_only_sign(self):
        h = np.array([-3., -1., 0., 1., 3.])
        gaussian = np.array([2., 0., -1., 0., 2.])
        s, c = self.logic._shape_index_curvedness(*self.logic._principal_from_HK(h, gaussian))
        reverse_s, reverse_c = self.logic._shape_index_curvedness(*self.logic._principal_from_HK(-h, gaussian))
        np.testing.assert_allclose(reverse_s, -s)
        np.testing.assert_allclose(reverse_c, c)

    def test_cotangent_operator_matches_analytic_triangle(self):
        vertices = np.array([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.]])
        laplace, mass = self.logic._cotangent_laplacian(vertices, np.array([[0, 1, 2]]))
        expected = np.array([[1.25, -.25, -1.], [-.25, .25, 0.], [-1., 0., 1.]])
        np.testing.assert_allclose(laplace.toarray(), expected, atol=1e-12)
        np.testing.assert_allclose(mass.diagonal(), np.full(3, 1 / 3))
        np.testing.assert_allclose(laplace.toarray().sum(axis=1), 0, atol=1e-14)
        self.assertGreaterEqual(np.linalg.eigvalsh(laplace.toarray()).min(), -1e-12)

    def test_cotangent_operator_face_order_winding_and_scale_invariance(self):
        vertices = np.array([[0., 0., 0.], [2., 0., 0.], [1., 3., 0.], [0., 1., 0.]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        laplace, mass = self.logic._cotangent_laplacian(vertices, faces)
        for altered in (faces[::-1], faces[:, ::-1], np.roll(faces, 1, axis=1)):
            other, other_mass = self.logic._cotangent_laplacian(vertices, altered)
            np.testing.assert_allclose(other.toarray(), laplace.toarray(), atol=1e-12)
            np.testing.assert_allclose(other_mass.diagonal(), mass.diagonal())
        scaled, scaled_mass = self.logic._cotangent_laplacian(vertices * 10, faces)
        np.testing.assert_allclose(scaled.toarray(), laplace.toarray(), atol=1e-12)
        np.testing.assert_allclose(scaled_mass.diagonal(), mass.diagonal() * 100)

    def test_knn_small_inputs_and_coincident_loci(self):
        for n in range(9):
            for coords in (np.c_[np.arange(n), np.zeros((n, 2))], np.zeros((n, 3))):
                graph = self.logic._knn_graph(coords, 12)
                self.assertEqual(graph.shape, (n, n))
                self.assertTrue(np.isfinite(graph.data).all())
                np.testing.assert_allclose(graph.diagonal(), 0)
                np.testing.assert_allclose(graph.toarray(), graph.T.toarray())
                if n > 1:
                    self.assertTrue(np.all(np.asarray(graph.sum(1)).ravel() > 0))

    def test_knn_rejects_invalid_coordinates_and_neighbor_count(self):
        for values in (np.zeros((3, 2)), np.array([[np.nan, 0, 0]])):
            with self.assertRaisesRegex(ValueError, "finite N-by-3"):
                self.logic._knn_graph(values, 12)
        for k in (0, -1, 1.5):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                self.logic._knn_graph(np.zeros((3, 3)), k)

    def test_spectral_small_inputs_reject_with_actionable_error(self):
        for n in range(3):
            with self.assertRaisesRegex(ValueError, "at least 3"):
                self.logic._spectral_cluster(coo_matrix((n, n)), 2)
        for k in (1, 3, 15, 2.5):
            with self.assertRaisesRegex(ValueError, "between 2 and 2"):
                self.logic._spectral_cluster(self.logic._knn_graph(np.eye(3), 12), k)

    def test_spectral_clustering_real_scipy_for_three_to_eight_loci(self):
        for n in range(3, 9):
            vertices = np.c_[np.arange(n), np.sin(np.arange(n)), np.zeros(n)]
            graph = self.logic._knn_graph(vertices, 12)
            labels = self.logic._spectral_cluster(graph, 2)
            self.assertEqual(labels.shape, (n,))
            self.assertEqual(set(labels.tolist()), {0, 1})

    def test_spectral_rejects_invalid_or_isolated_affinities(self):
        for graph, error in ((coo_matrix((3, 3)), "positive-weight"),
                             (coo_matrix(np.full((3, 3), np.nan)), "finite and nonnegative"),
                             (coo_matrix(-np.ones((3, 3))), "finite and nonnegative"),
                             (coo_matrix(np.triu(np.ones((3, 3)))), "symmetric")):
            with self.assertRaisesRegex(ValueError, error):
                self.logic._spectral_cluster(graph, 2)

    def test_hks_single_triangle_eigensolver_bound(self):
        vertices = np.array([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.]])
        faces = np.array([[0, 1, 2]])
        eigenvalues, eigenvectors = self.logic._lb_eigens(vertices, faces, k=30)
        self.assertEqual(eigenvalues.shape, (2,))
        self.assertEqual(eigenvectors.shape, (3, 2))
        self.assertTrue(np.isfinite(self.logic._hks_at_vertices(vertices, faces)).all())
        with self.assertRaisesRegex(ValueError, "at least 3"):
            self.logic._lb_eigens(vertices[:2], np.empty((0, 3), dtype=int))

    def test_labeled_vtp_preserves_complete_geometry_and_both_arrays(self):
        _, _, pd = square()
        raw = np.array([0, 0, 1, 1]); smoothed = np.array([0, 1, 1, 1])
        with tempfile.TemporaryDirectory() as folder:
            path = self.logic._write_vtp_with_labels_world("sample.ply", folder, raw, smoothed, pdW=pd)
            self.assertEqual(Path(path).name, "sample_seg.vtp")
            reader = vtk.vtkXMLPolyDataReader(); reader.SetFileName(path); reader.Update()
            actual = reader.GetOutput()
            self.assertEqual(actual.GetNumberOfPolys(), 2)
            self.assertEqual(actual.GetNumberOfPoints(), 4)
            np.testing.assert_array_equal(vtk_np.vtk_to_numpy(actual.GetPointData().GetArray("SegID")), raw)
            np.testing.assert_array_equal(vtk_np.vtk_to_numpy(actual.GetPointData().GetArray("SegID_smooth")), smoothed)
            self.assertEqual(actual.GetPointData().GetScalars().GetName(), "SegID_smooth")
            self.assertIsNone(pd.GetPointData().GetArray("SegID"))

    def test_mixed_label_ply_interiors_are_reported_not_counted_as_writes(self):
        _, _, pd = square()
        with tempfile.TemporaryDirectory() as folder:
            report = self.logic._write_ply_segments(pd, folder, np.array([0, 0, 1, 1]), "sample.ply")
            self.assertEqual(report['files'], [])
            self.assertEqual(report['empty_segments'], [0, 1])
            self.assertEqual(report['excluded_cells'], 2)
            self.assertEqual(report['written_cells'], 0)
            self.assertEqual(report['failures'], [])
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_single_label_ply_and_tiny_region_report_actual_outputs(self):
        _, _, pd = square()
        for labels, cell_count, excluded, empty in (([0, 0, 0, 0], 2, 0, []), ([0, 0, 0, 1], 1, 1, [1])):
            with tempfile.TemporaryDirectory() as folder:
                report = self.logic._write_ply_segments(pd, folder, np.array(labels), "sample.ply")
                self.assertEqual(len(report['files']), 1)
                self.assertEqual(report['excluded_cells'], excluded)
                self.assertEqual(report['empty_segments'], empty)
                self.assertEqual(report['written_cells'], cell_count)
                reader = vtk.vtkPLYReader(); reader.SetFileName(report['files'][0]); reader.Update()
                self.assertEqual(reader.GetOutput().GetNumberOfPolys(), cell_count)

    def test_invalid_label_count_and_noninteger_labels_fail_before_write(self):
        _, _, pd = square()
        with tempfile.TemporaryDirectory() as folder:
            for labels in (np.array([0, 1]), np.array([0, 1, 0, np.nan]), np.array([0, 1, 0, .5])):
                with self.assertRaises(ValueError):
                    self.logic._write_vtp_with_labels_world("sample.ply", folder, labels, labels, pdW=pd)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_failed_destination_is_not_counted_and_old_file_is_preserved(self):
        _, _, pd = square()
        labels = np.zeros(4, dtype=int)
        with tempfile.TemporaryDirectory() as folder:
            # A directory at the final path is a real filesystem replace failure,
            # including when tests run as root (chmod-based tests would not be).
            destination = Path(folder) / "sample_seg.vtp"; destination.mkdir()
            with self.assertRaises(OSError):
                self.logic._write_vtp_with_labels_world("sample.ply", folder, labels, labels, pdW=pd)
            self.assertTrue(destination.is_dir())
            self.assertEqual(list(Path(folder).iterdir()), [destination])
            (Path(folder) / "sample_seg_00.ply").mkdir()
            report = self.logic._write_ply_segments(pd, folder, labels, "sample.ply")
            self.assertEqual(report['files'], [])
            self.assertEqual(len(report['failures']), 1)
            self.assertEqual(report['written_cells'], 0)

    def test_writer_return_code_is_checked_and_partial_temp_is_removed(self):
        _, _, pd = square()
        class FailingWriter:
            # Deliberate writer boundary double for otherwise rare VTK failures.
            def SetFileName(self, path): self.path = path
            def SetInputData(self, data): pass
            def Write(self):
                Path(self.path).write_bytes(b"partial")
                return 0
            def GetErrorCode(self): return 1
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "existing.vtp"; out.write_bytes(b"old-valid-content")
            with self.assertRaisesRegex(IOError, "Failed to write mesh"):
                self.logic._checked_polydata_write(FailingWriter(), pd, str(out))
            self.assertEqual(out.read_bytes(), b"old-valid-content")
            self.assertEqual(list(Path(folder).iterdir()), [out])

    def test_pairing_case_insensitive_and_duplicate_stems_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            meshes = Path(folder) / "meshes"; meshes.mkdir()
            markups = Path(folder) / "markups"; markups.mkdir()
            (meshes / "Sample.PLY").touch(); (markups / "sample.MRK.JSON").touch()
            self.assertEqual(self.logic._pair_files(str(meshes), str(markups)),
                             [(str(meshes / "Sample.PLY"), str(markups / "sample.MRK.JSON"))])
            duplicate = meshes / "sample.vtp"; duplicate.touch()
            with self.assertRaisesRegex(ValueError, "Duplicate specimen stem"):
                self.logic._pair_files(str(meshes), str(markups))
            duplicate.unlink(); (markups / "Sample.mrk.json").touch()
            with self.assertRaisesRegex(ValueError, "Duplicate specimen stem"):
                self.logic._pair_files(str(meshes), str(markups))


class SurfaceSegmentationRunBoundaryTest(unittest.TestCase):
    """Replace only scene/loading boundaries; run all numerical methods/writers."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.meshes = self.folder / "meshes"; self.meshes.mkdir()
        self.markups = self.folder / "markups"; self.markups.mkdir()
        self.out = self.folder / "output"
        self.logic = Logic()
        self.vertices, self.faces, self.pd = octahedron()
        self.logic._clear_preview_nodes = lambda: None
        self.logic._load_mesh_VF_world = lambda path: (self.vertices.copy(), self.faces.copy(), self.pd)
        self.logic._load_mrk_positions_world = lambda path: self.vertices.copy()
        for name in ("good", "other"):
            (self.meshes / (name + ".vtp")).touch()
            (self.markups / (name + ".mrk.json")).touch()

    def run_segmentation(self, **kwargs):
        return self.logic.run(str(self.meshes), str(self.markups), str(self.out),
                              k=2, knn=12, previewN=0, **kwargs)

    def test_actual_run_writes_full_vtps_and_preserves_raw_vs_smooth_arrays(self):
        result = self.run_segmentation(smooth_iters=1)
        self.assertEqual(result['n_pairs'], 2)
        self.assertEqual(result['n_used'], 2)
        self.assertEqual(result['n_skipped'], 0)
        self.assertEqual(result['n_export_failed'], 0)
        self.assertEqual(result['wrote'], 2)
        self.assertEqual(result['failures'], [])
        self.assertEqual(len(result['lookup_files']), 2)
        labels = np.load(result['lookup_files'][0])
        expected_smooth = self.logic._smooth_labels(self.vertices, self.faces, labels.copy(), 1)
        for path in result['output_files']:
            reader = vtk.vtkXMLPolyDataReader(); reader.SetFileName(path); reader.Update()
            actual = reader.GetOutput()
            self.assertEqual(actual.GetNumberOfPolys(), len(self.faces))
            np.testing.assert_array_equal(vtk_np.vtk_to_numpy(actual.GetPointData().GetArray("SegID")), labels)
            np.testing.assert_array_equal(vtk_np.vtk_to_numpy(actual.GetPointData().GetArray("SegID_smooth")), expected_smooth)
        self.assertEqual(list(self.out.glob("*.ply")), [])

    def test_feature_failure_reports_specimen_without_hiding_success(self):
        def load_mesh(path):
            if Path(path).stem == "other": raise RuntimeError("unreadable specimen")
            return self.vertices.copy(), self.faces.copy(), self.pd
        self.logic._load_mesh_VF_world = load_mesh
        result = self.run_segmentation()
        self.assertEqual((result['n_pairs'], result['n_used'], result['n_skipped'], result['wrote']), (2, 1, 1, 1))
        self.assertEqual(len(result['failures']), 1)
        self.assertIn("other.vtp", result['failures'][0]['mesh'])
        self.assertIn("unreadable specimen", result['failures'][0]['error'])

    def test_all_feature_failures_raise_with_specimen_names(self):
        def fail(path): raise RuntimeError("unreadable specimen")
        self.logic._load_mesh_VF_world = fail
        with self.assertRaisesRegex(RuntimeError, "All meshes failed.*good.vtp.*other.vtp"):
            self.run_segmentation()
        self.assertEqual(list(self.out.glob("*.vtp")), [])

    def test_export_failure_reports_actual_files_and_continues(self):
        self.out.mkdir(); (self.out / "good_seg.vtp").mkdir()
        result = self.run_segmentation()
        self.assertEqual(result['n_export_failed'], 1)
        self.assertEqual(result['wrote'], 1)
        self.assertEqual([Path(p).name for p in result['output_files']], ["other_seg.vtp"])
        self.assertEqual(result['failures'][0]['stage'], "VTP export")

    def test_all_export_failures_are_not_reported_as_writes(self):
        self.out.mkdir()
        for name in ("good", "other"): (self.out / (name + "_seg.vtp")).mkdir()
        result = self.run_segmentation()
        self.assertEqual(result['n_export_failed'], 2)
        self.assertEqual(result['wrote'], 0)
        self.assertEqual(result['output_files'], [])
        self.assertEqual(len(result['failures']), 2)

    def test_ply_is_explicit_and_reports_omitted_cells_from_actual_run(self):
        result = self.run_segmentation(write_ply_segments=True, smooth_iters=0)
        self.assertEqual(len(result['segment_reports']), 2)
        actual = [str(path) for path in self.out.iterdir() if path.suffix in (".vtp", ".ply")]
        self.assertEqual(sorted(actual), sorted(result['output_files']))
        self.assertEqual(result['wrote'], len(actual))
        self.assertTrue(any(report['excluded_cells'] > 0 for report in result['segment_reports']))
        self.assertTrue(any("PLY interiors omit" in warning for warning in result['warnings']))
        for report in result['segment_reports']:
            self.assertEqual(report['total_cells'], len(self.faces))
            self.assertEqual(report['interior_cells'] + report['excluded_cells'], len(self.faces))

    def test_ply_pipeline_error_does_not_hide_successful_vtp_outputs(self):
        def fail(*args): raise RuntimeError("segment filter failed")
        self.logic._write_ply_segments = fail
        result = self.run_segmentation(write_ply_segments=True)
        self.assertEqual(result['wrote'], 2)
        self.assertEqual(result['n_export_failed'], 2)
        self.assertEqual(len(result['failures']), 2)
        self.assertTrue(all(error['stage'] == "PLY export" for error in result['failures']))

    def test_disabled_mesh_exports_still_report_label_lookup_files(self):
        result = self.run_segmentation(write_vtp=False)
        self.assertEqual(result['wrote'], 0)
        self.assertEqual(result['output_files'], [])
        self.assertTrue(all(Path(path).is_file() for path in result['lookup_files']))

    def test_blank_output_and_too_few_loci_reject_before_writing(self):
        with self.assertRaisesRegex(ValueError, "explicit output"):
            self.logic.run(str(self.meshes), str(self.markups), " ", k=2)
        for n in range(3):
            self.logic._load_mrk_positions_world = lambda path, n=n: np.zeros((n, 3))
            with self.assertRaisesRegex(ValueError, "at least 3"):
                self.run_segmentation()
        self.assertFalse(self.out.exists())

    def test_invalid_auto_tune_candidates_reject_before_writing(self):
        for grid, alpha in (([2, 6], [.7]), ([2], [1.5])):
            with self.assertRaises(ValueError):
                self.run_segmentation(auto_tune=True, k_grid=grid, alpha_grid=alpha)
        self.assertFalse(self.out.exists())

    def test_preview_failure_is_reported_but_does_not_hide_saved_files(self):
        def fail(**kwargs): raise RuntimeError("scene unavailable")
        self.logic._preview_in_scene = fail
        result = self.logic.run(str(self.meshes), str(self.markups), str(self.out), k=2, previewN=1)
        self.assertEqual(result['wrote'], 2)
        self.assertEqual(result['n_export_failed'], 0)
        self.assertIn("scene unavailable", result['warnings'][0])


class SurfaceSegmentationWidgetBoundaryTest(unittest.TestCase):
    """No live Qt: minimal text/control doubles exercise actual validation/UI reporting."""
    class Label:
        def setTextFormat(self, value): pass
        def setText(self, value): self.text = value
        def setAccessibleName(self, value): self.accessible = value

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.meshes = self.folder / "meshes"; self.meshes.mkdir()
        self.markups = self.folder / "markups"; self.markups.mkdir()
        (self.meshes / "SAMPLE.ply").touch(); (self.markups / "sample.mrk.json").touch()
        self.widget = Widget(); self.widget.logic = Logic()
        self.widget.meshDir = types.SimpleNamespace(text=str(self.meshes))
        self.widget.mrkDir = types.SimpleNamespace(text=str(self.markups))
        self.widget.outDir = types.SimpleNamespace(text="")
        self.widget.runBtn = types.SimpleNamespace(enabled=True)
        self.widget.statusLbl = self.Label()

    def test_empty_output_does_not_become_current_working_directory(self):
        self.widget._validateInputs()
        self.assertFalse(self.widget.runBtn.enabled)
        self.assertIn("Choose an output folder", self.widget.statusLbl.text)
        self.widget.outDir.text = str(self.folder / "output")
        self.widget._validateInputs()
        self.assertTrue(self.widget.runBtn.enabled)
        self.assertIn("1 paired specimens", self.widget.statusLbl.text)

    def test_duplicate_pair_stems_are_not_shown_ready(self):
        self.widget.outDir.text = str(self.folder / "output")
        (self.meshes / "sample.vtp").touch()
        self.widget._validateInputs()
        self.assertFalse(self.widget.runBtn.enabled)
        self.assertIn("Duplicate specimen stem", self.widget.statusLbl.text)

    def test_on_run_surfaces_skipped_and_export_failed_specimens(self):
        messages = []; warnings = []
        fake_slicer = types.SimpleNamespace(util=types.SimpleNamespace(
            tryWithErrorDisplay=lambda *a, **k: contextlib.nullcontext(),
            showStatusMessage=messages.append, warningDisplay=warnings.append))
        self.widget.outDir.text = str(self.folder / "output")
        self.widget.autoTune = types.SimpleNamespace(checked=False)
        self.widget.saveVTP = types.SimpleNamespace(checked=True)
        self.widget.savePLY = types.SimpleNamespace(checked=False)
        for name, value in (("kSeg", 2), ("knn", 12), ("smooth", 1), ("alpha", .7), ("previewN", 0)):
            setattr(self.widget, name, types.SimpleNamespace(value=value))
        self.widget.logic.run = lambda *a, **k: dict(
            n_used=2, n_pairs=3, n_skipped=1, n_export_failed=1, k=2, wrote=1,
            failures=[dict(mesh="failed.ply", stage="VTP export", error="writer failed")], warnings=[])
        with patch.dict(MODULE_GLOBALS, slicer=fake_slicer):
            self.widget.onRun()
        self.assertIn("2/3 specimens used, 1 skipped, 1 export failures", messages[0])
        self.assertIn("Finished with issues", self.widget.statusLbl.text)
        self.assertIn("failed.ply (VTP export): writer failed", warnings[0])


if __name__ == "__main__":
    unittest.main()
