"""Audit regression tests: actual core/VTK methods, with native checks when available."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
import numpy as np
import vtk
import vtk.util.numpy_support as vtk_np

ROOT = Path(__file__).resolve().parents[2]
RESOURCES = ROOT / 'Resources' / 'Python'
sys.path.insert(0, str(RESOURCES))
from MorphoWeaveShapeCompletionCore import (
    CalibrationAccumulator, CompletionSettings, RidgeLogisticCalibrator,
    auc_rank, coordinate_compatibility, finalize_calibration_entry,
    rms_radius, run_completion,
)
from MorphoWeaveShapeCompletionGeometry import closest_surface_points
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ShapeCompletionCoreTest import _FakeRustCPD, _Pose, synthetic_ssm


class CompletionAuditRegressionTest(unittest.TestCase):
    def test_atlas_normalizes_geometry_and_explicit_variance_once(self):
        mean, modes, eig = synthetic_ssm(40, 2)
        backend = _FakeRustCPD()
        result = run_completion(mean * 3. + [10., 20., 30.], mean, modes, eig,
                                CompletionSettings(), rustcpd_module=backend)
        target, _, _, kwargs = backend.atlas_call
        self.assertTrue(kwargs['normalize'])
        self.assertAlmostEqual(kwargs['sigma2'] * rms_radius(target)**2,
                               result.atlas_initial_sigma2)
        np.testing.assert_array_equal(backend.pose_call[2], modes.reshape(mean.size, 2))

    def test_coarse_triangle_landmarks_are_not_frame_mismatches(self):
        vertices = np.array([[0., 0., 0.], [10., 0., 0.], [0., 10., 0.]])
        query = np.array([[2., 2., 0.], [4., 2., 0.], [2., 4., 0.]])
        surface = vtk.vtkPolyData(); points = vtk.vtkPoints()
        points.SetData(vtk_np.numpy_to_vtk(vertices, deep=True)); surface.SetPoints(points)
        cells = vtk.vtkCellArray(); cells.InsertNextCell(3)
        for i in range(3): cells.InsertCellPoint(i)
        surface.SetPolys(cells)
        # Exercise the actual integration method, not a replacement formula.
        path = RESOURCES / 'MorphoWeaveShapeCompletionBase.py'
        tree = ast.parse(path.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == 'model_coordinate_compatibility')
        ns = {'np': np, 'vtk_np': vtk_np,
              'closest_surface_points': closest_surface_points,
              'coordinate_compatibility': coordinate_compatibility}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), ns)
        logic = types.SimpleNamespace(model_polydata_world=lambda node: surface)
        stats = ns[method.name](logic, object(), query, 'target landmarks',
                                max_median_fraction=.15, max_p95_fraction=.35)
        self.assertEqual(stats['median_distance'], 0.)
        with self.assertRaises(ValueError):
            ns[method.name](logic, object(), query + [0., 0., 100.], 'target landmarks')

    def test_supplied_surface_distances_fail_closed(self):
        points = np.eye(3)
        for bad in (np.zeros(2), np.array([0., -1., 0.]), np.full(3, np.nan)):
            with self.assertRaises(ValueError):
                coordinate_compatibility(points, points, 'points', surface_distances=bad)

    def test_reported_auc_matches_fitted_probability_both_slope_signs(self):
        for increasing in (True, False):
            accumulator = CalibrationAccumulator(.5, False, 'exact_dense_correspondence')
            for i in range(12):
                accumulator.add(fit_id=str(i), group=str(i), errors=np.ones(20),
                                variances=np.ones(20), normalized_sigma2=float(np.exp(i-6)),
                                successful=(i >= 6) == increasing, missing_rms=1.)
            entry = finalize_calibration_entry(accumulator, alpha=.1,
                                               success_threshold_fraction=.01)
            classifier = RidgeLogisticCalibrator.from_dict(entry['fit_success_calibrator'])
            probability = classifier.probability(accumulator.normalized_sigma2)
            self.assertEqual(entry['fit_success_apparent_metrics']['auc'],
                             auc_rank(probability, accumulator.successful))
            self.assertEqual(entry['fit_success_apparent_metrics']['auc'], 1.)

    def test_native_final_atlas_is_unit_equivariant_at_fixed_iterations(self):
        try:
            import rustcpd
        except ImportError:
            self.skipTest('rustcpd is not installed')
        if rustcpd.__version__ != '4.0.0':
            self.skipTest('requires pinned rustcpd 4.0.0')
        class NativeAtlasWithFixedPose:
            def pose_initialize(self, source, target, modes, eigenvalues, **kwargs):
                return _Pose(len(eigenvalues))
            def register_atlas(self, *args, **kwargs):
                return rustcpd.register_atlas(*args, **kwargs)
        rng = np.random.default_rng(42)
        mean = rng.normal(size=(48, 3)) * [3., 2., 1.]
        u, _ = np.linalg.qr(rng.normal(size=(mean.size, 2)))
        modes = u.reshape(len(mean), 3, 2); eigenvalues = np.array([4., 1.])
        target = mean + (u @ np.array([.6, -.3])).reshape(mean.shape) + [.3, -.2, .1]
        settings = CompletionSettings(coverage=1., variance_keep=1., coverage_prescale=False,
                                      residual_scale_policy='fixed', atlas_max_iterations=30,
                                      atlas_tolerance=0., parallel=False)
        results = []
        for unit in (1., .001, 1000.):
            r = run_completion(target*unit, mean*unit, modes, eigenvalues*unit**2,
                               settings, rustcpd_module=NativeAtlasWithFixedPose())
            results.append((r.completed_points/unit, r.total_variance/unit**2))
        for points, variance in results[1:]:
            np.testing.assert_allclose(points, results[0][0], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(variance, results[0][1], atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
