import ast
import tempfile
import sys
import unittest
from pathlib import Path

import numpy as np
import vtk
import vtk.util.numpy_support as vtk_np
from scipy.spatial.transform import Rotation

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from LogicTestSupport import Node, SOURCE_PATH, load_logic, polydata


class GeometryCorrectnessTest(unittest.TestCase):
    def setUp(self):
        self.logic, self.namespace = load_logic()
        self.vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
        self.faces = np.array([[0, 1, 2], [0, 2, 3]])
        self.plane = polydata(self.vertices, self.faces)

    def project(self, points, limit, surface=None, **kwargs):
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(vtk_np.numpy_to_vtk(np.asarray(points, float).reshape(-1, 3), deep=True))
        output = self.logic.projectPointsPolydataCoherent(
            self.plane if surface is None else surface, vtk_points, limit,
            use_outer_surface=False, **kwargs)
        return vtk_np.vtk_to_numpy(output.GetPoints().GetData())

    def test_projection_all_fixed_preserves_count_order_and_coordinates(self):
        for n in (0, 1, 3, 12):
            points = np.c_[np.linspace(.1, .9, n), np.full(n, .5), np.zeros(n)]
            with self.subTest(n=n):
                np.testing.assert_array_equal(self.project(points, .1), points)

    def test_projection_small_mixed_sets_and_duplicates_are_finite(self):
        for n in range(1, 9):
            points = np.c_[np.linspace(.1, .9, n), np.full(n, .5), np.full(n, .05)]
            if n > 1:
                points[0, 2] = 0
            output = self.project(points, .2)
            self.assertEqual(output.shape, points.shape)
            self.assertTrue(np.isfinite(output).all())
            np.testing.assert_allclose(output[:, 2], 0, atol=1e-14)
        L, W, _ = self.logic._knn_graph_laplacian(np.zeros((8, 3)), k=6)
        self.assertTrue(np.isfinite(L.data).all())
        np.testing.assert_array_equal(W.diagonal(), 0)
        for n in (0, 1):
            L, W, _ = self.logic._knn_graph_laplacian(np.zeros((n, 3)), k=6)
            self.assertEqual(L.shape, (n, n))
            self.assertEqual(L.nnz, 0)

    def test_projection_preserves_unreachable_points_and_honors_final_bound(self):
        points = np.c_[np.linspace(.1, .9, 8), np.full(8, .5), np.full(8, 10.)]
        np.testing.assert_array_equal(self.project(points, .05), points)
        # Nearby points on either side of an oriented surface and distant ones.
        points = np.array([[.2, .2, -.03], [.5, .5, .03], [.8, .8, 3.]])
        result = self.project(points, .1)
        self.assertTrue(np.all(np.linalg.norm(result - points, axis=1) <= .1))
        np.testing.assert_allclose(result[:2, 2], 0, atol=1e-14)
        np.testing.assert_array_equal(result[2], points[2])
        np.testing.assert_array_equal(self.project(points, 0), points)
        # When smoothing pushes a reachable point sideways, use its own nearest
        # surface point instead of abandoning an achievable projection.
        points = np.array([[.1, .1, .09], [.9, .9, .09]])
        result = self.project(points, .1, lambda_cohere=100)
        np.testing.assert_allclose(result[:, 2], 0, atol=1e-14)
        self.assertTrue(np.all(np.linalg.norm(result - points, axis=1) <= .1))

    def test_projection_without_snap_also_honors_travel_bound(self):
        points = np.array([[.2, .2, -1.], [.5, .5, 1.], [.8, .8, 3.]])
        result = self.project(points, .05, snap_to_surface=False)
        self.assertTrue(np.all(np.linalg.norm(result - points, axis=1) <= .05 + 1e-14))
        np.testing.assert_array_equal(self.project(points, 0, snap_to_surface=False), points)

    def test_projection_closed_surface_inside_outside_and_empty_surface(self):
        sphere = vtk.vtkSphereSource(); sphere.SetRadius(1); sphere.SetThetaResolution(30)
        sphere.SetPhiResolution(30); sphere.Update()
        points = np.array([[0, 0, .98], [0, 0, 1.02], [0, 0, 0], [0, 0, 5.]])
        result = self.project(points, .1, surface=sphere.GetOutput())
        self.assertTrue(np.all(np.linalg.norm(result - points, axis=1) <= .1))
        np.testing.assert_array_equal(result[2:], points[2:])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.project([[0, 0, 0]], .1, surface=vtk.vtkPolyData())

    def test_public_projection_preserves_markups_count(self):
        points = np.array([[.1, .1, 0], [.7, .2, 0], [.3, .8, 0]])
        output = self.logic.runPointProjection(Node(self.plane), Node(self.plane), Node(points=points), .1)
        np.testing.assert_allclose(output.points, points, atol=3e-8)

    def test_cotangent_matches_analytic_triangle_and_face_order(self):
        vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]], float)
        L, M = self.logic._cotangent_laplacian(vertices, np.array([[0, 1, 2]]))
        expected = -np.array([[1.25, -.25, -1], [-.25, .25, 0], [-1, 0, 1]])
        np.testing.assert_allclose(L.toarray(), expected)
        np.testing.assert_allclose(M.diagonal(), np.full(3, 1/3))
        vertices = np.array([[0, 0, 0], [2, 0, 0], [1, 3, 0], [0, 1, 0]], float)
        L, M = self.logic._cotangent_laplacian(vertices, self.faces)
        for faces in (self.faces[::-1], self.faces[:, ::-1], self.faces[::-1, ::-1]):
            other, mass = self.logic._cotangent_laplacian(vertices, faces)
            np.testing.assert_allclose(other.toarray(), L.toarray())
            np.testing.assert_allclose(mass.toarray(), M.toarray())

    def bump_fixture(self):
        vertices = np.vstack([self.vertices, [.5, .5, .5]])
        faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        return vertices, faces

    def test_biharmonic_identity_translation_and_nonvertex_controls(self):
        vertices, faces = self.bump_fixture()
        for offset in (np.zeros(3), np.array([.02, .01, .03])):
            for shift in (np.zeros(3), np.array([2., -3., .5])):
                controls = vertices[:4] + offset
                before, after, _ = self.logic.warp_model_biharmonic(
                    Node(polydata(vertices, faces)), controls, controls + shift)
                np.testing.assert_allclose(before, vertices, atol=1e-14)
                np.testing.assert_allclose(after, vertices + shift, atol=2e-10)

    def test_biharmonic_coordinate_rotation_equivariance_and_face_order(self):
        vertices, faces = self.bump_fixture()
        controls = vertices[:4]
        destination = controls.copy(); destination[0] += [.1, -.2, .15]
        _, expected, _ = self.logic.warp_model_biharmonic(Node(polydata(vertices, faces)), controls, destination)
        rotation = Rotation.from_euler("xyz", [23, -11, 43], degrees=True).as_matrix()
        translation = np.array([4, -3, 2])
        moved = lambda p: p @ rotation.T + translation
        for reordered in (faces[::-1], faces[:, ::-1]):
            _, actual, _ = self.logic.warp_model_biharmonic(
                Node(polydata(moved(vertices), reordered)), moved(controls), moved(destination))
            np.testing.assert_allclose(actual, moved(expected), atol=2e-10)

    def test_biharmonic_unconstrained_components_and_empty_controls_remain_unchanged(self):
        vertices, faces = self.bump_fixture()
        all_vertices = np.vstack([vertices, vertices + [5, 0, 0], [99, 99, 99]])
        all_faces = np.vstack([faces, faces + len(vertices)])
        _, after, _ = self.logic.warp_model_biharmonic(
            Node(polydata(all_vertices, all_faces)), vertices[:4], vertices[:4] + [1, 2, 3])
        np.testing.assert_allclose(after[:5], vertices + [1, 2, 3], atol=2e-10)
        np.testing.assert_array_equal(after[5:], all_vertices[5:])
        _, after, _ = self.logic.warp_model_biharmonic(
            Node(polydata(all_vertices, all_faces)), np.zeros((0, 3)), np.zeros((0, 3)))
        np.testing.assert_array_equal(after, all_vertices)

    def test_biharmonic_degenerate_triangle_does_not_bridge_unconstrained_island(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                             [3, 0, 0], [4, 0, 0], [3, 1, 0]], float)
        # The last triangle has zero area and is ignored by the stiffness
        # operator. It must also be ignored when choosing constrained components.
        faces = np.array([[0, 1, 2], [3, 4, 5], [1, 3, 4]])
        shift = np.array([.1, -.2, .3])
        _, after, _ = self.logic.warp_model_biharmonic(
            Node(polydata(vertices, faces)), vertices[:1], vertices[:1] + shift)
        np.testing.assert_allclose(after[:3], vertices[:3] + shift, atol=2e-10)
        np.testing.assert_array_equal(after[3:], vertices[3:])

    def test_tps_smoothing_damps_nonaffine_residual_and_zero_interpolates(self):
        source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [.25, .25, .25]])
        target = source.copy(); target[-1, 2] += 1
        affine_basis = np.c_[np.ones(len(source)), source]
        affine = affine_basis @ np.linalg.lstsq(affine_basis, target, rcond=None)[0]
        residuals = []
        for smoothing in (0, .01, .1, .3, 1, 10, 100):
            result = self.logic.warp_points_tps(source, source, target, lam=smoothing)
            self.assertTrue(np.isfinite(result).all())
            residuals.append(np.linalg.norm(result - affine))
            if smoothing == 0:
                np.testing.assert_allclose(result, target, atol=1e-13)
        self.assertTrue(np.all(np.diff(residuals) < 0))
        self.assertLess(residuals[-1], residuals[0] / 100)

    def test_tps_small_affine_fallback_and_no_controls(self):
        source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        for n in range(1, 5):
            target = source[:n] + [5, 6, 7]
            output = self.logic.warp_points_tps(source[:n], source[:n], target)
            np.testing.assert_allclose(output, target, atol=1e-12)
        np.testing.assert_array_equal(self.logic.warp_points_tps(source, source[:0], source[:0]), source)
        output = self.logic.warp_points_tps(source, source, source + [5, 6, 7], max_corr=3)
        self.assertTrue(np.isfinite(output).all())
        used = self.logic._fps_indices(source, 3)
        np.testing.assert_allclose(output[used], (source + [5, 6, 7])[used], atol=1e-12)
        for smoothing in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                self.logic.warp_points_tps(source, source, source, lam=smoothing)

    def test_vtp_writer_checks_errors_and_successful_file(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "mesh.vtp"
            self.logic.savePolyDataVTP(self.plane, str(output))
            reader = vtk.vtkXMLPolyDataReader(); reader.SetFileName(str(output)); reader.Update()
            self.assertEqual(reader.GetOutput().GetNumberOfPoints(), len(self.vertices))
            with self.assertRaises(RuntimeError):
                self.logic.savePolyDataVTP(self.plane, str(output / "cannot-write.vtp"))

    def test_standardized_legacy_prior_score_and_pruning_are_unit_invariant(self):
        # Execute the production nested score function with deterministic geometry
        # scores, not the unavailable tiny3d RANSAC boundary. No penalty reimplementation.
        tree = ast.parse(SOURCE_PATH.read_text())
        logic_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name.endswith("Logic"))
        method = next(n for n in logic_class.body if isinstance(n, ast.FunctionDef) and n.name == "initialize_template")
        nodes = [n for n in method.body if isinstance(n, ast.FunctionDef) and n.name in ("clip_b", "ssm_sample", "score_geom")]
        scores = []
        for units in (.001, 1., 1000.):
            calls = []
            namespace = dict(np=np, span=3., Uw_flat=np.array([[2.], [0], [0]]) * units,
                             mean_flat=np.zeros(3), M=1, oldCorr=np.zeros((1, 3)), w_rmse=.6,
                             w_reg=.1, bound_margin=1e-5, LARGE=1e6, cache={},
                             parameters={}, best={"score": -.5})
            def evaluate(*args):
                calls.append(True)
                return np.eye(4), .8, .1
            namespace["eval_bestof"] = evaluate
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
            score = namespace["score_geom"]([2.], 10, False, 1)
            self.assertAlmostEqual(score, -(.8 - .6 * .1 - .1 * 4))
            self.assertEqual(len(calls), 1)
            namespace["best"]["score"] = .9
            namespace["cache"].clear()
            pruned = namespace["score_geom"]([2.], 10, False, 1)
            self.assertEqual(pruned, 1e6 + 4)
            self.assertEqual(len(calls), 1)
            scores.append((score, pruned))
        np.testing.assert_array_equal(scores, [scores[0]] * 3)


class NativeCovarianceCorrectnessTest(unittest.TestCase):
    def test_native_atlas_is_unit_invariant_after_prescale_and_similarity(self):
        try:
            import rustcpd
        except ImportError:
            self.skipTest("rustcpd native API is unavailable")
        logic, _ = load_logic()
        rng = np.random.default_rng(11); count = 120
        z = np.sort(rng.uniform(0, 1, count)); theta = rng.uniform(0, 2*np.pi, count)
        radius = 4 + 6*z + 9*np.exp(-((z-.06)/.06)**2)
        mean = np.c_[100*z, radius*np.cos(theta), radius*np.sin(theta)]
        modes = np.zeros((count, 3, 2)); modes[:, 1, 0] = np.sin(np.pi*z); modes[:, 2, 1] = np.sin(2*np.pi*z)
        modes /= np.linalg.norm(modes.reshape(3*count, 2), axis=0)
        eigenvalues = np.array([900., 400.])
        target0 = mean + modes @ np.array([15., -10.]) + rng.normal(scale=.4, size=mean.shape)
        rotation = Rotation.from_euler("z", .3).as_matrix()
        for prescale in (1., 3.):
            for residual in (1., 1.4):
                matrix = np.eye(4); matrix[:3, :3] = residual * rotation
                source = (prescale * mean) @ matrix[:3, :3].T
                target = (prescale * target0) @ matrix[:3, :3].T
                outputs = []
                for units in (.001, 1., 1000.):
                    output = logic.runDeformable(object(), source * units, prescale, target * units,
                        dict(lambda_reg=5., w=.1, tolerance=1e-9, max_iterations=250,
                             targetCoverage=.8, skipFineCPD=True, variance_keep=1.),
                        rigidMatrix=matrix,
                        ssmData=dict(mean=mean*units, U=modes, eig=eigenvalues*units**2))
                    outputs.append(output / units)
                with self.subTest(prescale=prescale, residual=residual):
                    for output in outputs:
                        np.testing.assert_allclose(output, outputs[1], atol=2e-4, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
