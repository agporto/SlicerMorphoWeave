"""Real-VTK metric tests, including calibration code, without a Slicer runtime."""

import ast
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


MODULE_DIR = Path(__file__).resolve().parents[2]
RESOURCE_DIR = MODULE_DIR / "Resources" / "Python"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, RESOURCE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


geometry = load_module("shape_completion_geometry_test", "MorphoWeaveShapeCompletionGeometry.py")
core = load_module("shape_completion_geometry_core_test", "MorphoWeaveShapeCompletionCore.py")


def mesh(vertices, triangles=()):
    surface = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(vertices, dtype=np.float64), deep=True))
    surface.SetPoints(points)
    cells = vtk.vtkCellArray()
    for triangle in triangles:
        cells.InsertNextCell(len(triangle))
        for vertex in triangle:
            cells.InsertCellPoint(int(vertex))
    surface.SetPolys(cells)
    return surface


def calibration_branch(test="truth_type == 'exact_dense_correspondence'"):
    """Compile a real calibration branch, rather than a test copy of its formulas."""
    path = RESOURCE_DIR / "MorphoWeaveShapeCompletionBase.py"
    tree = ast.parse(path.read_text())
    widget = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "MorphoWeaveShapeCompletionWidget")
    method = next(node for node in widget.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_run_calibration_impl")
    branch = next(node for node in ast.walk(method) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == test)
    return compile(ast.Module(body=[branch], type_ignores=[]), str(path), "exec")


class ShapeCompletionGeometryTest(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0., 0., 0.], [4., 0., 0.], [0., 4., 0.]])
        self.surface = mesh(self.vertices, [(0, 1, 2)])

    def evaluate_calibration(self, predictions, *, surface=None, surface_points=None,
                             coverage=0.5, truth_type="surface_distance", posed_truth=None,
                             mask=None, closest=None):
        namespace = {
            "np": np, "truth_type": truth_type,
            "result": types.SimpleNamespace(completed_points=np.asarray(predictions, dtype=float)),
            "complete_polydata": self.surface if surface is None else surface,
            "posed_surface": self.vertices if surface_points is None else surface_points,
            "posed_truth": posed_truth, "coverage": coverage,
            "closest_surface_points": geometry.closest_surface_points if closest is None else closest,
            "mask": mask or core.PlaneMask(np.array([0, 2]), np.array([1., 0., 1.]), 1.5, True),
        }
        exec(calibration_branch(), namespace)
        return namespace["errors_all"], namespace["missing_mask"]

    def test_triangle_interior_is_zero_not_nearest_vertex_error(self):
        query = np.array([[1., 1., 0.], [0.2, 0.3, 0.]])
        closest, distance = geometry.closest_surface_points(self.surface, query)
        np.testing.assert_allclose(closest, query, atol=1e-14)
        np.testing.assert_allclose(distance, 0, atol=1e-14)
        _, old_distance = core.nearest_indices(self.vertices, query)
        self.assertTrue(np.all(old_distance > 0.3))

    def test_offsets_and_boundary_closest_locations(self):
        closest, distance = geometry.closest_surface_points(
            self.surface, [[1., 1., 3.], [-2., 1., 0.], [3., 3., 0.]])
        np.testing.assert_allclose(closest, [[1., 1., 0.], [0., 1., 0.], [2., 2., 0.]], atol=1e-14)
        np.testing.assert_allclose(distance, [3., 2., np.sqrt(2.)], atol=1e-14)

    def test_distance_is_invariant_to_equivalent_sparse_tessellation(self):
        refined = mesh(np.vstack((self.vertices, [[2., 0., 0.], [2., 2., 0.], [0., 2., 0.]])),
                       [(0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)])
        queries = np.array([[1., 1., 0.], [2., 0.4, 0.3], [-1., 2., 0.5], [3., 3., -1.]])
        coarse_closest, coarse_distance = geometry.closest_surface_points(self.surface, queries)
        fine_closest, fine_distance = geometry.closest_surface_points(refined, queries)
        np.testing.assert_allclose(fine_distance, coarse_distance, atol=1e-14)
        np.testing.assert_allclose(fine_closest, coarse_closest, atol=1e-14)

    def test_rigid_pose_preserves_distances_locations_and_input_mesh(self):
        queries = np.array([[1., 1., 0.], [1., 1., 0.6], [-1., 2., -0.5]])
        closest, distance = geometry.closest_surface_points(self.surface, queries)
        posed, rotation, translation = core.apply_random_pose(self.vertices, np.random.default_rng(123))
        posed_queries = queries @ rotation + translation
        posed_closest, posed_distance = geometry.closest_surface_points(
            self.surface, posed_queries, surface_points=posed)
        np.testing.assert_allclose(posed_distance, distance, atol=1e-13)
        np.testing.assert_allclose(posed_closest, closest @ rotation + translation, atol=1e-13)
        np.testing.assert_array_equal(vtk_to_numpy(self.surface.GetPoints().GetData()), self.vertices)

    def test_calibration_classifies_closest_truth_location_not_prediction_or_vertex(self):
        # The first query and its nearest vertex are retained, but its closest
        # truth-triangle point is withheld. The second query crosses the cut in
        # the other direction. This catches both incorrect classification rules.
        queries = np.array([[1.6, 0.6, -2.], [0.6, 0.6, 2.]])
        errors, missing = self.evaluate_calibration(queries)
        np.testing.assert_allclose(errors, [2., 2.], atol=1e-14)
        np.testing.assert_array_equal(missing, [True, False])
        mask = core.PlaneMask(np.array([0, 2]), np.array([1., 0., 1.]), 1.5, True)
        np.testing.assert_array_equal(~mask.contains(queries), [False, True])
        nearest, _ = core.nearest_indices(self.vertices, queries)
        np.testing.assert_array_equal(~mask.contains(self.vertices[nearest]), [False, False])

    def test_calibration_pose_uses_real_surface_topology_in_posed_frame(self):
        queries = np.array([[1.6, 0.6, -2.], [0.6, 0.6, 2.]])
        posed, rotation, translation = core.apply_random_pose(self.vertices, np.random.default_rng(76))
        normal = np.array([1., 0., 1.]) @ rotation
        mask = core.PlaneMask(np.array([0, 2]), normal, 1.5 + translation @ normal, True)
        errors, missing = self.evaluate_calibration(
            queries @ rotation + translation, surface_points=posed, mask=mask)
        np.testing.assert_allclose(errors, [2., 2.], atol=1e-13)
        np.testing.assert_array_equal(missing, [True, False])

    def test_complete_coverage_scores_all_dense_loci(self):
        errors, missing = self.evaluate_calibration([[1., 1., 0.], [2., 1., 0.2]], coverage=1.0)
        np.testing.assert_allclose(errors, [0., 0.2], atol=1e-14)
        np.testing.assert_array_equal(missing, [True, True])

    def test_exact_dense_truth_remains_homologous_and_needs_no_triangles(self):
        truth = np.array([[1., 1., 0.], [2., 1., 0.]])
        # Permuted truth loci are on the same surface but are not homologously
        # correct. Exact-truth scoring must still measure the unit displacement.
        no_surface_query = Mock(side_effect=AssertionError("must not use surface scoring"))
        errors, missing = self.evaluate_calibration(
            truth[::-1], surface=mesh(self.vertices), truth_type="exact_dense_correspondence",
            posed_truth=truth, closest=no_surface_query)
        np.testing.assert_allclose(errors, [1., 1.], atol=1e-14)
        np.testing.assert_array_equal(missing, [False, True])
        no_surface_query.assert_not_called()

    def test_point_cloud_and_line_only_surfaces_are_rejected(self):
        cloud = mesh(self.vertices)
        with self.assertRaisesRegex(ValueError, "triangle surfaces.*point clouds"):
            geometry.closest_surface_points(cloud, [[1., 1., 0.]])
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(2)
        cells.InsertCellPoint(0)
        cells.InsertCellPoint(1)
        cloud.SetLines(cells)
        with self.assertRaisesRegex(ValueError, "triangle surfaces.*point clouds"):
            geometry.closest_surface_points(cloud, [[1., 1., 0.]])

    def test_calibration_requires_surface_topology_only_without_paired_dense_truth(self):
        logic = types.SimpleNamespace(model_polydata_world=Mock(return_value=mesh(self.vertices)))
        namespace = {"self": types.SimpleNamespace(logic=logic), "mesh_node": object(),
                     "triangle_surface": geometry.triangle_surface,
                     "truth_type": "surface_distance"}
        preflight = calibration_branch("truth_type == 'surface_distance'")
        with self.assertRaisesRegex(ValueError, "triangle surfaces.*point clouds"):
            exec(preflight, namespace)
        logic.model_polydata_world.reset_mock()
        namespace["truth_type"] = "exact_dense_correspondence"
        exec(preflight, namespace)
        logic.model_polydata_world.assert_not_called()

    def test_polygon_reference_is_triangulated_without_changing_geometry(self):
        square = mesh([[0., 0., 0.], [4., 0., 0.], [4., 4., 0.], [0., 4., 0.]], [(0, 1, 2, 3)])
        surface, vertices, triangles = geometry.triangle_surface(square)
        self.assertEqual(surface.GetNumberOfPolys(), 2)
        self.assertEqual(triangles.shape, (2, 3))
        self.assertEqual(vertices.shape, (4, 3))
        closest, distance = geometry.closest_surface_points(square, [[3., 3., 0.], [1., 3., -0.2]])
        np.testing.assert_allclose(closest, [[3., 3., 0.], [1., 3., 0.]], atol=1e-14)
        np.testing.assert_allclose(distance, [0., 0.2], atol=1e-14)

    def test_isolated_vertex_cells_do_not_count_as_surface(self):
        surface = mesh(np.vstack((self.vertices, [[1., 1., 2.]])), [(0, 1, 2)])
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(1)
        cells.InsertCellPoint(3)
        surface.SetVerts(cells)
        closest, distance = geometry.closest_surface_points(surface, [[1., 1., 2.]])
        np.testing.assert_allclose(closest, [[1., 1., 0.]], atol=1e-14)
        np.testing.assert_allclose(distance, [2.], atol=1e-14)

    def test_invalid_queries_and_replacement_coordinates_are_rejected(self):
        for query in ([[np.nan, 0., 0.]], [1., 2., 3.], [[0., 1.]]):
            with self.assertRaisesRegex(ValueError, "finite.*array"):
                geometry.closest_surface_points(self.surface, query)
        for points in (np.zeros((2, 3)), np.full((3, 3), np.inf)):
            with self.assertRaisesRegex(ValueError, "Replacement surface points"):
                geometry.closest_surface_points(self.surface, [[1., 1., 0.]], surface_points=points)
        closest, distance = geometry.closest_surface_points(self.surface, np.empty((0, 3)))
        self.assertEqual(closest.shape, (0, 3))
        self.assertEqual(distance.shape, (0,))

    def test_optimize_uses_shared_metric_without_reversing_queries(self):
        path = MODULE_DIR / "MorphoWeaveShapeCompletion.py"
        tree = ast.parse(path.read_text())
        method = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_optimize_surface_distances")
        namespace = {"closest_surface_points": geometry.closest_surface_points}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        actual = namespace["_optimize_surface_distances"](self.surface, [[1., 1., 0.], [1., 1., 3.]])
        np.testing.assert_allclose(actual, [0., 3.], atol=1e-14)

    def test_geometry_helper_and_test_are_packaged(self):
        cmake = (MODULE_DIR / "CMakeLists.txt").read_text()
        self.assertIn("Resources/Python/MorphoWeaveShapeCompletionGeometry.py", cmake)
        self.assertIn("slicer_add_python_unittest(SCRIPT Testing/Python/ShapeCompletionGeometryTest.py)", cmake)


if __name__ == "__main__":
    unittest.main()
