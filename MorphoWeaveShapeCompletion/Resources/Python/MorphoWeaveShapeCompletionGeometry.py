"""Triangle-surface evaluation shared by calibration and Optimize.

These queries are unsigned and directional: each query is measured against the
provided reference surface. They are not a bidirectional surface discrepancy and
do not require a closed mesh. This module has no Slicer or registration dependency.
"""

import numpy as np
import vtk
import vtk.util.numpy_support as vtk_np


def triangle_surface(polydata):
    """Return a triangulated copy, its float64 vertices, and triangle indices.

    Discard isolated vertex/line cells so they cannot change a surface distance.
    Point-cloud-only references are not a triangle surface and must not silently
    fall back to nearest-vertex distances.
    """
    if polydata is None or polydata.GetPoints() is None:
        raise ValueError("Surface distance requires triangle surfaces, not empty meshes or point clouds")
    vertices = vtk_np.vtk_to_numpy(polydata.GetPoints().GetData())
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Surface vertices must be finite")
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(polydata)
    triangle_filter.PassVertsOff()
    triangle_filter.PassLinesOff()
    triangle_filter.Update()
    output = vtk.vtkPolyData()
    output.DeepCopy(triangle_filter.GetOutput())
    if output.GetNumberOfPoints() < 3 or output.GetNumberOfPolys() < 1:
        raise ValueError("Surface distance requires triangle surfaces, not empty meshes or point clouds")
    vertices = vtk_np.vtk_to_numpy(output.GetPoints().GetData()).astype(np.float64, copy=True)
    offsets = vtk_np.vtk_to_numpy(output.GetPolys().GetOffsetsArray())
    connectivity = vtk_np.vtk_to_numpy(output.GetPolys().GetConnectivityArray())
    if not np.all(np.diff(offsets) == 3):
        raise ValueError("Invalid triangulated surface")
    triangles = connectivity.reshape(-1, 3).astype(np.int64, copy=True)
    if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise ValueError("Invalid surface triangle indices")
    return output, vertices, triangles


def closest_surface_points(polydata, queries, *, surface_points=None):
    """Return nearest triangle locations and unsigned distances for queries.

    ``surface_points`` optionally replaces the reference coordinates, preserving
    its topology and float64 precision. Calibration uses this for each generated
    rigid pose. Neither the input mesh nor the supplied arrays are modified.
    """
    queries = np.asarray(queries, dtype=np.float64)
    if queries.ndim != 2 or queries.shape[1] != 3 or not np.all(np.isfinite(queries)):
        raise ValueError("Surface queries must be a finite (N, 3) array")
    if surface_points is not None:
        surface_points = np.asarray(surface_points, dtype=np.float64)
        if (polydata is None
                or surface_points.shape != (polydata.GetNumberOfPoints(), 3)
                or not np.all(np.isfinite(surface_points))):
            raise ValueError("Replacement surface points must be finite and match the mesh vertex count")
        posed = vtk.vtkPolyData()
        posed.ShallowCopy(polydata)
        points = vtk.vtkPoints()
        points.SetData(vtk_np.numpy_to_vtk(surface_points, deep=True))
        posed.SetPoints(points)
        polydata = posed
    surface, _, _ = triangle_surface(polydata)
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    closest_points = np.empty(queries.shape, dtype=np.float64)
    distances = np.empty(len(queries), dtype=np.float64)
    closest = [0.0, 0.0, 0.0]
    cell, sub, squared = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
    for index, point in enumerate(queries):
        locator.FindClosestPoint(point, closest, cell, sub, squared)
        if int(cell) < 0 or not np.isfinite(float(squared)):
            raise ValueError("Could not locate a finite closest point on the triangle surface")
        closest_points[index] = closest
        distances[index] = np.sqrt(max(float(squared), 0.0))
    if not np.all(np.isfinite(closest_points)) or not np.all(np.isfinite(distances)):
        raise ValueError("Nonfinite surface distances")
    return closest_points, distances
