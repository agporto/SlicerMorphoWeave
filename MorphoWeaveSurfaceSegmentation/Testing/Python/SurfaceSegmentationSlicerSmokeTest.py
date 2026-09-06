"""Native hdmseg + real MRML/VTK smoke test; requires a fresh Slicer process."""
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import slicer
import vtk
from vtk.util.numpy_support import numpy_to_vtk


def run_smoke_test():
    if not hasattr(slicer, "mrmlScene"):
        raise RuntimeError("Run this script with the 3D Slicer application")
    import hdmseg
    from MorphoWeaveSurfaceSegmentation import MorphoWeaveSurfaceSegmentationLogic, stableMajorMinor
    if stableMajorMinor(hdmseg.__version__) != (0, 2):
        raise RuntimeError("This smoke test requires hdmseg>=0.2,<0.3")
    logic = MorphoWeaveSurfaceSegmentationLogic()
    with tempfile.TemporaryDirectory(prefix="hdmseg-smoke-") as temporary:
        root = Path(temporary)
        meshes, markups, output = (root / name for name in ("meshes", "markups", "output"))
        for directory in (meshes, markups, output):
            directory.mkdir()
        xy = np.array([(x, y) for y in range(6) for x in range(6)], dtype=float)
        for specimen in range(3):
            points = np.column_stack([xy, 0.1 * (1 + specimen) * np.sin(xy[:, 0]) * np.cos(xy[:, 1])])
            surface = vtk.vtkPolyData()
            pts = vtk.vtkPoints()
            pts.SetData(numpy_to_vtk(points, deep=True))
            surface.SetPoints(pts)
            cells = vtk.vtkCellArray()
            for y in range(5):
                for x in range(5):
                    a = y * 6 + x
                    for face in ((a, a + 1, a + 6), (a + 1, a + 7, a + 6)):
                        cells.InsertNextCell(3)
                        for index in face:
                            cells.InsertCellPoint(index)
            surface.SetPolys(cells)
            stem = f"specimen_{specimen:02d}"
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetInputData(surface)
            writer.SetFileName(str(meshes / (stem + ".vtp")))
            assert writer.Write() == 1
            (markups / (stem + ".mrk.json")).write_text(json.dumps({
                "markups": [{"coordinateSystem": "RAS", "controlPoints": [
                    {"position": point.tolist(), "positionStatus": "defined"} for point in points]}]}))
        def run(k):
            return logic.run(str(meshes), str(markups), str(output), hdmseg,
                             fixedK=k, neighbors=8, components=6, smoothing=0, previewN=0)
        first = run(3)
        first_dir = Path(first["output_directory"])
        original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in first_dir.iterdir()}
        second = run(2)
        second_dir = Path(second["output_directory"])
        assert first["skipped"] == second["skipped"] == 0
        assert first_dir != second_dir
        assert original == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in first_dir.iterdir()}
        for result in (first, second):
            directory = Path(result["output_directory"])
            labels = np.load(directory / "MorphoWeaveSurfaceSegmentation_locus_labels.npy")
            assert len(np.unique(labels)) == result["k"]
            summary = json.loads(Path(result["summary"]).read_text())
            assert set(summary["outputs"]["region_ply_files"]) == {p.name for p in directory.glob("*.ply")}
            for path in directory.glob("*.ply"):
                reader = vtk.vtkPLYReader()
                reader.SetFileName(str(path))
                reader.Update()
                assert reader.GetOutput().GetNumberOfCells() > 0
        print("SURFACE_SEGMENTATION_SMOKE_PASSED " + json.dumps({
            "native_backend": hdmseg.__version__, "first_k": first["k"],
            "second_k": second["k"], "prior_outputs_preserved": True,
        }, sort_keys=True))


if __name__ == "__main__":
    try:
        run_smoke_test()
    except Exception:
        import traceback
        traceback.print_exc()
        slicer.app.exit(1)
    else:
        slicer.app.exit(0)
