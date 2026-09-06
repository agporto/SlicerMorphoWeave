"""Run inside a fresh 3D Slicer process with this extension and rustcpd installed.

This runs the native fitter on a small synthetic SSM. Reduced iteration budgets
exercise integration, not anatomical accuracy. It never installs dependencies.
"""
import csv
import json
from pathlib import Path
import tempfile

import numpy as np
import qt
import slicer
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def run_smoke_test():
    if not hasattr(slicer, "mrmlScene"):
        raise RuntimeError("Run this script with the 3D Slicer application, not ordinary Python")
    import rustcpd
    from MorphoWeaveShapeCompletion import (
        MorphoWeaveShapeCompletionWidget, validate_completion_backend,
    )
    validate_completion_backend(rustcpd)
    scene = slicer.mrmlScene
    existing = {scene.GetNthNode(i).GetID() for i in range(scene.GetNumberOfNodes())}
    parent = None
    widget = None
    try:
        sphere = vtk.vtkSphereSource()
        sphere.SetThetaResolution(12)
        sphere.SetPhiResolution(8)
        sphere.SetRadius(10)
        sphere.Update()
        surface = vtk.vtkPolyData()
        surface.DeepCopy(sphere.GetOutput())
        mean = vtk_to_numpy(surface.GetPoints().GetData()).astype(float)
        mean *= [1.5, 0.8, 1.1]
        surface.GetPoints().SetData(numpy_to_vtk(mean, deep=True))
        model = scene.AddNewNodeByClass("vtkMRMLModelNode", "merge_smoke_template")
        model.SetAndObservePolyData(surface)
        model.CreateDefaultDisplayNodes()
        dense = scene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "merge_smoke_template_correspondences")
        slicer.util.updateMarkupsControlPointsFromArray(dense, mean)
        sparse = scene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "merge_smoke_template_sparse_landmarks")
        slicer.util.updateMarkupsControlPointsFromArray(sparse, mean[[0, 7, 23, 51]])
        modes, _ = np.linalg.qr(np.random.default_rng(17).normal(size=(mean.size, 4)))
        table = scene.AddNewNodeByClass("vtkMRMLTableNode", "ssm_data_merge_smoke")
        for index, column in enumerate(np.column_stack([mean.ravel(), modes]).T):
            array = numpy_to_vtk(np.ascontiguousarray(column), deep=True)
            array.SetName("mean" if index == 0 else f"mode_{index}")
            table.GetTable().AddColumn(array)
        table.SetAttribute("ssm_npoints", str(len(mean)))
        table.SetAttribute("ssm_eigenvalues", json.dumps([0.5, 0.3, 0.2, 0.1]))
        parent = slicer.qMRMLWidget()
        parent.setLayout(qt.QVBoxLayout())
        parent.setMRMLScene(scene)
        widget = MorphoWeaveShapeCompletionWidget(parent)
        widget._deps_ready = True  # Already checked above; do not prompt/install.
        widget.setup()
        for left, right in (
            (widget.template_model_selector, widget.batch_template_model),
            (widget.template_dense_selector, widget.batch_template_dense),
            (widget.template_sparse_selector, widget.batch_template_sparse),
            (widget.ssm_table_selector, widget.batch_ssm_table),
        ):
            assert left.currentNode() is not None
            assert left.currentNode().GetID() == right.currentNode().GetID()
        widget.variance_keep.setValue(1.0)
        widget.posterior_samples.setValue(0)
        widget.pose_rotation_count.setValue(12)
        widget.pose_survivors.setValue(12)
        widget.pose_coarse_rank.setValue(4)
        widget.pose_coarse_source.setValue(40)
        widget.pose_coarse_target.setValue(40)
        widget.pose_coarse_iterations.setValue(2)
        widget.pose_screen_iterations.setValue(2)
        widget.pose_refine_count.setValue(2)
        widget.pose_refine_iterations.setValue(3)
        widget.atlas_iterations.setValue(5)
        widget.random_seed.setValue(17)
        with tempfile.TemporaryDirectory(prefix="morphoweave-smoke-") as temporary:
            root = Path(temporary)
            inputs, single, output = (root / name for name in ("inputs", "single", "batch"))
            for directory in (inputs, single, output):
                directory.mkdir()
            target = vtk.vtkPolyData()
            target.DeepCopy(surface)
            points = mean + 0.2 * modes[:, 0].reshape(-1, 3)
            target.GetPoints().SetData(numpy_to_vtk(points, deep=True))
            for name in ("target_a", "target_b"):
                writer = vtk.vtkXMLPolyDataWriter()
                writer.SetInputData(target)
                writer.SetFileName(str(inputs / f"{name}.vtp"))
                assert writer.Write() == 1
            widget.target_model_selector.setCurrentNode(slicer.util.loadModel(str(inputs / "target_a.vtp")))
            widget.output_directory.setCurrentPath(str(single))
            widget._run_completion_impl()
            single_dense = list(single.glob("*_dense.mrk.json"))
            assert len(single_dense) == 1
            single_node = slicer.util.loadMarkups(str(single_dense[0]))
            single_points = widget.logic.markups_points_world(single_node)
            widget.batch_input.setCurrentPath(str(inputs))
            widget.batch_output.setCurrentPath(str(output))
            progress = widget._batch_progress_update
            def cancel_after_first(row, completed, total):
                progress(row, completed, total)
                if row["status"] == "success" and completed == 1:
                    widget.on_cancel_batch()
            widget._batch_progress_update = cancel_after_first
            widget.on_run_batch()
            def statuses():
                with (output / "batch_summary.csv").open(newline="", encoding="utf-8") as stream:
                    return [row["status"] for row in csv.DictReader(stream)]
            assert statuses() == ["success", "cancelled"], statuses()
            widget._batch_progress_update = progress
            widget.on_run_batch()
            assert statuses() == ["skipped", "success"], statuses()
            widget.on_run_batch()
            assert statuses() == ["skipped", "skipped"], statuses()
            batch_dense = list((output / "target_a").glob("*_dense.mrk.json"))
            assert len(batch_dense) == 1
            batch_node = slicer.util.loadMarkups(str(batch_dense[0]))
            batch_points = widget.logic.markups_points_world(batch_node)
            np.testing.assert_allclose(single_points, batch_points, rtol=1e-10, atol=1e-7)
            report = {
                "native_backend": getattr(rustcpd, "__version__", "unknown"),
                "automatic_selectors_synchronized": True,
                "single_batch_max_coordinate_difference": float(np.max(np.abs(single_points - batch_points))),
                "cancellation_and_verified_resume": True,
            }
            print("SHAPE_COMPLETION_SMOKE_PASSED " + json.dumps(report, sort_keys=True))
    finally:
        if widget is not None:
            widget.on_clear_outputs()
            widget.cleanup()
        if parent is not None:
            parent.deleteLater()
        # Preserve pre-existing scene data. Run in a fresh process for merge signoff.
        for index in reversed(range(scene.GetNumberOfNodes())):
            node = scene.GetNthNode(index)
            if node.GetID() not in existing and not node.GetSingletonTag():
                scene.RemoveNode(node)


if __name__ == "__main__":
    try:
        run_smoke_test()
    except Exception:
        import traceback
        traceback.print_exc()
        slicer.app.exit(1)
    else:
        slicer.app.exit(0)
