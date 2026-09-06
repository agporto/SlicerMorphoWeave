"""Batch UI reusing Shape Completion's settings, fitting, and exporters unchanged."""
from collections import Counter
import hashlib
import importlib.metadata
import inspect
import json
import logging
from pathlib import Path

import ctk
import numpy as np
import qt
import slicer
import vtk.util.numpy_support as vtk_np
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from MorphoWeaveShapeCompletion import (
    MorphoWeaveShapeCompletionLogic,
    MorphoWeaveShapeCompletionWidget,
    run_completion,
)
from Resources.Python.MorphoWeaveCompletionBatch import (
    discover_specimens,
    file_sha256,
    run_batch,
)


class MorphoWeaveBatchShapeCompletion(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Batch Shape Completion"
        self.parent.categories = ["MorphoWeave"]
        self.parent.dependencies = ["MorphoWeaveShapeCompletion"]
        self.parent.contributors = ["Arthur Porto"]
        self.parent.helpText = (
            "Process a folder of fragments using the existing Shape Completion workflow. "
            "Select the shared SSM, coverage, landmarks mode and optional calibration profile "
            "in Complete Shape; use Advanced for fitting settings and Batch for folders. "
            "No manually loaded target is required for batch processing."
        )
        self.parent.acknowledgementText = "This module was developed by Arthur Porto"


class _BatchCompletionLogic(MorphoWeaveShapeCompletionLogic):
    """Cache SSM parsing and track only nodes created by our synchronous I/O.

    Do not snapshot the entire specimen run: processEvents() in the parent
    workflow may allow unrelated nodes to be added elsewhere in Slicer.
    """
    def __init__(self, scene):
        super().__init__()
        self.scene = scene
        self._owned_node_ids = []
        self._owned_folders = []
        self._ssm_node = None
        self._ssm_arrays = None

    def ssm_from_table(self, node):
        if self._ssm_node is None:
            self._ssm_arrays = super().ssm_from_table(node)
            self._ssm_node = node
        elif node.GetID() != self._ssm_node.GetID():
            raise RuntimeError("The selected SSM changed during the batch")
        # Fresh arrays isolate native in-place writes without reparsing the table.
        return tuple(array.copy() for array in self._ssm_arrays)

    def tracked_call(self, function, *args, **kwargs):
        before = {
            self.scene.GetNthNode(i).GetID()
            for i in range(self.scene.GetNumberOfNodes())
        }
        try:
            return function(*args, **kwargs)
        finally:
            for index in range(self.scene.GetNumberOfNodes()):
                node = self.scene.GetNthNode(index)
                if node.GetID() not in before and not node.GetSingletonTag():
                    self._owned_node_ids.append(node.GetID())

    def create_output_folder(self, *args, **kwargs):
        folder = super().create_output_folder(*args, **kwargs)
        self._owned_folders.append(folder)
        return folder

    def create_completion_outputs(self, *args, **kwargs):
        return self.tracked_call(super().create_completion_outputs, *args, **kwargs)

    def save_completion_outputs(self, *args, **kwargs):
        return self.tracked_call(super().save_completion_outputs, *args, **kwargs)

    def cleanup_specimen(self):
        hierarchy = self.scene.GetFirstNodeByClass("vtkMRMLSubjectHierarchyNode")
        if hierarchy is not None:
            for folder in self._owned_folders:
                hierarchy.RemoveItem(folder)
        self._owned_folders.clear()
        for node_id in reversed(self._owned_node_ids):
            node = self.scene.GetNodeByID(node_id)
            if node is not None:
                self.scene.RemoveNode(node)
        self._owned_node_ids.clear()


def _array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    if array.size:
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _shared_input_token(widget):
    """Cheap edit guard; world-coordinate hashes are also stored in the manifest."""
    token = []
    for selector in (widget.ssm_table_selector, widget.template_model_selector,
                     widget.template_dense_selector, widget.template_sparse_selector):
        node = selector.currentNode()
        if node is None:
            token.append(None)
            continue
        entry = [node.GetID(), node.GetMTime()]
        for getter in ("GetPolyData", "GetTable"):
            if hasattr(node, getter):
                data = getattr(node, getter)()
                entry.append(data.GetMTime() if data is not None else None)
        parent = node.GetParentTransformNode() if hasattr(node, "GetParentTransformNode") else None
        while parent is not None:
            entry.extend((parent.GetID(), parent.GetMTime()))
            parent = parent.GetParentTransformNode()
        token.append(tuple(entry))
    return tuple(token)


def _batch_context(widget):
    """Fingerprint inputs/settings relevant to fitting and full-resolution export."""
    import rustcpd

    mean, modes, eigenvalues = widget.logic.ssm_from_table(widget.ssm_table_selector.currentNode())
    template = widget.logic.model_polydata_world(widget.template_model_selector.currentNode())
    arrays = {"mean": mean, "modes": modes, "eigenvalues": eigenvalues,
              "surface_points": vtk_np.vtk_to_numpy(template.GetPoints().GetData())}
    for name, cells in (("polys", template.GetPolys()), ("strips", template.GetStrips()),
                        ("lines", template.GetLines()), ("verts", template.GetVerts())):
        arrays[name] = vtk_np.vtk_to_numpy(cells.GetData())
    labels = {}
    for name, selector in (("dense", widget.template_dense_selector),
                           ("sparse", widget.template_sparse_selector)):
        if selector.currentNode() is not None:
            labels[name], arrays[name] = widget.logic.markups_labels_points_world(selector.currentNode())
    try:
        backend_version = importlib.metadata.version("rustcpd")
    except importlib.metadata.PackageNotFoundError:
        backend_version = getattr(rustcpd, "__version__", "unknown")
    backend_files = {}
    backend_file = Path(rustcpd.__file__)
    if backend_file.is_file():
        backend_files[backend_file.name] = file_sha256(backend_file)
    if hasattr(rustcpd, "__path__"):
        for package_path in rustcpd.__path__:
            for path in sorted(Path(package_path).rglob("*")):
                if path.is_file() and path.suffix.lower() in (".so", ".pyd", ".dll", ".dylib"):
                    backend_files[str(path.relative_to(package_path))] = file_sha256(path)
    return {
        "settings": widget._settings_snapshot(widget._read_settings()),
        "interpolation": {
            "neighbors": int(widget.interpolation_neighbors.value),
            "sharpness": float(widget.interpolation_sharpness.value),
            "chunk_size": int(widget.interpolation_chunk_size.value),
            "restrict_to_components": bool(widget.component_interpolation_check.checked),
            "full_resolution_samples": bool(widget.full_resolution_samples_check.checked),
        },
        "arrays_sha256": {name: _array_hash(array) for name, array in arrays.items()},
        "landmark_labels": labels,
        "template_vertex_indices": widget.template_dense_selector.currentNode().GetAttribute(
            "MorphoWeave.TemplateVertexIndices"),
        "calibration_profile_sha256": hashlib.sha256(
            json.dumps(widget._profile, sort_keys=True).encode("utf-8")).hexdigest(),
        "rustcpd_version": backend_version,
        "rustcpd_files_sha256": backend_files,
        "completion_source_sha256": file_sha256(inspect.getfile(MorphoWeaveShapeCompletionWidget)),
        "completion_core_sha256": file_sha256(inspect.getfile(run_completion)),
        "batch_source_sha256": file_sha256(__file__),
        "batch_runner_sha256": file_sha256(inspect.getfile(run_batch)),
        "numpy_version": np.__version__,
    }


def _process_specimen(widget, specimen, output_directory):
    """Call the *same* single-specimen implementation, not a second pipeline."""
    previous_target = widget.target_model_selector.currentNode()
    previous_landmarks = widget.target_landmark_selector.currentNode()
    previous_directory = str(widget.output_directory.currentPath or "")
    previous_folder = widget._run_folder_item
    try:
        if _shared_input_token(widget) != widget._batch_input_token:
            raise RuntimeError("Shared SSM/template inputs changed during the batch; restart with unchanged inputs")
        # Clear both selectors before loading, so landmarks cannot leak across specimens.
        widget.target_model_selector.setCurrentNode(None)
        widget.target_landmark_selector.setCurrentNode(None)
        widget._run_folder_item = None
        target = widget.logic.tracked_call(slicer.util.loadModel, str(specimen.mesh))
        if target is None or target.GetPolyData() is None or target.GetPolyData().GetNumberOfPoints() == 0:
            raise ValueError(f"Could not load a nonempty mesh: {specimen.mesh}")
        widget.target_model_selector.setCurrentNode(target)
        if widget._read_settings().use_landmarks:
            if specimen.landmarks is None:
                raise ValueError("Landmark-assisted mode requires a matching landmark file")
            landmarks = widget.logic.tracked_call(slicer.util.loadMarkups, str(specimen.landmarks))
            if landmarks is None:
                raise ValueError(f"Could not load landmarks: {specimen.landmarks}")
            widget.target_landmark_selector.setCurrentNode(landmarks)
            if widget._preview_landmark_match_count() < 3:
                raise ValueError("Landmark-assisted mode requires at least three matched landmarks")
        widget.output_directory.setCurrentPath(str(output_directory))
        widget._run_completion_impl()
        if _shared_input_token(widget) != widget._batch_input_token:
            raise RuntimeError("Shared SSM/template inputs changed during fitting; result was not published")
    finally:
        try:
            widget.target_model_selector.setCurrentNode(previous_target)
            widget.target_landmark_selector.setCurrentNode(previous_landmarks)
            widget.output_directory.setCurrentPath(previous_directory)
            widget._run_folder_item = previous_folder
        finally:
            widget.logic.cleanup_specimen()


class MorphoWeaveBatchShapeCompletionWidget(MorphoWeaveShapeCompletionWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._batch_running = False
        self._cancel_batch = False

    def setup(self):
        super().setup()
        self._validate_batch_inputs()

    def _build_ui(self):
        super()._build_ui()
        self.batch_tab = qt.QWidget()
        self.tabs.insertTab(1, self.batch_tab, "Batch")
        layout = qt.QFormLayout(self.batch_tab)
        intro = qt.QLabel(
            "Select the shared SSM, coverage, landmark mode and optional calibration profile "
            "in Complete Shape, and fitting parameters in Advanced. The target selectors "
            "and single-specimen output directory are ignored here. Every specimen uses "
            "the same settings and random seed. Do not edit the shared scene inputs during a batch."
        )
        intro.setWordWrap(True)
        layout.addRow(intro)
        self.batch_input = ctk.ctkPathLineEdit()
        self.batch_landmarks = ctk.ctkPathLineEdit()
        self.batch_output = ctk.ctkPathLineEdit()
        for control in (self.batch_input, self.batch_landmarks, self.batch_output):
            control.filters = ctk.ctkPathLineEdit.Dirs
        layout.addRow("Fragment mesh directory:", self.batch_input)
        layout.addRow("Paired landmark directory (landmark mode only):", self.batch_landmarks)
        layout.addRow("Batch output directory:", self.batch_output)
        self.batch_resume = qt.QCheckBox("Skip verified completed specimens")
        self.batch_resume.checked = True
        layout.addRow(self.batch_resume)
        note = qt.QLabel(
            "Mesh formats: PLY, VTP, VTK, STL, OBJ. Landmarks: matching basename with "
            ".mrk.json, .fcsv or .json. Folders are not searched recursively. "
            "Existing specimen outputs are never overwritten; use a new output directory "
            "when changing settings. Scene outputs are saved and removed after each specimen."
        )
        note.setWordWrap(True)
        layout.addRow(note)
        self.batch_status = qt.QLabel("")
        self.batch_status.setWordWrap(True)
        layout.addRow(self.batch_status)
        self.batch_run = qt.QPushButton("Run Batch Shape Completion")
        self.batch_cancel = qt.QPushButton("Cancel after current specimen")
        self.batch_cancel.enabled = False
        layout.addRow(self.batch_run)
        layout.addRow(self.batch_cancel)
        self.batch_progress = qt.QProgressBar()
        self.batch_progress.setRange(0, 100)
        self.batch_progress.setValue(0)
        layout.addRow(self.batch_progress)
        self.batch_log = qt.QPlainTextEdit()
        self.batch_log.setReadOnly(True)
        self.batch_log.setMinimumHeight(200)
        layout.addRow(self.batch_log)
        self.tabs.setCurrentWidget(self.batch_tab)

    def _connect_ui(self):
        super()._connect_ui()
        for control in (self.batch_input, self.batch_landmarks, self.batch_output):
            control.currentPathChanged.connect(self._validate_batch_inputs)
        for selector in (self.template_model_selector, self.template_dense_selector,
                         self.template_sparse_selector, self.ssm_table_selector):
            selector.currentNodeChanged.connect(self._validate_batch_inputs)
        self.use_landmarks_check.toggled.connect(self._validate_batch_inputs)
        self.batch_run.clicked.connect(self.on_run_batch)
        self.batch_cancel.clicked.connect(self.on_cancel_batch)

    def enter(self):
        if self._batch_running:
            return
        super().enter()
        if self._ui_ready:
            self._validate_batch_inputs()

    def cleanup(self):
        self._cancel_batch = True
        super().cleanup()

    def _validate_batch_inputs(self, *args):
        if self._batch_running:
            self.batch_run.enabled = False
            return
        required = all(selector.currentNode() is not None for selector in (
            self.template_model_selector, self.template_dense_selector, self.ssm_table_selector))
        use_landmarks = bool(self.use_landmarks_check.checked)
        landmarks_ok = not use_landmarks or (
            self.template_sparse_selector.currentNode() is not None
            and bool(self.batch_landmarks.currentPath)
            and Path(str(self.batch_landmarks.currentPath)).is_dir())
        input_ok = bool(self.batch_input.currentPath) and Path(str(self.batch_input.currentPath)).is_dir()
        ready = required and input_ok and landmarks_ok and bool(self.batch_output.currentPath)
        self.batch_run.enabled = bool(ready)
        self.batch_status.setText(
            "Ready. Each specimen will be saved independently; failures will not stop the batch."
            if ready else "Select the shared SSM inputs, mesh and output folders, and paired landmarks when landmark mode is enabled."
        )

    def _set_batch_busy(self, busy):
        self._batch_running = busy
        for control in (self.complete_tab, self.calibration_tab, self.advanced_tab,
                        self.batch_input, self.batch_landmarks, self.batch_output, self.batch_resume):
            control.setEnabled(not busy)
        self.batch_run.enabled = not busy
        self.batch_cancel.enabled = busy

    def on_cancel_batch(self):
        self._cancel_batch = True
        self.batch_cancel.enabled = False
        self.batch_log.appendPlainText("Cancellation requested. The current native fit and its exports will finish first.")

    def _batch_progress_update(self, row, completed, total):
        self.batch_progress.setValue(round(100 * completed / max(total, 1)))
        self.batch_status.setText(f"{completed}/{total}: {row['specimen']} - {row['status']}")
        self.batch_log.appendPlainText(f"{row['specimen']}: {row['status']}" + (f" - {row['error']}" if row['error'] else ""))
        slicer.app.processEvents()

    def on_run_batch(self):
        if self._batch_running:
            return
        self._cancel_batch = False
        self._set_batch_busy(True)
        self.batch_log.clear()
        self.batch_progress.setValue(0)
        previous_logic = self.logic
        try:
            if not self._ensure_dependencies():
                return
            if self.profile_path.currentPath and self._profile is None:
                raise ValueError("The selected calibration profile could not be loaded")
            use_landmarks = bool(self.use_landmarks_check.checked)
            specimens = discover_specimens(
                str(self.batch_input.currentPath),
                str(self.batch_landmarks.currentPath) if use_landmarks else None,
            )
            # Keep the shared hierarchy outside the set of temporary batch nodes.
            slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
            self.logic = _BatchCompletionLogic(slicer.mrmlScene)
            self._batch_input_token = _shared_input_token(self)
            context = _batch_context(self)
            self.batch_log.appendPlainText(f"Discovered {len(specimens)} specimens. Starting sequential processing.")
            rows = run_batch(
                specimens, str(self.batch_output.currentPath),
                lambda specimen, directory: _process_specimen(self, specimen, directory),
                context=context, require_landmarks=use_landmarks,
                resume=bool(self.batch_resume.checked),
                should_cancel=lambda: self._cancel_batch,
                progress=self._batch_progress_update,
            )
            counts = Counter(row["status"] for row in rows)
            message = ", ".join(f"{counts[key]} {key}" for key in ("success", "skipped", "failed", "cancelled"))
            self.batch_log.appendPlainText(message)
            self.batch_log.appendPlainText(f"Summary: {Path(str(self.batch_output.currentPath)) / 'batch_summary.csv'}")
            self.batch_status.setText(message)
        except Exception as error:
            logging.exception("Batch Shape Completion failed")
            self.batch_log.appendPlainText(f"Batch error: {error}")
            self.batch_status.setText(f"Batch failed: {error}")
            slicer.util.errorDisplay(f"Batch Shape Completion failed:\n{error}")
        finally:
            try:
                if self.logic is not previous_logic:
                    self.logic.cleanup_specimen()
            finally:
                self.logic = previous_logic
                self._set_batch_busy(False)
                self._validate_complete_inputs()
                self._validate_calibration_inputs()
                # Keep the final counts visible instead of replacing them with Ready.
                final_status = str(self.batch_status.text)
                self._validate_batch_inputs()
                self.batch_status.setText(final_status)
