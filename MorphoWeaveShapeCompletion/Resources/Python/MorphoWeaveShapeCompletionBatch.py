"""Batch tab for Shape Completion; not a separately registered Slicer module."""
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

from Resources.Python.MorphoWeaveShapeCompletionBase import (
    MorphoWeaveShapeCompletionLogic,
    MorphoWeaveShapeCompletionWidget,
    run_completion,
)
from Resources.Python.MorphoWeaveCompletionBatch import (
    discover_specimens,
    file_sha256,
    run_batch,
)


class _BatchCompletionLogic(MorphoWeaveShapeCompletionLogic):
    """Parse the shared SSM once and own only synchronous batch I/O nodes."""

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
        return tuple(array.copy() for array in self._ssm_arrays)

    def tracked_call(self, function, *args, **kwargs):
        # Do not snapshot the entire run: processEvents() in the parent workflow
        # can allow unrelated nodes to be added elsewhere in Slicer.
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
    """Cheap guard against scene edits, including ancestor transforms."""
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


def _interpolation_settings(widget):
    return {
        "neighbors": int(widget.interpolation_neighbors.value),
        "sharpness": float(widget.interpolation_sharpness.value),
        "chunk_size": int(widget.interpolation_chunk_size.value),
        "restrict_to_components": bool(widget.component_interpolation_check.checked),
        "full_resolution_samples": bool(widget.full_resolution_samples_check.checked),
    }


def _batch_state_token(widget):
    return (_shared_input_token(widget), json.dumps({
        "settings": widget._settings_snapshot(widget._read_settings()),
        "interpolation": _interpolation_settings(widget),
        "profile": widget._profile,
    }, sort_keys=True))


def _batch_context(widget):
    """Fingerprint the shared inputs, settings and implementation for resume."""
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
        "interpolation": _interpolation_settings(widget),
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
        "entrypoint_source_sha256": file_sha256(inspect.getfile(type(widget))),
        "batch_source_sha256": file_sha256(__file__),
        "batch_runner_sha256": file_sha256(inspect.getfile(run_batch)),
        "numpy_version": np.__version__,
    }


def _process_specimen(widget, specimen, output_directory):
    """Delegate to the unchanged single-specimen implementation, then clean up."""
    previous_target = widget.target_model_selector.currentNode()
    previous_landmarks = widget.target_landmark_selector.currentNode()
    previous_directory = str(widget.output_directory.currentPath or "")
    previous_folder = widget._run_folder_item
    try:
        if _batch_state_token(widget) != widget._batch_input_token:
            raise RuntimeError("Shared inputs or settings changed during the batch; restart with unchanged inputs")
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
        if _batch_state_token(widget) != widget._batch_input_token:
            raise RuntimeError("Shared inputs or settings changed during fitting; result was not published")
    finally:
        try:
            widget.target_model_selector.setCurrentNode(previous_target)
            widget.target_landmark_selector.setCurrentNode(previous_landmarks)
            widget.output_directory.setCurrentPath(previous_directory)
            widget._run_folder_item = previous_folder
        finally:
            widget.logic.cleanup_specimen()


def _link_controls(left, right, signal, getter, setter):
    """Mirror controls in both tabs without feedback loops or duplicated state."""
    def copy_value(source, destination):
        value = getter(source)
        if getter(destination) != value:
            setter(destination, value)
    getattr(left, signal).connect(lambda *args: copy_value(left, right))
    getattr(right, signal).connect(lambda *args: copy_value(right, left))
    copy_value(left, right)


class ShapeCompletionBatchMixin:
    """Add an integrated Batch tab to the existing widget, not a second module."""

    def __init__(self, *args, **kwargs):
        self._workflow_busy = None
        self._cancel_batch = False
        super().__init__(*args, **kwargs)

    def setup(self):
        super().setup()
        self._validate_batch_inputs()

    def _build_ui(self):
        super()._build_ui()
        self.batch_tab = qt.QWidget()
        self.tabs.insertTab(1, self.batch_tab, "Batch")
        layout = qt.QFormLayout(self.batch_tab)
        intro = qt.QLabel(
            "Complete a directory of fragments with the same SSM and fitting settings. "
            "Model selections and options below are shared with Complete Shape. "
            "Advanced parameters apply to both tabs; no individual target needs to be loaded."
        )
        intro.setWordWrap(True)
        layout.addRow(intro)
        self.batch_inputs_box = ctk.ctkCollapsibleButton()
        self.batch_inputs_box.text = "Batch Inputs"
        self.batch_inputs_box.collapsed = False
        inputs = qt.QFormLayout(self.batch_inputs_box)
        layout.addRow(self.batch_inputs_box)
        self.batch_template_model = self._make_selector(["vtkMRMLModelNode"])
        self.batch_template_dense = self._make_selector(["vtkMRMLMarkupsFiducialNode"])
        self.batch_template_sparse = self._make_selector(["vtkMRMLMarkupsFiducialNode"])
        self.batch_ssm_table = self._make_selector(["vtkMRMLTableNode"], attribute="ssm_eigenvalues")
        inputs.addRow("Template surface:", self.batch_template_model)
        inputs.addRow("Template dense correspondences:", self.batch_template_dense)
        inputs.addRow("Template sparse landmarks:", self.batch_template_sparse)
        inputs.addRow("SSM Data Table:", self.batch_ssm_table)
        self.batch_input = ctk.ctkPathLineEdit()
        self.batch_landmarks = ctk.ctkPathLineEdit()
        self.batch_output = ctk.ctkPathLineEdit()
        for control in (self.batch_input, self.batch_landmarks, self.batch_output):
            control.filters = ctk.ctkPathLineEdit.Dirs
        inputs.addRow("Target mesh directory:", self.batch_input)
        inputs.addRow("Paired landmark directory (optional):", self.batch_landmarks)
        inputs.addRow("Output directory:", self.batch_output)
        self.batch_coverage = self._double(0.05, 1.0, float(self.coverage_spin.value), 0.05, 2)
        inputs.addRow("Target coverage (all specimens):", self.batch_coverage)
        self.batch_use_landmarks = qt.QCheckBox("Use matched landmarks")
        self.batch_order_fallback = qt.QCheckBox("Allow ordered landmark matching when labels do not match")
        inputs.addRow(self.batch_use_landmarks)
        inputs.addRow(self.batch_order_fallback)
        self.batch_profile = ctk.ctkPathLineEdit()
        self.batch_profile.filters = ctk.ctkPathLineEdit.Files
        self.batch_profile.nameFilters = ["Calibration profile (*.json)"]
        inputs.addRow("Calibration profile (optional):", self.batch_profile)
        self.batch_resume = qt.QCheckBox("Skip verified completed specimens")
        self.batch_resume.checked = True
        inputs.addRow(self.batch_resume)
        note = qt.QLabel(
            "PLY, VTP, VTK, STL and OBJ meshes; paired .mrk.json, .fcsv or .json landmarks "
            "must share the mesh basename. Folders are not searched recursively. "
            "All specimens use the same coverage and seed. Existing outputs are never overwritten."
        )
        note.setWordWrap(True)
        inputs.addRow(note)
        self.batch_status = qt.QLabel("")
        self.batch_status.setWordWrap(True)
        layout.addRow(self.batch_status)
        self.batch_run = qt.QPushButton("Run Batch Shape Completion")
        self.batch_run.enabled = False
        self.batch_cancel = qt.QPushButton("Cancel after current specimen")
        self.batch_cancel.enabled = False
        layout.addRow(self.batch_run)
        self.batch_progress = qt.QProgressBar()
        self.batch_progress.setRange(0, 100)
        self.batch_progress.setValue(0)
        layout.addRow("Progress:", self.batch_progress)
        layout.addRow(self.batch_cancel)
        self.batch_log = qt.QPlainTextEdit()
        self.batch_log.setReadOnly(True)
        self.batch_log.setMinimumHeight(180)
        layout.addRow(self.batch_log)
        # Keep Complete Shape as the default tab, just as before batch support.

    def _connect_ui(self):
        super()._connect_ui()
        for left, right in (
            (self.template_model_selector, self.batch_template_model),
            (self.template_dense_selector, self.batch_template_dense),
            (self.template_sparse_selector, self.batch_template_sparse),
            (self.ssm_table_selector, self.batch_ssm_table),
        ):
            _link_controls(left, right, "currentNodeChanged",
                           lambda w: w.currentNode(), lambda w, v: w.setCurrentNode(v))
            left.currentNodeChanged.connect(self._validate_batch_inputs)
        _link_controls(self.coverage_spin, self.batch_coverage, "valueChanged",
                       lambda w: float(w.value), lambda w, v: w.setValue(v))
        for left, right in ((self.use_landmarks_check, self.batch_use_landmarks),
                            (self.order_fallback_check, self.batch_order_fallback)):
            _link_controls(left, right, "toggled",
                           lambda w: bool(w.checked), lambda w, v: w.setChecked(v))
        _link_controls(self.profile_path, self.batch_profile, "currentPathChanged",
                       lambda w: str(w.currentPath or ""), lambda w, v: w.setCurrentPath(v))
        for control in (self.batch_input, self.batch_landmarks, self.batch_output):
            control.currentPathChanged.connect(self._validate_batch_inputs)
        self.use_landmarks_check.toggled.connect(self._validate_batch_inputs)
        self.batch_run.clicked.connect(self.on_run_batch)
        self.batch_cancel.clicked.connect(self.on_cancel_batch)

    def enter(self):
        if self._workflow_busy:
            return
        super().enter()
        if self._ui_ready:
            self._validate_batch_inputs()

    def cleanup(self):
        self._cancel_batch = True
        super().cleanup()

    def _validate_complete_inputs(self, *args):
        super()._validate_complete_inputs(*args)
        if self._workflow_busy:
            self.run_button.enabled = False

    def _validate_calibration_inputs(self, *args):
        super()._validate_calibration_inputs(*args)
        if self._workflow_busy:
            self.calibration_run_button.enabled = False

    def _validate_batch_inputs(self, *args):
        if self._workflow_busy:
            self.batch_run.enabled = False
            return
        required = all(selector.currentNode() is not None for selector in (
            self.template_model_selector, self.template_dense_selector, self.ssm_table_selector))
        use_landmarks = bool(self.use_landmarks_check.checked)
        self.batch_landmarks.enabled = use_landmarks
        landmark_path = str(self.batch_landmarks.currentPath or "")
        landmarks_ok = not use_landmarks or (
            self.template_sparse_selector.currentNode() is not None
            and bool(landmark_path) and Path(landmark_path).is_dir())
        input_path = str(self.batch_input.currentPath or "")
        output_path = str(self.batch_output.currentPath or "")
        input_ok = bool(input_path) and Path(input_path).is_dir()
        distinct_output = bool(output_path) and (
            not input_path or Path(input_path).resolve() != Path(output_path).resolve())
        ready = required and input_ok and landmarks_ok and distinct_output
        self.batch_run.enabled = bool(ready)
        self.batch_status.setText(
            "Ready. Each specimen is saved independently; failures will not stop the batch."
            if ready else "Select the template, dense correspondences, SSM table, target folder and a different output folder. Landmark mode also requires template and paired target landmarks."
        )

    def _set_operation(self, operation):
        self._workflow_busy = operation
        busy = bool(operation)
        for control in (self.complete_tab, self.advanced_tab, self.batch_inputs_box):
            control.setEnabled(not busy)
        # Calibration's own cancel button must remain usable during calibration.
        self.calibration_tab.setEnabled(not busy or operation == "calibration")
        self.batch_run.enabled = False
        self.batch_cancel.enabled = operation == "batch"
        self._validate_complete_inputs()
        self._validate_calibration_inputs()
        self._validate_batch_inputs()

    def on_run_completion(self):
        if self._workflow_busy:
            return
        self._set_operation("single")
        try:
            return super().on_run_completion()
        finally:
            self._set_operation(None)

    def on_run_calibration(self):
        if self._workflow_busy:
            return
        self._set_operation("calibration")
        try:
            return super().on_run_calibration()
        finally:
            self._set_operation(None)

    def on_clear_outputs(self):
        if not self._workflow_busy:
            return super().on_clear_outputs()

    def on_cancel_batch(self):
        self._cancel_batch = True
        self.batch_cancel.enabled = False
        self.batch_log.appendPlainText("Cancellation requested. The current fit and its exports will finish first.")

    def _batch_progress_update(self, row, completed, total):
        self.batch_progress.setValue(round(100 * completed / max(total, 1)))
        self.batch_status.setText(f"{completed}/{total}: {row['specimen']} - {row['status']}")
        self.batch_log.appendPlainText(f"{row['specimen']}: {row['status']}" + (f" - {row['error']}" if row['error'] else ""))
        slicer.app.processEvents()

    def on_run_batch(self):
        if self._workflow_busy:
            return
        self._cancel_batch = False
        previous_logic = self.logic
        single_view = (self.run_progress.value, self.diagnostics_text.toPlainText(),
                       str(self.complete_status.text))
        final_status = "Batch was not started."
        self._set_operation("batch")
        self.batch_log.clear()
        self.batch_progress.setValue(0)
        try:
            if not self._ensure_dependencies():
                final_status = "Batch was not started: dependencies are unavailable."
                return
            if not all(selector.currentNode() is not None for selector in (
                    self.template_model_selector, self.template_dense_selector, self.ssm_table_selector)):
                raise ValueError("Select the template surface, dense correspondences and SSM table")
            if self.profile_path.currentPath and self._profile is None:
                raise ValueError("The selected calibration profile could not be loaded")
            if not self.batch_input.currentPath or not self.batch_output.currentPath:
                raise ValueError("Select target mesh and output directories")
            use_landmarks = bool(self.use_landmarks_check.checked)
            if use_landmarks and (self.template_sparse_selector.currentNode() is None
                                  or not self.batch_landmarks.currentPath):
                raise ValueError("Landmark mode requires template sparse landmarks and a paired landmark directory")
            specimens = discover_specimens(
                str(self.batch_input.currentPath),
                str(self.batch_landmarks.currentPath) if use_landmarks else None,
            )
            slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
            self.logic = _BatchCompletionLogic(slicer.mrmlScene)
            self._batch_input_token = _batch_state_token(self)
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
            final_status = ", ".join(f"{counts[key]} {key}" for key in ("success", "skipped", "failed", "cancelled"))
            self.batch_log.appendPlainText(final_status)
            self.batch_log.appendPlainText(f"Summary: {Path(str(self.batch_output.currentPath)) / 'batch_summary.csv'}")
        except Exception as error:
            logging.exception("Batch shape completion failed")
            final_status = f"Batch failed: {error}"
            self.batch_log.appendPlainText(final_status)
            slicer.util.errorDisplay(final_status)
        finally:
            try:
                if self.logic is not previous_logic:
                    self.logic.cleanup_specimen()
            finally:
                self.logic = previous_logic
                self._set_operation(None)
                self.run_progress.setValue(single_view[0])
                self.diagnostics_text.setPlainText(single_view[1])
                self.complete_status.setText(single_view[2])
                self.batch_status.setText(final_status)
