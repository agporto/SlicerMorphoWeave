import html
import importlib
import inspect
import json
import logging
import os
from pathlib import Path
import time
import traceback

import ctk
import numpy as np
import qt
import slicer
import vtk
import vtk.util.numpy_support as vtk_np
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)

from Resources.Python.MorphoWeaveShapeCompletionCore import (
    CalibrationAccumulator,
    CompletionSettings,
    apply_similarity,
    apply_random_pose,
    bbox_diagonal,
    build_local_transfer_operator,
    build_calibration_profile,
    calibrated_radius,
    calibrated_simultaneous_radius,
    classify_completion_region,
    completion_diagnostics,
    contiguous_plane_mask,
    coordinate_compatibility,
    finalize_calibration_entry,
    indexed_correspondence_compatibility,
    infer_control_vertex_indices,
    landmark_recommendation,
    load_calibration_profile,
    match_landmarks,
    model_fingerprint,
    nearest_indices,
    pose_residual_interpolate,
    run_completion,
    save_calibration_profile,
    select_calibration_entry,
)


SUPPORTED_MODEL_EXTENSIONS = (".ply", ".vtp", ".vtk", ".stl", ".obj")
SUPPORTED_MARKUP_EXTENSIONS = (".mrk.json", ".fcsv", ".json")


def setWorkflowStatus(label, state, detail):
    title, color = {
        "needs_input": ("Needs input", "#c75b5b"),
        "ready": ("Ready", "#4f9d69"),
        "optional": ("Optional", "#c18b37"),
        "recommended": ("Recommended", "#c18b37"),
        "complete": ("Complete", "#4f9d69"),
    }[state]
    safe_detail = html.escape(str(detail)).replace("\n", "<br>")
    label.setTextFormat(qt.Qt.RichText)
    label.setText(
        f'<span style="color:{color}; font-weight:600">{title}</span> — {safe_detail}'
    )
    label.setAccessibleName(f"{title}: {detail}")


def canonical_ssm_node_names(database_name):
    return {
        "table": f"ssm_data_{database_name}",
        "model": f"{database_name}_template",
        "dense": f"{database_name}_template_correspondences",
        "sparse": f"{database_name}_template_sparse_landmarks",
    }


def latest_complete_ssm_set(tables, models, landmarks):
    models_by_name = {node.GetName(): node for node in models}
    landmarks_by_name = {node.GetName(): node for node in landmarks}
    for table in reversed(list(tables)):
        table_name = table.GetName() or ""
        if not table_name.startswith("ssm_data_"):
            continue
        database_name = table_name[len("ssm_data_") :]
        names = canonical_ssm_node_names(database_name)
        model = models_by_name.get(names["model"])
        dense = landmarks_by_name.get(names["dense"])
        sparse = landmarks_by_name.get(names["sparse"])
        if not all((database_name, model, dense, sparse)):
            continue
        try:
            expected_points = int(table.GetAttribute("ssm_npoints") or "")
        except (TypeError, ValueError):
            continue
        if expected_points > 0 and dense.GetNumberOfControlPoints() == expected_points:
            return {"table": table, "model": model, "dense": dense, "sparse": sparse}
    return None


def fill_empty_selectors(selector_nodes):
    changed = False
    for selector, node in selector_nodes:
        if selector.currentNode() is None:
            selector.setCurrentNode(node)
            changed = True
    return changed


def safe_stem(path):
    stem = Path(path).name
    lower = stem.lower()
    for suffix in (".mrk.json", ".fcsv", ".json", ".ply", ".vtp", ".vtk", ".stl", ".obj"):
        if lower.endswith(suffix):
            return stem[: -len(suffix)]
    return Path(stem).stem


class MorphoWeaveShapeCompletion(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Shape Completion"
        self.parent.categories = ["MorphoWeave"]
        self.parent.dependencies = []
        self.parent.contributors = ["Arthur Porto"]
        self.parent.helpText = (
            "Complete a partial surface with a loaded Statistical Shape Model using "
            "global pose search, finalist refinement, and pure SSM atlas registration. "
            "The module reports conditional spatial uncertainty and optionally applies "
            "a dataset-specific calibration profile."
        )
        self.parent.acknowledgementText = "This module was developed by Arthur Porto"


class MorphoWeaveShapeCompletionWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logic = None
        self._deps_ready = False
        self._dependency_preflight_attempted = False
        self._cancel_calibration = False
        self._run_folder_item = None
        self._profile = None

    def setup(self):
        super().setup()
        self.logic = MorphoWeaveShapeCompletionLogic()
        self._build_ui()
        self._connect_ui()
        self._auto_select_canonical_ssm_set()
        self._validate_complete_inputs()
        self._validate_calibration_inputs()
        self.layout.addStretch(1)
        # Resolve the native backend when the module opens so the user does not
        # first encounter installation latency after pressing Run.
        qt.QTimer.singleShot(0, self._preflight_dependencies)

    def _preflight_dependencies(self):
        if self._dependency_preflight_attempted:
            return
        self._dependency_preflight_attempted = True
        self._ensure_dependencies()

    def _make_selector(self, types, *, none=True, select_new=False, attribute=None, tooltip=""):
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = list(types)
        selector.selectNodeUponCreation = bool(select_new)
        selector.addEnabled = False
        selector.removeEnabled = False
        selector.noneEnabled = bool(none)
        selector.setMRMLScene(slicer.mrmlScene)
        if attribute:
            selector.addAttribute(types[0], attribute)
        if tooltip:
            selector.setToolTip(tooltip)
        return selector

    def _build_ui(self):
        self.tabs = qt.QTabWidget()
        self.layout.addWidget(self.tabs)
        self.complete_tab = qt.QWidget()
        self.calibration_tab = qt.QWidget()
        self.advanced_tab = qt.QWidget()
        self.tabs.addTab(self.complete_tab, "Complete Shape")
        self.tabs.addTab(self.calibration_tab, "Calibration")
        self.tabs.addTab(self.advanced_tab, "Advanced")
        complete_layout = qt.QFormLayout(self.complete_tab)
        calibration_layout = qt.QFormLayout(self.calibration_tab)
        advanced_layout = qt.QFormLayout(self.advanced_tab)

        intro = qt.QLabel(
            "Pipeline: deterministic global pose search → refinement of surviving poses → "
            "pure SSM atlas registration → conditional posterior completion. No unconstrained "
            "fine deformation is applied. Based on the current fragment experiments, homologous "
            "landmarks are recommended below 0.75 target coverage."
        )
        intro.setWordWrap(True)
        complete_layout.addRow(intro)

        required = ctk.ctkCollapsibleButton()
        required.text = "Required Inputs"
        required.collapsed = False
        required_layout = qt.QFormLayout(required)
        complete_layout.addRow(required)
        self.template_model_selector = self._make_selector(["vtkMRMLModelNode"])
        self.template_dense_selector = self._make_selector(["vtkMRMLMarkupsFiducialNode"])
        self.template_sparse_selector = self._make_selector(["vtkMRMLMarkupsFiducialNode"])
        self.ssm_table_selector = self._make_selector(
            ["vtkMRMLTableNode"], attribute="ssm_eigenvalues"
        )
        self.target_model_selector = self._make_selector(
            ["vtkMRMLModelNode"], select_new=True
        )
        self.target_landmark_selector = self._make_selector(
            ["vtkMRMLMarkupsFiducialNode"], none=True
        )
        required_layout.addRow("Template surface:", self.template_model_selector)
        required_layout.addRow("Template dense correspondences:", self.template_dense_selector)
        required_layout.addRow("Template sparse landmarks:", self.template_sparse_selector)
        required_layout.addRow("SSM Data Table:", self.ssm_table_selector)
        required_layout.addRow("Target fragment:", self.target_model_selector)
        required_layout.addRow("Target landmarks (optional):", self.target_landmark_selector)

        self.coverage_spin = ctk.ctkDoubleSpinBox()
        self.coverage_spin.minimum = 0.05
        self.coverage_spin.maximum = 1.0
        self.coverage_spin.singleStep = 0.05
        self.coverage_spin.value = 1.0
        self.coverage_spin.setDecimals(2)
        self.coverage_spin.setToolTip(
            "Estimated fraction of the complete target represented by the fragment. "
            "Defaults to 1.00. It controls fragment scaling policy, posterior completeness, "
            "calibration-profile matching, and the landmark recommendation."
        )
        required_layout.addRow("Target coverage:", self.coverage_spin)
        self.use_landmarks_check = qt.QCheckBox(
            "Use matched landmarks in pose search, pose refinement, and atlas registration"
        )
        self.use_landmarks_check.checked = False
        required_layout.addRow(self.use_landmarks_check)
        self.order_fallback_check = qt.QCheckBox(
            "Allow ordered landmark matching when labels do not match"
        )
        self.order_fallback_check.checked = False
        self.order_fallback_check.setToolTip(
            "Use only when template and target control-point order is known to be homologous. "
            "Label matching is safer and remains the default."
        )
        required_layout.addRow(self.order_fallback_check)
        self.output_directory = ctk.ctkPathLineEdit()
        self.output_directory.filters = ctk.ctkPathLineEdit.Dirs
        self.output_directory.setToolTip(
            "Optional. Scene outputs are always created; selecting a directory also writes VTP, markups, CSV, and JSON files."
        )
        required_layout.addRow("Output directory (optional):", self.output_directory)
        self.complete_status = qt.QLabel("")
        self.complete_status.setWordWrap(True)
        required_layout.addRow(self.complete_status)
        self.landmark_recommendation = qt.QLabel("")
        self.landmark_recommendation.setWordWrap(True)
        required_layout.addRow(self.landmark_recommendation)

        profile_box = ctk.ctkCollapsibleButton()
        profile_box.text = "Optional Calibration Profile"
        profile_box.collapsed = False
        profile_layout = qt.QFormLayout(profile_box)
        complete_layout.addRow(profile_box)
        self.profile_path = ctk.ctkPathLineEdit()
        self.profile_path.filters = ctk.ctkPathLineEdit.Files
        self.profile_path.nameFilters = ["Calibration profile (*.json)"]
        profile_layout.addRow("Calibration profile:", self.profile_path)
        self.profile_status = qt.QLabel(
            "No profile loaded. Model-based conditional uncertainty will still be produced."
        )
        self.profile_status.setWordWrap(True)
        profile_layout.addRow(self.profile_status)

        run_box = ctk.ctkCollapsibleButton()
        run_box.text = "Run + Diagnostics"
        run_box.collapsed = False
        run_layout = qt.QFormLayout(run_box)
        complete_layout.addRow(run_box)
        self.run_button = qt.QPushButton("Run Shape Completion")
        self.run_button.enabled = False
        run_layout.addRow(self.run_button)
        self.run_progress = qt.QProgressBar()
        self.run_progress.setRange(0, 100)
        self.run_progress.setValue(0)
        run_layout.addRow(self.run_progress)
        self.diagnostics_text = qt.QPlainTextEdit()
        self.diagnostics_text.setReadOnly(True)
        self.diagnostics_text.setMinimumHeight(230)
        self.diagnostics_text.setPlainText(
            "No completion yet. Uncalibrated posterior uncertainty is conditional on the selected pose and fitted surface correspondences."
        )
        run_layout.addRow(self.diagnostics_text)
        self.reset_button = qt.QPushButton("Clear Shape Completion Outputs")
        run_layout.addRow(self.reset_button)

        calibration_intro = qt.QLabel(
            "Build a model- and coverage-specific calibration profile from complete meshes. "
            "Paired dense truth gives exact correspondence error; otherwise calibration targets "
            "distance to the complete surface. Profiles estimate a marginal pointwise radius and, "
            "when enough independent meshes are available, a conservative meshwise simultaneous envelope. "
            "Profiles made from SSM training meshes are marked as internal and may be optimistic for new specimens."
        )
        calibration_intro.setWordWrap(True)
        calibration_layout.addRow(calibration_intro)
        cal_inputs = ctk.ctkCollapsibleButton()
        cal_inputs.text = "Calibration Dataset"
        cal_inputs.collapsed = False
        cal_input_layout = qt.QFormLayout(cal_inputs)
        calibration_layout.addRow(cal_inputs)
        self.calibration_mesh_directory = ctk.ctkPathLineEdit()
        self.calibration_mesh_directory.filters = ctk.ctkPathLineEdit.Dirs
        self.calibration_dense_directory = ctk.ctkPathLineEdit()
        self.calibration_dense_directory.filters = ctk.ctkPathLineEdit.Dirs
        self.calibration_landmark_directory = ctk.ctkPathLineEdit()
        self.calibration_landmark_directory.filters = ctk.ctkPathLineEdit.Dirs
        cal_input_layout.addRow("Complete mesh directory:", self.calibration_mesh_directory)
        cal_input_layout.addRow(
            "Paired dense truth directory (optional):", self.calibration_dense_directory
        )
        cal_input_layout.addRow(
            "Paired target landmark directory (optional):", self.calibration_landmark_directory
        )
        self.calibration_coverage_edit = qt.QLineEdit("1.00, 0.75, 0.60, 0.45")
        cal_input_layout.addRow("Coverage levels:", self.calibration_coverage_edit)
        self.calibration_replicates = qt.QSpinBox()
        self.calibration_replicates.minimum = 1
        self.calibration_replicates.maximum = 100
        self.calibration_replicates.value = 2
        cal_input_layout.addRow("Fragments per mesh and coverage:", self.calibration_replicates)
        self.calibration_use_landmarks = qt.QCheckBox(
            "Calibrate landmark-assisted completion"
        )
        self.calibration_use_landmarks.checked = False
        cal_input_layout.addRow(self.calibration_use_landmarks)
        self.calibration_random_pose = qt.QCheckBox(
            "Apply a deterministic random rigid pose to each generated fragment"
        )
        self.calibration_random_pose.checked = True
        cal_input_layout.addRow(self.calibration_random_pose)
        self.calibration_training_set = qt.QCheckBox(
            "These complete meshes were used to train the loaded SSM"
        )
        self.calibration_training_set.checked = False
        self.calibration_training_set.setToolTip(
            "Enable only when the calibration meshes are members of the SSM training set. "
            "Such profiles are labelled internal and may be optimistic on new specimens."
        )
        cal_input_layout.addRow(self.calibration_training_set)

        cal_parameters = ctk.ctkCollapsibleButton()
        cal_parameters.text = "Calibration Parameters"
        cal_parameters.collapsed = False
        cal_parameter_layout = qt.QFormLayout(cal_parameters)
        calibration_layout.addRow(cal_parameters)
        self.calibration_alpha = ctk.ctkDoubleSpinBox()
        self.calibration_alpha.minimum = 0.01
        self.calibration_alpha.maximum = 0.50
        self.calibration_alpha.singleStep = 0.01
        self.calibration_alpha.value = 0.10
        self.calibration_alpha.setDecimals(2)
        cal_parameter_layout.addRow("Miscoverage α:", self.calibration_alpha)
        self.calibration_success_threshold = ctk.ctkDoubleSpinBox()
        self.calibration_success_threshold.minimum = 0.1
        self.calibration_success_threshold.maximum = 25.0
        self.calibration_success_threshold.singleStep = 0.5
        self.calibration_success_threshold.value = 3.0
        self.calibration_success_threshold.setDecimals(1)
        self.calibration_success_threshold.setSuffix(" %")
        self.calibration_success_threshold.setToolTip(
            "A fit is labelled successful when inferred-region RMS error is no more than this percentage of the complete-shape bounding-box diagonal."
        )
        cal_parameter_layout.addRow("Fit-success threshold:", self.calibration_success_threshold)
        output_row = qt.QHBoxLayout()
        self.calibration_output_edit = qt.QLineEdit()
        self.calibration_browse_button = qt.QToolButton()
        self.calibration_browse_button.text = "…"
        output_row.addWidget(self.calibration_output_edit)
        output_row.addWidget(self.calibration_browse_button)
        cal_parameter_layout.addRow("Save profile as:", output_row)

        cal_run = ctk.ctkCollapsibleButton()
        cal_run.text = "Run Calibration"
        cal_run.collapsed = False
        cal_run_layout = qt.QFormLayout(cal_run)
        calibration_layout.addRow(cal_run)
        buttons = qt.QHBoxLayout()
        self.calibration_run_button = qt.QPushButton("Generate Calibration Profile")
        self.calibration_run_button.enabled = False
        self.calibration_cancel_button = qt.QPushButton("Cancel")
        self.calibration_cancel_button.enabled = False
        buttons.addWidget(self.calibration_run_button)
        buttons.addWidget(self.calibration_cancel_button)
        cal_run_layout.addRow(buttons)
        self.calibration_progress = qt.QProgressBar()
        self.calibration_progress.setRange(0, 100)
        self.calibration_progress.setValue(0)
        cal_run_layout.addRow(self.calibration_progress)
        self.calibration_status = qt.QLabel("")
        self.calibration_status.setWordWrap(True)
        cal_run_layout.addRow(self.calibration_status)
        self.calibration_log = qt.QPlainTextEdit()
        self.calibration_log.setReadOnly(True)
        self.calibration_log.setMinimumHeight(260)
        cal_run_layout.addRow(self.calibration_log)

        self._build_advanced_ui(advanced_layout)

    def _build_advanced_ui(self, layout):
        sampling = ctk.ctkCollapsibleButton()
        sampling.text = "SSM and Fragment Preparation"
        sampling.collapsed = False
        form = qt.QFormLayout(sampling)
        layout.addRow(sampling)
        self.variance_keep = ctk.ctkDoubleSpinBox()
        self.variance_keep.minimum = 0.50
        self.variance_keep.maximum = 1.00
        self.variance_keep.singleStep = 0.01
        self.variance_keep.value = 0.95
        self.variance_keep.setDecimals(2)
        form.addRow("Retained SSM variance:", self.variance_keep)
        self.registration_point_count = qt.QSpinBox()
        self.registration_point_count.minimum = 100
        self.registration_point_count.maximum = 1000000
        self.registration_point_count.singleStep = 100
        self.registration_point_count.value = 2000
        form.addRow("Target registration points:", self.registration_point_count)
        self.coverage_prescale_check = qt.QCheckBox(
            "Pre-scale the SSM using target extent and supplied coverage"
        )
        self.coverage_prescale_check.checked = True
        self.coverage_prescale_check.setToolTip(
            "For partial targets, estimate full linear extent as fragment extent / coverage, "
            "then fix residual scale during pose and atlas optimization. Disable when source and target are already in the same physical scale."
        )
        form.addRow(self.coverage_prescale_check)
        self.scale_policy_combo = qt.QComboBox()
        self.scale_policy_combo.addItems(
            ["Automatic (free at 1.0; fixed for fragments)", "Always free", "Always fixed"]
        )
        form.addRow("Residual scale policy:", self.scale_policy_combo)
        self.landmark_sigma_percent = ctk.ctkDoubleSpinBox()
        self.landmark_sigma_percent.minimum = 0.05
        self.landmark_sigma_percent.maximum = 25.0
        self.landmark_sigma_percent.singleStep = 0.25
        self.landmark_sigma_percent.value = 2.0
        self.landmark_sigma_percent.setDecimals(2)
        self.landmark_sigma_percent.setSuffix(" %")
        self.landmark_sigma_percent.setToolTip(
            "Target-landmark localization standard deviation τ as a percentage of the expected full-shape RMS radius. The same τ is used in pose scoring, pose refinement, and atlas registration."
        )
        form.addRow("Landmark localization σ:", self.landmark_sigma_percent)

        pose_box = ctk.ctkCollapsibleButton()
        pose_box.text = "Pose Search and Refinement"
        pose_box.collapsed = True
        pose = qt.QFormLayout(pose_box)
        layout.addRow(pose_box)
        self.pose_rotation_count = self._spin(3, 2000, 193)
        self.pose_coarse_source = self._spin(20, 100000, 350)
        self.pose_coarse_target = self._spin(20, 100000, 350)
        self.pose_coarse_rank = self._spin(1, 1000, 8)
        self.pose_coarse_iterations = self._spin(1, 1000, 8)
        self.pose_screen_iterations = self._spin(1, 1000, 8)
        self.pose_survivors = self._spin(1, 2000, 193)
        self.pose_refine_count = self._spin(1, 2000, 8)
        self.pose_refine_source = self._spin(0, 100000, 0)
        self.pose_refine_source.setToolTip("0 uses every retained SSM source point.")
        self.pose_refine_target = self._spin(20, 100000, 1200)
        self.pose_refine_iterations = self._spin(1, 1000, 25)
        self.pose_lambda = self._double(0.0, 1000.0, 5.0, 0.5, 4)
        self.pose_outlier = self._double(0.0, 0.95, 0.10, 0.01, 3)
        self.pose_identity_prior = self._double(0.001, 0.999, 0.20, 0.01, 3)
        pose.addRow("Rotation hypotheses:", self.pose_rotation_count)
        pose.addRow("Coarse source points:", self.pose_coarse_source)
        pose.addRow("Coarse target points:", self.pose_coarse_target)
        pose.addRow("Coarse SSM rank:", self.pose_coarse_rank)
        pose.addRow("Coarse iterations:", self.pose_coarse_iterations)
        pose.addRow("Screen iterations:", self.pose_screen_iterations)
        pose.addRow("Surviving hypotheses:", self.pose_survivors)
        pose.addRow("Refined hypotheses:", self.pose_refine_count)
        pose.addRow("Refine source points (0=all):", self.pose_refine_source)
        pose.addRow("Refine target points:", self.pose_refine_target)
        pose.addRow("Refine iterations:", self.pose_refine_iterations)
        pose.addRow("Pose SSM regularization:", self.pose_lambda)
        pose.addRow("Pose outlier weight:", self.pose_outlier)
        pose.addRow("Identity prior probability:", self.pose_identity_prior)

        atlas_box = ctk.ctkCollapsibleButton()
        atlas_box.text = "Pure Atlas Registration"
        atlas_box.collapsed = True
        atlas = qt.QFormLayout(atlas_box)
        layout.addRow(atlas_box)
        self.atlas_lambda = self._double(0.0, 1000.0, 5.0, 0.5, 4)
        self.atlas_outlier = self._double(0.0, 0.95, 0.10, 0.01, 3)
        self.atlas_iterations = self._spin(1, 5000, 150)
        self.atlas_tolerance = self._double(1e-10, 1e-2, 1e-7, 1e-7, 10)
        self.atlas_k = self._spin(0, 1000, 0)
        self.atlas_k.setToolTip("0 uses the exact dense E-step; a positive value uses sparse k-nearest correspondences.")
        atlas.addRow("Atlas SSM regularization:", self.atlas_lambda)
        atlas.addRow("Atlas outlier weight:", self.atlas_outlier)
        atlas.addRow("Maximum iterations:", self.atlas_iterations)
        atlas.addRow("Tolerance:", self.atlas_tolerance)
        atlas.addRow("Sparse E-step k (0=exact):", self.atlas_k)

        uncertainty_box = ctk.ctkCollapsibleButton()
        uncertainty_box.text = "Uncertainty and Output Surface"
        uncertainty_box.collapsed = False
        uncertainty = qt.QFormLayout(uncertainty_box)
        layout.addRow(uncertainty_box)
        self.estimate_discrepancy_check = qt.QCheckBox(
            "Include model discrepancy in total predictive uncertainty"
        )
        self.estimate_discrepancy_check.checked = True
        uncertainty.addRow(self.estimate_discrepancy_check)
        self.posterior_samples = self._spin(0, 100, 0)
        self.posterior_samples.setToolTip(
            "Samples are drawn from the latent coefficient posterior. They do not add independent observation noise or model-discrepancy noise."
        )
        uncertainty.addRow("Latent posterior samples:", self.posterior_samples)
        self.full_resolution_samples_check = qt.QCheckBox(
            "Create full-resolution sample surfaces (memory intensive)"
        )
        self.full_resolution_samples_check.checked = False
        self.full_resolution_samples_check.setToolTip(
            "Disabled by default: each million-vertex sample can consume substantial scene memory. When disabled, samples are stored as the dense SSM point cloud."
        )
        uncertainty.addRow(self.full_resolution_samples_check)
        self.interpolation_neighbors = self._spin(1, 64, 8)
        self.interpolation_neighbors.setToolTip(
            "Number of dense SSM controls retained per full-resolution vertex. Eight is a good starting point for approximately 5,000 controls."
        )
        uncertainty.addRow("Surface interpolation neighbors:", self.interpolation_neighbors)
        self.interpolation_sharpness = self._double(0.25, 8.0, 2.0, 0.25, 2)
        self.interpolation_sharpness.setToolTip(
            "Gaussian locality multiplier. A value of 2 gives the farthest retained neighbor an unnormalized weight of exp(-4)."
        )
        uncertainty.addRow("Interpolation locality:", self.interpolation_sharpness)
        self.interpolation_chunk_size = self._spin(10000, 1000000, 200000)
        self.interpolation_chunk_size.setSingleStep(10000)
        self.interpolation_chunk_size.setToolTip(
            "Number of full-resolution vertices processed per k-d-tree query chunk. Lower values reduce temporary memory without changing the result."
        )
        uncertainty.addRow("Transfer build chunk size:", self.interpolation_chunk_size)
        self.component_interpolation_check = qt.QCheckBox(
            "Restrict interpolation to connected template components"
        )
        self.component_interpolation_check.checked = True
        self.component_interpolation_check.setToolTip(
            "Prevents controls on one disconnected mesh component from deforming another. Components without any dense controls fall back to the global control set and are reported in diagnostics."
        )
        uncertainty.addRow(self.component_interpolation_check)
        self.random_seed = self._spin(0, 2147483647, 0)
        uncertainty.addRow("Random seed:", self.random_seed)
        self.parallel_check = qt.QCheckBox("Use deterministic Rust parallelism")
        self.parallel_check.checked = True
        uncertainty.addRow(self.parallel_check)

    def _spin(self, minimum, maximum, value):
        widget = qt.QSpinBox()
        widget.minimum = int(minimum)
        widget.maximum = int(maximum)
        widget.value = int(value)
        return widget

    def _double(self, minimum, maximum, value, step, decimals):
        widget = ctk.ctkDoubleSpinBox()
        widget.minimum = float(minimum)
        widget.maximum = float(maximum)
        widget.singleStep = float(step)
        widget.value = float(value)
        widget.setDecimals(int(decimals))
        return widget

    def _connect_ui(self):
        for selector in (
            self.template_model_selector,
            self.template_dense_selector,
            self.template_sparse_selector,
            self.ssm_table_selector,
            self.target_model_selector,
            self.target_landmark_selector,
        ):
            selector.currentNodeChanged.connect(self._validate_complete_inputs)
        for selector in (
            self.template_dense_selector,
            self.template_sparse_selector,
            self.ssm_table_selector,
        ):
            selector.currentNodeChanged.connect(self._validate_calibration_inputs)
        self.coverage_spin.valueChanged.connect(self._validate_complete_inputs)
        self.use_landmarks_check.toggled.connect(self._validate_complete_inputs)
        self.order_fallback_check.toggled.connect(self._validate_complete_inputs)
        self.profile_path.currentPathChanged.connect(self._on_profile_path_changed)
        self.run_button.clicked.connect(self.on_run_completion)
        self.reset_button.clicked.connect(self.on_clear_outputs)
        for path_widget in (
            self.calibration_mesh_directory,
            self.calibration_dense_directory,
            self.calibration_landmark_directory,
        ):
            path_widget.currentPathChanged.connect(self._validate_calibration_inputs)
        self.calibration_coverage_edit.textChanged.connect(self._validate_calibration_inputs)
        self.calibration_use_landmarks.toggled.connect(self._validate_calibration_inputs)
        self.calibration_output_edit.textChanged.connect(self._validate_calibration_inputs)
        self.calibration_browse_button.clicked.connect(self._choose_calibration_output)
        self.calibration_run_button.clicked.connect(self.on_run_calibration)
        self.calibration_cancel_button.clicked.connect(self.on_cancel_calibration)

    def enter(self):
        self._auto_select_canonical_ssm_set()
        self._validate_complete_inputs()
        self._validate_calibration_inputs()

    def cleanup(self):
        self._cancel_calibration = True
        super().cleanup()

    def _latest_complete_ssm_set(self):
        tables = slicer.util.getNodesByClass("vtkMRMLTableNode")
        models = slicer.util.getNodesByClass("vtkMRMLModelNode")
        landmarks = slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")
        return latest_complete_ssm_set(tables, models, landmarks)

    def _auto_select_canonical_ssm_set(self):
        resolved = self._latest_complete_ssm_set()
        if not resolved:
            return False
        selectors = (
            (self.template_model_selector, resolved["model"]),
            (self.template_dense_selector, resolved["dense"]),
            (self.template_sparse_selector, resolved["sparse"]),
            (self.ssm_table_selector, resolved["table"]),
        )
        blockers = []
        try:
            for selector, _ in selectors:
                if selector.currentNode() is None:
                    blockers.append(qt.QSignalBlocker(selector))
            return fill_empty_selectors(selectors)
        finally:
            blockers.clear()

    def _validate_complete_inputs(self, *args):
        required = (
            self.template_model_selector.currentNode(),
            self.template_dense_selector.currentNode(),
            self.ssm_table_selector.currentNode(),
            self.target_model_selector.currentNode(),
        )
        use_landmarks = bool(self.use_landmarks_check.checked)
        template_sparse = self.template_sparse_selector.currentNode()
        target_sparse = self.target_landmark_selector.currentNode()
        matched = 0
        landmark_error = None
        landmarks_present = bool(template_sparse and target_sparse)
        if landmarks_present:
            try:
                matched = self._preview_landmark_match_count()
            except Exception as error:
                landmark_error = str(error)
        landmarks_ok = not use_landmarks or (landmarks_present and landmark_error is None and matched >= 3)
        ready = all(required) and landmarks_ok
        self.run_button.enabled = bool(ready)
        if not all(required):
            setWorkflowStatus(
                self.complete_status,
                "needs_input",
                "Select template surface, dense correspondences, SSM table, and target fragment.",
            )
        elif use_landmarks and not landmarks_present:
            setWorkflowStatus(
                self.complete_status,
                "needs_input",
                "Landmark mode is enabled; select template and target sparse landmarks.",
            )
        elif use_landmarks and landmark_error:
            setWorkflowStatus(
                self.complete_status,
                "needs_input",
                f"Landmark configuration is not usable: {landmark_error}",
            )
        else:
            setWorkflowStatus(
                self.complete_status,
                "ready",
                "Inputs are available for the three-stage completion pipeline.",
            )
        state, message = landmark_recommendation(float(self.coverage_spin.value), matched)
        display_state = {
            "recommended": "recommended",
            "insufficient": "recommended",
            "available": "ready",
            "optional": "optional",
        }[state]
        if landmark_error and landmarks_present:
            message += f" Current landmark selection: {landmark_error}"
            display_state = "recommended" if use_landmarks else display_state
        setWorkflowStatus(self.landmark_recommendation, display_state, message)

    def _validate_calibration_inputs(self, *args):
        mesh_dir = str(self.calibration_mesh_directory.currentPath or "")
        output = str(self.calibration_output_edit.text or "").strip()
        try:
            coverage = self._parse_coverages()
            coverage_ok = bool(coverage)
        except Exception:
            coverage_ok = False
        landmark_mode = bool(self.calibration_use_landmarks.checked)
        landmarks_ok = (
            not landmark_mode
            or bool(self.calibration_landmark_directory.currentPath)
        )
        ssm_ok = bool(
            self.ssm_table_selector.currentNode()
            and self.template_dense_selector.currentNode()
            and (not landmark_mode or self.template_sparse_selector.currentNode())
        )
        ready = bool(
            ssm_ok and os.path.isdir(mesh_dir) and output and coverage_ok and landmarks_ok
        )
        self.calibration_run_button.enabled = ready
        if not ssm_ok:
            setWorkflowStatus(
                self.calibration_status,
                "needs_input",
                "Load an SSM and template dense correspondences; landmark calibration also requires template sparse landmarks.",
            )
        elif not os.path.isdir(mesh_dir):
            setWorkflowStatus(
                self.calibration_status,
                "needs_input",
                "Select a directory containing complete meshes.",
            )
        elif not coverage_ok:
            setWorkflowStatus(
                self.calibration_status,
                "needs_input",
                "Enter comma-separated coverage values between 0.05 and 1.00.",
            )
        elif not landmarks_ok:
            setWorkflowStatus(
                self.calibration_status,
                "needs_input",
                "Landmark calibration requires a paired target-landmark directory.",
            )
        elif not output:
            setWorkflowStatus(
                self.calibration_status,
                "needs_input",
                "Choose an output calibration-profile filename.",
            )
        else:
            note = (
                "Profile will be marked as internal because the meshes are declared part of SSM training."
                if self.calibration_training_set.checked
                else "Profile will be marked as externally calibrated."
            )
            setWorkflowStatus(self.calibration_status, "ready", note)

    def _parse_coverages(self):
        text = str(self.calibration_coverage_edit.text)
        values = []
        for part in text.replace(";", ",").split(","):
            if not part.strip():
                continue
            value = float(part)
            if not 0.05 <= value <= 1.0:
                raise ValueError("coverage outside [0.05, 1.0]")
            values.append(round(value, 6))
        return sorted(set(values), reverse=True)

    def _choose_calibration_output(self):
        start = str(self.calibration_output_edit.text or "")
        if not start:
            start = os.path.join(str(Path.home()), "MorphoWeaveShapeCompletionCalibration.json")
        selected = qt.QFileDialog.getSaveFileName(
            self.parent,
            "Save Shape Completion Calibration Profile",
            start,
            "JSON (*.json)",
        )
        if isinstance(selected, (tuple, list)):
            selected = selected[0]
        if selected:
            if not str(selected).lower().endswith(".json"):
                selected = str(selected) + ".json"
            self.calibration_output_edit.setText(str(selected))

    def _on_profile_path_changed(self, path):
        self._profile = None
        path = str(path or "")
        if not path:
            self.profile_status.setText(
                "No profile loaded. Model-based conditional uncertainty will still be produced."
            )
            return
        try:
            self._profile = load_calibration_profile(path)
            scope = self._profile.get("calibration_scope_warning", "")
            entries = len(self._profile.get("entries", []))
            self.profile_status.setText(f"Loaded {entries} calibration entries. {scope}")
        except Exception as error:
            self.profile_status.setText(f"Calibration profile could not be loaded: {error}")

    def _read_settings(self, *, coverage=None, use_landmarks=None, samples=None):
        policy_index = int(self.scale_policy_combo.currentIndex)
        policy = {0: "auto", 1: "free", 2: "fixed"}[policy_index]
        refine_source = int(self.pose_refine_source.value)
        atlas_k = int(self.atlas_k.value)
        return CompletionSettings(
            coverage=float(self.coverage_spin.value if coverage is None else coverage),
            variance_keep=float(self.variance_keep.value),
            target_point_count=int(self.registration_point_count.value),
            coverage_prescale=bool(self.coverage_prescale_check.checked),
            residual_scale_policy=policy,
            use_landmarks=bool(
                self.use_landmarks_check.checked if use_landmarks is None else use_landmarks
            ),
            landmark_sigma_percent=float(self.landmark_sigma_percent.value),
            rotation_count=int(self.pose_rotation_count.value),
            coarse_source_count=int(self.pose_coarse_source.value),
            coarse_target_count=int(self.pose_coarse_target.value),
            coarse_rank=int(self.pose_coarse_rank.value),
            coarse_iterations=int(self.pose_coarse_iterations.value),
            coarse_screen_iterations=int(self.pose_screen_iterations.value),
            coarse_survivor_count=int(self.pose_survivors.value),
            refine_count=int(self.pose_refine_count.value),
            refine_source_count=None if refine_source == 0 else refine_source,
            refine_target_count=int(self.pose_refine_target.value),
            refine_iterations=int(self.pose_refine_iterations.value),
            pose_lambda=float(self.pose_lambda.value),
            pose_outlier_weight=float(self.pose_outlier.value),
            identity_prior_probability=float(self.pose_identity_prior.value),
            atlas_lambda=float(self.atlas_lambda.value),
            atlas_outlier_weight=float(self.atlas_outlier.value),
            atlas_max_iterations=int(self.atlas_iterations.value),
            atlas_tolerance=float(self.atlas_tolerance.value),
            atlas_k=None if atlas_k == 0 else atlas_k,
            estimate_discrepancy=bool(self.estimate_discrepancy_check.checked),
            posterior_samples=int(self.posterior_samples.value if samples is None else samples),
            seed=int(self.random_seed.value),
            parallel=bool(self.parallel_check.checked),
        )

    def _settings_snapshot(self, settings):
        values = dict(settings.__dict__)
        # Ordered fallback changes the landmark observation map and therefore the
        # calibration regime even though it is not a native rustcpd parameter.
        values["allow_order_fallback"] = bool(self.order_fallback_check.checked)
        return values

    def _ensure_dependencies(self):
        if self._deps_ready:
            return True
        import slicer.packaging

        try:
            slicer.packaging.pip_ensure(
                ["rustcpd>=3.0,<4"],
                prompt_install=True,
                requester="Shape Completion",
            )
            importlib.invalidate_caches()
            rustcpd = importlib.import_module("rustcpd")
            pose_parameters = inspect.signature(rustcpd.pose_initialize).parameters
            atlas_parameters = inspect.signature(rustcpd.register_atlas).parameters
            missing = []
            pose_required = (
                "rotation_count",
                "coarse_source_count",
                "coarse_target_count",
                "coarse_rank",
                "coarse_iterations",
                "coarse_screen_iterations",
                "coarse_survivor_count",
                "coarse_score_mode",
                "refine_count",
                "refine_source_count",
                "refine_target_count",
                "refine_iterations",
                "lambda_regularization",
                "outlier_weight",
                "identity_prior_probability",
                "landmark_indices",
                "landmark_targets",
                "landmark_sigma",
                "refine_landmark_sigma",
                "with_scale",
                "seed",
                "parallel",
                "single_precision",
            )
            atlas_required = (
                "lambda_regularization",
                "normalize",
                "optimize_similarity",
                "with_scale",
                "initial_coefficients",
                "initial_rotation",
                "initial_scale",
                "initial_translation",
                "landmark_indices",
                "landmark_targets",
                "landmark_sigma",
                "max_iterations",
                "tolerance",
                "outlier_weight",
                "k",
                "parallel",
                "single_precision",
            )
            for name in pose_required:
                if name not in pose_parameters:
                    missing.append(f"pose_initialize.{name}")
            for name in atlas_required:
                if name not in atlas_parameters:
                    missing.append(f"register_atlas.{name}")
            required_classes = ("PoseInitialization", "ShapePosterior", "AtlasResult")
            for name in required_classes:
                if not hasattr(rustcpd, name):
                    missing.append(name)
            if hasattr(rustcpd, "PoseInitialization"):
                for name in (
                    "coefficients",
                    "rotation",
                    "scale",
                    "translation",
                    "score",
                    "score_margin",
                    "posterior_entropy",
                    "effective_hypotheses",
                    "hypotheses_evaluated",
                    "hypotheses_refined",
                ):
                    if not hasattr(rustcpd.PoseInitialization, name):
                        missing.append(f"PoseInitialization.{name}")
            if hasattr(rustcpd, "AtlasResult"):
                for name in (
                    "points",
                    "coefficients",
                    "rotation",
                    "scale",
                    "translation",
                    "sigma2",
                    "iterations",
                    "difference",
                    "landmark_rms",
                    "posterior",
                ):
                    if not hasattr(rustcpd.AtlasResult, name):
                        missing.append(f"AtlasResult.{name}")
                if hasattr(rustcpd.AtlasResult, "posterior"):
                    posterior_parameters = inspect.signature(
                        rustcpd.AtlasResult.posterior
                    ).parameters
                    for name in (
                        "completeness",
                        "prior_temperature",
                        "outlier_weight",
                        "estimate_discrepancy",
                    ):
                        if name not in posterior_parameters:
                            missing.append(f"AtlasResult.posterior.{name}")
            if hasattr(rustcpd, "ShapePosterior"):
                for name in (
                    "coefficient_covariance",
                    "noise_variance",
                    "discrepancy_variance",
                    "predict",
                    "predictive_variance",
                    "sample_shapes",
                ):
                    if not hasattr(rustcpd.ShapePosterior, name):
                        missing.append(f"ShapePosterior.{name}")
            if missing:
                raise RuntimeError(
                    "The installed rustcpd build lacks the constrained shape-completion API: "
                    + ", ".join(missing)
                    + ". Install a build containing the pose-landmark-keypoints functionality. "
                    "Because that development branch still reports version 3.0.0, an older 3.0.0 wheel may need to be explicitly uninstalled or force-reinstalled."
                )
            self._deps_ready = True
            return True
        except Exception as error:
            if isinstance(error, RuntimeError) and str(error) == "User declined package installation":
                slicer.util.showStatusMessage(
                    "Shape Completion dependencies were not installed.", 3000
                )
                return False
            logging.exception("Shape Completion dependency setup failed")
            slicer.util.errorDisplay(f"Shape Completion dependency setup failed:\n{error}")
            return False

    def _preview_landmark_match_count(self):
        template = self.template_sparse_selector.currentNode()
        target = self.target_landmark_selector.currentNode()
        dense = self.template_dense_selector.currentNode()
        if not all((template, target, dense)):
            return 0
        labels_t, points_t = self.logic.markups_labels_points_world(template)
        labels_x, points_x = self.logic.markups_labels_points_world(target)
        dense_points = self.logic.markups_points_world(dense)
        match = match_landmarks(
            labels_t,
            points_t,
            labels_x,
            points_x,
            dense_points,
            allow_order_fallback=bool(self.order_fallback_check.checked),
        )
        return match.count

    def _landmark_match_for_nodes(self, template_sparse, target_sparse, dense):
        if not all((template_sparse, target_sparse, dense)):
            return None
        labels_t, points_t = self.logic.markups_labels_points_world(template_sparse)
        labels_x, points_x = self.logic.markups_labels_points_world(target_sparse)
        dense_points = self.logic.markups_points_world(dense)
        return match_landmarks(
            labels_t,
            points_t,
            labels_x,
            points_x,
            dense_points,
            allow_order_fallback=bool(self.order_fallback_check.checked),
        )

    def on_run_completion(self):
        if not self._ensure_dependencies():
            return
        self.run_button.enabled = False
        self.run_progress.setValue(0)
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self._run_completion_impl()
        except Exception as error:
            logging.exception("Shape Completion failed")
            self.diagnostics_text.setPlainText(
                f"Shape Completion failed:\n{error}\n\n{traceback.format_exc()}"
            )
            slicer.util.errorDisplay(f"Shape Completion failed:\n{error}")
        finally:
            qt.QApplication.restoreOverrideCursor()
            self._validate_complete_inputs()

    def _run_completion_impl(self):
        import rustcpd

        template_model = self.template_model_selector.currentNode()
        dense_node = self.template_dense_selector.currentNode()
        sparse_node = self.template_sparse_selector.currentNode()
        table_node = self.ssm_table_selector.currentNode()
        target_node = self.target_model_selector.currentNode()
        if not all((template_model, dense_node, table_node, target_node)):
            raise ValueError("required completion inputs are missing")
        self.run_progress.setValue(5)
        slicer.app.processEvents()
        mean, modes, eigenvalues = self.logic.ssm_from_table(table_node)
        dense_points = self.logic.markups_points_world(dense_node)
        if len(dense_points) != len(mean):
            raise ValueError(
                f"Template dense correspondences contain {len(dense_points)} points, but the SSM contains {len(mean)}."
            )
        ssm_template_compatibility = indexed_correspondence_compatibility(
            mean, dense_points
        )
        template_surface_points = self.logic.model_points_world(template_model)
        template_compatibility = coordinate_compatibility(
            template_surface_points,
            dense_points,
            "template dense correspondences",
        )
        target_points = self.logic.model_points_world(target_node)
        settings = self._read_settings()
        landmarks = None
        if settings.use_landmarks:
            landmarks = self._landmark_match_for_nodes(
                sparse_node, self.target_landmark_selector.currentNode(), dense_node
            )
            coordinate_compatibility(
                target_points,
                landmarks.target_points,
                "target landmarks",
                max_median_fraction=0.15,
                max_p95_fraction=0.35,
            )
        self.run_progress.setValue(15)
        self.diagnostics_text.setPlainText(
            "Running global pose search and finalist refinement…\n"
            "Pose scores are within-run ranking diagnostics, not probabilities."
        )
        slicer.app.processEvents()
        result = run_completion(
            target_points,
            mean,
            modes,
            eigenvalues,
            settings,
            rustcpd_module=rustcpd,
            landmarks=landmarks,
        )
        self.run_progress.setValue(75)
        slicer.app.processEvents()
        fingerprint = model_fingerprint(mean, modes, eigenvalues)
        profile_entry = None
        profile_message = "No calibration profile selected."
        if self._profile is not None:
            profile_entry, profile_message = select_calibration_entry(
                self._profile,
                fingerprint=fingerprint,
                coverage=settings.coverage,
                landmark_mode=bool(landmarks is not None and settings.use_landmarks),
                current_settings=self._settings_snapshot(settings),
            )
        pointwise_radius = (
            calibrated_radius(result.total_variance, profile_entry)
            if profile_entry is not None
            else None
        )
        simultaneous_radius = (
            calibrated_simultaneous_radius(result.total_variance, profile_entry)
            if profile_entry is not None
            else None
        )
        diagnostics = completion_diagnostics(
            result,
            settings,
            landmark_count=0 if landmarks is None else landmarks.count,
            calibration_entry=profile_entry,
            calibration_message=profile_message,
        )
        diagnostics["ssm_mean_template_dense_indexed_rms"] = ssm_template_compatibility["rms_distance"]
        diagnostics["ssm_mean_template_dense_indexed_rms_fraction"] = ssm_template_compatibility["rms_fraction"]
        diagnostics["template_dense_surface_median_distance"] = template_compatibility["median_distance"]
        diagnostics["template_dense_surface_p95_distance"] = template_compatibility["p95_distance"]
        diagnostics["landmark_order_fallback_used"] = bool(
            landmarks is not None and landmarks.used_order_fallback
        )
        self.run_progress.setValue(82)
        self.diagnostics_text.setPlainText(
            "Registration finished. Building or reusing the sparse full-resolution transfer operator…"
        )
        slicer.app.processEvents()
        outputs = self.logic.create_completion_outputs(
            template_model=template_model,
            template_dense_node=dense_node,
            template_sparse_node=sparse_node,
            target_node=target_node,
            result=result,
            diagnostics=diagnostics,
            calibrated_pointwise_radius_values=pointwise_radius,
            calibrated_simultaneous_radius_values=simultaneous_radius,
            interpolation_neighbors=int(self.interpolation_neighbors.value),
            interpolation_sharpness=float(self.interpolation_sharpness.value),
            interpolation_chunk_size=int(self.interpolation_chunk_size.value),
            restrict_to_components=bool(self.component_interpolation_check.checked),
            full_resolution_samples=bool(self.full_resolution_samples_check.checked),
        )
        output_directory = str(self.output_directory.currentPath or "")
        if output_directory:
            self.logic.save_completion_outputs(outputs, diagnostics, output_directory)
        self._run_folder_item = outputs["folder_item"]
        self.run_progress.setValue(100)
        self.diagnostics_text.setPlainText(self.logic.format_diagnostics(diagnostics))
        setWorkflowStatus(
            self.complete_status,
            "complete",
            f"Completion created as '{outputs['model'].GetName()}'. {profile_message}",
        )
        slicer.util.showStatusMessage("Shape Completion finished.", 4000)

    def on_clear_outputs(self):
        if self._run_folder_item:
            hierarchy = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSubjectHierarchyNode")
            try:
                hierarchy.RemoveItem(self._run_folder_item)
            except Exception:
                pass
        self._run_folder_item = None
        self.run_progress.setValue(0)
        self.diagnostics_text.setPlainText("No completion yet.")
        self._validate_complete_inputs()

    def on_cancel_calibration(self):
        self._cancel_calibration = True
        self.calibration_cancel_button.enabled = False
        self._calibration_log("Cancellation requested; the current native registration will finish first.")

    def on_run_calibration(self):
        if not self._ensure_dependencies():
            return
        self._cancel_calibration = False
        self.calibration_run_button.enabled = False
        self.calibration_cancel_button.enabled = True
        self.calibration_progress.setValue(0)
        self.calibration_log.clear()
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self._run_calibration_impl()
        except Exception as error:
            logging.exception("Shape Completion calibration failed")
            self._calibration_log(f"Calibration failed: {error}\n{traceback.format_exc()}")
            slicer.util.errorDisplay(f"Shape Completion calibration failed:\n{error}")
        finally:
            qt.QApplication.restoreOverrideCursor()
            self.calibration_cancel_button.enabled = False
            self._validate_calibration_inputs()

    def _run_calibration_impl(self):
        import rustcpd

        table_node = self.ssm_table_selector.currentNode()
        dense_template = self.template_dense_selector.currentNode()
        sparse_template = self.template_sparse_selector.currentNode()
        if not all((table_node, dense_template)):
            raise ValueError("load an SSM and template dense correspondences before calibration")
        if self.calibration_use_landmarks.checked and sparse_template is None:
            raise ValueError("landmark calibration requires template sparse landmarks")
        mean, modes, eigenvalues = self.logic.ssm_from_table(table_node)
        dense_template_points = self.logic.markups_points_world(dense_template)
        if len(dense_template_points) != len(mean):
            raise ValueError("template dense correspondences do not match the SSM point count")
        ssm_template_compatibility = indexed_correspondence_compatibility(
            mean, dense_template_points
        )
        self._calibration_log(
            "SSM mean↔template indexed RMS: "
            f"{100 * ssm_template_compatibility['rms_fraction']:.3f}% of diagonal."
        )
        template_labels = template_sparse_points = None
        if sparse_template is not None:
            template_labels, template_sparse_points = self.logic.markups_labels_points_world(
                sparse_template
            )
        mesh_dir = str(self.calibration_mesh_directory.currentPath)
        mesh_files = self.logic.list_model_files(mesh_dir)
        if not mesh_files:
            raise ValueError("the calibration directory contains no supported meshes")
        dense_index = self.logic.index_markup_files(
            str(self.calibration_dense_directory.currentPath or "")
        )
        landmark_index = self.logic.index_markup_files(
            str(self.calibration_landmark_directory.currentPath or "")
        )
        mesh_key_list = [safe_stem(path).casefold() for path in mesh_files]
        duplicate_mesh_keys = sorted(
            key for key in set(mesh_key_list) if mesh_key_list.count(key) > 1
        )
        if duplicate_mesh_keys:
            raise ValueError(
                "Complete mesh filenames are ambiguous after removing extensions: "
                + ", ".join(duplicate_mesh_keys[:5])
            )
        mesh_keys = set(mesh_key_list)
        dense_truth_requested = bool(str(self.calibration_dense_directory.currentPath or ""))
        if dense_truth_requested:
            missing_dense = sorted(mesh_keys.difference(dense_index))
            if missing_dense:
                preview = ", ".join(missing_dense[:5])
                raise ValueError(
                    f"Paired dense truth was requested, but {len(missing_dense)} complete meshes have no matching markup (first: {preview}). "
                    "Use a complete paired directory or clear the dense-truth field for surface-distance calibration."
                )
        if bool(self.calibration_use_landmarks.checked):
            missing_landmarks = sorted(mesh_keys.difference(landmark_index))
            if missing_landmarks:
                preview = ", ".join(missing_landmarks[:5])
                raise ValueError(
                    f"Landmark-assisted calibration was requested, but {len(missing_landmarks)} complete meshes have no matching landmark file (first: {preview})."
                )
        coverages = self._parse_coverages()
        replicates = int(self.calibration_replicates.value)
        requested_landmarks = bool(self.calibration_use_landmarks.checked)
        total = len(mesh_files) * len(coverages) * replicates
        complete = 0
        skipped_fragments = 0
        failed_fragments = 0
        accumulators = {}
        seed_sequence = np.random.SeedSequence(int(self.random_seed.value))
        child_sequences = iter(seed_sequence.spawn(max(total, 1)))
        self._calibration_log(
            f"Starting {total} generated fragments from {len(mesh_files)} complete meshes."
        )
        if self.calibration_training_set.checked:
            self._calibration_log(
                "Scope warning: these meshes are declared SSM training data; final calibration will be labelled internal and potentially optimistic."
            )
        for mesh_path in mesh_files:
            if self._cancel_calibration:
                break
            key = safe_stem(mesh_path).casefold()
            mesh_node = None
            dense_truth_node = None
            landmark_node = None
            try:
                mesh_node = slicer.util.loadModel(mesh_path)
                complete_surface = self.logic.model_points_world(mesh_node)
                truth_type = "surface_distance"
                dense_truth = None
                dense_truth_path = dense_index.get(key) if dense_truth_requested else None
                if dense_truth_path:
                    dense_truth_node = slicer.util.loadMarkups(dense_truth_path)
                    dense_truth = self.logic.markups_points_world(dense_truth_node)
                    if len(dense_truth) != len(mean):
                        raise ValueError(
                            f"{Path(dense_truth_path).name} has {len(dense_truth)} points; expected {len(mean)}"
                        )
                    compatibility = coordinate_compatibility(
                        complete_surface,
                        dense_truth,
                        f"dense truth for {Path(mesh_path).name}",
                    )
                    self._calibration_log(
                        f"{Path(mesh_path).name}: dense-truth↔surface median distance "
                        f"{compatibility['median_fraction'] * 100:.3f}% of diagonal."
                    )
                    truth_type = "exact_dense_correspondence"
                target_labels = target_sparse_points = None
                landmark_path = landmark_index.get(key)
                if requested_landmarks:
                    if not landmark_path:
                        raise ValueError(f"No paired target landmarks for {Path(mesh_path).name}")
                    landmark_node = slicer.util.loadMarkups(landmark_path)
                    target_labels, target_sparse_points = self.logic.markups_labels_points_world(
                        landmark_node
                    )
                    coordinate_compatibility(
                        complete_surface,
                        target_sparse_points,
                        f"target landmarks for {Path(mesh_path).name}",
                        max_median_fraction=0.15,
                        max_p95_fraction=0.35,
                    )
                for coverage in coverages:
                    for replicate in range(replicates):
                        if self._cancel_calibration:
                            break
                        rng = np.random.default_rng(next(child_sequences))
                        if self.calibration_random_pose.checked:
                            posed_surface, rotation, translation = apply_random_pose(
                                complete_surface, rng
                            )
                            posed_truth = (
                                None
                                if dense_truth is None
                                else dense_truth @ rotation + translation
                            )
                            posed_sparse = (
                                None
                                if target_sparse_points is None
                                else target_sparse_points @ rotation + translation
                            )
                        else:
                            posed_surface = np.asarray(complete_surface, dtype=float)
                            posed_truth = (
                                None if dense_truth is None else np.asarray(dense_truth, dtype=float)
                            )
                            posed_sparse = (
                                None
                                if target_sparse_points is None
                                else np.asarray(target_sparse_points, dtype=float)
                            )
                        # Fragment geometry always comes from the complete mesh. Paired
                        # dense truth is reserved for evaluating homologous completion
                        # error, so exact-truth calibration is not given an artificially
                        # easy correspondence-only target cloud.
                        mask = contiguous_plane_mask(posed_surface, coverage, rng)
                        fragment = posed_surface[mask.indices]
                        landmarks = None
                        actual_landmark_mode = False
                        if requested_landmarks:
                            visible = mask.contains(posed_sparse)
                            if int(np.sum(visible)) < 3:
                                self._calibration_log(
                                    f"Skip {Path(mesh_path).name} coverage={coverage:.2f} rep={replicate + 1}: fewer than three visible landmarks."
                                )
                                skipped_fragments += 1
                                complete += 1
                                self._update_calibration_progress(complete, total)
                                continue
                            try:
                                landmarks = match_landmarks(
                                    template_labels,
                                    template_sparse_points,
                                    [target_labels[i] for i in np.flatnonzero(visible)],
                                    posed_sparse[visible],
                                    dense_template_points,
                                    allow_order_fallback=bool(self.order_fallback_check.checked),
                                )
                                actual_landmark_mode = True
                            except Exception as error:
                                self._calibration_log(
                                    f"Skip {Path(mesh_path).name} coverage={coverage:.2f} rep={replicate + 1}: {error}"
                                )
                                skipped_fragments += 1
                                complete += 1
                                self._update_calibration_progress(complete, total)
                                continue
                        try:
                            settings = self._read_settings(
                                coverage=coverage,
                                use_landmarks=actual_landmark_mode,
                                samples=0,
                            )
                            result = run_completion(
                                fragment,
                                mean,
                                modes,
                                eigenvalues,
                                settings,
                                rustcpd_module=rustcpd,
                                landmarks=landmarks,
                            )
                            if truth_type == "exact_dense_correspondence":
                                errors_all = np.linalg.norm(
                                    result.completed_points - posed_truth, axis=1
                                )
                                missing_mask = (
                                    np.ones(len(result.completed_points), dtype=bool)
                                    if np.isclose(float(coverage), 1.0)
                                    else ~mask.contains(posed_truth)
                                )
                            else:
                                nearest_truth, errors_all = nearest_indices(
                                    posed_surface, result.completed_points
                                )
                                missing_mask = (
                                    np.ones(len(result.completed_points), dtype=bool)
                                    if np.isclose(float(coverage), 1.0)
                                    else ~mask.contains(posed_surface[nearest_truth])
                                )
                            if not np.any(missing_mask):
                                # Extremely distorted failures can place every completed
                                # point on the retained side of the known calibration cut.
                                # Fall back to the declared-coverage proximity mask so the
                                # failed fit remains represented rather than disappearing.
                                region, _ = classify_completion_region(
                                    result.completed_points, fragment, coverage
                                )
                                missing_mask = region == 1
                            if not np.any(missing_mask):
                                missing_mask[:] = True
                            errors = errors_all[missing_mask]
                            variances = result.total_variance[missing_mask]
                            missing_rms = float(np.sqrt(np.mean(errors * errors)))
                            full_diagonal = max(bbox_diagonal(posed_surface), 1e-12)
                            threshold = (
                                float(self.calibration_success_threshold.value) / 100.0
                            ) * full_diagonal
                            successful = missing_rms <= threshold
                            accumulator_key = (
                                round(float(coverage), 6),
                                bool(actual_landmark_mode),
                                truth_type,
                            )
                            accumulator = accumulators.setdefault(
                                accumulator_key,
                                CalibrationAccumulator(
                                    coverage=float(coverage),
                                    landmark_mode=bool(actual_landmark_mode),
                                    truth_type=truth_type,
                                ),
                            )
                            fit_id = f"{key}:coverage={coverage:.6f}:rep={replicate}"
                            accumulator.add(
                                fit_id=fit_id,
                                group=key,
                                errors=errors,
                                variances=variances,
                                normalized_sigma2=result.normalized_sigma2,
                                successful=successful,
                                missing_rms=missing_rms,
                            )
                            complete += 1
                            self._update_calibration_progress(complete, total)
                            self._calibration_log(
                                f"[{complete}/{total}] {Path(mesh_path).name} coverage={coverage:.2f} "
                                f"landmarks={'yes' if actual_landmark_mode else 'no'} "
                                f"missing RMS={100 * missing_rms / full_diagonal:.2f}% diag "
                                f"fit={'pass' if successful else 'fail'}"
                            )
                        except Exception as error:
                            failed_fragments += 1
                            complete += 1
                            self._update_calibration_progress(complete, total)
                            self._calibration_log(
                                f"[{complete}/{total}] Failed {Path(mesh_path).name} "
                                f"coverage={coverage:.2f} rep={replicate + 1}: "
                                f"{type(error).__name__}: {error}"
                            )
                            logging.exception(
                                "Calibration fragment failed for %s at coverage %.3f",
                                mesh_path,
                                coverage,
                            )
                            continue
                    if self._cancel_calibration:
                        break
            finally:
                for node in (landmark_node, dense_truth_node, mesh_node):
                    if node is not None:
                        try:
                            slicer.mrmlScene.RemoveNode(node)
                        except Exception:
                            pass
        if self._cancel_calibration:
            self._calibration_log("Calibration cancelled; no partial profile was written.")
            setWorkflowStatus(self.calibration_status, "optional", "Calibration was cancelled.")
            return
        if not accumulators:
            raise RuntimeError("no calibration fits completed successfully")
        entries = []
        for key in sorted(accumulators):
            entries.append(
                finalize_calibration_entry(
                    accumulators[key],
                    alpha=float(self.calibration_alpha.value),
                    success_threshold_fraction=float(
                        self.calibration_success_threshold.value
                    )
                    / 100.0,
                )
            )
        fingerprint = model_fingerprint(mean, modes, eigenvalues)
        settings_snapshot = self._settings_snapshot(self._read_settings(samples=0))
        profile = build_calibration_profile(
            fingerprint=fingerprint,
            entries=entries,
            generated_from_ssm_training_set=bool(self.calibration_training_set.checked),
            source_description=(
                f"complete_meshes={mesh_dir}; dense_truth={self.calibration_dense_directory.currentPath or 'none'}; "
                f"target_landmarks={self.calibration_landmark_directory.currentPath or 'none'}"
            ),
            settings_snapshot=settings_snapshot,
        )
        profile["generation_summary"] = {
            "requested_fragment_count": int(total),
            "calibrated_fit_count": int(sum(len(value.fit_ids) for value in accumulators.values())),
            "skipped_for_landmarks_or_matching": int(skipped_fragments),
            "failed_registration_or_scoring": int(failed_fragments),
        }
        output_path = str(self.calibration_output_edit.text).strip()
        save_calibration_profile(profile, output_path)
        self.profile_path.currentPath = output_path
        self._profile = profile
        self.calibration_progress.setValue(100)
        self._calibration_log(
            f"Saved {len(entries)} calibration entries to {output_path}. "
            f"Fits={profile['generation_summary']['calibrated_fit_count']}; "
            f"skipped={skipped_fragments}; failed={failed_fragments}."
        )
        setWorkflowStatus(
            self.calibration_status,
            "complete",
            f"Calibration profile saved. {profile['calibration_scope_warning']}",
        )

    def _update_calibration_progress(self, complete, total):
        self.calibration_progress.setValue(int(round(100.0 * complete / max(total, 1))))
        slicer.app.processEvents()

    def _calibration_log(self, message):
        self.calibration_log.appendPlainText(str(message))
        now = time.monotonic()
        if now - getattr(self, "_last_calibration_pump", 0.0) >= 0.1:
            slicer.app.processEvents()
            self._last_calibration_pump = now


class MorphoWeaveShapeCompletionLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        # Keep one large transfer operator alive across repeated completions of
        # the same template. Replacing the key releases the previous CSR matrix,
        # bounding cache growth even for million-vertex surfaces.
        self._transfer_cache_key = None
        self._transfer_cache_value = None

    def table_to_array(self, table_node):
        table = table_node.GetTable()
        rows = table.GetNumberOfRows()
        columns = table.GetNumberOfColumns()
        array = np.zeros((rows, columns), dtype=np.float64)
        for column in range(columns):
            name = table.GetColumn(column).GetName()
            array[:, column] = slicer.util.arrayFromTableColumn(table_node, name)
        return array

    def ssm_from_table(self, table_node):
        if table_node is None:
            raise ValueError("SSM table is required")
        array = self.table_to_array(table_node)
        if array.shape[0] % 3 != 0 or array.shape[1] < 2:
            raise ValueError("SSM table must contain a flattened mean and at least one mode")
        mean = array[:, 0].reshape(-1, 3)
        modes = array[:, 1:].reshape(len(mean), 3, array.shape[1] - 1)
        try:
            eigenvalues = np.asarray(
                json.loads(table_node.GetAttribute("ssm_eigenvalues")), dtype=np.float64
            )
        except Exception as error:
            raise ValueError(f"SSM table has invalid eigenvalues: {error}") from error
        return mean, modes, eigenvalues

    def model_polydata_world(self, model_node):
        if model_node is None or model_node.GetPolyData() is None:
            raise ValueError("model node has no surface")
        output = vtk.vtkPolyData()
        output.DeepCopy(model_node.GetPolyData())
        parent = model_node.GetParentTransformNode()
        if parent is None:
            return output
        transform = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(parent, None, transform)
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetTransform(transform)
        transform_filter.SetInputData(output)
        transform_filter.Update()
        world = vtk.vtkPolyData()
        world.DeepCopy(transform_filter.GetOutput())
        return world

    def recompute_polydata_normals(self, polydata):
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(polydata)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.SplittingOff()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.Update()
        output = vtk.vtkPolyData()
        output.DeepCopy(normals.GetOutput())
        return output

    def model_points_world(self, model_node):
        polydata = self.model_polydata_world(model_node)
        if polydata.GetPoints() is None or polydata.GetNumberOfPoints() == 0:
            raise ValueError("model contains no points")
        return vtk_np.vtk_to_numpy(polydata.GetPoints().GetData()).astype(
            np.float64, copy=True
        )

    def markups_labels_points_world(self, node):
        if node is None:
            raise ValueError("markups node is required")
        labels = []
        points = []
        for index in range(node.GetNumberOfControlPoints()):
            position = [0.0, 0.0, 0.0]
            node.GetNthControlPointPositionWorld(index, position)
            points.append(position)
            labels.append(node.GetNthControlPointLabel(index))
        if not points:
            raise ValueError("markups node contains no control points")
        return labels, np.asarray(points, dtype=np.float64)

    def markups_points_world(self, node):
        return self.markups_labels_points_world(node)[1]

    def copy_markups_metadata(self, source_node, target_node):
        if source_node is None or target_node is None:
            return
        count = min(
            int(source_node.GetNumberOfControlPoints()),
            int(target_node.GetNumberOfControlPoints()),
        )
        for index in range(count):
            target_node.SetNthControlPointLabel(
                index, str(source_node.GetNthControlPointLabel(index))
            )
            description_getter = getattr(
                source_node, "GetNthControlPointDescription", None
            )
            description_setter = getattr(
                target_node, "SetNthControlPointDescription", None
            )
            if description_getter is not None and description_setter is not None:
                description_setter(index, str(description_getter(index) or ""))
            for getter_name, setter_name in (
                ("GetNthControlPointSelected", "SetNthControlPointSelected"),
                ("GetNthControlPointVisibility", "SetNthControlPointVisibility"),
            ):
                getter = getattr(source_node, getter_name, None)
                setter = getattr(target_node, setter_name, None)
                if getter is not None and setter is not None:
                    setter(index, bool(getter(index)))

    def _transfer_key(
        self,
        template_model,
        dense_node,
        *,
        neighbors,
        sharpness,
        chunk_size,
        restrict_to_components,
    ):
        template_polydata = template_model.GetPolyData()
        template_transform = template_model.GetParentTransformNode()
        dense_transform = dense_node.GetParentTransformNode()
        return (
            str(template_model.GetID() or template_model.GetName() or "template"),
            int(template_polydata.GetMTime()) if template_polydata is not None else -1,
            str(template_transform.GetID()) if template_transform is not None else "",
            int(template_transform.GetMTime()) if template_transform is not None else -1,
            str(dense_node.GetID() or dense_node.GetName() or "dense"),
            int(dense_node.GetMTime()),
            str(dense_transform.GetID()) if dense_transform is not None else "",
            int(dense_transform.GetMTime()) if dense_transform is not None else -1,
            int(neighbors),
            round(float(sharpness), 8),
            int(chunk_size),
            bool(restrict_to_components),
        )

    def polydata_point_component_ids(self, polydata):
        """Return a connected-component label for every original point ID."""

        point_count = int(polydata.GetNumberOfPoints())
        if point_count == 0:
            raise ValueError("cannot label components of an empty surface")
        if int(polydata.GetNumberOfCells()) == 0:
            # A point cloud has no surface connectivity with which to prevent
            # cross-sheet interpolation. Treat it as one group instead of
            # creating one million one-point groups and an impractical Python loop.
            return np.zeros(point_count, dtype=np.int32)

        original_id_name = "__MorphoWeaveOriginalPointId"
        tagged = vtk.vtkPolyData()
        tagged.ShallowCopy(polydata)
        while tagged.GetPointData().HasArray(original_id_name):
            original_id_name += "_"
        original_ids = vtk_np.numpy_to_vtkIdTypeArray(
            np.arange(point_count, dtype=np.int64), deep=True
        )
        original_ids.SetName(original_id_name)
        tagged.GetPointData().AddArray(original_ids)

        connectivity = vtk.vtkPolyDataConnectivityFilter()
        connectivity.SetInputData(tagged)
        connectivity.SetExtractionModeToAllRegions()
        connectivity.ColorRegionsOn()
        connectivity.Update()
        connected = connectivity.GetOutput()
        output_ids_array = connected.GetPointData().GetArray(original_id_name)
        region_array = connected.GetPointData().GetArray("RegionId")
        if output_ids_array is None or region_array is None:
            raise RuntimeError("VTK connectivity filter did not provide point-region labels")
        output_ids = vtk_np.vtk_to_numpy(output_ids_array).astype(np.int64, copy=False)
        output_regions = vtk_np.vtk_to_numpy(region_array).astype(np.int32, copy=False)
        if len(output_regions) < len(output_ids):
            raise RuntimeError("VTK connectivity output has inconsistent point arrays")
        # Some VTK builds retain RegionId tuples for isolated input points even
        # though those points are omitted from the connectivity output. The
        # tuples corresponding to emitted output points come first.
        output_regions = output_regions[: len(output_ids)]

        labels = np.full(point_count, -1, dtype=np.int32)
        valid = (output_ids >= 0) & (output_ids < point_count)
        labels[output_ids[valid]] = output_regions[valid]
        missing = np.flatnonzero(labels < 0)
        if len(missing):
            labelled = np.flatnonzero(labels >= 0)
            if len(labelled):
                mesh_points = vtk_np.vtk_to_numpy(polydata.GetPoints().GetData()).astype(
                    np.float64, copy=False
                )
                nearest_labelled, _ = nearest_indices(
                    mesh_points[labelled], mesh_points[missing]
                )
                labels[missing] = labels[labelled[nearest_labelled]]
            else:
                labels[missing] = 0
        return labels

    def _template_vertex_indices_from_attribute(
        self, dense_node, template_vertices, dense_points
    ):
        """Read and verify persisted dense-control vertex IDs when available."""

        raw = dense_node.GetAttribute("MorphoWeave.TemplateVertexIndices")
        if not raw:
            return None
        try:
            indices = np.asarray(json.loads(raw), dtype=np.int64).reshape(-1)
        except Exception as error:
            logging.warning(
                "Ignoring invalid MorphoWeave.TemplateVertexIndices JSON on '%s': %s",
                dense_node.GetName(),
                error,
            )
            return None
        if len(indices) != len(dense_points):
            logging.warning(
                "Ignoring MorphoWeave.TemplateVertexIndices on '%s': expected %d entries, found %d.",
                dense_node.GetName(),
                len(dense_points),
                len(indices),
            )
            return None
        if np.any(indices < -1) or np.any(indices >= len(template_vertices)):
            logging.warning(
                "Ignoring MorphoWeave.TemplateVertexIndices on '%s': an index is outside the template mesh.",
                dense_node.GetName(),
            )
            return None
        valid_mask = indices >= 0
        valid_indices = indices[valid_mask]
        if len(valid_indices) != len(np.unique(valid_indices)):
            logging.warning(
                "Ignoring MorphoWeave.TemplateVertexIndices on '%s': non-negative indices are not unique.",
                dense_node.GetName(),
            )
            return None
        tolerance = max(bbox_diagonal(template_vertices) * 1e-7, 1e-9)
        nearest_vertices = indices.copy()
        distances = np.empty(len(indices), dtype=np.float64)
        if np.any(valid_mask):
            distances[valid_mask] = np.linalg.norm(
                np.asarray(template_vertices, dtype=np.float64)[valid_indices]
                - np.asarray(dense_points, dtype=np.float64)[valid_mask],
                axis=1,
            )
            maximum = float(np.max(distances[valid_mask], initial=0.0))
            if np.any(~np.isfinite(distances[valid_mask])) or maximum > tolerance:
                logging.warning(
                    "Ignoring MorphoWeave.TemplateVertexIndices on '%s': coordinates disagree with the template (max %.6g; tolerance %.6g).",
                    dense_node.GetName(),
                    maximum,
                    tolerance,
                )
                return None
        missing = ~valid_mask
        if np.any(missing):
            nearest_missing, missing_distances = nearest_indices(
                template_vertices, np.asarray(dense_points, dtype=np.float64)[missing]
            )
            nearest_vertices[missing] = nearest_missing
            distances[missing] = missing_distances
        return indices.copy(), nearest_vertices, distances, float(tolerance)

    def _get_or_build_transfer_operator(
        self,
        *,
        template_model,
        template_polydata,
        template_vertices,
        dense_node,
        dense_points,
        neighbors,
        sharpness,
        chunk_size,
        restrict_to_components,
    ):
        key = self._transfer_key(
            template_model,
            dense_node,
            neighbors=neighbors,
            sharpness=sharpness,
            chunk_size=chunk_size,
            restrict_to_components=restrict_to_components,
        )
        cached = self._transfer_cache_value
        if (
            self._transfer_cache_key == key
            and cached is not None
            and cached["operator"].query_count == len(template_vertices)
            and cached["operator"].source_count == len(dense_points)
        ):
            return cached, True

        persisted_indices = self._template_vertex_indices_from_attribute(
            dense_node, template_vertices, dense_points
        )
        if persisted_indices is not None:
            exact_vertices, nearest_vertices, distances, tolerance = persisted_indices
            vertex_index_source = "markup_attribute"
        else:
            exact_vertices, nearest_vertices, distances, tolerance = infer_control_vertex_indices(
                template_vertices,
                dense_points,
                require_unique=True,
            )
            vertex_index_source = "coordinate_inference"
        template_components = None
        source_components = None
        component_error = None
        if restrict_to_components:
            try:
                template_components = self.polydata_point_component_ids(template_polydata)
                source_components = template_components[nearest_vertices]
            except Exception as error:
                component_error = str(error)
                logging.warning(
                    "Connected-component interpolation restriction could not be applied: %s",
                    error,
                )

        operator = build_local_transfer_operator(
            template_vertices,
            dense_points,
            neighbors=int(neighbors),
            sharpness=float(sharpness),
            chunk_size=int(chunk_size),
            exact_source_vertex_indices=exact_vertices,
            query_groups=template_components,
            source_groups=source_components,
            workers=-1,
        )
        value = {
            "operator": operator,
            "exact_vertices": exact_vertices,
            "nearest_vertices": nearest_vertices,
            "control_distances": distances,
            "control_tolerance": float(tolerance),
            "template_components": template_components,
            "source_components": source_components,
            "component_error": component_error,
            "vertex_index_source": vertex_index_source,
        }
        self._transfer_cache_key = key
        self._transfer_cache_value = value
        return value, False

    def polydata_with_points(self, template_polydata, points, *, recompute_normals=False):
        points = np.asarray(points, dtype=np.float64)
        if points.shape != (template_polydata.GetNumberOfPoints(), 3):
            raise ValueError("replacement surface points do not match the template vertex count")
        output = vtk.vtkPolyData()
        output.ShallowCopy(template_polydata)
        # Template point arrays, especially normals, are invalid after deformation.
        output.GetPointData().Initialize()
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(vtk_np.numpy_to_vtk(points.astype(np.float32), deep=True))
        output.SetPoints(vtk_points)
        output.GetPoints().Modified()
        output.Modified()
        return self.recompute_polydata_normals(output) if recompute_normals else output

    def create_completion_outputs(
        self,
        *,
        template_model,
        template_dense_node,
        template_sparse_node,
        target_node,
        result,
        diagnostics,
        calibrated_pointwise_radius_values,
        calibrated_simultaneous_radius_values,
        interpolation_neighbors,
        interpolation_sharpness,
        interpolation_chunk_size,
        restrict_to_components,
        full_resolution_samples,
    ):
        template_polydata = self.model_polydata_world(template_model)
        if template_polydata.GetPoints() is None or template_polydata.GetNumberOfPoints() == 0:
            raise ValueError("template model contains no surface vertices")
        template_vertices = vtk_np.vtk_to_numpy(
            template_polydata.GetPoints().GetData()
        ).astype(np.float64, copy=True)
        dense_points = self.markups_points_world(template_dense_node)
        if len(dense_points) != len(result.completed_points):
            raise ValueError(
                "completion point count does not match template dense correspondences"
            )

        transfer_bundle, transfer_cache_hit = self._get_or_build_transfer_operator(
            template_model=template_model,
            template_polydata=template_polydata,
            template_vertices=template_vertices,
            dense_node=template_dense_node,
            dense_points=dense_points,
            neighbors=int(interpolation_neighbors),
            sharpness=float(interpolation_sharpness),
            chunk_size=int(interpolation_chunk_size),
            restrict_to_components=bool(restrict_to_components),
        )
        transfer = transfer_bundle["operator"]

        # Reproduce the fitted global pose analytically. The sparse operator sees
        # only the nonrigid residual, avoiding artificial bending under rotations
        # and scale changes when local weights do not reproduce affine coordinates.
        posed_vertices = apply_similarity(
            template_vertices,
            result.world_scale,
            result.world_rotation,
            result.world_translation,
        )
        posed_dense = apply_similarity(
            dense_points,
            result.world_scale,
            result.world_rotation,
            result.world_translation,
        )
        dense_residual = np.asarray(result.completed_points, dtype=np.float64) - posed_dense

        target_points = self.model_points_world(target_node)
        region_dense, distance_dense = classify_completion_region(
            result.completed_points, target_points, diagnostics["coverage"]
        )
        epistemic_std_dense = np.sqrt(
            np.maximum(np.asarray(result.epistemic_variance, dtype=np.float64), 0.0)
        )
        total_std_dense = np.sqrt(
            np.maximum(np.asarray(result.total_variance, dtype=np.float64), 0.0)
        )
        scalar_dense_fields = [
            ("CompletionEpistemicStd", epistemic_std_dense),
            ("CompletionTotalStd", total_std_dense),
            ("CompletionDistanceToFragment", distance_dense),
        ]
        if calibrated_pointwise_radius_values is not None:
            scalar_dense_fields.append(
                (
                    "CompletionCalibratedPointwiseRadius",
                    np.asarray(calibrated_pointwise_radius_values, dtype=np.float64),
                )
            )
        if calibrated_simultaneous_radius_values is not None:
            scalar_dense_fields.append(
                (
                    "CompletionCalibratedSimultaneousRadius",
                    np.asarray(calibrated_simultaneous_radius_values, dtype=np.float64),
                )
            )
        for name, values in scalar_dense_fields:
            if np.asarray(values).shape != (len(dense_points),):
                raise ValueError(f"{name} does not match the dense correspondence count")

        # Geometry and every scalar display field share one sparse multiplication.
        dense_fields = np.column_stack(
            [dense_residual]
            + [np.asarray(values, dtype=np.float64) for _, values in scalar_dense_fields]
        ).astype(np.float32, copy=False)
        apply_started = time.perf_counter()
        full_fields = transfer.apply(dense_fields)
        transfer_apply_seconds = float(time.perf_counter() - apply_started)
        warped_vertices = posed_vertices + np.asarray(full_fields[:, :3], dtype=np.float64)
        scalar_arrays = {
            name: np.asarray(full_fields[:, 3 + index], dtype=np.float32)
            for index, (name, _) in enumerate(scalar_dense_fields)
        }

        output_polydata = self.polydata_with_points(
            template_polydata, warped_vertices, recompute_normals=True
        )
        point_data = output_polydata.GetPointData()
        for name, values in scalar_arrays.items():
            array = vtk_np.numpy_to_vtk(values, deep=True)
            array.SetName(name)
            point_data.AddArray(array)
        region_array = vtk_np.numpy_to_vtk(
            region_dense[transfer.nearest_source_indices].astype(np.int32), deep=True
        )
        region_array.SetName("CompletionRegion")
        point_data.AddArray(region_array)
        point_data.SetActiveScalars("CompletionEpistemicStd")

        exact_vertices = transfer_bundle["exact_vertices"]
        control_distances = transfer_bundle["control_distances"]
        exact_anchor_count = int(np.count_nonzero(exact_vertices >= 0))
        if result.samples:
            latent_sample_output = (
                "full-resolution surfaces"
                if full_resolution_samples
                else "dense SSM point clouds"
            )
            sample_representation = (
                "full_resolution_surface"
                if full_resolution_samples
                else "dense_latent_point_cloud"
            )
        else:
            latent_sample_output = "none"
            sample_representation = "none"

        diagnostics.update(
            {
                "full_resolution_vertex_count": int(len(template_vertices)),
                "transfer_method": "exact similarity pose plus cached sparse local residual",
                "full_resolution_transfer_method": "exact_similarity_plus_cached_sparse_local_residual",
                "transfer_operator_cache_hit": bool(transfer_cache_hit),
                "transfer_operator_neighbors": int(transfer.neighbors),
                "transfer_operator_sharpness": float(transfer.sharpness),
                "transfer_operator_chunk_size": int(transfer.chunk_size),
                "transfer_operator_nonzeros": int(transfer.matrix.nnz),
                "transfer_operator_build_seconds": float(transfer.build_seconds),
                "transfer_operator_apply_seconds": transfer_apply_seconds,
                "transfer_operator_memory_mib": float(
                    transfer.memory_bytes / (1024.0 * 1024.0)
                ),
                "transfer_exact_control_anchors": exact_anchor_count,
                "transfer_unmatched_control_count": int(
                    len(dense_points) - exact_anchor_count
                ),
                "transfer_control_to_vertex_rms": float(
                    np.sqrt(np.mean(control_distances * control_distances))
                ),
                "transfer_control_to_vertex_max": float(np.max(control_distances)),
                "transfer_control_vertex_tolerance": float(
                    transfer_bundle["control_tolerance"]
                ),
                "transfer_control_vertex_index_source": transfer_bundle[
                    "vertex_index_source"
                ],
                "transfer_mean_support_radius": float(transfer.mean_support_radius),
                "transfer_max_support_radius": float(transfer.max_support_radius),
                "transfer_component_restriction_requested": bool(
                    restrict_to_components
                ),
                "transfer_component_restriction_applied": bool(
                    transfer_bundle["template_components"] is not None
                ),
                "transfer_component_count": int(transfer.component_count),
                "transfer_component_fallback_vertices": int(
                    transfer.component_fallback_query_count
                ),
                "transfer_component_error": transfer_bundle["component_error"],
                "transfer_dense_residual_rms": float(
                    np.sqrt(np.mean(np.sum(dense_residual * dense_residual, axis=1)))
                ),
                "latent_sample_output": latent_sample_output,
                "posterior_sample_representation": sample_representation,
                "surface_scalar_transfer_scope": "interpolated_for_visualization",
            }
        )
        if exact_anchor_count < len(dense_points):
            logging.warning(
                "%d of %d dense controls were not coincident with a template vertex "
                "within tolerance %.6g; local interpolation remains available.",
                len(dense_points) - exact_anchor_count,
                len(dense_points),
                transfer_bundle["control_tolerance"],
            )

        target_name = target_node.GetName() or "Target"
        folder_item = self.create_output_folder(target_name)
        model_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode", f"{target_name}_CompletedShape"
        )
        model_node.SetAndObservePolyData(output_polydata)
        model_node.SetAttribute(
            "MorphoWeave.ShapeCompletion.TransferMethod",
            "exact_similarity_plus_cached_sparse_local_residual",
        )
        model_node.SetAttribute(
            "MorphoWeave.ShapeCompletion.TransferOperatorMemoryMiB",
            f"{transfer.memory_bytes / (1024.0 * 1024.0):.6g}",
        )
        model_node.CreateDefaultDisplayNodes()
        display = model_node.GetDisplayNode()
        display.SetScalarVisibility(True)
        display.SetActiveScalarName("CompletionEpistemicStd")
        finite_std = scalar_arrays["CompletionEpistemicStd"]
        display.SetScalarRange(float(np.min(finite_std)), float(np.max(finite_std)))
        self.parent_node(model_node, folder_item)

        dense_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode",
            f"{target_name}_CompletedDenseCorrespondences",
        )
        slicer.util.updateMarkupsControlPointsFromArray(
            dense_node, result.completed_points.astype(np.float32)
        )
        dense_node.CreateDefaultDisplayNodes()
        self.copy_markups_metadata(template_dense_node, dense_node)
        dense_node.SetAttribute(
            "MorphoWeave.TemplateVertexIndices",
            json.dumps(exact_vertices.astype(int).tolist(), separators=(",", ":")),
        )
        dense_node.SetLocked(True)
        dense_node.SetFixedNumberOfControlPoints(True)
        dense_node.GetDisplayNode().SetVisibility(False)
        self.parent_node(dense_node, folder_item)

        sparse_node = None
        if template_sparse_node is not None:
            labels, sparse_points = self.markups_labels_points_world(template_sparse_node)
            sparse_query_groups = None
            if transfer_bundle["template_components"] is not None:
                sparse_nearest_vertices, _ = nearest_indices(
                    template_vertices, sparse_points
                )
                sparse_query_groups = transfer_bundle["template_components"][
                    sparse_nearest_vertices
                ]
            sparse_transfer = build_local_transfer_operator(
                sparse_points,
                dense_points,
                neighbors=int(interpolation_neighbors),
                sharpness=float(interpolation_sharpness),
                chunk_size=max(1, min(int(interpolation_chunk_size), len(sparse_points))),
                query_groups=sparse_query_groups,
                source_groups=transfer_bundle["source_components"],
                workers=-1,
            )
            completed_sparse = pose_residual_interpolate(
                sparse_points,
                dense_points,
                result.completed_points,
                operator=sparse_transfer,
                scale=result.world_scale,
                rotation=result.world_rotation,
                translation=result.world_translation,
            )
            sparse_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", f"{target_name}_CompletedLandmarks"
            )
            slicer.util.updateMarkupsControlPointsFromArray(
                sparse_node, completed_sparse.astype(np.float32)
            )
            sparse_node.CreateDefaultDisplayNodes()
            self.copy_markups_metadata(template_sparse_node, sparse_node)
            for index, label in enumerate(labels):
                sparse_node.SetNthControlPointLabel(index, str(label))
            sparse_node.SetLocked(True)
            sparse_node.SetFixedNumberOfControlPoints(True)
            self.parent_node(sparse_node, folder_item)

        dense_cloud = self.point_cloud_model(
            result.completed_points,
            f"{target_name}_CompletionPointCloud",
            epistemic_std_dense,
            "CompletionEpistemicStd",
        )
        dense_cloud.GetDisplayNode().SetVisibility(False)
        self.parent_node(dense_cloud, folder_item)

        sample_nodes = []
        for index, sample in enumerate(result.samples):
            if full_resolution_samples:
                sample_vertices = pose_residual_interpolate(
                    template_vertices,
                    dense_points,
                    sample,
                    operator=transfer,
                    scale=result.world_scale,
                    rotation=result.world_rotation,
                    translation=result.world_translation,
                )
                sample_polydata = self.polydata_with_points(
                    template_polydata, sample_vertices, recompute_normals=False
                )
                sample_node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLModelNode",
                    f"{target_name}_LatentShapeSampleSurface_{index + 1:02d}",
                )
                sample_node.SetAndObservePolyData(sample_polydata)
                sample_node.CreateDefaultDisplayNodes()
                sample_node.GetDisplayNode().SetOpacity(0.20)
                representation = "full_resolution_surface"
            else:
                sample_node = self.point_cloud_model(
                    sample,
                    f"{target_name}_LatentShapeSampleDense_{index + 1:02d}",
                )
                display_node = sample_node.GetDisplayNode()
                if hasattr(display_node, "SetPointSize"):
                    display_node.SetPointSize(2.0)
                display_node.SetOpacity(0.35)
                representation = "dense_latent_point_cloud"
            sample_node.SetAttribute(
                "MorphoWeave.ShapeCompletion.SampleRepresentation", representation
            )
            sample_node.SetAttribute(
                "MorphoWeave.SampleDistribution", "latent coefficient posterior"
            )
            sample_node.GetDisplayNode().SetVisibility(False)
            self.parent_node(sample_node, folder_item)
            sample_nodes.append(sample_node)

        table_node = self.diagnostics_table(
            diagnostics, f"{target_name}_CompletionDiagnostics"
        )
        self.parent_node(table_node, folder_item)
        target_reference = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode", f"{target_name}_FragmentReference"
        )
        target_reference.SetAndObservePolyData(self.model_polydata_world(target_node))
        target_reference.CreateDefaultDisplayNodes()
        target_reference.GetDisplayNode().SetColor(0.2, 0.45, 0.9)
        target_reference.GetDisplayNode().SetOpacity(0.75)
        target_reference.GetDisplayNode().SetVisibility(True)
        self.parent_node(target_reference, folder_item)
        return {
            "folder_item": folder_item,
            "model": model_node,
            "dense": dense_node,
            "sparse": sparse_node,
            "dense_cloud": dense_cloud,
            "samples": sample_nodes,
            "sample_representation": sample_representation,
            "diagnostics_table": table_node,
            "target_reference": target_reference,
            "dense_region": region_dense,
            "dense_distance": distance_dense,
            "dense_epistemic_std": epistemic_std_dense,
            "dense_total_std": total_std_dense,
            "dense_calibrated_pointwise_radius": calibrated_pointwise_radius_values,
            "dense_calibrated_simultaneous_radius": calibrated_simultaneous_radius_values,
            "dense_completed_points": np.asarray(
                result.completed_points, dtype=np.float64
            ),
        }

    def point_cloud_model(self, points, name, values=None, value_name=None):
        points = np.asarray(points, dtype=np.float64)
        polydata = vtk.vtkPolyData()
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(vtk_np.numpy_to_vtk(points.astype(np.float32), deep=True))
        polydata.SetPoints(vtk_points)
        vertices = vtk.vtkCellArray()
        for index in range(len(points)):
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(index)
        polydata.SetVerts(vertices)
        if values is not None:
            array = vtk_np.numpy_to_vtk(
                np.asarray(values, dtype=np.float32), deep=True
            )
            array.SetName(str(value_name))
            polydata.GetPointData().AddArray(array)
            polydata.GetPointData().SetActiveScalars(str(value_name))
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        node.SetAndObservePolyData(polydata)
        node.CreateDefaultDisplayNodes()
        if values is not None:
            node.GetDisplayNode().SetScalarVisibility(True)
            node.GetDisplayNode().SetActiveScalarName(str(value_name))
        return node

    def create_output_folder(self, target_name):
        hierarchy = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSubjectHierarchyNode")
        root_name = "MorphoWeave Shape Completion runs"
        root = hierarchy.GetItemByName(root_name)
        invalid = hierarchy.GetInvalidItemID()
        if root in (None, invalid):
            root = hierarchy.CreateFolderItem(hierarchy.GetSceneItemID(), root_name)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return hierarchy.CreateFolderItem(root, f"Run {timestamp} - {target_name}")

    def parent_node(self, node, folder_item):
        hierarchy = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSubjectHierarchyNode")
        item = hierarchy.GetItemByDataNode(node)
        invalid = hierarchy.GetInvalidItemID()
        if item in (None, invalid):
            hierarchy.CreateItem(folder_item, node)
        else:
            hierarchy.SetItemParent(item, folder_item)

    def diagnostics_table(self, diagnostics, name):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", name)
        metric = vtk.vtkStringArray()
        metric.SetName("Metric")
        value = vtk.vtkStringArray()
        value.SetName("Value")
        for key, item in diagnostics.items():
            metric.InsertNextValue(str(key))
            if isinstance(item, float):
                value.InsertNextValue(f"{item:.8g}")
            elif item is None:
                value.InsertNextValue("not available")
            else:
                value.InsertNextValue(str(item))
        node.AddColumn(metric)
        node.AddColumn(value)
        return node

    def format_diagnostics(self, diagnostics):
        lines = [
            "Shape completion finished",
            "=========================",
            f"Target coverage: {diagnostics['coverage']:.2f}",
            f"Landmark-assisted: {'yes' if diagnostics['landmark_mode'] else 'no'} ({diagnostics['landmark_count']} matched)",
            f"Retained SSM: {diagnostics['retained_modes']} modes ({100 * diagnostics['retained_variance']:.2f}% variance)",
            f"Surface residual σ²: {diagnostics['surface_sigma2']:.6g}",
            f"Scale-normalized surface σ²: {diagnostics['normalized_surface_sigma2']:.6g}",
            f"Conditional epistemic std, median: {diagnostics['conditional_epistemic_std_median']:.6g}",
            f"Conditional total std, median: {diagnostics['conditional_total_std_median']:.6g}",
            f"Surface-only posterior mean RMS from constrained atlas estimate: {diagnostics['posterior_surface_only_mean_rms_from_constrained_atlas']:.6g}",
        ]
        if diagnostics.get("transfer_method") is not None:
            lines.extend(
                [
                    "",
                    "Full-resolution transfer",
                    f"  vertices: {diagnostics['full_resolution_vertex_count']:,}",
                    f"  method: {diagnostics['transfer_method']}",
                    f"  sparse operator: k={diagnostics['transfer_operator_neighbors']}, locality={diagnostics['transfer_operator_sharpness']:.3g}, memory={diagnostics['transfer_operator_memory_mib']:.2f} MiB",
                    f"  cache: {'reused' if diagnostics['transfer_operator_cache_hit'] else 'built'}; original build time={diagnostics['transfer_operator_build_seconds']:.3f} s",
                    f"  exact dense-control anchors: {diagnostics['transfer_exact_control_anchors']} ({diagnostics.get('transfer_control_vertex_index_source', 'unknown')})",
                    f"  connected-component restriction: {'applied' if diagnostics['transfer_component_restriction_applied'] else 'not applied'}",
                    f"  latent samples: {diagnostics['latent_sample_output']}",
                ]
            )
            if diagnostics.get("transfer_component_fallback_vertices", 0):
                lines.append(
                    "  warning: "
                    f"{diagnostics['transfer_component_fallback_vertices']} vertices belonged to components without dense controls and used the global control set."
                )
            if diagnostics.get("transfer_component_error"):
                lines.append(
                    "  component restriction warning: "
                    + str(diagnostics["transfer_component_error"])
                )
        if diagnostics.get("landmark_sigma") is not None:
            lines.append(
                f"Effective landmark σ: {diagnostics['landmark_sigma']:.6g} "
                f"(annotation σ={diagnostics['landmark_nominal_sigma']:.6g}; "
                f"dense-mapping RMS={diagnostics['landmark_dense_mapping_rms']:.6g})"
            )
        if diagnostics.get("landmark_rms") is not None:
            lines.append(f"Landmark RMS: {diagnostics['landmark_rms']:.6g}")
            lines.append(
                f"Landmark reduced χ²: {diagnostics['landmark_reduced_chi_square']:.4g} (≈1 is the stated localization noise floor)"
            )
        if diagnostics.get("calibration_truth_type") is not None:
            if diagnostics.get("calibration_pointwise_radius_available"):
                lines.append(
                    "Calibrated pointwise radius: marginal Euclidean-error coverage on inferred loci; "
                    "not a simultaneous whole-shape guarantee."
                )
            else:
                lines.append(
                    "Calibrated pointwise radius unavailable: the matching profile entry had too few calibration scores at this alpha."
                )
            if diagnostics.get("calibration_simultaneous_radius_available"):
                lines.append(
                    "Calibrated simultaneous radius: conservative meshwise envelope under "
                    "the profile's fragment-generation protocol."
                )
        probability = diagnostics.get("calibrated_fit_success_probability")
        if probability is not None:
            lines.append(f"Calibrated P(fit within threshold): {probability:.3f}")
        lines.extend(
            [
                "",
                "Pose ambiguity diagnostics (within this run only)",
                f"  score margin: {diagnostics['pose_score_margin_within_run_only']:.6g}",
                f"  entropy: {diagnostics['pose_posterior_entropy_within_run_only']:.6g}",
                f"  effective hypotheses: {diagnostics['pose_effective_hypotheses_within_run_only']:.3f}",
                "",
                f"Calibration: {diagnostics['calibration_message']}",
                f"Uncertainty scope: {diagnostics['uncertainty_scope']}",
                "CompletionRegion is a proximity-based display mask (0=data-proximal, 1=inferred), not observed ground truth.",
            ]
        )
        return "\n".join(lines)

    def save_node_checked(self, node, path):
        path = os.fspath(path)
        if node is None:
            raise ValueError(f"cannot save an empty node to {path}")
        if not slicer.util.saveNode(node, path):
            raise IOError(f"3D Slicer could not save '{node.GetName()}' to {path}")
        return path

    def save_completion_outputs(self, outputs, diagnostics, output_directory):
        os.makedirs(output_directory, exist_ok=True)
        base = safe_stem(outputs["model"].GetName())
        self.save_node_checked(
            outputs["model"], os.path.join(output_directory, base + ".vtp")
        )
        self.save_node_checked(
            outputs["dense"], os.path.join(output_directory, base + "_dense.mrk.json")
        )
        if outputs["sparse"] is not None:
            self.save_node_checked(
                outputs["sparse"], os.path.join(output_directory, base + "_landmarks.mrk.json")
            )
        for index, sample_node in enumerate(outputs.get("samples", []), start=1):
            representation = sample_node.GetAttribute(
                "MorphoWeave.ShapeCompletion.SampleRepresentation"
            )
            sample_kind = (
                "latent_surface"
                if representation == "full_resolution_surface"
                else "latent_dense"
            )
            self.save_node_checked(
                sample_node,
                os.path.join(output_directory, f"{base}_{sample_kind}_{index:02d}.vtp"),
            )
        with open(
            os.path.join(output_directory, base + "_diagnostics.json"),
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(diagnostics, stream, indent=2, sort_keys=True)
        dense_points = np.asarray(outputs["dense_completed_points"], dtype=np.float64)
        columns = [
            np.arange(len(outputs["dense_region"]), dtype=np.int64),
            dense_points[:, 0],
            dense_points[:, 1],
            dense_points[:, 2],
            outputs["dense_region"],
            outputs["dense_distance"],
            outputs["dense_epistemic_std"],
            outputs["dense_total_std"],
        ]
        header = [
            "dense_index",
            "x",
            "y",
            "z",
            "completion_region",
            "distance_to_fragment",
            "epistemic_std",
            "total_std",
        ]
        if outputs["dense_calibrated_pointwise_radius"] is not None:
            columns.append(outputs["dense_calibrated_pointwise_radius"])
            header.append("calibrated_pointwise_radius")
        if outputs["dense_calibrated_simultaneous_radius"] is not None:
            columns.append(outputs["dense_calibrated_simultaneous_radius"])
            header.append("calibrated_simultaneous_radius")
        np.savetxt(
            os.path.join(output_directory, base + "_uncertainty.csv"),
            np.column_stack(columns),
            delimiter=",",
            header=",".join(header),
            comments="",
        )

    def list_model_files(self, directory):
        if not directory or not os.path.isdir(directory):
            return []
        return [
            os.path.join(directory, name)
            for name in sorted(os.listdir(directory))
            if name.lower().endswith(SUPPORTED_MODEL_EXTENSIONS)
        ]

    def index_markup_files(self, directory):
        if not directory or not os.path.isdir(directory):
            return {}
        result = {}
        for name in sorted(os.listdir(directory)):
            if name.lower().endswith(SUPPORTED_MARKUP_EXTENSIONS):
                key = safe_stem(name).casefold()
                if key in result:
                    raise ValueError(
                        f"Ambiguous paired markup files share the stem '{key}': "
                        f"{Path(result[key]).name} and {name}"
                    )
                result[key] = os.path.join(directory, name)
        return result


class MorphoWeaveShapeCompletionTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.assertIsNotNone(MorphoWeaveShapeCompletionLogic())
