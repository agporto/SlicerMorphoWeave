"""Discoverable single/batch entry point around the uploaded fragment workflow.

The numerical core is restored byte-for-byte from the user's working archive.
This entry point handles deployment, synchronization, and the Optimize tab.
The numerical fitting implementation and existing files are not reorganized.
"""
import importlib
import inspect
import logging
import sys
import json
import os
from pathlib import Path
import time

import ctk
import numpy as np
import qt
import vtk
import vtk.util.numpy_support as vtk_np

import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from Resources.Python.MorphoWeaveShapeCompletionBase import *  # noqa: F401,F403
from Resources.Python.MorphoWeaveShapeCompletionBase import (
    MorphoWeaveShapeCompletion as _CompletionModule,
    MorphoWeaveShapeCompletionWidget as _SingleCompletionWidget,
    MorphoWeaveShapeCompletionTest as _CompletionTest,
)
from Resources.Python.MorphoWeaveShapeCompletionBatch import ShapeCompletionBatchMixin, _link_controls

RUSTCPD_REQUIREMENT = "rustcpd==4.0.0"


# Optimize is deliberately implemented in this entry-point file. The existing
# fragment fitter, calibration code, batch adapter, and exporters are unchanged.
OPTIMIZATION_PRESET_SCHEMA = "MorphoWeaveShapeCompletionHyperparameters"
OPTIMIZATION_PRESET_VERSION = 1

# Settings with GUI controls. Fields without controls must equal the existing
# core defaults; a preset must never claim to apply a value the GUI cannot use.
_OPTIMIZE_SCALARS = {
    "coverage": "coverage_spin", "variance_keep": "variance_keep",
    "target_point_count": "registration_point_count",
    "landmark_sigma_percent": "landmark_sigma_percent",
    "rotation_count": "pose_rotation_count", "coarse_source_count": "pose_coarse_source",
    "coarse_target_count": "pose_coarse_target", "coarse_rank": "pose_coarse_rank",
    "coarse_iterations": "pose_coarse_iterations",
    "coarse_screen_iterations": "pose_screen_iterations",
    "coarse_survivor_count": "pose_survivors", "refine_count": "pose_refine_count",
    "refine_source_count": "pose_refine_source", "refine_target_count": "pose_refine_target",
    "refine_iterations": "pose_refine_iterations", "pose_lambda": "pose_lambda",
    "pose_outlier_weight": "pose_outlier", "identity_prior_probability": "pose_identity_prior",
    "atlas_lambda": "atlas_lambda", "atlas_outlier_weight": "atlas_outlier",
    "atlas_max_iterations": "atlas_iterations", "atlas_tolerance": "atlas_tolerance",
    "atlas_k": "atlas_k", "posterior_samples": "posterior_samples", "seed": "random_seed",
    "translation_anchor_count": "pose_anchor_count",
    "anchor_completeness_threshold": "pose_anchor_threshold",
    "free_scale_bounds_fraction": "free_scale_bounds", "adaptive_mixing": "pose_adaptive_mixing",
    "initial_sigma2": "pose_initial_sigma2", "merge_tolerance": "pose_merge_tolerance",
}
_OPTIMIZE_CHECKS = {
    "coverage_prescale": "coverage_prescale_check", "use_landmarks": "use_landmarks_check",
    "estimate_discrepancy": "estimate_discrepancy_check", "parallel": "parallel_check",
    "allow_order_fallback": "order_fallback_check",
}
_OPTIMIZE_NULLABLE = {"refine_source_count", "atlas_k", "free_scale_bounds_fraction",
                      "adaptive_mixing", "initial_sigma2"}
_OPTIMIZE_INTERPOLATION = {
    "neighbors": "interpolation_neighbors", "sharpness": "interpolation_sharpness",
    "chunk_size": "interpolation_chunk_size",
}


class _OptimizationCancelled(Exception):
    """Cooperative cancellation; never count this as a failed candidate."""


def _optimize_numbers(text, minimum, maximum, *, integer=False, zero_is_none=False):
    """Parse finite comma-separated values without eval or executable presets."""
    values = []
    for part in str(text).replace(";", ",").split(","):
        if not part.strip():
            continue
        value = float(part)
        if not np.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"Candidate values must be finite and between {minimum} and {maximum}")
        if integer:
            if not value.is_integer():
                raise ValueError("Translation seed counts must be integers")
            value = int(value)
        if zero_is_none and value == 0:
            value = None
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("Each candidate list must contain at least one value")
    return values


def _optimize_candidates(base, regularization, margins, anchors):
    """A small exhaustive grid plus the current settings as an explicit control."""
    from dataclasses import asdict, replace
    from itertools import product
    if len(regularization) * len(margins) * len(anchors) > 200:
        raise ValueError("Search is limited to 200 combinations; shorten the candidate lists")
    candidates, seen = [], set()
    settings_to_try = [("Current settings", base)]
    settings_to_try.extend(
        (f"lambda={lam:g}, margin={margin}, seeds={anchor}",
         replace(base, pose_lambda=lam, atlas_lambda=lam,
                 free_scale_bounds_fraction=margin, translation_anchor_count=anchor))
        for lam, margin, anchor in product(regularization, margins, anchors)
    )
    for name, settings in settings_to_try:
        settings.validate()
        # Samples are not needed for geometry ranking. Saved/applicable settings
        # retain the user's requested sample count for normal completion.
        signature = json.dumps(asdict(replace(settings, posterior_samples=0)), sort_keys=True,
                               allow_nan=False)
        if signature not in seen:
            seen.add(signature)
            candidates.append({"id": f"C{len(candidates):03d}", "name": name,
                               "settings": asdict(settings)})
    return candidates


def _optimize_index(directory, extensions):
    path = Path(directory).expanduser()
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")
    result = {}
    for item in sorted(path.iterdir(), key=lambda p: (p.name.casefold(), p.name)):
        if not item.is_file() or not item.name.lower().endswith(extensions):
            continue
        key = safe_stem(item).casefold()
        if key in result:
            raise ValueError(f"Ambiguous specimen basename: {result[key].name} and {item.name}")
        result[key] = item.resolve()
    if not result:
        raise ValueError(f"No supported files in {path}")
    return result


def _optimize_pairs(complete_directory, fragment_directory="", landmark_directory=""):
    complete = _optimize_index(complete_directory, SUPPORTED_MODEL_EXTENSIONS)
    fragments = _optimize_index(fragment_directory, SUPPORTED_MODEL_EXTENSIONS) if fragment_directory else {}
    landmarks = _optimize_index(landmark_directory, SUPPORTED_MARKUP_EXTENSIONS) if landmark_directory else {}
    keys = sorted(complete.keys() & fragments.keys()) if fragment_directory else sorted(complete)
    if not keys:
        raise ValueError("No partial/complete specimens have matching basenames")
    if landmark_directory:
        missing = [key for key in keys if key not in landmarks]
        if missing:
            raise ValueError("Missing landmark files for: " + ", ".join(missing[:6]))
    pairs = [{"specimen": key, "complete": complete[key], "fragment": fragments.get(key),
              "landmarks": landmarks.get(key)} for key in keys]
    ignored = {"unpaired_fragments": sorted(fragments.keys() - complete.keys()),
               "complete_without_fragment": sorted(complete.keys() - fragments.keys()) if fragment_directory else []}
    return pairs, ignored


def _optimize_triangles(polydata):
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(polydata)
    triangle_filter.PassVertsOff()
    triangle_filter.PassLinesOff()
    triangle_filter.Update()
    output = vtk.vtkPolyData()
    output.DeepCopy(triangle_filter.GetOutput())
    if output.GetNumberOfPoints() < 3 or output.GetNumberOfPolys() < 1:
        raise ValueError("Optimization needs triangle surfaces, not empty meshes or point clouds")
    vertices = vtk_np.vtk_to_numpy(output.GetPoints().GetData()).astype(np.float64, copy=True)
    offsets = vtk_np.vtk_to_numpy(output.GetPolys().GetOffsetsArray())
    connectivity = vtk_np.vtk_to_numpy(output.GetPolys().GetConnectivityArray())
    if not np.all(np.diff(offsets) == 3) or not np.all(np.isfinite(vertices)):
        raise ValueError("Invalid triangulated surface")
    return output, vertices, connectivity.reshape(-1, 3).astype(np.int64, copy=True)


def _optimize_surface_samples(vertices, triangles, count, rng, mask=None):
    tri = vertices[triangles]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if mask is not None:
        areas = areas * np.asarray(mask, dtype=bool)
    total = float(areas.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("No nondegenerate withheld surface is available for scoring")
    chosen = rng.choice(len(triangles), size=int(count), p=areas / total)
    root = np.sqrt(rng.random(int(count)))
    other = rng.random(int(count))
    barycentric = np.column_stack((1 - root, root * (1 - other), root * other))
    return (tri[chosen] * barycentric[:, :, None]).sum(axis=1)


def _optimize_surface_distances(polydata, queries):
    """Unsigned closest-triangle distances; do not require a closed surface."""
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(polydata)
    locator.BuildLocator()
    closest, cell, sub, squared = [0.0] * 3, vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
    distances = np.empty(len(queries), dtype=np.float64)
    for index, point in enumerate(queries):
        locator.FindClosestPoint(point, closest, cell, sub, squared)
        distances[index] = np.sqrt(max(float(squared), 0.0))
    if not np.all(np.isfinite(distances)):
        raise ValueError("Nonfinite surface distances")
    return distances


def _optimize_paired_queries(complete, fragment, count, rng, tolerance_fraction):
    """Find genuinely missing surface independently of fitted predictions.

    Exact shared triangles are recognized for cropped meshes. Other aligned
    surfaces use area-uniform reference samples outside the fragment tolerance.
    Full references are never used to initialize pose, size, or coefficients.
    """
    from scipy.spatial import cKDTree
    truth, vertices, faces = _optimize_triangles(complete)
    partial, partial_vertices, partial_faces = _optimize_triangles(fragment)
    diagonal = bbox_diagonal(vertices)
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("Complete reference has zero or invalid extent")
    probes = _optimize_surface_samples(partial_vertices, partial_faces, 256, rng)
    distances = _optimize_surface_distances(truth, probes)
    tolerance = max(diagonal * float(tolerance_fraction), 1e-9)
    if np.quantile(distances, 0.95) > tolerance:
        raise ValueError("Partial and complete surfaces must already share coordinates; "
                         "their 95th-percentile separation exceeds the matching tolerance")
    distance, mapping = cKDTree(vertices).query(partial_vertices)
    if float(distance.max()) <= max(diagonal * 1e-7, 1e-9):
        # Structured triples avoid overflow from packing vertex indices into n^3.
        def keys(values):
            values = np.ascontiguousarray(np.sort(values, axis=1), dtype=np.int64)
            return values.view(np.dtype((np.void, 3 * values.dtype.itemsize))).ravel()
        full_keys, part_keys = keys(faces), keys(mapping[partial_faces])
        if np.all(np.isin(part_keys, full_keys)):
            missing = ~np.isin(full_keys, part_keys)
            queries = _optimize_surface_samples(vertices, faces, count, rng, missing)
            return queries, "exact_missing_triangles"
    selected, remaining = [], int(count)
    for _ in range(12):
        samples = _optimize_surface_samples(vertices, faces, max(4 * remaining, 1000), rng)
        samples = samples[_optimize_surface_distances(partial, samples) > tolerance]
        accepted = samples[:remaining]
        selected.append(accepted)
        remaining -= len(accepted)
        if remaining == 0:
            return np.concatenate(selected), "area_samples_outside_fragment_tolerance"
    raise ValueError("Too little withheld surface; check that a partial, not complete, mesh was supplied")


def _optimize_rank(candidates, rows, expected_cases):
    """Equal specimen weights; any failed/missing case disqualifies a candidate."""
    ranked = []
    for candidate in candidates:
        results = [row for row in rows if row["candidate"] == candidate["id"]]
        failed = expected_cases - sum(row["status"] == "success" for row in results)
        entry = dict(candidate, failed_cases=failed, successful_cases=expected_cases - failed,
                     mean_missing_rmse_fraction=None, mean_extent_bias_percent=None)
        if not failed:
            groups = {}
            for row in results:
                groups.setdefault(row["specimen"], []).append(row)
            means = [np.mean([r["missing_rmse_fraction"] for r in group]) for group in groups.values()]
            biases = [np.mean([r["long_axis_extent_bias_percent"] for r in group]) for group in groups.values()]
            entry.update(mean_missing_rmse_fraction=float(np.mean(means)),
                         mean_extent_bias_percent=float(np.mean(biases)),
                         worst_specimen_missing_rmse_fraction=float(np.max(means)))
        ranked.append(entry)
    return sorted(ranked, key=lambda e: (e["failed_cases"],
                  e["mean_missing_rmse_fraction"] if e["mean_missing_rmse_fraction"] is not None else float("inf"),
                  e["id"]))


def _optimize_file_hash(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optimize_save_json(path, value):
    """Write a complete preset atomically; failed writes leave old files intact."""
    import tempfile
    path = Path(path).expanduser()
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_completion_backend(backend):
    """Require the pinned release and every fragment API this core actually uses."""
    required_parameters = {
        "pose_initialize": (
            "rotation_count", "coarse_source_count", "coarse_target_count",
            "coarse_rank", "coarse_iterations", "coarse_screen_iterations",
            "coarse_survivor_count", "coarse_score_mode", "refine_count",
            "refine_source_count", "refine_target_count", "refine_iterations",
            "lambda_regularization", "outlier_weight", "identity_prior_probability",
            "landmark_indices", "landmark_targets", "landmark_sigma",
            "refine_landmark_sigma", "with_scale", "seed", "parallel", "single_precision",
            "translation_anchor_count", "anchor_completeness_threshold",
            "scale_bounds", "adaptive_mixing", "initial_sigma2", "merge_tolerance",
        ),
        "register_atlas": (
            "lambda_regularization", "normalize", "optimize_similarity", "with_scale",
            "initial_coefficients", "initial_rotation", "initial_scale", "initial_translation",
            "landmark_indices", "landmark_targets", "landmark_sigma", "max_iterations",
            "tolerance", "outlier_weight", "k", "parallel", "single_precision",
            "sigma2", "scale_bounds", "adaptive_mixing",
        ),
        "AtlasResult.posterior": (
            "completeness", "prior_temperature", "outlier_weight", "estimate_discrepancy",
        ),
    }
    required_attributes = {
        "PoseInitialization": (
            "coefficients", "rotation", "scale", "translation", "score", "score_margin",
            "posterior_entropy", "effective_hypotheses", "hypotheses_evaluated", "hypotheses_refined",
            "distinct_hypotheses", "winner_support", "translation_anchors_used",
        ),
        "AtlasResult": (
            "points", "coefficients", "rotation", "scale", "translation", "sigma2",
            "iterations", "difference", "landmark_rms", "posterior", "mixing_weights",
        ),
        "ShapePosterior": (
            "coefficient_covariance", "noise_variance", "discrepancy_variance",
            "predict", "predictive_variance", "sample_shapes",
        ),
    }
    missing = []
    for qualified_name, names in required_parameters.items():
        function = backend
        for part in qualified_name.split("."):
            function = getattr(function, part, None)
        if not callable(function):
            missing.append(qualified_name)
            continue
        parameters = inspect.signature(function).parameters
        missing.extend(f"{qualified_name}.{name}" for name in names if name not in parameters)
    for class_name, names in required_attributes.items():
        cls = getattr(backend, class_name, None)
        if cls is None:
            missing.append(class_name)
        else:
            missing.extend(f"{class_name}.{name}" for name in names if not hasattr(cls, name))
    version = getattr(backend, "__version__", None)
    if version != "4.0.0":
        missing.append(f"loaded rustcpd==4.0.0 (found {version!r})")
    if missing:
        raise RuntimeError(
            "Shape Completion requires " + RUSTCPD_REQUIREMENT
            + " with the partial-target completion API. Missing or incompatible: "
            + ", ".join(missing)
            + ". Install the released rustcpd==4.0.0 wheel, then restart Slicer."
        )


# Extension Wizard requires this literal base name; it does not resolve aliases.
class MorphoWeaveShapeCompletion(_CompletionModule, ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.helpText += (
            " The Batch tab uses the same fragment-fitting implementation and "
            "Advanced settings as Complete Shape. Requires rustcpd==4.0.0. "
            "Optimize compares reconstruction settings and saves loadable JSON presets. "
            "See RESTORATION.md for preserved behavior and validation limits."
        )


class MorphoWeaveShapeCompletionWidget(ShapeCompletionBatchMixin, _SingleCompletionWidget):
    def __init__(self, *args, **kwargs):
        self._optimization_result = None
        self._cancel_optimization = False
        super().__init__(*args, **kwargs)

    def setup(self):
        super().setup()
        # Auto-selection deliberately suppresses selector signals during setup.
        self._validate_optimize_inputs()

    def _build_ui(self):
        super()._build_ui()
        self.optimize_tab = qt.QWidget()
        self.tabs.insertTab(2, self.optimize_tab, "Optimize")
        layout = qt.QFormLayout(self.optimize_tab)
        intro = qt.QLabel(
            "Find reconstruction settings using complete reference specimens. Uses the SSM, "
            "coverage and landmark mode shared with Complete Shape; all other settings come "
            "from Advanced. This tunes reconstruction, not uncertainty calibration."
        )
        intro.setWordWrap(True)
        layout.addRow(intro)
        self.optimize_inputs_box = ctk.ctkCollapsibleButton()
        self.optimize_inputs_box.text = "Training specimens and search"
        self.optimize_inputs_box.collapsed = False
        inputs = qt.QFormLayout(self.optimize_inputs_box)
        layout.addRow(self.optimize_inputs_box)
        self.optimize_coverage = self._double(0.05, 1.0, float(self.coverage_spin.value), 0.05, 12)
        self.optimize_coverage.setToolTip(
            "Shared with Complete Shape and Batch. A missing-anatomy search needs coverage "
            "below 1.0; for the proximal-humerus examples use 0.40. Coverage is not "
            "estimated or changed automatically."
        )
        inputs.addRow("Target coverage (shared; below 1.0):", self.optimize_coverage)
        self.optimize_use_landmarks = qt.QCheckBox("Use matched landmarks (shared)")
        self.optimize_use_landmarks.setChecked(bool(self.use_landmarks_check.checked))
        inputs.addRow(self.optimize_use_landmarks)
        self.optimize_complete = ctk.ctkPathLineEdit()
        self.optimize_fragments = ctk.ctkPathLineEdit()
        self.optimize_landmarks = ctk.ctkPathLineEdit()
        for control in (self.optimize_complete, self.optimize_fragments, self.optimize_landmarks):
            control.filters = ctk.ctkPathLineEdit.Dirs
        inputs.addRow("Complete reference meshes:", self.optimize_complete)
        inputs.addRow("Paired fragments (optional):", self.optimize_fragments)
        self.optimize_fragments.setToolTip(
            "Pair by basename in separate folders, in the same coordinate frame. "
            "Leave empty to generate plane-cut fragments from the complete meshes."
        )
        inputs.addRow("Paired landmark files (optional):", self.optimize_landmarks)
        self.optimize_visible_labels = qt.QLineEdit()
        self.optimize_visible_labels.setToolTip(
            "In landmark-assisted paired-fragment mode, explicitly list ONLY labels available "
            "on the fragment (for example LT, GT, SCT, HHC). All other labels are excluded. "
            "Generated cuts instead use only landmarks on the retained side of each cut."
        )
        inputs.addRow("Visible labels for paired fragments:", self.optimize_visible_labels)
        self.optimize_lambdas = qt.QLineEdit("5, 20, 100")
        self.optimize_margins = qt.QLineEdit("0.025, 0.05, 0.10")
        self.optimize_anchors = qt.QLineEdit("6")
        inputs.addRow("Regularization values (both stages):", self.optimize_lambdas)
        inputs.addRow("Fragment scale margins (0 = off):", self.optimize_margins)
        inputs.addRow("Translation seed counts:", self.optimize_anchors)
        self.optimize_replicates = self._spin(1, 10, 1)
        inputs.addRow("Generated cuts per specimen:", self.optimize_replicates)
        self.optimize_samples = self._spin(100, 10000, 3000)
        inputs.addRow("Withheld-surface scoring samples:", self.optimize_samples)
        self.optimize_tolerance = self._double(1e-7, 0.02, 0.001, 0.0001, 7)
        self.optimize_tolerance.setToolTip(
            "Fraction of the complete mesh diagonal for matching independently tessellated "
            "partial/reference surfaces. Exact shared triangles take precedence. "
            "This is a scoring/matching tolerance, not a fitting hyperparameter."
        )
        inputs.addRow("Paired-surface matching tolerance:", self.optimize_tolerance)
        note = qt.QLabel(
            "Every candidate sees the same fragments and scoring samples. The current settings "
            "are also evaluated. Rank: mean withheld-surface RMSE / complete-bone diagonal, "
            "with equal specimen weights. All supplied specimens are used for tuning, not "
            "held-out validation. Keep separate specimens for calibration/testing."
        )
        note.setWordWrap(True)
        inputs.addRow(note)
        self.optimize_run = qt.QPushButton("Run Parameter Search")
        self.optimize_cancel = qt.QPushButton("Cancel after current fit")
        self.optimize_cancel.enabled = False
        self.optimize_progress = qt.QProgressBar()
        self.optimize_progress.setRange(0, 100)
        self.optimize_status = qt.QLabel("")
        self.optimize_status.setWordWrap(True)
        self.optimize_log = qt.QPlainTextEdit()
        self.optimize_log.setReadOnly(True)
        self.optimize_log.setMinimumHeight(160)
        layout.addRow(self.optimize_status)
        layout.addRow(self.optimize_run)
        layout.addRow(self.optimize_progress)
        layout.addRow(self.optimize_cancel)
        layout.addRow(self.optimize_log)
        self.optimize_apply = qt.QPushButton("Apply Best Settings")
        self.optimize_save = qt.QPushButton("Save Best Preset...")
        self.optimize_load = qt.QPushButton("Load and Apply Preset...")
        layout.addRow(self.optimize_apply)
        layout.addRow(self.optimize_save)
        layout.addRow(self.optimize_load)
        self._validate_optimize_inputs()

    def _connect_ui(self):
        super()._connect_ui()
        _link_controls(self.coverage_spin, self.optimize_coverage, "valueChanged",
                       lambda w: float(w.value), lambda w, v: w.setValue(v))
        _link_controls(self.use_landmarks_check, self.optimize_use_landmarks, "toggled",
                       lambda w: bool(w.checked), lambda w, v: w.setChecked(v))
        self.tabs.currentChanged.connect(self._validate_optimize_inputs)
        for control in (self.optimize_complete, self.optimize_fragments, self.optimize_landmarks):
            control.currentPathChanged.connect(self._validate_optimize_inputs)
        for control in (self.optimize_lambdas, self.optimize_margins, self.optimize_anchors,
                        self.optimize_visible_labels):
            control.textChanged.connect(self._validate_optimize_inputs)
        for selector in (self.template_model_selector, self.template_dense_selector,
                         self.template_sparse_selector, self.ssm_table_selector):
            selector.currentNodeChanged.connect(self._validate_optimize_inputs)
        # Invalid Advanced settings can disable the search. Revalidate when
        # those settings are repaired instead of waiting for a directory edit.
        for name in sorted(set(_OPTIMIZE_SCALARS.values())):
            getattr(self, name).valueChanged.connect(self._validate_optimize_inputs)
        for name in sorted(set(_OPTIMIZE_CHECKS.values())):
            getattr(self, name).toggled.connect(self._validate_optimize_inputs)
        self.scale_policy_combo.currentIndexChanged.connect(self._validate_optimize_inputs)
        self.optimize_run.clicked.connect(self.on_run_optimization)
        self.optimize_cancel.clicked.connect(self.on_cancel_optimization)
        self.optimize_apply.clicked.connect(self.on_apply_optimized_settings)
        self.optimize_save.clicked.connect(self.on_save_optimization_preset)
        self.optimize_load.clicked.connect(self.on_load_optimization_preset)

    def _optimization_candidates(self):
        return _optimize_candidates(
            self._read_settings(),
            _optimize_numbers(self.optimize_lambdas.text, 0, 1000),
            _optimize_numbers(self.optimize_margins.text, 0, 0.9, zero_is_none=True),
            _optimize_numbers(self.optimize_anchors.text, 1, 32, integer=True),
        )

    def _optimization_visible_labels(self):
        return [part.strip() for part in str(self.optimize_visible_labels.text).split(",") if part.strip()]

    def _sync_optimization_inputs(self):
        # Preset application and automatic setup may block the source signals.
        # The existing controls remain the single source of fitting settings.
        blockers = [qt.QSignalBlocker(self.optimize_coverage),
                    qt.QSignalBlocker(self.optimize_use_landmarks)]
        try:
            self.optimize_coverage.setValue(float(self.coverage_spin.value))
            self.optimize_use_landmarks.setChecked(bool(self.use_landmarks_check.checked))
        finally:
            blockers.clear()

    def _validate_optimize_inputs(self, *args):
        if not hasattr(self, "optimize_run"):
            return
        self._sync_optimization_inputs()
        busy = bool(getattr(self, "_workflow_busy", None))
        paired = bool(str(self.optimize_fragments.currentPath or ""))
        assisted = bool(self.use_landmarks_check.checked)
        self.optimize_replicates.enabled = not busy and not paired
        self.optimize_landmarks.enabled = not busy and assisted
        self.optimize_visible_labels.enabled = not busy and assisted and paired
        self.optimize_apply.enabled = not busy and self._optimization_result is not None
        self.optimize_save.enabled = not busy and self._optimization_result is not None
        self.optimize_load.enabled = not busy
        self.optimize_run.enabled = False
        if busy:
            self.optimize_run.setToolTip("A workflow is running; finish or cancel it before starting a search.")
            return
        try:
            if not all(s.currentNode() is not None for s in (
                    self.template_model_selector, self.template_dense_selector, self.ssm_table_selector)):
                raise ValueError("Select the SSM, template surface and dense correspondences in Complete Shape")
            if not 0.05 <= float(self.coverage_spin.value) < 1.0:
                raise ValueError(
                    "Set Target coverage below 1.0 above (for example 0.40 for the proximal-humerus "
                    "fragments). At 1.0 there is no withheld anatomy to optimize."
                )
            if not Path(str(self.optimize_complete.currentPath or "__missing__")).is_dir():
                raise ValueError("Select the complete-reference mesh directory")
            if paired and not Path(str(self.optimize_fragments.currentPath)).is_dir():
                raise ValueError("The paired-fragment directory does not exist")
            if assisted:
                if (self.template_sparse_selector.currentNode() is None
                        or not self.optimize_landmarks.currentPath
                        or not Path(str(self.optimize_landmarks.currentPath)).is_dir()):
                    raise ValueError("Landmark mode requires template landmarks and paired target landmark files")
                if paired and len(set(self._optimization_visible_labels())) < 3:
                    raise ValueError("List at least three visible landmark labels; never supply missing-side landmarks")
            candidates = self._optimization_candidates()
            self.optimize_run.enabled = True
            self.optimize_status.setText(
                f"Ready: {len(candidates)} candidate(s), including current settings. "
                f"Coverage {float(self.coverage_spin.value):.2f}; "
                + ("landmark-assisted." if assisted else "surface-only.")
            )
            self.optimize_run.setToolTip("Run the candidate search; current fitting settings will not be changed.")
        except (TypeError, ValueError) as error:
            self.optimize_status.setText(str(error))
            self.optimize_run.setToolTip(str(error))

    def _set_operation(self, operation):
        super()._set_operation(operation)
        if hasattr(self, "optimize_inputs_box"):
            self.optimize_inputs_box.setEnabled(not bool(operation))
            self.optimize_cancel.enabled = operation == "optimize"
            self._validate_optimize_inputs()

    def enter(self):
        super().enter()
        if getattr(self, "_ui_ready", False):
            self._validate_optimize_inputs()

    def cleanup(self):
        self._cancel_optimization = True
        super().cleanup()

    def on_cancel_optimization(self):
        self._cancel_optimization = True
        self.optimize_cancel.enabled = False
        self.optimize_status.setText("Cancellation requested. The current fit will finish; no partial preset is saved.")

    def _optimization_interpolation(self):
        values = {key: getattr(self, name).value for key, name in _OPTIMIZE_INTERPOLATION.items()}
        values["restrict_to_components"] = bool(self.component_interpolation_check.checked)
        return values

    def _optimization_load_input(self, filename, landmarks=False):
        """Copy data out of transient Slicer nodes, then remove only owned nodes."""
        scene = slicer.mrmlScene
        before = {scene.GetNthNode(i).GetID() for i in range(scene.GetNumberOfNodes())}
        target, sparse = self.target_model_selector.currentNode(), self.target_landmark_selector.currentNode()
        try:
            node = (slicer.util.loadMarkups if landmarks else slicer.util.loadModel)(str(filename))
            if node is None:
                raise ValueError(f"Could not load {filename}")
            if landmarks:
                labels, points = self.logic.markups_labels_points_world(node)
                if len(labels) != len(set(labels)) or not np.all(np.isfinite(points)):
                    raise ValueError(f"Landmarks must have unique labels and finite coordinates: {filename}")
                return list(labels), np.array(points, dtype=float, copy=True)
            return self.logic.model_polydata_world(node)
        finally:
            self.target_model_selector.setCurrentNode(target)
            self.target_landmark_selector.setCurrentNode(sparse)
            for index in reversed(range(scene.GetNumberOfNodes())):
                node = scene.GetNthNode(index)
                if node.GetID() not in before and not node.GetSingletonTag():
                    scene.RemoveNode(node)

    def _optimization_cases(self, pairs, base, protocol, checkpoint):
        from hashlib import sha256
        sample_count = protocol["scoring_sample_count"]
        for pair in pairs:
            checkpoint()
            identity = {"specimen": pair["specimen"], "files": {}}
            for kind in ("complete", "fragment", "landmarks"):
                path = pair[kind]
                if path is not None:
                    identity["files"][kind] = {"name": path.name, "sha256": _optimize_file_hash(path)}
            complete, vertices, faces = _optimize_triangles(self._optimization_load_input(pair["complete"]))
            diagonal = bbox_diagonal(vertices)
            if not np.isfinite(diagonal) or diagonal <= 0:
                raise ValueError("Complete-reference extent must be positive")
            axis = np.linalg.svd(vertices - vertices.mean(axis=0), full_matrices=False)[2][0]
            extent = float(np.ptp(vertices @ axis))
            target_labels, target_sparse = (None, None)
            if base.use_landmarks:
                target_labels, target_sparse = self._optimization_load_input(pair["landmarks"], landmarks=True)
                template_labels, template_sparse = self.logic.markups_labels_points_world(
                    self.template_sparse_selector.currentNode())
                dense_points = self.logic.markups_points_world(self.template_dense_selector.currentNode())
            partial = self._optimization_load_input(pair["fragment"]) if pair["fragment"] else None
            # Check that the recorded bytes are the inputs actually read.
            for kind, file_record in identity["files"].items():
                if _optimize_file_hash(pair[kind]) != file_record["sha256"]:
                    raise RuntimeError(f"Input changed while loading: {pair[kind]}")
            protocol["input_files"].append(identity)
            repeats = 1 if partial is not None else protocol["generated_cuts_per_specimen"]
            for replicate in range(repeats):
                checkpoint()
                seed_text = f"{base.seed}:{pair['specimen']}:{replicate}".encode("utf-8")
                rng = np.random.default_rng(int(sha256(seed_text).hexdigest()[:16], 16))
                if partial is None:
                    mask = contiguous_plane_mask(vertices, base.coverage, rng)
                    target = vertices[mask.indices].copy()
                    missing_faces = np.all(~mask.contains(vertices)[faces], axis=1)
                    queries = _optimize_surface_samples(vertices, faces, sample_count, rng, missing_faces)
                    scoring_mode = "generated_cut_wholly_missing_triangles"
                    visible = mask.contains(target_sparse) if base.use_landmarks else None
                else:
                    queries, scoring_mode = _optimize_paired_queries(
                        complete, partial, sample_count, rng, protocol["matching_tolerance_fraction"])
                    # Preserve raw fragment vertex order for the existing fitter.
                    target = vtk_np.vtk_to_numpy(partial.GetPoints().GetData()).astype(np.float64, copy=True)
                    if base.use_landmarks:
                        wanted = set(protocol["visible_landmark_labels"])
                        if not wanted.issubset(target_labels):
                            raise ValueError(f"Visible landmark labels missing from {pair['specimen']}: "
                                             + ", ".join(sorted(wanted - set(target_labels))))
                        visible = np.array([label in wanted for label in target_labels])
                matched = None
                if base.use_landmarks:
                    if int(visible.sum()) < 3:
                        raise ValueError(f"{pair['specimen']} cut {replicate + 1} has fewer than three visible "
                                         "landmarks. Use surface-only mode or representative paired fragments.")
                    matched = match_landmarks(
                        template_labels, template_sparse,
                        [label for label, keep in zip(target_labels, visible) if keep],
                        target_sparse[visible], dense_points,
                        allow_order_fallback=protocol["allow_order_fallback"],
                    )
                    coordinate_compatibility(target, matched.target_points, "visible target landmarks",
                                             max_median_fraction=0.15, max_p95_fraction=0.35)
                yield {"specimen": pair["specimen"], "replicate": replicate,
                       "fragment": target, "landmarks": matched, "queries": queries,
                       "diagonal": diagonal, "axis": axis, "extent": extent,
                       "scoring_mode": scoring_mode}

    def _run_optimization_impl(self):
        from dataclasses import replace
        from datetime import datetime, timezone
        from Resources.Python.MorphoWeaveShapeCompletionBatch import _batch_state_token
        import rustcpd

        base = self._read_settings()
        base.validate()
        if not 0.05 <= base.coverage < 1.0:
            raise ValueError("Optimization needs coverage below 1.0")
        candidates = self._optimization_candidates()
        fragments = str(self.optimize_fragments.currentPath or "")
        labels = self._optimization_visible_labels()
        if base.use_landmarks and fragments and len(set(labels)) < 3:
            raise ValueError("Explicit visible landmark labels are required for paired fragments")
        pairs, ignored = _optimize_pairs(
            str(self.optimize_complete.currentPath), fragments,
            str(self.optimize_landmarks.currentPath or "") if base.use_landmarks else "",
        )
        if base.use_landmarks and any(pair["landmarks"] is None for pair in pairs):
            raise ValueError("Landmark mode requires paired landmark files")
        model = self.template_model_selector.currentNode()
        dense_node = self.template_dense_selector.currentNode()
        table = self.ssm_table_selector.currentNode()
        if model is None or dense_node is None or table is None:
            raise ValueError("Select a template, dense correspondences and SSM table")
        mean, modes, eigenvalues = self.logic.ssm_from_table(table)
        dense = self.logic.markups_points_world(dense_node)
        indexed_correspondence_compatibility(mean, dense)
        template = self.logic.model_polydata_world(model)
        template_vertices = vtk_np.vtk_to_numpy(template.GetPoints().GetData()).astype(float, copy=True)
        _optimize_triangles(template)  # Fail before fitting if it is not a surface.
        interpolation = self._optimization_interpolation()
        transfer, _ = self.logic._get_or_build_transfer_operator(
            template_model=model, template_polydata=template, template_vertices=template_vertices,
            dense_node=dense_node, dense_points=dense, neighbors=interpolation["neighbors"],
            sharpness=interpolation["sharpness"], chunk_size=interpolation["chunk_size"],
            restrict_to_components=interpolation["restrict_to_components"],
        )
        protocol = {
            "source": "paired_fragments" if fragments else "generated_plane_cuts",
            "coverage": base.coverage,
            "generated_cuts_per_specimen": 1 if fragments else int(self.optimize_replicates.value),
            "scoring_sample_count": int(self.optimize_samples.value),
            "matching_tolerance_fraction": float(self.optimize_tolerance.value),
            "visible_landmark_labels": labels if fragments and base.use_landmarks else [],
            "allow_order_fallback": bool(self.order_fallback_check.checked),
            "seed": base.seed, "input_files": [], "ignored_unpaired_files": ignored,
            "selection_metric": "equal_specimen_mean_missing_surface_rmse_divided_by_reference_diagonal",
            "post_fit_alignment": False,
            "scope": "All supplied specimens are tuning data, not held-out validation. "
                     "SSM training membership is not established. Use separate specimens for uncertainty calibration.",
            "generated_cut_note": "Existing plane-mask routine uses vertex fraction; scoring is surface-area weighted.",
        }
        token = _batch_state_token(self)
        def checkpoint():
            if self._cancel_optimization:
                raise _OptimizationCancelled()
            if token != _batch_state_token(self):
                raise RuntimeError("Shared model or settings changed during optimization; no preset was produced")
        case_count = len(pairs) * protocol["generated_cuts_per_specimen"]
        total = case_count * len(candidates)
        self.optimize_log.appendPlainText(f"{len(pairs)} specimens; {len(candidates)} candidates; {total} fits.")
        if ignored["unpaired_fragments"] or ignored["complete_without_fragment"]:
            self.optimize_log.appendPlainText("Excluded unmatched basenames: " + json.dumps(ignored))
        rows = []
        for case in self._optimization_cases(pairs, base, protocol, checkpoint):
            for candidate in candidates:
                checkpoint()
                self.optimize_status.setText(f"{len(rows) + 1}/{total}: {case['specimen']} / {candidate['name']}")
                slicer.app.processEvents()
                checkpoint()
                started = time.perf_counter()
                row = {"specimen": case["specimen"], "replicate": case["replicate"],
                       "candidate": candidate["id"], "status": "failed", "scoring_mode": case["scoring_mode"]}
                try:
                    settings = replace(CompletionSettings(**candidate["settings"]), posterior_samples=0)
                    result = run_completion(case["fragment"].copy(), mean, modes, eigenvalues, settings,
                                            rustcpd_module=rustcpd, landmarks=case["landmarks"])
                    # Same exact-pose plus float32 residual transfer as normal output.
                    posed = apply_similarity(template_vertices, result.world_scale,
                                             result.world_rotation, result.world_translation)
                    residual = result.completed_points - apply_similarity(
                        dense, result.world_scale, result.world_rotation, result.world_translation)
                    warped = posed + transfer["operator"].apply(residual.astype(np.float32))
                    predicted = self.logic.polydata_with_points(template, warped, recompute_normals=False)
                    surface, exported_vertices, _ = _optimize_triangles(predicted)
                    distances = _optimize_surface_distances(surface, case["queries"])
                    error = float(np.sqrt(np.mean(distances ** 2)))
                    row.update(status="success", missing_rmse=error,
                               missing_rmse_fraction=error / case["diagonal"],
                               long_axis_extent_bias_percent=100.0 * (
                                   float(np.ptp(exported_vertices @ case["axis"])) / case["extent"] - 1),
                               world_scale=float(result.world_scale))
                    del result
                except _OptimizationCancelled:
                    raise
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                    logging.exception("Optimize candidate %s failed for %s", candidate["id"], case["specimen"])
                row["elapsed_seconds"] = time.perf_counter() - started
                rows.append(row)
                self.optimize_log.appendPlainText(
                    f"{row['specimen']} {row['candidate']}: " +
                    (f"missing RMSE {100 * row['missing_rmse_fraction']:.3f}% diagonal; "
                     f"long-axis extent bias {row['long_axis_extent_bias_percent']:+.2f}%"
                     if row["status"] == "success" else row["error"]))
                self.optimize_progress.setValue(round(100 * len(rows) / total))
                slicer.app.processEvents()
                checkpoint()
        ranked = _optimize_rank(candidates, rows, case_count)
        if not ranked or ranked[0]["failed_cases"]:
            raise RuntimeError("No candidate completed every case. Inspect the failures; no preset was selected")
        winner = ranked[0]
        values = dict(winner["settings"], allow_order_fallback=protocol["allow_order_fallback"])
        report = {
            "schema": OPTIMIZATION_PRESET_SCHEMA, "version": OPTIMIZATION_PRESET_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "rustcpd_version": str(rustcpd.__version__),
            "model_fingerprint_sha256": model_fingerprint(mean, modes, eigenvalues),
            "completion_core_sha256": _optimize_file_hash(inspect.getfile(run_completion)),
            "settings": values, "interpolation": interpolation,
            "best_candidate": winner["id"], "training_summary": winner,
            "protocol": protocol, "ranked_candidates": ranked, "evaluations": rows,
        }
        self.optimize_log.appendPlainText("\nRanking (tuning data; not an independent accuracy estimate):")
        for entry in ranked:
            label = (f"{100 * entry['mean_missing_rmse_fraction']:.3f}% diagonal"
                     if entry["failed_cases"] == 0 else f"disqualified: {entry['failed_cases']} failed case(s)")
            self.optimize_log.appendPlainText(f"{entry['id']} {entry['name']}: {label}")
        self._optimization_result = report
        return f"Best: {winner['name']} ({100 * winner['mean_missing_rmse_fraction']:.3f}% diagonal on tuning data). Settings are not applied until you click Apply or load a saved preset."

    def on_run_optimization(self):
        if self._workflow_busy:
            return
        self._cancel_optimization = False
        self._optimization_result = None
        self._set_operation("optimize")
        self.optimize_log.clear()
        self.optimize_progress.setValue(0)
        final_status = "Optimization did not start."
        try:
            if not self._ensure_dependencies():
                final_status = "Optimization did not start: dependencies unavailable."
                return
            final_status = self._run_optimization_impl()
        except _OptimizationCancelled:
            final_status = "Optimization cancelled. No incomplete winner or preset was saved."
        except Exception as error:
            logging.exception("Shape Completion optimization failed")
            final_status = f"Optimization failed: {error}"
            self.optimize_log.appendPlainText(final_status)
            slicer.util.errorDisplay(final_status)
        finally:
            self._set_operation(None)
            self.optimize_status.setText(final_status)

    def _apply_optimization_preset(self, preset):
        """Validate completely, apply transactionally, verify readback. Never run a fit."""
        from dataclasses import asdict
        if self._workflow_busy:
            raise RuntimeError("Cannot apply settings during another workflow")
        if not isinstance(preset, dict) or preset.get("schema") != OPTIMIZATION_PRESET_SCHEMA:
            raise ValueError("Not a Shape Completion hyperparameter preset (calibration profiles are different)")
        if type(preset.get("version")) is not int or preset["version"] != OPTIMIZATION_PRESET_VERSION:
            raise ValueError("Unsupported preset version")
        if preset.get("rustcpd_version") != "4.0.0":
            raise ValueError("This preset must use rustcpd 4.0.0")
        loaded = sys.modules.get("rustcpd")
        if loaded is not None and getattr(loaded, "__version__", None) != "4.0.0":
            raise ValueError("Restart Slicer with rustcpd 4.0.0 before applying this preset")
        mean, modes, eigenvalues = self.logic.ssm_from_table(self.ssm_table_selector.currentNode())
        if preset.get("model_fingerprint_sha256") != model_fingerprint(mean, modes, eigenvalues):
            raise ValueError("Preset belongs to a different SSM; load the matching model first")
        if preset.get("completion_core_sha256") != _optimize_file_hash(inspect.getfile(run_completion)):
            raise ValueError("Preset was produced by a different completion core")
        values = preset.get("settings")
        current = self._settings_snapshot(self._read_settings())
        if not isinstance(values, dict) or values.keys() != current.keys():
            raise ValueError("Preset settings have missing or unsupported fields")
        for key, value in values.items():
            reference = current[key]
            if key in _OPTIMIZE_CHECKS or type(reference) is bool:
                if type(value) is not bool:
                    raise ValueError(f"{key} must be true or false")
            elif key == "residual_scale_policy" or key == "coarse_score_mode":
                if not isinstance(value, str):
                    raise ValueError(f"{key} must be a string")
            elif value is None:
                if key not in _OPTIMIZE_NULLABLE and key != "atlas_sigma2_from_pose_factor":
                    raise ValueError(f"{key} cannot be null")
            elif type(value) not in (int, float) or not np.isfinite(value):
                raise ValueError(f"{key} must be finite numeric data")
            elif (type(reference) is int or key in ("refine_source_count", "atlas_k")) and type(value) is not int:
                raise ValueError(f"{key} must be an integer")
        settings = CompletionSettings(**{key: value for key, value in values.items() if key != "allow_order_fallback"})
        settings.validate()
        hidden = current.keys() - _OPTIMIZE_SCALARS.keys() - _OPTIMIZE_CHECKS.keys() - {"residual_scale_policy"}
        if any(values[key] != current[key] for key in hidden):
            raise ValueError("Preset changes settings without GUI controls: " + ", ".join(sorted(hidden)))
        interpolation = preset.get("interpolation")
        if not isinstance(interpolation, dict) or interpolation.keys() != self._optimization_interpolation().keys():
            raise ValueError("Preset has missing or unsupported interpolation fields")
        if type(interpolation["restrict_to_components"]) is not bool:
            raise ValueError("restrict_to_components must be true or false")
        numeric = [(getattr(self, name), values[key], key) for key, name in _OPTIMIZE_SCALARS.items()]
        numeric += [(getattr(self, name), interpolation[key], key) for key, name in _OPTIMIZE_INTERPOLATION.items()]
        for control, value, key in numeric:
            numeric_value = 0 if value is None else value
            if type(numeric_value) not in (int, float) or not np.isfinite(numeric_value):
                raise ValueError(f"{key} must be finite")
            if key in ("neighbors", "chunk_size") and type(value) is not int:
                raise ValueError(f"{key} must be an integer")
            if not control.minimum <= numeric_value <= control.maximum:
                raise ValueError(f"{key} lies outside the supported UI range")
        # No controls have changed before this point.
        controls = [control for control, _, _ in numeric]
        controls += [getattr(self, name) for name in _OPTIMIZE_CHECKS.values()]
        controls += [self.scale_policy_combo, self.component_interpolation_check]
        numeric_before = [(control, control.value) for control, _, _ in numeric]
        checks_before = [(getattr(self, name), getattr(self, name).checked) for name in _OPTIMIZE_CHECKS.values()]
        checks_before.append((self.component_interpolation_check, self.component_interpolation_check.checked))
        old_index = int(self.scale_policy_combo.currentIndex)
        blockers = [qt.QSignalBlocker(control) for control in controls]
        try:
            for control, value, key in numeric:
                if key not in ("target_point_count", "rotation_count", "coarse_source_count", "coarse_target_count",
                               "coarse_rank", "coarse_iterations", "coarse_screen_iterations", "coarse_survivor_count",
                               "refine_count", "refine_source_count", "refine_target_count", "refine_iterations",
                               "atlas_max_iterations", "atlas_k", "posterior_samples", "seed", "translation_anchor_count",
                               "neighbors", "chunk_size"):
                    control.setDecimals(12)  # In particular, preserve 0.025, not a rounded 0.03.
                control.setValue(0 if value is None else value)
            for key, name in _OPTIMIZE_CHECKS.items():
                getattr(self, name).setChecked(values[key])
            self.component_interpolation_check.setChecked(interpolation["restrict_to_components"])
            self.scale_policy_combo.setCurrentIndex({"auto": 0, "free": 1, "fixed": 2}[values["residual_scale_policy"]])
            if self._settings_snapshot(self._read_settings()) != values or self._optimization_interpolation() != interpolation:
                raise ValueError("The controls could not represent the preset exactly; settings were restored")
        except Exception:
            for control, value in numeric_before:
                control.setValue(value)
            for control, value in checks_before:
                control.setChecked(value)
            self.scale_policy_combo.setCurrentIndex(old_index)
            raise
        finally:
            blockers.clear()
        # Source signals were blocked: explicitly mirror the shared Batch controls.
        self.batch_coverage.setDecimals(12)
        self.batch_coverage.setValue(float(self.coverage_spin.value))
        self.batch_use_landmarks.setChecked(bool(self.use_landmarks_check.checked))
        self.batch_order_fallback.setChecked(bool(self.order_fallback_check.checked))
        self.profile_path.setCurrentPath("")
        self._on_profile_path_changed("")
        self.batch_profile.setCurrentPath("")
        self._validate_complete_inputs()
        self._validate_calibration_inputs()
        self._validate_batch_inputs()
        self._validate_optimize_inputs()
        self.optimize_status.setText("Preset applied to Complete Shape and Batch. No fit was started. Previous uncertainty calibration cleared; recalibrate separately.")

    def on_apply_optimized_settings(self):
        if self._workflow_busy or self._optimization_result is None:
            return
        try:
            self._apply_optimization_preset(self._optimization_result)
        except Exception as error:
            slicer.util.errorDisplay(f"Could not apply preset: {error}")

    def on_save_optimization_preset(self):
        if self._workflow_busy or self._optimization_result is None:
            return
        path = qt.QFileDialog.getSaveFileName(self.parent, "Save Shape Completion Hyperparameters",
                                              "ShapeCompletionHyperparameters.json", "JSON (*.json)")
        if isinstance(path, (tuple, list)):
            path = path[0]
        if not path:
            return
        path = str(path) if str(path).lower().endswith(".json") else str(path) + ".json"
        try:
            _optimize_save_json(path, self._optimization_result)
            self.optimize_status.setText(f"Saved preset and search scores to {path}. Load and Apply restores its hyperparameters.")
        except Exception as error:
            slicer.util.errorDisplay(f"Could not save preset: {error}")

    def on_load_optimization_preset(self):
        if self._workflow_busy:
            return
        path = qt.QFileDialog.getOpenFileName(self.parent, "Load Shape Completion Hyperparameters", "", "JSON (*.json)")
        if isinstance(path, (tuple, list)):
            path = path[0]
        if not path:
            return
        try:
            if Path(str(path)).stat().st_size > 64 * 1024 * 1024:
                raise ValueError("Preset exceeds the 64 MiB limit")
            def invalid_constant(value):
                raise ValueError(f"Invalid JSON numeric constant: {value}")
            with open(str(path), encoding="utf-8") as stream:
                preset = json.load(stream, parse_constant=invalid_constant)
            self._apply_optimization_preset(preset)
        except Exception as error:
            slicer.util.errorDisplay(f"Could not load preset: {error}")

    def _ensure_dependencies(self):
        if self._deps_ready:
            return True
        import slicer.packaging
        try:
            loaded = sys.modules.get("rustcpd")
            if loaded is not None and getattr(loaded, "__version__", None) != "4.0.0":
                raise RuntimeError(
                    "Shape Completion requires rustcpd==4.0.0. Restart Slicer before "
                    "replacing an already loaded backend. No packages were changed."
                )
            slicer.packaging.pip_ensure(
                [RUSTCPD_REQUIREMENT], prompt_install=True, requester="Shape Completion"
            )
            importlib.invalidate_caches()
            validate_completion_backend(importlib.import_module("rustcpd"))
            self._deps_ready = True
            return True
        except Exception as error:
            if isinstance(error, RuntimeError) and str(error) == "User declined package installation":
                slicer.util.showStatusMessage("Shape Completion dependencies were not installed.", 3000)
                return False
            logging.exception("Shape Completion dependency setup failed")
            slicer.util.errorDisplay(f"Shape Completion dependency setup failed:\n{error}")
            return False

    def _auto_select_canonical_ssm_set(self):
        changed = super()._auto_select_canonical_ssm_set()
        # Source signals were intentionally blocked and are not replayed by Qt.
        for source, destination in (
            (self.template_model_selector, self.batch_template_model),
            (self.template_dense_selector, self.batch_template_dense),
            (self.template_sparse_selector, self.batch_template_sparse),
            (self.ssm_table_selector, self.batch_ssm_table),
        ):
            if destination.currentNode() != source.currentNode():
                destination.setCurrentNode(source.currentNode())
        self._validate_batch_inputs()
        self._validate_optimize_inputs()
        return changed


class MorphoWeaveShapeCompletionTest(_CompletionTest):
    pass
