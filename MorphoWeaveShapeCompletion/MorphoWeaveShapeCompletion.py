"""Discoverable single/batch entry point around the uploaded fragment workflow.

The numerical core is restored byte-for-byte from the user's working archive.
This wrapper handles deployment and synchronization, not fitting.
"""
import importlib
import inspect
import logging
import sys

import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from Resources.Python.MorphoWeaveShapeCompletionBase import *  # noqa: F401,F403
from Resources.Python.MorphoWeaveShapeCompletionBase import (
    MorphoWeaveShapeCompletion as _CompletionModule,
    MorphoWeaveShapeCompletionWidget as _SingleCompletionWidget,
    MorphoWeaveShapeCompletionTest as _CompletionTest,
)
from Resources.Python.MorphoWeaveShapeCompletionBatch import ShapeCompletionBatchMixin

RUSTCPD_REQUIREMENT = "rustcpd==4.0.0"


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
            "See RESTORATION.md for preserved behavior and validation limits."
        )


class MorphoWeaveShapeCompletionWidget(ShapeCompletionBatchMixin, _SingleCompletionWidget):
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
        return changed


class MorphoWeaveShapeCompletionTest(_CompletionTest):
    pass
