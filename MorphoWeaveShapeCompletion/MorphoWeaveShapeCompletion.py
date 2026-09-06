"""Slicer entry point for single and batch shape completion.

The numerical pipeline remains in MorphoWeaveShapeCompletionBase. Deployment
requirements and synchronization of the two tabs belong to this entry point.
"""
import importlib
import inspect
import logging
import re

import slicer

# Preserve the module's existing public helpers and logic for Slicer scripts.
from Resources.Python.MorphoWeaveShapeCompletionBase import *  # noqa: F401,F403
from Resources.Python.MorphoWeaveShapeCompletionBase import (
    MorphoWeaveShapeCompletion as _CompletionModule,
    MorphoWeaveShapeCompletionWidget as _SingleCompletionWidget,
    MorphoWeaveShapeCompletionTest as _CompletionTest,
)
from Resources.Python.MorphoWeaveShapeCompletionBatch import ShapeCompletionBatchMixin


RUSTCPD_REQUIREMENT = "rustcpd>=3.1,<5"


def validate_completion_backend(backend):
    """Keep capability checks even when a wheel satisfies the version range."""
    required_parameters = {
        "pose_initialize": (
            "rotation_count", "coarse_source_count", "coarse_target_count",
            "coarse_rank", "coarse_iterations", "coarse_screen_iterations",
            "coarse_survivor_count", "coarse_score_mode", "refine_count",
            "refine_source_count", "refine_target_count", "refine_iterations",
            "lambda_regularization", "outlier_weight", "identity_prior_probability",
            "landmark_indices", "landmark_targets", "landmark_sigma",
            "refine_landmark_sigma", "with_scale", "seed", "parallel", "single_precision",
        ),
        "register_atlas": (
            "lambda_regularization", "normalize", "optimize_similarity", "with_scale",
            "initial_coefficients", "initial_rotation", "initial_scale", "initial_translation",
            "landmark_indices", "landmark_targets", "landmark_sigma", "max_iterations",
            "tolerance", "outlier_weight", "k", "parallel", "single_precision",
        ),
        "AtlasResult.posterior": (
            "completeness", "prior_temperature", "outlier_weight", "estimate_discrepancy",
        ),
    }
    required_attributes = {
        "PoseInitialization": (
            "coefficients", "rotation", "scale", "translation", "score", "score_margin",
            "posterior_entropy", "effective_hypotheses", "hypotheses_evaluated", "hypotheses_refined",
        ),
        "AtlasResult": (
            "points", "coefficients", "rotation", "scale", "translation", "sigma2",
            "iterations", "difference", "landmark_rms", "posterior",
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
    # pip may have upgraded the files while an older extension is still loaded.
    version = getattr(backend, "__version__", None)
    if version is not None:
        release = re.fullmatch(
            r"([34])\.(\d+)\.\d+(?:\.post\d+)?(?:\+[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)?",
            str(version),
        )
        if release is None or (int(release.group(1)) == 3 and int(release.group(2)) < 1):
            missing.append(f"supported loaded version (found {version})")
    if missing:
        raise RuntimeError(
            "Shape Completion requires " + RUSTCPD_REQUIREMENT
            + " with the constrained completion API. Missing or incompatible: "
            + ", ".join(missing)
            + ". Install or upgrade to a supported released wheel, then restart Slicer "
            "if an older rustcpd version was already imported."
        )


class MorphoWeaveShapeCompletion(_CompletionModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.helpText += (
            " Use the Batch tab to process a directory of fragments with the "
            "same model and settings, optional paired landmarks, and resumable exports."
            f" Shape Completion requires {RUSTCPD_REQUIREMENT}."
            " When upgrading from rustcpd 3.1 to 4.0, revalidate completions and "
            "regenerate calibration profiles; posterior results are not numerically equivalent."
        )


class MorphoWeaveShapeCompletionWidget(ShapeCompletionBatchMixin, _SingleCompletionWidget):
    """Complete Shape, Batch, Calibration and Advanced in one module."""

    def _ensure_dependencies(self):
        # Override the historical base widget's deployment preflight for every
        # entry point: setup, single completion, calibration, and batch.
        if self._deps_ready:
            return True
        import slicer.packaging

        try:
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
        # The base intentionally blocks source-selector signals while filling
        # empty inputs. Qt does not replay those signals, so explicitly mirror
        # the final selections after all of the blockers have been released.
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
