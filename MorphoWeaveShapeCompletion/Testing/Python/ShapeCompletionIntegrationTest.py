"""Entry-point regression tests; no native Slicer/rustcpd required.

The selector double implements blocked-signal semantics (signals are discarded,
not queued). These tests do not replace the live Slicer smoke test.
"""
import importlib.util
import inspect
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

MODULE_DIR = Path(__file__).resolve().parents[2]


class Signal:
    def __init__(self, owner):
        self.owner = owner
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self):
        if not self.owner.blocked:
            for callback in self.callbacks:
                callback()


class Selector:
    def __init__(self, node=None):
        self.node = node
        self.blocked = False
        self.currentNodeChanged = Signal(self)

    def currentNode(self):
        return self.node

    def setCurrentNode(self, node):
        if self.node != node:
            self.node = node
            self.currentNodeChanged.emit()

    def blockSignals(self, blocked):
        previous = self.blocked
        self.blocked = blocked
        return previous


PAIRS = (
    ("template_model_selector", "batch_template_model"),
    ("template_dense_selector", "batch_template_dense"),
    ("template_sparse_selector", "batch_template_sparse"),
    ("ssm_table_selector", "batch_ssm_table"),
)


class BaseWidget:
    """Minimal base with the production automatic-selection signal policy."""
    def __init__(self):
        self._deps_ready = False
        self.resolved = [object() for _ in PAIRS]
        self.validations = 0
        self.target_model_selector = Selector(object())
        self.target_landmark_selector = Selector(object())
        for left, right in PAIRS:
            source, destination = Selector(), Selector()
            setattr(self, left, source)
            setattr(self, right, destination)
            source.currentNodeChanged.connect(
                lambda s=source, d=destination: d.setCurrentNode(s.currentNode()))
            destination.currentNodeChanged.connect(
                lambda s=destination, d=source: d.setCurrentNode(s.currentNode()))

    def setup(self):
        self._auto_select_canonical_ssm_set()

    def enter(self):
        self._auto_select_canonical_ssm_set()

    def _auto_select_canonical_ssm_set(self):
        if self.resolved is None:
            return False
        changed = False
        blocked = []
        try:
            for left, _ in PAIRS:
                source = getattr(self, left)
                if source.currentNode() is None:
                    blocked.append((source, source.blockSignals(True)))
            for (left, _), node in zip(PAIRS, self.resolved):
                source = getattr(self, left)
                if source.currentNode() is None:
                    source.setCurrentNode(node)
                    changed = True
            return changed
        finally:
            for source, previous in blocked:
                source.blockSignals(previous)

    def _validate_batch_inputs(self):
        self.validations += 1

    def _ensure_dependencies(self):
        raise AssertionError("The obsolete base dependency gate must not be called")


def function_with_parameters(names):
    def function(**kwargs):
        return None
    function.__signature__ = inspect.Signature([
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
        for name in names.split()
    ])
    return function


def valid_backend():
    backend = types.SimpleNamespace(__version__="4.0.0")
    backend.pose_initialize = function_with_parameters(
        "rotation_count coarse_source_count coarse_target_count coarse_rank coarse_iterations "
        "coarse_screen_iterations coarse_survivor_count coarse_score_mode refine_count "
        "refine_source_count refine_target_count refine_iterations lambda_regularization "
        "outlier_weight identity_prior_probability landmark_indices landmark_targets "
        "landmark_sigma refine_landmark_sigma with_scale seed parallel single_precision")
    backend.register_atlas = function_with_parameters(
        "lambda_regularization normalize optimize_similarity with_scale initial_coefficients "
        "initial_rotation initial_scale initial_translation landmark_indices landmark_targets "
        "landmark_sigma max_iterations tolerance outlier_weight k parallel single_precision")
    for cls, names in {
        "PoseInitialization": "coefficients rotation scale translation score score_margin posterior_entropy effective_hypotheses hypotheses_evaluated hypotheses_refined",
        "AtlasResult": "points coefficients rotation scale translation sigma2 iterations difference landmark_rms",
        "ShapePosterior": "coefficient_covariance noise_variance discrepancy_variance predict predictive_variance sample_shapes",
    }.items():
        setattr(backend, cls, type(cls, (), {name: None for name in names.split()}))
    backend.AtlasResult.posterior = function_with_parameters(
        "completeness prior_temperature outlier_weight estimate_discrepancy")
    return backend


class ShapeCompletionIntegrationTest(unittest.TestCase):
    def setUp(self):
        base = types.ModuleType("Resources.Python.MorphoWeaveShapeCompletionBase")
        base.MorphoWeaveShapeCompletion = type("Module", (), {})
        base.MorphoWeaveShapeCompletionWidget = BaseWidget
        base.MorphoWeaveShapeCompletionTest = type("Test", (), {})
        batch = types.ModuleType("Resources.Python.MorphoWeaveShapeCompletionBatch")
        batch.ShapeCompletionBatchMixin = type("BatchMixin", (), {})
        self.slicer = types.ModuleType("slicer")
        self.slicer.__path__ = []
        self.slicer.util = types.SimpleNamespace(errorDisplay=Mock(), showStatusMessage=Mock())
        self.packaging = types.ModuleType("slicer.packaging")
        self.packaging.pip_ensure = Mock()
        self.slicer.packaging = self.packaging
        modules = {
            "Resources": types.ModuleType("Resources"),
            "Resources.Python": types.ModuleType("Resources.Python"),
            base.__name__: base, batch.__name__: batch,
            "slicer": self.slicer, "slicer.packaging": self.packaging,
        }
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        spec = importlib.util.spec_from_file_location(
            "completion_entrypoint_under_test", MODULE_DIR / "MorphoWeaveShapeCompletion.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.widget = self.module.MorphoWeaveShapeCompletionWidget()
        self.backend = valid_backend()
        importer = patch.object(self.module.importlib, "import_module", return_value=self.backend)
        importer.start()
        self.addCleanup(importer.stop)
        logger = patch.object(self.module.logging, "exception")
        logger.start()
        self.addCleanup(logger.stop)

    def assert_mirrors(self):
        for left, right in PAIRS:
            self.assertIs(getattr(self.widget, left).currentNode(),
                          getattr(self.widget, right).currentNode())

    def test_double_reproduces_blocked_base_bug(self):
        BaseWidget._auto_select_canonical_ssm_set(self.widget)
        self.assertTrue(all(getattr(self.widget, left).currentNode() is not None for left, _ in PAIRS))
        self.assertTrue(all(getattr(self.widget, right).currentNode() is None for _, right in PAIRS))

    def test_setup_mirrors_automatically_loaded_model(self):
        self.widget.setup()
        self.assert_mirrors()
        self.assertEqual(self.widget.validations, 1)

    def test_enter_mirrors_newly_loaded_model(self):
        self.widget.resolved = None
        self.widget.setup()
        self.widget.resolved = [object() for _ in PAIRS]
        self.widget.enter()
        self.assert_mirrors()
        self.assertIsNotNone(self.widget.batch_ssm_table.currentNode())

    def test_manual_inputs_and_targets_are_preserved(self):
        manual = object()
        self.widget.template_model_selector.setCurrentNode(manual)
        target = self.widget.target_model_selector.currentNode()
        landmarks = self.widget.target_landmark_selector.currentNode()
        self.widget.setup()
        self.assertIs(self.widget.batch_template_model.currentNode(), manual)
        self.assertIs(self.widget.target_model_selector.currentNode(), target)
        self.assertIs(self.widget.target_landmark_selector.currentNode(), landmarks)
        self.assert_mirrors()

    def test_idempotent_reentry_does_not_emit_extra_changes(self):
        self.widget.setup()
        callback = Mock()
        self.widget.batch_ssm_table.currentNodeChanged.connect(callback)
        self.assertFalse(self.widget._auto_select_canonical_ssm_set())
        callback.assert_not_called()

    def test_partial_selection_is_filled_and_synchronized(self):
        self.widget.template_dense_selector.setCurrentNode(object())
        self.assertTrue(self.widget._auto_select_canonical_ssm_set())
        self.assert_mirrors()

    def test_no_canonical_set_still_synchronizes_existing_selections(self):
        self.widget.resolved = None
        source = self.widget.ssm_table_selector
        source.blockSignals(True)
        source.setCurrentNode(object())
        source.blockSignals(False)
        self.assertFalse(self.widget._auto_select_canonical_ssm_set())
        self.assert_mirrors()

    def test_manual_batch_change_still_updates_single(self):
        self.widget.setup()
        value = object()
        self.widget.batch_ssm_table.setCurrentNode(value)
        self.assertIs(self.widget.ssm_table_selector.currentNode(), value)

    def test_requires_supported_release_range_and_retains_cache(self):
        self.assertTrue(self.widget._ensure_dependencies())
        self.packaging.pip_ensure.assert_called_once_with(
            ["rustcpd>=3.1,<5"], prompt_install=True, requester="Shape Completion")
        self.assertTrue(self.widget._ensure_dependencies())
        self.assertEqual(self.packaging.pip_ensure.call_count, 1)

    def test_old_loaded_extension_is_rejected_with_restart_guidance(self):
        self.backend.__version__ = "3.0.0"
        self.assertFalse(self.widget._ensure_dependencies())
        self.assertFalse(self.widget._deps_ready)
        message = self.slicer.util.errorDisplay.call_args.args[0]
        self.assertIn("restart Slicer", message)
        self.assertNotIn("development branch", message)

    def test_future_major_and_prerelease_are_not_accepted(self):
        for version in ("5.0.0", "3.0.0", "3.1.0rc1", "4.0.0rc1",
                        "4.0.0.dev1", "4.0.0+", "4.0.0+foo..bar", "garbage"):
            with self.subTest(version=version):
                self.backend.__version__ = version
                with self.assertRaises(RuntimeError):
                    self.module.validate_completion_backend(self.backend)

    def test_minor_post_and_local_versions_are_supported(self):
        for version in ("3.1.0", "3.2.1", "3.1.0.post1", "3.1.0+local",
                        "4.0.0", "4.1.2", "4.0.0.post1", "4.0.0+local"):
            self.backend.__version__ = version
            self.module.validate_completion_backend(self.backend)

    def test_released_400_passes_public_preflight(self):
        self.backend.__version__ = "4.0.0"
        self.assertTrue(self.widget._ensure_dependencies())
        self.slicer.util.errorDisplay.assert_not_called()
        self.assertTrue(self.widget._deps_ready)

    def test_31_is_still_accepted_for_existing_environments(self):
        self.backend.__version__ = "3.1.0"
        self.assertTrue(self.widget._ensure_dependencies())
        self.slicer.util.errorDisplay.assert_not_called()

    def test_400_with_missing_posterior_api_is_not_accepted(self):
        self.backend.__version__ = "4.0.0"
        del self.backend.AtlasResult.posterior
        self.assertFalse(self.widget._ensure_dependencies())
        self.assertIn("AtlasResult.posterior", self.slicer.util.errorDisplay.call_args.args[0])

    def test_capability_checks_reject_missing_landmark_parameter(self):
        signature = self.backend.pose_initialize.__signature__
        self.backend.pose_initialize.__signature__ = signature.replace(parameters=[
            value for key, value in signature.parameters.items() if key != "refine_landmark_sigma"])
        self.assertFalse(self.widget._ensure_dependencies())
        self.assertIn("pose_initialize.refine_landmark_sigma", self.slicer.util.errorDisplay.call_args.args[0])

    def test_missing_classes_or_posterior_parameters_are_reported(self):
        del self.backend.ShapePosterior
        self.backend.AtlasResult.posterior = function_with_parameters("completeness")
        with self.assertRaisesRegex(RuntimeError, "ShapePosterior"):
            self.module.validate_completion_backend(self.backend)

    def test_install_decline_does_not_display_failure_dialog(self):
        self.packaging.pip_ensure.side_effect = RuntimeError("User declined package installation")
        self.assertFalse(self.widget._ensure_dependencies())
        self.slicer.util.errorDisplay.assert_not_called()
        self.assertFalse(self.widget._deps_ready)

    def test_install_failure_is_not_cached_and_retry_succeeds(self):
        self.packaging.pip_ensure.side_effect = RuntimeError("installation failed")
        self.assertFalse(self.widget._ensure_dependencies())
        self.packaging.pip_ensure.side_effect = None
        self.assertTrue(self.widget._ensure_dependencies())


if __name__ == "__main__":
    unittest.main()
