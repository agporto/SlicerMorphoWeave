import ast
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

MODULE_DIR = Path(__file__).resolve().parents[2]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


def _load_source_function(name):
    source_path = MODULE_DIR / "MorphoWeaveLandmarkTransfer.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"re": re}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            filename=str(source_path),
            mode="exec",
        ),
        namespace,
    )
    return namespace[name]


stable_distribution_release = _load_source_function("stable_distribution_release")


from Resources.Python.PoseEMTemplate import (
    LEGACY_BACKEND,
    POSE_EM_BACKEND,
    PoseEMSettings,
    _adapt_rustcpd_initialization,
    coverage_prescale,
    optimization_backend,
    pose_em_enabled,
    run_pose_em_registration,
    similarity_matrix,
    ssm_sample,
    transform_points,
)


def _initializer(coefficients, rotation, scale, translation):
    def initialize(*args, **kwargs):
        initialize.args.append(args)
        initialize.calls.append(kwargs)
        return SimpleNamespace(
            coefficients=np.asarray(coefficients),
            rotation=np.asarray(rotation),
            scale=float(scale),
            translation=np.asarray(translation).reshape(1, 3),
            score=12.5,
            score_margin=3.25,
            posterior_entropy=0.4,
            effective_hypotheses=1.49,
            hypotheses_evaluated=61,
            hypotheses_refined=2,
        )
    initialize.args = []
    initialize.calls = []
    return initialize


class PoseEMTemplateUnitTest(unittest.TestCase):
    def test_helper_is_packaged_as_a_resource_not_a_slicer_module(self):
        cmake = (MODULE_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
        scripts = cmake.split("set(MODULE_PYTHON_SCRIPTS", 1)[1].split(")", 1)[0]
        resources = cmake.split("set(MODULE_PYTHON_RESOURCES", 1)[1].split(")", 1)[0]
        self.assertNotIn("PoseEMTemplate.py", scripts)
        self.assertIn("Resources/Python/PoseEMTemplate.py", resources)
        self.assertFalse((MODULE_DIR / "PoseEMTemplate.py").exists())

    def test_slicer_module_wires_backend_selector_to_standalone_and_batch_paths(self):
        source = (MODULE_DIR / "MorphoWeaveLandmarkTransfer.py").read_text(encoding="utf-8")
        compile(source, str(MODULE_DIR / "MorphoWeaveLandmarkTransfer.py"), "exec")
        self.assertIn("Pose-marginalized EM (experimental)", source)
        self.assertIn("backend = optimization_backend(parameters)", source)
        self.assertIn("pose_em = pose_em_enabled(parameters, skip_optimization=skipOpt)", source)
        self.assertIn("logic.initialize_template_pose_em", source)
        self.assertIn("self.runPoseEMDeformable", source)
        self.assertIn('"pose_em_diagnostics.json"', source)
        self.assertIn('"spec": "tiny3d-rs>=2.1,<3"', source)
        self.assertIn('"spec": "rustcpd>=3.0,<4"', source)
        self.assertIn('distributionInstalled("tiny3d")', source)
        self.assertIn('pip_uninstall("tiny3d")', source)
        self.assertIn("loaded_replacements", source)
        self.assertIn("Restart Slicer, then reopen", source)
        self.assertIn("rustcpd.register_atlas", source)
        self.assertIn("rustcpd.register_deformable", source)
        self.assertIn("np.asarray(pca.points)", source)
        self.assertIn("np.asarray(final.points)", source)
        self.assertIn('getattr(rustcpd_module, "pose_initialize")', source)
        self.assertNotIn("pose_marginalized_initialization", source)
        self.assertNotIn("runFineDeformable", source)
        self.assertNotIn("blas_threads_limited", source)
        self.assertNotIn("dense_completion_skipped", source)
        self.assertNotIn("downstream_rigid", source)
        self.assertIn('parameters.get("lambda_reg", 0.01)', source)
        self.assertIn('parameters.get("w", 0.10)', source)
        self.assertIn('parameters.get("max_iterations", 250)', source)
        self.assertNotIn("from biocpd", source)
        self.assertNotIn("from packaging", source)
        self.assertIn("slicer.util.pip_install", source)
        self.assertNotIn("subprocess.check_call", source)
        self.assertNotIn("git+https://", source)

    def test_distribution_release_parser_rejects_prereleases(self):
        self.assertEqual(stable_distribution_release("3.0.0"), (3, 0, 0))
        self.assertEqual(stable_distribution_release("2.1.0.post1"), (2, 1, 0))
        self.assertEqual(stable_distribution_release("3.0+cpu"), (3, 0))
        self.assertIsNone(stable_distribution_release("3.0.0rc1"))
        self.assertIsNone(stable_distribution_release("2.1.0.dev2"))

    def test_pose_em_never_bypasses_standard_alignment_stages(self):
        source = (MODULE_DIR / "MorphoWeaveLandmarkTransfer.py").read_text(encoding="utf-8")
        self.assertNotIn("poseEMPrealigned", source)
        self.assertNotIn("Pose-EM Identity Handoff", source)
        self.assertNotIn("MorphoWeaveLandmarkTransfer.pose_em_registered", source)

        single_rigid = source.split("  def onAlignButton(self):", 1)[1].split(
            "  def onDisplayMeshButton(self):", 1
        )[0]
        self.assertIn("logic.estimateTransform", single_rigid)

        single_deformable = source.split("  def onCPDRegistration(self):", 1)[1].split(
            "  def onDisplayWarpedModel", 1
        )[0]
        self.assertIn("logic.runDeformable", single_deformable)

        batch = source.split("  def runLandmarkBatch(", 1)[1].split(
            "  def _tableToArray", 1
        )[0]
        self.assertIn("self.initialize_template_pose_em", batch)
        self.assertIn("RANSAC+ICP rigid", batch)
        self.assertIn("self.estimateTransform", batch)
        self.assertIn("self.runDeformable", batch)

        pose_optimizer = source.split("  def initialize_template_pose_em(", 1)[1].split(
            "  def smoothPolyData", 1
        )[0]
        self.assertIn("newCorr = ssm_sample(mean, modes, result.coefficients)", pose_optimizer)
        self.assertIn("updateMarkupsControlPointsFromArray(srcCorrNode, newCorr)", pose_optimizer)

    def test_backend_is_explicit_and_legacy_by_default(self):
        self.assertEqual(optimization_backend({}), LEGACY_BACKEND)
        self.assertEqual(optimization_backend({"optimizationBackend": "pose_em"}), POSE_EM_BACKEND)
        self.assertTrue(pose_em_enabled({"optimizationBackend": "pose_em"}))
        self.assertFalse(pose_em_enabled({"optimizationBackend": "pose_em"}, skip_optimization=True))
        self.assertFalse(pose_em_enabled({"optimizationBackend": "ransac"}))
        with self.assertRaises(ValueError):
            optimization_backend({"optimizationBackend": "unknown"})

    def test_settings_map_ui_values_to_native_rustcpd_arguments(self):
        settings = PoseEMSettings.from_mapping({
            "poseRotationCount": 24,
            "poseCoarseSourceCount": 80,
            "poseCoarseTargetCount": 90,
            "poseCoarseRank": 3,
            "poseCoarseIterations": 5,
            "poseCoarseScreenIterations": 3,
            "poseCoarseSurvivorCount": 8,
            "poseCoarseScoreMode": "final",
            "poseRefineCount": 1,
            "poseRefineSourceCount": 70,
            "poseRefineTargetCount": 120,
            "poseRefineIterations": 9,
            "poseLambdaReg": 0.03,
            "poseOutlierWeight": 0.15,
            "poseIdentityPrior": 0.25,
            "poseSeed": 7,
            "poseParallel": False,
        })
        self.assertEqual(settings.rotation_count, 24)
        self.assertEqual(settings.refine_count, 1)
        self.assertEqual(settings.seed, 7)
        self.assertEqual(settings.initializer_kwargs(), {
            "rotation_count": 24,
            "coarse_source_count": 80,
            "coarse_target_count": 90,
            "coarse_rank": 3,
            "coarse_iterations": 5,
            "coarse_screen_iterations": 3,
            "coarse_survivor_count": 8,
            "coarse_score_mode": "final",
            "refine_count": 1,
            "refine_source_count": 70,
            "refine_target_count": 120,
            "refine_iterations": 9,
            "lambda_regularization": 0.03,
            "outlier_weight": 0.15,
            "identity_prior_probability": 0.25,
            "seed": 7,
            "parallel": False,
        })

    def test_legacy_pose_worker_input_maps_to_parallel_boolean(self):
        self.assertFalse(PoseEMSettings.from_mapping({"poseNJobs": 1}).parallel)
        self.assertTrue(PoseEMSettings.from_mapping({"poseNJobs": 2}).parallel)
        self.assertFalse(
            PoseEMSettings.from_mapping({
                "poseNJobs": 4,
                "poseParallel": False,
            }).parallel
        )

    def test_defaults_match_morphoweave_rustcpd_configuration(self):
        settings = PoseEMSettings.from_mapping({})
        self.assertEqual(settings.rotation_count, 193)
        self.assertEqual(settings.coarse_screen_iterations, 8)
        self.assertEqual(settings.coarse_survivor_count, 193)
        self.assertEqual(settings.coarse_score_mode, "trajectory")
        self.assertIsNone(settings.refine_source_count)
        self.assertEqual(settings.lambda_reg, 5.0)
        self.assertEqual(settings.outlier_weight, 0.15)
        self.assertTrue(settings.parallel)

    def test_ui_uses_morphoweave_native_rustcpd_defaults(self):
        source = (MODULE_DIR / "MorphoWeaveLandmarkTransfer.py").read_text(encoding="utf-8")
        self.assertIn('poseOptL.addRow("Total pose hypotheses:", self.poseRotationCount)', source)
        self.assertIn("self.poseRotationCount.value=193", source)
        self.assertIn('"poseCoarseScoreMode": "trajectory" if scoreModeIndex == 0 else "final"', source)
        self.assertIn('self.poseRefineSourceCount.setSpecialValueText("Full source")', source)
        self.assertIn("self.poseLambdaReg.maximum=20.0", source)
        self.assertIn("self.poseLambdaReg.value=5.0", source)
        self.assertIn("self.poseOutlierWeight.value=0.15", source)
        self.assertIn("self.poseParallel.checked=True", source)
        self.assertIn('"poseParallel": bool(self.poseParallel.checked)', source)
        self.assertNotIn('"poseNJobs":', source)

    def test_parallel_setting_and_flattened_modes_are_forwarded_to_native_rustcpd(self):
        mean = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        modes = np.zeros((2, 3, 1))
        initializer = _initializer([0.0], np.eye(3), 1.0, [[0.0, 0.0, 0.0]])
        result = run_pose_em_registration(
            mean, mean, modes, np.array([1.0]),
            PoseEMSettings(rotation_count=12, parallel=True),
            with_scale=True,
            initializer=initializer,
        )
        self.assertTrue(initializer.calls[0]["parallel"])
        self.assertEqual(initializer.calls[0]["lambda_regularization"], 5.0)
        self.assertEqual(initializer.args[0][2].shape, (mean.size, 1))
        self.assertTrue(result.final_parameters["pose_parallel"])
        self.assertNotIn("blas_threads_limited", result.final_parameters)
        self.assertNotIn("dense_completion_skipped", result.final_parameters)

    def test_serial_pose_setting_is_forwarded_to_native_rustcpd(self):
        mean = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        modes = np.zeros((2, 3, 1))
        initializer = _initializer([0.0], np.eye(3), 1.0, [[0.0, 0.0, 0.0]])
        result = run_pose_em_registration(
            mean, mean, modes, np.array([1.0]),
            PoseEMSettings(rotation_count=12, parallel=False),
            with_scale=True,
            initializer=initializer,
        )
        self.assertFalse(initializer.calls[0]["parallel"])
        self.assertFalse(result.final_parameters["pose_parallel"])

    def test_template_selection_omits_redundant_dense_completion(self):
        helper = (MODULE_DIR / "Resources/Python/PoseEMTemplate.py").read_text(encoding="utf-8")
        self.assertNotIn("registration.register()", helper)
        self.assertNotIn("dense_completion_skipped", helper)
        self.assertNotIn("controller_factory", helper)
        self.assertNotIn("n_jobs", helper)
        self.assertNotIn("pose_marginalized_initialization", helper)
        self.assertIn("from rustcpd import pose_initialize", helper)

    def test_ssm_sample_uses_native_coefficients_without_sqrt_eigen_scaling(self):
        mean = np.zeros((2, 3))
        modes = np.zeros((2, 3, 2))
        modes[0, 0, 0] = 1.0
        modes[1, 1, 1] = 2.0
        sample = ssm_sample(mean, modes, np.array([3.0, -0.5]))
        np.testing.assert_allclose(sample, [[3.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

    def test_similarity_matrix_preserves_morphoweave_transform_convention(self):
        points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
        rotation = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
        translation = np.array([[4.0, -2.0, 0.5]])
        matrix = similarity_matrix(rotation, 1.5, translation)
        expected = 1.5 * (points @ rotation.T) + translation
        np.testing.assert_allclose(transform_points(points, matrix), expected, atol=1e-12)

    def test_rustcpd_direct_row_rotation_is_transposed_once_at_adapter_boundary(self):
        points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
        direct_rotation = Rotation.from_euler(
            "xyz", [18.0, -27.0, 41.0], degrees=True
        ).as_matrix().T
        translation = np.array([[4.0, -2.0, 0.5]])
        raw = SimpleNamespace(rotation=direct_rotation, marker="preserved")
        adapted = _adapt_rustcpd_initialization(raw)

        np.testing.assert_allclose(adapted.rotation, direct_rotation.T, atol=1e-12)
        self.assertEqual(adapted.marker, "preserved")
        matrix = similarity_matrix(adapted.rotation, 1.5, translation)
        expected = 1.5 * (points @ direct_rotation) + translation
        np.testing.assert_allclose(transform_points(points, matrix), expected, atol=1e-12)

    def test_coverage_prescale_uses_expected_full_target_size(self):
        source = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        target = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        self.assertAlmostEqual(coverage_prescale(source, target, 0.25), 1.0)

    def test_centered_warm_state_is_composed_back_to_world_coordinates(self):
        mean = np.array([[-1.0, -0.5, 0.25], [1.0, 0.5, -0.25]])
        modes = np.zeros((2, 3, 1)); modes[:, 2, 0] = [0.5, -0.5]
        eigenvalues = np.array([0.4])
        coefficients = np.array([0.3])
        rotation = Rotation.from_euler("xyz", [10.0, -15.0, 25.0], degrees=True).as_matrix()
        translation = np.array([[2.0, -1.0, 0.4]])
        target = 1.2 * (ssm_sample(mean, modes, coefficients) @ rotation.T) + translation
        initializer = _initializer(coefficients, rotation, 1.2, np.zeros((1, 3)))
        result = run_pose_em_registration(
            mean, target, modes, eigenvalues, PoseEMSettings(rotation_count=12),
            with_scale=True,
            initializer=initializer,
        )
        np.testing.assert_allclose(result.points, target, atol=1e-12)
        np.testing.assert_allclose(result.translation, translation, atol=1e-12)
        np.testing.assert_allclose(
            transform_points(ssm_sample(mean, modes, coefficients), result.similarity_matrix()),
            target,
            atol=1e-12,
        )
        self.assertEqual(result.hypotheses_evaluated, 61)

    def test_fixed_scale_handoff_recenters_pose_and_disables_scale_optimization(self):
        mean = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        modes = np.zeros((2, 3, 1))
        eigenvalues = np.array([1.0])
        rotation = Rotation.from_euler("z", 35.0, degrees=True).as_matrix()
        target = mean @ rotation.T + np.array([3.0, -2.0, 0.5])
        result = run_pose_em_registration(
            mean, target, modes, eigenvalues, PoseEMSettings(rotation_count=12),
            with_scale=False,
            initializer=_initializer([0.0], rotation, 0.4, [[9.0, 9.0, 9.0]]),
        )
        self.assertAlmostEqual(result.scale, 1.0)
        np.testing.assert_allclose(result.points.mean(0), target.mean(0), atol=1e-12)

    def test_partial_target_prescale_is_carried_into_fixed_scale_registration(self):
        mean = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        modes = np.zeros((2, 3, 1))
        target = np.array([[4.0, 0.0, 0.0], [4.5, 0.0, 0.0]])
        source_scale = coverage_prescale(mean, target, coverage=0.5)
        result = run_pose_em_registration(
            mean, target, modes, np.array([1.0]), PoseEMSettings(rotation_count=12),
            with_scale=False, source_scale=source_scale,
            initializer=_initializer([0.0], np.eye(3), 0.5, [[0.0, 0.0, 0.0]]),
        )
        self.assertAlmostEqual(source_scale, 0.5)
        self.assertAlmostEqual(result.scale, source_scale)
        np.testing.assert_allclose(result.points.mean(0), target.mean(0), atol=1e-12)


class PoseEMTemplateIntegrationTest(unittest.TestCase):
    def test_small_offset_target_retains_template_shape_without_redundant_completion(self):
        try:
            from rustcpd import pose_initialize
        except (ImportError, AttributeError):
            self.skipTest("rustcpd pose-EM API is unavailable")
        rng = np.random.default_rng(17)
        mean = rng.normal(size=(80, 3)) * np.array([1.0, 0.55, 0.25])
        mean[:, 0] += 0.15 * mean[:, 1] ** 2
        modes = np.zeros((80, 3, 2))
        modes[:, 0, 0] = 0.03 * mean[:, 0]
        modes[:, 1, 1] = 0.02 * mean[:, 1]
        eigenvalues = np.array([0.4, 0.2])
        rotation = Rotation.from_euler("xyz", [40.0, -30.0, 28.0], degrees=True).as_matrix()
        target = 0.08 * (mean @ rotation.T) + np.array([25.0, -12.0, 4.0])
        settings = PoseEMSettings(
            rotation_count=24,
            coarse_source_count=60,
            coarse_target_count=60,
            coarse_rank=2,
            coarse_iterations=5,
            coarse_screen_iterations=5,
            refine_count=2,
            refine_target_count=80,
            refine_iterations=10,
            seed=5,
        )
        result = run_pose_em_registration(
            mean, target, modes, eigenvalues, settings,
            with_scale=True,
        )
        selected_shape = ssm_sample(mean, modes, result.coefficients)
        self.assertGreater(np.linalg.norm(np.ptp(selected_shape, axis=0)), 0.9)
        self.assertTrue(np.isfinite(result.coefficients).all())

    def test_large_rotation_recovery_through_rustcpd_adapter(self):
        try:
            from rustcpd import pose_initialize
        except (ImportError, AttributeError):
            self.skipTest("rustcpd pose-EM API is unavailable")
        rng = np.random.default_rng(19)
        mean = rng.normal(size=(60, 3)) * np.array([2.0, 1.1, 0.6])
        mean[:, 0] += 0.2 * mean[:, 1] ** 2
        modes = np.zeros((60, 3, 2))
        modes[:, 0, 0] = 0.03 * mean[:, 0]
        modes[:, 1, 1] = 0.02 * mean[:, 1]
        eigenvalues = np.array([0.4, 0.2])
        coefficients = np.array([0.35, -0.25])
        deformed = ssm_sample(mean, modes, coefficients)
        rotation = Rotation.from_euler("xyz", [36.0, -24.0, 31.0], degrees=True).as_matrix()
        target = 1.08 * (deformed @ rotation.T) + np.array([0.8, -0.4, 0.2])
        settings = PoseEMSettings(
            rotation_count=24,
            coarse_source_count=50,
            coarse_target_count=50,
            coarse_rank=2,
            coarse_iterations=4,
            coarse_screen_iterations=4,
            refine_count=2,
            refine_target_count=60,
            refine_iterations=8,
            seed=7,
        )
        result = run_pose_em_registration(
            mean, target, modes, eigenvalues, settings,
            with_scale=True,
        )
        error = np.median(np.linalg.norm(result.points - target, axis=1))
        diagonal = np.linalg.norm(np.ptp(target, axis=0))
        self.assertLess(error / diagonal, 0.05)
        self.assertLess(result.final_parameters.get("b").size, 3)


if __name__ == "__main__":
    unittest.main()
