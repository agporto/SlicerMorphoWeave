import ast
from pathlib import Path
import tempfile
import unittest

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[2]
CORE_DIR = MODULE_DIR / "Resources" / "Python"
import sys
sys.path.insert(0, str(CORE_DIR))

from MorphoWeaveShapeCompletionCore import (
    CalibrationAccumulator,
    CompletionSettings,
    RidgeLogisticCalibrator,
    apply_similarity,
    auc_rank,
    build_local_transfer_operator,
    build_calibration_profile,
    calibrated_radius,
    calibrated_simultaneous_radius,
    calibration_method_signature,
    classify_completion_region,
    conformal_quantile,
    contiguous_plane_mask,
    coordinate_compatibility,
    finalize_calibration_entry,
    indexed_correspondence_compatibility,
    infer_control_vertex_indices,
    landmark_recommendation,
    load_calibration_profile,
    local_displacement_interpolate,
    local_scalar_interpolate,
    match_landmarks,
    model_fingerprint,
    nonconformity_scores,
    pose_residual_interpolate,
    run_completion,
    save_calibration_profile,
    select_calibration_entry,
    validate_and_truncate_ssm,
)


def synthetic_ssm(m=12, k=4):
    x = np.linspace(-1.0, 1.0, m)
    mean = np.column_stack((x, x * x + 0.2 * x, np.sin(2.0 * x)))
    rng = np.random.default_rng(7)
    modes, _ = np.linalg.qr(rng.normal(size=(3 * m, k)))
    eigenvalues = np.array([4.0, 2.0, 1.0, 0.5])[:k]
    return mean, modes.reshape(m, 3, k), eigenvalues


class _Posterior:
    def __init__(self, points, rank):
        self._points = np.asarray(points)
        self._cov = np.eye(rank) * 0.02
        self.noise_variance = 0.01
        self.discrepancy_variance = 0.03

    @property
    def coefficient_covariance(self):
        return self._cov

    @property
    def coefficient_mean(self):
        return np.zeros(len(self._cov))

    def predict(self):
        return self._points

    def predictive_variance(self):
        return np.linspace(0.02, 0.08, len(self._points))

    def sample_shapes(self, count, seed):
        rng = np.random.default_rng(seed)
        return [self._points + rng.normal(scale=0.01, size=self._points.shape) for _ in range(count)]


class _Fit:
    def __init__(self, mean, rank, kwargs):
        self.points = np.asarray(mean) + 0.1
        self.coefficients = np.zeros(rank)
        self.rotation = np.eye(3)
        self.scale = 1.0
        self.translation = np.array([0.1, 0.1, 0.1])
        self.sigma2 = 0.02
        self.iterations = 17
        self.difference = 1e-8
        self.landmark_rms = 0.03 if "landmark_sigma" in kwargs else float("nan")
        self.posterior_kwargs = None

    def posterior(self, target, mean, modes, eigenvalues, **kwargs):
        self.posterior_kwargs = kwargs
        return _Posterior(self.points, len(eigenvalues))


class _Pose:
    def __init__(self, rank):
        self.coefficients = np.zeros(rank)
        self.rotation = np.eye(3)
        self.scale = 1.0
        self.translation = np.zeros(3)
        self.score = 3.0
        self.score_margin = 2.0
        self.posterior_entropy = 0.2
        self.effective_hypotheses = 1.2
        self.hypotheses_evaluated = 193
        self.hypotheses_refined = 8


class _FakeRustCPD:
    def __init__(self):
        self.pose_call = None
        self.atlas_call = None
        self.fit = None

    def pose_initialize(self, source, target, modes, eigenvalues, **kwargs):
        self.pose_call = (np.asarray(source), np.asarray(target), np.asarray(modes), dict(kwargs))
        return _Pose(len(eigenvalues))

    def register_atlas(self, target, mean, modes, eigenvalues, **kwargs):
        self.atlas_call = (np.asarray(target), np.asarray(mean), np.asarray(modes), dict(kwargs))
        self.fit = _Fit(mean, len(eigenvalues), kwargs)
        return self.fit


class _DivergentPosterior(_Posterior):
    def __init__(self, points, rank):
        super().__init__(np.asarray(points) + np.array([5.0, -3.0, 2.0]), rank)


class _DivergentFit(_Fit):
    def posterior(self, target, mean, modes, eigenvalues, **kwargs):
        self.posterior_kwargs = kwargs
        return _DivergentPosterior(self.points, len(eigenvalues))


class _DivergentRustCPD(_FakeRustCPD):
    def register_atlas(self, target, mean, modes, eigenvalues, **kwargs):
        self.atlas_call = (np.asarray(target), np.asarray(mean), np.asarray(modes), dict(kwargs))
        self.fit = _DivergentFit(mean, len(eigenvalues), kwargs)
        return self.fit


class ShapeCompletionCoreUnitTest(unittest.TestCase):
    def test_defaults_and_landmark_recommendation(self):
        settings = CompletionSettings()
        self.assertEqual(settings.coverage, 1.0)
        self.assertTrue(settings.with_scale())
        settings.coverage = 0.60
        self.assertFalse(settings.with_scale())
        state, message = landmark_recommendation(0.74, 0)
        self.assertEqual(state, "recommended")
        self.assertIn("below 0.75", message)
        self.assertEqual(landmark_recommendation(0.75, 0)[0], "optional")
        self.assertEqual(landmark_recommendation(0.60, 3)[0], "available")

    def test_ssm_validation_and_truncation(self):
        mean, modes, eigenvalues = synthetic_ssm()
        ssm = validate_and_truncate_ssm(mean, modes, eigenvalues, 0.75)
        self.assertEqual(ssm.retained_modes, 2)
        self.assertGreaterEqual(ssm.retained_variance, 0.75)
        self.assertEqual(ssm.modes_flat.shape, (mean.size, 2))
        with self.assertRaises(ValueError):
            validate_and_truncate_ssm(mean, modes, np.array([1, 0, 1, 1]), 1.0)

    def test_label_matching_and_dense_index_mapping(self):
        dense = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]
        ], dtype=float)
        source = dense[[0, 1, 2, 3]] + 1e-4
        target = source + np.array([2, 3, 4])
        match = match_landmarks(
            ["A", "B", "C", "D"], source,
            ["D", "B", "A", "C"], target[[3, 1, 0, 2]], dense
        )
        self.assertEqual(match.count, 4)
        self.assertEqual(match.labels, ("A", "B", "C", "D"))
        np.testing.assert_array_equal(match.source_indices, [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            match_landmarks(["", "", "", ""], source, ["", "", "", ""], target, dense)
        ordered = match_landmarks(
            ["", "", "", ""], source, ["", "", "", ""], target, dense,
            allow_order_fallback=True,
        )
        self.assertTrue(ordered.used_order_fallback)

    def test_collinear_landmarks_rejected(self):
        dense = np.column_stack((np.arange(5), np.zeros(5), np.zeros(5)))
        with self.assertRaisesRegex(ValueError, "collinear"):
            match_landmarks(
                ["a", "b", "c"], dense[:3], ["a", "b", "c"], dense[:3] + 2, dense
            )

    def test_pipeline_forwards_one_physical_landmark_sigma_everywhere(self):
        mean, modes, eigenvalues = synthetic_ssm(m=30)
        target = mean[:24] + np.array([0.3, -0.2, 0.4])
        source_points = mean[[0, 4, 8, 11]]
        target_points = source_points + np.array([0.3, -0.2, 0.4])
        landmarks = match_landmarks(
            ["a", "b", "c", "d"], source_points,
            ["a", "b", "c", "d"], target_points,
            mean,
        )
        settings = CompletionSettings(
            coverage=0.60,
            use_landmarks=True,
            coverage_prescale=False,
            target_point_count=100,
            posterior_samples=2,
            atlas_lambda=7.5,
        )
        fake = _FakeRustCPD()
        result = run_completion(
            target, mean, modes, eigenvalues, settings,
            rustcpd_module=fake, landmarks=landmarks,
        )
        pose_kwargs = fake.pose_call[3]
        atlas_kwargs = fake.atlas_call[3]
        self.assertFalse(pose_kwargs["with_scale"])
        self.assertFalse(atlas_kwargs["with_scale"])
        self.assertIn("landmark_sigma", pose_kwargs)
        self.assertEqual(pose_kwargs["landmark_sigma"], pose_kwargs["refine_landmark_sigma"])
        self.assertEqual(pose_kwargs["landmark_sigma"], atlas_kwargs["landmark_sigma"])
        self.assertNotIn("landmark_weight", pose_kwargs)
        self.assertAlmostEqual(fake.fit.posterior_kwargs["prior_temperature"], 1.0 / 7.5)
        self.assertEqual(len(result.samples), 2)
        self.assertEqual(result.completed_points.shape, mean.shape)
        self.assertTrue(np.all(result.epistemic_variance >= 0))
        self.assertGreater(result.normalized_sigma2, 0)


    def test_constrained_atlas_remains_the_completion_point_estimate(self):
        mean, modes, eigenvalues = synthetic_ssm(m=30)
        settings = CompletionSettings(coverage=0.60, target_point_count=100, posterior_samples=2)
        fake = _DivergentRustCPD()
        result = run_completion(
            mean[:24], mean, modes, eigenvalues, settings, rustcpd_module=fake
        )
        expected = np.asarray(fake.fit.points) + result.target_centroid
        np.testing.assert_allclose(result.completed_points, expected)
        self.assertGreater(result.posterior_mean_rms_from_atlas, 1.0)
        self.assertFalse(np.allclose(result.posterior_mean_points, result.completed_points))
        # Samples are recentered on the constrained atlas estimate while keeping
        # the posterior covariance/noise realization around that center.
        sample_mean = np.mean(np.stack(result.samples), axis=0)
        self.assertLess(float(np.mean(np.linalg.norm(sample_mean - expected, axis=1))), 0.05)

    def test_single_neighbor_interpolation_uses_the_nearest_control_exactly(self):
        source = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        target = source + np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        query = np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0], [1000.0, 0.0, 0.0]])
        interpolated = local_displacement_interpolate(
            query, source, target, neighbors=1
        )
        np.testing.assert_allclose(interpolated[0], query[0] + np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(interpolated[1:], query[1:] + np.array([-1.0, -2.0, -3.0]))

    def test_unavailable_pointwise_calibration_is_not_applied(self):
        self.assertIsNone(calibrated_radius(np.ones(4), {"conformal_scale": None}))

    def test_reciprocal_temperature_matches_atlas_normal_equation(self):
        rng = np.random.default_rng(31)
        rows, rank = 24, 4
        modes = rng.normal(size=(rows, rank))
        weights = rng.uniform(0.2, 1.0, size=rows)
        residual = rng.normal(size=rows)
        eigenvalues = np.array([3.0, 1.7, 0.8, 0.4])
        sigma_eff2 = 0.35
        regularization = 6.0
        data = modes.T @ (weights[:, None] * modes)
        rhs = modes.T @ (weights * residual)
        atlas_coefficients = np.linalg.solve(
            data + regularization * sigma_eff2 * np.diag(1.0 / eigenvalues),
            rhs,
        )
        temperature = 1.0 / regularization
        posterior_precision = (
            data / sigma_eff2
            + np.diag(1.0 / (temperature * eigenvalues))
        )
        posterior_coefficients = np.linalg.solve(
            posterior_precision, rhs / sigma_eff2
        )
        np.testing.assert_allclose(
            posterior_coefficients, atlas_coefficients, rtol=1e-12, atol=1e-12
        )

    def test_landmark_vertex_mapping_is_included_in_effective_sigma(self):
        mean, modes, eigenvalues = synthetic_ssm(m=30)
        indices = np.array([0, 4, 8, 11])
        source_points = mean[indices] + np.array([0.0, 0.03, 0.0])
        target_points = source_points + np.array([0.3, -0.2, 0.4])
        landmarks = match_landmarks(
            ["a", "b", "c", "d"], source_points,
            ["a", "b", "c", "d"], target_points,
            mean,
        )
        settings = CompletionSettings(
            coverage=0.60,
            use_landmarks=True,
            coverage_prescale=False,
            target_point_count=100,
        )
        fake = _FakeRustCPD()
        result = run_completion(
            mean[:24], mean, modes, eigenvalues, settings,
            rustcpd_module=fake, landmarks=landmarks,
        )
        self.assertGreater(result.landmark_mapping_rms, 0.0)
        self.assertGreater(result.landmark_sigma, result.landmark_nominal_sigma)
        self.assertEqual(fake.pose_call[3]["landmark_sigma"], result.landmark_sigma)
        self.assertEqual(fake.atlas_call[3]["landmark_sigma"], result.landmark_sigma)

    def test_indexed_ssm_template_compatibility_detects_mixed_quartets(self):
        mean, _, _ = synthetic_ssm()
        report = indexed_correspondence_compatibility(mean, mean + 1e-4)
        self.assertLess(report["rms_fraction"], 0.01)
        with self.assertRaisesRegex(ValueError, "canonical Model Library quartet"):
            indexed_correspondence_compatibility(mean, mean * 4.0 + 10.0)

    def test_tiny_fragments_are_rejected_before_native_registration(self):
        mean, modes, eigenvalues = synthetic_ssm()
        with self.assertRaisesRegex(ValueError, "at least 20"):
            run_completion(
                mean[:10], mean, modes, eigenvalues,
                CompletionSettings(target_point_count=100),
                rustcpd_module=_FakeRustCPD(),
            )

    def test_coordinate_compatibility_rejects_gross_frame_mismatch(self):
        surface = np.random.default_rng(2).normal(size=(100, 3))
        close = surface[::10] + 1e-4
        report = coordinate_compatibility(surface, close, "close points")
        self.assertLess(report["median_fraction"], 0.01)
        with self.assertRaisesRegex(ValueError, "different coordinate frame"):
            coordinate_compatibility(surface, close + 100.0, "shifted points")

    def test_surface_only_pipeline_does_not_pass_landmark_arguments(self):
        mean, modes, eigenvalues = synthetic_ssm(m=30)
        settings = CompletionSettings(
            coverage=1.0, use_landmarks=False, target_point_count=100
        )
        fake = _FakeRustCPD()
        run_completion(mean, mean, modes, eigenvalues, settings, rustcpd_module=fake)
        for kwargs in (fake.pose_call[3], fake.atlas_call[3]):
            self.assertNotIn("landmark_sigma", kwargs)
            self.assertNotIn("landmark_indices", kwargs)
        self.assertTrue(fake.pose_call[3]["with_scale"])

    def test_landmark_mode_never_silently_falls_back_to_surface_only(self):
        mean, modes, eigenvalues = synthetic_ssm(m=30)
        settings = CompletionSettings(
            coverage=0.60, use_landmarks=True, target_point_count=100
        )
        with self.assertRaisesRegex(ValueError, "no matched landmarks"):
            run_completion(
                mean[:20], mean, modes, eigenvalues, settings,
                rustcpd_module=_FakeRustCPD(),
            )

    def test_local_interpolation_exact_at_correspondences(self):
        source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        target = source + np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        warped = local_displacement_interpolate(source, source, target, neighbors=3)
        np.testing.assert_allclose(warped, target)
        scalar = local_scalar_interpolate(source, source, [1.0, 2.0, 3.0], neighbors=3)
        np.testing.assert_allclose(scalar, [1.0, 2.0, 3.0])

    def test_pose_residual_transfer_reproduces_pure_similarity_exactly(self):
        rng = np.random.default_rng(22)
        query = rng.normal(size=(80, 3))
        source = query[[0, 5, 11, 17, 23, 31, 44, 57, 66, 79]]
        angle = np.deg2rad(63.0)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        scale = 1.7
        translation = np.array([2.0, -4.0, 0.5])
        completed_source = apply_similarity(source, scale, rotation, translation)
        operator = build_local_transfer_operator(
            query, source, neighbors=5, sharpness=2.0, chunk_size=17
        )
        transferred = pose_residual_interpolate(
            query,
            source,
            completed_source,
            operator=operator,
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        np.testing.assert_allclose(
            transferred,
            apply_similarity(query, scale, rotation, translation),
            rtol=0.0,
            atol=1e-12,
        )

    def test_sparse_operator_batches_fields_and_anchors_controls(self):
        query = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        source = query[[0, 2, 3, 4]]
        exact, nearest, distances, tolerance = infer_control_vertex_indices(query, source)
        np.testing.assert_array_equal(exact, [0, 2, 3, 4])
        np.testing.assert_array_equal(nearest, exact)
        self.assertTrue(np.all(distances <= tolerance))
        operator = build_local_transfer_operator(
            query,
            source,
            neighbors=3,
            sharpness=2.0,
            chunk_size=2,
            exact_source_vertex_indices=exact,
        )
        fields = np.column_stack(
            (
                np.arange(len(source), dtype=np.float32),
                np.arange(len(source), dtype=np.float32) ** 2,
                np.ones(len(source), dtype=np.float32),
            )
        )
        transferred = operator.apply(fields)
        np.testing.assert_allclose(transferred[exact], fields)
        self.assertEqual(operator.exact_anchor_count, len(source))
        self.assertEqual(operator.matrix.data.dtype, np.float32)
        self.assertEqual(operator.matrix.indices.dtype, np.int32)

    def test_component_restriction_prevents_cross_component_blending(self):
        source = np.array([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]])
        query = np.array([[0.03, 0.0, 0.0], [0.05, 0.0, 0.0]])
        operator = build_local_transfer_operator(
            query,
            source,
            neighbors=2,
            sharpness=1.0,
            query_groups=np.array([0, 1]),
            source_groups=np.array([0, 1]),
        )
        np.testing.assert_allclose(operator.apply(np.array([1.0, 9.0])), [1.0, 9.0])
        self.assertEqual(operator.component_count, 2)
        self.assertEqual(operator.component_fallback_query_count, 0)

        fallback = build_local_transfer_operator(
            query,
            source[:1],
            neighbors=1,
            query_groups=np.array([0, 1]),
            source_groups=np.array([0]),
        )
        np.testing.assert_allclose(fallback.apply(np.array([4.0])), [4.0, 4.0])
        self.assertEqual(fallback.component_fallback_query_count, 1)

    def test_duplicate_exact_control_vertices_are_rejected(self):
        mesh = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        controls = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "same template vertex"):
            infer_control_vertex_indices(mesh, controls)

    def test_plane_mask_is_deterministic_and_exact_count(self):
        points = np.random.default_rng(0).normal(size=(101, 3))
        first = contiguous_plane_mask(points, 0.60, np.random.default_rng(42))
        second = contiguous_plane_mask(points, 0.60, np.random.default_rng(42))
        np.testing.assert_array_equal(first.indices, second.indices)
        self.assertEqual(len(first.indices), round(0.60 * len(points)))
        self.assertEqual(int(first.contains(points).sum()), len(first.indices))

    def test_region_display_mask_has_requested_count(self):
        complete = np.column_stack((np.arange(10), np.zeros(10), np.zeros(10)))
        fragment = complete[:4]
        region, distance = classify_completion_region(complete, fragment, 0.4)
        self.assertEqual(int((region == 0).sum()), 4)
        self.assertEqual(distance.shape, (10,))

    def test_conformal_quantile_and_calibrated_radii(self):
        errors = np.array([1, 2, 3, 4, 5], dtype=float)
        variances = np.ones(5)
        q = conformal_quantile(nonconformity_scores(errors, variances), 0.20)
        self.assertEqual(q, 5.0)
        entry = {
            "conformal_scale": 2.0,
            "simultaneous_meshwise_conformal_scale": 4.0,
        }
        radii = calibrated_radius(np.array([1.0, 4.0]), entry)
        np.testing.assert_allclose(radii, [2.0, 4.0])
        simultaneous = calibrated_simultaneous_radius(np.array([1.0, 4.0]), entry)
        np.testing.assert_allclose(simultaneous, [4.0, 8.0])
        self.assertIsNone(calibrated_simultaneous_radius(
            np.array([1.0]), {"simultaneous_meshwise_conformal_scale": None}
        ))

    def test_ridge_logistic_remains_finite_under_near_separation(self):
        sigma = np.r_[np.geomspace(1e-6, 1e-3, 30), np.geomspace(1e-1, 10, 30)]
        success = np.r_[np.ones(30, dtype=bool), np.zeros(30, dtype=bool)]
        model = RidgeLogisticCalibrator.fit(sigma, success, l2=1.0)
        probability = model.probability(sigma)
        self.assertTrue(np.all(np.isfinite(probability)))
        self.assertGreater(float(probability[:30].mean()), 0.9)
        self.assertLess(float(probability[30:].mean()), 0.1)
        self.assertGreater(auc_rank(-np.log(sigma), success), 0.99)

    def test_profile_fingerprint_mode_and_coverage_gating(self):
        mean, modes, eigenvalues = synthetic_ssm()
        fingerprint = model_fingerprint(mean, modes, eigenvalues)
        profile = build_calibration_profile(
            fingerprint=fingerprint,
            entries=[{
                "coverage": 0.60,
                "landmark_mode": True,
                "truth_type": "exact_dense_correspondence",
                "nominal_coverage": 0.9,
                "conformal_scale": 2.0,
            }],
            generated_from_ssm_training_set=True,
            source_description="test",
            settings_snapshot=CompletionSettings().__dict__,
        )
        selected, _ = select_calibration_entry(
            profile, fingerprint=fingerprint, coverage=0.62, landmark_mode=True,
            current_settings=CompletionSettings(),
        )
        self.assertIsNotNone(selected)
        self.assertIsNone(select_calibration_entry(
            profile, fingerprint="wrong", coverage=0.60, landmark_mode=True
        )[0])
        self.assertIsNone(select_calibration_entry(
            profile, fingerprint=fingerprint, coverage=0.60, landmark_mode=False
        )[0])
        self.assertIsNone(select_calibration_entry(
            profile, fingerprint=fingerprint, coverage=0.80, landmark_mode=True
        )[0])
        changed = CompletionSettings(atlas_lambda=9.0)
        self.assertIsNone(select_calibration_entry(
            profile, fingerprint=fingerprint, coverage=0.60, landmark_mode=True,
            current_settings=changed,
        )[0])
        changed_seed = CompletionSettings(seed=19)
        self.assertIsNone(select_calibration_entry(
            profile, fingerprint=fingerprint, coverage=0.60, landmark_mode=True,
            current_settings=changed_seed,
        )[0])
        self.assertEqual(
            profile["method_signature_sha256"],
            calibration_method_signature(CompletionSettings()),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            save_calibration_profile(profile, path)
            loaded = load_calibration_profile(path)
            self.assertEqual(loaded["model_fingerprint_sha256"], fingerprint)
            self.assertTrue(loaded["generated_from_ssm_training_set"])

    def test_calibration_entry_reports_scope_and_probability_status(self):
        accumulator = CalibrationAccumulator(0.6, False, "surface_distance")
        for index in range(10):
            error = np.array([0.01, 0.02, 0.03]) * (1 + index / 20)
            variance = np.array([0.01, 0.01, 0.01])
            success = index < 5
            accumulator.add(
                fit_id=str(index), group=f"mesh-{index}", errors=error,
                variances=variance, normalized_sigma2=0.001 if success else 0.1,
                successful=success, missing_rms=float(error.mean()),
            )
        entry = finalize_calibration_entry(
            accumulator, alpha=0.1, success_threshold_fraction=0.03
        )
        self.assertEqual(entry["fit_count"], 10)
        self.assertIn("held_out_mesh_audit", entry)
        self.assertIsNotNone(entry["fit_success_calibrator"])
        self.assertGreater(entry["conformal_scale"], 0)
        self.assertEqual(entry["conformal_scope"], "marginal_pointwise_euclidean_error")
        self.assertIsNotNone(entry["simultaneous_meshwise_conformal_scale"])
        self.assertIn("meshwise_simultaneous_empirical_coverage", entry["held_out_mesh_audit"])

    def test_main_module_contains_required_ui_and_no_unconstrained_cpd(self):
        source_path = MODULE_DIR / "MorphoWeaveShapeCompletion.py"
        source = source_path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(source_path))
        self.assertIn("Target coverage:", source)
        self.assertIn("below 0.75", source)
        self.assertIn("Optional Calibration Profile", source)
        self.assertIn("Pure Atlas Registration", source)
        self.assertNotIn("register_deformable(", source)
        resolver = source.split("    def _auto_select_canonical_ssm_set", 1)[1].split(
            "    def _validate_complete_inputs", 1
        )[0]
        self.assertNotIn("target_model_selector", resolver)
        self.assertNotIn("target_landmark_selector", resolver)


if __name__ == "__main__":
    unittest.main()
