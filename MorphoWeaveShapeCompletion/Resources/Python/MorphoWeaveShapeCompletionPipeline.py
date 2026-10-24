"""Robust fragment-completion registration (pipeline="robust").

Validated on held-out mouse skulls, molars and lizard vertebrae (see the
benchmark report shipped with this change). Compared with the legacy path it

1. estimates the fragment size from the convex-hull volume (robust to vertex
   noise and mesh resolution) combined with the coverage-consistent radii of
   the legacy path, and bounds the residual scale tightly around it;
2. runs rustcpd's pose-marginalized search with adaptive mixing and continues
   the atlas fit from its full state (pose, coefficients, variance, mixture);
3. widens the global search without feature matching: several independent
   pose-search lattices (seeds) plus "SSM region" proposals that align the
   fragment's principal axes with coverage-consistent cuts of the mean shape
   resembling it; every candidate is refined and the best penalized CPD
   likelihood wins. (Optional FPFH + RANSAC proposals are kept but off: on
   held-out data the SSM-only search matched the FPFH hybrid, 22.4% vs 24.3%
   failed fragments.)
4. optionally re-fits on a density-matched resampling of the fragment so
   model points do not slide toward over-sampled regions;
5. builds the posterior with the prior precision matched to the atlas fit
   (rustcpd >= 4: prior_temperature == lambda_regularization).
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

_TINY = 1e-12


def convex_hull_volume(points: np.ndarray) -> float:
    from scipy.spatial import ConvexHull

    return float(ConvexHull(np.asarray(points, dtype=np.float64)).volume)


def hull_scale(fragment: np.ndarray, template_points: np.ndarray, coverage: float) -> float:
    """Scale s with hull_volume(fragment) = coverage * hull_volume(s * template)."""
    return float(np.cbrt(convex_hull_volume(fragment) / (float(coverage) * convex_hull_volume(template_points))))


def penalized_nll(target, fitted, sigma2, outlier_weight, mixing, coefficients, eigenvalues, lam) -> float:
    """CPD mixture negative log-likelihood of the target under a fitted model,
    plus the shape prior 0.5*lam*b'L^-1 b. Comparable across candidates that
    share the same target cloud."""
    x = np.asarray(target, dtype=np.float64)
    y = np.asarray(fitted, dtype=np.float64)
    m, d = y.shape
    sigma2 = max(float(sigma2), _TINY)
    w = float(outlier_weight)
    if mixing is None:
        log_pi = np.full(m, -math.log(m))
    else:
        log_pi = np.log(np.maximum(np.asarray(mixing, dtype=np.float64), 1e-300))
    volume = max(float(np.prod(np.ptp(x, axis=0))), _TINY)
    out = (math.log(w) - math.log(volume)) if w > 0 else -np.inf
    base = math.log(max(1.0 - w, _TINY)) - 0.5 * d * math.log(2.0 * math.pi * sigma2)
    total = 0.0
    for start in range(0, len(x), 512):
        xs = x[start:start + 512]
        d2 = np.sum((xs[:, None, :] - y[None]) ** 2, axis=2)
        comp = base + log_pi[None, :] - d2 / (2.0 * sigma2)
        top = np.maximum(comp.max(axis=1), out)
        total -= float(np.sum(top + np.log(np.exp(comp - top[:, None]).sum(axis=1) + np.exp(out - top))))
    b = np.asarray(coefficients, dtype=np.float64)
    return total + 0.5 * float(lam) * float(np.sum(b * b / np.asarray(eigenvalues, dtype=np.float64)))


def fpfh_pose_proposals(source_points, target_points, *, hypotheses=16, iterations=100_000,
                        voxel_fraction=1.0 / 40.0, seed=0):
    """Rigid proposals (R, t), row convention target ~ source @ R + t, from FPFH + RANSAC."""
    import tiny3d as t3d

    source_points = np.asarray(source_points, dtype=np.float64)
    voxel = float(np.linalg.norm(np.ptp(source_points, axis=0))) * float(voxel_fraction)
    reg = t3d.pipelines.registration

    def features(points):
        cloud = t3d.geometry.PointCloud()
        cloud.points = t3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        down = cloud.voxel_down_sample(voxel)
        down.estimate_normals(t3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel, max_nn=30))
        fpfh = reg.compute_fpfh_feature(down, t3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel, max_nn=100))
        return down, fpfh

    source_down, source_fpfh = features(source_points)
    target_down, target_fpfh = features(target_points)
    proposals = []
    for index in range(int(hypotheses)):
        t3d.utility.random.seed(int(seed) + 7919 * index)
        result = reg.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, False, 1.5 * voxel,
            reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(1.5 * voxel)],
            reg.RANSACConvergenceCriteria(int(iterations), 0.999),
        )
        transform = np.asarray(result.transformation, dtype=np.float64)
        proposals.append((transform[:3, :3].T.copy(), transform[:3, 3].copy()))
    # Drop proposals that place the source within 2% of the diagonal of an earlier one.
    sample = source_points[:: max(1, len(source_points) // 500)]
    diagonal = float(np.linalg.norm(np.ptp(source_points, axis=0)))
    kept, placed = [], []
    for rotation, translation in proposals:
        posed = sample @ rotation + translation
        if all(np.sqrt(np.mean(np.sum((posed - other) ** 2, axis=1))) > 0.02 * diagonal for other in placed):
            kept.append((rotation, translation))
            placed.append(posed)
    return kept


def region_pose_proposals(model, target, coverage, *, regions=16, directions=64, fibonacci=None):
    """Pose hypotheses (R, t), row convention target ~ model @ R + t, without features.

    The mean shape is cut by planes in `directions` directions keeping the
    requested coverage; the `regions` cuts whose scale-free covariance spectrum
    best matches the fragment's are aligned to it by principal axes (four
    proper sign combinations each).
    """
    model = np.asarray(model, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    def frame(points):
        centre = points.mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov((points - centre).T))
        return centre, vectors[:, ::-1], values[::-1]

    count = max(3, int(round(float(coverage) * len(model))))
    target_centre, target_axes, target_values = frame(target)
    target_spectrum = np.log(np.maximum(target_values[1:], _TINY) / max(target_values[0], _TINY))
    cuts = []
    for normal in fibonacci(int(directions)):
        order = np.argsort(model @ normal, kind="mergesort")
        for indices in (order[:count], order[-count:]):
            centre, axes, values = frame(model[indices])
            spectrum = np.log(np.maximum(values[1:], _TINY) / max(values[0], _TINY))
            cuts.append((float(np.linalg.norm(spectrum - target_spectrum)), centre, axes))
    cuts.sort(key=lambda item: item[0])
    proposals = []
    for _, centre, axes in cuts[: int(regions)]:
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            rotation = axes @ np.diag([sx, sy, sx * sy]) @ target_axes.T
            if np.linalg.det(rotation) < 0:
                rotation = axes @ np.diag([sx, sy, -sx * sy]) @ target_axes.T
            proposals.append((rotation, target_centre - centre @ rotation))
    return proposals


def density_matched_target(surface_points, fitted_points, count, seed):
    """Resample the observed surface so its density follows the fitted model points'."""
    from scipy.spatial import cKDTree

    fitted_points = np.asarray(fitted_points, dtype=np.float64)
    spacing = float(np.median(cKDTree(fitted_points).query(fitted_points, k=2)[0][:, 1]))
    neighbours = cKDTree(fitted_points).query_ball_point(surface_points, 2.0 * spacing)
    density = np.array([len(item) for item in neighbours], dtype=np.float64) + 0.5
    rng = np.random.default_rng(int(seed))
    count = min(int(count), len(surface_points))
    chosen = rng.choice(len(surface_points), count, replace=False, p=density / density.sum())
    return surface_points[chosen]


def fibonacci_directions(count: int) -> np.ndarray:
    index = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden = np.pi * (3.0 - np.sqrt(5.0))
    return np.column_stack((radius * np.cos(golden * index), radius * np.sin(golden * index), z))


def global_pose_search(
    rustcpd_module: Any,
    working_mean: np.ndarray,
    working_target: np.ndarray,
    working_flat: np.ndarray,
    eigenvalues: np.ndarray,
    *,
    coverage: float = 1.0,
    bounds: Optional[tuple] = (0.9, 1.0 / 0.9),
    outlier_weight: float = 0.10,
    adaptive_mixing: Optional[float] = 1.0,
    pose_lambda: float = 5.0,
    atlas_lambda: float = 5.0,
    rotation_count: int = 193,
    coarse_source_count: int = 350,
    coarse_target_count: int = 350,
    coarse_rank: int = 8,
    coarse_iterations: int = 8,
    coarse_screen_iterations: int = 8,
    coarse_survivor_count: int = 193,
    coarse_score_mode: str = "trajectory",
    refine_count: int = 8,
    refine_target_count: int = 1200,
    refine_iterations: int = 25,
    identity_prior_probability: float = 0.20,
    translation_anchor_count: int = 8,
    anchor_completeness_threshold: float = 0.9,
    merge_tolerance: float = 0.02,
    atlas_max_iterations: int = 150,
    atlas_tolerance: float = 1e-7,
    atlas_k: Optional[int] = None,
    seeds: int = 4,
    region_proposals: int = 16,
    screen_iterations: int = 30,
    screen_points: int = 400,
    refine_proposals: int = 2,
    seed: int = 0,
    parallel: bool = True,
    single_precision: bool = False,
    landmark_kwargs: Optional[dict] = None,
    pose_landmark_kwargs: Optional[dict] = None,
    extra_proposals: Any = None,
) -> dict:
    """SSM-only global pose search shared by Shape Completion and Landmark Transfer.

    Model `working_mean + working_flat @ b` is posed onto `working_target` (both
    centred, same units; the model pre-scaled so that the residual scale lies in
    `bounds`). Candidates come from `seeds` independent pose-marginalized searches
    and from coverage-consistent SSM-region proposals (for complete targets: the
    four principal-axis alignments); `extra_proposals` may add (R, t) hypotheses
    (e.g. FPFH + RANSAC). Proposals are screened on subsampled clouds, the best
    `refine_proposals` are refined, and the fit with the lowest penalized CPD
    negative log-likelihood is returned together with the atlas options used.
    """
    landmark_kwargs = dict(landmark_kwargs or {})
    pose_landmark_kwargs = dict(pose_landmark_kwargs or {})
    coverage = float(coverage)
    is_fragment = coverage < 0.95
    rank = working_flat.shape[1]
    common = dict(
        outlier_weight=float(outlier_weight),
        with_scale=bounds is not None,
        scale_bounds=bounds,
        adaptive_mixing=adaptive_mixing,
    )
    pose_kwargs = dict(
        rotation_count=int(rotation_count),
        coarse_source_count=min(int(coarse_source_count), len(working_mean)),
        coarse_target_count=min(int(coarse_target_count), len(working_target)),
        coarse_rank=min(int(coarse_rank), rank),
        coarse_iterations=int(coarse_iterations),
        coarse_screen_iterations=int(coarse_screen_iterations),
        coarse_survivor_count=min(int(coarse_survivor_count), int(rotation_count)),
        coarse_score_mode=str(coarse_score_mode),
        refine_count=min(int(refine_count), int(rotation_count)),
        refine_target_count=min(int(refine_target_count), len(working_target)),
        refine_iterations=int(refine_iterations),
        lambda_regularization=float(pose_lambda),
        identity_prior_probability=float(identity_prior_probability),
        translation_anchor_count=int(translation_anchor_count) if is_fragment else 1,
        anchor_completeness_threshold=float(anchor_completeness_threshold),
        merge_tolerance=float(merge_tolerance),
        parallel=bool(parallel),
        single_precision=bool(single_precision),
        **common,
        **pose_landmark_kwargs,
    )
    atlas_kwargs = dict(
        common,
        lambda_regularization=float(atlas_lambda),
        normalize=True,
        max_iterations=int(atlas_max_iterations),
        tolerance=float(atlas_tolerance),
        k=None if atlas_k is None else int(atlas_k),
        parallel=bool(parallel),
        single_precision=bool(single_precision),
        **landmark_kwargs,
    )

    def score(candidate, target=working_target):
        return penalized_nll(target, candidate.points, candidate.sigma2, outlier_weight,
                             getattr(candidate, "mixing_weights", None), candidate.coefficients,
                             eigenvalues, atlas_lambda)

    seeds = max(1, int(seeds))
    pose, fit, best = None, None, np.inf
    for index in range(seeds):
        trial_pose = rustcpd_module.pose_initialize(
            working_mean, working_target, working_flat, eigenvalues, seed=int(seed) + 1000 * index, **pose_kwargs,
        )
        trial = rustcpd_module.register_atlas(
            working_target, working_mean, working_flat, eigenvalues, initial_state=trial_pose.state, **atlas_kwargs
        )
        value = score(trial) if seeds > 1 else 0.0
        if pose is None or value < best:
            pose, fit, best = trial_pose, trial, value
    chosen, candidates = "pose_search", seeds
    proposals = []
    if not landmark_kwargs and int(region_proposals) > 0:
        # For complete targets every "cut" is the whole model, so this reduces to the
        # four proper principal-axis alignments (guards against near-symmetric flips).
        proposals += region_pose_proposals(
            working_mean, working_target, coverage if is_fragment else 1.0,
            regions=int(region_proposals) if is_fragment else 1,
            fibonacci=fibonacci_directions,
        )
    if extra_proposals is not None:
        try:
            proposals += list(extra_proposals() if callable(extra_proposals) else extra_proposals)
        except ImportError:
            pass
    if proposals:
        best = score(fit)
        # Screen on subsampled clouds (as the pose search's coarse stage does),
        # then refine the best few at full resolution.
        rng = np.random.default_rng(int(seed))
        mean_idx = np.sort(rng.choice(len(working_mean), min(int(screen_points), len(working_mean)), replace=False))
        target_idx = np.sort(rng.choice(len(working_target), min(int(screen_points), len(working_target)), replace=False))
        screen_mean, screen_target = working_mean[mean_idx], working_target[target_idx]
        screen_modes = working_flat.reshape(len(working_mean), 3, -1)[mean_idx].reshape(3 * len(mean_idx), -1)
        screened = []
        for rotation, translation in proposals:
            try:
                trial = rustcpd_module.register_atlas(
                    screen_target, screen_mean, screen_modes, eigenvalues,
                    initial_rotation=rotation, initial_scale=1.0, initial_translation=translation,
                    **dict(atlas_kwargs, max_iterations=int(screen_iterations)),
                )
            except Exception:
                continue
            screened.append((score(trial, screen_target), trial))
        candidates += len(screened)
        screened.sort(key=lambda item: item[0])
        for _, trial in screened[: int(refine_proposals)]:
            refined = rustcpd_module.register_atlas(
                working_target, working_mean, working_flat, eigenvalues, initial_state=trial.state, **atlas_kwargs,
            )
            value = score(refined)
            if value < best:
                best, fit, chosen = value, refined, "proposal"
    return dict(pose=pose, fit=fit, chosen=chosen, candidates=candidates, atlas_kwargs=atlas_kwargs, score=best)


def register_robust(
    *,
    core: Any,
    rustcpd_module: Any,
    settings: Any,
    ssm: Any,
    target_full: np.ndarray,
    target: np.ndarray,
    source_centroid: np.ndarray,
    target_centroid: np.ndarray,
    template_points: Optional[np.ndarray],
    landmark_kwargs: dict,
    pose_landmark_kwargs: dict,
) -> dict:
    """Pose search, atlas fit and candidate selection in the working frame."""
    coverage = float(settings.coverage)
    is_fragment = coverage < 0.95
    working_target = target - target_centroid
    # 1. Size: hull volume x coverage-consistent radii (geometric mean).
    if coverage < 1.0:
        radii_scale = float(np.median(core.coverage_scale_candidates(ssm.mean, target, coverage)))
    else:
        radii_scale = core.rms_radius(target) / max(core.rms_radius(ssm.mean), _TINY)
    source_scale = radii_scale
    if template_points is not None and len(template_points) >= 4:
        try:
            source_scale = math.sqrt(hull_scale(target_full, template_points, coverage) * radii_scale)
        except Exception:  # degenerate (planar) fragment: keep the radii estimate
            source_scale = radii_scale
    margin = float(settings.robust_scale_margin)
    bounds = (1.0 - margin, 1.0 / (1.0 - margin))
    working_mean = (ssm.mean - source_centroid) * source_scale
    working_modes = ssm.modes * source_scale
    working_flat = working_modes.reshape(working_mean.size, ssm.retained_modes)
    # 2-3. Global pose search shared with Landmark Transfer.
    search = global_pose_search(
        rustcpd_module, working_mean, working_target, working_flat, ssm.eigenvalues,
        coverage=coverage,
        bounds=bounds,
        outlier_weight=float(settings.atlas_outlier_weight),
        adaptive_mixing=float(settings.robust_adaptive_mixing),
        pose_lambda=float(settings.pose_lambda),
        atlas_lambda=float(settings.atlas_lambda),
        rotation_count=int(settings.rotation_count),
        coarse_source_count=int(settings.coarse_source_count),
        coarse_target_count=int(settings.coarse_target_count),
        coarse_rank=int(settings.coarse_rank),
        coarse_iterations=int(settings.coarse_iterations),
        coarse_screen_iterations=int(settings.coarse_screen_iterations),
        coarse_survivor_count=int(settings.coarse_survivor_count),
        coarse_score_mode=str(settings.coarse_score_mode),
        refine_count=int(settings.refine_count),
        refine_target_count=int(settings.refine_target_count),
        refine_iterations=int(settings.refine_iterations),
        identity_prior_probability=float(settings.identity_prior_probability),
        translation_anchor_count=int(settings.translation_anchor_count),
        anchor_completeness_threshold=float(settings.anchor_completeness_threshold),
        merge_tolerance=float(settings.merge_tolerance),
        atlas_max_iterations=int(settings.atlas_max_iterations),
        atlas_tolerance=float(settings.atlas_tolerance),
        atlas_k=settings.atlas_k,
        seeds=int(settings.robust_pose_seeds),
        region_proposals=int(settings.robust_region_proposals),
        screen_iterations=int(settings.robust_screen_iterations),
        screen_points=int(settings.robust_screen_points),
        refine_proposals=int(settings.robust_refine_proposals),
        seed=int(settings.seed),
        parallel=bool(settings.parallel),
        single_precision=bool(settings.single_precision),
        landmark_kwargs=landmark_kwargs,
        pose_landmark_kwargs=pose_landmark_kwargs,
        extra_proposals=lambda: (
            fpfh_pose_proposals(
                (np.asarray(template_points, dtype=np.float64) - source_centroid) * source_scale,
                target_full - target_centroid,
                hypotheses=int(settings.robust_ransac_hypotheses), seed=int(settings.seed),
            )
            if is_fragment and not landmark_kwargs and int(settings.robust_ransac_hypotheses) > 0
            and template_points is not None else []
        ),
    )
    pose, fit, chosen, candidates = search["pose"], search["fit"], search["chosen"], search["candidates"]
    atlas_kwargs = search["atlas_kwargs"]
    # 4. Density-matched refit on a dense, near-uniform sample of the fragment.
    if bool(settings.robust_density_refit):
        dense = core.deterministic_downsample(target_full, 30_000) - target_centroid
        working_target = density_matched_target(dense, fit.points, len(working_target), settings.seed)
        fit = rustcpd_module.register_atlas(
            working_target, working_mean, working_flat, ssm.eigenvalues, initial_state=fit.state, **atlas_kwargs
        )
    return dict(
        pose=pose, fit=fit, working_mean=working_mean, working_modes=working_modes,
        working_flat=working_flat, working_target=working_target, source_scale=float(source_scale),
        scale_bounds=bounds, chosen=chosen, candidates=candidates,
    )
