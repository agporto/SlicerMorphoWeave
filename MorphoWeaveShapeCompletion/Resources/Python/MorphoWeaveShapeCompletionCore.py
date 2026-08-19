from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

DEFAULT_COVERAGE = 1.0
LANDMARK_RECOMMENDED_BELOW = 0.75
PROFILE_VERSION = 1
_TINY = 1e-12


@dataclass(frozen=True)
class SSMData:
    mean: np.ndarray
    modes: np.ndarray
    modes_flat: np.ndarray
    eigenvalues: np.ndarray
    retained_modes: int
    retained_variance: float


@dataclass(frozen=True)
class LandmarkMatch:
    source_indices: np.ndarray
    target_points: np.ndarray
    labels: tuple[str, ...]
    source_points: np.ndarray
    mapping_distances: np.ndarray
    used_order_fallback: bool = False

    @property
    def count(self) -> int:
        return int(len(self.source_indices))


@dataclass
class CompletionSettings:
    coverage: float = DEFAULT_COVERAGE
    variance_keep: float = 0.95
    target_point_count: int = 2000
    coverage_prescale: bool = True
    residual_scale_policy: str = "auto"  # auto, free, fixed
    use_landmarks: bool = False
    landmark_sigma_percent: float = 2.0
    rotation_count: int = 193
    coarse_source_count: int = 350
    coarse_target_count: int = 350
    coarse_rank: int = 8
    coarse_iterations: int = 8
    coarse_screen_iterations: int = 8
    coarse_survivor_count: int = 193
    coarse_score_mode: str = "trajectory"
    refine_count: int = 8
    refine_source_count: Optional[int] = None
    refine_target_count: int = 1200
    refine_iterations: int = 25
    pose_lambda: float = 5.0
    pose_outlier_weight: float = 0.10
    identity_prior_probability: float = 0.20
    atlas_lambda: float = 5.0
    atlas_outlier_weight: float = 0.10
    atlas_max_iterations: int = 150
    atlas_tolerance: float = 1e-7
    atlas_k: Optional[int] = None
    estimate_discrepancy: bool = True
    posterior_samples: int = 0
    seed: int = 0
    parallel: bool = True
    single_precision: bool = False

    def validate(self) -> None:
        if not 0.05 <= float(self.coverage) <= 1.0:
            raise ValueError("coverage must be between 0.05 and 1.0")
        if not 0.0 < float(self.variance_keep) <= 1.0:
            raise ValueError("variance_keep must be in (0, 1]")
        if int(self.target_point_count) < 20:
            raise ValueError("target_point_count must be at least 20")
        if self.residual_scale_policy not in {"auto", "free", "fixed"}:
            raise ValueError("residual_scale_policy must be auto, free, or fixed")
        if float(self.landmark_sigma_percent) <= 0:
            raise ValueError("landmark_sigma_percent must be positive")
        positive = {
            "rotation_count": self.rotation_count,
            "coarse_source_count": self.coarse_source_count,
            "coarse_target_count": self.coarse_target_count,
            "coarse_rank": self.coarse_rank,
            "coarse_iterations": self.coarse_iterations,
            "coarse_screen_iterations": self.coarse_screen_iterations,
            "coarse_survivor_count": self.coarse_survivor_count,
            "refine_count": self.refine_count,
            "refine_target_count": self.refine_target_count,
            "refine_iterations": self.refine_iterations,
            "atlas_max_iterations": self.atlas_max_iterations,
        }
        bad = [name for name, value in positive.items() if int(value) < 1]
        if bad:
            raise ValueError("settings must be positive: " + ", ".join(bad))
        if self.coarse_screen_iterations > self.coarse_iterations:
            raise ValueError("coarse_screen_iterations cannot exceed coarse_iterations")
        if self.coarse_survivor_count < min(self.refine_count, self.rotation_count):
            raise ValueError("coarse_survivor_count must cover the finalists")
        if self.refine_source_count is not None and int(self.refine_source_count) < 1:
            raise ValueError("refine_source_count must be positive or None")
        if self.pose_lambda < 0 or self.atlas_lambda < 0:
            raise ValueError("regularization strengths must be non-negative")
        if not 0 <= self.pose_outlier_weight < 1 or not 0 <= self.atlas_outlier_weight < 1:
            raise ValueError("outlier weights must be in [0, 1)")
        if not 0 < self.identity_prior_probability < 1:
            raise ValueError("identity_prior_probability must be in (0, 1)")
        if self.atlas_k is not None and int(self.atlas_k) < 1:
            raise ValueError("atlas_k must be positive or None")
        if int(self.posterior_samples) < 0:
            raise ValueError("posterior_samples cannot be negative")

    def with_scale(self) -> bool:
        if self.residual_scale_policy == "free":
            return True
        if self.residual_scale_policy == "fixed":
            return False
        return bool(np.isclose(float(self.coverage), 1.0))


@dataclass(frozen=True)
class CompletionResult:
    completed_points: np.ndarray
    fitted_points: np.ndarray
    posterior_mean_points: np.ndarray
    posterior_mean_rms_from_atlas: float
    target_points_used: np.ndarray
    epistemic_variance: np.ndarray
    total_variance: np.ndarray
    samples: tuple[np.ndarray, ...]
    pose: Any
    atlas: Any
    posterior: Any
    source_scale: float
    landmark_sigma: Optional[float]
    landmark_nominal_sigma: Optional[float]
    landmark_mapping_rms: Optional[float]
    landmark_mapping_max: Optional[float]
    normalized_sigma2: float
    world_scale: float
    world_rotation: np.ndarray
    world_translation: np.ndarray
    source_centroid: np.ndarray
    target_centroid: np.ndarray
    retained_modes: int
    retained_variance: float


@dataclass(frozen=True)
class LocalTransferOperator:
    """Sparse local interpolation from dense SSM controls to output vertices.

    The matrix is intentionally retained independently of any completed shape so
    it can be reused for geometry, scalar arrays, and latent posterior samples.
    Weights are stored as float32 and column indices as int32 whenever the matrix
    dimensions permit, keeping an eight-neighbor one-million-vertex operator near
    70 MiB rather than rebuilding k-nearest-neighbor searches for every field.
    """

    matrix: Any
    nearest_source_indices: np.ndarray
    source_count: int
    query_count: int
    neighbors: int
    sharpness: float
    chunk_size: int
    exact_anchor_count: int
    component_count: int
    component_fallback_query_count: int
    mean_support_radius: float
    max_support_radius: float
    build_seconds: float
    memory_bytes: int

    def apply(self, values: np.ndarray | Sequence[float]) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim not in (1, 2):
            raise ValueError("transfer values must be a vector or a two-dimensional field matrix")
        if array.shape[0] != int(self.source_count):
            raise ValueError(
                f"transfer values contain {array.shape[0]} rows; expected {self.source_count}"
            )
        return np.asarray(self.matrix @ array)


def apply_similarity(
    points: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply MorphoWeave/rustcpd's row-vector similarity convention."""

    points = _points(points, "similarity points")
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    if translation.shape != (3,):
        raise ValueError("translation must contain three coordinates")
    if not np.isfinite(float(scale)) or np.any(~np.isfinite(rotation)) or np.any(
        ~np.isfinite(translation)
    ):
        raise ValueError("similarity parameters must be finite")
    return float(scale) * (points @ rotation) + translation


@dataclass(frozen=True)
class PlaneMask:
    indices: np.ndarray
    normal: np.ndarray
    threshold: float
    keep_low: bool

    def contains(self, points: np.ndarray) -> np.ndarray:
        projection = np.asarray(points, dtype=np.float64) @ self.normal
        return projection <= self.threshold if self.keep_low else projection >= self.threshold


@dataclass(frozen=True)
class RidgeLogisticCalibrator:
    intercept: float
    slope: float
    feature_mean: float
    feature_scale: float
    l2: float

    @classmethod
    def fit(
        cls,
        normalized_sigma2: Sequence[float],
        correct: Sequence[bool],
        *,
        l2: float = 1.0,
        max_iter: int = 100,
        tolerance: float = 1e-10,
    ) -> "RidgeLogisticCalibrator":
        s = np.asarray(normalized_sigma2, dtype=np.float64).reshape(-1)
        y = np.asarray(correct, dtype=np.float64).reshape(-1)
        if s.shape != y.shape or len(s) == 0:
            raise ValueError("normalized_sigma2 and correct must be non-empty and equally sized")
        if np.any(~np.isfinite(s)) or np.any(s <= 0):
            raise ValueError("normalized_sigma2 must be finite and positive")
        if np.unique(y).size != 2:
            raise ValueError("both successful and failed examples are required")
        x0 = np.log(np.maximum(s, _TINY))
        mu = float(x0.mean())
        sd = float(x0.std())
        if sd <= _TINY:
            sd = 1.0
        x = (x0 - mu) / sd
        design = np.column_stack((np.ones_like(x), x))
        coef = np.zeros(2, dtype=np.float64)
        penalty = np.diag([0.0, max(float(l2), 0.0)])
        for _ in range(int(max_iter)):
            eta = design @ coef
            p = stable_sigmoid(eta)
            weights = np.maximum(p * (1.0 - p), 1e-9)
            gradient = design.T @ (p - y) + penalty @ coef
            hessian = (design * weights[:, None]).T @ design + penalty
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            candidate = coef - step
            if not np.all(np.isfinite(candidate)):
                raise RuntimeError("ridge logistic calibration did not remain finite")
            coef = candidate
            if float(np.max(np.abs(step))) < float(tolerance):
                break
        return cls(
            intercept=float(coef[0]),
            slope=float(coef[1]),
            feature_mean=mu,
            feature_scale=sd,
            l2=float(l2),
        )

    def probability(self, normalized_sigma2: Sequence[float] | float) -> np.ndarray:
        s = np.asarray(normalized_sigma2, dtype=np.float64)
        x = (np.log(np.maximum(s, _TINY)) - self.feature_mean) / self.feature_scale
        return stable_sigmoid(self.intercept + self.slope * x)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RidgeLogisticCalibrator":
        return cls(**{name: float(value[name]) for name in cls.__dataclass_fields__})


@dataclass
class CalibrationAccumulator:
    coverage: float
    landmark_mode: bool
    truth_type: str
    fit_ids: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    errors: list[np.ndarray] = field(default_factory=list)
    variances: list[np.ndarray] = field(default_factory=list)
    normalized_sigma2: list[float] = field(default_factory=list)
    successful: list[bool] = field(default_factory=list)
    missing_rms: list[float] = field(default_factory=list)

    def add(
        self,
        *,
        fit_id: str,
        group: str,
        errors: np.ndarray,
        variances: np.ndarray,
        normalized_sigma2: float,
        successful: bool,
        missing_rms: float,
    ) -> None:
        e = np.asarray(errors, dtype=np.float64).reshape(-1)
        v = np.asarray(variances, dtype=np.float64).reshape(-1)
        if e.shape != v.shape or len(e) == 0:
            raise ValueError("calibration errors and variances must match and be non-empty")
        keep = np.isfinite(e) & np.isfinite(v) & (v >= 0)
        if not np.any(keep):
            raise ValueError("calibration example has no finite pointwise values")
        self.fit_ids.append(str(fit_id))
        self.groups.append(str(group))
        self.errors.append(e[keep])
        self.variances.append(v[keep])
        self.normalized_sigma2.append(float(normalized_sigma2))
        self.successful.append(bool(successful))
        self.missing_rms.append(float(missing_rms))


def stable_sigmoid(value: np.ndarray | float) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def rms_radius(points: np.ndarray) -> float:
    p = _points(points, "points")
    centered = p - p.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def bbox_diagonal(points: np.ndarray) -> float:
    p = _points(points, "points")
    return float(np.linalg.norm(np.ptp(p, axis=0)))


def validate_and_truncate_ssm(
    mean: np.ndarray,
    modes: np.ndarray,
    eigenvalues: np.ndarray,
    variance_keep: float = 0.95,
) -> SSMData:
    mean = _points(mean, "SSM mean")
    modes = np.asarray(modes, dtype=np.float64)
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    if modes.ndim == 3:
        if modes.shape[:2] != mean.shape:
            raise ValueError("3-D modes must have shape (M, 3, K)")
        modes3 = modes
    elif modes.ndim == 2:
        if modes.shape[0] != mean.size:
            raise ValueError("flattened modes must have M*3 rows")
        modes3 = modes.reshape(len(mean), 3, modes.shape[1])
    else:
        raise ValueError("modes must have shape (M, 3, K) or (M*3, K)")
    if modes3.shape[2] != len(eigenvalues) or len(eigenvalues) == 0:
        raise ValueError("eigenvalues must match the number of modes")
    if not np.all(np.isfinite(modes3)) or not np.all(np.isfinite(eigenvalues)):
        raise ValueError("SSM arrays must be finite")
    if np.any(eigenvalues <= 0):
        raise ValueError("SSM eigenvalues must be positive")
    variance_keep = float(variance_keep)
    if not 0 < variance_keep <= 1:
        raise ValueError("variance_keep must be in (0, 1]")
    total = float(eigenvalues.sum())
    if variance_keep >= 1.0 or total <= 0:
        k = len(eigenvalues)
    else:
        k = int(np.searchsorted(np.cumsum(eigenvalues), variance_keep * total) + 1)
    k = max(1, min(k, len(eigenvalues)))
    retained = float(eigenvalues[:k].sum() / total) if total > 0 else 1.0
    modes3 = np.ascontiguousarray(modes3[:, :, :k])
    return SSMData(
        mean=np.ascontiguousarray(mean),
        modes=modes3,
        modes_flat=np.ascontiguousarray(modes3.reshape(mean.size, k)),
        eigenvalues=np.ascontiguousarray(eigenvalues[:k]),
        retained_modes=k,
        retained_variance=retained,
    )


def calibration_method_signature(settings: Mapping[str, Any] | CompletionSettings) -> str:
    """Fingerprint inference settings that affect fit or uncertainty.

    Coverage, landmark mode, and posterior sample count are intentionally
    excluded because profiles store coverage/mode per entry and posterior samples
    do not alter the fitted distribution. The seed remains in the signature because
    it changes the deterministic rotation lattice and can therefore change the fit.
    """
    if isinstance(settings, CompletionSettings):
        values = dict(settings.__dict__)
    else:
        values = dict(settings)
    excluded = {"coverage", "use_landmarks", "posterior_samples"}
    canonical = {key: values[key] for key in sorted(values) if key not in excluded}
    payload = json.dumps(_jsonable(canonical), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_fingerprint(mean: np.ndarray, modes: np.ndarray, eigenvalues: np.ndarray) -> str:
    ssm = validate_and_truncate_ssm(mean, modes, eigenvalues, 1.0)
    digest = hashlib.sha256()
    for array in (ssm.mean, ssm.modes_flat, ssm.eigenvalues):
        canonical = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(canonical.shape).encode("ascii"))
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def landmark_recommendation(coverage: float, matched_count: int = 0) -> tuple[str, str]:
    coverage = float(coverage)
    if coverage < LANDMARK_RECOMMENDED_BELOW and matched_count < 3:
        return (
            "recommended",
            "At target coverage below 0.75, at least three homologous target landmarks are recommended by the current fragment experiments.",
        )
    if matched_count > 0 and matched_count < 3:
        return "insufficient", "Landmark-constrained completion requires at least three matched landmarks."
    if matched_count >= 3:
        return "available", f"{matched_count} homologous landmarks are available for pose search, refinement, and atlas registration."
    return "optional", "Landmarks are optional at this coverage; surface-only pose and atlas registration will be used."


def normalize_label(label: str) -> str:
    return " ".join(str(label or "").strip().casefold().split())


def match_landmarks(
    template_labels: Sequence[str],
    template_points: np.ndarray,
    target_labels: Sequence[str],
    target_points: np.ndarray,
    dense_points: np.ndarray,
    *,
    allow_order_fallback: bool = False,
    minimum_count: int = 3,
) -> LandmarkMatch:
    template_points = _points(template_points, "template landmarks")
    target_points = _points(target_points, "target landmarks")
    dense_points = _points(dense_points, "dense correspondences")
    if len(template_labels) != len(template_points) or len(target_labels) != len(target_points):
        raise ValueError("landmark labels and points must have matching lengths")
    template_map = _unique_label_map(template_labels, "template")
    target_map = _unique_label_map(target_labels, "target")
    common = sorted(set(template_map).intersection(target_map))
    used_order = False
    if common:
        source_sparse = np.asarray([template_map[label] for label in common], dtype=np.int64)
        target_sparse = np.asarray([target_map[label] for label in common], dtype=np.int64)
        labels = tuple(str(template_labels[i]) for i in source_sparse)
    elif allow_order_fallback and len(template_points) == len(target_points):
        source_sparse = np.arange(len(template_points), dtype=np.int64)
        target_sparse = np.arange(len(target_points), dtype=np.int64)
        labels = tuple(str(template_labels[i] or f"Point-{i + 1}") for i in source_sparse)
        used_order = True
    else:
        raise ValueError("no common landmark labels were found; enable ordered fallback only when order is known to be homologous")
    if len(source_sparse) < int(minimum_count):
        raise ValueError(f"at least {minimum_count} matched landmarks are required")
    source_points = template_points[source_sparse]
    paired_targets = target_points[target_sparse]
    indices, distances = nearest_indices(dense_points, source_points)
    unique_dense: dict[int, int] = {}
    for position, dense_index in enumerate(indices.tolist()):
        current = unique_dense.get(dense_index)
        if current is None or distances[position] < distances[current]:
            unique_dense[dense_index] = position
    positions = np.asarray(sorted(unique_dense.values()), dtype=np.int64)
    if len(positions) < int(minimum_count):
        raise ValueError("matched sparse landmarks collapse to fewer than three unique dense SSM points")
    source_points = source_points[positions]
    paired_targets = paired_targets[positions]
    indices = indices[positions]
    distances = distances[positions]
    labels = tuple(labels[i] for i in positions)
    _require_noncollinear(source_points, "template")
    _require_noncollinear(paired_targets, "target")
    return LandmarkMatch(
        source_indices=indices,
        target_points=paired_targets,
        labels=labels,
        source_points=source_points,
        mapping_distances=distances,
        used_order_fallback=used_order,
    )


def nearest_indices(reference: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = _points(reference, "reference")
    query = _points(query, "query")
    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(reference).query(query, k=1)
        return np.asarray(indices, dtype=np.int64), np.asarray(distances, dtype=np.float64)
    except Exception:
        indices = np.empty(len(query), dtype=np.int64)
        distances = np.empty(len(query), dtype=np.float64)
        chunk = 256
        for start in range(0, len(query), chunk):
            q = query[start : start + chunk]
            distance2 = np.sum((q[:, None, :] - reference[None, :, :]) ** 2, axis=2)
            local = np.argmin(distance2, axis=1)
            indices[start : start + len(q)] = local
            distances[start : start + len(q)] = np.sqrt(distance2[np.arange(len(q)), local])
        return indices, distances


def deterministic_downsample(points: np.ndarray, target_count: int) -> np.ndarray:
    points = _points(points, "target points")
    target_count = int(target_count)
    if target_count < 1:
        raise ValueError("target_count must be positive")
    if len(points) <= target_count:
        return points.copy()
    origin = points.min(axis=0)
    diagonal = bbox_diagonal(points)
    if diagonal <= _TINY:
        return points[:target_count].copy()
    low, high = diagonal * 1e-9, diagonal
    best = np.arange(len(points), dtype=np.int64)
    best_error = len(points) - target_count
    for _ in range(42):
        voxel = math.sqrt(low * high)
        keys = np.floor((points - origin) / voxel).astype(np.int64)
        order = np.lexsort((np.arange(len(points)), keys[:, 2], keys[:, 1], keys[:, 0]))
        ordered_keys = keys[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = np.any(ordered_keys[1:] != ordered_keys[:-1], axis=1)
        indices = np.sort(order[first])
        count = len(indices)
        error = abs(count - target_count)
        if error < best_error:
            best, best_error = indices, error
        if count > target_count:
            low = voxel
        elif count < target_count:
            high = voxel
        else:
            return points[indices]
    if len(best) > target_count:
        best = farthest_point_subset(points, best, target_count)
    elif len(best) < target_count:
        missing = np.setdiff1d(np.arange(len(points)), best, assume_unique=False)
        add = farthest_point_subset(points, missing, target_count - len(best))
        best = np.sort(np.concatenate((best, add)))
    return points[best[:target_count]].copy()


def farthest_point_subset(points: np.ndarray, candidate_indices: Sequence[int], count: int) -> np.ndarray:
    points = _points(points, "points")
    candidates = np.unique(np.asarray(candidate_indices, dtype=np.int64))
    count = int(count)
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if len(candidates) <= count:
        return np.sort(candidates)
    values = points[candidates]
    first = int(np.argmax(np.sum((values - values.mean(axis=0)) ** 2, axis=1)))
    chosen = [first]
    selected = np.zeros(len(candidates), dtype=bool)
    selected[first] = True
    nearest = np.sum((values - values[first]) ** 2, axis=1)
    while len(chosen) < count:
        nearest[selected] = -1
        next_index = int(np.argmax(nearest))
        chosen.append(next_index)
        selected[next_index] = True
        nearest = np.minimum(nearest, np.sum((values - values[next_index]) ** 2, axis=1))
    return np.sort(candidates[np.asarray(chosen, dtype=np.int64)])


def indexed_correspondence_compatibility(
    model_points: np.ndarray,
    template_points: np.ndarray,
    *,
    max_rms_fraction: float = 0.35,
) -> dict[str, float]:
    """Check that an indexed template belongs to the selected SSM frame.

    Point count alone cannot detect a manually mixed SSM/template pair. Because
    both arrays are dense homologous correspondences, index-wise displacement is
    the relevant diagnostic; the threshold is deliberately generous enough for
    a non-mean template but rejects gross coordinate-frame or database mismatch.
    """
    model = _points(model_points, "SSM mean")
    template = _points(template_points, "template dense correspondences")
    if model.shape != template.shape:
        raise ValueError("SSM mean and template dense correspondences must have the same shape")
    displacement = np.linalg.norm(model - template, axis=1)
    diagonal = max(bbox_diagonal(model), bbox_diagonal(template), _TINY)
    rms = float(np.sqrt(np.mean(displacement * displacement)))
    maximum = float(np.max(displacement))
    fraction = rms / diagonal
    if fraction > float(max_rms_fraction):
        raise ValueError(
            "The selected SSM and template dense correspondences are not compatible "
            f"index by index (RMS difference={100 * fraction:.1f}% of diagonal). "
            "Load the canonical Model Library quartet or remove any transform applied only to the template nodes."
        )
    return {
        "rms_distance": rms,
        "max_distance": maximum,
        "rms_fraction": fraction,
    }


def coordinate_compatibility(
    reference_surface_points: np.ndarray,
    candidate_points: np.ndarray,
    label: str,
    *,
    max_median_fraction: float = 0.10,
    max_p95_fraction: float = 0.25,
) -> dict[str, float]:
    reference = _points(reference_surface_points, "reference surface points")
    candidate = _points(candidate_points, label)
    _, distances = nearest_indices(reference, candidate)
    diagonal = max(bbox_diagonal(reference), _TINY)
    median = float(np.median(distances))
    p95 = float(np.quantile(distances, 0.95))
    median_fraction = median / diagonal
    p95_fraction = p95 / diagonal
    if median_fraction > float(max_median_fraction) or p95_fraction > float(max_p95_fraction):
        raise ValueError(
            f"{label} appears to use a different coordinate frame from its surface "
            f"(median nearest-surface distance={100 * median_fraction:.1f}% and "
            f"95th percentile={100 * p95_fraction:.1f}% of the surface diagonal)."
        )
    return {
        "median_distance": median,
        "p95_distance": p95,
        "median_fraction": median_fraction,
        "p95_fraction": p95_fraction,
    }


def coverage_prescale(mean: np.ndarray, target: np.ndarray, coverage: float) -> float:
    source_diagonal = bbox_diagonal(mean)
    target_diagonal = bbox_diagonal(target)
    coverage = float(np.clip(coverage, 1e-3, 1.0))
    if source_diagonal <= _TINY:
        return 1.0
    return float((target_diagonal / coverage) / source_diagonal)


def run_completion(
    target_points: np.ndarray,
    mean: np.ndarray,
    modes: np.ndarray,
    eigenvalues: np.ndarray,
    settings: CompletionSettings,
    *,
    rustcpd_module: Any,
    landmarks: Optional[LandmarkMatch] = None,
) -> CompletionResult:
    settings.validate()
    ssm = validate_and_truncate_ssm(mean, modes, eigenvalues, settings.variance_keep)
    target_full = _points(target_points, "target fragment")
    if len(target_full) < 20:
        raise ValueError("target fragment must contain at least 20 surface points")
    target = deterministic_downsample(target_full, settings.target_point_count)
    source_centroid = ssm.mean.mean(axis=0, keepdims=True)
    target_centroid = target.mean(axis=0, keepdims=True)
    source_scale = 1.0
    if settings.coverage < 1.0 and settings.coverage_prescale:
        source_scale = coverage_prescale(ssm.mean, target, settings.coverage)
    working_mean = (ssm.mean - source_centroid) * source_scale
    working_modes = ssm.modes * source_scale
    working_modes_flat = working_modes.reshape(working_mean.size, ssm.retained_modes)
    working_target = target - target_centroid
    with_scale = settings.with_scale()

    if settings.use_landmarks and landmarks is None:
        raise ValueError("landmark mode was requested but no matched landmarks were supplied")
    use_landmarks = bool(settings.use_landmarks)
    landmark_sigma = None
    landmark_nominal_sigma = None
    landmark_mapping_rms = None
    landmark_mapping_max = None
    landmark_indices = None
    landmark_targets = None
    if use_landmarks:
        if landmarks.count < 3:
            raise ValueError("landmark-constrained completion requires at least three landmarks")
        if np.any(landmarks.source_indices < 0) or np.any(landmarks.source_indices >= len(working_mean)):
            raise ValueError("a landmark source index is outside the SSM")
        landmark_indices = [int(value) for value in landmarks.source_indices]
        landmark_targets = np.asarray(landmarks.target_points, dtype=np.float64) - target_centroid
        radius = (
            rms_radius(working_target)
            if np.isclose(float(settings.coverage), 1.0)
            else rms_radius(working_mean)
        )
        if radius <= _TINY:
            raise ValueError("cannot define landmark uncertainty for a zero-radius expected shape")
        landmark_nominal_sigma = float(settings.landmark_sigma_percent) / 100.0 * radius
        source_radius = max(rms_radius(ssm.mean), _TINY)
        mapping_scale = radius / source_radius
        mapping_distances = np.asarray(landmarks.mapping_distances, dtype=np.float64).reshape(-1)
        if len(mapping_distances) != landmarks.count or np.any(~np.isfinite(mapping_distances)):
            raise ValueError("landmark mapping distances must be finite and match the landmarks")
        landmark_mapping_rms = float(np.sqrt(np.mean(mapping_distances * mapping_distances))) * mapping_scale
        landmark_mapping_max = float(np.max(mapping_distances)) * mapping_scale
        # The native keypoint API anchors dense SSM vertices. Sparse template
        # landmarks generally lie between dense vertices, so their vertex-mapping
        # approximation is an additional target-frame uncertainty source.
        landmark_sigma = float(math.hypot(landmark_nominal_sigma, landmark_mapping_rms))

    pose_kwargs: dict[str, Any] = {
        "rotation_count": int(settings.rotation_count),
        "coarse_source_count": min(int(settings.coarse_source_count), len(working_mean)),
        "coarse_target_count": min(int(settings.coarse_target_count), len(working_target)),
        "coarse_rank": min(int(settings.coarse_rank), ssm.retained_modes),
        "coarse_iterations": int(settings.coarse_iterations),
        "coarse_screen_iterations": int(settings.coarse_screen_iterations),
        "coarse_survivor_count": min(int(settings.coarse_survivor_count), int(settings.rotation_count)),
        "coarse_score_mode": str(settings.coarse_score_mode),
        "refine_count": min(int(settings.refine_count), int(settings.rotation_count)),
        "refine_source_count": (
            None
            if settings.refine_source_count is None
            else min(int(settings.refine_source_count), len(working_mean))
        ),
        "refine_target_count": min(int(settings.refine_target_count), len(working_target)),
        "refine_iterations": int(settings.refine_iterations),
        "lambda_regularization": float(settings.pose_lambda),
        "outlier_weight": float(settings.pose_outlier_weight),
        "identity_prior_probability": float(settings.identity_prior_probability),
        "with_scale": bool(with_scale),
        "seed": int(settings.seed),
        "parallel": bool(settings.parallel),
        "single_precision": bool(settings.single_precision),
    }
    if use_landmarks:
        pose_kwargs.update(
            landmark_indices=landmark_indices,
            landmark_targets=landmark_targets,
            landmark_sigma=landmark_sigma,
            refine_landmark_sigma=landmark_sigma,
        )
    pose = rustcpd_module.pose_initialize(
        working_mean,
        working_target,
        working_modes_flat,
        ssm.eigenvalues,
        **pose_kwargs,
    )

    atlas_kwargs: dict[str, Any] = {
        "lambda_regularization": float(settings.atlas_lambda),
        "normalize": False,
        "optimize_similarity": True,
        "with_scale": bool(with_scale),
        "initial_coefficients": np.asarray(pose.coefficients, dtype=np.float64),
        "initial_rotation": np.asarray(pose.rotation, dtype=np.float64),
        "initial_scale": float(pose.scale if with_scale else 1.0),
        "initial_translation": np.asarray(pose.translation, dtype=np.float64).reshape(-1),
        "max_iterations": int(settings.atlas_max_iterations),
        "tolerance": float(settings.atlas_tolerance),
        "outlier_weight": float(settings.atlas_outlier_weight),
        "k": None if settings.atlas_k is None else int(settings.atlas_k),
        "parallel": bool(settings.parallel),
        "single_precision": bool(settings.single_precision),
    }
    if use_landmarks:
        atlas_kwargs.update(
            landmark_indices=landmark_indices,
            landmark_targets=landmark_targets,
            landmark_sigma=landmark_sigma,
        )
    fit = rustcpd_module.register_atlas(
        working_target,
        working_mean,
        working_modes_flat,
        ssm.eigenvalues,
        **atlas_kwargs,
    )
    posterior = fit.posterior(
        working_target,
        working_mean,
        working_modes_flat,
        ssm.eigenvalues,
        completeness=float(settings.coverage),
        # Atlas registration adds λ·Λ⁻¹ to the coefficient precision after
        # division by σ_eff². ShapePosterior parameterizes that same prior as
        # (temperature·Λ)⁻¹, so the matching temperature is 1/λ.
        prior_temperature=(
            1.0 / float(settings.atlas_lambda)
            if float(settings.atlas_lambda) > _TINY
            else 1e12
        ),
        outlier_weight=float(settings.atlas_outlier_weight),
        estimate_discrepancy=bool(settings.estimate_discrepancy),
    )
    posterior_mean_centered = np.asarray(posterior.predict(), dtype=np.float64)
    fitted_centered = np.asarray(fit.points, dtype=np.float64)
    if posterior_mean_centered.shape != working_mean.shape:
        raise RuntimeError("rustcpd returned an unexpected posterior-mean shape size")
    if fitted_centered.shape != working_mean.shape:
        raise RuntimeError("rustcpd returned an unexpected atlas-fit shape size")
    # The constrained atlas fit is the scientific point estimate.  The current
    # posterior intentionally conditions on surface correspondences only and does
    # not yet add target landmarks to coefficient precision.  Replacing the atlas
    # estimate with posterior.predict() would therefore partially discard the very
    # landmark information used to make severe-fragment completion identifiable.
    # We retain the atlas mean and use the posterior for covariance, discrepancy,
    # and sampling.  Posterior samples are translated to the constrained atlas
    # center below; their covariance is unchanged.
    completed_centered = fitted_centered
    posterior_center_shift = fitted_centered - posterior_mean_centered
    posterior_mean_rms_from_atlas = float(
        np.sqrt(np.mean(np.sum(posterior_center_shift * posterior_center_shift, axis=1)))
    )
    completed = completed_centered + target_centroid
    fitted = fitted_centered + target_centroid
    posterior_mean_points = posterior_mean_centered + target_centroid
    coefficient_covariance = np.asarray(posterior.coefficient_covariance, dtype=np.float64)
    epistemic = np.einsum(
        "pdk,kl,pdl->p", working_modes, coefficient_covariance, working_modes, optimize=True
    ) * float(fit.scale) ** 2
    epistemic = np.maximum(epistemic, 0.0)
    total = np.maximum(np.asarray(posterior.predictive_variance(), dtype=np.float64), 0.0)
    if total.shape != (len(completed),):
        raise RuntimeError("rustcpd returned an unexpected predictive-variance size")
    samples: list[np.ndarray] = []
    if int(settings.posterior_samples) > 0:
        samples = [
            np.asarray(sample, dtype=np.float64)
            + posterior_center_shift
            + target_centroid
            for sample in posterior.sample_shapes(int(settings.posterior_samples), int(settings.seed))
        ]
    normalization_radius = max(rms_radius(working_mean) * abs(float(fit.scale)), _TINY)
    normalized_sigma2 = float(fit.sigma2) / (normalization_radius * normalization_radius)
    rotation = np.asarray(fit.rotation, dtype=np.float64)
    world_scale = float(source_scale) * float(fit.scale)
    world_translation = (
        np.asarray(fit.translation, dtype=np.float64).reshape(1, 3)
        + target_centroid
        - world_scale * (source_centroid @ rotation)
    ).reshape(3)
    return CompletionResult(
        completed_points=completed,
        fitted_points=fitted,
        posterior_mean_points=posterior_mean_points,
        posterior_mean_rms_from_atlas=posterior_mean_rms_from_atlas,
        target_points_used=target,
        epistemic_variance=epistemic,
        total_variance=total,
        samples=tuple(samples),
        pose=pose,
        atlas=fit,
        posterior=posterior,
        source_scale=float(source_scale),
        landmark_sigma=landmark_sigma,
        landmark_nominal_sigma=landmark_nominal_sigma,
        landmark_mapping_rms=landmark_mapping_rms,
        landmark_mapping_max=landmark_mapping_max,
        normalized_sigma2=normalized_sigma2,
        world_scale=world_scale,
        world_rotation=rotation,
        world_translation=world_translation,
        source_centroid=source_centroid.reshape(3),
        target_centroid=target_centroid.reshape(3),
        retained_modes=ssm.retained_modes,
        retained_variance=ssm.retained_variance,
    )


def classify_completion_region(
    completed_points: np.ndarray, target_fragment: np.ndarray, coverage: float
) -> tuple[np.ndarray, np.ndarray]:
    completed = _points(completed_points, "completed points")
    target = _points(target_fragment, "target fragment")
    _, distance = nearest_indices(target, completed)
    observed_count = int(np.clip(round(float(coverage) * len(completed)), 1, len(completed)))
    order = np.argsort(distance, kind="mergesort")
    region = np.ones(len(completed), dtype=np.int32)
    region[order[:observed_count]] = 0
    return region, distance


def infer_control_vertex_indices(
    mesh_vertices: np.ndarray,
    control_points: np.ndarray,
    *,
    tolerance_fraction: float = 1e-7,
    require_unique: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Map controls to their nearest template vertices and identify exact anchors.

    Returns ``(exact_vertex_indices, nearest_vertex_indices, distances, tolerance)``.
    ``exact_vertex_indices`` contains ``-1`` where a control is not coincident with
    a template vertex within the scale-relative tolerance.  Atlas Builder normally
    exports existing template vertices, so a complete exact mapping is expected for
    current Model Library entries while legacy or externally authored inputs remain
    usable through ordinary local interpolation.
    """

    mesh = _points(mesh_vertices, "mesh vertices")
    controls = _points(control_points, "control points")
    nearest, distances = nearest_indices(mesh, controls)
    tolerance_fraction = float(tolerance_fraction)
    if not np.isfinite(tolerance_fraction) or tolerance_fraction <= 0:
        raise ValueError("tolerance_fraction must be finite and positive")
    tolerance = max(bbox_diagonal(mesh) * tolerance_fraction, 1e-9)
    exact = np.full(len(controls), -1, dtype=np.int64)
    within = distances <= tolerance
    exact[within] = nearest[within]
    if require_unique:
        valid = exact[exact >= 0]
        if len(valid) != len(np.unique(valid)):
            raise ValueError(
                "multiple dense controls map to the same template vertex within the exact-anchor tolerance"
            )
    return exact, nearest, distances, float(tolerance)


def _tree_query(tree: Any, points: np.ndarray, *, k: int, workers: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        distances, indices = tree.query(points, k=k, workers=int(workers))
    except TypeError:
        # Slicer releases carrying older SciPy builds may not expose ``workers``.
        distances, indices = tree.query(points, k=k)
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if int(k) == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)
    return distances, indices


def _normalized_local_weights(
    distances: np.ndarray,
    *,
    sharpness: float,
    tolerance: float,
) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[1] < 1:
        raise ValueError("neighbor distances must be a non-empty matrix")
    if distances.shape[1] == 1:
        return np.ones(distances.shape, dtype=np.float32)
    bandwidth = np.maximum(distances[:, -1:], float(tolerance))
    weights = np.exp(-((float(sharpness) * distances / bandwidth) ** 2))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), np.finfo(np.float64).tiny)
    exact = distances[:, 0] <= float(tolerance)
    if np.any(exact):
        weights[exact] = 0.0
        weights[exact, 0] = 1.0
    return weights.astype(np.float32, copy=False)


def build_local_transfer_operator(
    query_points: np.ndarray,
    source_points: np.ndarray,
    *,
    neighbors: int = 8,
    sharpness: float = 2.0,
    chunk_size: int = 200_000,
    exact_source_vertex_indices: Optional[Sequence[int]] = None,
    query_groups: Optional[Sequence[int]] = None,
    source_groups: Optional[Sequence[int]] = None,
    workers: int = -1,
) -> LocalTransferOperator:
    """Build a reusable sparse local interpolation operator.

    ``query_groups`` and ``source_groups`` may be used to prevent interpolation
    between disconnected mesh components.  A query component with no dense
    controls falls back to the global control set and is counted in the returned
    diagnostics rather than silently producing an empty row.
    """

    query = _points(query_points, "query points")
    source = _points(source_points, "source points")
    if len(source) == 0:
        raise ValueError("at least one source control is required")
    neighbors = max(1, min(int(neighbors), len(source)))
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    sharpness = float(sharpness)
    if not np.isfinite(sharpness) or sharpness <= 0:
        raise ValueError("sharpness must be finite and positive")

    try:
        from scipy.sparse import csr_matrix
        from scipy.spatial import cKDTree
    except Exception as error:
        raise RuntimeError(
            "SciPy spatial and sparse modules are required for full-resolution transfer"
        ) from error

    started = time.perf_counter()
    query_count = len(query)
    source_count = len(source)
    indices = np.empty((query_count, neighbors), dtype=np.int32)
    weights = np.zeros((query_count, neighbors), dtype=np.float32)
    nearest_source = np.empty(query_count, dtype=np.int32)
    tolerance = max(
        max(bbox_diagonal(query), bbox_diagonal(source), 1.0) * 1e-10,
        _TINY,
    )
    support_radius_sum = 0.0
    max_support_radius = 0.0

    if (query_groups is None) != (source_groups is None):
        raise ValueError("query_groups and source_groups must be supplied together")
    component_count = 1
    fallback_query_count = 0

    if query_groups is None:
        tree = cKDTree(source)
        for start in range(0, query_count, chunk_size):
            stop = min(start + chunk_size, query_count)
            distances, local_indices = _tree_query(
                tree, query[start:stop], k=neighbors, workers=workers
            )
            support_radius_sum += float(np.sum(distances[:, -1]))
            max_support_radius = max(max_support_radius, float(np.max(distances[:, -1])))
            indices[start:stop] = local_indices.astype(np.int32, copy=False)
            weights[start:stop] = _normalized_local_weights(
                distances, sharpness=sharpness, tolerance=tolerance
            )
            nearest_source[start:stop] = indices[start:stop, 0]
    else:
        query_groups_array = np.asarray(query_groups).reshape(-1)
        source_groups_array = np.asarray(source_groups).reshape(-1)
        if len(query_groups_array) != query_count or len(source_groups_array) != source_count:
            raise ValueError("component labels must match query and source point counts")
        if np.any(~np.isfinite(query_groups_array.astype(np.float64))) or np.any(
            ~np.isfinite(source_groups_array.astype(np.float64))
        ):
            raise ValueError("component labels must be finite")
        unique_groups = np.unique(query_groups_array)
        component_count = int(len(unique_groups))
        all_source_indices = np.arange(source_count, dtype=np.int64)
        global_tree = None
        for group in unique_groups:
            query_rows = np.flatnonzero(query_groups_array == group)
            source_rows = np.flatnonzero(source_groups_array == group)
            if len(source_rows) == 0:
                source_rows = all_source_indices
                fallback_query_count += int(len(query_rows))
                if global_tree is None:
                    global_tree = cKDTree(source)
                tree = global_tree
            else:
                tree = cKDTree(source[source_rows])
            local_k = min(neighbors, len(source_rows))
            for local_start in range(0, len(query_rows), chunk_size):
                row_ids = query_rows[local_start : local_start + chunk_size]
                distances, local_indices = _tree_query(
                    tree, query[row_ids], k=local_k, workers=workers
                )
                support_radius_sum += float(np.sum(distances[:, -1]))
                max_support_radius = max(
                    max_support_radius, float(np.max(distances[:, -1]))
                )
                global_indices = source_rows[local_indices]
                indices[row_ids, :local_k] = global_indices.astype(np.int32, copy=False)
                weights[row_ids, :local_k] = _normalized_local_weights(
                    distances, sharpness=sharpness, tolerance=tolerance
                )
                nearest_source[row_ids] = indices[row_ids, 0]
                if local_k < neighbors:
                    # Repeated zero-weight columns preserve a compact fixed-width
                    # CSR construction without introducing cross-component support.
                    indices[row_ids, local_k:] = indices[row_ids, :1]

    exact_anchor_count = 0
    if exact_source_vertex_indices is not None:
        exact_vertices = np.asarray(exact_source_vertex_indices, dtype=np.int64).reshape(-1)
        if len(exact_vertices) != source_count:
            raise ValueError("exact_source_vertex_indices must match the source controls")
        valid_controls = np.flatnonzero(exact_vertices >= 0)
        valid_vertices = exact_vertices[valid_controls]
        if np.any(valid_vertices >= query_count):
            raise ValueError("an exact source vertex index is outside the query mesh")
        if len(valid_vertices) != len(np.unique(valid_vertices)):
            raise ValueError("exact source vertex indices must be unique")
        for control_index, vertex_index in zip(valid_controls, valid_vertices):
            indices[vertex_index, :] = int(control_index)
            weights[vertex_index, :] = 0.0
            weights[vertex_index, 0] = 1.0
            nearest_source[vertex_index] = int(control_index)
        exact_anchor_count = int(len(valid_controls))

    nnz = query_count * neighbors
    index_dtype = np.int32 if nnz < np.iinfo(np.int32).max else np.int64
    indptr = np.arange(0, nnz + 1, neighbors, dtype=index_dtype)
    matrix = csr_matrix(
        (weights.reshape(-1), indices.astype(index_dtype, copy=False).reshape(-1), indptr),
        shape=(query_count, source_count),
        copy=False,
    )
    memory_bytes = int(
        matrix.data.nbytes
        + matrix.indices.nbytes
        + matrix.indptr.nbytes
        + nearest_source.nbytes
    )
    return LocalTransferOperator(
        matrix=matrix,
        nearest_source_indices=nearest_source,
        source_count=source_count,
        query_count=query_count,
        neighbors=neighbors,
        sharpness=sharpness,
        chunk_size=chunk_size,
        exact_anchor_count=exact_anchor_count,
        component_count=component_count,
        component_fallback_query_count=int(fallback_query_count),
        mean_support_radius=float(support_radius_sum / max(query_count, 1)),
        max_support_radius=float(max_support_radius),
        build_seconds=float(time.perf_counter() - started),
        memory_bytes=memory_bytes,
    )


def pose_residual_interpolate(
    query_points: np.ndarray,
    source_points: np.ndarray,
    completed_source_points: np.ndarray,
    *,
    operator: LocalTransferOperator,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply similarity pose exactly and interpolate only nonrigid residuals."""

    query = _points(query_points, "query points")
    source = _points(source_points, "source points")
    completed = _points(completed_source_points, "completed source points")
    if source.shape != completed.shape:
        raise ValueError("source and completed source points must have the same shape")
    if operator.query_count != len(query) or operator.source_count != len(source):
        raise ValueError("transfer operator dimensions do not match the supplied points")
    posed_query = apply_similarity(query, scale, rotation, translation)
    posed_source = apply_similarity(source, scale, rotation, translation)
    residual = completed - posed_source
    return posed_query + operator.apply(residual)


def local_displacement_interpolate(
    query_points: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    neighbors: int = 8,
) -> np.ndarray:
    """Backward-compatible direct displacement interpolation for small callers.

    Full-resolution Shape Completion output uses :func:`pose_residual_interpolate`
    with a cached operator so that global similarity pose is reproduced exactly.
    """

    query = _points(query_points, "query points")
    source = _points(source_points, "source points")
    target = _points(target_points, "target points")
    if source.shape != target.shape:
        raise ValueError("source and target correspondences must have the same shape")
    operator = build_local_transfer_operator(
        query,
        source,
        neighbors=neighbors,
        sharpness=1.0,
        chunk_size=max(1, min(len(query), 200_000)),
    )
    return query + operator.apply(target - source)


def local_scalar_interpolate(
    query_points: np.ndarray,
    source_points: np.ndarray,
    values: Sequence[float],
    *,
    neighbors: int = 8,
) -> np.ndarray:
    query = _points(query_points, "query points")
    source = _points(source_points, "source points")
    values_array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values_array) != len(source):
        raise ValueError("scalar values must match source points")
    operator = build_local_transfer_operator(
        query,
        source,
        neighbors=neighbors,
        sharpness=1.0,
        chunk_size=max(1, min(len(query), 200_000)),
    )
    return operator.apply(values_array)

def random_rotation(rng: np.random.Generator) -> np.ndarray:
    quaternion = np.asarray(rng.normal(size=4), dtype=np.float64)
    quaternion /= max(float(np.linalg.norm(quaternion)), _TINY)
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def apply_random_pose(
    points: np.ndarray,
    rng: np.random.Generator,
    *,
    translation_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = _points(points, "points")
    rotation = random_rotation(rng)
    diagonal = max(bbox_diagonal(points), 1.0)
    translation = rng.uniform(-translation_fraction, translation_fraction, size=3) * diagonal
    return points @ rotation + translation, rotation, translation


def contiguous_plane_mask(
    points: np.ndarray, coverage: float, rng: np.random.Generator
) -> PlaneMask:
    points = _points(points, "points")
    coverage = float(coverage)
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    count = max(3, min(len(points), int(round(coverage * len(points)))))
    normal = np.asarray(rng.normal(size=3), dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), _TINY)
    projection = points @ normal
    keep_low = bool(rng.integers(0, 2) == 0)
    order = np.argsort(projection if keep_low else -projection, kind="mergesort")
    indices = np.sort(order[:count])
    selected_projection = projection[indices]
    threshold = float(selected_projection.max() if keep_low else selected_projection.min())
    return PlaneMask(indices=indices, normal=normal, threshold=threshold, keep_low=keep_low)


def nonconformity_scores(errors: np.ndarray, variances: np.ndarray) -> np.ndarray:
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    variances = np.asarray(variances, dtype=np.float64).reshape(-1)
    if errors.shape != variances.shape:
        raise ValueError("errors and variances must have the same shape")
    if np.any(variances < 0):
        raise ValueError("variances cannot be negative")
    return errors / np.sqrt(np.maximum(variances, _TINY))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must be in (0, 1)")
    if len(scores) == 0:
        raise ValueError("at least one score is required")
    rank = int(math.ceil((len(scores) + 1) * (1.0 - float(alpha))))
    if rank > len(scores):
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])


def auc_rank(score: Sequence[float], positive: Sequence[bool]) -> float:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    positive = np.asarray(positive, dtype=bool).reshape(-1)
    if score.shape != positive.shape or np.unique(positive).size != 2:
        raise ValueError("AUC requires equally sized scores and both classes")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def expected_calibration_error(
    probability: Sequence[float], truth: Sequence[bool], bins: int = 10
) -> float:
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    y = np.asarray(truth, dtype=np.float64).reshape(-1)
    if p.shape != y.shape or len(p) == 0:
        raise ValueError("probability and truth must match and be non-empty")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = 0.0
    for index in range(int(bins)):
        if index == bins - 1:
            mask = (p >= edges[index]) & (p <= edges[index + 1])
        else:
            mask = (p >= edges[index]) & (p < edges[index + 1])
        if np.any(mask):
            total += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return total


def _groupwise_max_nonconformity(
    accumulator: CalibrationAccumulator,
    positions: Optional[Sequence[int]] = None,
) -> dict[str, float]:
    selected = range(len(accumulator.errors)) if positions is None else positions
    maxima: dict[str, float] = {}
    for position in selected:
        score = nonconformity_scores(
            accumulator.errors[position], accumulator.variances[position]
        )
        value = float(np.max(score))
        group = accumulator.groups[position]
        maxima[group] = max(value, maxima.get(group, -float("inf")))
    return maxima


def finalize_calibration_entry(
    accumulator: CalibrationAccumulator,
    *,
    alpha: float,
    success_threshold_fraction: float,
    logistic_l2: float = 1.0,
) -> dict[str, Any]:
    if not accumulator.errors:
        raise ValueError("calibration accumulator is empty")
    errors = np.concatenate(accumulator.errors)
    variances = np.concatenate(accumulator.variances)
    marginal_scale = conformal_quantile(nonconformity_scores(errors, variances), alpha)
    result: dict[str, Any] = {
        "coverage": float(accumulator.coverage),
        "landmark_mode": bool(accumulator.landmark_mode),
        "truth_type": str(accumulator.truth_type),
        "alpha": float(alpha),
        "nominal_coverage": float(1.0 - alpha),
        # Keep the original field name for profile compatibility. Its scope is
        # deliberately explicit: this is a marginal pointwise radius, not a
        # simultaneous guarantee for an entire completed surface.
        "conformal_scale": float(marginal_scale) if np.isfinite(marginal_scale) else None,
        "conformal_scope": "marginal_pointwise_euclidean_error",
        "conformal_scope_note": (
            "Calibrated on pooled inferred-locus Euclidean errors. It targets marginal "
            "pointwise coverage and is not a simultaneous whole-shape guarantee."
        ),
        "fit_count": int(len(accumulator.fit_ids)),
        "mesh_count": int(len(set(accumulator.groups))),
        "point_count": int(len(errors)),
        "success_threshold_fraction_of_bbox_diagonal": float(success_threshold_fraction),
        "success_rate": float(np.mean(accumulator.successful)),
        "missing_rms_median": float(np.median(accumulator.missing_rms)),
        "normalized_sigma2_median": float(np.median(accumulator.normalized_sigma2)),
    }
    if not np.isfinite(marginal_scale):
        result["conformal_calibrator_status"] = (
            "Not available: too few inferred-locus calibration scores for the requested miscoverage alpha."
        )

    # A second, much more conservative scale treats complete meshes as the
    # exchangeable calibration units. Each mesh contributes the maximum
    # standardized inferred-locus error over all generated fragments. When the
    # finite conformal quantile exists, multiplying every point's posterior SD by
    # this scale gives a meshwise simultaneous envelope under the calibration
    # masking protocol.
    group_maxima = _groupwise_max_nonconformity(accumulator)
    simultaneous_scale = conformal_quantile(
        np.asarray(list(group_maxima.values()), dtype=np.float64), alpha
    )
    result["simultaneous_scope"] = (
        "meshwise_all_evaluated_inferred_loci_under_the_calibration_masking_protocol"
    )
    result["simultaneous_calibration_mesh_count"] = int(len(group_maxima))
    if np.isfinite(simultaneous_scale):
        result["simultaneous_meshwise_conformal_scale"] = float(simultaneous_scale)
    else:
        result["simultaneous_meshwise_conformal_scale"] = None
        result["simultaneous_calibrator_status"] = (
            "Not available: too few independent complete meshes for the requested miscoverage alpha."
        )

    unique_groups = sorted(set(accumulator.groups))
    if len(unique_groups) >= 5:
        audit_groups = set(unique_groups[::5])
        calibration_positions = [
            i for i, group in enumerate(accumulator.groups) if group not in audit_groups
        ]
        audit_positions = [
            i for i, group in enumerate(accumulator.groups) if group in audit_groups
        ]
        if calibration_positions and audit_positions:
            calibration_errors = np.concatenate(
                [accumulator.errors[i] for i in calibration_positions]
            )
            calibration_variances = np.concatenate(
                [accumulator.variances[i] for i in calibration_positions]
            )
            audit_errors = np.concatenate([accumulator.errors[i] for i in audit_positions])
            audit_variances = np.concatenate(
                [accumulator.variances[i] for i in audit_positions]
            )
            audit_scale = conformal_quantile(
                nonconformity_scores(calibration_errors, calibration_variances), alpha
            )
            audit_radius = audit_scale * np.sqrt(np.maximum(audit_variances, _TINY))
            audit: dict[str, Any] = {
                "calibration_mesh_count": len(
                    set(accumulator.groups[i] for i in calibration_positions)
                ),
                "audit_mesh_count": len(audit_groups),
                "audit_point_count": int(len(audit_errors)),
                "marginal_pointwise_empirical_coverage": float(
                    np.mean(audit_errors <= audit_radius)
                ),
                # Backward-compatible name retained alongside the explicit one.
                "empirical_coverage": float(np.mean(audit_errors <= audit_radius)),
                "conformal_scale": float(audit_scale),
            }
            calibration_group_maxima = _groupwise_max_nonconformity(
                accumulator, calibration_positions
            )
            audit_simultaneous_scale = conformal_quantile(
                np.asarray(list(calibration_group_maxima.values()), dtype=np.float64), alpha
            )
            if np.isfinite(audit_simultaneous_scale):
                audit_group_success = []
                for group in sorted(audit_groups):
                    positions = [
                        i
                        for i in audit_positions
                        if accumulator.groups[i] == group
                    ]
                    group_errors = np.concatenate(
                        [accumulator.errors[i] for i in positions]
                    )
                    group_variances = np.concatenate(
                        [accumulator.variances[i] for i in positions]
                    )
                    group_radius = audit_simultaneous_scale * np.sqrt(
                        np.maximum(group_variances, _TINY)
                    )
                    audit_group_success.append(bool(np.all(group_errors <= group_radius)))
                audit["meshwise_simultaneous_empirical_coverage"] = float(
                    np.mean(audit_group_success)
                )
                audit["simultaneous_meshwise_conformal_scale"] = float(
                    audit_simultaneous_scale
                )
            else:
                audit["meshwise_simultaneous_empirical_coverage"] = None
                audit["simultaneous_meshwise_conformal_scale"] = None
            result["held_out_mesh_audit"] = audit

    success = np.asarray(accumulator.successful, dtype=bool)
    sigma = np.asarray(accumulator.normalized_sigma2, dtype=np.float64)
    if len(success) >= 8 and np.unique(success).size == 2:
        classifier = RidgeLogisticCalibrator.fit(sigma, success, l2=logistic_l2)
        probability = classifier.probability(sigma)
        result["fit_success_calibrator"] = classifier.to_dict()
        result["fit_success_apparent_metrics"] = {
            "auc": auc_rank(-np.log(np.maximum(sigma, _TINY)), success),
            "brier": float(np.mean((probability - success.astype(float)) ** 2)),
            "ece_10bin": expected_calibration_error(probability, success, 10),
        }
    else:
        result["fit_success_calibrator"] = None
        result["fit_success_calibrator_status"] = (
            "Not fitted: at least eight fits and both success classes are required."
        )
    return result

def build_calibration_profile(
    *,
    fingerprint: str,
    entries: Sequence[Mapping[str, Any]],
    generated_from_ssm_training_set: bool,
    source_description: str,
    settings_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    warning = (
        "Internal calibration: these complete meshes were used to fit the SSM. Intervals and probabilities may be optimistic for new specimens."
        if generated_from_ssm_training_set
        else "External calibration: the supplied complete meshes were declared independent of SSM fitting."
    )
    return {
        "schema": "MorphoWeaveShapeCompletionCalibration",
        "version": PROFILE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_fingerprint_sha256": str(fingerprint),
        "method_signature_sha256": calibration_method_signature(settings_snapshot),
        "generated_from_ssm_training_set": bool(generated_from_ssm_training_set),
        "calibration_scope_warning": warning,
        "source_description": str(source_description),
        "settings": _jsonable(dict(settings_snapshot)),
        "entries": [_jsonable(dict(entry)) for entry in entries],
    }


def save_calibration_profile(profile: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(profile)), indent=2, sort_keys=True), encoding="utf-8")


def load_calibration_profile(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "MorphoWeaveShapeCompletionCalibration":
        raise ValueError("not a MorphoWeave shape-completion calibration profile")
    if int(data.get("version", -1)) != PROFILE_VERSION:
        raise ValueError("unsupported calibration profile version")
    if not isinstance(data.get("entries"), list):
        raise ValueError("calibration profile has no entries")
    return data


def select_calibration_entry(
    profile: Mapping[str, Any],
    *,
    fingerprint: str,
    coverage: float,
    landmark_mode: bool,
    current_settings: Optional[Mapping[str, Any] | CompletionSettings] = None,
    max_coverage_delta: float = 0.05,
) -> tuple[Optional[dict[str, Any]], str]:
    if profile.get("model_fingerprint_sha256") != fingerprint:
        return None, "Calibration profile belongs to a different SSM."
    if current_settings is not None:
        expected_signature = profile.get("method_signature_sha256")
        current_signature = calibration_method_signature(current_settings)
        if expected_signature and expected_signature != current_signature:
            return None, "Calibration profile was generated with different completion or uncertainty settings."
    candidates = [
        dict(entry)
        for entry in profile.get("entries", [])
        if bool(entry.get("landmark_mode")) == bool(landmark_mode)
    ]
    if not candidates:
        mode = "landmark-assisted" if landmark_mode else "surface-only"
        return None, f"Calibration profile has no {mode} entry."
    selected = min(
        candidates,
        key=lambda entry: (
            abs(float(entry["coverage"]) - float(coverage)),
            0 if entry.get("truth_type") == "exact_dense_correspondence" else 1,
        ),
    )
    delta = abs(float(selected["coverage"]) - float(coverage))
    if delta > float(max_coverage_delta):
        return None, (
            f"Nearest calibrated coverage is {float(selected['coverage']):.2f}, which is farther than the allowed {max_coverage_delta:.2f}."
        )
    return selected, (
        f"Using {selected['truth_type']} calibration at coverage {float(selected['coverage']):.2f} "
        f"(requested {float(coverage):.2f})."
    )


def calibrated_radius(
    total_variance: np.ndarray, entry: Mapping[str, Any]
) -> Optional[np.ndarray]:
    value = entry.get("conformal_scale")
    if value is None or not np.isfinite(float(value)):
        return None
    variance = np.asarray(total_variance, dtype=np.float64)
    return float(value) * np.sqrt(np.maximum(variance, _TINY))


def calibrated_simultaneous_radius(
    total_variance: np.ndarray, entry: Mapping[str, Any]
) -> Optional[np.ndarray]:
    value = entry.get("simultaneous_meshwise_conformal_scale")
    if value is None or not np.isfinite(float(value)):
        return None
    variance = np.asarray(total_variance, dtype=np.float64)
    return float(value) * np.sqrt(np.maximum(variance, _TINY))


def fit_success_probability(normalized_sigma2: float, entry: Mapping[str, Any]) -> Optional[float]:
    value = entry.get("fit_success_calibrator")
    if not value:
        return None
    calibrator = RidgeLogisticCalibrator.from_dict(value)
    return float(np.asarray(calibrator.probability(float(normalized_sigma2))))


def completion_diagnostics(
    result: CompletionResult,
    settings: CompletionSettings,
    *,
    landmark_count: int,
    calibration_entry: Optional[Mapping[str, Any]] = None,
    calibration_message: str = "No calibration profile selected.",
) -> dict[str, Any]:
    pose = result.pose
    atlas = result.atlas
    diagnostics: dict[str, Any] = {
        "coverage": float(settings.coverage),
        "landmark_mode": bool(settings.use_landmarks and landmark_count >= 3),
        "landmark_count": int(landmark_count),
        "landmark_sigma": result.landmark_sigma,
        "landmark_nominal_sigma": result.landmark_nominal_sigma,
        "landmark_dense_mapping_rms": result.landmark_mapping_rms,
        "landmark_dense_mapping_max": result.landmark_mapping_max,
        "retained_modes": int(result.retained_modes),
        "retained_variance": float(result.retained_variance),
        "source_prescale": float(result.source_scale),
        "residual_scale_free": bool(settings.with_scale()),
        "world_scale": float(result.world_scale),
        "surface_sigma2": float(atlas.sigma2),
        "normalized_surface_sigma2": float(result.normalized_sigma2),
        "atlas_iterations": int(atlas.iterations),
        "atlas_difference": float(atlas.difference),
        "pose_score_within_run_only": float(pose.score),
        "pose_score_margin_within_run_only": float(pose.score_margin),
        "pose_posterior_entropy_within_run_only": float(pose.posterior_entropy),
        "pose_effective_hypotheses_within_run_only": float(pose.effective_hypotheses),
        "pose_hypotheses_evaluated": int(pose.hypotheses_evaluated),
        "pose_hypotheses_refined": int(pose.hypotheses_refined),
        "conditional_epistemic_std_median": float(np.median(np.sqrt(result.epistemic_variance))),
        "conditional_total_std_median": float(np.median(np.sqrt(result.total_variance))),
        "posterior_surface_only_mean_rms_from_constrained_atlas": float(
            result.posterior_mean_rms_from_atlas
        ),
        "posterior_noise_variance": float(result.posterior.noise_variance),
        "posterior_discrepancy_variance": float(result.posterior.discrepancy_variance),
        "uncertainty_scope": (
            "The completed point estimate is the constrained atlas fit. Its covariance is conditional on the selected pose and fitted surface correspondences; the current posterior does not integrate alternate pose basins or add target landmarks to posterior precision."
        ),
        "calibration_message": str(calibration_message),
    }
    landmark_rms = float(getattr(atlas, "landmark_rms", float("nan")))
    diagnostics["landmark_rms"] = landmark_rms if np.isfinite(landmark_rms) else None
    if result.landmark_sigma is not None and np.isfinite(landmark_rms):
        diagnostics["landmark_reduced_chi_square"] = float(
            (landmark_rms / (math.sqrt(3.0) * result.landmark_sigma)) ** 2
        )
    else:
        diagnostics["landmark_reduced_chi_square"] = None
    if calibration_entry is not None:
        diagnostics["calibration_truth_type"] = calibration_entry.get("truth_type")
        diagnostics["calibration_coverage"] = float(calibration_entry["coverage"])
        diagnostics["calibration_nominal_coverage"] = float(calibration_entry["nominal_coverage"])
        diagnostics["calibration_pointwise_radius_available"] = bool(
            calibration_entry.get("conformal_scale") is not None
        )
        diagnostics["calibration_conformal_scope"] = calibration_entry.get(
            "conformal_scope", "marginal_pointwise_euclidean_error"
        )
        diagnostics["calibration_conformal_scope_note"] = calibration_entry.get(
            "conformal_scope_note"
        )
        diagnostics["calibration_simultaneous_scope"] = calibration_entry.get(
            "simultaneous_scope"
        )
        diagnostics["calibration_simultaneous_radius_available"] = bool(
            calibration_entry.get("simultaneous_meshwise_conformal_scale") is not None
        )
        diagnostics["calibrated_fit_success_probability"] = fit_success_probability(
            result.normalized_sigma2, calibration_entry
        )
    else:
        diagnostics["calibration_pointwise_radius_available"] = False
        diagnostics["calibration_simultaneous_radius_available"] = False
        diagnostics["calibrated_fit_success_probability"] = None
    return diagnostics


def _points(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        raise ValueError(f"{name} must have non-empty shape (N, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _unique_label_map(labels: Sequence[str], source: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, label in enumerate(labels):
        normalized = normalize_label(label)
        if not normalized:
            continue
        if normalized in result:
            raise ValueError(f"duplicate {source} landmark label: {label}")
        result[normalized] = index
    return result


def _require_noncollinear(points: np.ndarray, label: str) -> None:
    centered = np.asarray(points, dtype=np.float64) - np.mean(points, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    if len(singular) < 2 or singular[0] <= _TINY or singular[1] / singular[0] < 1e-4:
        raise ValueError(f"{label} landmarks are collinear or nearly collinear")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
