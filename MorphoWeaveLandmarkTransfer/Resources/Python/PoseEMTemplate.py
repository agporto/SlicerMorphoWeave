from dataclasses import dataclass
import inspect
from typing import Any, Callable, Mapping, Optional

import numpy as np


LEGACY_BACKEND = "ransac"
POSE_EM_BACKEND = "pose_em"


def optimization_backend(parameters: Mapping[str, Any]) -> str:
    backend = str(parameters.get("optimizationBackend", LEGACY_BACKEND)).strip().lower()
    if backend not in (LEGACY_BACKEND, POSE_EM_BACKEND):
        raise ValueError(f"Unknown template optimization backend: {backend}")
    return backend


def pose_em_enabled(parameters: Mapping[str, Any], skip_optimization: bool = False) -> bool:
    return bool(not skip_optimization and optimization_backend(parameters) == POSE_EM_BACKEND)


@dataclass(frozen=True)
class PoseEMSettings:
    rotation_count: int = 193
    coarse_source_count: int = 400
    coarse_target_count: int = 400
    coarse_rank: int = 12
    coarse_iterations: int = 8
    coarse_screen_iterations: int = 8
    coarse_survivor_count: int = 193
    coarse_score_mode: str = "trajectory"
    refine_count: int = 12
    refine_source_count: Optional[int] = None
    refine_target_count: int = 1600
    refine_iterations: int = 30
    lambda_reg: float = 0.1
    outlier_weight: float = 0.05
    identity_prior_probability: float = 0.2
    seed: int = 0
    parallel: bool = True

    @classmethod
    def from_mapping(cls, parameters: Mapping[str, Any]) -> "PoseEMSettings":
        parallel = (
            bool(parameters["poseParallel"])
            if "poseParallel" in parameters
            else int(parameters.get("poseNJobs", -1)) != 1
        )
        settings = cls(
            rotation_count=int(parameters.get("poseRotationCount", 193)),
            coarse_source_count=int(parameters.get("poseCoarseSourceCount", 400)),
            coarse_target_count=int(parameters.get("poseCoarseTargetCount", 400)),
            coarse_rank=int(parameters.get("poseCoarseRank", 12)),
            coarse_iterations=int(parameters.get("poseCoarseIterations", 8)),
            coarse_screen_iterations=int(parameters.get("poseCoarseScreenIterations", 8)),
            coarse_survivor_count=int(parameters.get("poseCoarseSurvivorCount", 193)),
            coarse_score_mode=str(parameters.get("poseCoarseScoreMode", "trajectory")).strip().lower(),
            refine_count=int(parameters.get("poseRefineCount", 12)),
            refine_source_count=(
                None
                if int(parameters.get("poseRefineSourceCount", 0)) == 0
                else int(parameters.get("poseRefineSourceCount", 0))
            ),
            refine_target_count=int(parameters.get("poseRefineTargetCount", 1600)),
            refine_iterations=int(parameters.get("poseRefineIterations", 30)),
            lambda_reg=float(parameters.get("poseLambdaReg", 0.1)),
            outlier_weight=float(parameters.get("poseOutlierWeight", 0.05)),
            identity_prior_probability=float(parameters.get("poseIdentityPrior", 0.2)),
            seed=int(parameters.get("poseSeed", 0)),
            parallel=parallel,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
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
        }
        invalid = [name for name, value in positive.items() if value < 1]
        if invalid:
            raise ValueError(f"Pose EM settings must be positive: {', '.join(invalid)}")
        if self.coarse_screen_iterations > self.coarse_iterations:
            raise ValueError("coarse_screen_iterations must not exceed coarse_iterations")
        if self.coarse_survivor_count < min(self.refine_count, self.rotation_count):
            raise ValueError("coarse_survivor_count must cover the requested finalists")
        if self.coarse_score_mode not in ("trajectory", "final"):
            raise ValueError("coarse_score_mode must be 'trajectory' or 'final'")
        if self.refine_source_count is not None and self.refine_source_count < 1:
            raise ValueError("refine_source_count must be positive or None")
        if self.lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative")
        if not 0 <= self.outlier_weight < 1:
            raise ValueError("outlier_weight must be in [0, 1)")
        if not 0 < self.identity_prior_probability < 1:
            raise ValueError("identity_prior_probability must be in (0, 1)")

    def initializer_kwargs(self) -> dict[str, Any]:
        return {
            "rotation_count": self.rotation_count,
            "coarse_source_count": self.coarse_source_count,
            "coarse_target_count": self.coarse_target_count,
            "coarse_rank": self.coarse_rank,
            "coarse_iterations": self.coarse_iterations,
            "coarse_screen_iterations": self.coarse_screen_iterations,
            "coarse_survivor_count": self.coarse_survivor_count,
            "coarse_score_mode": self.coarse_score_mode,
            "refine_count": self.refine_count,
            "refine_source_count": self.refine_source_count,
            "refine_target_count": self.refine_target_count,
            "refine_iterations": self.refine_iterations,
            "lambda_regularization": self.lambda_reg,
            "outlier_weight": self.outlier_weight,
            "identity_prior_probability": self.identity_prior_probability,
            "seed": self.seed,
            "parallel": self.parallel,
        }


def _require_native_pose_initializer(initializer: Callable[..., Any]) -> None:
    """Fail clearly when the rustcpd native pose initializer is incomplete."""
    parameters = inspect.signature(initializer).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return
    required = {
        "coarse_screen_iterations",
        "coarse_survivor_count",
        "coarse_score_mode",
        "refine_source_count",
        "lambda_regularization",
        "parallel",
    }
    missing = sorted(required.difference(parameters))
    if missing:
        raise RuntimeError(
            "Pose EM template optimization requires the rustcpd native "
            "initializer API; the installed build is missing: " + ", ".join(missing)
        )


@dataclass(frozen=True)
class PoseEMRegistrationResult:
    points: np.ndarray
    coefficients: np.ndarray
    rotation: np.ndarray
    scale: float
    translation: np.ndarray
    score: float
    score_margin: float
    posterior_entropy: float
    effective_hypotheses: float
    hypotheses_evaluated: int
    hypotheses_refined: int
    final_parameters: Mapping[str, Any]

    def similarity_matrix(self) -> np.ndarray:
        return similarity_matrix(self.rotation, self.scale, self.translation)


def validate_ssm(mean: np.ndarray, modes: np.ndarray, eigenvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(mean, dtype=np.float64)
    modes = np.asarray(modes, dtype=np.float64)
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if mean.ndim != 2 or mean.shape[1] != 3 or len(mean) == 0:
        raise ValueError("SSM mean must have non-empty shape (M, 3)")
    if modes.ndim == 2:
        if modes.shape[0] != mean.size:
            raise ValueError("Flattened SSM modes do not match the mean")
        modes = modes.reshape(len(mean), 3, modes.shape[1])
    if modes.ndim != 3 or modes.shape[:2] != mean.shape:
        raise ValueError("SSM modes must have shape (M, 3, K) or (M*3, K)")
    if eigenvalues.ndim != 1 or modes.shape[2] != len(eigenvalues) or len(eigenvalues) == 0:
        raise ValueError("SSM eigenvalues do not match the modes")
    if not np.isfinite(mean).all() or not np.isfinite(modes).all() or not np.isfinite(eigenvalues).all():
        raise ValueError("SSM arrays must be finite")
    if np.any(eigenvalues <= 0):
        raise ValueError("SSM eigenvalues must be positive")
    return mean, modes, eigenvalues


def ssm_sample(mean: np.ndarray, modes: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    mean, modes, eigenvalues = validate_ssm(
        mean,
        modes,
        np.ones(np.asarray(modes).shape[-1], dtype=np.float64),
    )
    coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if len(coefficients) != len(eigenvalues):
        raise ValueError("SSM coefficients do not match the modes")
    return mean + (modes.reshape(mean.size, -1) @ coefficients).reshape(mean.shape)


def similarity_matrix(rotation: np.ndarray, scale: float, translation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Similarity rotation and translation must be 3D")
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Similarity scale must be finite and positive")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = float(scale) * rotation
    matrix[:3, 3] = translation
    return matrix


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or matrix.shape != (4, 4):
        raise ValueError("Expected points (N, 3) and a 4x4 transform")
    return np.c_[points, np.ones(len(points), dtype=np.float64)] @ matrix.T[:, :3]


def coverage_prescale(mean: np.ndarray, target: np.ndarray, coverage: float) -> float:
    mean = np.asarray(mean, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    coverage = float(np.clip(coverage, 1e-3, 1.0))
    source_diag = float(np.linalg.norm(np.ptp(mean, axis=0)))
    target_diag = float(np.linalg.norm(np.ptp(target, axis=0)))
    if source_diag <= np.finfo(float).eps:
        return 1.0
    return (target_diag / coverage) / source_diag


def fixed_scale_translation(source: np.ndarray, target: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    return (target.mean(axis=0) - source.mean(axis=0) @ rotation.T).reshape(1, 3)


class _RustPoseInitializationAdapter:
    """Expose rustcpd's direct-row rotation in MorphoWeave's legacy convention."""

    def __init__(self, initialization: Any):
        self._initialization = initialization
        # rustcpd 3 applies row points as points @ rotation. MorphoWeave's
        # established internal convention applies points @ rotation.T.
        self.rotation = np.asarray(initialization.rotation, dtype=np.float64).T.copy()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._initialization, name)


def _adapt_rustcpd_initialization(initialization: Any) -> Any:
    return _RustPoseInitializationAdapter(initialization)


def _rustcpd_api() -> Callable[..., Any]:
    try:
        from rustcpd import pose_initialize
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Pose EM template optimization requires rustcpd 3 or newer. "
            "Upgrade rustcpd in the Slicer Python environment."
        ) from exc

    def initialize(*args: Any, **kwargs: Any) -> Any:
        return _adapt_rustcpd_initialization(
            pose_initialize(*args, **kwargs)
        )

    return initialize


def run_pose_em_registration(
    mean: np.ndarray,
    target: np.ndarray,
    modes: np.ndarray,
    eigenvalues: np.ndarray,
    settings: PoseEMSettings,
    *,
    with_scale: bool,
    source_scale: float = 1.0,
    initializer: Optional[Callable[..., Any]] = None,
) -> PoseEMRegistrationResult:
    mean, modes, eigenvalues = validate_ssm(mean, modes, eigenvalues)
    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 3 or len(target) == 0 or not np.isfinite(target).all():
        raise ValueError("Target must have non-empty finite shape (N, 3)")
    settings.validate()
    if source_scale <= 0 or not np.isfinite(source_scale):
        raise ValueError("source_scale must be finite and positive")

    # Work in independent centroid frames so large Slicer world-coordinate
    # offsets do not reduce the precision of the similarity initialization.
    source_centroid = mean.mean(axis=0, keepdims=True)
    target_centroid = target.mean(axis=0, keepdims=True)
    working_mean = (mean - source_centroid) * float(source_scale)
    working_target = target - target_centroid
    working_modes = modes * float(source_scale)
    if initializer is None:
        initializer = _rustcpd_api()
    _require_native_pose_initializer(initializer)

    working_modes_flat = working_modes.reshape(
        working_mean.size,
        working_modes.shape[2],
    )
    initial = initializer(
        working_mean,
        working_target,
        working_modes_flat,
        eigenvalues,
        **settings.initializer_kwargs(),
    )
    coefficients = np.asarray(initial.coefficients, dtype=np.float64).reshape(-1)
    rotation = np.asarray(initial.rotation, dtype=np.float64)
    initial_shape = ssm_sample(working_mean, working_modes, coefficients)
    if with_scale:
        scale = float(initial.scale)
        translation = np.asarray(initial.translation, dtype=np.float64).reshape(1, 3)
    else:
        scale = 1.0
        translation = fixed_scale_translation(initial_shape, working_target, rotation)

    # The initializer has already refined its finalists. Template optimization
    # needs only those coefficients; a second dense atlas pass was redundant
    # and its registered coordinates were discarded by MorphoWeaveLandmarkTransfer.
    points_centered = scale * (initial_shape @ rotation.T) + translation
    final_scale = float(source_scale) * scale
    final_translation = (
        translation
        + target_centroid
        - final_scale * (source_centroid @ rotation.T)
    )
    points = np.asarray(points_centered, dtype=np.float64) + target_centroid
    final_parameters = {
        "b": coefficients.reshape(-1, 1),
        "R_world": rotation,
        "s_world": final_scale,
        "t_world": final_translation,
        "s_preconditioned": scale,
        "t_preconditioned": translation,
        "source_centroid": source_centroid.copy(),
        "target_centroid": target_centroid.copy(),
        "source_scale": float(source_scale),
        "pose_parallel": bool(settings.parallel),
    }
    return PoseEMRegistrationResult(
        points=points,
        coefficients=coefficients,
        rotation=rotation,
        scale=final_scale,
        translation=final_translation,
        score=float(initial.score),
        score_margin=float(initial.score_margin),
        posterior_entropy=float(initial.posterior_entropy),
        effective_hypotheses=float(initial.effective_hypotheses),
        hypotheses_evaluated=int(initial.hypotheses_evaluated),
        hypotheses_refined=int(initial.hypotheses_refined),
        final_parameters=final_parameters,
    )
