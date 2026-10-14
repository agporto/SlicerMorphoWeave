#!/usr/bin/env python3
"""Native-only, paired completion smoke benchmark; see VALIDATION_PROTOCOL.md."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
from importlib.machinery import EXTENSION_SUFFIXES
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

CORE_DIR = Path(__file__).resolve().parents[2] / "Resources" / "Python"
sys.path.insert(0, str(CORE_DIR))
import MorphoWeaveShapeCompletionCore as core


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_backend():
    try:
        backend = importlib.import_module("rustcpd")
        native = importlib.import_module("rustcpd._core")
    except ImportError as exc:
        raise RuntimeError("Native rustcpd==4.0.0 is required; no fake-backend accuracy run is allowed.") from exc
    if (backend.__version__ != "4.0.0"
            or not any(str(native.__file__).endswith(suffix) for suffix in EXTENSION_SUFFIXES)
            or backend.register_atlas is not native.register_atlas
            or backend.pose_initialize is not native.pose_initialize):
        raise RuntimeError("Require unmodified native rustcpd==4.0.0 registration functions.")
    return backend, {"version": backend.__version__, "extension_sha256": sha256(native.__file__)}


def fixed_settings(**overrides):
    # Small, frozen smoke preset, NOT a biologically optimized production preset.
    values = dict(coverage=1.0, variance_keep=1.0, target_point_count=200,
                  coverage_prescale=True, residual_scale_policy="auto", use_landmarks=False,
                  landmark_sigma_percent=2.0, rotation_count=25, coarse_source_count=300,
                  coarse_target_count=200, coarse_rank=2, coarse_iterations=8,
                  coarse_screen_iterations=8, coarse_survivor_count=25, coarse_score_mode="trajectory",
                  refine_count=6, refine_source_count=None, refine_target_count=200, refine_iterations=25,
                  pose_lambda=5.0, pose_outlier_weight=.10, identity_prior_probability=.20,
                  atlas_lambda=5.0, atlas_outlier_weight=.10, atlas_max_iterations=100,
                  atlas_tolerance=1e-7, atlas_k=None, estimate_discrepancy=True,
                  posterior_samples=0, seed=0, parallel=False, single_precision=False,
                  translation_anchor_count=6, anchor_completeness_threshold=.9,
                  free_scale_bounds_fraction=.10, adaptive_mixing=None, initial_sigma2=None,
                  merge_tolerance=.02, atlas_sigma2_from_pose_factor=2.0)
    values.update(overrides)
    settings = core.CompletionSettings(**values)
    settings.validate()
    return settings


class HandoffAdapter:
    """Only changes the handoff in this experiment; Core.run_completion is unchanged."""
    def __init__(self, backend, continuation):
        self.backend, self.continuation = backend, continuation
        self.pose = None
        self.timings = {}
        self.initial_sigma2 = None

    def pose_initialize(self, *args, **kwargs):
        start = time.perf_counter()
        self.pose = self.backend.pose_initialize(*args, **kwargs)
        self.timings["pose_seconds"] = time.perf_counter() - start
        return self.pose

    def register_atlas(self, *args, **kwargs):
        if self.continuation:
            if self.pose is None:
                raise RuntimeError("State continuation requires a completed pose initialization.")
            for key in ("sigma2", "initial_coefficients", "initial_rotation", "initial_scale", "initial_translation"):
                kwargs.pop(key, None)
            kwargs["initial_state"] = self.pose.state
            self.initial_sigma2 = float(self.pose.state.sigma2)
        else:
            self.initial_sigma2 = kwargs.get("sigma2")
        start = time.perf_counter()
        result = self.backend.register_atlas(*args, **kwargs)
        self.timings["atlas_seconds"] = time.perf_counter() - start
        return result


def synthetic_cases():
    rng = np.random.default_rng(11)
    z = np.sort(rng.uniform(0, 1, 600))
    theta = rng.uniform(0, 2 * np.pi, len(z))
    radius = 4 + 6 * z + 9 * np.exp(-((z - 0.06) / 0.06) ** 2)
    mean = np.column_stack((100 * z, radius * np.cos(theta), radius * np.sin(theta)))
    modes = np.zeros((len(z), 3, 2))
    modes[:, 1, 0], modes[:, 2, 1] = np.sin(np.pi * z), np.sin(2 * np.pi * z)
    modes /= np.linalg.norm(modes.reshape(-1, 2), axis=0)
    rotation = np.array([[np.cos(.5), -np.sin(.5), 0], [np.sin(.5), np.cos(.5), 0], [0, 0, 1]])
    truth = (mean + modes @ np.array([1.5, -1.0])) @ rotation + [30, -20, 10]
    noisy = truth + rng.normal(scale=.05, size=truth.shape)
    cases = []
    for name, mask, coverage in (
            ("complete", np.ones(len(z), dtype=bool), 1.0),
            ("proximal_third", z < .33, .33),
            ("middle_third", (z >= .33) & (z < .66), .33),
            ("distal_third", z >= .67, .33),
            ("proximal_coverage_misspecified", z < .33, .60)):
        cases.append(dict(specimen_id="synthetic_tube_11", case_id=name, split="synthetic_smoke",
                          points=noisy[mask], complete_dense=truth, observed_mask=mask,
                          coverage=coverage, true_scale=1.0, external_status="not_external"))
    return (mean, modes, np.array([9., 4.])), cases


def validate_manifest(path):
    """NPZ-only exact-correspondence scaffold; external status is declared, not certified."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("version") != 1:
        raise ValueError("Manifest version must be 1.")
    training = manifest.get("ssm_training_ids")
    if not isinstance(training, list) or not training or any(not isinstance(x, str) or not x or x != x.strip() for x in training):
        raise ValueError("Declare nonempty ssm_training_ids; unknown training provenance cannot establish held-out evaluation.")
    if len(set(training)) != len(training):
        raise ValueError("Duplicate SSM training IDs.")
    if not isinstance(manifest.get("units"), str) or not manifest["units"].strip():
        raise ValueError("Declare common physical units for model, fragment, and complete truth.")
    model_path = path.parent / manifest["model_npz"]
    with np.load(model_path, allow_pickle=False) as data:
        model = tuple(data[key].copy() for key in ("mean", "modes", "eigenvalues"))
    core.validate_and_truncate_ssm(*model, 1.0)
    settings = asdict(fixed_settings(**manifest.get("settings", {})))
    if settings["use_landmarks"]:
        raise ValueError("This compact real-data scaffold does not support landmarks yet.")
    seen_ids, seen_files, seen_truth, cases, records = set(), set(), set(), [], []
    for item in manifest["specimens"]:
        specimen_id = item["specimen_id"]
        if not isinstance(specimen_id, str) or not specimen_id or specimen_id != specimen_id.strip() or specimen_id in seen_ids:
            raise ValueError("Invalid or duplicate specimen ID.")
        if specimen_id in training:
            raise ValueError("Split leakage: evaluation specimen is in SSM training IDs.")
        if item["split"] not in {"tuning", "calibration", "test"}:
            raise ValueError("Split must be tuning, calibration, or test.")
        if item.get("dense_correspondence_verified") is not True:
            raise ValueError("Exact dense correspondence must be explicitly verified by the data curator.")
        status = item.get("external_status", "unverified")
        if status not in {"user_declared_external", "not_external", "unverified"}:
            raise ValueError("Invalid external_status declaration.")
        specimen_path = path.parent / item["npz"]
        digest = sha256(specimen_path)
        with np.load(specimen_path, allow_pickle=False) as data:
            points, truth, mask = (data[key].copy() for key in ("points", "complete_dense", "observed_mask"))
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 20
                or truth.shape != model[0].shape or mask.shape != (len(truth),)
                or mask.dtype != np.bool_ or not mask.any()
                or not np.isfinite(points).all() or not np.isfinite(truth).all()):
            raise ValueError("Invalid points/complete_dense/observed_mask arrays.")
        truth_digest = hashlib.sha256(np.ascontiguousarray(truth, dtype="<f8").tobytes()).hexdigest()
        if digest in seen_files or truth_digest in seen_truth:
            raise ValueError("Duplicate specimen file or complete truth content, possibly under another ID.")
        seen_ids.add(specimen_id); seen_files.add(digest); seen_truth.add(truth_digest)
        coverage = float(item["coverage"])
        fixed_settings(**dict(settings, coverage=coverage))
        if coverage < 1.0 and mask.all():
            raise ValueError("Fragment coverage requires unobserved dense truth loci for missing-region evaluation.")
        scale = item.get("true_scale")
        if scale is not None and (not np.isfinite(scale) or scale <= 0):
            raise ValueError("true_scale must be positive and finite, or omitted.")
        record = dict(specimen_id=specimen_id, split=item["split"], coverage=coverage,
                      external_status=status, file_sha256=digest, truth_sha256=truth_digest)
        records.append(record)
        cases.append(dict(record, case_id=specimen_id, points=points, complete_dense=truth,
                          observed_mask=mask, true_scale=scale))
    if not cases:
        raise ValueError("Manifest has no specimens.")
    provenance = dict(manifest_sha256=sha256(path), model_sha256=sha256(model_path), units=manifest["units"],
                      ssm_training_ids=training, specimens=records,
                      external_status_note="User declarations only; provenance has not been independently audited.")
    return model, cases, settings, provenance


def evaluate_case(model, case, settings, backend, continuation):
    adapter = HandoffAdapter(backend, continuation)
    row = {key: case[key] for key in ("specimen_id", "case_id", "split", "coverage", "external_status")}
    row.update(handoff="experimental_state" if continuation else "current", settings=asdict(settings))
    start = time.perf_counter()
    try:
        result = core.run_completion(case["points"], *model, settings, rustcpd_module=adapter)
        errors = np.linalg.norm(result.completed_points - case["complete_dense"], axis=1)
        if not all(np.isfinite(values).all() for values in (
                errors, result.epistemic_variance, result.total_variance,
                result.world_scale, result.atlas.sigma2)):
            raise RuntimeError("Nonfinite completion errors, scale, sigma2, or uncertainty.")
        mask = case["observed_mask"]
        rms = lambda x: float(np.sqrt(np.mean(x * x))) if len(x) else None
        region_error = rms(errors[~mask]) if (~mask).any() else rms(errors)
        threshold = .03 * core.bbox_diagonal(case["complete_dense"])
        row.update(status="returned", rms=rms(errors), p95=float(np.quantile(errors, .95)),
                   observed_rms=rms(errors[mask]), missing_rms=rms(errors[~mask]),
                   scale_bias=None if case["true_scale"] is None else result.world_scale / case["true_scale"] - 1,
                   smoke_threshold=threshold, smoke_accuracy_failure=region_error > threshold,
                   actual_atlas_initial_sigma2=adapter.initial_sigma2,
                   diagnostics=core.completion_diagnostics(result, settings, landmark_count=0))
    except Exception as exc:
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    row.update(adapter.timings, elapsed_seconds=time.perf_counter() - start)
    return row


def json_clean(value):
    if isinstance(value, dict):
        return {k: json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", choices=["tuning", "calibration", "test"])
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.manifest:
        model, cases, preset, provenance = validate_manifest(args.manifest)
        if not args.validate_only and not args.split:
            parser.error("Select exactly one --split for a real-data run.")
        selected = [case for case in cases if case["split"] == args.split]
        if not args.validate_only and not selected:
            parser.error("The selected split has no specimens.")
        cases = selected
    else:
        if args.validate_only or args.split:
            parser.error("--validate-only and --split require --manifest.")
        model, cases = synthetic_cases()
        preset, provenance = asdict(fixed_settings()), {"synthetic_seed": 11, "units": "arbitrary_synthetic_units", "biological_validation": False}
    report = dict(schema_version=1, provenance=provenance, python=platform.python_version(),
                  numpy=np.__version__, platform=platform.platform(), settings=preset,
                  source_sha256={p.name: sha256(p) for p in [Path(__file__), *sorted(CORE_DIR.glob("*.py"))]},
                  uncertainty_note="Conditional on chosen pose/correspondences; no calibrated coverage guarantee.",
                  failure_definition="Exception/nonfinite output or missing-region RMS > 3% truth bbox diagonal (full RMS for complete cases). Descriptive smoke criterion only.",
                  results=[])
    if not args.validate_only:
        backend, report["native_backend"] = native_backend()
        for case in cases:
            settings = fixed_settings(**dict(preset, coverage=case["coverage"]))
            for continuation in (False, True):
                report["results"].append(evaluate_case(model, case, settings, backend, continuation))
    report["runtime_failures"] = sum(row["status"] == "failed" for row in report["results"])
    report["smoke_accuracy_failures"] = sum(row.get("smoke_accuracy_failure", False) for row in report["results"])
    report["exit_status_meaning"] = "Nonzero for runtime failures; inspect smoke_accuracy_failures separately."
    args.json.write_text(json.dumps(json_clean(report), indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.json}: {len(report['results'])} fits, {report['runtime_failures']} runtime failures, "
          f"{report['smoke_accuracy_failures']} smoke accuracy failures (exit status reflects runtime only).")
    return 1 if report["runtime_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
