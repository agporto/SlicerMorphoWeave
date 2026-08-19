#!/usr/bin/env python3
"""Synthetic benchmark for the Shape Completion sparse transfer operator.

This script is not registered as a unit test because its default one-million-vertex
case is intentionally a performance and memory exercise.  It can be run with a
smaller ``--vertices`` value for quick checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    import resource
except ImportError:  # Windows
    resource = None
import time

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[2]
CORE_DIR = MODULE_DIR / "Resources" / "Python"
sys.path.insert(0, str(CORE_DIR))

from MorphoWeaveShapeCompletionCore import build_local_transfer_operator  # noqa: E402


def peak_rss_mib() -> float | None:
    if resource is None:
        return None
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    if sys.platform == "darwin":
        value /= 1024.0
    return value / 1024.0


def run_benchmark(
    *,
    vertices: int,
    controls: int,
    neighbors: int,
    fields: int,
    chunk_size: int,
    seed: int,
) -> dict[str, object]:
    if vertices < 1 or controls < 1 or controls > vertices:
        raise ValueError("require 1 <= controls <= vertices")
    if fields < 1:
        raise ValueError("fields must be positive")

    rng = np.random.default_rng(seed)
    query = rng.normal(size=(vertices, 3)).astype(np.float64)
    source_ids = np.linspace(0, vertices - 1, controls, dtype=np.int64)
    source = query[source_ids].copy()

    started = time.perf_counter()
    operator = build_local_transfer_operator(
        query,
        source,
        neighbors=neighbors,
        sharpness=2.0,
        chunk_size=chunk_size,
        exact_source_vertex_indices=source_ids,
        workers=-1,
    )
    build_wall = time.perf_counter() - started

    values = rng.normal(size=(controls, fields)).astype(np.float32)
    started = time.perf_counter()
    output = operator.apply(values)
    apply_seconds = time.perf_counter() - started

    row_sums = np.asarray(operator.matrix.sum(axis=1)).reshape(-1)
    anchor_error = float(np.max(np.abs(output[source_ids] - values), initial=0.0))
    return {
        "query_vertices": int(vertices),
        "source_controls": int(controls),
        "neighbors": int(operator.neighbors),
        "fields": int(fields),
        "nonzeros": int(operator.matrix.nnz),
        "build_seconds_internal": float(operator.build_seconds),
        "build_seconds_wall": float(build_wall),
        "apply_fields_seconds": float(apply_seconds),
        "operator_memory_mib": float(operator.memory_bytes / (1024.0 * 1024.0)),
        "peak_rss_mib": peak_rss_mib(),
        "exact_anchor_count": int(operator.exact_anchor_count),
        "max_anchor_abs_error": anchor_error,
        "max_row_sum_error": float(np.max(np.abs(row_sums - 1.0), initial=0.0)),
        "output_shape": [int(value) for value in output.shape],
        "output_dtype": str(output.dtype),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertices", type=int, default=1_000_000)
    parser.add_argument("--controls", type=int, default=5_000)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--fields", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--json", type=Path, help="Optional path for JSON output")
    arguments = parser.parse_args(argv)
    result = run_benchmark(
        vertices=arguments.vertices,
        controls=arguments.controls,
        neighbors=arguments.neighbors,
        fields=arguments.fields,
        chunk_size=arguments.chunk_size,
        seed=arguments.seed,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if arguments.json:
        arguments.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
