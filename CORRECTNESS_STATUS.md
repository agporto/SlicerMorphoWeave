# Correctness review branch

Base: `shape-completion` at `e397882d54994479ef19279756b8beb82d382856`.
Branch: `fix/shape-completion-correctness-audit`.

These are scoped repairs with regression evidence, not certification that the
extension or reconstruction uncertainties are scientifically validated. Existing
outputs can change. Use a separate checkout and output directory, retain the
baseline, and regenerate Shape Completion calibration profiles.

## Implemented audit findings

| Area | Finding IDs | Repair |
|---|---|---|
| Model Library | AM01, AM02 | Interpret numeric RAS/LPS metadata; reject malformed, undefined, nonfinite or dimensionally incompatible points; check ordering where shared complete unique IDs/labels make it verifiable. |
| Model Library | AM03 | Stage and validate database entries before replacement; roll back ordinary copy/write/rename failures without destroying the existing entry. |
| Model Library | AM07, AM08, AM10 | Correct opposite-angle cotangents and rest-shape displacement solving; update visible geometry after SSM slider changes. |
| Model Library | AM11–AM13 | Reject zero-variance/unusable SSMs and mismatched template/population dimensions before persistence; filter numerical null PCA modes. Standard PCA eigenvalue units are unchanged. |
| Atlas Builder | AM04–AM06, AM09 | Propagate rigid/similarity normalization choice, preserve vertex IDs/topology, transfer control displacements, check output writers/cohort completeness, handle identity/all-constrained cases, and expose solve failures. |
| Landmark Transfer | LT01 | Transport mode displacements by the same prescale, full similarity linear block and external normalization as geometry. Do not rescale eigenvalues a second time. |
| Landmark Transfer | LT02–LT04 | Preserve landmark count/order, bound small-neighbor queries and final projection travel. If no surface point is reachable within the bound, retain the original landmark. |
| Landmark Transfer | LT05 | Forward native fixed-scale pose policy and preserve its optimized translation. |
| Landmark Transfer | LT06–LT08 | Correct TPS smoothing sign and small-control affine basis; correct cotangents; solve biharmonic displacements and retain unconstrained components. |
| Landmark Transfer | LT09, LT10 | Report true batch success/failure/cancellation, validate writes and duplicate stems; use the standardized coefficient prior without double eigenvalue weighting. |
| Shape Completion | Earlier correctness package | Match posterior prior precision to atlas lambda; reject invalid lambda; require calibration implementation provenance; evaluate calibration against closest triangle locations, not vertices. |
| Shape Completion | Scale-wrapper finding | Normalize the final atlas stage and convert explicit initial variance exactly once, preserving the SSM coefficient/eigenvalue convention. |
| Shape Completion | SC2, SC3 | Calculate classifier AUC from its actual probability ranking; validate candidate landmarks/dense truth against actual triangle surfaces when available. |
| Surface Segmentation | SS01, SS02 | Correct signed shape index; write and verify complete labeled VTP output; expose output counts/failures. Optional PLY remains explicitly eroded interiors, not an exhaustive partition. |
| Additional narrow cases | Transfer/segmentation audit | Correct HKS opposite-angle assembly, validate small-N settings, align pairing/output selection, and reject case-insensitive duplicate stems that would overwrite outputs. |

See each module's README (Surface Segmentation: `Testing/README.md`) for exact
behavior and testing boundaries. IDs refer to the full correctness audit; grouped
rows describe only the repaired scope, not every related design limitation.

## Explicitly not fixed here

- No rustcpd source changes. Native single-candidate failure handling (C1),
  zero-outlier extreme underflow (C2), and default-translation normalization (C3)
  remain upstream issues. The current wrapper explicitly supplies its translation.
- Unit-sensitive native coefficient stopping thresholds and tiny-eigenvalue
  floors remain. Optional TPS/biharmonic regularizers and some calibration
  variance floors also retain unit-dependent conventions. The wrapper fixes do
  not establish arbitrary-unit invariance for the entire pipeline.
- Surface-only calibration's one-direction success metric (SC1) is not a measure
  of complete shape coverage or homologous error. Conditional covariance still
  excludes alternate pose basins and landmark precision. Neither apparent AUC
  nor synthetic passing tests establishes calibrated biological uncertainty.
- Coverage/prescale sensitivity, posterior landmark precision, sampling-density
  dependence, full parent-transform support, and broad `skipScaling` semantics
  are not redesigned.
- Experimental biharmonic warps remain nearest-vertex/soft-constraint methods,
  not anatomical or as-rigid-as-possible guarantees. Segmentation lacks a new
  exhaustive mixed-boundary partition policy or stable cross-run segment IDs.
- Database replacement is rollback-safe for ordinary failures, not a
  crash-durable/concurrent-writer transaction. Multi-file specimen exports can
  retain partial successful outputs after a later failure; this is reported.

## Reproduce the headless regression gate

Use an isolated Python environment with NumPy, SciPy, VTK and `rustcpd==4.0.0`:

```sh
python Testing/run_headless_tests.py --require-native
```

The runner executes each module suite in a separate interpreter because Qt/MRML
boundary doubles deliberately modify import state. Tested dependency versions:
Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, VTK 9.3.1, rustcpd 4.0.0. There is no new
installer, vendored wheel, credential, biological dataset or audit bundle in this
branch. Native dependencies come from the normal package environment.

The suites exercise actual production methods, real NumPy/SciPy/VTK and native
rustcpd. Slicer scene/UI/markup I/O and selected registration boundaries are
explicit doubles. Four existing installer tests skip because this repository has
no installer script. Native tests must not be counted as passed when skipped.
No live Slicer, CMake configure/build, tiny3d end-to-end optimizer, or independent
biological validation was available. These remain separate release gates.

## Verified regression snapshot

On 2026-10-14 the combined command ran **283 tests: 279 passed, 4 skipped**
(the installer-only skips described above). Module totals were Atlas Builder 33,
Model Library 29, Landmark Transfer 71, Shape Completion 118 and Surface
Segmentation 32. Native rustcpd tests ran; none were skipped. Python compilation
and `git diff --check` also passed. Independent reviews challenged the numerical
contracts and output/error paths, including additional unit-rescaling cases.

No GitHub workflow is added by this branch. Use the command above for the local
gate; an absence of GitHub checks must not be read as a passing CI result.
