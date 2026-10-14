# Landmark Transfer correctness notes

These corrections retain the public workflow and default settings. They change
incorrect results or previously undefined edge cases; existing outputs should not
be assumed to reproduce bit-for-bit.

## Numerical contracts

- **PCA-CPD (LT-01):** SSM mode displacements receive the same source prescale,
  rigid/similarity linear transform, and external normalization as source points.
  Eigenvalues and coefficients retain their existing convention. The wrapper
  regression runs the same geometry in three length units, with source prescales
  1 and 3 and residual similarity scales 1 and 1.4.
- **Projection (LT-02, LT-03, LT-04):** Landmark count and ordering are preserved,
  including empty, singleton, and already-on-surface sets. The requested maximum
  movement bounds the *final* position, not only an intermediate solve. If the
  coherent surface candidate is too far away, the nearest point to the original
  landmark is tried. If no surface point is within the limit, the original
  landmark is retained, even if it remains inside or off the surface. The default
  still selects one outer connected region; this is not a multipart projection
  algorithm. Empty inputs return empty output; non-empty inputs need a polygonal
  surface. kNN graphs cap the number of neighbors to the available points.
- **Pose-EM (LT-05):** `with_scale` is forwarded to native `pose_initialize`, not
  merely applied after native shape/pose selection. Fixed-scale fitting keeps
  the optimized native translation instead of forcibly aligning the full-model
  and fragment centroids. Any external source prescale still composes into the
  final transform. Scores and diagnostics now correspond to this same model;
  an initializer that ignores the fixed-scale flag is rejected, not silently
  rewritten after optimization.
- **TPS (LT-06):** The 3D kernel uses `-r` consistently in fitting and evaluation,
  with positive diagonal smoothing. Positive lambda damps non-affine deformation
  toward the affine fit. Lambda zero retains interpolation for nondegenerate
  controls. The small-control affine fallback now uses the same coefficient order
  as evaluation, including after a constraint cap; fewer than four controls do
  not uniquely determine a 3D affine transform. No controls means no deformation.
  Negative/non-finite smoothing and nonpositive constraint caps are rejected.
- **Experimental biharmonic warp (LT-07, LT-08):** Half-cotangent edge weights use
  the correct opposite angles and face-major indexing. The solve is for
  displacements added to the original mesh, preserving its rest shape under
  identity constraints. A nearest-vertex handle receives `destination-source`,
  not an absolute destination that silently discards an off-vertex offset.
  Duplicate controls retain the existing policy: average their displacements,
  with one soft penalty per constrained vertex. Unconstrained connected
  components remain unchanged. The soft penalty is not a hard interpolant or an
  as-rigid-as-possible warp. Barycentric landmark transfer still projects each
  landmark onto the source mesh and carries that surface location; off-surface
  landmark offsets are not retained by that separate step.
- **Legacy candidate search (LT-10):** Candidates already multiply standardized
  coefficients by the square root of eigenvalues. Their prior is therefore
  `b @ b`, without a second inverse-eigenvalue weighting. Both scoring and pruning
  use the corrected prior. Full tiny3d optimization is not exercised by the
  standalone regression; the production nested scoring/pruning functions are.

## Batch result and failure contract (LT-09)

`runLandmarkBatch` now returns a dictionary:

- `total`, `attempted`, `succeeded`, `failed`, and `unprocessed` are counts.
- `cancelled` is a boolean. `cancelled_count` is one if an in-progress specimen
  was interrupted, zero for cancellation between specimens.
- `attempted == succeeded + failed + cancelled_count` in normal operation.
- `failures` contains each failed target filename, its error, and any requested
  specimen output paths already successfully written for it.
- `outputs` lists verified successful specimen output paths, including files
  retained when a later output for the same specimen fails.
- `warnings` reports auxiliary diagnostics-write errors. A fatal batch/setup
  exception is returned as `error`; validation errors before processing raise.

A specimen succeeds only after every requested mesh/landmark output succeeds.
Markup save booleans, VTK writer errors, output existence, and nonempty files are
checked. Successful files are not deleted because another target or later output
failed. The UI reports counts and errors, not unconditional completion. Cancellation
is checked between work stages and before writes; a running native solver is not
preempted internally. Duplicate case-insensitive output stems are rejected before
processing so different mesh extensions cannot silently overwrite each other.

## Tests and remaining integration scope

From the repository root, with NumPy, SciPy, VTK, and rustcpd 4.0.0 installed:

```sh
python -m unittest discover -s MorphoWeaveLandmarkTransfer/Testing/Python -p '*Test.py' -v
```

The geometry suite AST-loads actual production logic to avoid importing Qt/Slicer.
VTK, SciPy solves, and rustcpd are real. The batch suite substitutes MRML/application,
file-loading/markup-writing, and registration-stage boundaries while executing
actual batch control flow, TPS/projection, and VTK mesh writing. Native tests are
explicitly skipped if rustcpd is unavailable. These are not live-Slicer tests or
anatomical accuracy validation.

Still required before clinical/scientific reliance: live Slicer single and batch
runs on representative held-out specimens, real markup serialization, interrupted
and repeated UI stages, and transformed model/markup nodes. Broad parent-transform
support is not changed here. “Skip scaling” retains its existing behavior and is
not newly advertised as forbidding scale optimization in every downstream stage.
