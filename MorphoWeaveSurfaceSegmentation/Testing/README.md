# Surface segmentation correctness regression tests

Run from the repository root with Python, NumPy, SciPy and VTK installed:

```sh
PYTHONDONTWRITEBYTECODE=1 python MorphoWeaveSurfaceSegmentation/Testing/Python/SurfaceSegmentationCorrectnessTest.py -v
```

The suite is also registered with `slicer_add_python_unittest` when
`BUILD_TESTING` is enabled. The methods are extracted unchanged from the
production module using Python's AST so the standalone run does not need a
running Slicer/Qt application. NumPy, SciPy, VTK filters and VTK writers are real.
Scene access, Slicer file-loading and Qt controls use explicitly documented
boundary doubles. One intentionally failing writer double verifies that a
writer failure return cannot be counted as success; other file-output tests
use actual VTK writers and readers.

## Protected contracts

- SS-01: shape index uses signed-ordered principal curvatures internally, stays
  within [-1, 1], is continuous around a saddle and changes sign under global
  normal inversion. The standalone principal-curvature feature columns still
  use their existing magnitude ordering. A flat point is defined as S=0, C=0.
- SS-02: “Write .vtp with SegID & SegID_smooth” writes a complete labeled VTP,
  retaining raw labels and smoothed labels as distinct point arrays. Export
  uses the same cached world-coordinate mesh used for feature extraction.
- PLY interiors are now an explicit, separately selectable option, disabled by
  default. They deliberately retain only cells whose vertices all have the
  same nonnegative label. Mixed/unlabeled boundary cells and labels without an
  interior cell are reported, not silently counted as successful writes.
  This is an eroded-interior policy, not an exhaustive triangle partition;
  the labeled VTP is the complete-surface output. No majority-vote boundary
  assignment or triangle splitting has been introduced.
- Mesh writers must return success, have no VTK error code and create a
  nonempty new file. Each writes to a temporary file in the destination
  directory and replaces the final path only after those checks. Failed writes
  preserve an existing destination; temporary files are removed.
- The run result's `wrote` is the number of actual mesh files successfully
  written, not specimen attempts. `output_files` lists those mesh paths;
  `lookup_files` lists the NPY/TXT locus-label files separately. `n_used` means
  used to fit the segmentation, not “all outputs succeeded.” `n_skipped`,
  `n_export_failed`, `failures`, `warnings` and `segment_reports` expose partial
  outcomes. The UI reports those counts and specimen-specific issues.
- Pair matching is case-insensitive in both UI validation and execution.
  Duplicate stems, including different mesh extensions or case-only filename
  differences, fail preflight rather than silently selecting/overwriting one.
  A blank output field cannot implicitly select the current directory.
- Segmentation requires at least three corresponding loci, positive integer
  kNN, and integer `2 <= k < number_of_loci`. Invalid manual or auto-tune counts
  fail clearly instead of silently clamping or reaching an invalid sparse
  eigensolve. Graph helpers handle 0/1/2 loci without out-of-bounds indexing;
  clustering explicitly rejects fewer than three loci. Affinities must be
  finite, symmetric, nonnegative and have positive degree at every vertex.
- The optional HKS cotangent operator assigns each angle to its opposite edge
  with standard half-cotangent weights, passes analytic and face-order/winding
  tests, and bounds the sparse eigensolve below the vertex count. HKS remains
  disabled in the default run.

## Explicit limits and remaining validation

This suite is a numerical/file-output regression gate, not a live-Slicer
integration or anatomical-accuracy certification. Live scene node loading,
Qt signal wiring, preview rendering, parent transforms, cancellation and
held-out anatomical data still need Slicer testing. Inputs must already share
an atlas/world coordinate frame and consistent locus identities; this change
adds no independent specimen pose alignment. K-means/eigensolver randomness
and anatomical label interpretation are unchanged. Segment numbers are not
promised stable across separate runs.

PLY interiors can contain holes and can be empty for a small/mixed segment by
design. This change documents and reports those omissions; it does not decide
an exhaustive partition policy. The returned paths identify files from the
current run. Unrelated or stale files already in the output folder are not
deleted; use a separate output folder to keep run outputs isolated.
