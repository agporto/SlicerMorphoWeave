# Model Library correctness and validation

## Input contract

Population files and the dense template must contain the same nonzero number of
finite, defined 3D points in correspondence order. Import rejects the entire
cohort on a malformed row, missing/undefined position, non-finite coordinate,
empty file, or count mismatch. No point or specimen is silently skipped.

FCSV accepts the Slicer coordinate tags `RAS`/`0` and `LPS`/`1`, ignoring outer
whitespace and letter case. JSON accepts the same explicit tags. Unrecognized
tags are rejected. Legacy files with no coordinate tag retain the existing RAS
assumption. JSON files must contain exactly one markup object. FCSV `columns`
headers are honored, including CSV-quoted labels; files without a columns header
use the standard id/x/y/z and optional label positions.

Point order is checked when two files share a complete, unique label set or ID
set. References are tracked for every comparable set across the full cohort,
not just the dense template and first specimen. A permutation is rejected, not
automatically reordered. File-specific or
missing labels/IDs cannot prove biological correspondence: those files continue
to use their given point order. The template and population must already be in
a compatible coordinate frame. The library does not perform GPA or infer
homology during ingestion.

At least two distinct specimens with nonzero numerical variation are required.
A singleton or identical cohort receives an actionable rejection before anything
is persisted. Old mean-only database entries also fail validation before scene
mutation rather than failing in a zero-column reshape.

## PCA convention

The mean and modes retain the existing storage precision (float32), while
covariance eigenvalues are float64. Modes have unit Euclidean norm and
eigenvalues are in squared coordinate units. Explorer sliders multiply each
mode by `z * sqrt(eigenvalue)`. Full retention (`variance_threshold=1`, `None`,
or a large `max_modes`) now retains only positive numerical-rank modes, never
centered nullspace or negative Gram eigenvalues. SVD and Gram calculations use
scale-relative rank tolerances.

## Database replacement

Names must be safe single directory names. The UI checks names before offering
an overwrite prompt and passes the selected library root to ingestion. The
logic checks direct-child containment before staging and again before swapping,
rejects traversal, symlinks/aliased roots, source-population ancestors, and
nonempty directories that do not identify as individual databases. Direct API
callers may pass `libraryRootPath` to enforce their selected root; omitting it
does not bypass the existing-directory/source protections. Existing database
entries containing subdirectories or symlinks require a different destination,
not recursive replacement. These checks prevent a name such as `.` from
replacing the library and removing sibling entries.

Inputs are validated first, including real VTK template geometry. A complete
new generation is written to a sibling temporary directory and read back before
publication. Replacing an existing entry uses a same-filesystem directory
rename to a backup, followed by publication and removal of the backup. An
ordinary publication failure restores the previous directory; copy, NPZ,
manifest, and pre-publication verification failures leave it untouched.

This is a rollback-safe replacement, not a multi-process transaction or a
power-loss durability guarantee. A reader can observe a brief missing-directory
interval between renames. Concurrent writers are unsupported. If the filesystem
also prevents rollback, the error identifies the retained backup directory for
recovery; do not discard that backup. Backup cleanup failure after publication
is logged without incorrectly reporting the completed save as failed.

## Explorer

Changing the displayed PC count resets sliders, mesh vertices, and landmarks to
the same template-centered baseline immediately. It does not redefine zero as
the learned population mean.

The experimental biharmonic backend now uses correctly assigned cotangents and
minimizes **displacement** bending relative to immutable rest geometry.
Triangulation carries explicit original vertex IDs; switching from Local RBF
never recaches the currently deformed display as the biharmonic rest shape.
Repeated nearest-vertex constraints contribute an order-independent soft
least-squares average. An unconstrained disconnected component rejects the
biharmonic precompute so the caller can use its existing Local RBF fallback.
These repairs do not make the optional backend biologically validated, exact at
every control, or invariant to arbitrary mesh resolution/scale. Local RBF is
still a normalized Gaussian KNN smoother, not an exact interpolant.

## Regression tests

Run outside Slicer with NumPy, SciPy, and VTK installed:

```sh
python -W error::ResourceWarning MorphoWeaveModelLibrary/Testing/Python/ModelLibraryCorrectnessTest.py
```

The suite executes the actual production logic/widget classes compiled verbatim
from their AST. NumPy/SciPy algebra, VTK geometry/readers/writers, and temporary
filesystem operations are real. Only Slicer MRML scene/node adapters and Qt
slider/layout boundaries are explicit doubles. The test is registered with
`slicer_add_python_unittest` in CMake, alongside the existing module smoke test.

Coverage includes:

- AM01/AM02: numeric/string RAS/LPS equivalence, malformed rows, undefined/empty/
  non-finite points, shared-label/ID permutations, and all-zero valid positions.
- AM03: failure at each of three copies, NPZ and manifest writes, both directory
  rename steps, missing source, corrupt staged NPZ, successful replacement,
  unsafe UI names, library/source preservation, symlink/ancestor aliases,
  unrelated manifests, and backup cleanup failure.
- AM07/AM08: identity, tiny and finite translations, analytic triangle weights,
  winding/vertex permutation, duplicate controls, immutable backend switching,
  and unconstrained disconnected-component rejection.
- AM10: rebuilding sliders immediately restores both mesh and landmark geometry.
- AM11/AM12: zero-variance rejection, template/population mismatch, invalid mesh,
  invalid arrays before scene mutation, authoritative manifest roles, and the
  actual loader with explicit scene doubles.
- AM13: both PCA branches, full retention and low rank, covariance reconstruction,
  positive eigenvalues, orthonormal modes, parameter validation, and input-unit
  scales from 1e-9 to 1e9.

Not covered here: live Slicer reader/writer behavior, MRML observers and parent
transform hierarchies, actual Qt signal ordering, reselecting nodes while they
are deformed, CMake configure/build, biological correspondence accuracy, or
crash/concurrent-writer recovery. Live Slicer integration validation remains
necessary before claiming those behaviors are tested.
