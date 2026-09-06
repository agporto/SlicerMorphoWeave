# Local restoration of the working fragment implementation

This module restores the matched widget and numerical core from the user's
`MorphoWeaveShapeCompletion.zip` and retains Batch **as a tab inside Shape
Completion**. It is a local deliverable. No repository branch has been changed.

## Preserved exactly

`Resources/Python/MorphoWeaveShapeCompletionCore.py` is byte-for-byte identical
to the uploaded core. Its SHA-256 is:

```
aa936d91e962a0c14b1a7a7737df6330ce672f1c99291f43bcffce24f183b274
```

The uploaded widget lives in `Resources/Python/MorphoWeaveShapeCompletionBase.py`.
All 62 original non-installer functions/methods retain identical Python ASTs,
including UI construction and defaults, settings extraction, single completion,
calibration, geometry transfer, and export. Only dependency preflight was changed
there, to enforce the previously requested `rustcpd==4.0.0` requirement.

This preserves, rather than reinvents:

- Coverage-consistent scale candidates from model regions, bounded automatic
  fragment scaling, and the 0.10 scale-bound margin.
- Six requested translation anchors for fragments below 0.95 coverage, with the
  original backend completeness threshold and activation rules.
- Optional adaptive mixing, initial pose variance and hypothesis merging controls.
- Atlas initialization variance based on the selected pose's nearest-distance
  residual, with the original default multiplier of 2.0.

The two Batch implementation files are copied **unchanged** from
`agporto/SlicerMorphoWeave` commit `eea6dcf0609726ef936224bd7c676fd9623020d8`.
Their Git blob hashes are recorded in `RESTORATION_PROVENANCE.json`. Batch calls
the same `_run_completion_impl` and settings extractor as Complete Shape; it
is not a second fitting implementation. Resume fingerprints include the core
and settings, so outputs from a different implementation cannot be silently
reused. Choose a fresh output directory for restored runs.

## Integration fixes retained

The public entry point explicitly names `ScriptedLoadableModule` as a base so
Extension Wizard can discover it. It also mirrors automatic SSM selections
between the two tabs after signal blockers are released. The installed backend
must report exactly `4.0.0` and expose the required fragment APIs, including
`register_atlas(..., sigma2=...)`. No installer requests an older rustcpd version.
A different already loaded backend requires restarting before package changes.

## Install without losing the working baseline

Close Slicer. Keep the uploaded working module and your original checkout intact.
In a **separate copy** of your existing `shape-completion` checkout, replace the
entire `MorphoWeaveShapeCompletion` folder with this folder. Put any backup outside
the scanned extension directory; do not leave a second discoverable module in it.
The branch already registers this directory in its top-level CMakeLists.txt.

Open Slicer, select that checkout in Extension Wizard, and load Shape Completion.
The expected tabs are **Complete Shape / Batch / Calibration / Advanced**.
Remove stale Additional module paths pointing to other copies before testing.
This ZIP contains only Shape Completion and does not change Landmark Transfer,
Surface Segmentation, the top-level extension files, or your input data.

Start with the same fragment, atlas, coverage, and settings that worked in the
uploaded version. Check Complete Shape first. Batch uses the same model and
Advanced settings; all specimens in one batch share the same coverage and seed.
Optional paired landmarks must have the same basename as the mesh. Use separate
batches when coverage or other settings differ among specimens.

## Tests and what they establish

From this module's parent directory, run:

```bash
python MorphoWeaveShapeCompletion/Testing/Python/ShapeCompletionCoreTest.py
python MorphoWeaveShapeCompletion/Testing/Python/ShapeCompletionModuleSourceTest.py
python MorphoWeaveShapeCompletion/Testing/Python/FragmentRestorationTest.py
```

The restoration suite uses the actual widget setup, settings extraction,
single-run implementation, numerical Python core, and batch adapter/runner.
Qt/MRML and native registration are simulated; VTK file I/O is real. It compares
all prepared registration arguments and the exported arrays for single versus
batch, including landmark-assisted fragments, modified fragment settings,
complete-coverage mode, fixed scale, and posterior sampling. It also checks
resume, changed-settings rejection, missing landmarks, cancellation, automatic
selection, discovery, exact pinning, and source preservation.

The original core suite includes an optional native displaced-fragment regression.
That test cannot pass by silently skipping when used for native signoff: check
that rustcpd 4.0.0 is installed and the native test actually executes. The
package-level installer tests refer to an installer not supplied in this archive
and are skipped. This is not a complete rerun of the repository's test suites.

**Not validated here:** live Slicer GUI behavior, native rustcpd fitting, your
biological specimens, or uncertainty calibration. The native backend is not
installed in this execution environment. Passing simulated-backend tests does
not prove native fitting accuracy or single/batch numerical equality with Rust.

## Deliberately preserved limitation: uncertainty under rustcpd 4.0.0

The uploaded core passes `prior_temperature = 1 / atlas_lambda`. That is preserved
exactly here, along with the rest of the successful implementation. The upstream
4.0 migration guide instead defines this parameter as a **precision multiplier**
and says matching the atlas prior requires `prior_temperature = atlas_lambda`.
Thus this restoration does **not** establish a matching posterior prior or
calibrated uncertainty under 4.0.0. Its posterior comments also retain assumptions
from the earlier backend. These are not silently corrected in a restoration
whose purpose is to preserve the user's successful numerical path.

The point estimate is the atlas-fit geometry, not the posterior mean. Addressing
the posterior mismatch should be a separately validated change, after restored
fragment fitting is confirmed. Likewise, replacing the pose-residual variance
heuristic with `initial_state` continuation is not part of this restoration.

Upstream source:
https://github.com/agporto/rustcpd/blob/v4.0.0/docs/MIGRATING_4.md
