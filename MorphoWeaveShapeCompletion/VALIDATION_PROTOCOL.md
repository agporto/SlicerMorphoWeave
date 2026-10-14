# Shape Completion: reproducible validation scaffold

This is a small **native numerical smoke experiment**, not biological validation,
uncertainty calibration, or evidence that a production handoff should be changed.
It runs the actual `Core.run_completion` against **rustcpd 4.0.0**. There is no
Slicer runtime, mesh-file importer, simulated native accuracy, or hidden fallback.
The paired comparison changes no production source. No real specimens are bundled.

## Run

From the repository root, use an environment containing NumPy, SciPy and the pinned
native rustcpd. An isolated installation can be placed outside the repository:

```sh
python -m pip install --target /tmp/morphoweave-native-deps --no-deps rustcpd==4.0.0
PYTHONPATH=/tmp/morphoweave-native-deps python \
  MorphoWeaveShapeCompletion/Testing/Python/ShapeCompletionValidation.py \
  --json /tmp/morphoweave-validation-results.json
python -m unittest discover -s MorphoWeaveShapeCompletion/Testing/Python \
  -p ShapeCompletionValidationTest.py -v
```

The benchmark requires the compiled extension, verifies its version and the native
registration function identities, and records its SHA-256. Missing/incompatible
native code is a hard error, not a skipped successful accuracy test. Contract tests
use small stubs only to verify argument routing/failure handling, never accuracy.
A JSON report is written after the fits; runtime failures are retained in it and
produce exit status 1. Exit status 0 means operational completion, not scientific
accuracy: `smoke_accuracy_failures` separately counts the descriptive failures.
A returned fit with poor accuracy is a separate metric and
does not turn the benchmark into a prespecified clinical acceptance test.

## Frozen synthetic experiment

The generator uses seed 11, 600 samples of an asymmetric tapered tube, two smooth
normalized modes, eigenvalues `[9, 4]`, coefficients `[1.5, -1]`, rotation 0.5 rad,
translation `[30, -20, 10]`, true scale 1, and independent coordinate noise SD 0.05.
Units are arbitrary. This is intentionally comparable, not identical, to the
native displaced-fragment core test: it shares one noisy whole shape across cases.
Cases are complete, proximal/middle/distal thirds, and a proximal third knowingly
misreported as coverage 0.60. Coverage here describes the cut range; the report's
complete settings fix the numerical prescription and fitting seed 0. The compact
25-rotation, 200-target-point preset is not tuned for anatomy or representative of
all UI defaults. All paired cases use identical arrays and settings.

- `current`: passes Core's pose coefficients/rotation/scale/translation and
  pose-residual-derived `sigma2` unchanged to native atlas registration.
- `experimental_state`: captures that same pose result, removes those five
  initialization arguments, then passes `pose.state` as `initial_state`. This also
  carries native noise/outlier/mixing state. It is not a sigma-only experiment.
  Each arm independently reruns deterministic pose initialization, without caching.

Reports include full and missing/observed-region Euclidean correspondence RMS,
95th-percentile point error, relative scale bias, surface and normalized sigma2,
conditional epistemic/total SD, within-run pose ambiguity, exceptions and pose,
atlas, and overall wall times. Empty missing regions have `missing_rms: null`.
The descriptive smoke-failure flag uses missing RMS >3% of the truth bounding-box
diagonal (full RMS for a complete shape). This arbitrary prespecified smoke
threshold is not a scientific success probability. State-arm diagnostics retain
Core's computed but unused `atlas_initial_sigma2_from_pose`; the separate
`actual_atlas_initial_sigma2` records what its native state actually supplied.
Timing includes warm-up effects and is not a rigorous performance comparison.

A low conditional SD is **not** proof of correct pose, scale, missing-region shape,
or calibrated coverage. Posterior covariance conditions on one selected pose and
surface correspondences and currently excludes target-landmark precision. Pose
scores/margins are comparable only within a run. No empirical confidence interval,
external-validity claim, or claim of superiority follows from these five cases.

## Real-specimen NPZ manifest

Copy `VALIDATION_MANIFEST.example.json` and replace every placeholder. Relative
paths resolve beside the manifest. Nothing is downloaded or uploaded.

- Model NPZ: `mean` `(M,3)`, `modes` `(M,3,K)` or `(3*M,K)`, positive `eigenvalues`
  `(K,)`, following Core's SSM conventions. Model, fragment and truth must use the
  declared common physical units; the script cannot infer coordinate correctness.
- Specimen NPZ: `points` `(N,3)`, at least 20 observed target points;
  `complete_dense` `(M,3)` in the target frame with **known pointwise correspondence
  to model vertices**; boolean `observed_mask` `(M,)` defining observed versus
  withheld truth vertices. A declared fragment (`coverage < 1`) must have at
  least one withheld truth vertex. Points need not index directly into the truth array.
- Each entry requires an explicit specimen ID, one `tuning` / `calibration` / `test`
  split, expected `coverage`, and `dense_correspondence_verified: true` from the
  curator. This assertion is not independently verified by the script. Optional
  positive `true_scale` enables scale bias; omit it for real data with unknown scale.
- `ssm_training_ids` must enumerate the actual unique model-training specimens;
  empty/unknown training provenance is rejected rather than called held-out.
- `external_status` is `user_declared_external`, `not_external`, or `unverified`
  (default). An external declaration is a user assertion, not an independent audit.
  Explain provenance and institution/population differences separately.

```sh
python MorphoWeaveShapeCompletion/Testing/Python/ShapeCompletionValidation.py \
  --manifest manifest.json --validate-only --json /tmp/manifest-audit.json
PYTHONPATH=/tmp/morphoweave-native-deps python \
  MorphoWeaveShapeCompletion/Testing/Python/ShapeCompletionValidation.py \
  --manifest manifest.json --split tuning --json /tmp/tuning-results.json
# After freezing settings, use --split calibration, then finally --split test.
```

Manifest audit needs no native backend. Actual runs require exactly one explicit
split. All manifest rows are audited before any selected split runs. Duplicate
IDs, training/evaluation ID overlap, identical specimen files, and identical
complete-truth arrays repacked under another ID are rejected. Full file and truth
content hashes, manifest/model hashes, settings, backend, environment and source
hashes are retained. These checks catch obvious leakage, **not** renamed or
transformed reuses, related specimens, bilateral elements, individuals, sites,
or earlier analyst exposure. Curators must group such related data appropriately.
One manifest row per specimen is deliberate; do not manufacture independent cases
by renaming multiple fragments from one individual. Cross-run manifest changes
must be reviewed manually. Keep original locked manifests and reports.

This initial reader does not accept arbitrary meshes as exact dense truth, create
correspondence, calibrate uncertainties, or support landmark cases. For independent
meshes, use and test a genuine surface-distance protocol rather than silently
substituting nearest-vertex distance or claiming pointwise anatomical truth.

## Before drawing scientific conclusions

1. Separate SSM training, **preset tuning**, **uncertainty calibration**, and final
   test specimens before inspecting outcomes. Derive all fragments of one specimen
   within its assigned split; freeze mask generation, coverage perturbations,
   numerical settings, handoff, success thresholds and endpoint definitions.
2. On tuning only, choose a prescription. Compare multiple cuts, severe losses,
   coverage misspecification, scale, pose seeds, noise/outliers and relevant anatomy.
   Record all failures, not just returned fits or successful basins. Do not optimize
   defaults on the tiny smoke generator and report that as external improvement.
3. On separate calibration specimens, generate a compatible profile using the
   calibration workflow, with the fixed model/method/backend/settings. Merely
   labeling rows `calibration` here does not construct a calibrator. Calibration
   based on SSM-training shapes is not independent held-out calibration.
4. Evaluate once on untouched test specimens, retaining failed reconstructions.
   Report specimen-level failure rates and error distributions by cut/coverage;
   avoid treating thousands of correlated vertices as independent sample sizes.
   Audit both pointwise and specimen-wise uncertainty coverage under explicit
   endpoints, and distinguish coordinate correspondence from surface error.
5. Establish external provenance with a curator, adequate representative sample
   size, leakage review, blinded endpoints and independent Slicer/UI integration
   tests before any anatomical or uncertainty guarantee. If test outcomes drive
   further choices, that set has become tuning data and a new test set is needed.
