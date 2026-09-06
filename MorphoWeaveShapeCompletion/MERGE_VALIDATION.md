# Shape Completion integration validation

All registration installers use **`rustcpd==4.0.0`**: Landmark Transfer, the
public Shape Completion widget and its base implementation. Imported native
modules must report exactly `4.0.0`; older releases, later patches, local or
post releases, and prereleases are rejected. Completion retains its required
API checks. A different already loaded version is rejected before any pip
operation, with instructions to restart Slicer.

## rustcpd 4.0 migration scope

The 4.0 Python signatures retain the calls used by this pipeline. This change
updates the dependency gate, tests, and help text, not the fitting equations,
settings, or stage-initialization strategy. In particular, it does not migrate
the pipeline to the new `initial_state` continuation API.

API compatibility does not imply numerical equivalence. rustcpd 4.0 corrects
fitted-state handling, posterior observation models and the non-default
`prior_temperature` convention. Revalidate representative completions and
regenerate calibration profiles with the backend used for production. Do not
reuse a 3.1 calibration profile as evidence of calibrated 4.0 uncertainty.
See the upstream [4.0 migration guide](https://github.com/agporto/rustcpd/blob/v4.0.0/docs/MIGRATING_4.md).

Landmark Transfer and Shape Completion now share the exact dependency pin,
so switching modules does not request conflicting backend versions. This
dependency-policy change does not itself establish native numerical or live
Slicer compatibility; the smoke-test gate below remains in place.

Automatic SSM selection intentionally blocks signals in the base widget. The
public widget explicitly mirrors the resulting model selections into Batch
after those blockers are released. This runs on setup and reentry and preserves
manual selections and target nodes.

## Ordinary Python regression tests

From the repository root:

```bash
python MorphoWeaveShapeCompletion/Testing/Python/ShapeCompletionIntegrationTest.py
```

These tests load the actual module entry point with interface doubles. They
exercise discarded signals, initial setup, reentry, partial/manual selection,
exact 4.0.0 acceptance, rejection of other versions, required capabilities,
declined installs, failure, and retry. They are not a native fitter or live
Qt/MRML validation.

## Native Slicer smoke test

Install this branch of the extension and the released rustcpd==4.0.0 wheel,
then run the following in a **fresh Slicer process** (replace `Slicer` with the
path to the application executable). To validate 4.0 specifically, ensure the
loaded backend is 4.0.0. The module paths must refer to this checkout or an
installation built from this branch, not an older extension. The test does not
install packages or show installation prompts.

```bash
Slicer --no-main-window --python-script MorphoWeaveShapeCompletion/Testing/Python/ShapeCompletionSlicerSmokeTest.py
```

The script creates a small synthetic SSM, checks real automatic selector
synchronization, runs a native single completion and a two-specimen batch,
cancels after the first specimen, resumes, checks verified skips, and compares
the exported single/batch coordinates. The iteration budgets are reduced to
exercise integration, not anatomical reconstruction accuracy. It uses temporary
files, cleans up its scene nodes, prints `SHAPE_COMPLETION_SMOKE_PASSED` with a
JSON report on success, and exits Slicer with status 0 (or 1 on failure).

Passing the interface-double tests alone does **not** close this native smoke
test gate. The script must actually be run successfully in the target Slicer
installation before merge signoff.
