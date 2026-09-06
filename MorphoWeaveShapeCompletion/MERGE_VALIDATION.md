# Shape Completion integration validation

The public Shape Completion widget now requires **`rustcpd>=3.1,<4`** for all
entry points (startup, single completion, calibration, and batch). Its deployment
preflight supersedes the historical preflight preserved in the base widget.
It retains the capability checks and reports supported released wheels and a
Slicer restart, rather than referring users to a development build reporting
version 3.0.0. Registration and uncertainty calculations are unchanged.

Automatic SSM selection intentionally blocks signals in the base widget. The
public widget now explicitly mirrors the resulting model selections into Batch
after those blockers are released. This runs on setup and reentry and preserves
manual selections and target nodes.

## Ordinary Python regression tests

From the repository root:

```bash
python MorphoWeaveShapeCompletion/Testing/Python/ShapeCompletionIntegrationTest.py
```

These tests load the actual module entry point with interface doubles. They
exercise discarded signals, initial setup, reentry, partial/manual selection,
required backend versions and capabilities, declined installs, failure, and
retry. They are not a native fitter or live Qt/MRML validation.

## Native Slicer smoke test

Install this branch of the extension and a compatible released rustcpd wheel,
then run the following in a **fresh Slicer process** (replace `Slicer` with the
path to the application executable). The module paths must refer to this
checkout or an installation built from this branch, not an older extension.
The test does not install packages or show installation prompts.

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
