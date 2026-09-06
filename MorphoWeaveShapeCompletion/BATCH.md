# Batch shape completion

Open **MorphoWeave > Shape Completion > Batch**. Batch processing is a tab in
the existing Shape Completion module, like the Batch workflow in Landmark
Transfer. There is no separate Batch Shape Completion module to install.

## Inputs and settings

In the Batch tab, select the template surface, dense correspondences and SSM
data table. Select template sparse landmarks when needed, then choose the
target mesh directory and a different output directory. Click **Run Batch Shape
Completion**. Individual target meshes do not need to be loaded into the scene.

The template selectors, coverage, landmark mode, ordered-matching option and
calibration-profile path are synchronized with **Complete Shape** in both
directions. **Advanced** settings are shared. All specimens in a batch use the
same coverage and random seed; use separate batches for different coverage
estimates. Changing the order of specimens does not change a specimen's seed.

Supported mesh formats are `.ply`, `.vtp`, `.vtk`, `.stl` and `.obj`, including
uppercase extensions. Only the selected directory is scanned, not subfolders.
In landmark mode, supply a paired directory containing `.mrk.json`, `.fcsv` or
`.json` files with matching basenames, for example:

```text
fragments/specimen_01.ply
landmarks/specimen_01.mrk.json
```

Matching is case-insensitive. Duplicate basenames are rejected. A missing file,
unusable mesh or fewer than three matched landmarks fails that specimen and
is reported, but other specimens continue. Landmark mode never silently falls
back to an unassisted fit. The landmark directory is ignored when that mode is
off. Label matching and the explicit ordered fallback are unchanged.

## Outputs and resume

```text
results/
  batch_summary.csv
  specimen_01/
    <completed VTP, dense and optional sparse markups>
    <uncertainty CSV, diagnostics JSON, optional posterior samples>
    completion_batch.json
  specimen_02/
    ...
```

Exports use the existing single-specimen exporter. A specimen's files are
written to a staging directory and published only after all exports succeed.
The summary CSV is updated after each specimen and records input/output paths,
status, elapsed time, settings fingerprint and errors. The summary describes
the latest invocation; successful specimen manifests preserve provenance.

**Skip verified completed specimens** is on by default. A specimen is skipped
only when its successful manifest matches the input files, model/template,
settings, calibration profile, implementation/backend fingerprints and intact
output hashes. Existing specimen folders are never overwritten. Use a new
output directory for changed settings or damaged results.

A directory lock prevents concurrent writers. Following an abrupt Slicer
crash, verify that no batch is still running before removing the stale
`.morphoweave_batch.lock` file. Abandoned `.specimen.tmp-*` directories are not
considered completed specimens and may then be removed.

## Scene state and cancellation

Specimens are processed sequentially, retaining Rust parallelism within each
fit. The SSM is parsed once per batch, with fresh arrays for each fit; the
full-resolution transfer cache is reused. Temporary input, landmark, output,
display and storage nodes are removed after each specimen. Existing scene
nodes, selected single-specimen targets, output path and single-run diagnostics
are preserved. Load saved files to inspect a completed specimen.

Single completion, batch completion and calibration cannot run simultaneously
from this module. **Cancel after current specimen** lets the current native fit
and exports finish before stopping. Shared settings are disabled during the
batch, and detected edits to shared scene inputs/settings reject the affected
result rather than publishing inconsistent provenance.

## Updating from the previous separate module

The separate `MorphoWeaveBatchShapeCompletion` directory and extension build
entry have been removed. Update/rebuild the extension and restart Slicer. A
source checkout needs only its existing `MorphoWeaveShapeCompletion` module
path. Remove a manually added path to the old batch directory from Additional
module paths. The new workflow is **Shape Completion > Batch**.

## Implementation and tests

The original single-specimen/calibration implementation is preserved byte for
byte as `Resources/Python/MorphoWeaveShapeCompletionBase.py`. The public Slicer
entry point composes that widget with the integrated Batch tab. Both modes
invoke the same `_run_completion_impl()`; fitting, pose search, scaling,
calibration, uncertainty and exporters have not been reimplemented.

```bash
python MorphoWeaveShapeCompletion/Testing/Python/BatchRunnerTest.py
python MorphoWeaveShapeCompletion/Testing/Python/BatchTabTest.py
```

The runner suite uses ordinary Python. The tab/adapter suite uses NumPy and
fake Qt/MRML objects, and checks tab order, bidirectional shared controls,
workflow exclusion, state restoration, cancellation, resume, failure isolation,
SSM-cache isolation and delegation to the single-specimen fitter. It is not a
live Slicer GUI test or a native rustcpd accuracy benchmark.
