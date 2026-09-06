# Batch Shape Completion

Process a folder of specimens with the **existing Shape Completion pipeline**.
The original module, fitting equations, scale policy, pose search, calibration,
uncertainty calculations, and exporters are not changed.

## Using the module

After rebuilding/reinstalling the extension from this branch, select
**MorphoWeave > Batch Shape Completion**. For a source checkout loaded through
Slicer's Additional module paths, also add the `MorphoWeaveBatchShapeCompletion`
directory; the existing `MorphoWeaveShapeCompletion` module must remain available.

1. Load your SSM as usual. In **Complete Shape**, select the template surface,
   template dense correspondences, SSM table, and template sparse landmarks
   when applicable. Choose coverage, landmark mode, and an optional calibration
   profile. The target selectors and single-specimen output directory are
   **not required** for a batch.
2. Set fitting and export options in **Advanced**, exactly as for one specimen.
3. Open **Batch**, select the fragment mesh folder and a different output folder.
   In landmark mode, also select the paired landmark folder. Click
   **Run Batch Shape Completion**.

All specimens share the selected settings, coverage, calibration profile and
random seed. The seed is not incremented by specimen index, so changing batch
order does not change an individual specimen's seed. Use separate batches for
specimens requiring different coverage estimates or fitting settings.

The batch entry has the same Complete Shape, Calibration, and Advanced controls
as the original module, plus a Batch tab. Its fitting call is the inherited
`_run_completion_impl()`, not a separately implemented approximation. The
original **Shape Completion** module remains available and unchanged.

## Input files and pairing

The mesh folder is scanned non-recursively for `.ply`, `.vtp`, `.vtk`, `.stl`, and
`.obj` files, including uppercase extensions. For example:

```text
fragments/
  specimen_01.ply
  specimen_02.stl
landmarks/
  specimen_01.mrk.json
  specimen_02.fcsv
```

Landmarks are paired by the complete basename, case-insensitively; `.mrk.json`,
`.fcsv`, and `.json` are supported. Duplicate mesh or landmark basenames are
rejected before processing because they would make pairing/output paths
ambiguous. Landmark files are ignored when landmark mode is off.

In landmark mode, a missing file or fewer than three matched landmarks fails
that specimen and is recorded in the CSV; other specimens continue. The batch
never silently falls back to an unassisted fit. Label matching and the explicit
ordered-fallback option are the same as in single-specimen completion.

## Outputs, failures, and resume

```text
results/
  batch_summary.csv
  specimen_01/
    <existing completed-shape VTP, markups, uncertainty CSV and diagnostics JSON>
    <posterior samples when requested>
    completion_batch.json
  specimen_02/
    ...
```

Each specimen uses the existing exporter, including its uncertainty arrays,
calibrated fields and optional samples. `batch_summary.csv` is updated after
each specimen and records input paths, output directory, status, elapsed time,
settings fingerprint and errors. Statuses are `pending`, `running`, `success`,
`failed`, `skipped`, or `cancelled`. The summary describes the latest invocation;
the per-specimen manifests preserve the provenance of successful results.

A result is first written to a temporary directory and published only after
all exports succeed. Failed fits/exports do not create apparently completed
specimen folders. The summary records their error and the batch continues.

**Skip verified completed specimens** is enabled by default. Resume requires a
successful manifest with matching input-file hashes, SSM/template geometry,
settings, calibration profile, code/backend fingerprints and intact output-file
hashes. A corrupt or mismatched result is not silently reused. Existing specimen
folders are never overwritten: select a new output folder to rerun with changed
inputs/settings or to replace damaged results. Empty/partial folders created by
other workflows are likewise not overwritten.

A directory lock prevents two batches from writing to the same output root.
After an abrupt Slicer/process crash, first verify that no batch is still
running, then remove the stale `.morphoweave_batch.lock` file to resume. Abandoned
`.specimen.tmp-*` directories are not treated as successful results and may be
removed after the same check.

## Memory and cancellation

Specimens run sequentially; the existing Rust parallelism within an individual
fit is retained. SSM arrays are parsed once per batch and copied for each fit to
prevent state leakage. The existing full-resolution transfer-operator cache is
reused across specimens.

Temporary target, landmark, output, display and storage nodes are removed after
each specimen, including failed loads/exports. Pre-existing scene inputs and
outputs are preserved, and target selectors/output-path state is restored.
Completed specimens are saved to disk rather than accumulated in the scene;
load an exported VTP or markup file to inspect it.

**Cancel after current specimen** finishes the current native fit and exports,
then stops before the next specimen. Native registration is not interrupted
mid-call. Shared SSM/template inputs should not be edited while a batch runs;
changes detected by the edit guard cause the affected result to be rejected.

## Tests

From the repository root:

```bash
python MorphoWeaveBatchShapeCompletion/Testing/Python/BatchRunnerTest.py
python MorphoWeaveBatchShapeCompletion/Testing/Python/BatchSlicerAdapterTest.py
```

The runner suite uses the Python standard library. The adapter suite uses NumPy
and fake Slicer/Qt/VTK objects. It checks delegation to the single-specimen
runner, identical results with a deterministic fake fitter, SSM-cache isolation,
landmark isolation, scene cleanup on success/failure, preservation of unrelated
nodes, and shared-input edit detection. These tests do **not** constitute a
native `rustcpd` accuracy benchmark or a live 3D Slicer GUI test.
