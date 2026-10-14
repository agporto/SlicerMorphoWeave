# Correctness update of the restored fragment implementation

This separate working copy is based on `shape-completion` commit
`e397882d54994479ef19279756b8beb82d382856`. It is a reviewable update on the separate
`fix/shape-completion-correctness-audit` branch. The historical upload and batch provenance
remain in `RESTORATION_PROVENANCE.json`; its `correctness_update` and `audit_branch_update` sections record
intentional differences. Earlier delivery descriptions in the provenance file
are historical, not the current branch publication status. The core is no longer byte-for-byte identical to the
historical restoration.

## What changed

- The posterior uses `prior_temperature = atlas_lambda`, matching rustcpd 4.0's
  prior **precision** multiplier. Finite positive atlas regularization is
  required. Zero is rejected before registration because the pinned posterior
  API cannot represent an exact flat prior; no arbitrary epsilon is substituted.
- Calibration profiles require current schema, settings, backend and
  implementation provenance. Old profiles must be regenerated. A changed core,
  calibration widget or surface metric cannot silently reuse old radii.
- Surface-distance calibration evaluates predicted dense loci against closest
  points on the complete triangle surface, not nearest vertices. Missing-region
  membership uses the corresponding closest truth-side locations. Exact dense
  correspondence truth is unchanged.
- The final atlas fit is normalized, with explicit initial variance converted
  into that frame. Coarse triangle-surface landmark validation and classifier
  AUC scoring are corrected. Native stopping/floor limitations remain.
- Regression tests and a native validation scaffold accompany the changes.
  See `VALIDATION_PROTOCOL.md` and the delivered test report for measured results.

## What deliberately did not change

The scientific point estimate remains the constrained atlas fit. The production
pose search, coverage-consistent prescale, residual scale bounds, translation
seeding, landmark matching, default fragment settings, exact-pose-plus-residual
mesh transfer, and batch implementation remain intact. In particular, production
atlas initialization still uses the pose-residual variance heuristic (factor
2.0), not `initial_state` continuation. The validation scaffold compares state
continuation experimentally; a synthetic comparison is not grounds to replace
successful biological fitting behavior automatically.

The posterior remains conditional on one selected pose and fitted surface
assignments. It does not integrate alternate anatomical interpretations or add
landmark precision. Model-based sample spread is not a calibrated reconstruction
error envelope. Pooled pointwise and meshwise simultaneous calibration have
different scopes, detailed in the README.

## Install without replacing your baseline

1. Close Slicer and preserve your working installation and inputs.
2. Check out `fix/shape-completion-correctness-audit` in a separate repository
   copy. Keep backups outside Slicer's scanned extension paths.
3. Use Extension Wizard to load that separate checkout. Expected tabs are
   **Complete Shape / Batch / Optimize / Calibration / Advanced**.
4. Keep `rustcpd==4.0.0`. Restart Slicer if another backend version was loaded.
5. Use a fresh output directory. Re-run calibration with the updated pipeline;
   never edit a legacy JSON profile to make it appear compatible.
6. Check a known successful fragment first, then single/batch equivalence,
   surface-only/landmark modes, calibration with real triangle meshes, and export.

No input datasets are included or modified. This branch also repairs other
MorphoWeave modules; see `../CORRECTNESS_STATUS.md` for the complete scope.

## Running regression tests

From the extension root with NumPy, SciPy and VTK available:

```bash
python -m unittest discover -s MorphoWeaveShapeCompletion/Testing/Python -p '*Test.py'
```

Install the pinned rustcpd package in an isolated environment to execute native
contracts and the displaced-fragment regression. Check the test report for
skips: a skipped native test is not a native pass. The installer suite references
an installer absent from this module and is expected to skip.

The restoration harness simulates Qt/MRML and registration while exercising the
actual widget, core and batch path; its VTK I/O is real. It is not a live Slicer
GUI test. Native synthetic tests establish only their stated synthetic cases.
Independent biological validation and live Slicer smoke testing remain separate
release gates. See `VALIDATION_PROTOCOL.md`.

Upstream semantics:
https://github.com/agporto/rustcpd/blob/v4.0.0/docs/MIGRATING_4.md
