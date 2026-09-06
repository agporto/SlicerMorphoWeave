# MorphoWeave Shape Completion

`MorphoWeaveShapeCompletion` completes a partial target surface using the Statistical Shape Model already loaded by **Model Library**. It is a separate module rather than an option inside Landmark Transfer because its estimand and outputs are different: Landmark Transfer predicts sparse landmarks on an observed surface, whereas Shape Completion infers the unobserved part of the surface and reports uncertainty in that inference.

See [TUTORIAL.md](TUTORIAL.md) for a complete first-run walkthrough, high-resolution settings, output interpretation, calibration, and troubleshooting.

## Installing this module patch

The full integration package places `install_into_checkout.py` beside the
`MorphoWeaveShapeCompletion` directory. From that package root, preview the
repository changes with:

```bash
python install_into_checkout.py /path/to/SlicerMorphoWeave --dry-run
```

Install into the checkout with:

```bash
python install_into_checkout.py /path/to/SlicerMorphoWeave
```

Use `--force` to replace an existing Shape Completion directory. The previous
directory is moved to `.shape-completion-backup/MorphoWeaveShapeCompletion`
before replacement. The installer updates the root CMake file, root README, and
main MorphoWeave tutorial idempotently.

When using a module-only archive, copy `MorphoWeaveShapeCompletion` into the
repository root and add `add_subdirectory(MorphoWeaveShapeCompletion)` to the
root `CMakeLists.txt`. The full integration package is preferred because it also
updates the project-level documentation and includes update and validation notes.

## Registration pipeline

The production path is deliberately restricted to the learned SSM:

1. deterministic global pose search over rotation hypotheses;
2. joint pose/shape refinement of surviving hypotheses;
3. pure SSM atlas registration initialized from the winning hypothesis;
4. conditional Gaussian shape completion and uncertainty propagation.

No unconstrained fine CPD is applied after the atlas fit. The completed dense correspondences therefore remain in the SSM shape space. For the full-resolution output, the fitted similarity pose is applied analytically to every template vertex and a cached sparse operator interpolates only the nonrigid SSM residual. This is a rendering/geometry-transfer step, not another statistical fit.

The module runs in either mode:

- **surface only**, using the target fragment alone; or
- **landmark assisted**, using homologous target landmarks in pose scoring, pose refinement, and atlas registration.

Landmark mode is strict: when selected, an invalid or absent landmark match stops the run rather than silently reverting to surface-only completion.

## Target coverage

**Target coverage** is an explicit input and defaults to `1.00`. It controls:

- whether residual scale is free or fixed under the default automatic policy;
- the optional extent/coverage source pre-scale for partial targets;
- posterior visibility/completeness;
- calibration-profile matching; and
- the landmark recommendation.

Based on the current fragment experiments, the UI recommends at least three homologous landmarks below `0.75` coverage. The recommendation is not a substitute for checking whether the landmarks span a stable, non-collinear configuration. The extent/coverage pre-scale is a heuristic inherited from the existing fragment workflow and can be disabled when source and target already share meaningful physical scale.

## Full-resolution surface transfer

The completed dense SSM points are transferred to the original template mesh with an exact-pose-plus-residual formulation:

`completed surface = similarity(template) + W × (completed dense − similarity(dense template))`.

Applying the fitted similarity transform analytically ensures that a pure rotation, translation, or scale is reproduced exactly. Only the nonrigid SSM residual is locally interpolated. `W` is a chunk-built SciPy CSR matrix with a small fixed number of nonzero weights per output vertex. It is cached for the current template, dense node, transform state, and transfer settings, then reused for geometry, scalar display arrays, repeated runs, and optional full-resolution latent samples.

The defaults target approximately 5,000 controls and meshes up to about 1,000,000 vertices: eight neighbors, Gaussian locality `2.0`, and 200,000 query vertices per k-d-tree chunk. An eight-neighbor one-million-row operator is approximately 70 MiB, excluding VTK geometry and output arrays.

Dense controls that coincide with template vertices are made exact one-hot anchors. The module infers and validates those vertex IDs from coordinates, then stores them on the completed dense markup node as `MorphoWeave.TemplateVertexIndices`. Connected-component restriction is enabled by default to prevent interpolation between disconnected surface components. Components without dense controls fall back to the global control set and are reported explicitly. The restriction does not prevent Euclidean support from crossing nearby opposing sheets within one connected component.

Full-resolution posterior sample surfaces are disabled by default because duplicating a very large mesh can dominate memory. Latent samples are stored as dense SSM point clouds unless the user opts into full-resolution surfaces. Sample spread represents coefficient-posterior uncertainty, not the additional noise and discrepancy terms in `CompletionTotalStd`.

## Landmark matching and uncertainty

Template and target landmarks are matched by normalized control-point labels. Ordered matching is disabled by default and must be explicitly enabled. The matched template sparse landmarks are mapped to unique dense SSM indices. The module rejects:

- fewer than three matches;
- duplicate labels;
- multiple sparse landmarks collapsing to fewer than three dense indices; and
- collinear or nearly collinear source or target configurations.

The user-specified annotation standard deviation `τ_annotation` defaults to 2% of the expected full-shape RMS radius. Because rustcpd currently anchors dense SSM vertices while a sparse template landmark may lie between vertices, the sparse-to-dense mapping error is treated as an additional uncertainty source:

`τ_effective = sqrt(τ_annotation² + RMS(mapping error)²)`.

That same effective target-coordinate standard deviation is passed to:

- pose-hypothesis scoring;
- landmark-anchored pose refinement; and
- constrained atlas registration.

The nominal annotation standard deviation, effective standard deviation, mapping RMS, mapping maximum, final landmark RMS, and reduced chi-square are reported separately. This vertex approximation can later be replaced by a barycentric observation operator without changing the UI contract.

## Model-based uncertainty outputs

The completed model contains point-data arrays:

- `CompletionEpistemicStd`: square root of the trace of uncertainty due to the conditional SSM coefficient covariance;
- `CompletionTotalStd`: coefficient uncertainty plus posterior noise and model-discrepancy terms;
- `CompletionDistanceToFragment`: distance from each dense locus to the observed target fragment, interpolated to the surface; and
- `CompletionRegion`: `0` for data-proximal and `1` for inferred surface, based on a coverage-sized proximity mask. This is a display aid, not observed ground truth.

The scalar arrays on the full-resolution surface are interpolated visualization fields. Native SSM-locus values are retained in the dense uncertainty CSV for quantitative analysis.

The diagnostics table and JSON distinguish quantities that should not be conflated:

1. pose score margin, entropy, and effective hypotheses, which rank ambiguity **within one run only**;
2. surface `sigma2`, also reported after scale normalization;
3. landmark RMS and reduced chi-square when target landmarks are used; and
4. conditional spatial uncertainty from the shape posterior.

The completed coordinates are the final constrained atlas fit. This is especially important in landmark-assisted runs: the current rustcpd posterior conditions on the fitted surface correspondences but does not yet add target landmarks to posterior precision, so replacing the completion with `ShapePosterior.predict()` would partially discard the anchor information. The surface-only posterior mean is retained only as an RMS drift diagnostic, while posterior samples are recentered on the constrained atlas estimate without changing their covariance.

The current posterior is conditional on the selected pose and fitted surface correspondences. It does not integrate alternate pose basins, and its covariance does not yet include target-landmark precision. These limitations are displayed in the UI and saved diagnostics.

The posterior coefficient prior is matched to atlas registration exactly: atlas regularization `λ` corresponds to posterior prior temperature `1/λ` (`λ=0` is represented by an effectively flat prior). This keeps the post-hoc uncertainty calculation consistent with the atlas coefficient objective.

## Calibration from complete meshes

The **Calibration** tab accepts a directory of complete meshes and generates deterministic contiguous plane-cut fragments at one or more coverage levels. Fragment geometry always comes from the complete mesh surface. Optional paired files change the calibration target or enable landmarks; they never replace the fragment with a correspondence point cloud.

Two calibration truth targets are supported:

- **Exact dense correspondence error** when a complete paired dense-truth markup directory is supplied. Every mesh must have a paired file with the SSM point count and matching point order.
- **Distance to complete surface** when only complete meshes are supplied. Predicted loci are evaluated against their nearest complete-surface locations, and inferred-region membership is determined from those truth-side locations. This calibrates surface recovery, not homologous point error.

A complete paired sparse-landmark directory may be supplied for landmark-assisted calibration. Visible landmarks are selected using the same generated plane cut; examples with fewer than three usable landmarks are skipped rather than silently converted to surface-only calibration.

Each profile entry is specific to coverage, landmark mode, and truth target and contains two different spatial calibration products:

- `CompletionCalibratedPointwiseRadius`: a **marginal pointwise** Euclidean-error radius based on pooled inferred-locus nonconformity scores. It is not a simultaneous guarantee for the entire completed shape.
- `CompletionCalibratedSimultaneousRadius`: a more conservative **meshwise simultaneous** envelope, available only when the finite conformal quantile can be estimated from enough independent complete meshes. Each mesh contributes its maximum standardized inferred-locus error across generated fragments under the calibration masking protocol.

When at least five meshes are available, the profile also stores a deterministic held-out-mesh audit. It reports marginal pointwise empirical coverage and, when estimable, meshwise simultaneous empirical coverage.

The profile additionally stores a separation-safe ridge-logistic mapping from scale-normalized surface `sigma2` to the user-defined fit-success event when at least eight fits and both outcome classes are present. Its in-sample metrics are explicitly labelled apparent; the held-out real-data behavior is the relevant evidence for use.

## Calibration provenance and reuse

Every profile stores:

- SHA-256 of the full SSM mean, modes, and eigenvalues;
- a signature of all completion and uncertainty settings that affect the fit, including the pose-search seed;
- coverage, landmark mode, and truth target;
- fit, mesh, and point counts;
- spatial calibration scales and audits;
- optional fit-success calibration; and
- whether the calibration meshes were used to train the SSM.

Profiles made from SSM training meshes are explicitly labelled **internal calibration** and may be optimistic for new specimens. A profile is applied only when:

- the SSM fingerprint matches exactly;
- the inference/uncertainty settings signature matches;
- landmark mode matches; and
- a calibrated coverage lies within `0.05` of the requested coverage.

Otherwise calibrated radii and probabilities are omitted. Exact and surface-distance truth targets are never mixed silently.

## Scene and file outputs

Every run creates a folder under `MorphoWeave Shape Completion runs` containing:

- completed full-resolution surface;
- completed dense correspondences;
- completed sparse landmarks, when available;
- hidden dense uncertainty point cloud;
- optional latent posterior samples, stored as dense point clouds by default or full-resolution surfaces by explicit opt-in; and
- a diagnostics table.

When an output directory is selected, the module also writes:

- completed `.vtp` surface;
- dense and sparse `.mrk.json` files;
- diagnostics `.json`;
- dense uncertainty `.csv` with coordinates, model-based uncertainty, region/distance fields, and any compatible calibrated radii; and
- latent samples as `.vtp` point clouds or surfaces when requested.

## Runtime dependency

The module requires exactly **`rustcpd==4.0.0`**, shared with Landmark Transfer, containing the pose-landmark-keypoints and completion APIs, including:

- `pose_initialize(..., landmark_sigma=..., refine_landmark_sigma=..., with_scale=...)`;
- `register_atlas(..., landmark_sigma=..., initial_*=...)`;
- `AtlasResult.posterior(...)`; and
- `ShapePosterior.predict`, `predictive_variance`, and `sample_shapes`.

The dependency is checked when the module opens. The preflight validates the required signatures and result fields so that an incompatible build fails with an actionable capability report before registration starts.

## Tests

The Slicer-independent test suites currently contain 53 tests covering:

- SSM validation and truncation;
- landmark matching, geometry checks, and strict no-fallback behavior;
- exact forwarding of one effective physical landmark standard deviation to all constrained stages;
- inclusion of sparse-to-dense mapping uncertainty;
- surface-only API behavior;
- reciprocal atlas/posterior prior consistency;
- deterministic fragment masks and truth-side inferred-region calibration;
- coordinate-frame compatibility checks;
- exact similarity reproduction under full-resolution pose/residual transfer;
- sparse batched field transfer, exact control anchors, and component restriction;
- backward-compatible local interpolation at correspondence controls;
- marginal and meshwise simultaneous conformal calibration;
- separation-safe ridge-logistic calibration;
- profile fingerprint, settings, seed, mode, and coverage gating; and
- source-contract checks that enforce pose search → refinement → pure atlas → posterior with no unconstrained CPD;
- source contracts for cached full-resolution transfer, opt-in large sample surfaces, and checked file saves; and
- installer dry-run, idempotence, backup, refusal, and forced-replacement behavior.

The pure-Python suites do not replace an in-Slicer smoke test against the native `rustcpd` build.

A standalone stress benchmark is available at `Testing/Python/ShapeCompletionTransferBenchmark.py`. Its default case builds an eight-neighbor transfer from 5,000 controls to 1,000,000 vertices and applies eight fields; it is intentionally not registered as a routine unit test because of its memory footprint.
