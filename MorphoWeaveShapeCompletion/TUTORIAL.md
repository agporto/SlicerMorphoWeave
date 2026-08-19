# MorphoWeave Shape Completion Tutorial

This tutorial walks through completing a partial anatomical surface with a Statistical Shape Model (SSM), inspecting uncertainty, and using the scalable exact-pose-plus-residual transfer on meshes containing about 1,000,000 vertices.

## What the module does

Shape Completion estimates a complete shape from a partial target surface. Its registration path is restricted to the loaded PCA shape model:

1. deterministic global pose search;
2. pose and SSM-shape refinement of surviving hypotheses;
3. SSM-constrained atlas registration;
4. conditional posterior uncertainty; and
5. transfer of the dense SSM result to the full-resolution template mesh.

The module does not apply an unconstrained fine CPD after the SSM fit. The dense completed correspondences therefore remain in the learned shape space. The final full-resolution mesh is a geometric rendering of that dense completion.

## Before starting

You need:

- a MorphoWeave SSM loaded through **Model Library**;
- the matching template surface;
- the matching dense template correspondences;
- the matching sparse template landmarks when landmark-assisted completion is used; and
- a partial target surface loaded as a Slicer model node.

For landmark-assisted completion, also load a target markup node containing at least three non-collinear homologous landmarks. Matching control-point labels are preferred.

The `rustcpd` build must contain the `pose-landmark-keypoints` and shape-posterior APIs expected by the module.

## Quick start

### 1. Load the model

1. Open **MorphoWeave > Model Library**.
2. Select the appropriate database.
3. Load the SSM into the scene.
4. Confirm that the scene contains the canonical quartet:
   - `ssm_data_<name>`;
   - `<name>_template`;
   - `<name>_template_correspondences`; and
   - `<name>_template_sparse_landmarks`.

When Shape Completion opens, it fills only empty template and SSM selectors from the most recently loaded compatible quartet. It does not replace a manually selected target.

### 2. Load the target fragment

Load the partial mesh with **File > Add Data** or drag it into Slicer. The target should use the same physical units as the training data whenever possible. A parent transform is supported; the module reads all geometry in world coordinates.

Optional target landmarks should lie in the target mesh coordinate frame. Use labels that match the template sparse-landmark labels. Ordered fallback is available, but should be enabled only when the control-point ordering is known to be homologous.

### 3. Populate Required Inputs

Open **MorphoWeave > Shape Completion > Complete Shape** and confirm:

| Input | Selection |
|---|---|
| Template surface | Full-resolution template associated with the SSM |
| Template dense correspondences | Dense SSM control points, typically about 5,000 |
| Template sparse landmarks | Matching template markup set |
| SSM Data Table | Table loaded by Model Library |
| Target fragment | Partial target surface |
| Target landmarks | Optional homologous target markups |

The module checks point counts, indexed SSM/template compatibility, and coordinate-frame consistency before enabling the run.

### 4. Set Target coverage

Set **Target coverage** to the estimated fraction of the complete anatomy represented by the fragment. Examples:

- `1.00`: effectively complete target;
- `0.75`: approximately three quarters retained;
- `0.50`: approximately half retained; and
- `0.25`: severe fragment.

Coverage affects scale policy, posterior completeness, calibration-profile matching, and the display mask separating data-proximal from inferred regions.

The default advanced setting pre-scales a partial target using fragment extent divided by coverage, then fixes residual scale. This is a practical fragment heuristic, not a geometric identity. Disable **Pre-scale the SSM using target extent and supplied coverage** when source and target already share a trustworthy physical scale or when validation shows the heuristic is biased for the expected cut pattern.

### 5. Choose surface-only or landmark-assisted completion

For distinctive fragments with broad coverage, start with surface-only completion.

Enable **Use matched landmarks in pose search, pose refinement, and atlas registration** when:

- coverage is low;
- the anatomy is symmetric or nearly symmetric;
- the fragment admits multiple plausible poses; or
- a small number of reliable anatomical anchors are available.

Landmark mode is strict. Missing, duplicated, collinear, or otherwise invalid matches stop the run instead of silently changing the scientific model to surface-only completion.

### 6. Run the completion

Optionally select an output directory, then click **Run Shape Completion**. The progress display separates registration from construction or reuse of the full-resolution transfer operator.

The first run for a template builds the sparse transfer operator. Repeating a completion with the same template, dense controls, transforms, and interpolation settings reuses the cached operator.

## Full-resolution transfer

### Why the exact-pose-plus-residual transfer is separated

The dense completion includes a fitted similarity pose and a nonrigid SSM-shape residual. Ordinary weighted displacement interpolation does not necessarily reproduce rotations or scale exactly. Shape Completion therefore computes:

```text
posed full-resolution template = similarity(template vertices)
posed dense template           = similarity(dense controls)
dense nonrigid residual         = completed dense - posed dense
completed full-resolution       = posed full-resolution + W × dense residual
```

The fitted rotation, scale, and translation are therefore exact on every output vertex. Only the nonrigid residual is interpolated.

### Sparse transfer operator

`W` is a cached CSR sparse matrix from dense SSM controls to full-resolution vertices. It is constructed in chunks using a k-d tree, then reused for:

- the three residual-displacement coordinates;
- epistemic uncertainty;
- total uncertainty;
- distance to the fragment;
- calibrated radii; and
- optional full-resolution latent samples.

All geometry and scalar fields for the completed surface are transferred in one sparse multiplication.

### Recommended defaults for 5,000 controls

| Setting | Default | Guidance |
|---|---:|---|
| Surface interpolation neighbors | 8 | Good balance between locality, smoothness, and memory |
| Interpolation locality | 2.0 | Gives the farthest retained neighbor an unnormalized Gaussian weight of `exp(-4)` |
| Transfer build chunk size | 200,000 | Lower this when temporary memory is constrained; it does not change the result |
| Restrict interpolation to connected template components | On | Prevents deformation from crossing disconnected components |

With one million output vertices and eight retained neighbors, the stored transfer operator is approximately 69 MiB using float32 weights and int32 indices. The completed surface, VTK topology, normals, temporary arrays, and optional samples require additional memory.

### Exact control anchors

Current Atlas Builder dense correspondences are normally selected from existing template vertices. Shape Completion detects those coincident vertices and makes their transfer rows exactly one-hot. The full-resolution surface therefore matches the completed dense control exactly at every recognized anchor.

The output dense markup node stores the inferred template vertex indices in the `MorphoWeave.TemplateVertexIndices` attribute. A value of `-1` indicates a control that did not match a template vertex within tolerance and was handled by ordinary local interpolation.

### Connected components

When component restriction is enabled, each full-resolution vertex receives support only from dense controls on the same connected template component. If a component contains no dense controls, its vertices fall back to the global control set and the fallback count is reported in diagnostics.

This restriction prevents leakage between disconnected structures. It does not prevent leakage between nearby opposing sheets within the same component; inspect thin or tightly folded anatomy carefully.

### Latent samples and large meshes

The posterior samples are samples from the latent coefficient posterior. They do not add independent observation noise or model-discrepancy noise.

By default, samples are created as hidden dense SSM point clouds. This keeps scene memory manageable. Enable **Create full-resolution sample surfaces (memory intensive)** only when full-resolution sampled geometry is specifically needed. Each million-vertex sample can consume substantial memory even though the transfer itself is fast.

## Inspecting outputs

Every run creates a folder under **MorphoWeave Shape Completion runs** containing:

- `<target>_CompletedShape`;
- `<target>_CompletedDenseCorrespondences`;
- `<target>_CompletedLandmarks`, when a template sparse set is available;
- `<target>_CompletionPointCloud`;
- optional latent sample nodes;
- `<target>_CompletionDiagnostics`; and
- `<target>_FragmentReference`.

### Surface arrays

The completed full-resolution model contains:

- `CompletionEpistemicStd`: square root of the coefficient-posterior predictive variance trace;
- `CompletionTotalStd`: coefficient uncertainty plus posterior noise and estimated model discrepancy;
- `CompletionDistanceToFragment`: dense-locus distance to the observed fragment, interpolated to the surface;
- `CompletionRegion`: `0` for data-proximal and `1` for inferred display regions; and
- compatible calibrated radius arrays when a matching calibration profile is loaded.

The full-resolution scalar arrays are interpolated visualization fields. The dense uncertainty CSV contains the native values at SSM loci and should be preferred for quantitative locus-level analysis.

### Diagnostics to review

Check the following after every run:

- pose score margin, entropy, and effective hypotheses;
- scale-normalized surface residual variance;
- landmark RMS and reduced chi-square, when landmarks are used;
- median epistemic and total standard deviation;
- posterior-mean drift from the constrained atlas estimate;
- transfer operator memory and build/apply time;
- number of exact control anchors;
- control-to-template vertex RMS and maximum distance;
- connected-component fallback count; and
- calibration compatibility and scope.

Pose ambiguity statistics compare hypotheses within the current run. They are not calibrated probabilities of correctness.

## Saving results

Selecting an output directory writes:

- completed surface: `<base>.vtp`;
- dense correspondences: `<base>_dense.mrk.json`;
- completed sparse landmarks: `<base>_landmarks.mrk.json`, when present;
- latent dense samples: `<base>_latent_dense_XX.vtp`;
- latent full-resolution samples: `<base>_latent_surface_XX.vtp`;
- diagnostics: `<base>_diagnostics.json`; and
- dense uncertainty table: `<base>_uncertainty.csv`.

A failed Slicer save operation raises an explicit error rather than being silently ignored.

## Calibration tutorial

Calibration is optional and dataset-specific. It estimates how model-based uncertainty relates to observed completion error under a chosen fragment-generation protocol.

### Prepare calibration data

Create a directory of complete meshes. Optional paired directories can contain:

- complete dense truth markups with the same point order as the SSM; and
- complete sparse target landmarks for landmark-assisted calibration.

Pairing is by normalized file stem. Avoid duplicate stems.

### Configure the Calibration tab

1. Select the complete mesh directory.
2. Optionally select paired dense-truth and sparse-landmark directories.
3. Enter one or more coverage values.
4. Choose the number of deterministic plane cuts per mesh and other calibration controls.
5. State whether the complete meshes were used to train the SSM.
6. Choose the output calibration profile path.
7. Click **Generate Calibration Profile**.

The generated fragments always come from complete mesh surfaces. Dense truth changes the error target; it never replaces fragment geometry.

### Interpret calibrated outputs

A matching profile may provide:

- `CompletionCalibratedPointwiseRadius`: marginal pointwise Euclidean-error coverage; and
- `CompletionCalibratedSimultaneousRadius`: a more conservative meshwise envelope.

Profile reuse requires an exact SSM fingerprint, matching inference settings, matching landmark mode, and sufficiently close coverage. Internal calibration on SSM training meshes may be optimistic for new specimens.

## Troubleshooting

### Run button remains disabled

Confirm that the template, dense correspondences, SSM table, and target fragment are selected and compatible. Read the status text directly above the run controls.

### Dense correspondences do not match the template

Reload the canonical quartet from the same Model Library entry. Remove transforms that were applied to only some nodes. The SSM mean and dense template points must match by index and coordinate frame.

### Landmark mode fails before registration

Check for:

- fewer than three matched labels;
- duplicate labels;
- ordered fallback disabled when labels are absent;
- several sparse landmarks mapping to the same dense vertex; or
- a nearly collinear landmark arrangement.

### Completion has the wrong scale

Verify the physical units and coverage estimate. Compare the default coverage pre-scale with:

- pre-scale disabled and residual scale fixed; and
- pre-scale disabled and residual scale free.

Use a dataset-specific validation set to choose the policy.

### Thin surfaces deform across nearby sheets

Keep component restriction enabled. Increasing locality can reduce distant influence, but Euclidean neighbors on the same connected component may still cross thin gaps. Validate the result visually and with surface-quality checks for the anatomy of interest.

### Transfer operator uses more memory than expected

Reduce interpolation neighbors. Lowering chunk size reduces temporary construction memory but not the stored CSR matrix. Keep full-resolution latent sample surfaces disabled.

### A component fallback is reported

The template contains a connected component with no dense SSM controls. Rebuild the dense sampling so every scientifically relevant component contains controls, or accept that the reported component used the global support set.

### Saved samples look less variable than total uncertainty

This is expected. Samples represent the latent coefficient posterior, while `CompletionTotalStd` also includes posterior noise and model-discrepancy terms.

## Suggested validation before production use

For each anatomical dataset, evaluate:

1. several coverage levels;
2. several cut directions and retained regions;
3. surface-only and landmark-assisted modes;
4. the coverage pre-scale ablation;
5. pose correctness and reconstruction error;
6. full-resolution triangle flips, self-intersections, and edge stretch; and
7. uncertainty calibration on specimens independent of SSM training when possible.

The module is designed to expose the diagnostics needed for that validation, but it cannot replace dataset-specific evidence.
