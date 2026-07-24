# MorphoWeave: Scaling 3D morphometrics across biodiversity with atlas-based automated landmarking

`MorphoWeave` is a companion extension to SlicerMorph, focused on high‑fidelity, scalable generation and transfer of 3D anatomical landmarks and dense correspondences using atlas construction, Statistical Shape Models (SSMs), and PCA‑guided deformable registration.

## Overview
`MorphoWeave` streamlines:
* Building an atlas (mean model + dense correspondences) from meshes and sparse landmarks
* Constructing and persisting PCA Statistical Shape Models for variation exploration
* Single and batch automated landmark transfer with robust rigid + PCA‑CPD deformable alignment and optional surface projection
* Continuous optimization of the template (pose + shape) before large batch runs
* Segmentation of the 3D model into parts

## Module Descriptions
Open MorphoWeave ⟶ Atlas Builder.
You should observe the following screen:

<p align="center">
<img src="images/1.png" width = 600>
</p>

`MorphoWeave` organizes functionality into four scripted modules: `Atlas Builder`, `Model Library`, `Landmark Transfer`, and `Surface Segmentation`.

#### Atlas Builder
**Category**: Atlas & Correspondence Generation
**Purpose**: Align a set of meshes and sparse landmark sets; automatically pick a reference (closest to mean), similarity-align all specimens, generate a mean (atlas) surface, and derive dense correspondences via TPS warp + k‑d tree mapping. Optionally downsamples to produce a sparser, index‑stable subset.
**Key Outputs**: Timestamped output folder containing atlas files (atlas/atlas_model.ply, atlas/atlas_sparse_landmarks.mrk.json), sampled dense set (atlas/atlas_dense_correspondences.mrk.json), per-specimen dense correspondences in population_correspondences/, and optionally retained aligned meshes (alignedModels/*) and landmarks (alignedLMs/*).
**Notable Features**: Robust base selection; flexible file stem resolution; index preservation during downsampling.

<p align="center">
<img src="images/2.png" width = 600>
</p>

#### Model Library
**Category**: Statistical Shape Modeling
**Purpose**: Build a PCA SSM from a folder of dense correspondences; store as a lightweight on‑disk database with manifest; load into scene for interactive mode exploration.
**Workflow**: Ingest ⟶ validate consistent point counts ⟶ SVD (variance threshold) ⟶ save mean, modes, eigenvalues.
**Visualization**: Single-PC slider + spinbox; deforms mesh in real time using k‑NN interpolation (weights from inverse distance) between landmark displacements and all vertices.
**Outputs**: manifest.json, ssm_model.npz, copied template and markup files.

<p align="center">
<img src="images/3.png" width = 600>
</p>

#### Landmark Transfer
**Category**: Automated Landmark Transfer
**Purpose**: Transfer template sparse landmarks to a target mesh (single or batch) via multi-stage alignment: subsampling & FPFH features ⟶ RANSAC + ICP rigid alignment ⟶ PCA‑guided Coherent Point Drift (CPD) deformable registration ⟶ optional surface projection refinement. Batch mode can also optionally export the warped template mesh for each target.
**Modes**:
  * Single specimen alignment (tune parameters, inspect intermediate clouds/models)
  * Batch processing (reuse tuned parameters; cancellation + progress)
  * Template optimization (continuous search in SSM space + RANSAC scoring to refine initial template pose/shape)

**Outputs**: Predicted landmark .mrk.json files, warped template models (scene), optionally refined projected landmarks, and optional batch warped mesh exports as .vtp files.

#### Surface Segmentation
**Category**: Correspondence-Guided Surface Segmentation
**Purpose**: Use population-wide dense correspondences and geometric features to divide paired meshes into consistent anatomical surface regions.
**Outputs**: Per-region mesh files, preview models, and locus-label lookup files.

<p align="center">
<img src="images/4.png" width = 600>
</p>

## Atlas Builder
Now that we are acquainted with the overall layout of MorphoWeave, let's start by building a mean (atlas) surface and landmarks (sparse and dense).

### Step 1. Download sample data
Download the MorphoWeave sample dataset from [here](https://github.com/SlicerMorph/Mouse_Models)*. Click `Code` at the upper right corner then `Download Zip`. Extract all the files to a local directory. Return to 3D Slicer, go to the `MorphoWeave` category, then to `Atlas Builder`. * *Note: Pending pull request to fix broken mesh, FVB_NJ.ply. Otherwise, download data and delete the broken mesh manually or repair it using Slicer's Surface Toolbox.*

### Step 2. Populate `Required Inputs`
Set `Model directory` with source models (.ply format), `Landmark directory` with source landmarks (.mrk.json format), and `Output directory`. In `Output directory`, a timestamped folder will be made containing dense correspondences/semilandmarks (population_correspondences/*), atlas files (atlas/atlas_model.ply, atlas/atlas_sparse_landmarks.mrk.json, atlas/atlas_dense_correspondences.mrk.json), and—when retained—aligned meshes (alignedModels/*) and aligned landmarks (alignedLMs/*). Original input files are never copied or modified.

Optionally expand `Optional Model Library Save`, enable `Save SSM to Model Library`, and enter a model name. The displayed library location comes from the Model Library configuration (by default `Documents/MorphoWeaveModels`). If that name already exists, MorphoWeave asks for confirmation before the run; declining leaves the existing entry unchanged. The SSM is ingested only after atlas and dense correspondence export succeeds.

<p align="center">
<img src="images/5.png" width = 600>
</p>

### Step 3. Specify `Advanced Options` for atlas and dense correspondences generation.
* **Normalize scale**: Defaults to True. Normalizes the scale of specimens and landmarks.
* **Override landmark ⟷ mesh coordinate check**: Defaults to False. Checks that meshes and landmarks are in the same coordinate system (ex: RAS).
* **Keep aligned models and landmarks**: Defaults to True. Retains transformed derivatives in alignedModels/* and alignedLMs/*. Disable it to use a temporary processing workspace and omit these folders from the final output.
* **Warp method**: Defaults to TPS (recommended). Warp using thin plate splines (TPS; more smooth and flexible) or biharmonic (more rigid).
  * * **Auto-fallback to TPS if biharmonic fails**: Defaults to True.
* **Sampling radius (% of diag)**: Choose a value that will produce 2000 - 4000 expected dense correspondence points.
* **Expected points**: Adjust `Sampling radius` and click `Preview Point Count` until desired count is achieved.

<p align="center">
<img src="images/6.png" width = 600>
</p>

### Step 4. Under `Run + Status`, Click `Run Atlas Builder Pipeline`.
Progress will be reported in the window below `Run Atlas Builder Pipeline` and a model with landmarks should appear in the 3D viewer.

<p align="center">
<img src="images/7.png" width = 600>
</p>

<p align="center">
<img src="images/6a.png" width = 600>
</p>

If everything worked, you will see something indicating saved outputs and their locations.

<p align="center">
<img src="images/8.png" width = 600>
</p>

## Model Library
Use atlas and dense correspondences from `Atlas Builder` to make a PCA-based statistical shape model (SSM) database.

### Step 1. `Database Library`: Specify the persistent Model Library location.
The default is `Documents/MorphoWeaveModels`. This is also the destination displayed by Atlas Builder's optional auto-save workflow.

<p align="center">
<img src="images/8a.png" width = 600>
</p>

### Step 2. `Ingest New SSM`: Select corresponding files and folders found within `Database location` above.
* **Template Model**: atlas/atlas_model.ply
* **Dense Correspondences**: atlas/atlas_dense_correspondences.mrk.json
* **Sparse Landmarks**: atlas/atlas_sparse_landmarks.mrk.json
* **Population Correspondences Folder**: population_correspondences/*
* **New Database Name**: your database name (ex: mouse_db)

<p align="center">
<img src="images/9.png" width = 600>
</p>

Click `Ingest into Database`

### Step 3. `Database library`: Load database into MorphoWeave.
Under `Database Library`, your new database should appear. Select your new database and click `Load Selected Database`.

<p align="center">
<img src="images/10.png" width = 600>
</p>

### Step 4. `SSM Explorer`: Explore SSM changes through principal component space.
The form fields under `SSM Explorer` will automatically populate and your atlas model and landmarks will open in the 3D viewer.

<p align="center">
<img src="images/11.png" width = 600>
</p>

Adjust positions on PC sliders to visualize morphological change across &plusmn;SD's for each principal component.

<p align="center">
<img src="images/12.png" width = 600>
</p>

<p align="center">
<img src="images/13.png" width = 600>
</p>


## Landmark Transfer
Use the SSM database from `Model Library` to predict landmark positions on new specimens.

### Step 1. `Prepare`: Ensure SSM is loaded.
Follow [`Model Library` ⟶ Step 3 above](#step-3-database-library-load-database-into-morphoweave).

On setup and module re-entry, Landmark Transfer finds the most recently loaded complete canonical SSM set and fills only empty template, correspondence, landmark, and SSM table selectors in Single Run, Batch, and Template Optimization. It validates dense-point count against the SSM table. Existing manual selections and target models are never replaced by this resolver.

### Step 2. `Single Run`: Run landmark prediction for a single target mesh.
* **Template Model**: <your_ssm>_template
* **Template Correspondences**: <your_ssm>_template_correspondences
* **Template Landmarks**: <your_ssm>_template_sparse_landmarks
* **Target model**: Choose a model to predict landmarks for
* **SSM Data Table**: ssm_data_<your_ssm>

Load a model to predict landmarks for into your Scene using the 3D Slicer Add Data button.
<p align="center">
<img src="images/13a.png" width = 600>
</p>

Ensure all form fields are correctly filled.

Workflow labels use compact **Needs input**, **Ready**, **Optional**, and **Complete** states with restrained color accents. Optimization backend settings remain collapsed when the backend changes. In Advanced Settings, **Rigid registration** and **Deformation backend** also start collapsed.

<p align="center">
<img src="images/14.png" width = 600>
</p>

1) Subsample source/target.
Downsample source and target point clouds based on value set in Advanced ⟶ Point density and max projection ⟶ Point Density.

<p align="center">
<img src="images/15.png" width = 600>
</p>

2) Run rigid alignment.
Global (RANSAC) and rigid (ICP) registration that register the source point cloud to the target.

<p align="center">
<img src="images/16.png" width = 600>
</p>

2) b. Preview Rigid Alignment.

<p align="center">
<img src="images/17.png" width = 600>
</p>

3) Run deformable alignment.
Registration where source point cloud is deformed to target point cloud, then the registration is used to propagate the source landmarks to target specimen. Uses rustcpd atlas/SSM registration as a biological prior to avoid biologically implausible deformations, followed by deformable CPD to capture local details.

<p align="center">
<img src="images/18.png" width = 600>
</p>

4) Show final registration.

<p align="center">
<img src="images/19.png" width = 600>
</p>

### Step 3. Display and evaluate MorphoWeave results.
All results are saved in the Landmark Transfer runs folder that can be viewed in the 3D Slicer Data module. Rotate and inspect results (meshes, pointclouds, landmarks) to ensure the alignment and landmark prediction behaviors are as expected.

By default, only the Warped Source model (green), Target Model (blue), and the final MorphoWeave Refined Predicted Landmarks are displayed.

<p align="center">
<img src="images/20.png" width = 600>
</p>

To display the Landmark Predictions (pink) versus the Refined Predicted Landmarks (pink), change the color of the Refined Predicted Landmarks to blue. Then, toggle the visibility (eyeball button) to on.

<p align="center">
<img src="images/21.png" width = 600>
</p>

To display the rigidly registered Source Pointcloud (red) and Target Pointcloud (blue), toggle the visibility (eyeball button) to on and turn off any other visible nodes.

<p align="center">
<img src="images/22.png" width = 600>
</p>

### Step 4. (Optional): Fine tune landmark prediction parameters and repeat Steps 2-3 until desired results are achieved.
These steps are especially important for macroevolutionary comparative analyses that may deviate from the original SSM.

### Step 4a. (Optional): `Template Optimization`: Optimize template for Target model to improve landmark transfer prediction.
* **Template Model**: <your_ssm>_template
* **Template Correspondences**: <your_ssm>_template_correspondences
* **Template Landmarks**: <your_ssm>_template_sparse_landmarks
* **Target model**: Choose a model to predict landmarks for
* **SSM Data Table**: ssm_data_<your_ssm>
* **Optimization backend**:
  * **FPFH + RANSAC (current)** preserves the established feature-based template search and remains the default.
  * **Pose-marginalized EM (experimental)** evaluates deterministic global rotations and jointly refines SSM coefficients and similarity pose to select a target-informed template shape.

The current backend will either produce an optimized template or retain the baseline when it is already a good fit. The experimental backend applies the selected SSM shape in the original template frame and reports score margin, effective pose count, and evaluated/refined hypothesis counts. Pose-EM does not bypass any downstream stage: Single Run and Batch still perform the standard prescaling, RANSAC + ICP rigid alignment, PCA-CPD, and optional fine deformation.

Pose-EM settings use the recommended rustcpd configuration: an exact budget of 193 total pose hypotheses, trajectory scoring after eight coarse iterations with all 193 hypotheses retained, 12 finalists, full-source/1,600-target refinement for 30 iterations, and pose-specific SSM/outlier weights of 0.1/0.05. These pose-specific weights are independent of downstream PCA-CPD. Parallel pose search is enabled by default and uses rustcpd's deterministic internal pool. The initializer's refined coefficients are applied directly in the template frame. For incomplete targets, `Target completeness` prescales the SSM and fixes initializer scale; inspect ambiguity diagnostics carefully because partial or symmetric anatomy may support several poses.

<p align="center">
<img src="images/23.png" width = 600>
</p>

### Step 4b. (Optional): `Advanced`: Adjust advanced parameters.
`General Settings`
* **Skip scaling**:  Disables size normalization between template and target
* **Skip projection**: Skips final surface snap of landmarks
* **Skip template optimization (batch)**: Uses same template for all targets (faster)
* **Target completeness (linear fraction)**: Fraction of specimen intact (0.5 = 50% complete)

`Point density and max projection`
* **Point Density**: Higher = more points, slower. Default 1.3
* **Max projection factor (%)**: Max distance landmarks move to surface (% of size)

`Rigid registration`
* **Normal search radius**: Radius for surface normal estimation. Higher = smoother
* **FPFH search radius**: Feature descriptor neighborhood size. Higher = more robust
* **RANSAC distance threshold**: Max distance for point match. Higher = more tolerant
* **Max RANSAC iterations**: Max attempts to find alignment. Default 400k
* **RANSAC confidence**: Probability of finding good match. Default 0.999
* **ICP distance threshold**: Refinement tolerance after RANSAC. Default 0.4

`Deformation backend`
* **Experimental: Use biharmonic surface warp**: Experimental alternative to TPS (less stable)
* **Biharmonic stiffness (lambda)**: Constraint strength for biharmonic warp
* **TPS smoothing (λ)**: Regularization. 0=exact fit, higher=smoother approximation
* **TPS max constraints**: Max correspondences used. Higher=detail but slower

`PCA-CPD registration`
* **Rigidity (alpha)**: Smoothness. Higher = stiffer, more global deformation
* **Motion coherence (beta)**: Spatial correlation width. Higher = smoother coupling
* **Fossil mode (SSM-only)**: Skip free-form stage for incomplete specimens
* **Outlier weight (w)**: Expected outlier fraction. Higher for partial data
* **Tolerance**: Convergence threshold. Lower = more precise, slower
* **Max iterations**: Iteration limit for CPD. Default 250
* **SSM weight (lambda_reg)**: SSM constraint strength. Higher = closer to mean

### Step 5. `Batch`: Run landmark prediction for a directory of target meshes.
Use defaults or parameters chosen under `Template Optimization` and `Advanced` to run landmark prediction in batch mode.

* **Source mesh**: the database template, `GridRANSAC_TemplateModel`, or `PoseEM_TemplateModel`
* **Source Correspondences**: the matching template correspondence node
* **Source landmarks**: the matching template landmark node
* **Target mesh directory**: folder with your meshes to be landmarked
* **Target output landmark directory**: your output landmark directory name (ex: predictedLMs)
* **Save warped meshes**: Defaults to False. Saves warped meshes used for landmark transfer.
* **Warped mesh output directory**: If saving warped meshes, specify your warped mesh output landmark directory name (ex: warpedMeshes)
* **Smooth exported warped meshes**: Defaults to False. If saving warped meshes, specify to smooth them.
* **SSM Data Table**: ssm_data_<your_ssm>
* **Skip template optimization**: Defaults to False. Skips the backend selected in the Template Optimization tab. Both backends feed the same downstream scaling, rigid, and deformable pipeline. Pose-EM batch diagnostics are saved as `pose_em_diagnostics.json` in the landmark output directory.

<p align="center">
<img src="images/24.png" width = 600>
</p>

## Surface Segmentation

Use the dense population correspondences generated by **Atlas Builder** to partition homologous surface regions consistently across specimens.

### Step 1. Select paired inputs

- **Meshes folder**: the directory containing target meshes.
- **Dense Correspondences folder**: the directory containing matching `.mrk.json` correspondence files.
- **Output folder**: the directory where segmented region meshes and label lookup files will be written.

Mesh and correspondence filenames must share the same base name.

### Step 2. Configure segmentation

Choose the number of segments, correspondence-neighborhood size, smoothing iterations, and feature-versus-spatial weighting. The optional auto-tuning mode evaluates candidate segment counts and feature weights using graph modularity.

### Step 3. Run and inspect

Click **Run Segmentation**. MorphoWeave writes each segmented region as a PLY file, saves the correspondence-locus labels, and previews the requested number of segmented models in the Slicer scene. Inspect the previews for anatomical consistency before using the labels downstream.
