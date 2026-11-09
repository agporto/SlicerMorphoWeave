# Your first landmark prediction with MorphoWeave

**Build an atlas. Explore how skull shape varies. Transfer landmarks to one skull.**

You’ll start with mouse skull surfaces that already have landmarks and finish with predicted landmarks on a skull you kept out of the atlas. Everything happens in 3D Slicer.

<img src="images/20.png" width="880" alt="Slicer Data panel beside a mouse skull showing a blue target, a green warped template, and predicted landmarks">

*The finish line. The blue surface is the target skull. The green surface is the atlas warped to fit it. The labeled points are the predictions you’ll inspect and save.*

[Get ready](#get-ready) · [1. Build](#1-build-an-atlas) · [2. Explore](#2-explore-shape-variation) · [3. Transfer](#3-transfer-to-one-skull) · [4. Check and save](#4-check-and-save)

## Get ready

1. **Install MorphoWeave.** Open 3D Slicer and search for **MorphoWeave** in the **Extension Manager**. If it appears, install it and restart Slicer. Otherwise, follow the [repository installation instructions](../README.md#developer-installation). The module menu has a **MorphoWeave** category.
2. **Download and extract the data.** Start with the [MorphoWeave sample data on Figshare](https://figshare.com/s/3ff773754624c9132e3e). If the link is unavailable, use [SlicerMorph Mouse Models](https://github.com/SlicerMorph/Mouse_Models) → **Code** → **Download ZIP**. The captures below use this mouse-skull collection, with meshes in `Models` and manual landmarks in `LMs`.
3. **Set one skull aside.** In a working copy, move one complete skull mesh and its matching manual landmark file outside the training folders. For **Mouse Models**, use `Models/A_J.ply` and `LMs/A_J.mrk.json`. Keep the manual landmarks for the last screen. Create a separate empty output folder such as `morphoweave-output`.

> The screen captures show an earlier build. Folder names, colors, and some labels may look different. Follow the control names in the numbered instructions.

## 1. Build an atlas

<img src="images/5.png" width="760" alt="Atlas Builder Required Inputs with Create atlas from inputs selected and model, landmark, and output directories filled">

*Find these three folder fields in **Atlas Builder**.*

1. Open **MorphoWeave → Atlas Builder** and select **Create atlas from inputs**.
2. Set **Model directory** to your training meshes, **Landmark directory** to their matching manual landmarks, and **Output directory** to `morphoweave-output`. Each mesh should have a corresponding landmark file with the same specimen name. The landmarks must identify the same anatomical locations in the same order across specimens.
3. Leave **Advanced Options** at their defaults. Under **Run + Status**, click **Run Atlas Builder Pipeline** and wait for the output location to appear in the log.

<img src="images/6a.png" width="760" alt="Atlas Builder showing the generated yellow mouse skull atlas in the 3D viewer">

**Checkpoint.** An atlas skull appears in the 3D viewer. Open the timestamped run folder reported in the log. You’ll use its `atlas/` and `population_correspondences/` folders in the next screen.

The **atlas** is a shared reference surface built from the aligned skulls. **Sparse landmarks** are the named points you want to predict. **Dense correspondences** give the shape model many more points to work with. Point 500, for example, represents the same mapped location across all training skulls.

## 2. Explore shape variation

<img src="images/9.png" width="800" alt="Model Library Ingest New SSM form with the atlas mesh, dense correspondences, sparse landmarks, and population correspondence folder selected">

*Use four outputs from the same Atlas Builder run.*

1. Open **MorphoWeave → Model Library**. In **Database Library**, keep the default **Database Location** or choose a writable folder where you want models to persist.
2. Expand **Ingest New SSM**. Choose the following paths **inside the timestamped run folder**, enter `mouse_db` under **New Database Name**, then click **Ingest into Database**.

   | Field | Atlas Builder output |
   | --- | --- |
   | **Template Model** | `atlas/atlas_model.ply` |
   | **Dense Correspondences** | `atlas/atlas_dense_correspondences.mrk.json` |
   | **Sparse Landmarks** | `atlas/atlas_sparse_landmarks.mrk.json` |
   | **Population Correspondences Folder** | `population_correspondences/` |

3. Under **Database Library**, select `mouse_db` and click **Load Selected Database**.
4. In **SSM Explorer**, move **PC1** to about `-2` and then `+2`. Watch the snout and braincase from the same view. Return the slider to `0` before continuing.

| PC1 toward the left | PC1 toward the right |
| --- | --- |
| <img src="images/13.png" width="490" alt="SSM Explorer with PC1 at the left end and the resulting mouse skull shape"> | <img src="images/12.png" width="490" alt="SSM Explorer with PC1 at the right end and the resulting mouse skull shape"> |

**Checkpoint.** The skull changes shape as you move the slider. A **statistical shape model (SSM)** describes how the training skulls vary around the reference. **PC1** is the strongest pattern of shape variation in that training set. The slider shows that pattern in opposite directions.

## 3. Transfer to one skull

<img src="images/14.png" width="800" alt="Landmark Transfer Single Run tab with a template model, dense correspondences, sparse landmarks, target model, and SSM data table selected">

*The template is where the landmarks come from. The target is the skull that receives them.*

1. Click **Add Data** in Slicer and load the skull mesh you set aside. Leave its manual landmarks outside the scene for now.
2. Open **MorphoWeave → Landmark Transfer → Single Run**. Select your held-out skull as **Target model**. If the other selectors are empty or show a different model, choose the matching `mouse_db` entries.

   | Field | Scene node |
   | --- | --- |
   | **Template model** | `mouse_db_template` |
   | **Template correspondences** | `mouse_db_template_correspondences` |
   | **Template landmarks** | `mouse_db_template_sparse_landmarks` |
   | **SSM Data Table** | `ssm_data_mouse_db` |

3. Click **1) Subsample source/target** to create smaller point clouds for registration. On first use, Slicer may ask to install registration dependencies. Allow the installation and follow any restart prompt, then reload your model and target if needed.
4. Click **2) Run rigid alignment**, then **2b) Preview Rigid Alignment**. Rotate the skulls. Check that the snout, eye sockets, and back of the skull broadly overlap. If they point in different directions or the surfaces barely overlap, stop here and check the target and template selections.
5. Click **3) Run deformable alignment**, then **4) Show final registration**. Rigid alignment brings the skulls into the same pose. Deformable alignment changes the template’s shape to fit the target and carries the landmarks along with it.

<img src="images/17.png" width="880" alt="Rigid alignment preview with the red source skull and blue target skull in the same pose">

**Checkpoint.** At the rigid preview, the skulls face the same way and their main structures broadly line up. After deformation, the landmarks should be visible on the target. The opening capture shows that final view.

## 4. Check and save

<img src="images/21.png" width="880" alt="Slicer Data panel with a transfer run expanded beside the skull and its labeled points">

*Use the eye icons in **Data** to compare the prediction with the manual landmarks.*

1. Open Slicer’s **Data** module. Expand **MorphoWeave Landmark Transfer runs** and use the eye icons to hide **Warped Source** so you can see the target and predictions clearly.
2. Load the held-out manual landmark file with **Add Data**. Toggle between it and **Refined Predicted Landmarks**. Compare several labeled points from both sides and underneath the skull. Check the anatomical feature and side for each point. A point can be on the surface and still be on the wrong feature.
3. Click Slicer’s **Save** button. Select the predicted markup, choose **Markups JSON** (`.mrk.json`), and save it under a distinct name such as `A_J_predicted.mrk.json`. Keep it separate from the manual annotation. Save the scene too if you want to reopen the comparison.

**You’re done when** you can point to a prediction that agrees with the manual annotation, describe any mismatch you find, and locate your saved markup file. One skull is a practice check. To assess accuracy for a study, compare predictions against manual landmarks across multiple held-out specimens.

<details>
<summary>If you get stuck</summary>

| What you see | What to check |
| --- | --- |
| Atlas Builder cannot pair inputs | The paths should point directly to the mesh and landmark folders. Confirm matching specimen stems and consistent landmark counts. |
| Atlas Builder flags landmark coordinates | Open one mesh with its own manual markups in Slicer. Check that the landmarks sit on the correct anatomy and verify the mesh and markup coordinate conventions. |
| The mouse dataset fails on `FVB_NJ` | Some copies contain a mesh that needs repair. For this exercise, move both `FVB_NJ.ply` and `FVB_NJ.mrk.json` out of the training folders and retry. |
| The SSM is missing or the dense point counts differ | Load the database again. Select its template, correspondence, landmark, and table nodes as a set. |
| Predictions disappear or are hard to see | In **Data**, turn on the target and prediction nodes, hide the warped template and manual markups, and center the 3D view. With projection disabled, look for **Predicted Landmarks (no projection)**. |

</details>

Once you’ve checked a single skull, the [module walkthrough](README.md) covers batch transfer, template optimization, and surface segmentation.
