# Publishing MorphoWeave in the 3D Slicer Extensions Catalog

MorphoWeave is prepared for an initial Tier 1 submission to the Slicer ExtensionsIndex.

## Repository configuration

Before opening the ExtensionsIndex pull request:

- Add the GitHub topic `3d-slicer-extension` to the SlicerMorphoWeave repository.
- Keep `SlicerMorph` documented as a complementary extension, not a build dependency.
- Confirm that the renamed module icons load and that `EXTENSION_ICONURL` resolves from the `main` branch.
- Confirm that `EXTENSION_SCREENSHOTURLS` resolves to `tutorial/images/20.png` from the `main` branch.
- Merge this publication-preparation branch into `main`.

## Catalog entry

Copy `ExtensionsIndex/MorphoWeave.json` into the root of a fork of `Slicer/ExtensionsIndex`, then open a pull request against its `main` branch.

The entry intentionally uses:

- Repository: `https://github.com/agporto/SlicerMorphoWeave.git`
- Category: `Registration`
- Revision: `main`
- Build dependencies: none
- Tier: 1

## Minimal validation

MorphoWeave retains its existing lightweight scripted-module tests. Before submission, perform a clean package/install smoke test in a current Slicer Preview build:

1. Configure and package the extension through Slicer's extension build workflow.
2. Install the generated package into a clean Slicer profile.
3. Restart Slicer and open Atlas Builder, Model Library, Landmark Transfer, and Surface Segmentation.
4. Confirm that module icons and packaged Python resources load.
5. Confirm that Landmark Transfer offers installation of compatible `tiny3d-rs` and `rustcpd` releases when they are absent, and that replacing a loaded legacy `tiny3d` installation requests a Slicer restart.

No SlicerMorph dependency or bundled test dataset is required for the initial Tier 1 submission.

## ExtensionsIndex pull-request notes

State that:

- The repository follows the recommended `Slicer+ExtensionName` naming convention while the extension remains displayed as MorphoWeave in Slicer.
- The extension operates locally and does not upload user data.
- Python dependencies are obtained from the configured Python package index with explicit user approval.
- The source is released under the BSD 2-Clause License.
