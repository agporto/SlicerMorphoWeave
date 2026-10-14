# Atlas Builder lineage and attribution

The MorphoWeave **Atlas Builder** module was originally adapted from the Dense Correspondence Landmarking (**DeCAL**) workflow in [SlicerDenseCorrespondenceAnalysis](https://github.com/SlicerMorph/SlicerDenseCorrespondenceAnalysis), developed by the SlicerMorph project.

Atlas Builder retains the central DeCAL concept of using sparse landmark correspondences to align specimens, deform a template surface, and export index-consistent dense correspondence points. It has since been substantially restructured and extended for the MorphoWeave workflow.

Major differences and additions include:

- a focused atlas-construction workflow designed to feed the MorphoWeave Model Library and Landmark Transfer modules;
- robust, case-insensitive model-landmark pairing across supported file formats;
- coordinate-system validation for detecting likely landmark/mesh orientation mismatches;
- mesh cleaning and warnings for unusually dense surfaces;
- pre-alignment to the specimen closest to the Procrustes mean before mean-atlas construction;
- optional biharmonic deformation with automatic thin-plate spline fallback;
- explicit point-count preview and index-stable dense correspondence export; and
- standardized output directories for aligned models, aligned landmarks, atlas assets, and population correspondences.

Dense export uses deterministic target-count voxel sampling. The default target is 2,000 existing surface-connected atlas vertices and can be overridden in Advanced Options. The same selected vertex indices are propagated to every specimen; when the atlas contains fewer surface-connected vertices than requested, all available surface vertices are used. Unreferenced vertices are excluded because they have no surface location to preserve during projection.

Users of Atlas Builder should cite both MorphoWeave and the relevant SlicerDenseCorrespondenceAnalysis/DeCA software and publication.

The upstream BSD 2-Clause copyright and license notice is reproduced in [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Correctness contracts and regression tests

- **Scale normalization:** enabled (the default) uses similarity GPA and a normalized mean atlas. Disabled propagates rigid-only GPA through closest-specimen selection, mean-atlas construction, and final alignment, preserving physical size in both the atlas and population. The mesh and sparse landmarks use the same mean frame.
- **Checked exports:** every aligned, atlas, and dense file must have a successful Slicer writer result and a nonempty output file. Dense coordinates and the expected specimen set are checked before writing. A write failure stops the workflow, prevents Model Library ingestion, and is reported as failure rather than completion. Partial files may remain for diagnosis; this is not an atomic multi-file transaction.
- **Biharmonic vertex identity:** the solver triangulates only to build its operator; it does not clean/reorder the geometry used for output. Sample IDs still name the same original atlas loci even when cells/vertices are reordered or unused/duplicate points exist.
- **Experimental biharmonic constraints:** the solver transfers target-minus-source displacements to nearest vertices, averaging displacements when multiple controls select one vertex. Identity, exact global rigid motion, and fully constrained meshes are handled directly. In general this is an approximate nearest-vertex deformation, not exact off-vertex/barycentric interpolation. Unconstrained disconnected components remain at rest for nonrigid solves. Numerical solve failure raises an error when fallback is disabled, or uses a finite, vertex-count-checked TPS result when enabled. Malformed/non-finite inputs are rejected rather than sent to TPS.

The focused regressions execute the module's actual method bodies with real NumPy, SciPy, and VTK:

```sh
python MorphoWeaveAtlasBuilder/Testing/Python/AtlasBuilderWorkflowTest.py
python MorphoWeaveAtlasBuilder/Testing/Python/AtlasBuilderCorrectnessTest.py
```

The correctness suite covers audit findings AM04, AM05, AM06, and AM09: physical-size/unit-conversion contracts, default normalization, control propagation, original vertex IDs, identity/rigid/all-constrained/duplicate-control/disconnected cases, solver-failure fallback, each aligned/atlas/dense writer failure, missing/empty writer outputs, and failed-run ingestion guards. Both scripts are registered with Slicer's CTest integration.

Outside Slicer, AST loading bypasses application setup, not the methods under test. MRML nodes, UI controls, and file writers are explicit test doubles; real temporary files exercise the write-result checks. A live Slicer run is still needed to validate Qt signals, MRML observers/hardening, and actual file-format writer roundtrips. These tests do not establish biological correspondence accuracy or validate the experimental backend as a scientific alternative to TPS.
