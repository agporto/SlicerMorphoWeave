import csv
import html
import importlib
import importlib.metadata
import json
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import ctk
import numpy as np
import qt
import slicer
import vtk
import vtk.util.numpy_support as vtk_np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)


HDMSEG_REQUIREMENT = "hdmseg>=0.2,<0.3"
MODEL_EXTENSIONS = {".ply", ".stl", ".obj", ".vtp", ".vtk"}


def setWorkflowStatus(label, state, detail):
    title, color = {
        "needs_input": ("Needs input", "#c75b5b"),
        "ready": ("Ready", "#4f9d69"),
        "optional": ("Optional", "#c18b37"),
        "complete": ("Complete", "#4f9d69"),
    }[state]
    safe = html.escape(str(detail)).replace("\n", "<br>")
    label.setTextFormat(qt.Qt.RichText)
    label.setText(
        f'<span style="color:{color}; font-weight:600">{title}</span> — {safe}'
    )
    label.setAccessibleName(f"{title}: {detail}")


def markupsStem(path):
    name = Path(path).name
    suffix = ".mrk.json"
    return name[: -len(suffix)] if name.lower().endswith(suffix) else Path(name).stem


def stableMajorMinor(version):
    match = re.fullmatch(
        r"\s*(\d+)\.(\d+)(?:\.\d+)*(?:\.post\d+)?"
        r"(?:\+[A-Za-z0-9][A-Za-z0-9._-]*)?\s*",
        str(version),
    )
    return (int(match.group(1)), int(match.group(2))) if match else None


class MorphoWeaveSurfaceSegmentation(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "Surface Segmentation"
        parent.categories = ["MorphoWeave"]
        parent.dependencies = []
        parent.contributors = ["Arthur Porto"]
        parent.helpText = (
            "Partition population-wide dense correspondences with hdmseg consensus "
            "diffusion and propagate the shared labels to paired full-resolution meshes."
        )
        parent.acknowledgementText = "This module was developed by Arthur Porto"


class MorphoWeaveSurfaceSegmentationWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logic = None
        self._running = False

    def setup(self):
        super().setup()
        self.logic = MorphoWeaveSurfaceSegmentationLogic()
        self._buildUI()
        self._refreshDependencyStatus()
        self._validateInputs()

    @staticmethod
    def _path(widget):
        value = getattr(widget, "currentPath", "")
        return str(value() if callable(value) else value).strip()

    @staticmethod
    def _value(widget):
        value = getattr(widget, "value", 0)
        return value() if callable(value) else value

    @staticmethod
    def _checked(widget):
        method = getattr(widget, "isChecked", None)
        return bool(method() if callable(method) else getattr(widget, "checked", False))

    @staticmethod
    def _index(widget):
        value = getattr(widget, "currentIndex", 0)
        return int(value() if callable(value) else value)

    def _pathWidget(self, files=False, nameFilters=None):
        widget = ctk.ctkPathLineEdit()
        widget.filters = ctk.ctkPathLineEdit.Files if files else ctk.ctkPathLineEdit.Dirs
        if nameFilters:
            widget.nameFilters = list(nameFilters)
        return widget

    def _buildUI(self):
        root = ctk.ctkCollapsibleButton()
        root.text = "Surface Segmentation Workflow"
        root.collapsed = False
        self.layout.addWidget(root)
        form = qt.QFormLayout(root)

        info = qt.QLabel(
            "hdmseg builds one population-consensus diffusion operator from the "
            "dense correspondence stack, clusters homologous loci once, and applies "
            "the same labels to every paired mesh."
        )
        info.setWordWrap(True)
        form.addRow(info)

        required = ctk.ctkCollapsibleButton()
        required.text = "Required Inputs"
        required.collapsed = False
        requiredForm = qt.QFormLayout(required)
        form.addRow(required)

        self.meshDir = self._pathWidget()
        self.mrkDir = self._pathWidget()
        self.referencePath = self._pathWidget(
            files=True, nameFilters=["Slicer Markups (*.mrk.json)"]
        )
        self.outDir = self._pathWidget()
        requiredForm.addRow("Meshes folder:", self.meshDir)
        requiredForm.addRow("Dense correspondences folder:", self.mrkDir)
        requiredForm.addRow("Reference correspondences (optional):", self.referencePath)
        requiredForm.addRow("Output folder:", self.outDir)
        self.outDir.setToolTip(
            "Each run creates a new subfolder here. Previous runs and unrelated files "
            "are never overwritten or mixed with the new segmentation."
        )

        backend = ctk.ctkCollapsibleButton()
        backend.text = "hdmseg Backend"
        backend.collapsed = False
        backendForm = qt.QFormLayout(backend)
        form.addRow(backend)
        self.backendStatus = qt.QLabel("")
        self.backendStatus.setWordWrap(True)
        backendForm.addRow(self.backendStatus)
        self.installButton = qt.QPushButton("Install or update hdmseg")
        self.installButton.clicked.connect(self._installBackend)
        backendForm.addRow(self.installButton)

        settings = ctk.ctkCollapsibleButton()
        settings.text = "Segmentation"
        settings.collapsed = False
        settingsForm = qt.QFormLayout(settings)
        form.addRow(settings)

        self.selection = qt.QComboBox()
        self.selection.addItems(
            [
                "Fixed number of regions",
                "Automatic: bootstrap stability",
                "Automatic: modularity",
                "Automatic: eigengap",
            ]
        )
        self.selection.setCurrentIndex(0)
        self.fixedK = qt.QSpinBox()
        self.fixedK.setRange(2, 128)
        self.fixedK.setValue(15)
        self.maxK = qt.QSpinBox()
        self.maxK.setRange(2, 128)
        self.maxK.setValue(20)
        self.neighbors = qt.QSpinBox()
        self.neighbors.setRange(1, 128)
        self.neighbors.setValue(12)
        self.components = qt.QSpinBox()
        self.components.setRange(1, 128)
        self.components.setValue(20)
        self.bootstraps = qt.QSpinBox()
        self.bootstraps.setRange(1, 500)
        self.bootstraps.setValue(20)
        self.seed = qt.QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(0)
        self.parallel = qt.QCheckBox("Use hdmseg parallel execution")
        self.parallel.setChecked(True)
        settingsForm.addRow("Region selection:", self.selection)
        settingsForm.addRow("Fixed regions (k):", self.fixedK)
        settingsForm.addRow("Maximum automatic k:", self.maxK)
        settingsForm.addRow("Reference neighbors:", self.neighbors)
        settingsForm.addRow("Diffusion coordinates:", self.components)
        settingsForm.addRow("Bootstrap replicates:", self.bootstraps)
        settingsForm.addRow("Random seed:", self.seed)
        settingsForm.addRow(self.parallel)

        output = ctk.ctkCollapsibleButton()
        output.text = "Mesh Projection and Outputs"
        output.collapsed = True
        outputForm = qt.QFormLayout(output)
        form.addRow(output)
        self.smoothing = qt.QSpinBox()
        self.smoothing.setRange(0, 20)
        self.smoothing.setValue(1)
        self.writeVtp = qt.QCheckBox("Write labeled VTP for each specimen")
        self.writeVtp.setChecked(True)
        self.writePly = qt.QCheckBox("Write one PLY file per region")
        self.writePly.setChecked(True)
        self.previewN = qt.QSpinBox()
        self.previewN.setRange(0, 50)
        self.previewN.setValue(3)
        outputForm.addRow("Mesh label smoothing:", self.smoothing)
        outputForm.addRow(self.writeVtp)
        outputForm.addRow(self.writePly)
        outputForm.addRow("Preview first N specimens:", self.previewN)

        self.status = qt.QLabel("")
        self.status.setWordWrap(True)
        form.addRow(self.status)
        self.runButton = qt.QPushButton("Run hdmseg Surface Segmentation")
        self.runButton.clicked.connect(self.onRun)
        form.addRow(self.runButton)
        self.layout.addStretch(1)

        self.selection.currentIndexChanged.connect(self._selectionChanged)
        for widget in (self.meshDir, self.mrkDir, self.referencePath, self.outDir):
            widget.connect("validInputChanged(bool)", self._validateInputs)
        self._selectionChanged()

    def _selectionMode(self):
        return {0: "fixed", 1: "stability", 2: "modularity", 3: "eigengap"}[
            self._index(self.selection)
        ]

    def _selectionChanged(self, *_):
        mode = self._selectionMode()
        self.fixedK.enabled = not self._running and mode == "fixed"
        self.maxK.enabled = not self._running and mode != "fixed"
        self.bootstraps.enabled = not self._running and mode == "stability"

    def _setRunning(self, running):
        """Prevent reentry, including while dependency installation pumps events."""
        self._running = bool(running)
        for widget in (
            self.meshDir, self.mrkDir, self.referencePath, self.outDir,
            self.selection, self.fixedK, self.maxK, self.neighbors,
            self.components, self.bootstraps, self.seed, self.parallel,
            self.smoothing, self.writeVtp, self.writePly, self.previewN,
            self.installButton,
        ):
            widget.enabled = not self._running
        self.runButton.enabled = False
        if not self._running:
            self._selectionChanged()
            self._validateInputs()

    def _installedVersion(self):
        try:
            version = importlib.metadata.version("hdmseg")
        except importlib.metadata.PackageNotFoundError:
            return None, False
        return version, stableMajorMinor(version) == (0, 2)

    def _refreshDependencyStatus(self):
        version, compatible = self._installedVersion()
        if compatible:
            setWorkflowStatus(self.backendStatus, "ready", f"hdmseg {version} is installed.")
        elif version is None:
            setWorkflowStatus(
                self.backendStatus,
                "optional",
                f"{HDMSEG_REQUIREMENT} is not installed. Installation is offered when needed.",
            )
        else:
            setWorkflowStatus(
                self.backendStatus,
                "optional",
                f"Installed hdmseg {version} is outside the supported 0.2.x series.",
            )

    def _ensureBackend(self):
        import slicer.packaging

        version, compatible = self._installedVersion()
        loaded = any(name == "hdmseg" or name.startswith("hdmseg.") for name in sys.modules)
        if loaded and not compatible:
            slicer.util.infoDisplay(
                "An incompatible hdmseg version is already loaded. Restart Slicer, "
                "then install the supported 0.2.x release."
            )
            return None
        try:
            slicer.packaging.pip_ensure(
                [HDMSEG_REQUIREMENT],
                prompt_install=True,
                requester="Surface Segmentation",
            )
            importlib.invalidate_caches()
            module = importlib.import_module("hdmseg")
        except Exception as error:
            if isinstance(error, RuntimeError) and str(error) == "User declined package installation":
                slicer.util.showStatusMessage("hdmseg was not installed.", 3000)
                return None
            logging.exception("hdmseg setup failed")
            slicer.util.errorDisplay(f"hdmseg setup failed:\n{error}")
            return None
        if stableMajorMinor(getattr(module, "__version__", "")) != (0, 2):
            slicer.util.errorDisplay(
                f"Surface Segmentation requires {HDMSEG_REQUIREMENT}; imported "
                f"hdmseg {getattr(module, '__version__', 'unknown')}."
            )
            return None
        self._refreshDependencyStatus()
        return module

    def _installBackend(self):
        if self._running:
            return
        self._setRunning(True)
        try:
            if self._ensureBackend() is not None:
                slicer.util.showStatusMessage("Surface Segmentation: hdmseg ready", 3000)
        finally:
            self._setRunning(False)

    def _validateInputs(self, *_):
        if self._running:
            self.runButton.enabled = False
            return
        meshDir, mrkDir, outDir = map(
            self._path, (self.meshDir, self.mrkDir, self.outDir)
        )
        reference = self._path(self.referencePath)
        if not (os.path.isdir(meshDir) and os.path.isdir(mrkDir) and os.path.isdir(outDir)):
            self.runButton.enabled = False
            setWorkflowStatus(
                self.status,
                "needs_input",
                "Select valid mesh, correspondence, and output folders.",
            )
            return
        if reference and not os.path.isfile(reference):
            self.runButton.enabled = False
            setWorkflowStatus(self.status, "needs_input", "Reference file does not exist.")
            return
        try:
            pairs = self.logic.pairFiles(meshDir, mrkDir)
        except Exception as error:
            self.runButton.enabled = False
            setWorkflowStatus(self.status, "needs_input", str(error))
            return
        self.runButton.enabled = bool(pairs)
        if pairs:
            setWorkflowStatus(
                self.status,
                "ready",
                f"{len(pairs)} paired specimens detected. Blank reference uses the "
                "Atlas Builder reference when available, otherwise the specimen mean.",
            )
        else:
            setWorkflowStatus(
                self.status,
                "needs_input",
                "No matching mesh and .mrk.json specimen names were found.",
            )

    def onRun(self):
        if self._running:
            return
        self._setRunning(True)
        result = None
        try:
            backend = self._ensureBackend()
            if backend is None:
                return
            with slicer.util.tryWithErrorDisplay(
                "Surface Segmentation failed", waitCursor=True
            ):
                result = self.logic.run(
                    meshesDir=self._path(self.meshDir),
                    mrkDir=self._path(self.mrkDir),
                    outDir=self._path(self.outDir),
                    hdmseg=backend,
                    referencePath=self._path(self.referencePath) or None,
                    selection=self._selectionMode(),
                    fixedK=int(self._value(self.fixedK)),
                    maxK=int(self._value(self.maxK)),
                    neighbors=int(self._value(self.neighbors)),
                    components=int(self._value(self.components)),
                    bootstraps=int(self._value(self.bootstraps)),
                    seed=int(self._value(self.seed)),
                    parallel=self._checked(self.parallel),
                    smoothing=int(self._value(self.smoothing)),
                    writeVtp=self._checked(self.writeVtp),
                    writePly=self._checked(self.writePly),
                    previewN=int(self._value(self.previewN)),
                )
        finally:
            self._setRunning(False)
        if result is None:
            return
        message = (
            f"Surface Segmentation: {result['specimens']} specimens, "
            f"k={result['k']}, modularity={result['modularity']:.3f}"
        )
        if result["stability"] is not None:
            message += f", stability={result['stability']:.3f}"
        if result["skipped"]:
            message += f", {result['skipped']} mesh exports skipped"
        slicer.util.showStatusMessage(message, 8000)
        setWorkflowStatus(self.status, "complete", f"Results written to {result['output_directory']}")


class MorphoWeaveSurfaceSegmentationLogic(ScriptedLoadableModuleLogic):
    def run(
        self,
        meshesDir,
        mrkDir,
        outDir,
        hdmseg,
        referencePath=None,
        selection="fixed",
        fixedK=15,
        maxK=20,
        neighbors=12,
        components=20,
        bootstraps=20,
        seed=0,
        parallel=True,
        smoothing=1,
        writeVtp=True,
        writePly=True,
        previewN=3,
    ):
        if selection not in {"fixed", "stability", "modularity", "eigengap"}:
            raise ValueError(f"Unknown selection mode: {selection}")
        if not callable(getattr(hdmseg, "segment", None)):
            raise RuntimeError("The loaded hdmseg module does not provide segment().")

        outputRoot = Path(outDir).expanduser().resolve()
        outputRoot.mkdir(parents=True, exist_ok=True)
        self._clearPreviews()
        pairs = self.pairFiles(meshesDir, mrkDir)
        if not pairs:
            raise RuntimeError("No mesh ↔ .mrk.json pairs were found.")

        started = time.perf_counter()
        X = self._loadStack([mrk for _, mrk in pairs])
        n, m, d = X.shape
        if d != 3 or m < 3:
            raise RuntimeError(f"Expected at least three 3-D loci; received {X.shape}.")
        reference, referenceName = self._reference(referencePath, mrkDir, X)
        neighbors = min(max(1, int(neighbors)), m - 1)
        components = min(max(1, int(components)), m - 1)
        fixedK = min(max(2, int(fixedK)), m)
        maxK = min(max(2, int(maxK)), m)
        graphComponents = self._graphComponents(reference, neighbors)
        warnings = []
        if selection == "stability" and n < 15:
            warnings.append(
                "Bootstrap stability is most informative with roughly 15 or more specimens."
            )
        if graphComponents > 1:
            warnings.append(
                f"The reference kNN graph has {graphComponents} connected components; "
                "hdmseg will use its validated dense eigensolver fallback."
            )

        slicer.util.showStatusMessage(
            f"Surface Segmentation: running hdmseg on {n} × {m} loci…", 0
        )
        slicer.app.processEvents()
        solveStarted = time.perf_counter()
        result = hdmseg.segment(
            np.ascontiguousarray(X, dtype=np.float64),
            k=fixedK if selection == "fixed" else None,
            n_neighbors=neighbors,
            n_components=components,
            diffusion_time=1.0,
            select="stability" if selection == "fixed" else selection,
            max_k=maxK,
            n_boot=max(1, int(bootstraps)),
            reference=np.ascontiguousarray(reference, dtype=np.float64),
            seed=max(0, int(seed)),
            parallel=bool(parallel),
        )
        solveSeconds = time.perf_counter() - solveStarted
        labels = np.asarray(result.labels, dtype=np.int32)
        if labels.shape != (m,):
            raise RuntimeError(f"hdmseg returned labels with shape {labels.shape}; expected {(m,)}.")

        embedding = np.asarray(result.embedding, dtype=np.float64)
        eigenvalues = np.asarray(result.eigenvalues, dtype=np.float64)
        # Exclusive directory creation isolates reruns even when they start in
        # the same second. Never delete or reuse an earlier run's region files.
        output = Path(tempfile.mkdtemp(
            prefix="MorphoWeaveSurfaceSegmentation_" + time.strftime("%Y%m%d_%H%M%S") + "_",
            dir=str(outputRoot),
        ))
        self._writeLocusOutputs(output, labels, embedding, eigenvalues)

        vtkFiles, plyFiles, skipped = [], [], []
        meshStarted = time.perf_counter()
        self._colorTable(int(result.k))
        for index, (meshPath, mrkPath) in enumerate(pairs):
            slicer.util.showStatusMessage(
                f"Surface Segmentation: projecting labels {index + 1}/{len(pairs)}…", 0
            )
            if index % 5 == 0:
                slicer.app.processEvents()
            try:
                vertices, faces, polydata = self._loadMesh(
                    meshPath, needFaces=int(smoothing) > 0 or bool(writePly)
                )
                specimenLabels = self._project(vertices, X[index], labels)
                smoothLabels = self._smooth(
                    vertices, faces, specimenLabels.copy(), int(smoothing)
                )
                stem = Path(meshPath).stem
                if writeVtp:
                    vtkFiles.append(
                        self._writeVtp(output, stem, polydata, specimenLabels, smoothLabels)
                    )
                if writePly:
                    plyFiles.extend(
                        self._writePlys(output, stem, polydata, faces, smoothLabels)
                    )
                if index < max(0, int(previewN)):
                    self._preview(stem, polydata, smoothLabels)
            except Exception as error:
                logging.exception("Could not export %s", Path(meshPath).name)
                skipped.append(
                    {"mesh": str(meshPath), "error": f"{type(error).__name__}: {error}"}
                )
        meshSeconds = time.perf_counter() - meshStarted

        stability = None if result.stability is None else float(result.stability)
        summary = {
            "backend": "hdmseg",
            "backend_version": str(getattr(hdmseg, "__version__", "unknown")),
            "specimen_count": int(n),
            "locus_count": int(m),
            "reference_correspondences": str(referenceName),
            "reference_graph_components": int(graphComponents),
            "selection": selection,
            "k": int(result.k),
            "n_neighbors": int(neighbors),
            "n_components": int(components),
            "max_k": int(maxK),
            "n_boot": int(max(1, bootstraps)) if selection == "stability" else None,
            "seed": int(max(0, seed)),
            "parallel": bool(parallel),
            "modularity": float(result.modularity),
            "stability": stability,
            "mesh_label_smoothing_iterations": int(max(0, smoothing)),
            "timing_seconds": {
                "hdmseg": float(solveSeconds),
                "mesh_projection_and_export": float(meshSeconds),
                "total": float(time.perf_counter() - started),
            },
            "outputs": {
                "labeled_vtp_files": [Path(path).name for path in vtkFiles],
                "region_ply_files": [Path(path).name for path in plyFiles],
            },
            "warnings": warnings,
            "skipped_meshes": skipped,
        }
        summaryPath = output / "MorphoWeaveSurfaceSegmentation_summary.json"
        summaryPath.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        for warning in warnings:
            logging.warning("[MorphoWeaveSurfaceSegmentation] %s", warning)
        return {
            "specimens": int(n),
            "loci": int(m),
            "k": int(result.k),
            "modularity": float(result.modularity),
            "stability": stability,
            "skipped": len(skipped),
            "summary": str(summaryPath),
            "output_directory": str(output),
        }

    @staticmethod
    def pairFiles(meshesDir, mrkDir):
        meshesDir, mrkDir = Path(meshesDir), Path(mrkDir)
        if not meshesDir.is_dir() or not mrkDir.is_dir():
            return []
        meshes, markups = {}, {}
        for path in meshesDir.iterdir():
            if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS:
                meshes.setdefault(path.stem.lower(), []).append(path)
        for path in mrkDir.iterdir():
            if path.is_file() and path.name.lower().endswith(".mrk.json"):
                markups.setdefault(markupsStem(path).lower(), []).append(path)
        duplicates = [key for key, values in meshes.items() if len(values) > 1]
        duplicates += [key for key, values in markups.items() if len(values) > 1]
        if duplicates:
            raise RuntimeError("Ambiguous duplicate specimen stems: " + ", ".join(sorted(duplicates)))
        return [
            (str(meshes[key][0]), str(markups[key][0]))
            for key in sorted(set(meshes) & set(markups))
        ]

    @staticmethod
    def _loadMarkups(path):
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
            markup = document["markups"][0]
            controlPoints = markup["controlPoints"]
        except Exception as error:
            raise RuntimeError(f"Could not read markups '{Path(path).name}': {error}") from error
        positions = []
        for index, point in enumerate(controlPoints):
            position = point.get("position")
            if str(point.get("positionStatus", "defined")).lower() == "undefined" or not isinstance(position, list) or len(position) != 3:
                raise RuntimeError(f"Control point {index} in '{Path(path).name}' is undefined.")
            positions.append(position)
        points = np.asarray(positions, dtype=np.float64)
        coordinateSystem = markup.get(
            "coordinateSystem", document.get("coordinateSystem", "RAS")
        )
        isLps = (
            isinstance(coordinateSystem, str) and coordinateSystem.upper() == "LPS"
        ) or (isinstance(coordinateSystem, (int, float)) and int(coordinateSystem) == 1)
        if isLps:
            points[:, :2] *= -1.0
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise RuntimeError(f"Invalid coordinates in '{Path(path).name}'.")
        return points

    def _loadStack(self, paths):
        arrays, expected = [], None
        for path in paths:
            points = self._loadMarkups(path)
            expected = len(points) if expected is None else expected
            if len(points) != expected:
                raise RuntimeError(
                    f"'{Path(path).name}' has {len(points)} loci; expected {expected}."
                )
            arrays.append(points)
        return np.ascontiguousarray(np.stack(arrays), dtype=np.float64)

    def _reference(self, explicitPath, mrkDir, X):
        candidates = []
        if explicitPath:
            candidates.append(Path(explicitPath))
        else:
            folder = Path(mrkDir)
            candidates.extend(
                [
                    folder / "atlas_dense_correspondences.mrk.json",
                    folder.parent / "atlas" / "atlas_dense_correspondences.mrk.json",
                ]
            )
        for path in candidates:
            if not path.is_file():
                continue
            points = self._loadMarkups(path)
            if points.shape == X.shape[1:]:
                return np.ascontiguousarray(points), path
            if explicitPath:
                raise RuntimeError(
                    f"Reference has shape {points.shape}; expected {X.shape[1:]}."
                )
            logging.warning("Ignoring incompatible automatic reference %s", path)
        return np.ascontiguousarray(X.mean(axis=0)), "specimen_mean"

    @staticmethod
    def _graphComponents(reference, neighbors):
        m = len(reference)
        _, indices = cKDTree(reference).query(reference, k=min(neighbors + 1, m))
        indices = np.atleast_2d(indices)[:, 1:]
        rows = np.repeat(np.arange(m), indices.shape[1])
        graph = coo_matrix(
            (np.ones(rows.size, dtype=np.uint8), (rows, indices.ravel())),
            shape=(m, m),
        ).tocsr()
        graph = graph.maximum(graph.T)
        graph.setdiag(0)
        graph.eliminate_zeros()
        return int(connected_components(graph, directed=False, return_labels=False))

    @staticmethod
    def _nodeIds(className):
        collection = slicer.mrmlScene.GetNodesByClass(className)
        collection.UnRegister(None)
        return {
            collection.GetItemAsObject(index).GetID()
            for index in range(collection.GetNumberOfItems())
        }

    def _loadMesh(self, path, needFaces=True):
        before = self._nodeIds("vtkMRMLModelNode")
        try:
            node = slicer.util.loadModel(path, properties={"coordinateSystem": "RAS"})
        except TypeError:
            node = slicer.util.loadModel(path)
        if node is True or node is False or node is None:
            collection = slicer.mrmlScene.GetNodesByClass("vtkMRMLModelNode")
            collection.UnRegister(None)
            node = next(
                (
                    collection.GetItemAsObject(i)
                    for i in range(collection.GetNumberOfItems())
                    if collection.GetItemAsObject(i).GetID() not in before
                ),
                None,
            )
        if node is None or node.GetPolyData() is None:
            raise RuntimeError(f"Failed to load model: {path}")
        try:
            polydata = vtk.vtkPolyData()
            polydata.DeepCopy(node.GetPolyData())
            transformNode = node.GetParentTransformNode()
            if transformNode:
                transform = vtk.vtkGeneralTransform()
                transformNode.GetTransformToWorld(transform)
                transformFilter = vtk.vtkTransformPolyDataFilter()
                transformFilter.SetInputData(polydata)
                transformFilter.SetTransform(transform)
                transformFilter.Update()
                polydata.DeepCopy(transformFilter.GetOutput())
            clean = vtk.vtkCleanPolyData()
            clean.SetInputData(polydata)
            clean.Update()
            triangles = vtk.vtkTriangleFilter()
            triangles.SetInputConnection(clean.GetOutputPort())
            triangles.PassLinesOff()
            triangles.PassVertsOff()
            triangles.Update()
            polydata.DeepCopy(triangles.GetOutput())
        finally:
            slicer.mrmlScene.RemoveNode(node)
        if polydata.GetNumberOfPoints() == 0 or polydata.GetNumberOfPolys() == 0:
            raise RuntimeError(f"'{Path(path).name}' has no triangular surface.")
        vertices = vtk_np.vtk_to_numpy(polydata.GetPoints().GetData()).astype(
            np.float64, copy=False
        )
        if not np.isfinite(vertices).all():
            raise RuntimeError(f"'{Path(path).name}' contains non-finite vertices.")
        faces = self._faces(polydata) if needFaces else None
        return vertices, faces, polydata

    @staticmethod
    def _faces(polydata):
        cells = polydata.GetPolys()
        cells.InitTraversal()
        ids, faces = vtk.vtkIdList(), []
        while cells.GetNextCell(ids):
            if ids.GetNumberOfIds() == 3:
                faces.append([ids.GetId(0), ids.GetId(1), ids.GetId(2)])
        if not faces:
            raise RuntimeError("Mesh contains no triangles.")
        return np.asarray(faces, dtype=np.int32)

    @staticmethod
    def _project(vertices, loci, labels):
        tree = cKDTree(loci)
        try:
            nearest = tree.query(vertices, k=1, workers=-1)[1]
        except TypeError:
            nearest = tree.query(vertices, k=1)[1]
        return np.asarray(labels[nearest], dtype=np.int32)

    @staticmethod
    def _adjacency(faces, vertexCount):
        rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
        graph = coo_matrix(
            (np.ones(rows.size, dtype=np.uint8), (rows, cols)),
            shape=(vertexCount, vertexCount),
        ).tocsr()
        graph = graph.maximum(graph.T)
        graph.setdiag(0)
        graph.eliminate_zeros()
        return graph

    def _smooth(self, vertices, faces, labels, iterations):
        if iterations <= 0:
            return labels
        graph = self._adjacency(faces, len(vertices))
        degree = np.asarray(graph.sum(axis=1)).ravel()
        for _ in range(iterations):
            coo = graph.tocoo()
            votes = np.zeros((len(vertices), int(labels.max()) + 1), dtype=np.int32)
            np.add.at(votes, (coo.row, labels[coo.col]), 1)
            labels = np.where(degree > 0, votes.argmax(axis=1), labels).astype(np.int32)
        return labels

    @staticmethod
    def _vtkArray(values, name):
        array = vtk_np.numpy_to_vtk(
            np.asarray(values, dtype=np.int32), deep=True, array_type=vtk.VTK_INT
        )
        array.SetName(name)
        return array

    def _writeVtp(self, output, stem, polydata, rawLabels, labels):
        result = vtk.vtkPolyData()
        result.DeepCopy(polydata)
        result.GetPointData().AddArray(self._vtkArray(rawLabels, "SegID"))
        result.GetPointData().AddArray(self._vtkArray(labels, "SegID_smooth"))
        result.GetPointData().SetActiveScalars("SegID_smooth")
        path = output / f"{stem}_segmented.vtp"
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(path))
        writer.SetInputData(result)
        writer.SetDataModeToBinary()
        if writer.Write() != 1:
            raise RuntimeError(f"Could not write {path}")
        return str(path)

    @staticmethod
    def _cellLabels(faces, labels):
        a, b, c = labels[faces[:, 0]], labels[faces[:, 1]], labels[faces[:, 2]]
        return np.where(
            a == b,
            a,
            np.where(a == c, a, np.where(b == c, b, np.minimum(np.minimum(a, b), c))),
        ).astype(np.int32)

    def _writePlys(self, output, stem, polydata, faces, labels):
        result = vtk.vtkPolyData()
        result.DeepCopy(polydata)
        cellLabels = self._cellLabels(faces, labels)
        result.GetCellData().AddArray(self._vtkArray(cellLabels, "SegID"))
        written = []
        for region in sorted(int(value) for value in np.unique(cellLabels)):
            threshold = vtk.vtkThreshold()
            threshold.SetInputData(result)
            threshold.SetInputArrayToProcess(
                0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "SegID"
            )
            if hasattr(threshold, "SetLowerThreshold"):
                threshold.SetLowerThreshold(region)
                threshold.SetUpperThreshold(region)
            else:
                threshold.ThresholdBetween(region, region)
            geometry = vtk.vtkGeometryFilter()
            geometry.SetInputConnection(threshold.GetOutputPort())
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputConnection(geometry.GetOutputPort())
            normals.SplittingOff()
            normals.ConsistencyOn()
            path = output / f"{stem}_seg_{region:02d}.ply"
            writer = vtk.vtkPLYWriter()
            writer.SetFileName(str(path))
            writer.SetInputConnection(normals.GetOutputPort())
            writer.SetFileTypeToBinary()
            if writer.Write() != 1:
                raise RuntimeError(f"Could not write {path}")
            written.append(str(path))
        return written

    def _preview(self, stem, polydata, labels):
        result = vtk.vtkPolyData()
        result.DeepCopy(polydata)
        result.GetPointData().AddArray(self._vtkArray(labels, "SegID_smooth"))
        result.GetPointData().SetActiveScalars("SegID_smooth")
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        node.SetName(f"{stem}_seg_preview")
        node.SetAndObservePolyData(result)
        node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        colors = self._colorTable(int(labels.max()) + 1)
        display.SetAndObserveColorNodeID(colors.GetID())
        display.SetScalarRange(0.0, float(max(0, labels.max())))
        display.SetAutoScalarRange(False)
        display.SetScalarVisibility(True)

    @staticmethod
    def _clearPreviews():
        collection = slicer.mrmlScene.GetNodesByClass("vtkMRMLModelNode")
        collection.UnRegister(None)
        nodes = [
            collection.GetItemAsObject(index)
            for index in range(collection.GetNumberOfItems())
        ]
        for node in nodes:
            if node and (node.GetName() or "").endswith("_seg_preview"):
                slicer.mrmlScene.RemoveNode(node)

    def _colorTable(self, count):
        name = "MorphoWeaveSurfaceSegmentation_Colors"
        table = slicer.mrmlScene.GetFirstNodeByName(name)
        if table is None:
            table = slicer.vtkMRMLColorTableNode()
            table.SetTypeToUser()
            table.SetName(name)
            slicer.mrmlScene.AddNode(table)
        table.SetNumberOfColors(max(1, count))
        for index in range(max(1, count)):
            hue = (index * 0.61803398875) % 1.0
            red, green, blue = self._hsv(hue, 0.65, 0.95)
            table.SetColor(index, red, green, blue, 1.0)
            try:
                table.SetColorName(index, f"Region {index}")
            except AttributeError:
                pass
        return table

    @staticmethod
    def _hsv(h, s, v):
        i = int(h * 6.0) % 6
        f = h * 6.0 - int(h * 6.0)
        p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
        return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]

    @staticmethod
    def _writeLocusOutputs(output, labels, embedding, eigenvalues):
        np.save(output / "MorphoWeaveSurfaceSegmentation_locus_labels.npy", labels)
        np.save(output / "MorphoWeaveSurfaceSegmentation_embedding.npy", embedding)
        with (output / "MorphoWeaveSurfaceSegmentation_locus_labels.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["locus_index", "segment_id"])
            writer.writerows(enumerate(int(value) for value in labels))
        with (output / "MorphoWeaveSurfaceSegmentation_eigenvalues.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["mode_index", "eigenvalue"])
            writer.writerows(enumerate(float(value) for value in eigenvalues))


class MorphoWeaveSurfaceSegmentationTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.assertEqual(markupsStem("specimen.mrk.json"), "specimen")
        self.assertEqual(stableMajorMinor("0.2.0"), (0, 2))
        self.assertIsNone(stableMajorMinor("0.3.0rc1"))
        self.assertIsNotNone(MorphoWeaveSurfaceSegmentationLogic())
