import os
import time
import html
import tempfile
from datetime import datetime
from pathlib import Path
import numpy as np
import vtk, qt, ctk, slicer
import vtk.util.numpy_support as vtk_np
from slicer.ScriptedLoadableModule import *
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags, eye as speye
from scipy.sparse.linalg import splu


def voxel_representative_indices(points, voxel_size, origin):
  points = np.asarray(points, dtype=np.float64)
  if len(points) == 0:
    return np.empty((0,), dtype=np.int64)
  if not np.isfinite(voxel_size) or voxel_size <= 0:
    raise ValueError("voxel_size must be positive and finite")
  keys = np.floor((points - np.asarray(origin, dtype=np.float64)) / voxel_size).astype(np.int64)
  original = np.arange(len(points), dtype=np.int64)
  order = np.lexsort((original, keys[:, 2], keys[:, 1], keys[:, 0]))
  sorted_keys = keys[order]
  first = np.ones(len(order), dtype=bool)
  first[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
  return np.sort(order[first]).astype(np.int64, copy=False)


def farthest_point_subset_indices(points, candidate_indices, target_count):
  points = np.asarray(points, dtype=np.float64)
  candidates = np.unique(np.asarray(candidate_indices, dtype=np.int64))
  target_count = int(target_count)
  if target_count <= 0:
    raise ValueError("target_count must be positive")
  if len(candidates) <= target_count:
    return np.sort(candidates)

  candidate_points = points[candidates]
  centroid = candidate_points.mean(axis=0)
  first = int(np.argmax(np.sum((candidate_points - centroid) ** 2, axis=1)))
  selected = np.zeros(len(candidates), dtype=bool)
  selected[first] = True
  chosen = [first]
  nearest = np.sum((candidate_points - candidate_points[first]) ** 2, axis=1)

  while len(chosen) < target_count:
    nearest[selected] = -1.0
    next_index = int(np.argmax(nearest))
    selected[next_index] = True
    chosen.append(next_index)
    distance = np.sum((candidate_points - candidate_points[next_index]) ** 2, axis=1)
    nearest = np.minimum(nearest, distance)

  return np.sort(candidates[np.asarray(chosen, dtype=np.int64)])


def sample_indices_voxel_target_points(points, target_count=2000, tolerance=0.02, max_iterations=40):
  points = np.asarray(points, dtype=np.float64)
  if points.ndim != 2 or points.shape[1] != 3:
    raise ValueError("points must have shape (N, 3)")
  if not np.all(np.isfinite(points)):
    raise ValueError("points must contain only finite coordinates")
  target_count = int(target_count)
  if target_count <= 0:
    raise ValueError("target_count must be positive")
  if len(points) <= target_count:
    return np.arange(len(points), dtype=np.int64)

  bounds_min = points.min(axis=0)
  diagonal = float(np.linalg.norm(points.max(axis=0) - bounds_min))
  if diagonal <= np.finfo(np.float64).eps:
    return np.arange(target_count, dtype=np.int64)

  low = diagonal * 1e-9
  high = diagonal
  best_indices = np.arange(len(points), dtype=np.int64)
  best_overshoot = len(points) - target_count
  tolerance_count = max(1, int(round(target_count * float(tolerance))))

  for _ in range(max(1, int(max_iterations))):
    voxel_size = float(np.sqrt(low * high))
    indices = voxel_representative_indices(points, voxel_size, bounds_min)
    count = len(indices)

    if target_count <= count and count - target_count < best_overshoot:
      best_indices = indices
      best_overshoot = count - target_count
      if best_overshoot <= tolerance_count:
        break

    if count > target_count:
      low = voxel_size
    elif count < target_count:
      high = voxel_size
    else:
      return indices

  return farthest_point_subset_indices(points, best_indices, target_count)


def setWorkflowStatus(label, state, detail):
  title, color = {
    "needs_input": ("Needs input", "#c75b5b"),
    "ready": ("Ready", "#4f9d69"),
    "optional": ("Optional", "#c18b37"),
    "complete": ("Complete", "#4f9d69"),
  }[state]
  safeDetail = html.escape(str(detail)).replace("\n", "<br>")
  label.setTextFormat(qt.Qt.RichText)
  label.setText(f'<span style="color:{color}; font-weight:600">{title}</span> — {safeDetail}')
  label.setAccessibleName(f"{title}: {detail}")


# =========================
# Module + Widget
# =========================

class MorphoWeaveAtlasBuilder(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Atlas Builder"
    self.parent.categories = ["MorphoWeave"]
    self.parent.dependencies = ["MorphoWeaveModelLibrary"]
    self.parent.contributors = ["Arthur Porto"]
    self.parent.helpText = "Align meshes/landmarks to an atlas and export dense correspondences as .mrk.json"
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = ""

class MorphoWeaveAtlasBuilderWidget(ScriptedLoadableModuleWidget):
  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    self.logic = MorphoWeaveAtlasBuilderLogic()

    root = ctk.ctkCollapsibleButton(); root.text = "Atlas Builder Workflow"; root.collapsed = False
    lay = qt.QFormLayout(root); self.layout.addWidget(root)
    flowHelp = qt.QLabel("Step order: choose atlas source -> select input/output folders -> tune advanced options -> run.")
    flowHelp.setWordWrap(True)
    lay.addRow(flowHelp)

    requiredBox = ctk.ctkCollapsibleButton(); requiredBox.text = "Required Inputs"; requiredBox.collapsed = False
    reqLay = qt.QFormLayout(requiredBox); lay.addRow(requiredBox)

    # Atlas source
    self.createAtlasRadio = qt.QRadioButton("Create atlas from inputs")
    self.loadAtlasRadio   = qt.QRadioButton("Load existing atlas")
    self.createAtlasRadio.setChecked(True)
    g = qt.QButtonGroup(root); g.addButton(self.createAtlasRadio); g.addButton(self.loadAtlasRadio)
    reqLay.addRow(self.createAtlasRadio); reqLay.addRow(self.loadAtlasRadio)

    # Load-atlas widgets
    self.atlasBox = ctk.ctkCollapsibleButton(); self.atlasBox.text = "Atlas files (only when loading)"; self.atlasBox.collapsed = True; self.atlasBox.enabled = False
    aLay = qt.QFormLayout(self.atlasBox); reqLay.addRow(self.atlasBox)
    self.atlasModelPath = ctk.ctkPathLineEdit(); self.atlasModelPath.filters = ctk.ctkPathLineEdit().Files; self.atlasModelPath.nameFilters = ["Model (*.ply *.stl *.obj *.vtk *.vtp)"]
    self.atlasLMPath    = ctk.ctkPathLineEdit(); self.atlasLMPath.filters    = ctk.ctkPathLineEdit().Files; self.atlasLMPath.nameFilters    = ["Point set (*.fcsv *.mrk.json *.json)"]
    aLay.addRow("Atlas model:", self.atlasModelPath); aLay.addRow("Atlas landmarks:", self.atlasLMPath)

    # IO
    self.modelDir = ctk.ctkPathLineEdit(); self.modelDir.filters = ctk.ctkPathLineEdit.Dirs
    self.lmDir    = ctk.ctkPathLineEdit(); self.lmDir.filters    = ctk.ctkPathLineEdit.Dirs
    self.outDir   = ctk.ctkPathLineEdit(); self.outDir.filters   = ctk.ctkPathLineEdit.Dirs
    reqLay.addRow("Model directory:", self.modelDir)
    reqLay.addRow("Landmark directory:", self.lmDir)
    reqLay.addRow("Output directory:", self.outDir)

    self.librarySaveBox = ctk.ctkCollapsibleButton()
    self.librarySaveBox.text = "Optional Model Library Save"
    self.librarySaveBox.collapsed = True
    libraryLay = qt.QFormLayout(self.librarySaveBox)
    reqLay.addRow(self.librarySaveBox)
    self.saveToLibraryCheck = qt.QCheckBox("Save SSM to Model Library after a successful run")
    libraryLay.addRow(self.saveToLibraryCheck)
    self.libraryNameEdit = qt.QLineEdit()
    self.libraryNameEdit.enabled = False
    self.libraryNameEdit.setPlaceholderText("Enter a unique model name")
    libraryLay.addRow("Model name:", self.libraryNameEdit)
    self.libraryPathLabel = qt.QLabel("")
    self.libraryPathLabel.setWordWrap(True)
    self.libraryPathLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
    libraryLay.addRow("Library location:", self.libraryPathLabel)

    self.prereqLabel = qt.QLabel("")
    self.prereqLabel.setWordWrap(True)
    reqLay.addRow(self.prereqLabel)

    advancedBox = ctk.ctkCollapsibleButton(); advancedBox.text = "Advanced Options"; advancedBox.collapsed = True
    advLay = qt.QFormLayout(advancedBox); lay.addRow(advancedBox)

    # Alignment choice
    self.useSimilarity = qt.QCheckBox("Normalize scale"); self.useSimilarity.setChecked(True)
    self.useSimilarity.setToolTip("Checked (recommended): allow isotropic scaling so size differences are removed. Unchecked: rigid only.")
    advLay.addRow(self.useSimilarity)

    # Override LM on surface check
    self.overrideCoordCheck = qt.QCheckBox("Override landmark↔mesh coordinate check"); self.overrideCoordCheck.setChecked(False)
    advLay.addRow(self.overrideCoordCheck)

    self.keepAlignedOutputs = qt.QCheckBox("Keep aligned models and landmarks")
    self.keepAlignedOutputs.setChecked(True)
    self.keepAlignedOutputs.setToolTip(
      "Keep transformed copies in alignedModels/ and alignedLMs/. "
      "Original input files are never copied or modified."
    )
    advLay.addRow(self.keepAlignedOutputs)

    # Warp method (NEW)
    self.warpMethod = qt.QComboBox()
    self.warpMethod.addItems(["TPS (recommended)", "Biharmonic (experimental)"])
    self.warpMethod.setCurrentIndex(0)
    self.autoFallback = qt.QCheckBox("Auto-fallback to TPS if biharmonic fails"); self.autoFallback.setChecked(True)
    hwarp = qt.QHBoxLayout(); hwarp.addWidget(self.warpMethod); hwarp.addWidget(self.autoFallback)
    advLay.addRow("Warp method:", hwarp)

    # Target-count sampling + preview
    self.targetPointCount = qt.QSpinBox(); self.targetPointCount.minimum = 100; self.targetPointCount.maximum = 1_000_000; self.targetPointCount.singleStep = 100; self.targetPointCount.value = 2000
    self.targetPointCount.setToolTip("Requested number of existing atlas vertices to export. If the atlas has fewer vertices, all are used.")
    advLay.addRow("Target dense points:", self.targetPointCount)

    self.previewBtn = qt.QPushButton("Preview Point Count"); self.previewBtn.enabled = False
    self.previewLbl = qt.QLabel("")
    h = qt.QHBoxLayout(); h.addWidget(self.previewBtn); h.addWidget(self.previewLbl, 1)
    advLay.addRow("Selected points:", h)
    self._previewTimer = qt.QTimer(); self._previewTimer.setSingleShot(True); self._previewTimer.setInterval(250)
    self._previewTimer.timeout.connect(self._onPreview)
    self.targetPointCount.valueChanged.connect(lambda v: self._previewTimer.start())

    runBox = ctk.ctkCollapsibleButton(); runBox.text = "Run + Status"; runBox.collapsed = False
    runLay = qt.QFormLayout(runBox); lay.addRow(runBox)
    self.runBtn = qt.QPushButton("Run Atlas Builder Pipeline"); self.runBtn.enabled = False
    runLay.addRow(self.runBtn)

    # Log
    self.log = qt.QPlainTextEdit(); self.log.setReadOnly(True); runLay.addRow(self.log)

    # Signals
    self.createAtlasRadio.toggled.connect(self._toggleAtlasLoad)
    self.loadAtlasRadio.toggled.connect(self._toggleAtlasLoad)
    self.previewBtn.clicked.connect(self._onPreview)
    for w in [self.atlasModelPath, self.atlasLMPath, self.modelDir, self.lmDir, self.outDir]:
      w.connect('validInputChanged(bool)', self._reeval)
    self.saveToLibraryCheck.toggled.connect(self._onLibrarySaveToggled)
    self.libraryNameEdit.textChanged.connect(self._reeval)
    self.runBtn.clicked.connect(self._onRun)

    self._previewPD = None
    self._refreshLibraryPath()
    self._reeval()

  # ---------- UI helpers ----------
  def _status(self, msg):
    self.log.appendPlainText(msg)
    try: slicer.util.showStatusMessage(msg, 2500)
    except Exception: pass
    now = time.monotonic()
    last = getattr(self, "_lastStatusProcessTs", 0.0)
    if ("completed" in msg.lower()) or ("failed" in msg.lower()) or (now - last >= 0.10):
      qt.QApplication.processEvents()
      self._lastStatusProcessTs = now

  def enter(self):
    self._refreshLibraryPath()
    self._reeval()

  def _configuredLibraryPath(self):
    settings = qt.QSettings()
    newKey = "MorphoWeaveModelLibrary/databasePath"
    legacyKey = "DATABASE/databasePath"
    defaultPath = os.path.join(Path.home(), "Documents", "MorphoWeaveModels")
    if settings.contains(newKey):
      return str(settings.value(newKey, defaultPath))
    if settings.contains(legacyKey):
      path = str(settings.value(legacyKey, defaultPath))
      settings.setValue(newKey, path)
      return path
    return defaultPath

  def _refreshLibraryPath(self):
    self.libraryRootPath = self._configuredLibraryPath()
    if hasattr(self, "libraryPathLabel"):
      self.libraryPathLabel.setText(self.libraryRootPath)

  def _libraryName(self):
    text = getattr(self.libraryNameEdit, "text", "")
    return str(text() if callable(text) else text).strip()

  def _onLibrarySaveToggled(self, enabled):
    self.libraryNameEdit.enabled = bool(enabled)
    self._refreshLibraryPath()
    self._reeval()

  def _libraryTargetPath(self):
    return os.path.join(self.libraryRootPath, self._libraryName())

  def _toggleAtlasLoad(self):
    useLoad = self.loadAtlasRadio.isChecked()
    self.atlasBox.collapsed = not useLoad
    self.atlasBox.enabled = useLoad
    self._invalidatePreviewCache()
    self._reeval()

  def _reeval(self, *args):
    atlasOK = self.createAtlasRadio.isChecked() or bool(self.atlasModelPath.currentPath and self.atlasLMPath.currentPath)
    ioOK    = bool(self.modelDir.currentPath and self.lmDir.currentPath and self.outDir.currentPath)
    saveToLibrary = self.saveToLibraryCheck.isChecked()
    libraryNameOK = (not saveToLibrary) or self.logic.isSafeLibraryName(self._libraryName())
    self.runBtn.enabled = bool(atlasOK and ioOK and libraryNameOK)
    hasRef = (self.loadAtlasRadio.isChecked() and bool(self.atlasModelPath.currentPath)) or bool(self.modelDir.currentPath)
    self.previewBtn.enabled = hasRef
    if not atlasOK and not ioOK:
      setWorkflowStatus(self.prereqLabel, "needs_input", "Choose atlas source/files and input/output directories.")
    elif not atlasOK:
      setWorkflowStatus(self.prereqLabel, "needs_input", "Choose atlas creation mode or load atlas model and landmarks.")
    elif not ioOK:
      setWorkflowStatus(self.prereqLabel, "needs_input", "Select model, landmark, and output directories.")
    elif not libraryNameOK:
      setWorkflowStatus(self.prereqLabel, "needs_input", "Enter a model name without path separators or reserved filename characters.")
    else:
      detail = "Required inputs are complete."
      if saveToLibrary:
        detail += f" The SSM will be saved as '{self._libraryName()}'."
      setWorkflowStatus(self.prereqLabel, "ready", detail)
    self._invalidatePreviewCache()

  def _invalidatePreviewCache(self, *_):
    self._previewPD = None
    if hasattr(self, "previewLbl"): self.previewLbl.setText("")

  def _getPreviewPolyData(self):
    if getattr(self, "_previewPD", None) is not None: return self._previewPD
    try:
      if self.loadAtlasRadio.isChecked() and self.atlasModelPath.currentPath:
        n = slicer.util.loadModel(self.atlasModelPath.currentPath)
        pd = vtk.vtkPolyData(); pd.DeepCopy(n.GetPolyData()); slicer.mrmlScene.RemoveNode(n)
        pd = self.logic.cleanPolyData(pd, tri=True, merge_tol=0.0, smooth_iter=0, normals=True)
        self._previewPD = pd; return pd
      d = self.modelDir.currentPath
      if d:
        for f in sorted(os.listdir(d)):
          if f.lower().endswith(('.ply', '.stl', '.vtp', '.vtk', '.obj')):
            n = slicer.util.loadModel(os.path.join(d, f))
            pd = vtk.vtkPolyData(); pd.DeepCopy(n.GetPolyData()); slicer.mrmlScene.RemoveNode(n)
            pd = self.logic.cleanPolyData(pd, tri=True, merge_tol=0.0, smooth_iter=0, normals=True)
            self._previewPD = pd; return pd
    except Exception:
      pass
    self._previewPD = None; return None

  def _onPreview(self):
    pd = self._getPreviewPolyData()
    if pd: self.logic.maybeWarnDense(pd, label="preview")
    if not pd: self.previewLbl.setText("n/a"); return
    target = int(self.targetPointCount.value)
    n, tot = self.logic.previewCountForTarget(pd, target)
    suffix = " (all available)" if tot < target else ""
    self.previewLbl.setText(f"{n} of {tot} (~{(100.0*n/max(1,tot)):.1f}%){suffix}")

  def _loadAtlas(self):
    try:
      m = slicer.util.loadModel(self.atlasModelPath.currentPath)
      l = slicer.util.loadMarkups(self.atlasLMPath.currentPath)
      return m, l
    except Exception:
      self._status("Failed to load atlas model/landmarks."); return None, None

  def _outFolders(self, base, alignedRoot=None):
    ts = datetime.now().strftime('%Y_%m-%d_%H_%M_%S')
    root = os.path.join(base, ts); os.makedirs(root, exist_ok=True)
    d = {'output': root}
    for k in ['atlas', 'population_correspondences']:
      p = os.path.join(root, k); os.makedirs(p, exist_ok=True); d[k] = p
    alignedBase = alignedRoot or root
    for k in ['alignedModels', 'alignedLMs']:
      p = os.path.join(alignedBase, k); os.makedirs(p, exist_ok=True); d[k] = p
    return d

  def _onRun(self):
    logic = self.logic
    saveToLibrary = self.saveToLibraryCheck.isChecked()
    if saveToLibrary:
      self._refreshLibraryPath()
      libraryTarget = self._libraryTargetPath()
      if os.path.exists(libraryTarget):
        name = self._libraryName()
        if not slicer.util.confirmOkCancelDisplay(
          f"A Model Library entry named '{name}' already exists.\n\n"
          "Continuing will overwrite files with matching names. Continue?"
        ):
          self._status("Run cancelled; the existing Model Library entry was not changed.")
          return

    keepAlignedOutputs = self.keepAlignedOutputs.isChecked()
    alignedWorkspace = None
    try:
      if not keepAlignedOutputs:
        alignedWorkspace = tempfile.TemporaryDirectory(prefix="MorphoWeave-aligned-")

      F = self._outFolders(
        self.outDir.currentPath,
        alignedRoot=alignedWorkspace.name if alignedWorkspace else None,
      )
      F['originalModels'] = self.modelDir.currentPath
      F['originalLMs']    = self.lmDir.currentPath
      useSimilarity = self.useSimilarity.isChecked()
      method = "biharmonic" if int(self.warpMethod.currentIndex) == 1 else "tps"
      allowFallback = self.autoFallback.isChecked()

      # 1) Get atlas (load or build)
      if self.loadAtlasRadio.isChecked():
        self._status("Loading atlas model and landmarks…")
        atlasModel, atlasLMs = self._loadAtlas()
        if not atlasModel or not atlasLMs:
          return
      else:
        try:
          self._status("Generating atlas: finding closest-to-mean, pre-aligning, and averaging…")
          atlasModel, atlasLMs = self._generateAtlas(F, useSimilarity, method, allowFallback)
        except Exception as e:
          self._status(f"Atlas generation failed: {e}"); return

      # 2) Align all to atlas
      try:
        self._status("Aligning all specimens to the atlas…")
        logic.runAlign(
          atlasModel, atlasLMs,
          F['originalModels'], F['originalLMs'],
          F['alignedModels'],  F['alignedLMs'],
          useSimilarity,
          allowCoordMismatch=self.overrideCoordCheck.isChecked(),
          progress=lambda i,n,s: self._status(f"[{i}/{n}] aligned {s}")
        )
      except ValueError as e:
        self._status(str(e)); return

      # 3) Save atlas + export dense
      logic.saveAtlasOnly(atlasModel, atlasLMs, F['atlas'])
      dense_ok = True
      try:
        self._status(f"Exporting dense correspondences using {method.upper()} projection…")
        logic.exportDenseLMs(
          atlasModel, atlasLMs,
          F['alignedModels'], F['alignedLMs'],
          F['population_correspondences'], F['atlas'],
          int(self.targetPointCount.value),
          warp_method=method,
          allow_fallback=allowFallback,
          progress=lambda msg: self._status(msg)
        )
      except Exception as e:
        dense_ok = False
        self._status(f"Dense export failed: {e}")
      slicer.mrmlScene.RemoveNode(atlasModel); slicer.mrmlScene.RemoveNode(atlasLMs)
      library_ok = None
      if dense_ok and saveToLibrary:
        library_ok = self._saveSsmToLibrary(F)
      if dense_ok:
        retainedFolders = ["atlas/", "population_correspondences/"]
        if keepAlignedOutputs:
          retainedFolders.extend(["alignedModels/", "alignedLMs/"])
        library_note = ""
        if library_ok is True:
          library_note = f" SSM saved to Model Library as '{self._libraryName()}'."
        elif library_ok is False:
          library_note = " Atlas outputs are complete, but the Model Library save failed."
        self._status(f"Completed. Output: {F['output']} ({', '.join(retainedFolders)}).{library_note}")
      else:
        retainedFolders = ["atlas/"]
        if keepAlignedOutputs:
          retainedFolders.extend(["alignedModels/", "alignedLMs/"])
        self._status(
          f"Completed with warnings. Output: {F['output']} ({', '.join(retainedFolders)}). "
          "Dense export did not finish."
        )
    finally:
      if alignedWorkspace is not None:
        alignedWorkspace.cleanup()

  def _saveSsmToLibrary(self, F):
    try:
      from MorphoWeaveModelLibrary import MorphoWeaveModelLibraryLogic

      os.makedirs(self.libraryRootPath, exist_ok=True)
      target = self._libraryTargetPath()

      def progress(value, text):
        self._status(f"Model Library [{int(value)}%]: {text}")

      success, message = MorphoWeaveModelLibraryLogic().ingestSSMDatabase(
        dbPath=target,
        templateModelPath=os.path.join(F["atlas"], "atlas_model.ply"),
        templateLandmarksPath=os.path.join(F["atlas"], "atlas_dense_correspondences.mrk.json"),
        sparseLandmarksPath=os.path.join(F["atlas"], "atlas_sparse_landmarks.mrk.json"),
        populationDir=F["population_correspondences"],
        progress_callback=progress,
      )
      self._status(message)
      return bool(success)
    except Exception as error:
      self._status(f"Model Library save failed: {error}")
      return False

  def _generateAtlas(self, F, useSimilarity, warp_method, allowFallback):
    logic = self.logic
    closest_key = logic.getClosestToMeanPath(F['originalLMs'])
    model_index = logic._list_keys(F['originalModels'], ('.ply', '.stl', '.vtp', '.vtk', '.obj'))
    lm_index = logic._list_keys(F['originalLMs'], ('.mrk.json', '.fcsv', '.json'))
    lm_path = lm_index.get(str(closest_key).lower())
    if not lm_path: raise ValueError(f"Could not resolve LM file for key '{closest_key}'")
    baseLM = slicer.util.loadMarkups(lm_path)
    model_path = model_index.get(str(closest_key).lower())
    baseModel = slicer.util.loadModel(model_path) if model_path else None
    if not baseModel: raise ValueError(f"Could not resolve model for key '{closest_key}'")
    self._status(f"Closest sample to mean: {closest_key}")

    self._status("Pre-aligning all specimens to base (reduces warp bending)…")
    logic.runAlign(baseModel, baseLM, F['originalModels'], F['originalLMs'], F['alignedModels'], F['alignedLMs'], useSimilarity, allowCoordMismatch=self.overrideCoordCheck.isChecked())


    self._status(f"Building mean atlas in mean-shape space (template via {warp_method.upper()})…")
    atlasModel, atlasLMs = logic.runMean(F['alignedLMs'], F['alignedModels'], warp_method=warp_method, allow_fallback=allowFallback)

    slicer.mrmlScene.RemoveNode(baseModel); slicer.mrmlScene.RemoveNode(baseLM)
    return atlasModel, atlasLMs

# =========================
# Logic
# =========================

class MorphoWeaveAtlasBuilderLogic(ScriptedLoadableModuleLogic):

  DENSE_MAX_POINTS = 300_000
  DENSE_MAX_TRIS   = 500_000
  DENSE_EDGE_FRAC  = 1e-3  # mean edge length < 0.1% of bbox diag

  @staticmethod
  def isSafeLibraryName(name):
    name = str(name or "").strip()
    if not name or name in {".", ".."} or name.endswith((".", " ")):
      return False
    if any(ord(char) < 32 or char in '<>:"/\\|?*' for char in name):
      return False
    stem = name.split(".", 1)[0].upper()
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    return stem not in reserved

  def __init__(self):
    super().__init__()
    self._denseWarned = False
  # -------- Utilities: pairing, IO, conversions --------
  def numpyToFiducialNode(self, arr, name):
    n = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', name)
    slicer.util.updateMarkupsControlPointsFromArray(n, arr)
    return n

  def fiducialNodeToPolyData(self, nodeOrPath, load=True):
    n = slicer.util.loadMarkups(nodeOrPath) if isinstance(nodeOrPath, str) else nodeOrPath
    arr = slicer.util.arrayFromMarkupsControlPoints(n).astype(np.float64, copy=False)
    pts = vtk.vtkPoints(); pts.SetData(vtk_np.numpy_to_vtk(arr, deep=True))
    pd = vtk.vtkPolyData(); pd.SetPoints(pts)
    if load and isinstance(nodeOrPath, str): slicer.mrmlScene.RemoveNode(n)
    return pd

  def importLandmarks(self, topDir):
    g = vtk.vtkMultiBlockDataGroupFilter(); names=[]
    for f in sorted(os.listdir(topDir)):
      fl = f.lower()
      if fl.endswith((".fcsv",".json",".mrk.json")):
        stem = Path(f)
        while stem.suffix in {'.fcsv','.mrk','.json'}: stem = stem.with_suffix('')
        names.append(stem.name)
        poly = self.fiducialNodeToPolyData(os.path.join(topDir,f), load=True)
        g.AddInputData(poly)
    g.Update()
    return names, g.GetOutput()

  def _median_abs_signed_distance(self, ipd, pts_np):
    if ipd is None or pts_np.size==0:
      return float('inf')
    return float(np.median([abs(ipd.EvaluateFunction(float(p[0]), float(p[1]), float(p[2]))) for p in pts_np]))

  def _coord_check_ratio(self, pd, pts_np, threshold=None):
    diag = max(self._bbox_diag(pd), 1e-9)
    base = self.cleanPolyData(pd, tri=True, normals=False)
    if base.GetNumberOfPoints()==0:
      return float('inf'), float('inf')
    ipd = vtk.vtkImplicitPolyDataDistance()
    ipd.SetInput(base)
    r0 = self._median_abs_signed_distance(ipd, pts_np) / diag
    if threshold is not None and r0 <= threshold:
      return r0, float('inf')
    pts_flip = pts_np.copy(); pts_flip[:,0]*=-1.0; pts_flip[:,1]*=-1.0
    rflip = self._median_abs_signed_distance(ipd, pts_flip) / diag
    return r0, rflip

  def _key_no_ext(self, f):
    s = Path(f)
    while s.suffix.lower() in ('.json','.mrk','.fcsv'): s = s.with_suffix('')
    if s.suffix.lower() in ('.ply','.stl','.vtp','.vtk','.obj'): s = s.with_suffix('')
    return s.name

  def _resolve_by_key(self, directory, key, exts):
    kl = str(key).lower()
    for f in os.listdir(directory):
      if self._key_no_ext(f).lower() == kl:
        fl = f.lower()
        for ext in exts:
          if fl.endswith(ext): return os.path.join(directory, f)
        return os.path.join(directory, f)
    return None

  def _list_keys(self, d, ok_exts):
    out={}
    for f in os.listdir(d):
      fl=f.lower()
      if fl.endswith(tuple(ok_exts)): out[self._key_no_ext(f).lower()] = os.path.join(d,f)
    return out

  def _load_clean_model_polydata(self, modelPath, *, normals=False):
    mnode = slicer.util.loadModel(modelPath)
    mpd = vtk.vtkPolyData(); mpd.DeepCopy(mnode.GetPolyData()); slicer.mrmlScene.RemoveNode(mnode)
    return self.cleanPolyData(mpd, tri=True, merge_tol=0.0, smooth_iter=0, normals=normals)

  def importPaired(self, modelsDir, lmsDir, *, normals=False):
    md = self._list_keys(modelsDir, ('.ply','.stl','.vtp','.vtk','.obj'))
    ld = self._list_keys(lmsDir,    ('.mrk.json','.fcsv','.json'))
    keys = sorted(set(md) & set(ld))
    gm = vtk.vtkMultiBlockDataGroupFilter()
    gl = vtk.vtkMultiBlockDataGroupFilter()
    names=[]
    for k in keys:
      mpd = self._load_clean_model_polydata(md[k], normals=normals)
      gm.AddInputData(mpd)
      lnode = slicer.util.loadMarkups(ld[k]); gl.AddInputData(self.fiducialNodeToPolyData(lnode, load=False)); slicer.mrmlScene.RemoveNode(lnode)
      names.append(k)
    gm.Update(); gl.Update()
    return names, gm.GetOutput(), names, gl.GetOutput()

  def _mesh_density_score(self, pd):
    if pd is None or pd.GetNumberOfPoints()==0: return False, 0, 0, float('inf')
    tri = vtk.vtkTriangleFilter(); tri.SetInputData(pd); tri.PassLinesOff(); tri.PassVertsOff(); tri.Update()
    tpd = tri.GetOutput()
    npts = tpd.GetNumberOfPoints()
    ntris = tpd.GetNumberOfCells()
    if npts==0 or ntris==0: return False, npts, ntris, float('inf')

    V = vtk_np.vtk_to_numpy(tpd.GetPoints().GetData()).astype(np.float64, copy=False)
    c = vtk.vtkCellArray(); c.DeepCopy(tpd.GetPolys())
    F = np.empty((c.GetNumberOfCells(), 3), dtype=np.int64); il = vtk.vtkIdList()
    for i in range(c.GetNumberOfCells()):
      c.GetCellAtId(i, il); F[i] = [il.GetId(0), il.GetId(1), il.GetId(2)]

    if F.shape[0] > 200_000:
      idx = np.random.RandomState(0).choice(F.shape[0], size=200_000, replace=False)
      F = F[idx]

    e01 = np.linalg.norm(V[F[:,0]] - V[F[:,1]], axis=1)
    e12 = np.linalg.norm(V[F[:,1]] - V[F[:,2]], axis=1)
    e20 = np.linalg.norm(V[F[:,2]] - V[F[:,0]], axis=1)
    mean_edge = float(np.mean(np.concatenate([e01, e12, e20])))

    diag = max(self._bbox_diag(tpd), 1e-9)
    edge_frac = mean_edge / diag

    dense = (npts > self.DENSE_MAX_POINTS) or (ntris > self.DENSE_MAX_TRIS) or (edge_frac < self.DENSE_EDGE_FRAC)
    return dense, npts, ntris, edge_frac

  def maybeWarnDense(self, pd, label=""):
    if self._denseWarned: return
    dense, npts, ntris, edge_frac = self._mesh_density_score(pd)
    if not dense: return
    self._denseWarned = True
    msg = (f"{label + ': ' if label else ''}mesh appears very dense "
          f"(verts={npts:,}, tris={ntris:,}, mean edge ≈ {edge_frac*100:.3f}% of bbox diag).\n\n"
          "This can slow alignment and dense projection. Consider decimation (e.g., 50–80% reduction):\n"
          "• Slicer: Surface Toolbox → Decimate\n"
          "• Or pre-process with vtkDecimatePro/vtkQuadricDecimation.")
    try:
      slicer.util.warningDisplay(msg, windowTitle="Atlas Builder: Dense mesh")
    except Exception:
      print("[MorphoWeave Atlas Builder][WARN]", msg)

  # -------- Pre/post processing --------
  def cleanPolyData(self, pd, *, tri=True, merge_tol=0.0, fill_holes=0.0, decimate=0.0, smooth_iter=0, normals=True):
    src = pd
    if tri:
      t=vtk.vtkTriangleFilter(); t.SetInputData(src); t.PassLinesOff(); t.PassVertsOff(); t.Update(); src=t.GetOutput()
    c=vtk.vtkCleanPolyData(); c.SetInputData(src); c.ConvertLinesToPointsOff(); c.ConvertPolysToLinesOff()
    c.PointMergingOn(); c.SetToleranceIsAbsolute(True); c.SetAbsoluteTolerance(float(merge_tol)); c.Update(); src=c.GetOutput()
    if fill_holes>0:
      fh=vtk.vtkFillHolesFilter(); fh.SetInputData(src); fh.SetHoleSize(float(fill_holes)); fh.Update(); src=fh.GetOutput()
    if decimate>0:
      d=vtk.vtkDecimatePro(); d.SetInputData(src); d.SetTargetReduction(float(decimate)); d.PreserveTopologyOn()
      d.SplittingOff(); d.BoundaryVertexDeletionOff(); d.Update(); src=d.GetOutput()
    if smooth_iter>0:
      s=vtk.vtkWindowedSincPolyDataFilter(); s.SetInputData(src); s.SetNumberOfIterations(int(smooth_iter))
      s.BoundarySmoothingOff(); s.FeatureEdgeSmoothingOff(); s.NonManifoldSmoothingOn(); s.NormalizeCoordinatesOn(); s.Update(); src=s.GetOutput()
    if normals:
      n=vtk.vtkPolyDataNormals(); n.SetInputData(src); n.SplittingOff(); n.ConsistencyOn(); n.AutoOrientNormalsOn()
      n.ComputeCellNormalsOff(); n.Update(); src=n.GetOutput()
    out = vtk.vtkPolyData(); out.DeepCopy(src); return out

  # -------- Procrustes + atlas building --------
  def procrustesImposition(self, originalLandmarks, rigidOnly):
    flt = vtk.vtkProcrustesAlignmentFilter()
    if rigidOnly: flt.GetLandmarkTransform().SetModeToRigidBody()
    if hasattr(flt, "StartFromCentroidOn"): flt.StartFromCentroidOn()
    flt.SetInputData(originalLandmarks); flt.Update()
    return flt.GetMeanPoints(), flt.GetOutput()

  def getClosestToMeanIndex(self, meanShape, alignedPoints):
    N = alignedPoints.GetNumberOfBlocks()
    if N == 0: return 0
    mean_np = vtk_np.vtk_to_numpy(meanShape.GetData()).astype(np.float32, copy=False)
    best = 0; best_val = float('inf')
    for i in range(N):
      blk = alignedPoints.GetBlock(i)
      if not blk or blk.GetNumberOfPoints()==0: continue
      a = vtk_np.vtk_to_numpy(blk.GetPoints().GetData()).astype(np.float32, copy=False)
      v = np.sum((mean_np - a)**2)
      if v < best_val: best_val, best = v, i
    return best

  def getClosestToMeanPath(self, landmarkDirectory):
    names, L = self.importLandmarks(landmarkDirectory)
    mean, AL = self.procrustesImposition(L, False)
    idx = self.getClosestToMeanIndex(mean, AL)
    return names[idx]

  # -------- Warps (TPS + Biharmonic) --------
  def _warp_polydata_tps(self, pd, srcLM_pts, tgtLM_pts):
    t = vtk.vtkThinPlateSplineTransform(); t.SetSourceLandmarks(srcLM_pts); t.SetTargetLandmarks(tgtLM_pts); t.SetBasisToR()
    f = vtk.vtkTransformPolyDataFilter(); f.SetInputData(pd); f.SetTransform(t); f.Update()
    out = vtk.vtkPolyData(); out.DeepCopy(f.GetOutput()); return out

  def _pd_to_VF(self, pd):
    tri=vtk.vtkTriangleFilter(); tri.SetInputData(pd); tri.PassLinesOff(); tri.PassVertsOff(); tri.Update()
    pd=tri.GetOutput()
    V = vtk_np.vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64, copy=False)
    c=vtk.vtkCellArray(); c.DeepCopy(pd.GetPolys())
    F=np.empty((c.GetNumberOfCells(),3),dtype=np.int64); idl=vtk.vtkIdList()
    for i in range(c.GetNumberOfCells()):
      c.GetCellAtId(i, idl); F[i]=[idl.GetId(0),idl.GetId(1),idl.GetId(2)]
    return V,F

  def _cot_laplacian(self, V, F, area_eps=1e-12, cot_cap=1e6):
    I=[]; J=[]; W=[]
    for a,b,c in F:
      va,vb,vc = V[a],V[b],V[c]
      def cot_at(x,y,z):
        u = y-x; v = z-x
        cr = np.cross(u,v); s = np.linalg.norm(cr)
        if not np.isfinite(s) or s < area_eps: return None
        w = (u@v)/s
        if not np.isfinite(w): return None
        return float(np.clip(w, -cot_cap, cot_cap))
      w_a = cot_at(va,vb,vc); w_b = cot_at(vb,vc,va); w_c = cot_at(vc,va,vb)
      if w_a is None or w_b is None or w_c is None: continue
      for p,q,w in [(a,b,w_c),(b,a,w_c),(b,c,w_a),(c,b,w_a),(c,a,w_b),(a,c,w_b)]:
        I.append(p); J.append(q); W.append(w)
    n = V.shape[0]
    W = coo_matrix((W,(I,J)), shape=(n,n)).tocsr()
    d = np.asarray(W.sum(axis=1)).ravel()
    L = diags(d) - W
    return L.tocsr()

  def _nearest_vertex_ids(self, V, lm_xyz):
    tree=cKDTree(V); _,idx=tree.query(lm_xyz, k=1); return idx.astype(np.int64,copy=False)

  def biharmonicWarpPolyData(self, pd, srcLM_pts, tgtLM_pts, reg=1e-8, allow_fallback=True):
    src = vtk_np.vtk_to_numpy(srcLM_pts.GetData()) if isinstance(srcLM_pts, vtk.vtkPoints) else np.asarray(srcLM_pts, float)
    tgt = vtk_np.vtk_to_numpy(tgtLM_pts.GetData()) if isinstance(tgtLM_pts, vtk.vtkPoints) else np.asarray(tgtLM_pts, float)
    base = self.cleanPolyData(pd, tri=True, merge_tol=0.0, normals=False)
    V, F = self._pd_to_VF(base)
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(tgt)) and np.all(np.isfinite(V))):
      return self._warp_polydata_tps(pd, srcLM_pts, tgtLM_pts) if allow_fallback else base
    for reg_try in (reg, reg*10, reg*100):
      try:
        L = self._cot_laplacian(V, F)
        A = (L.T @ L) + reg_try * speye(L.shape[0], format="csr")
        c_idx = self._nearest_vertex_ids(V, src)
        uniq, inv = np.unique(c_idx, return_inverse=True)
        tgt_at = np.zeros((uniq.size,3)); cnt = np.zeros(uniq.size)
        for i,u in enumerate(inv): tgt_at[u] += tgt[i]; cnt[u] += 1
        tgt_at /= np.maximum(cnt[:,None], 1.0)
        src_at = V[uniq]
        Uc = tgt_at - src_at
        all_idx = np.arange(V.shape[0], dtype=np.int64)
        f_idx = np.setdiff1d(all_idx, uniq, assume_unique=False)
        if f_idx.size == 0: return self._warp_polydata_tps(pd, srcLM_pts, tgtLM_pts) if allow_fallback else base
        Aff = A[f_idx][:, f_idx]; Afc = A[f_idx][:, uniq]; rhs = -Afc @ Uc
        pref = splu(Aff.tocsc()); Uf = pref.solve(rhs)
        U = np.zeros_like(V); U[uniq] = Uc; U[f_idx] = Uf
        Vn = V + U
        if np.all(np.isfinite(Vn)):
          out = vtk.vtkPolyData(); out.DeepCopy(base)
          pts = vtk.vtkPoints(); pts.SetData(vtk_np.numpy_to_vtk(Vn.astype(np.float32), deep=True))
          out.SetPoints(pts); out.GetPoints().Modified()
          return out
      except Exception:
        continue
    return self._warp_polydata_tps(pd, srcLM_pts, tgtLM_pts) if allow_fallback else base

  # -------- Dense correspondence in mean space --------
  def denseCorrespondence(self, LMs, meshes, warp_method="tps", allow_fallback=True, progress=None):
    mean, AL = self.procrustesImposition(LMs, False)
    N = AL.GetNumberOfBlocks()
    if N == 0:
      out = vtk.vtkMultiBlockDataSet(); return out, 0, mean
    baseIdx  = self.getClosestToMeanIndex(mean, AL)
    baseMesh = meshes.GetBlock(baseIdx)
    baseLM   = LMs.GetBlock(baseIdx).GetPoints()
    if baseMesh is None or baseLM is None:
      raise ValueError("denseCorrespondence: invalid base mesh/landmarks.")

    if warp_method == "biharmonic":
      if progress: progress("Template: warping base mesh to mean via BIHARMONIC…")
      bWarp = self.biharmonicWarpPolyData(baseMesh, baseLM, mean, allow_fallback=allow_fallback)
    else:
      if progress: progress("Template: warping base mesh to mean via TPS…")
      t2 = vtk.vtkThinPlateSplineTransform(); t2.SetSourceLandmarks(baseLM); t2.SetTargetLandmarks(mean); t2.SetBasisToR()
      f2 = vtk.vtkTransformPolyDataFilter(); f2.SetInputData(baseMesh); f2.SetTransform(t2); f2.Update()
      bWarp = f2.GetOutput()

    bWarpPts_np = vtk_np.vtk_to_numpy(bWarp.GetPoints().GetData()).astype(np.float64, copy=False)
    bad = ~np.isfinite(bWarpPts_np).all(axis=1)
    if bad.any():
      raise ValueError(
        f"Template warp to mean produced {int(bad.sum())} non-finite vertices. "
        "This usually means the base mesh has NaN/Inf coordinates or the base sparse landmarks are duplicate/degenerate for TPS."
      )

    bWarpPts = bWarp.GetPoints(); basePolys = bWarp.GetPolys()
    grp = vtk.vtkMultiBlockDataGroupFilter()
    for i in range(N):
      lm_i = LMs.GetBlock(i).GetPoints()
      if progress: progress(f"Specimen {i+1}/{N}: warping specimen to mean via {warp_method.upper()} and projecting…")
      corr = self._corr_pair_kdtree(meshes.GetBlock(i), lm_i, bWarpPts, basePolys, mean, warp_method, allow_fallback)
      grp.AddInputData(corr)
    grp.Update()
    return grp.GetOutput(), baseIdx, mean

  def _corr_pair_kdtree(self, mesh, lm_pts, bWarpPts, basePolys, mean_pts, warp_method, allow_fallback):
    if warp_method == "biharmonic":
      mWarp = self.biharmonicWarpPolyData(mesh, lm_pts, mean_pts, allow_fallback=allow_fallback)
    else:
      t1 = vtk.vtkThinPlateSplineTransform(); t1.SetSourceLandmarks(lm_pts); t1.SetTargetLandmarks(mean_pts); t1.SetBasisToR()
      f1 = vtk.vtkTransformPolyDataFilter(); f1.SetInputData(mesh); f1.SetTransform(t1); f1.Update()
      mWarp = f1.GetOutput()

    wpts = vtk_np.vtk_to_numpy(mWarp.GetPoints().GetData())
    finite_mask = np.all(np.isfinite(wpts), axis=1)
    clean_spec = wpts[finite_mask] if not np.all(finite_mask) else wpts
    if clean_spec.shape[0] == 0:
      raise ValueError("Warp produced no finite specimen points for KD-tree matching.")

    tree = cKDTree(clean_spec)
    tmpl = vtk_np.vtk_to_numpy(bWarpPts.GetData()).astype(np.float64, copy=False)
    bad_tmpl = ~np.isfinite(tmpl).all(axis=1)
    if bad_tmpl.any():
      raise ValueError(
        f"Template query points contain {int(bad_tmpl.sum())} non-finite rows before KD-tree search."
      )
    try: distances, idxs = tree.query(tmpl, k=1, workers=-1)
    except TypeError: distances, idxs = tree.query(tmpl, k=1, n_jobs=-1)
    corr_np = clean_spec[idxs]

    pts_vtk = vtk.vtkPoints(); pts_vtk.SetData(vtk_np.numpy_to_vtk(corr_np, deep=True))
    corr = vtk.vtkPolyData(); corr.SetPoints(pts_vtk); corr.SetPolys(basePolys)
    return corr

  def computeAverageModelFromGroup(self, grp, baseIdx):
    n = grp.GetNumberOfBlocks()
    base = grp.GetBlock(baseIdx)
    if n == 0 or base is None or base.GetNumberOfPoints() == 0:
      raise ValueError("computeAverageModelFromGroup: empty dense correspondence group.")
    accum = None
    for i in range(n):
      blk = grp.GetBlock(i)
      if blk is None or blk.GetNumberOfPoints() != base.GetNumberOfPoints():
        raise ValueError("computeAverageModelFromGroup: inconsistent block sizes.")
      pts = vtk_np.vtk_to_numpy(blk.GetPoints().GetData()).astype(np.float64, copy=False)
      if accum is None:
        accum = np.zeros_like(pts, dtype=np.float64)
      accum += pts
    mean = (accum / float(n)).astype(np.float32, copy=False)
    pts  = self.convertPointsToVTK(mean)
    out  = vtk.vtkPolyData(); out.SetPoints(pts.GetPoints()); out.SetPolys(base.GetPolys())
    return out

  def convertPointsToVTK(self, pts):
    a = vtk_np.numpy_to_vtk(pts, deep=True, array_type=vtk.VTK_FLOAT)
    p = vtk.vtkPoints(); p.SetData(a)
    poly = vtk.vtkPolyData(); poly.SetPoints(p)
    return poly

  def runMean(self, lmDir, meshDir, warp_method="tps", allow_fallback=True):
    names, Ms, _, LMs = self.importPaired(meshDir, lmDir, normals=False)
    G, baseIdx, meanPts = self.denseCorrespondence(LMs, Ms, warp_method=warp_method, allow_fallback=allow_fallback)
    avg = self.computeAverageModelFromGroup(G, baseIdx)
    M = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode','Atlas Model'); M.CreateDefaultDisplayNodes(); M.SetAndObservePolyData(avg)
    mean_np = vtk_np.vtk_to_numpy(meanPts.GetData())
    L = self.numpyToFiducialNode(mean_np, "Atlas Landmarks")
    return M, L

  # -------- Alignment (preserve base names + extensions) --------
  def _points_from_markups(self, nodeOrArray):
    if isinstance(nodeOrArray, np.ndarray):
      arr = nodeOrArray.astype(np.float64, copy=False)
    else:
      arr = slicer.util.arrayFromMarkupsControlPoints(nodeOrArray).astype(np.float64, copy=False)
    pts = vtk.vtkPoints(); pts.SetData(vtk_np.numpy_to_vtk(arr, deep=True))
    return pts

  def runAlign(self, baseMeshNode, baseLMNode, meshDir, lmDir, outMeshDir, outLMDir, useSimilarity,
              slmDir=False, outSLMDir=False, progress=None, allowCoordMismatch=False):
    semis = bool(slmDir and outSLMDir)
    scratch = None; model_scratch = None
    md = self._list_keys(meshDir, ('.ply', '.stl', '.vtp', '.vtk', '.obj'))
    ld = self._list_keys(lmDir,  ('.mrk.json', '.fcsv', '.json'))
    sd = self._list_keys(slmDir, ('.mrk.json', '.fcsv', '.json')) if semis else {}
    keys = sorted(set(md) & set(ld))
    total = len(keys); done = 0
    if total == 0:
      raise ValueError("No matching model/landmark file pairs found.")

    try:
      try: slicer.app.setRenderPaused(True)
      except Exception: pass

      scratch = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', 'scratch_lm')
      model_scratch = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'scratch_model')
      scratch.SetHideFromEditors(True); model_scratch.SetHideFromEditors(True)
      if scratch.GetDisplayNode():
        scratch.GetDisplayNode().SetVisibility(False); scratch.GetDisplayNode().SetPointLabelsVisibility(False)
      if model_scratch.GetDisplayNode():
        model_scratch.GetDisplayNode().SetVisibility(False)

      tgt_arr = slicer.util.arrayFromMarkupsControlPoints(baseLMNode).astype(np.float64, copy=False)
      tgt_pts = self._points_from_markups(tgt_arr)
      tf = vtk.vtkLandmarkTransform(); tpf = vtk.vtkTransformPolyDataFilter()
      threshold = 0.03
      last_ui_pump = 0.0

      for sid in keys:
        lmNode = slicer.util.loadMarkups(ld[sid])
        modelNode = slicer.util.loadModel(md[sid])

        if lmNode.GetNumberOfControlPoints() != baseLMNode.GetNumberOfControlPoints():
          slicer.mrmlScene.RemoveNode(lmNode)
          raise ValueError(f"Landmark mismatch for {sid}")

        mesh_pd = vtk.vtkPolyData(); mesh_pd.DeepCopy(modelNode.GetPolyData())
        slicer.mrmlScene.RemoveNode(modelNode)

        self.maybeWarnDense(mesh_pd, label=sid)

        if mesh_pd.GetNumberOfPoints() == 0 or lmNode.GetNumberOfControlPoints() == 0:
          slicer.mrmlScene.RemoveNode(lmNode)
          raise ValueError(f"[{sid}] Empty mesh or landmark set.")

        lm_arr = slicer.util.arrayFromMarkupsControlPoints(lmNode).astype(np.float64, copy=False)
        r0, rflip = self._coord_check_ratio(mesh_pd, lm_arr, threshold=threshold)
        if not allowCoordMismatch and r0 > threshold:
          hint = " Likely LPS landmarks (X,Y need sign flip)." if (rflip < 0.5*r0 and rflip < threshold) else ""
          slicer.mrmlScene.RemoveNode(lmNode)
          raise ValueError(f"[{sid}] Landmark↔mesh coordinate mismatch: median surface distance ≈ {r0*100:.1f}% of bbox diag.{hint} Set 'Override landmark↔mesh coordinate check' to proceed.")

        src_pts = self._points_from_markups(lm_arr)
        tf.SetModeToSimilarity() if useSimilarity else tf.SetModeToRigidBody()
        tf.SetSourceLandmarks(src_pts); tf.SetTargetLandmarks(tgt_pts); tf.Update()
        tpf.SetInputData(mesh_pd); tpf.SetTransform(tf); tpf.Update()
        aligned_mesh_pd = tpf.GetOutput()

        # Save aligned model (preserve original extension)
        ext = os.path.splitext(md[sid])[1] or ".ply"
        with slicer.util.NodeModify(model_scratch):
          model_scratch.SetAndObservePolyData(aligned_mesh_pd)
        slicer.util.saveNode(model_scratch, os.path.join(outMeshDir, f"{sid}{ext}"))

        # Transform and save landmarks (keep base name)
        lm_h = np.hstack((lm_arr, np.ones((lm_arr.shape[0], 1), dtype=np.float64)))
        M = vtk.vtkMatrix4x4(); tf.GetMatrix(M)
        Mn = np.array([[M.GetElement(r,c) for c in range(4)] for r in range(4)], dtype=np.float64)
        aligned = (Mn @ lm_h.T).T[:, :3]
        with slicer.util.NodeModify(scratch):
          slicer.util.updateMarkupsControlPointsFromArray(scratch, aligned.astype(np.float32, copy=False))
        slicer.util.saveNode(scratch, os.path.join(outLMDir, f"{sid}.mrk.json"))
        slicer.mrmlScene.RemoveNode(lmNode)

        if semis:
          slm_path = sd.get(sid)
          slm = slicer.util.loadMarkups(slm_path) if slm_path else None
          if slm:
            slm_arr = slicer.util.arrayFromMarkupsControlPoints(slm).astype(np.float64, copy=False)
            slm_h = np.hstack((slm_arr, np.ones((slm_arr.shape[0], 1), dtype=np.float64)))
            slm_al = (Mn @ slm_h.T).T[:, :3]
            with slicer.util.NodeModify(scratch):
              slicer.util.updateMarkupsControlPointsFromArray(scratch, slm_al.astype(np.float32, copy=False))
            slicer.util.saveNode(scratch, os.path.join(outSLMDir, f"{sid}.mrk.json"))
            slicer.mrmlScene.RemoveNode(slm)

        done += 1
        if progress: progress(done, total, sid)
        now = time.monotonic()
        if done == total or (now - last_ui_pump >= 0.10):
          qt.QApplication.processEvents()
          last_ui_pump = now

    finally:
      if scratch: slicer.mrmlScene.RemoveNode(scratch)
      if model_scratch: slicer.mrmlScene.RemoveNode(model_scratch)
      try: slicer.app.setRenderPaused(False)
      except Exception: pass

  def _bbox_diag(self, pd):
    bounds = pd.GetBounds()
    return float(np.linalg.norm([
      bounds[1] - bounds[0],
      bounds[3] - bounds[2],
      bounds[5] - bounds[4],
    ]))

  # -------- Dense export (method-selectable) --------
  def sample_indices_voxel_target(self, pd, target_count=2000, tolerance=0.02):
    if pd is None or pd.GetPoints() is None:
      return np.empty((0,), dtype=np.int64)
    points = vtk_np.vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64, copy=False)
    return sample_indices_voxel_target_points(
      points,
      target_count=target_count,
      tolerance=tolerance,
    )

  def previewCountForTarget(self, polyDataOrNode, targetCount):
    pd = polyDataOrNode if isinstance(polyDataOrNode, vtk.vtkPolyData) else polyDataOrNode.GetPolyData()
    keep_idx = self.sample_indices_voxel_target(pd, int(targetCount))
    return int(keep_idx.size), int(pd.GetNumberOfPoints())

  def _build_surface_locator(self, mesh_pd):
    loc = vtk.vtkStaticCellLocator()
    loc.SetDataSet(mesh_pd)
    loc.BuildLocator()
    return loc

  def sparseCorrespondenceBaseMesh(self, LMs, meshes, atlasMesh, atlasLM, keep_idx, *,
                                  warp_method="tps", allow_fallback=True, progress=None, mesh_locators=None):
    keep_idx = np.asarray(keep_idx, np.int64)
    if keep_idx.size == 0: return []

    out=[]; N=LMs.GetNumberOfBlocks()
    n_atlas_lm = int(atlasLM.GetNumberOfPoints())
    if warp_method == "biharmonic":
      if progress: progress("Warping entire atlas mesh → each specimen via BIHARMONIC, then projecting…")
      for i in range(N):
        lm_i = LMs.GetBlock(i).GetPoints()
        if lm_i is None or int(lm_i.GetNumberOfPoints()) != n_atlas_lm:
          raise ValueError(f"Landmark count mismatch for specimen {i+1}: expected {n_atlas_lm}, got {0 if lm_i is None else int(lm_i.GetNumberOfPoints())}.")
        warped_atlas = self.biharmonicWarpPolyData(atlasMesh, atlasLM, lm_i, allow_fallback=allow_fallback)
        V = vtk_np.vtk_to_numpy(warped_atlas.GetPoints().GetData())
        qInSpec = V[keep_idx].astype(np.float32, copy=False)
        loc = mesh_locators[i] if mesh_locators is not None else self._build_surface_locator(meshes.GetBlock(i))
        outPts = vtk.vtkPoints(); outPts.SetNumberOfPoints(qInSpec.shape[0])
        hit=[0.,0.,0.]; cid=vtk.reference(0); sid=vtk.reference(0); d2=vtk.reference(0.0)
        for j in range(qInSpec.shape[0]):
          loc.FindClosestPoint(qInSpec[j], hit, cid, sid, d2)
          outPts.SetPoint(j, hit)
        arr = vtk_np.vtk_to_numpy(outPts.GetData()).astype(np.float32, copy=False)
        out.append(arr)
    else:
      if progress: progress("Warping sampled atlas points → each specimen via TPS, then projecting…")
      aPts = atlasMesh.GetPoints()
      qpts = vtk.vtkPoints(); qpts.SetNumberOfPoints(len(keep_idx))
      tmp = [0.0, 0.0, 0.0]
      for i,k in enumerate(keep_idx):
        aPts.GetPoint(int(k), tmp); qpts.SetPoint(i, tmp)
      for i in range(N):
        lm_i = LMs.GetBlock(i).GetPoints()
        if lm_i is None or int(lm_i.GetNumberOfPoints()) != n_atlas_lm:
          raise ValueError(f"Landmark count mismatch for specimen {i+1}: expected {n_atlas_lm}, got {0 if lm_i is None else int(lm_i.GetNumberOfPoints())}.")
        t = vtk.vtkThinPlateSplineTransform()
        t.SetSourceLandmarks(atlasLM); t.SetTargetLandmarks(lm_i); t.SetBasisToR()
        qPoly = vtk.vtkPolyData(); qPoly.SetPoints(qpts)
        f = vtk.vtkTransformPolyDataFilter(); f.SetInputData(qPoly); f.SetTransform(t); f.Update()
        qInSpec = f.GetOutput()
        loc = mesh_locators[i] if mesh_locators is not None else self._build_surface_locator(meshes.GetBlock(i))
        outPts = vtk.vtkPoints(); outPts.SetNumberOfPoints(qInSpec.GetNumberOfPoints())
        qp = [0.0,0.0,0.0]; hit=[0.0,0.0,0.0]; cid=vtk.reference(0); sid=vtk.reference(0); d2=vtk.reference(0.0)
        for j in range(qInSpec.GetNumberOfPoints()):
          qInSpec.GetPoint(j, qp); loc.FindClosestPoint(qp, hit, cid, sid, d2); outPts.SetPoint(j, hit)
        arr = vtk_np.vtk_to_numpy(outPts.GetData()).astype(np.float32, copy=False)
        out.append(arr)
    return out

  def exportDenseLMs(self, atlasModelNode, atlasLMNode,
                     alignedModelsDir, alignedLMsDir,
                     denseDir, atlasDir,
                     targetPointCount,
                     *, warp_method="tps", allow_fallback=True, progress=None):
    self._harden_if_parented(atlasModelNode)
    self._harden_if_parented(atlasLMNode)
    self.maybeWarnDense(atlasModelNode.GetPolyData(), label="atlas")

    def _has_files(d, exts):
      try: return any(fn.lower().endswith(exts) for fn in os.listdir(d))
      except Exception: return False
    if not _has_files(alignedModelsDir, ('.ply', '.stl', '.vtp', '.vtk', '.obj')) \
       or not _has_files(alignedLMsDir, ('.mrk.json', '.fcsv', '.json')):
      raise ValueError("No aligned models/landmarks found to export.")

    scratch = None
    try:
      try: slicer.app.setRenderPaused(True)
      except Exception: pass

      keep_idx = self.sample_indices_voxel_target(
        atlasModelNode.GetPolyData(),
        int(targetPointCount),
      )
      if keep_idx.size == 0: raise ValueError("Target sampling produced 0 template points.")

      names, models, _, lms = self.importPaired(alignedModelsDir, alignedLMsDir, normals=False)
      if len(names) == 0: raise ValueError("No paired aligned models/landmarks found to export.")
      mesh_locators = [self._build_surface_locator(models.GetBlock(i)) for i in range(models.GetNumberOfBlocks())]

      baseLM_pts = self.fiducialNodeToPolyData(atlasLMNode, load=False).GetPoints()
      if progress: progress(f"Computing specimen-space correspondences ({warp_method.upper()})…")
      pts_list = self.sparseCorrespondenceBaseMesh(
        lms, models, atlasModelNode.GetPolyData(), baseLM_pts, keep_idx,
        warp_method=warp_method, allow_fallback=allow_fallback, progress=progress, mesh_locators=mesh_locators
      )

      scratch = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', 'scratch_export')
      scratch.SetHideFromEditors(True)
      sd = scratch.GetDisplayNode()
      if sd: sd.SetVisibility(False); sd.SetPointLabelsVisibility(False)

      for i, arr in enumerate(pts_list):
        with slicer.util.NodeModify(scratch):
          slicer.util.updateMarkupsControlPointsFromArray(scratch, arr)
        slicer.util.saveNode(scratch, os.path.join(denseDir, f"{names[i]}.mrk.json"))
        if progress: progress(f"Saved dense points for {names[i]}")

      atlas_all = vtk_np.vtk_to_numpy(atlasModelNode.GetPolyData().GetPoints().GetData())
      atlas_tpl = atlas_all[keep_idx].astype(np.float32, copy=False)
      with slicer.util.NodeModify(scratch):
        slicer.util.updateMarkupsControlPointsFromArray(scratch, atlas_tpl)
      slicer.util.saveNode(scratch, os.path.join(atlasDir, "atlas_dense_correspondences.mrk.json"))
      if progress: progress("Saved template points to atlas/atlas_dense_correspondences.mrk.json")
    finally:
      if scratch:
        slicer.mrmlScene.RemoveNode(scratch)
      try: slicer.app.setRenderPaused(False)
      except Exception: pass

  # -------- Misc helpers --------
  def _harden_if_parented(self, node):
    tn = node.GetParentTransformNode()
    if tn: slicer.vtkSlicerTransformLogic().hardenTransform(node)

  def saveAtlasOnly(self, atlasModelNode, atlasLMNode, atlasDir):
    os.makedirs(atlasDir, exist_ok=True)
    self._harden_if_parented(atlasModelNode); self._harden_if_parented(atlasLMNode)
    slicer.util.saveNode(atlasModelNode, os.path.join(atlasDir, "atlas_model.ply"))
    slicer.util.saveNode(atlasLMNode,   os.path.join(atlasDir, "atlas_sparse_landmarks.mrk.json"))


class MorphoWeaveAtlasBuilderTest(ScriptedLoadableModuleTest):
  def setUp(self):
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    self.setUp()
    self.assertIsNotNone(MorphoWeaveAtlasBuilderLogic())
    self.assertTrue(MorphoWeaveAtlasBuilderLogic.isSafeLibraryName("mouse_atlas"))
    self.assertTrue(MorphoWeaveAtlasBuilderLogic.isSafeLibraryName("Mouse Atlas 2026"))
    for invalidName in ("", ".", "..", "../atlas", "atlas/model", "atlas\\model", "CON", "name."):
      self.assertFalse(MorphoWeaveAtlasBuilderLogic.isSafeLibraryName(invalidName))
