import os, logging, copy, json, time, html, re, numpy as np
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import vtk.util.numpy_support as vtk_np


from Resources.Python.PoseEMTemplate import (
    LEGACY_BACKEND,
    POSE_EM_BACKEND,
    PoseEMSettings,
    coverage_prescale,
    optimization_backend,
    pose_em_enabled,
    run_pose_em_registration,
    ssm_sample,
)

# ---------- Small utilities----------

def stable_distribution_release(version_text):
  """Return stable release components, rejecting prerelease version strings."""
  match = re.fullmatch(
    r"\s*(\d+(?:\.\d+)*)(?:\.post\d+)?(?:\+[A-Za-z0-9][A-Za-z0-9._-]*)?\s*",
    str(version_text),
  )
  return tuple(int(part) for part in match.group(1).split(".")) if match else None

def target_count_voxel_size(points, target_count, tolerance=0.02, max_iterations=40):
  points = np.asarray(points, dtype=np.float64)
  if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
    raise ValueError("points must be a non-empty array with shape (N, 3)")
  if not np.all(np.isfinite(points)):
    raise ValueError("points must contain only finite coordinates")
  target_count = int(target_count)
  if target_count <= 0:
    raise ValueError("target_count must be positive")

  origin = points.min(axis=0)
  diagonal = float(np.linalg.norm(points.max(axis=0) - origin))
  if diagonal <= np.finfo(np.float64).eps:
    return float(np.finfo(np.float64).eps)

  low = diagonal * 1e-9
  high = diagonal
  best_size = low
  best_error = abs(len(points) - target_count)
  tolerance_count = max(1, int(round(target_count * float(tolerance))))

  for _ in range(max(1, int(max_iterations))):
    voxel_size = float(np.sqrt(low * high))
    keys = np.floor((points - origin) / voxel_size).astype(np.int64)
    count = int(np.unique(keys, axis=0).shape[0])
    error = abs(count - target_count)
    if error < best_error:
      best_error = error
      best_size = voxel_size
      if best_error <= tolerance_count:
        break
    if count > target_count:
      low = voxel_size
    elif count < target_count:
      high = voxel_size
    else:
      return voxel_size

  return best_size

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

def canonical_ssm_node_names(database_name):
  return {
    "table": f"ssm_data_{database_name}",
    "model": f"{database_name}_template",
    "dense": f"{database_name}_template_correspondences",
    "sparse": f"{database_name}_template_sparse_landmarks",
  }

def latest_complete_ssm_set(tables, models, landmarks):
  models_by_name = {node.GetName(): node for node in models}
  landmarks_by_name = {node.GetName(): node for node in landmarks}
  for table in reversed(list(tables)):
    table_name = table.GetName() or ""
    if not table_name.startswith("ssm_data_"):
      continue
    database_name = table_name[len("ssm_data_"):]
    names = canonical_ssm_node_names(database_name)
    model = models_by_name.get(names["model"])
    dense = landmarks_by_name.get(names["dense"])
    sparse = landmarks_by_name.get(names["sparse"])
    if not all((database_name, model, dense, sparse)):
      continue
    try:
      expected_points = int(table.GetAttribute("ssm_npoints") or "")
    except (TypeError, ValueError):
      continue
    if expected_points > 0 and dense.GetNumberOfControlPoints() == expected_points:
      return {"table": table, "model": model, "dense": dense, "sparse": sparse}
  return None

def fill_empty_selectors(selector_nodes):
  changed = False
  for selector, node in selector_nodes:
    if selector.currentNode() is None:
      selector.setCurrentNode(node)
      changed = True
  return changed

COLORS = {"red":(1,0.2,0.2), "green":(0.2,1,0.2), "blue":(0.2,0.2,1)}

def np_to_vtk_mat(M):
    M = np.asarray(M, float); assert M.shape==(4,4)
    vm = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4): vm.SetElement(i,j,float(M[i,j]))
    return vm

def vtk_mat_to_np(vm):
    return np.array([[vm.GetElement(i,j) for j in range(4)] for i in range(4)], float)

def as_np4x4(M):
    if isinstance(M, vtk.vtkMatrix4x4): return vtk_mat_to_np(M)
    A = np.asarray(M, float)
    return A if A.shape==(4,4) else np.eye(4, dtype=float)

def make_transform_node(M, name):
    t = vtk.vtkTransform(); t.SetMatrix(np_to_vtk_mat(as_np4x4(M)))
    tn = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode', name)
    tn.SetAndObserveTransformToParent(t); return tn

def transform_polydata(pd, M):
    tf = vtk.vtkTransformPolyDataFilter(); tf.SetTransform(vtk.vtkTransform())
    tf.GetTransform().SetMatrix(np_to_vtk_mat(as_np4x4(M)))
    tf.SetInputData(pd); tf.Update(); out = vtk.vtkPolyData(); out.DeepCopy(tf.GetOutput()); return out

def clone_polydata_points_only(pd):
    out = vtk.vtkPolyData()
    out.ShallowCopy(pd)
    src_pts = pd.GetPoints()
    pts = vtk.vtkPoints()
    src_arr = src_pts.GetData()
    vtk_type = src_arr.GetDataType() if src_arr else vtk.VTK_FLOAT
    pts.SetData(vtk_np.numpy_to_vtk(vtk_np.vtk_to_numpy(src_arr).copy(), deep=1, array_type=vtk_type))
    out.SetPoints(pts)
    out.GetPoints().Modified()
    return out

def bounds_diag(node_or_pd):
    pd = node_or_pd if isinstance(node_or_pd, vtk.vtkPolyData) else node_or_pd.GetPolyData()
    b = np.array(pd.GetBounds(), float).reshape(3,2)
    return float(np.linalg.norm(b[:,1]-b[:,0]))

# ---------- Module ----------
class MorphoWeaveLandmarkTransfer(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Landmark Transfer"
    self.parent.categories = ["MorphoWeave"]
    self.parent.dependencies = []
    self.parent.contributors = ["Arthur Porto"]
    self.parent.helpText = (
      """
          MorphoWeave Landmark Transfer combines global rigid registration with
          Statistical Shape Model-guided Coherent Point Drift for non-rigid alignment,
          followed by optional surface projection refinement. It supports both
          single-specimen optimization and batch landmark transfer.
          <p>For more information see the
          <a href=\"https://github.com/agporto/SlicerMorphoWeave/blob/main/tutorial/README.md#landmark-transfer\">online documentation</a>.</p>
      """
    )
    self.parent.acknowledgementText = "This module was developed by Arthur Porto"

# ---------- Widget ----------
class MorphoWeaveLandmarkTransferWidget(ScriptedLoadableModuleWidget):
  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    self.cloneFolderItemID = None
    self._deps_ready = False
    qt.QTimer.singleShot(0, self._ensure_dependencies)

  # ----- Slicer-native dependency installer -----
  def _ensure_dependencies(self):
    import importlib, importlib.metadata, inspect, traceback, sys

    required = {
      "tiny3d-rs": {
        "spec": "tiny3d-rs>=2.1,<3",
        "module": "tiny3d",
        "minimum": (2, 1),
        "maximum": (3,),
      },
      "rustcpd": {
        "spec": "rustcpd>=3.0,<4",
        "module": "rustcpd",
        "minimum": (3,),
        "maximum": (4,),
      },
    }

    def distributionVersion(distribution_name):
      try:
        version_text = importlib.metadata.version(distribution_name)
      except importlib.metadata.PackageNotFoundError:
        return None
      # Accept stable, post, and local releases. Reject prereleases because
      # they do not satisfy pip's stable >= floor without an explicit opt-in.
      return stable_distribution_release(version_text)

    def distributionInstalled(distribution_name):
      try:
        importlib.metadata.version(distribution_name)
        return True
      except importlib.metadata.PackageNotFoundError:
        return False

    def hasCompatibleDistribution(distribution_name):
      requirement = required[distribution_name]
      version = distributionVersion(distribution_name)
      return bool(
        version is not None
        and requirement["minimum"] <= version < requirement["maximum"]
      )

    def hasPoseAPI():
      if not hasCompatibleDistribution("rustcpd"):
        return False
      try:
        rustcpd_module = importlib.import_module("rustcpd")
        initializer = getattr(rustcpd_module, "pose_initialize")
        parameters = inspect.signature(initializer).parameters
        return all(name in parameters for name in (
          "coarse_screen_iterations", "coarse_survivor_count",
          "coarse_score_mode", "refine_source_count",
          "lambda_regularization", "parallel",
        ))
      except Exception:
        return False

    legacy_tiny3d_installed = distributionInstalled("tiny3d")
    missing = [
      name for name in required
      if not hasCompatibleDistribution(name)
    ]
    if not hasPoseAPI() and "rustcpd" not in missing:
      missing.append("rustcpd")
    if legacy_tiny3d_installed and "tiny3d-rs" not in missing:
      # Both distributions own the same tiny3d import package. Reinstall the
      # Rust distribution after removing the legacy wheel so shared files are
      # not left missing.
      missing.append("tiny3d-rs")

    loaded_replacements = [
      required[name]["module"] for name in missing
      if any(
        module_name == required[name]["module"]
        or module_name.startswith(required[name]["module"] + ".")
        for module_name in sys.modules
      )
    ]
    if loaded_replacements:
      slicer.util.infoDisplay(
        "Landmark Transfer must replace an already loaded native backend "
        f"({', '.join(loaded_replacements)}). Restart Slicer, then reopen "
        "Landmark Transfer to install the Rust backend safely. No packages "
        "were changed."
      )
      return

    if missing:
      labels = [required[name]["spec"] for name in missing]
      msg = "Landmark Transfer needs or must upgrade: " + ", ".join(labels) + "."
      if legacy_tiny3d_installed:
        msg += "\nThe legacy tiny3d distribution must be removed before tiny3d-rs is installed."
      msg += "\nInstall now?"
      if not slicer.util.confirmOkCancelDisplay(msg):
        slicer.util.errorDisplay("Dependencies not installed; some actions may fail.")
        return

    self._deps_ready = False
    try:
      if legacy_tiny3d_installed:
        pip_uninstall = getattr(slicer.util, "pip_uninstall", None)
        if pip_uninstall is None:
          raise RuntimeError(
            "This Slicer build cannot remove the legacy tiny3d distribution. "
            "Uninstall tiny3d manually, restart Slicer, and reopen Landmark Transfer."
          )
        pip_uninstall("tiny3d")

      if missing:
        for package_name in [required[name]["module"] for name in missing]:
          for module_name in [
            name for name in sys.modules
            if name == package_name or name.startswith(package_name + ".")
          ]:
            sys.modules.pop(module_name, None)
        importlib.invalidate_caches()
        specs = [required[name]["spec"] for name in missing]
        install_args = ["--upgrade"]
        if legacy_tiny3d_installed:
          install_args.append("--force-reinstall")
        slicer.util.pip_install([*install_args, *specs])
        importlib.invalidate_caches()

      for module_name in ("tiny3d", "rustcpd", "scipy.spatial", "scipy.optimize"):
        importlib.import_module(module_name)
      if not hasCompatibleDistribution("tiny3d-rs"):
        raise RuntimeError("A compatible tiny3d-rs>=2.1,<3 distribution is not installed.")
      if not hasPoseAPI():
        raise RuntimeError(
          "The installed rustcpd>=3.0,<4 wheel does not provide the required native Pose-EM API."
        )
    except Exception as error:
      slicer.util.errorDisplay(f"Landmark Transfer dependency installation failed:\n{error}\n{traceback.format_exc()}")
      return
    self._deps_ready = True
    slicer.util.showStatusMessage("Landmark Transfer: dependencies ready", 3000)


  # ----- Visual helpers -----
  def _activeCameraAndView(self):
    lm = slicer.app.layoutManager()
    view = lm.threeDWidget(0).threeDView()
    viewNode = view.mrmlViewNode()
    camNode = slicer.modules.cameras.logic().GetViewActiveCameraNode(viewNode)
    return camNode.GetCamera(), view

  def _focusOnNode(self, node, keepOrientation=True, margin=1.15):
      cam, view = self._activeCameraAndView()

      # bounds → center & radius
      b = np.array(node.GetPolyData().GetBounds(), float).reshape(3, 2)
      center = b.mean(1)
      maxdim = np.max(b[:, 1] - b[:, 0])
      r = 0.5 * maxdim * float(margin)

      # keep current az/el unless asked otherwise
      if keepOrientation:
          fp = np.array(cam.GetFocalPoint())
          pos = np.array(cam.GetPosition())
          vdir = fp - pos
          if np.linalg.norm(vdir) < 1e-9:
              vdir = np.array([0, 0, -1.0])
          vdir = vdir / np.linalg.norm(vdir)
          up = np.array(cam.GetViewUp())
      else:
          vdir = np.array([0, 0, -1.0]); up = np.array([0, 1, 0])

      # distance to fit vertically
      fovy = np.deg2rad(float(cam.GetViewAngle()))
      dist = r / max(np.tan(fovy / 2.0), 1e-6)

      # set camera
      cam.SetFocalPoint(*center.tolist())
      cam.SetPosition(*(center - vdir * dist).tolist())
      cam.SetViewUp(*up.tolist())
      cam.SetClippingRange(max(dist - 3 * r, 1e-3), dist + 3 * r)
      view.scheduleRender()

  # ----- Parameter schema (declarative Advanced tab) -----
  PARAMS = [
    {"key":"skipScaling","kind":"check","label":"Skip scaling","section":"General Settings","value":False},
    {"key":"skipProjection","kind":"check","label":"Skip projection","section":"General Settings","value":False},
    {"key":"skipOptimization","kind":"check","label":"Skip template optimization (batch)","section":"General Settings","value":False},
    {"key":"targetCoverage","kind":"dspin","label":"Target completeness (linear fraction)","section":"General Settings", "min":0.05,"max":1.0,"step":0.01,"value":1.0,"decimals":2},
    {"key":"registrationPointCount","kind":"spin","label":"Target registration points","section":"Point sampling and max projection","min":100,"max":1_000_000,"step":100,"value":2000,
     "tooltip":"Target size for temporary registration clouds. Values below 1,600 limit Pose-EM refinement."},
    {"key":"projectionFactor","kind":"slider","label":"Max projection factor (%)","section":"Point sampling and max projection","min":0,"max":10,"step":0.1,"value":1.0},
    {"key":"normalSearchRadius","kind":"slider","label":"Normal search radius","section":"Rigid registration","min":2,"max":12,"step":1,"value":2},
    {"key":"FPFHSearchRadius","kind":"slider","label":"FPFH search radius","section":"Rigid registration","min":3,"max":20,"step":1,"value":5},
    {"key":"distanceThreshold","kind":"slider","label":"RANSAC distance threshold","section":"Rigid registration","min":0.5,"max":4.0,"step":0.25,"value":1.5},
    {"key":"maxRANSAC","kind":"spin","label":"Max RANSAC iterations","section":"Rigid registration","min":1,"max":50_000_000,"step":1,"value":400_000},
    {"key":"confidence","kind":"dspin","label":"RANSAC confidence","section":"Rigid registration","min":0.0,"max":1.0,"step":0.001,"value":0.999},
    {"key":"ICPDistanceThreshold","kind":"slider","label":"ICP distance threshold","section":"Rigid registration","min":0.1,"max":2.0,"step":0.1,"value":0.4},
    {"key":"useBiharmonic","kind":"check","label":"Experimental: Use biharmonic surface warp","section":"Deformation backend","value":False},
    {"key":"bih_lam","kind":"dspin","label":"Biharmonic stiffness (lambda)","section":"Deformation backend","min":1.0,"max":1000000.0,"step":100.0,"value":10000.0},
    {"key":"tpsLambda","kind":"dspin","label":"TPS smoothing (λ)","section":"Deformation backend","min":0.0,"max":10.0,"step":0.01,"value":0.0},
    {"key":"tpsMaxCorr","kind":"spin","label":"TPS max constraints","section":"Deformation backend","min":20,"max":5000,"step":20,"value":800},
    {"key":"alpha","kind":"dspin","label":"Rigidity (alpha)","section":"PCA-CPD registration","min":0.1,"max":10.0,"step":0.1,"value":2.0},
    {"key":"beta","kind":"dspin","label":"Motion coherence (beta)","section":"PCA-CPD registration","min":0.1,"max":10.0,"step":0.1,"value":2.0},
    {"key":"skipFineCPD","kind":"check", "label":"Fossil mode (SSM-only)", "section":"PCA-CPD registration","value":False},
    {"key":"w","kind":"dspin","label":"Outlier weight (w)","section":"PCA-CPD registration","min":0.0,"max":0.5,"step":0.01,"value":0.10,"decimals":3},
    {"key":"tolerance","kind":"dspin","label":"Tolerance","section":"PCA-CPD registration","min":1e-8,"max":1e-2,"step":1e-6,"value":1e-6,"decimals":8},
    {"key":"max_iterations","kind":"spin","label":"Max iterations","section":"PCA-CPD registration","min":100,"max":1000,"step":50,"value":250},
    {"key":"lambda_reg","kind":"dspin","label":"SSM weight (lambda_reg)","section":"PCA-CPD registration","min":0.0,"max":5.0,"step":0.01,"value":0.01}
  ]

  def _make_selector(self, types, attr_key=None, attr_val=None, tooltip=None, none=True, select_new=False):
    sel = slicer.qMRMLNodeComboBox()
    sel.nodeTypes = types; sel.selectNodeUponCreation=bool(select_new)
    sel.addEnabled=False; sel.removeEnabled=False; sel.noneEnabled=bool(none)
    sel.setMRMLScene(slicer.mrmlScene)
    if attr_key:
      if attr_val is None:
        sel.addAttribute(types[0], attr_key)
      else:
        sel.addAttribute(types[0], attr_key, attr_val)
    if tooltip: sel.setToolTip(tooltip)
    return sel

  def _build_advanced_tab(self, layout):
    sections={}
    def section(name):
      if name not in sections:
        cb=ctk.ctkCollapsibleButton(); cb.text=name
        if name in ("Rigid registration", "Deformation backend"):
          cb.collapsed = True
        layout.addRow(cb); sections[name]=qt.QFormLayout(cb)
      return sections[name]
    for p in self.PARAMS:
      kind=p["kind"]; lab=p["label"]; key=p["key"]; sec=p["section"]; v=p.get("value")
      if kind=="check":
        w=qt.QCheckBox(); w.checked=bool(v); w.toggled.connect(self._on_param_changed)
      elif kind=="slider":
        w=ctk.ctkSliderWidget(); w.minimum=p["min"]; w.maximum=p["max"]; w.singleStep=p["step"]; w.value=float(v); w.valueChanged.connect(self._on_param_changed)
      elif kind=="spin":
        w=qt.QSpinBox(); w.minimum=int(p["min"]); w.maximum=int(p["max"]); w.singleStep=int(p["step"]); w.value=int(v); w.valueChanged.connect(self._on_param_changed)
      elif kind=="dspin":
        w=ctk.ctkDoubleSpinBox(); w.minimum=float(p["min"]); w.maximum=float(p["max"]); w.singleStep=float(p["step"]); w.value=float(v); w.setDecimals(int(p.get("decimals",3))); w.valueChanged.connect(self._on_param_changed)
      else:
        continue
      if p.get("tooltip"):
        w.setToolTip(p["tooltip"])
      setattr(self, f"param_{key}", w)
      section(sec).addRow(lab+": ", w)
    self.parameterDictionary = self._read_params()

  def _read_params(self):
    d={}
    for p in self.PARAMS:
      w=getattr(self, f"param_{p['key']}")
      if p["kind"]=="check": d[p['key']] = bool(w.checked)
      else: d[p['key']] = float(w.value) if hasattr(w,'value') else w.value()
    # integers for specific keys
    d["maxRANSAC"] = int(d["maxRANSAC"]); d["max_iterations"] = int(d["max_iterations"])
    d["registrationPointCount"] = int(d["registrationPointCount"])
    d["normalSearchRadius"] = int(d["normalSearchRadius"]); d["FPFHSearchRadius"] = int(d["FPFHSearchRadius"])
    return d

  def _on_param_changed(self, *args):
    self.parameterDictionary = self._read_params()

  def _optimization_backend(self):
    try:
      index = int(self.optimizationBackendCombo.currentIndex)
    except TypeError:
      index = int(self.optimizationBackendCombo.currentIndex())
    return POSE_EM_BACKEND if index == 1 else LEGACY_BACKEND

  def _optimization_parameters(self):
    try:
      scoreModeIndex = int(self.poseCoarseScoreMode.currentIndex)
    except TypeError:
      scoreModeIndex = int(self.poseCoarseScoreMode.currentIndex())
    parameters = dict(self.parameterDictionary)
    parameters.update({
      "optimizationBackend": self._optimization_backend(),
      "opt_pcGridSteps": int(self.pcGridSteps.value),
      "opt_ransac_per_cand": int(self.ransacItersPerCand.value),
      "poseRotationCount": int(self.poseRotationCount.value),
      "poseCoarseSourceCount": int(self.poseCoarseSourceCount.value),
      "poseCoarseTargetCount": int(self.poseCoarseTargetCount.value),
      "poseCoarseRank": int(self.poseCoarseRank.value),
      "poseCoarseIterations": int(self.poseCoarseIterations.value),
      "poseCoarseScreenIterations": int(self.poseCoarseScreenIterations.value),
      "poseCoarseSurvivorCount": int(self.poseCoarseSurvivorCount.value),
      "poseCoarseScoreMode": "trajectory" if scoreModeIndex == 0 else "final",
      "poseRefineCount": int(self.poseRefineCount.value),
      "poseRefineSourceCount": int(self.poseRefineSourceCount.value),
      "poseRefineTargetCount": int(self.poseRefineTargetCount.value),
      "poseRefineIterations": int(self.poseRefineIterations.value),
      "poseLambdaReg": float(self.poseLambdaReg.value),
      "poseOutlierWeight": float(self.poseOutlierWeight.value),
      "poseIdentityPrior": float(self.poseIdentityPrior.value),
      "poseSeed": int(self.poseSeed.value),
      "poseParallel": bool(self.poseParallel.checked),
    })
    return parameters

  def _onOptimizationBackendChanged(self, *args):
    pose = self._optimization_backend() == POSE_EM_BACKEND
    self.legacyOptimizationBox.visible = not pose
    self.poseOptimizationBox.visible = pose
    self.batchBackendLabel.setText(
      "Template optimization backend: " +
      ("Pose-marginalized EM (experimental)" if pose else "FPFH + RANSAC (current)")
    )
    if pose:
      self.optimizeButton.setText("Run Experimental Pose-EM Optimization")
      self.optimizeButton.setToolTip(
        "Use pose-marginalized registration to select an SSM template shape; standard rigid alignment still runs afterward."
      )
    else:
      self.optimizeButton.setText("Run Template Optimization")
      self.optimizeButton.setToolTip("Run the current FPFH + RANSAC template optimizer.")

  # ----- UI setup -----
  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    tabsWidget = qt.QTabWidget(); self.layout.addWidget(tabsWidget)
    prepareTab=qt.QWidget(); alignSingleTab=qt.QWidget(); alignMultiTab=qt.QWidget(); d2Tab=qt.QWidget(); advancedSettingsTab=qt.QWidget()
    tabsWidget.addTab(prepareTab, "Prepare")
    tabsWidget.addTab(alignSingleTab, "Single Run")
    tabsWidget.addTab(alignMultiTab, "Batch")
    tabsWidget.addTab(d2Tab, "Template Optimization")
    tabsWidget.addTab(advancedSettingsTab, "Advanced")
    prepareLayout = qt.QFormLayout(prepareTab)
    alignSingleTabLayout = qt.QFormLayout(alignSingleTab)
    alignMultiTabLayout  = qt.QFormLayout(alignMultiTab)
    d2Layout = qt.QFormLayout(d2Tab)
    advancedSettingsTabLayout = qt.QFormLayout(advancedSettingsTab)

    prepInfo = qt.QLabel(
      "Recommended flow:\n"
      "1) Ensure a valid SSM table is loaded (from Model Library)\n"
      "2) Run Single mode to tune parameters\n"
      "3) Optionally optimize template\n"
      "4) Run Batch mode for full dataset"
    )
    prepInfo.setWordWrap(True)
    prepareLayout.addRow(prepInfo)
    prepButtons = qt.QHBoxLayout()
    for name in ("Model Library", "Single Run", "Batch", "Template Optimization"):
      btn = qt.QPushButton(name)
      if name == "Model Library":
        btn.clicked.connect(lambda: slicer.util.selectModule("MorphoWeaveModelLibrary"))
      elif name == "Single Run":
        btn.clicked.connect(lambda: tabsWidget.setCurrentWidget(alignSingleTab))
      elif name == "Batch":
        btn.clicked.connect(lambda: tabsWidget.setCurrentWidget(alignMultiTab))
      else:
        btn.clicked.connect(lambda: tabsWidget.setCurrentWidget(d2Tab))
      prepButtons.addWidget(btn)
    prepareLayout.addRow(prepButtons)

    # --- Single Alignment ---
    single=ctk.ctkCollapsibleButton(); single.text="Align and subsample a source and target mesh"; singleL=qt.QFormLayout(single)
    alignSingleTabLayout.addRow(single)
    self.singlePrereqLabel = qt.QLabel("")
    self.singlePrereqLabel.setWordWrap(True)
    singleL.addRow(self.singlePrereqLabel)
    self.sourceModelSelector            = self._make_selector(["vtkMRMLModelNode"]) ; singleL.addRow("Template model:", self.sourceModelSelector)
    self.sourceFiducialSelector         = self._make_selector(["vtkMRMLMarkupsFiducialNode"]) ; singleL.addRow("Template correspondences:", self.sourceFiducialSelector)
    self.sourceSparseFiducialSelector   = self._make_selector(["vtkMRMLMarkupsFiducialNode"]) ; singleL.addRow("Template landmarks:", self.sourceSparseFiducialSelector)
    self.targetModelSelector            = self._make_selector(["vtkMRMLModelNode"], select_new=True) ; singleL.addRow("Target model:", self.targetModelSelector)
    self.ssmTableSelector               = self._make_selector(["vtkMRMLTableNode"], attr_key="ssm_eigenvalues", attr_val=None, tooltip="If you have an SSM table loaded, select it here to use PCA-CPD.") ; singleL.addRow("SSM Data Table:", self.ssmTableSelector)
    self.subsampleButton=qt.QPushButton("1) Subsample source/target"); self.subsampleButton.enabled=False; singleL.addRow(self.subsampleButton)
    self.subsampleInfo=qt.QPlainTextEdit(); self.subsampleInfo.setPlaceholderText("Subsampling information"); self.subsampleInfo.setReadOnly(True); singleL.addRow(self.subsampleInfo)
    self.alignButton=qt.QPushButton("2) Run rigid alignment"); self.alignButton.enabled=False; singleL.addRow(self.alignButton)
    self.displayMeshButton=qt.QPushButton("2b) Preview Rigid Alignment"); self.displayMeshButton.enabled=False; singleL.addRow(self.displayMeshButton)
    self.CPDRegistrationButton=qt.QPushButton("3) Run deformable alignment"); self.CPDRegistrationButton.enabled=False; singleL.addRow(self.CPDRegistrationButton)
    self.displayWarpedModelButton=qt.QPushButton("4) Show final registration"); self.displayWarpedModelButton.enabled=False; singleL.addRow(self.displayWarpedModelButton)
    self.resetButton=qt.QPushButton("Reset Single Run"); singleL.addRow(self.resetButton)


    # --- Batch Processing ---
    multi=ctk.ctkCollapsibleButton(); multi.text="Transfer landmark points from a source mesh to a directory of target meshes"; multiL=qt.QFormLayout(multi)
    alignMultiTabLayout.addRow(multi)
    self.batchPrereqLabel = qt.QLabel("")
    self.batchPrereqLabel.setWordWrap(True)
    multiL.addRow(self.batchPrereqLabel)
    self.sourceModelMultiSelector          = self._make_selector(["vtkMRMLModelNode"]) ; multiL.addRow("Source mesh:", self.sourceModelMultiSelector)
    self.sourceFiducialMultiSelector       = self._make_selector(["vtkMRMLMarkupsFiducialNode"]) ; multiL.addRow("Source correspondences:", self.sourceFiducialMultiSelector)
    self.sourceSparseFiducialMultiSelector = self._make_selector(["vtkMRMLMarkupsFiducialNode"]) ; multiL.addRow("Source landmarks:", self.sourceSparseFiducialMultiSelector)
    self.targetModelMultiSelector=ctk.ctkPathLineEdit(); self.targetModelMultiSelector.filters=ctk.ctkPathLineEdit.Dirs; multiL.addRow("Target mesh directory:", self.targetModelMultiSelector)
    self.landmarkOutputSelector=ctk.ctkPathLineEdit(); self.landmarkOutputSelector.filters=ctk.ctkPathLineEdit.Dirs; multiL.addRow("Target output landmark directory:", self.landmarkOutputSelector)
    self.saveWarpedMeshesBatchChk = qt.QCheckBox("Save warped meshes")
    self.saveWarpedMeshesBatchChk.setToolTip("Also save the final warped mesh for each target as a .vtp file.")
    multiL.addRow(self.saveWarpedMeshesBatchChk)
    self.meshOutputSelector = ctk.ctkPathLineEdit(); self.meshOutputSelector.filters=ctk.ctkPathLineEdit.Dirs; self.meshOutputSelector.enabled=False; multiL.addRow("Warped mesh output directory:", self.meshOutputSelector)
    self.smoothExportedMeshesBatchChk = qt.QCheckBox("Smooth exported warped meshes")
    self.smoothExportedMeshesBatchChk.setToolTip("Apply cosmetic smoothing only to the saved mesh files. Landmark predictions are unchanged.")
    self.smoothExportedMeshesBatchChk.enabled=False
    multiL.addRow(self.smoothExportedMeshesBatchChk)
    self.ssmTableMultiSelector = self._make_selector(["vtkMRMLTableNode"], attr_key="ssm_eigenvalues", attr_val=None, tooltip="Select the SSM table."); multiL.addRow("SSM Data Table:", self.ssmTableMultiSelector)
    self.skipOptBatchChk = qt.QCheckBox("Skip template optimization"); self.skipOptBatchChk.setToolTip("Skip the selected target-specific template initialization backend.");multiL.addRow(self.skipOptBatchChk)
    self.batchBackendLabel = qt.QLabel("")
    self.batchBackendLabel.setWordWrap(True)
    multiL.addRow(self.batchBackendLabel)
    self.applyLandmarkMultiButton=qt.QPushButton("Run Batch Auto-Landmarking"); self.applyLandmarkMultiButton.enabled=False; multiL.addRow(self.applyLandmarkMultiButton)
    self.batchProgress=qt.QProgressBar(); self.batchProgress.minimum=0; self.batchProgress.maximum=100; self.batchProgress.value=0; multiL.addRow("Progress:", self.batchProgress)
    self.batchCancelButton=qt.QPushButton("Cancel Batch Run"); self.batchCancelButton.enabled=False; multiL.addRow(self.batchCancelButton)
    self._cancelBatch=False; self.batchCancelButton.connect('clicked()', self.onCancelBatch)


    # --- Optimization Tab ---
    opt=ctk.ctkCollapsibleButton(); opt.text="Optimization Inputs"; optL=qt.QFormLayout(opt); d2Layout.addRow(opt)
    self.optimizePrereqLabel = qt.QLabel("")
    self.optimizePrereqLabel.setWordWrap(True)
    optL.addRow(self.optimizePrereqLabel)
    self.d2sourceModelSelector          = self._make_selector(["vtkMRMLModelNode"]); optL.addRow("Template model:", self.d2sourceModelSelector)
    self.d2sourceFiducialSelector       = self._make_selector(["vtkMRMLMarkupsFiducialNode"]); optL.addRow("Template correspondences:", self.d2sourceFiducialSelector)
    self.d2sourceSparseFiducialSelector = self._make_selector(["vtkMRMLMarkupsFiducialNode"]); optL.addRow("Template landmarks:", self.d2sourceSparseFiducialSelector)
    self.d2targetModelSelector          = self._make_selector(["vtkMRMLModelNode"], select_new=True); optL.addRow("Target model:", self.d2targetModelSelector)
    self.d2ssmTableSelector             = self._make_selector(["vtkMRMLTableNode"], attr_key="ssm_eigenvalues", attr_val=None); optL.addRow("SSM Data Table:", self.d2ssmTableSelector)
    self.optimizationBackendCombo = qt.QComboBox()
    self.optimizationBackendCombo.addItem("FPFH + RANSAC (current)")
    self.optimizationBackendCombo.addItem("Pose-marginalized EM (experimental)")
    self.optimizationBackendCombo.setToolTip("Select the target-specific template initialization method. The current method remains the default.")
    optL.addRow("Optimization backend:", self.optimizationBackendCombo)

    self.legacyOptimizationBox=ctk.ctkCollapsibleButton(); self.legacyOptimizationBox.text="FPFH + RANSAC settings"; self.legacyOptimizationBox.collapsed=True
    legacyOptL=qt.QFormLayout(self.legacyOptimizationBox); optL.addRow(self.legacyOptimizationBox)
    self.pcGridSteps=qt.QSpinBox(); self.pcGridSteps.minimum=2; self.pcGridSteps.maximum=35; self.pcGridSteps.value=4; self.pcGridSteps.setToolTip("Number of leading SSM modes searched by the current optimizer."); legacyOptL.addRow("SSM modes searched:", self.pcGridSteps)
    self.ransacItersPerCand=qt.QSpinBox(); self.ransacItersPerCand.minimum=10000; self.ransacItersPerCand.maximum=2000000; self.ransacItersPerCand.singleStep=10000; self.ransacItersPerCand.value=300000; legacyOptL.addRow("RANSAC iters (per candidate):", self.ransacItersPerCand)

    self.poseOptimizationBox=ctk.ctkCollapsibleButton(); self.poseOptimizationBox.text="Experimental pose-EM settings"; self.poseOptimizationBox.collapsed=True
    poseOptL=qt.QFormLayout(self.poseOptimizationBox); optL.addRow(self.poseOptimizationBox)
    poseHelp=qt.QLabel("The defaults below use MorphoWeave's rustcpd registration configuration: trajectory scoring, all coarse poses retained, and full-source refinement. Only the selected SSM shape is applied to the template; standard scaling, rigid alignment, and deformable registration still run afterward.")
    poseHelp.setWordWrap(True); poseOptL.addRow(poseHelp)
    self.poseRotationCount=qt.QSpinBox(); self.poseRotationCount.minimum=12; self.poseRotationCount.maximum=2048; self.poseRotationCount.value=193; self.poseRotationCount.setToolTip("Exact total hypothesis budget, including the identity and global/local rotation samples."); poseOptL.addRow("Total pose hypotheses:", self.poseRotationCount)
    self.poseCoarseSourceCount=qt.QSpinBox(); self.poseCoarseSourceCount.minimum=20; self.poseCoarseSourceCount.maximum=10000; self.poseCoarseSourceCount.value=400; poseOptL.addRow("Coarse source points:", self.poseCoarseSourceCount)
    self.poseCoarseTargetCount=qt.QSpinBox(); self.poseCoarseTargetCount.minimum=20; self.poseCoarseTargetCount.maximum=10000; self.poseCoarseTargetCount.value=400; poseOptL.addRow("Coarse target points:", self.poseCoarseTargetCount)
    self.poseCoarseRank=qt.QSpinBox(); self.poseCoarseRank.minimum=1; self.poseCoarseRank.maximum=100; self.poseCoarseRank.value=12; poseOptL.addRow("Coarse SSM rank:", self.poseCoarseRank)
    self.poseCoarseIterations=qt.QSpinBox(); self.poseCoarseIterations.minimum=1; self.poseCoarseIterations.maximum=100; self.poseCoarseIterations.value=8; poseOptL.addRow("Coarse EM iterations:", self.poseCoarseIterations)
    self.poseCoarseScreenIterations=qt.QSpinBox(); self.poseCoarseScreenIterations.minimum=1; self.poseCoarseScreenIterations.maximum=100; self.poseCoarseScreenIterations.value=8; self.poseCoarseScreenIterations.setToolTip("Iterations completed before optional screening; equal to coarse iterations by default, so every pose completes the coarse stage."); poseOptL.addRow("Screen after iteration:", self.poseCoarseScreenIterations)
    self.poseCoarseSurvivorCount=qt.QSpinBox(); self.poseCoarseSurvivorCount.minimum=1; self.poseCoarseSurvivorCount.maximum=2048; self.poseCoarseSurvivorCount.value=193; self.poseCoarseSurvivorCount.setToolTip("Number of hypotheses retained after screening. The default retains the entire 193-pose budget."); poseOptL.addRow("Coarse survivors:", self.poseCoarseSurvivorCount)
    self.poseCoarseScoreMode=qt.QComboBox(); self.poseCoarseScoreMode.addItem("Trajectory (recommended)"); self.poseCoarseScoreMode.addItem("Final iteration"); self.poseCoarseScoreMode.setToolTip("Rank coarse poses by their EM objective trajectory or only their final objective."); poseOptL.addRow("Coarse scoring:", self.poseCoarseScoreMode)
    self.poseRefineCount=qt.QSpinBox(); self.poseRefineCount.minimum=1; self.poseRefineCount.maximum=64; self.poseRefineCount.value=12; poseOptL.addRow("Pose finalists:", self.poseRefineCount)
    self.poseRefineSourceCount=qt.QSpinBox(); self.poseRefineSourceCount.minimum=0; self.poseRefineSourceCount.maximum=50000; self.poseRefineSourceCount.value=0; self.poseRefineSourceCount.setSpecialValueText("Full source"); self.poseRefineSourceCount.setToolTip("Use 0 for all source points, matching the recommended default."); poseOptL.addRow("Refinement source points:", self.poseRefineSourceCount)
    self.poseRefineTargetCount=qt.QSpinBox(); self.poseRefineTargetCount.minimum=50; self.poseRefineTargetCount.maximum=50000; self.poseRefineTargetCount.value=1600; poseOptL.addRow("Refinement target points:", self.poseRefineTargetCount)
    self.poseRefineIterations=qt.QSpinBox(); self.poseRefineIterations.minimum=1; self.poseRefineIterations.maximum=250; self.poseRefineIterations.value=30; poseOptL.addRow("Refinement EM iterations:", self.poseRefineIterations)
    self.poseLambdaReg=qt.QDoubleSpinBox(); self.poseLambdaReg.minimum=0.0; self.poseLambdaReg.maximum=20.0; self.poseLambdaReg.singleStep=0.10; self.poseLambdaReg.decimals=3; self.poseLambdaReg.value=5.0; self.poseLambdaReg.setToolTip("Pose-initializer SSM regularization; independent of downstream PCA-CPD settings."); poseOptL.addRow("Pose SSM weight:", self.poseLambdaReg)
    self.poseOutlierWeight=qt.QDoubleSpinBox(); self.poseOutlierWeight.minimum=0.0; self.poseOutlierWeight.maximum=0.99; self.poseOutlierWeight.singleStep=0.01; self.poseOutlierWeight.decimals=2; self.poseOutlierWeight.value=0.15; self.poseOutlierWeight.setToolTip("Pose-initializer outlier weight; independent of downstream PCA-CPD settings."); poseOptL.addRow("Pose outlier weight:", self.poseOutlierWeight)
    self.poseIdentityPrior=qt.QDoubleSpinBox(); self.poseIdentityPrior.minimum=0.01; self.poseIdentityPrior.maximum=0.99; self.poseIdentityPrior.singleStep=0.05; self.poseIdentityPrior.decimals=2; self.poseIdentityPrior.value=0.20; poseOptL.addRow("Identity-pose prior:", self.poseIdentityPrior)
    self.poseSeed=qt.QSpinBox(); self.poseSeed.minimum=0; self.poseSeed.maximum=2147483647; self.poseSeed.value=0; poseOptL.addRow("Deterministic seed:", self.poseSeed)
    self.poseParallel=qt.QCheckBox(); self.poseParallel.checked=True; self.poseParallel.setToolTip("Use rustcpd's deterministic internal parallel pool for pose hypotheses."); poseOptL.addRow("Parallel pose search:", self.poseParallel)
    self.optimizeButton=qt.QPushButton("Run Template Optimization"); optL.addRow(self.optimizeButton)

    # --- Optimization Tab (append this) ---
    self.diagText = qt.QPlainTextEdit()
    self.diagText.setReadOnly(True)
    self.diagText.setPlaceholderText("Template optimization diagnostics will appear here…")
    optL.addRow(self.diagText)

    def _log_opt(msg):
        if hasattr(self, "diagText"):
            self.diagText.appendPlainText(str(msg))
        slicer.util.showStatusMessage(str(msg), 3000)
    self._log_opt = _log_opt  # keep a handle


    # --- Advanced Settings (declarative) ---
    self._build_advanced_tab(advancedSettingsTabLayout)


    # Sync Batch checkbox <-> Advanced param widget
    self.skipOptBatchChk.setChecked(bool(self.param_skipOptimization.checked))
    self.skipOptBatchChk.toggled.connect(
        lambda v: (setattr(self.param_skipOptimization, "checked", v), self._on_param_changed())
    )
    self.param_skipOptimization.toggled.connect(
        lambda v: self.skipOptBatchChk.setChecked(v)
    )

    # --- Signals ---
    self.sourceModelSelector.currentNodeChanged.connect(self.onSelect)
    self.sourceFiducialSelector.currentNodeChanged.connect(self.onSelect)
    self.sourceSparseFiducialSelector.currentNodeChanged.connect(self.onSelect)
    self.targetModelSelector.currentNodeChanged.connect(self.onSelect)
    self.ssmTableSelector.currentNodeChanged.connect(self.onSelect)
    self.subsampleButton.clicked.connect(self.onSubsampleButton)
    self.alignButton.clicked.connect(self.onAlignButton)
    self.displayMeshButton.clicked.connect(self.onDisplayMeshButton)
    self.CPDRegistrationButton.clicked.connect(self.onCPDRegistration)
    self.displayWarpedModelButton.clicked.connect(self.onDisplayWarpedModel)
    self.resetButton.clicked.connect(self.onResetButton)

    self.sourceModelMultiSelector.currentNodeChanged.connect(self.onSelectMultiProcess)
    self.sourceFiducialMultiSelector.currentNodeChanged.connect(self.onSelectMultiProcess)
    self.sourceSparseFiducialMultiSelector.currentNodeChanged.connect(self.onSelectMultiProcess)
    self.targetModelMultiSelector.validInputChanged.connect(self.onSelectMultiProcess)
    self.landmarkOutputSelector.validInputChanged.connect(self.onSelectMultiProcess)
    self.saveWarpedMeshesBatchChk.toggled.connect(self._onBatchMeshExportToggled)
    self.meshOutputSelector.validInputChanged.connect(self.onSelectMultiProcess)
    self.applyLandmarkMultiButton.clicked.connect(self.onApplyLandmarkMulti)
    self.ssmTableMultiSelector.currentNodeChanged.connect(self.onSelectMultiProcess)
    self.d2sourceModelSelector.currentNodeChanged.connect(self._enable_optimize)
    self.d2sourceFiducialSelector.currentNodeChanged.connect(self._enable_optimize)
    self.d2targetModelSelector.currentNodeChanged.connect(self._enable_optimize)
    self.d2ssmTableSelector.currentNodeChanged.connect(self._enable_optimize)
    self.optimizationBackendCombo.currentIndexChanged.connect(self._onOptimizationBackendChanged)

    self.optimizeButton.clicked.connect(self.onOptimize)

    self._autoSelectCanonicalSsmSet()
    self.onSelect(); self.onSelectMultiProcess(); self._enable_optimize(); self.parameterDictionary=self._read_params(); self._onOptimizationBackendChanged()
    self.layout.addStretch(1)

  # ----- Small helpers -----
  @staticmethod
  def canonicalSsmNodeNames(databaseName):
    return canonical_ssm_node_names(databaseName)

  def _sceneNodesByClass(self, className):
    collection = slicer.mrmlScene.GetNodesByClass(className)
    collection.UnRegister(None)
    return [collection.GetItemAsObject(index) for index in range(collection.GetNumberOfItems())]

  def _latestCompleteSsmSet(self):
    return latest_complete_ssm_set(
      self._sceneNodesByClass("vtkMRMLTableNode"),
      self._sceneNodesByClass("vtkMRMLModelNode"),
      self._sceneNodesByClass("vtkMRMLMarkupsFiducialNode"),
    )

  def _autoSelectCanonicalSsmSet(self):
    ssmSet = self._latestCompleteSsmSet()
    if not ssmSet:
      return False
    selectorNodes = (
      (self.sourceModelSelector, ssmSet["model"]),
      (self.sourceFiducialSelector, ssmSet["dense"]),
      (self.sourceSparseFiducialSelector, ssmSet["sparse"]),
      (self.ssmTableSelector, ssmSet["table"]),
      (self.sourceModelMultiSelector, ssmSet["model"]),
      (self.sourceFiducialMultiSelector, ssmSet["dense"]),
      (self.sourceSparseFiducialMultiSelector, ssmSet["sparse"]),
      (self.ssmTableMultiSelector, ssmSet["table"]),
      (self.d2sourceModelSelector, ssmSet["model"]),
      (self.d2sourceFiducialSelector, ssmSet["dense"]),
      (self.d2sourceSparseFiducialSelector, ssmSet["sparse"]),
      (self.d2ssmTableSelector, ssmSet["table"]),
    )
    blockers = []
    try:
      for selector, node in selectorNodes:
        if selector.currentNode() is None:
          blockers.append(qt.QSignalBlocker(selector))
      changed = fill_empty_selectors(selectorNodes)
    finally:
      blockers.clear()
    return changed

  def enter(self):
    self._autoSelectCanonicalSsmSet()
    self.onSelect()
    self.onSelectMultiProcess()
    self._enable_optimize()

  def updateLayout(self, focusNode=None, keepOrientation=True, margin=1.15):
      lm = slicer.app.layoutManager()
      if lm.layout != 9:
          lm.setLayout(9)
      if focusNode is not None:
          self._focusOnNode(focusNode, keepOrientation=keepOrientation, margin=margin)

  def _enable_single(self):
    ready = all([self.sourceModelSelector.currentNode(), self.targetModelSelector.currentNode(),
                 self.sourceFiducialSelector.currentNode(), self.sourceSparseFiducialSelector.currentNode()])
    self.subsampleButton.enabled = ready
    has_ssm = self.ssmTableSelector.currentNode() is not None
    if not ready:
      setWorkflowStatus(self.singlePrereqLabel, "needs_input", "Select template model, correspondences, sparse landmarks, and target model.")
    elif not has_ssm:
      setWorkflowStatus(self.singlePrereqLabel, "optional", "Rigid stage is ready. Select an SSM Data Table for deformable alignment.")
    else:
      setWorkflowStatus(self.singlePrereqLabel, "ready", "Single-run prerequisites are available.")

  def _enable_batch(self):
    needs_mesh_output = bool(self.saveWarpedMeshesBatchChk.checked)
    ready = all([self.sourceModelMultiSelector.currentNode(), self.sourceFiducialMultiSelector.currentNode(),
                 self.sourceSparseFiducialMultiSelector.currentNode(), self.ssmTableMultiSelector.currentNode(),
                 self.targetModelMultiSelector.currentPath, self.landmarkOutputSelector.currentPath]) and (
                 (not needs_mesh_output) or bool(self.meshOutputSelector.currentPath))
    self.applyLandmarkMultiButton.enabled = ready
    if ready:
      setWorkflowStatus(self.batchPrereqLabel, "ready", "Batch prerequisites are available.")
    elif needs_mesh_output:
      setWorkflowStatus(self.batchPrereqLabel, "needs_input", "Select source nodes, SSM table, target directory, landmark output directory, and warped mesh output directory.")
    else:
      setWorkflowStatus(self.batchPrereqLabel, "needs_input", "Select source nodes, SSM table, target directory, and output directory.")

  def _onBatchMeshExportToggled(self, checked):
    enabled = bool(checked)
    self.meshOutputSelector.enabled = enabled
    self.smoothExportedMeshesBatchChk.enabled = enabled
    self._enable_batch()

  def _enable_optimize(self):
    ready = all([
      self.d2sourceModelSelector.currentNode(),
      self.d2sourceFiducialSelector.currentNode(),
      self.d2targetModelSelector.currentNode(),
      self.d2ssmTableSelector.currentNode()
    ])
    self.optimizeButton.enabled = ready
    if ready:
      setWorkflowStatus(self.optimizePrereqLabel, "ready", "Optimization prerequisites are available.")
    else:
      setWorkflowStatus(self.optimizePrereqLabel, "needs_input", "Select template model, correspondences, target model, and SSM table.")

  def ensureCloneFolder(self, label=None):
    shNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSubjectHierarchyNode")
    if not shNode: raise RuntimeError("SubjectHierarchy node not found")
    invalid = shNode.GetInvalidItemID() if hasattr(shNode, "GetInvalidItemID") else -1
    rootName = "MorphoWeave Landmark Transfer runs"; rootID = shNode.GetItemByName(rootName)
    if rootID in (None, invalid): rootID = shNode.CreateFolderItem(shNode.GetSceneItemID(), rootName)
    if getattr(self, "cloneFolderItemID", None) not in (None, invalid): return self.cloneFolderItemID
    from datetime import datetime
    runName = f"Run {datetime.now().strftime('%Y%m%d-%H%M%S')}";
    if label: runName += f" - {label}"
    self.cloneFolderItemID = shNode.CreateFolderItem(rootID, runName)
    return self.cloneFolderItemID

  def cloneNode(self, originalModelNode, customName=None):
    shNode  = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSubjectHierarchyNode")
    shLogic = slicer.modules.subjecthierarchy.logic()
    originalItemID = shNode.GetItemByDataNode(originalModelNode)
    newItemID      = shLogic.CloneSubjectHierarchyItem(shNode, originalItemID)
    cloneNode      = shNode.GetItemDataNode(newItemID)
    cloneNode.SetName(customName or (originalModelNode.GetName()+"_clone"))
    self.ensureCloneFolder(); shNode.SetItemParent(newItemID, self.cloneFolderItemID)
    return cloneNode

  def _parentNode(self, mrmlNode):
    shNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSubjectHierarchyNode")
    if not shNode: raise RuntimeError("SubjectHierarchy node not found")
    self.ensureCloneFolder(); invalid = shNode.GetInvalidItemID() if hasattr(shNode, "GetInvalidItemID") else -1
    itemID = shNode.GetItemByDataNode(mrmlNode)
    if itemID == invalid or itemID is None: shNode.CreateItem(self.cloneFolderItemID, mrmlNode)
    else: shNode.SetItemParent(itemID, self.cloneFolderItemID)

  def clearCloneFolder(self):
    shNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSubjectHierarchyNode")
    if not shNode: return
    if getattr(self, "cloneFolderItemID", None): shNode.RemoveItem(self.cloneFolderItemID)
    self.cloneFolderItemID = None

  def _hideInputNodes(self):
    for selector in (self.sourceModelSelector, self.sourceFiducialSelector, self.sourceSparseFiducialSelector, self.targetModelSelector):
      node = selector.currentNode()
      if node and node.GetDisplayNode(): node.GetDisplayNode().SetVisibility(False)

  def _showInputNodes(self, visible=True):
    for selector in (self.sourceModelSelector, self.sourceFiducialSelector, self.sourceSparseFiducialSelector, self.targetModelSelector):
      n = selector.currentNode()
      if n and n.GetDisplayNode(): n.GetDisplayNode().SetVisibility(visible)

  def onSelect(self, node=None):
    src = self.sourceModelSelector.currentNode(); tgt = self.targetModelSelector.currentNode()
    lm  = self.sourceSparseFiducialSelector.currentNode(); slm = self.sourceFiducialSelector.currentNode()
    self._enable_single()
    has_table = self.ssmTableSelector.currentNode() is not None
    if not has_table:
      self.CPDRegistrationButton.setToolTip("Select an SSM Data Table before running deformable alignment.")
      if not self.CPDRegistrationButton.enabled:
        self.CPDRegistrationButton.enabled = False
    else:
      self.CPDRegistrationButton.setToolTip("")
    if src and self.sourceModelMultiSelector.currentNode() is None:
      self.sourceModelMultiSelector.setCurrentNode(src)
    if lm and self.sourceSparseFiducialMultiSelector.currentNode() is None:
      self.sourceSparseFiducialMultiSelector.setCurrentNode(lm)
    if slm and self.sourceFiducialMultiSelector.currentNode() is None:
      self.sourceFiducialMultiSelector.setCurrentNode(slm)

  def onCancelBatch(self):
    self._cancelBatch = True; self.batchCancelButton.enabled = False
    slicer.util.showStatusMessage("Cancelling batch…", 2000)


  def onSelectMultiProcess(self): self._enable_batch()

  def _proj_frac(self):
    d=self.parameterDictionary; return 0.0 if d.get("skipProjection", False) else float(d.get("projectionFactor",1.0))/100.0

  # ----- Single alignment flow -----
  def onSubsampleButton(self):
    logic = MorphoWeaveLandmarkTransferLogic(); self._hideInputNodes()
    srcOrig = self.sourceModelSelector.currentNode(); tgtOrig = self.targetModelSelector.currentNode()
    corresOrig  = self.sourceFiducialSelector.currentNode(); lmOrig  = self.sourceSparseFiducialSelector.currentNode()
    self.clearCloneFolder(); run_label = f"{srcOrig.GetName()}→{tgtOrig.GetName()}"; self.ensureCloneFolder(label=run_label)
    self.srcTemp = self.cloneNode(srcOrig, customName="Source")
    self.tgtTemp = self.cloneNode(tgtOrig, customName="Target")
    self.corresTemp  = self.cloneNode(corresOrig,  customName="Source correspondences")
    self.lmTemp  = self.cloneNode(lmOrig,  customName="Landmarks")
    if not self.parameterDictionary.get("skipScaling", False):
        cov=float(self.parameterDictionary.get("targetCoverage",1.0))
        cov=float(np.clip(cov, 1e-3, 1.0))
        size_src=bounds_diag(self.srcTemp); size_tgt=bounds_diag(self.tgtTemp)
        size_tgt_eff = size_tgt / cov   # “expected full” linear size
        self.scale = (size_tgt_eff/size_src) if size_src>0 else 1.0
        t = vtk.vtkTransform(); t.Scale(self.scale, self.scale, self.scale)
        tn = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode','PreScale'); tn.SetAndObserveTransformToParent(t)
        for node in (self.srcTemp, self.corresTemp, self.lmTemp): node.SetAndObserveTransformNodeID(tn.GetID()); slicer.vtkSlicerTransformLogic().hardenTransform(node)
        slicer.mrmlScene.RemoveNode(tn)
    else: self.scale = 1.0
    _, _, self.sourcePoints, self.targetPoints, self.sourceFeatures, self.targetFeatures, self.voxelSize = \
      logic.runSubsample(self.srcTemp, self.tgtTemp, self.parameterDictionary.get("skipScaling", False), self.parameterDictionary)
    if self.sourcePoints is None or self.targetPoints is None:
      slicer.util.errorDisplay("Subsampling failed. Check input files and logs."); return
    src_np = np.asarray(self.sourcePoints.points); tgt_np = np.asarray(self.targetPoints.points)
    self.sourceSLM_vtk = logic.convertPointsToVTK(src_np); self.targetSLM_vtk = logic.convertPointsToVTK(tgt_np)
    self.targetCloudNode = logic.displayPointCloud(self.targetSLM_vtk, 'Target Pointcloud', COLORS["blue"], frac=0.004, refNode=self.tgtTemp)
    self._parentNode(self.targetCloudNode); self.updateLayout(); self.alignButton.enabled = True
    self.subsampleInfo.clear(); self.subsampleInfo.insertPlainText(f':: Your subsampled source pointcloud has {len(src_np)} points.\n')
    self.subsampleInfo.insertPlainText(f':: Your subsampled target pointcloud has {len(tgt_np)} points.')
    self.updateLayout(focusNode=self.tgtTemp)


  def onAlignButton(self):
    logic = MorphoWeaveLandmarkTransferLogic()
    self.transformMatrix = logic.estimateTransform(self.sourcePoints, self.targetPoints, self.sourceFeatures, self.targetFeatures, self.voxelSize, self.parameterDictionary)
    self.ICPTransformNode = logic.convertMatrixToTransformNode(self.transformMatrix, 'Rigid Transformation Matrix')
    self._parentNode(self.ICPTransformNode)
    self.sourceCloudNode = logic.displayPointCloud(self.sourceSLM_vtk, 'Source Pointcloud', COLORS["red"],  frac=0.004, refNode=self.tgtTemp)
    self.sourceCloudNode.SetAndObserveTransformNodeID(self.ICPTransformNode.GetID()); slicer.vtkSlicerTransformLogic().hardenTransform(self.sourceCloudNode)
    self._parentNode(self.sourceCloudNode); self.displayMeshButton.enabled = True; self.alignButton.enabled = False


  def onDisplayMeshButton(self):
    self.tgtTemp.CreateDefaultDisplayNodes(); self.tgtTemp.GetDisplayNode().SetColor(*COLORS["blue"]); self.tgtTemp.GetDisplayNode().SetVisibility(True)
    self.srcTemp.CreateDefaultDisplayNodes(); self.srcTemp.SetAndObserveTransformNodeID(self.ICPTransformNode.GetID()); slicer.vtkSlicerTransformLogic().hardenTransform(self.srcTemp)
    self.srcTemp.GetDisplayNode().SetColor(*COLORS["red"]); self.srcTemp.GetDisplayNode().SetVisibility(True)
    self.sourceCloudNode.GetDisplayNode().SetVisibility(False); self.targetCloudNode.GetDisplayNode().SetVisibility(False)
    self.corresTemp.SetAndObserveTransformNodeID(self.ICPTransformNode.GetID()); slicer.vtkSlicerTransformLogic().hardenTransform(self.corresTemp); self.corresTemp.GetDisplayNode().SetVisibility(False)
    self.lmTemp.SetAndObserveTransformNodeID(self.ICPTransformNode.GetID()); slicer.vtkSlicerTransformLogic().hardenTransform(self.lmTemp); self.lmTemp.GetDisplayNode().SetVisibility(True)
    has_table = self.ssmTableSelector.currentNode() is not None
    self.CPDRegistrationButton.enabled = has_table
    if not has_table:
      self.CPDRegistrationButton.setToolTip("Select an SSM Data Table before running deformable alignment.")
    self.displayMeshButton.enabled = False


  def onCPDRegistration(self):
      logic = MorphoWeaveLandmarkTransferLogic()

      alignedSourceCorres_np = slicer.util.arrayFromMarkupsControlPoints(self.corresTemp)
      alignedSourceLM_np     = slicer.util.arrayFromMarkupsControlPoints(self.lmTemp)

      tableNode = self.ssmTableSelector.currentNode()
      if tableNode is None:
          slicer.util.errorDisplay("Select an SSM Data Table before running the deformable step.")
          return
      logic.rigidTransformNode = self.ICPTransformNode

      # SSM-guided deformable (PCA-CPD)
      deformedLandmark_np = logic.runDeformable(
          tableNode=tableNode,
          sourceLM=alignedSourceCorres_np,
          scale=self.scale,
          targetSLM=self.targetPoints.points,
          parameters=self.parameterDictionary
      )

      # Preview warped correspondences
      if getattr(self, "deformedCloudNode", None):
          try: slicer.mrmlScene.RemoveNode(self.deformedCloudNode)
          except: pass
      pc_vtk = logic.convertPointsToVTK(deformedLandmark_np)
      self.deformedCloudNode = logic.displayPointCloud(pc_vtk, "Warped Source Pointcloud", COLORS["green"], frac=0.004, refNode=self.tgtTemp)

      self.deformedCloudNode.GetDisplayNode().SetVisibility(False)
      self._parentNode(self.deformedCloudNode)

      # Choose deformation backend
      use_bih = bool(self.parameterDictionary.get("useBiharmonic", False))

      self.warpedMeshNode = self.cloneNode(self.srcTemp, customName="Warped Source")
      self.warpedLM       = self.cloneNode(self.lmTemp,  customName="Landmark Predictions")

      if use_bih:
          # Experimental surface biharmonic + barycentric LM transfer
          lam = float(self.parameterDictionary.get("bih_lam", 1e4))
          V0, V1, F = logic.warp_model_biharmonic(
              modelNode=self.warpedMeshNode,
              src_corr=alignedSourceCorres_np,
              dst_corr=deformedLandmark_np,
              lam=lam
          )
          logic.warp_markups_barycentric(self.warpedLM, V0, V1, F)
      else:
          # Default: stable TPS warp (optionally smoothed, constraint-capped)
          lam_tps  = float(self.parameterDictionary.get("tpsLambda", 0.0))
          max_corr = int(self.parameterDictionary.get("tpsMaxCorr", 800))
          logic.warp_model_tps(
              modelNode=self.warpedMeshNode,
              src_corr=alignedSourceCorres_np,
              dst_corr=deformedLandmark_np,
              lam=lam_tps,
              max_corr=max_corr,
              seed=0,
              landmarksNode=self.warpedLM
          )

      # Optional smoothing for visual niceness
      logic.smoothModel(self.warpedMeshNode)

      # Respect Skip projection
      proj = self._proj_frac()
      if proj > 0.0:
          self.refinedLM = logic.runPointProjection(
              template=self.warpedMeshNode, model=self.tgtTemp,
              templateLandmarks=self.warpedLM, maxProjectionFactor=proj
          )
          logic.propagateLandmarkTypes(self.sourceSparseFiducialSelector.currentNode(), self.refinedLM)
          self._parentNode(self.refinedLM)
          self.warpedLM.GetDisplayNode().SetVisibility(False)
          self.refinedLM.GetDisplayNode().SetVisibility(True)
      else:
          self.refinedLM = self.warpedLM
          try:
              self.refinedLM.SetName("Predicted Landmarks (no projection)")
              self.refinedLM.SetLocked(True)
              self.refinedLM.SetFixedNumberOfControlPoints(True)
          except: pass
          self._parentNode(self.refinedLM)
          self.refinedLM.GetDisplayNode().SetVisibility(True)

      self.warpedMeshNode.GetDisplayNode().SetColor(*COLORS["green"])
      self.warpedMeshNode.GetDisplayNode().SetVisibility(False)
      self.lmTemp.GetDisplayNode().SetVisibility(False)
      self.srcTemp.GetDisplayNode().SetVisibility(False)

      self.displayWarpedModelButton.enabled = True
      self.CPDRegistrationButton.enabled = False

  def onDisplayWarpedModel(self):
    self.warpedMeshNode.GetDisplayNode().SetColor(*COLORS["green"]); self.warpedMeshNode.GetDisplayNode().SetVisibility(True)

  def onResetButton(self):
    self.clearCloneFolder(); self._showInputNodes(True)
    for btn in (self.subsampleButton, self.alignButton, self.displayMeshButton, self.CPDRegistrationButton, self.displayWarpedModelButton, self.applyLandmarkMultiButton): btn.enabled=False
    self.onSelect(); slicer.util.showStatusMessage("Reset complete: cleared previous runs", 2000)
    if hasattr(self, 'diagText'):
      self.diagText.setPlainText("No diagnostics yet.")

  # ----- Batch processing -----
  def onApplyLandmarkMulti(self):
    logic = MorphoWeaveLandmarkTransferLogic(); projectionFactor = self._proj_frac(); d=self.parameterDictionary
    self._cancelBatch=False; self.batchProgress.setValue(0); self.batchCancelButton.enabled=True
    last_ui_pump = [0.0]
    last_status = [0.0]
    def progress_cb(done, total, label=None):
      pct = int(100.0 * done / max(1, total)); self.batchProgress.setValue(pct)
      now = time.monotonic()
      if label and (done == total or (now - last_status[0] >= 0.10)):
        slicer.util.showStatusMessage(f"Landmark Transfer batch: {label}", 1500)
        last_status[0] = now
      if done == total or (now - last_ui_pump[0] >= 0.10):
        slicer.app.processEvents()
        last_ui_pump[0] = now
    def status_cb(label):
      now = time.monotonic()
      if now - last_status[0] >= 0.10:
        slicer.util.showStatusMessage(label, 1500)
        last_status[0] = now
      if now - last_ui_pump[0] >= 0.10:
        slicer.app.processEvents()
        last_ui_pump[0] = now
    def cancel_cb(): return self._cancelBatch
    qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
    try:
      batchParameters = self._optimization_parameters()
      saveWarpedMeshes = bool(self.saveWarpedMeshesBatchChk.checked)
      meshOutputDir = self.meshOutputSelector.currentPath if saveWarpedMeshes else ""
      smoothExportedMesh = bool(self.smoothExportedMeshesBatchChk.checked)

      logic.runLandmarkBatch(
        sourceModelNode=self.sourceModelMultiSelector.currentNode(),
        sourceCorrNode=self.sourceFiducialMultiSelector.currentNode(),
        sourceLMNode=self.sourceSparseFiducialMultiSelector.currentNode(),
        targetModelDir=self.targetModelMultiSelector.currentPath,
        outputDir=self.landmarkOutputSelector.currentPath,
        skipScaling=d.get("skipScaling", False),
        projectionFactor=projectionFactor,
        parameters=batchParameters,
        tableNode=self.ssmTableMultiSelector.currentNode(),
        saveWarpedMeshes=saveWarpedMeshes,
        meshOutputDir=meshOutputDir,
        smoothExportedMesh=smoothExportedMesh,
        progress_callback=progress_cb,
        status_callback=status_cb,
        cancel_callback=cancel_cb
      )
    finally:
      qt.QApplication.restoreOverrideCursor(); self.batchCancelButton.enabled=False
      if not self._cancelBatch:
        self.batchProgress.setValue(100); slicer.util.showStatusMessage("Batch processing complete.", 3000)
      else:
        slicer.util.showStatusMessage("Batch cancelled.", 3000)

  # ----- Optimization -----
  def onOptimize(self):
      logic = MorphoWeaveLandmarkTransferLogic()
      tplModelOrig = self.d2sourceModelSelector.currentNode()
      tplCorrOrig  = self.d2sourceFiducialSelector.currentNode()
      tplLandOrig  = self.d2sourceSparseFiducialSelector.currentNode()
      targetNode   = self.d2targetModelSelector.currentNode()
      tableNode    = self.d2ssmTableSelector.currentNode()
      if not all([tplModelOrig, tplCorrOrig, targetNode, tableNode]):
          slicer.util.errorDisplay("Select template model/correspondences, target, and SSM table.")
          return

      label = f"{tplModelOrig.GetName()}→{targetNode.GetName()}"
      self.clearCloneFolder()
      self.ensureCloneFolder(label=label)

      parameters = self._optimization_parameters()
      backend = optimization_backend(parameters)
      prefix = "PoseEM" if backend == POSE_EM_BACKEND else "GridRANSAC"
      tplModel = self.cloneNode(tplModelOrig, f"{prefix}_TemplateModel")
      tplCorr  = self.cloneNode(tplCorrOrig,  f"{prefix}_TemplateCorrespondences")
      tplLand  = self.cloneNode(tplLandOrig,  f"{prefix}_TemplateLandmarks") if tplLandOrig else None

      qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
      try:
          if backend == POSE_EM_BACKEND:
              result = logic.initialize_template_pose_em(
                  tableNode, tplModel, tplCorr, tplLand, targetNode,
                  parameters=parameters,
              )
              b = result.coefficients
          else:
              k = int(self.pcGridSteps.value)
              b, _ = logic.initialize_template(
                  tableNode, tplModel, tplCorr, tplLand, targetNode,
                  parameters=parameters,
                  k=k, span=3.0, optimizer="powell",
                  max_evals=240,
                  eval_ransac_iters=int(self.ransacItersPerCand.value * 0.2),
                  final_ransac_iters=int(self.ransacItersPerCand.value),
                  seed=0
              )
      except Exception as e:
          slicer.util.errorDisplay(f"Template optimization failed:\n{e}")
          return
      finally:
          qt.QApplication.restoreOverrideCursor()


      self._parentNode(tplModel)
      self._parentNode(tplCorr)
      if tplLand: self._parentNode(tplLand)

      # Park results under a folder
      sh = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSubjectHierarchyNode")
      resRoot = sh.GetItemByName("Template optimization") or sh.CreateFolderItem(sh.GetSceneItemID(), "Template optimization")
      for n in [tplModel, tplCorr] + ([tplLand] if tplLand else []):
          sh.SetItemParent(sh.GetItemByDataNode(n), resRoot)

      # Report what happened
      info = getattr(logic, "last_opt_info", {}) or {}
      dec  = info.get("decision", "unknown")
      fit  = info.get("fit", None)
      rmse = info.get("rmse", None)
      nb   = info.get("norm_b", None)

      if dec == "pose_em_template_chosen":
          self._log_opt(
              "[pose-em] Selected template shape "
              f"(||b||={nb:.3f}, scale={info['scale']:.4g}, size ratio={info['size_ratio']:.3f}, "
              f"score={info['score']:.3f}, margin={info['score_margin']:.3f}, "
              f"effective poses={info['effective_hypotheses']:.2f}, "
              f"evaluated/refined={info['hypotheses_evaluated']}/{info['hypotheses_refined']}, "
              f"parallel={'yes' if info['pose_parallel'] else 'no'})."
          )
          if info["effective_hypotheses"] >= 2.0:
              self._log_opt(
                "[pose-em] Ambiguity warning: multiple pose hypotheses retain substantial support; inspect orientation before batch use."
              )
      elif dec == "baseline_good":
          self._log_opt(f"[opt] Baseline template was already good (fit={fit:.3f}, rmse_n={rmse:.3f}).")
      elif dec == "candidate_chosen":
          self._log_opt(f"[opt] New template chosen from SSM (||b||={nb:.3f}, fit={fit:.3f}, rmse_n={rmse:.3f}).")
      elif dec == "baseline_kept":
          self._log_opt(f"[opt] Baseline kept after search (fit={fit:.3f}, rmse_n={rmse:.3f}).")
      else:
          self._log_opt("[opt] Optimization finished (details unavailable).")

      # Hand the optimized, template-frame shape to Single Run. The regular
      # scaling, rigid, and deformable stages remain unchanged for both
      # optimization backends.
      self.sourceModelSelector.setCurrentNode(tplModel)
      self.sourceFiducialSelector.setCurrentNode(tplCorr)
      if tplLand: self.sourceSparseFiducialSelector.setCurrentNode(tplLand)


# ---------- Logic ----------
class MorphoWeaveLandmarkTransferLogic(ScriptedLoadableModuleLogic):
  def __init__(self):
    super().__init__()
    self.rigidTransformNode=None
    self._ssm_cache={}

  def runLandmarkBatch(self, sourceModelNode, sourceCorrNode, sourceLMNode, targetModelDir, outputDir, skipScaling, projectionFactor, parameters, tableNode, saveWarpedMeshes=False, meshOutputDir="", smoothExportedMesh=False, progress_callback=None, status_callback=None, cancel_callback=None):
    import tiny3d as t3d
    t3d.utility.random.seed(int(42))
    parameters = dict(parameters)
    parameters["skipScaling"] = bool(skipScaling)
    skipOpt = bool(parameters.get("skipOptimization", False))
    use_bih = bool(parameters.get("useBiharmonic", False))
    lam_bih = float(parameters.get("bih_lam", 1e4))
    lam_tps = float(parameters.get("tpsLambda", 0.0))
    max_corr = int(parameters.get("tpsMaxCorr", 800))
    projection_enabled = bool(projectionFactor and projectionFactor > 0)
    need_final_mesh = bool(saveWarpedMeshes or projection_enabled)
    need_runtime_nodes = bool(need_final_mesh or use_bih)
    need_template_nodes = bool((not skipOpt) or need_runtime_nodes)
    ssmData = self._get_ssm_base(tableNode) if tableNode is not None else None

    if tableNode is None: raise ValueError("Batch requires an SSM table.")
    if sourceCorrNode is None: raise ValueError("Batch requires template correspondences.")
    if sourceLMNode   is None: raise ValueError("Batch requires template landmarks.")
    os.makedirs(outputDir, exist_ok=True)
    if saveWarpedMeshes:
      if not meshOutputDir:
        raise ValueError("Batch mesh export requires a warped mesh output directory.")
      os.makedirs(meshOutputDir, exist_ok=True)
    targets=[f for f in os.listdir(targetModelDir) if f.lower().endswith((".ply",".vtp",".vtk",".stl",".obj"))]
    total=len(targets); done=0; pose_diagnostics=[]
    if total==0:
      if status_callback: status_callback("No target meshes found in the selected folder.")
      if progress_callback: progress_callback(0,1,"No work"); return
    app=slicer.app
    try:
      prev_render = app.isRenderPaused()
    except Exception:
      prev_render = False
    app.setRenderPaused(True); scene=slicer.mrmlScene; scene.StartState(slicer.vtkMRMLScene.BatchProcessState)
    def _apply_pd_transform(pd, T): return transform_polydata(pd, T)
    def _mat4(M): return as_np4x4(M)
    def _apply_M_to_np(P, M4): P=np.asarray(P, np.float32); M4=_mat4(M4); Ph=np.c_[P, np.ones((len(P),1), np.float32)]; return (Ph @ M4.T)[:, :3]
    tpl_model = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode','_tmp_tpl_model'); tpl_model.SetAttribute('fastmorph.temp','1')
    tpl_corr  = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode','_tmp_tpl_corr'); tpl_corr.SetAttribute('fastmorph.temp','1')
    tpl_lm    = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode','_tmp_tpl_lm'); tpl_lm.SetAttribute('fastmorph.temp','1')
    save_scratch = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode','_tmp_save'); save_scratch.SetAttribute('fastmorph.temp','1')
    for n in (tpl_model, tpl_corr, tpl_lm, save_scratch):
      dn=n.GetDisplayNode()
      if dn:
        dn.SetVisibility(False)
        if hasattr(dn,'SetPointLabelsVisibility'): dn.SetPointLabelsVisibility(False)
    tpl_model_orig=vtk.vtkPolyData(); tpl_model_orig.DeepCopy(sourceModelNode.GetPolyData())
    tpl_corr_orig =slicer.util.arrayFromMarkupsControlPoints(sourceCorrNode).astype(np.float32,copy=True)
    tpl_lm_orig   =slicer.util.arrayFromMarkupsControlPoints(sourceLMNode).astype(np.float32,copy=True)
    try:
      if progress_callback: progress_callback(0, total, "Starting…")
      for i,fname in enumerate(targets):
        if cancel_callback and cancel_callback(): logging.info("Batch cancel requested."); break
        tgt_node = pred_node = None; pose_entry = None
        try:
          tgt_path=os.path.join(targetModelDir,fname)
          if status_callback: status_callback(f"[{i+1}/{total}] Load target")
          tgt_node=slicer.util.loadModel(tgt_path)
          if tgt_node is None or tgt_node.GetPolyData() is None: raise RuntimeError(f"Failed to load: {tgt_path}")
          dn=tgt_node.GetDisplayNode();
          if dn: dn.SetVisibility(False)
          tpl_model.SetAndObservePolyData(clone_polydata_points_only(tpl_model_orig))
          corr_np = tpl_corr_orig.copy()
          lm_np = tpl_lm_orig.copy()
          if need_template_nodes:
            with slicer.util.NodeModify(tpl_corr): slicer.util.updateMarkupsControlPointsFromArray(tpl_corr, corr_np)
            with slicer.util.NodeModify(tpl_lm):   slicer.util.updateMarkupsControlPointsFromArray(tpl_lm,   lm_np)
          pose_em = pose_em_enabled(parameters, skip_optimization=skipOpt)
          if not skipOpt:
            if pose_em:
              if cancel_callback and cancel_callback(): raise KeyboardInterrupt("Cancel requested")
              if status_callback: status_callback(f"[{i+1}/{total}] Pose EM template-shape optimization…")
              pose_result = self.initialize_template_pose_em(
                tableNode, tpl_model, tpl_corr, tpl_lm, tgt_node,
                parameters=parameters,
                ssmData=ssmData,
              )
              corr_np = slicer.util.arrayFromMarkupsControlPoints(tpl_corr).astype(np.float32, copy=True)
              lm_np   = slicer.util.arrayFromMarkupsControlPoints(tpl_lm).astype(np.float32, copy=True)
              pose_info = self.last_opt_info or {}
              pose_entry = {
                "target": fname,
                "score": float(pose_result.score),
                "score_margin": (
                  float(pose_result.score_margin)
                  if np.isfinite(pose_result.score_margin) else None
                ),
                "posterior_entropy": float(pose_result.posterior_entropy),
                "effective_hypotheses": float(pose_result.effective_hypotheses),
                "hypotheses_evaluated": int(pose_result.hypotheses_evaluated),
                "hypotheses_refined": int(pose_result.hypotheses_refined),
                "coefficient_norm": float(np.linalg.norm(pose_result.coefficients)),
                "registration_scale": float(pose_result.scale),
                "registration_size_ratio": float(pose_info.get("size_ratio", np.nan)),
                "pose_parallel": bool(pose_info.get("pose_parallel", True)),
              }
              if status_callback:
                status_callback(
                  f"[{i+1}/{total}] Pose EM selected template shape; continuing with standard rigid alignment "
                  f"(margin={pose_result.score_margin:.3g}, effective poses={pose_result.effective_hypotheses:.2f})"
                )
            else:
              if status_callback: status_callback(f"[{i+1}/{total}] Optimize (SSM+RANSAC)…")
              k = int(parameters.get("opt_pcGridSteps", 4))
              rper = int(parameters.get("opt_ransac_per_cand", 300000))
              _b,_T=self.initialize_template(tableNode, tpl_model, tpl_corr, tpl_lm, tgt_node, parameters=parameters, k=k, span=3.0, optimizer="powell", max_evals=240, eval_ransac_iters=int(0.2 * rper), final_ransac_iters=rper, seed=0, ssmData=ssmData)
              corr_np = slicer.util.arrayFromMarkupsControlPoints(tpl_corr).astype(np.float32, copy=True)
              lm_np   = slicer.util.arrayFromMarkupsControlPoints(tpl_lm).astype(np.float32, copy=True)
          else:
            if status_callback: status_callback(f"[{i+1}/{total}] Skip optimization")

          # Both template optimizers return a shape in the template/SSM frame.
          # Preserve the established downstream contract: scale, rigidly align,
          # and run PCA-CPD regardless of which optimizer selected the shape.
          b_src=np.array(tpl_model.GetPolyData().GetBounds()).reshape(3,2); b_tgt=np.array(tgt_node.GetPolyData().GetBounds()).reshape(3,2)
          size_src=np.linalg.norm(b_src[:,1]-b_src[:,0]); size_tgt=np.linalg.norm(b_tgt[:,1]-b_tgt[:,0])
          cov=float(parameters.get("targetCoverage",1.0))
          cov=float(np.clip(cov, 1e-3, 1.0))
          if not skipScaling:
            s=((size_tgt/cov)/size_src) if size_src>0 else 1.0
            T_scale=np.eye(4, dtype=np.float32); T_scale[:3,:3]*=s
            tpl_model.SetAndObservePolyData(_apply_pd_transform(tpl_model.GetPolyData(), T_scale))
            Ms=T_scale
          else:
            s=1.0
            Ms=np.eye(4, dtype=np.float32)
          corr_np=_apply_M_to_np(corr_np, Ms)
          lm_np  =_apply_M_to_np(lm_np,   Ms)
          if need_runtime_nodes:
            with slicer.util.NodeModify(tpl_corr): slicer.util.updateMarkupsControlPointsFromArray(tpl_corr, corr_np)
            with slicer.util.NodeModify(tpl_lm):   slicer.util.updateMarkupsControlPointsFromArray(tpl_lm,   lm_np)
          if cancel_callback and cancel_callback(): raise KeyboardInterrupt("Cancel requested")
          if status_callback: status_callback(f"[{i+1}/{total}] Subsample & features…")
          _src_pc,_tgt_pc,src_down,tgt_down,src_fpfh,tgt_fpfh,voxel=self.runSubsample(tpl_model, tgt_node, skipScaling, parameters)
          if status_callback: status_callback(f"[{i+1}/{total}] RANSAC+ICP rigid…")
          M_icp=_mat4(self.estimateTransform(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel, parameters))
          if need_final_mesh or use_bih:
            tpl_model.SetAndObservePolyData(_apply_pd_transform(tpl_model.GetPolyData(), M_icp))
          corr_np=_apply_M_to_np(corr_np, M_icp); lm_np=_apply_M_to_np(lm_np, M_icp)
          if need_runtime_nodes:
            with slicer.util.NodeModify(tpl_corr): slicer.util.updateMarkupsControlPointsFromArray(tpl_corr, corr_np)
            with slicer.util.NodeModify(tpl_lm):   slicer.util.updateMarkupsControlPointsFromArray(tpl_lm,   lm_np)
          if cancel_callback and cancel_callback(): raise KeyboardInterrupt("Cancel requested")
          if status_callback: status_callback(f"[{i+1}/{total}] PCA-CPD deformable…")
          aligned_corr=corr_np
          deformed_corr=self.runDeformable(tableNode=tableNode, sourceLM=aligned_corr, scale=s, targetSLM=np.asarray(tgt_down.points), parameters=parameters, rigidMatrix=M_icp, ssmData=ssmData)
          if cancel_callback and cancel_callback(): raise KeyboardInterrupt("Cancel requested")
          pred_np = None
          if use_bih:
              try:
                  V0, V1, F = self.warp_model_biharmonic(
                      modelNode=tpl_model,
                      src_corr=aligned_corr,
                      dst_corr=deformed_corr,
                      lam=lam_bih
                  )
                  if tpl_lm is not None:
                      self.warp_markups_barycentric(tpl_lm, V0, V1, F)
              except Exception as e:
                  logging.warning(f"[batch] Biharmonic failed; falling back to TPS. Reason: {e}")
                  if need_final_mesh:
                    self.warp_model_tps(tpl_model, aligned_corr, deformed_corr, lam=lam_tps, max_corr=max_corr, seed=0, landmarksNode=tpl_lm)
                  else:
                    pred_np = self.warp_points_tps(lm_np, aligned_corr, deformed_corr, lam=lam_tps, max_corr=max_corr, seed=0).astype(np.float32, copy=False)
          elif need_final_mesh:
              self.warp_model_tps(tpl_model, aligned_corr, deformed_corr, lam=lam_tps, max_corr=max_corr, seed=0, landmarksNode=tpl_lm)
          else:
              pred_np = self.warp_points_tps(lm_np, aligned_corr, deformed_corr, lam=lam_tps, max_corr=max_corr, seed=0).astype(np.float32, copy=False)

          if projection_enabled:
            if status_callback: status_callback(f"[{i+1}/{total}] Project to surface…")
            pred_node=self.runPointProjection(tpl_model, tgt_node, tpl_lm, projectionFactor)
            self.propagateLandmarkTypes(sourceLMNode, pred_node)
            pred_np=slicer.util.arrayFromMarkupsControlPoints(pred_node).astype(np.float32,copy=False)
          elif pred_np is None:
            self.propagateLandmarkTypes(sourceLMNode, tpl_lm)
            pred_np=slicer.util.arrayFromMarkupsControlPoints(tpl_lm).astype(np.float32,copy=False)
          root=os.path.splitext(fname)[0]
          if saveWarpedMeshes:
            if status_callback: status_callback(f"[{i+1}/{total}] Save warped mesh…")
            mesh_pd = tpl_model.GetPolyData()
            if mesh_pd.GetNumberOfPoints() == 0:
              raise RuntimeError("Warped mesh is empty; cannot save batch mesh output.")
            if smoothExportedMesh:
              mesh_pd = self.smoothPolyData(mesh_pd)
            self.savePolyDataVTP(mesh_pd, os.path.join(meshOutputDir, root + ".vtp"))
          out_path=os.path.join(outputDir, root+".mrk.json")
          if status_callback: status_callback(f"[{i+1}/{total}] Save landmarks…")
          with slicer.util.NodeModify(save_scratch):
            slicer.util.updateMarkupsControlPointsFromArray(save_scratch, pred_np)
          self.propagateLandmarkTypes(sourceLMNode, save_scratch)
          slicer.util.saveNode(save_scratch, out_path)
          if pose_entry is not None:
            pose_entry["landmarks"] = os.path.basename(out_path)
            pose_diagnostics.append(pose_entry)
        except KeyboardInterrupt:
          logging.info(f"[batch] Cancelled during {fname}"); raise
        except Exception as e:
          logging.error(f"[batch] Failed on {fname}: {e}")
        finally:
          for n in (pred_node, tgt_node):
            if n:
              try: scene.RemoveNode(n)
              except: pass
          done += 1
          if progress_callback: progress_callback(done,total,f"Done {done}/{total}")
      if status_callback and done==total: status_callback("Batch processing complete.")
    except KeyboardInterrupt:
      logging.info(f"[batch] Cancelled at {done}/{total}")
      if progress_callback: progress_callback(done,total,f"Cancelled at {done}/{total}")
      if status_callback: status_callback("Batch cancelled.")
    except Exception as e:
      logging.exception("Batch crashed.")
      if status_callback: status_callback(f"Batch error: {e}")
    finally:
      if pose_diagnostics:
        diagnostics_path = os.path.join(outputDir, "pose_em_diagnostics.json")
        try:
          with open(diagnostics_path, "w", encoding="utf-8") as stream:
            json.dump(pose_diagnostics, stream, indent=2, sort_keys=True)
        except Exception:
          logging.exception(f"Could not save pose-EM diagnostics: {diagnostics_path}")
      for n in (tpl_model, tpl_corr, tpl_lm, save_scratch):
        if n:
          try: scene.RemoveNode(n)
          except: pass
      scene.EndState(slicer.vtkMRMLScene.BatchProcessState)
      try: app.setRenderPaused(prev_render)
      except:
        try: app.setRenderPaused(False)
        except: pass

  def _tableToArray(self, tableNode):
    vtk_table = tableNode.GetTable(); n_rows = vtk_table.GetNumberOfRows(); n_cols = vtk_table.GetNumberOfColumns()
    arr = np.zeros((n_rows, n_cols), dtype=float)
    for j in range(n_cols): name = vtk_table.GetColumn(j).GetName(); arr[:, j] = slicer.util.arrayFromTableColumn(tableNode, name)
    return arr

  def _get_ssm_base(self, tableNode):
    if tableNode is None:
      raise ValueError("SSM cache requires a table node.")
    key = tableNode.GetID()
    cached = self._ssm_cache.get(key)
    if cached is not None:
      return cached
    A = self._tableToArray(tableNode)
    mean = A[:,0].reshape(-1, 3)
    U = A[:,1:].reshape(mean.shape[0], 3, -1)
    eig = np.array(json.loads(tableNode.GetAttribute("ssm_eigenvalues")), float)
    eig[eig < 1e-10] = 1e-10
    cached = {"flat": A, "mean": mean, "U": U, "eig": eig}
    self._ssm_cache[key] = cached
    return cached

  def _truncated_ssm(self, tableNode, parameters, ssmData=None):
    ssmData = self._get_ssm_base(tableNode) if ssmData is None else ssmData
    mean = np.asarray(ssmData["mean"], dtype=float)
    modes = np.asarray(ssmData["U"], dtype=float)
    eig = np.asarray(ssmData["eig"], dtype=float)
    variance_keep = float(parameters.get("variance_keep", 0.95))
    if eig.size == 0:
      raise ValueError("The SSM contains no retained modes.")
    if eig.sum() <= 0 or variance_keep >= 1.0:
      k = eig.size
    else:
      k = int(np.searchsorted(np.cumsum(eig), variance_keep * eig.sum()) + 1)
    k = max(1, min(k, eig.size))
    return mean, modes[:, :, :k], eig[:k]

  def _pose_em_target_points(self, targetModelNode, parameters):
    import tiny3d as t3d
    target = np.asarray(slicer.util.arrayFromModelPoints(targetModelNode), dtype=float)
    if target.ndim != 2 or target.shape[1] != 3 or len(target) == 0:
      raise ValueError("Pose EM requires a non-empty target model.")
    diag = float(np.linalg.norm(np.ptp(target, axis=0)))
    if "registrationPointCount" in parameters:
      voxel = target_count_voxel_size(
        target,
        int(parameters["registrationPointCount"]),
      )
    else:
      voxel = diag / (55.0 * max(float(parameters.get("pointDensity", 1.3)), 0.1))
    if voxel <= np.finfo(float).eps:
      return target
    cloud = t3d.geometry.PointCloud(); cloud.points = t3d.utility.Vector3dVector(target)
    down = cloud.voxel_down_sample(voxel)
    points = np.asarray(down.points, dtype=float)
    return points if len(points) else target

  def runPoseEMDeformable(self, tableNode, targetSLM, parameters, ssmData=None):
    if tableNode is None:
      raise ValueError("Pose EM registration requires an SSM table.")
    mean, modes, eig = self._truncated_ssm(tableNode, parameters, ssmData=ssmData)
    target = np.asarray(targetSLM, dtype=float)
    coverage = float(np.clip(parameters.get("targetCoverage", 1.0), 1e-3, 1.0))
    skip_scaling = bool(parameters.get("skipScaling", False))
    complete = np.isclose(coverage, 1.0)
    with_scale = bool(complete and not skip_scaling)
    source_scale = 1.0
    if not complete and not skip_scaling:
      source_scale = coverage_prescale(mean, target, coverage)

    settings = PoseEMSettings.from_mapping(parameters)
    result = run_pose_em_registration(
      mean,
      target,
      modes,
      eig,
      settings,
      with_scale=with_scale,
      source_scale=source_scale,
    )
    points = result.points
    target_diag = float(np.linalg.norm(np.ptp(target, axis=0)))
    registered_diag = float(np.linalg.norm(np.ptp(points, axis=0)))
    size_ratio = registered_diag / max(target_diag, np.finfo(float).eps)
    if with_scale and size_ratio < 0.1:
      logging.warning(
        "[pose-em] Diagnostic similarity pose has a small size ratio "
        f"({size_ratio:.4g}, scale={result.scale:.4g}); retaining the selected "
        "SSM coefficients because downstream scaling and rigid alignment remain active."
      )
    return np.asarray(points, dtype=float), result

  def initialize_template_pose_em(self, tableNode, srcModelNode, srcCorrNode, srcLmNode, tgtModelNode, parameters, ssmData=None):
    self.last_opt_info = None
    oldCorr = np.asarray(slicer.util.arrayFromMarkupsControlPoints(srcCorrNode), dtype=float)
    mean, modes, _ = self._truncated_ssm(tableNode, parameters, ssmData=ssmData)
    if oldCorr.shape != mean.shape:
      raise ValueError(
        f"Template correspondences have shape {oldCorr.shape}, but the SSM mean has shape {mean.shape}."
      )
    target = self._pose_em_target_points(tgtModelNode, parameters)
    registeredCorr, result = self.runPoseEMDeformable(
      tableNode,
      target,
      parameters,
      ssmData=ssmData,
    )
    # Pose EM uses global pose to evaluate shape hypotheses, but template
    # optimization returns only the selected SSM shape. Keep it in the SSM's
    # original frame so the established MorphoWeave scaling, rigid, and deformable
    # stages always run afterward.
    newCorr = ssm_sample(mean, modes, result.coefficients)

    use_bih = bool(parameters.get("useBiharmonic", False))
    if use_bih:
      lam = float(parameters.get("bih_lam_init", parameters.get("bih_lam", 1e4)))
      try:
        V0, V1, F = self.warp_model_biharmonic(srcModelNode, oldCorr, newCorr, lam=lam)
        if srcLmNode is not None:
          self.warp_markups_barycentric(srcLmNode, V0, V1, F)
      except Exception as exc:
        logging.warning(f"[pose-em] Biharmonic warp failed; falling back to TPS. Reason: {exc}")
        self.warp_model_tps(
          srcModelNode, oldCorr, newCorr,
          lam=float(parameters.get("tpsLambda", 0.0)),
          max_corr=int(parameters.get("tpsMaxCorr", 800)),
          seed=int(parameters.get("poseSeed", 0)),
          landmarksNode=srcLmNode,
        )
    else:
      self.warp_model_tps(
        srcModelNode, oldCorr, newCorr,
        lam=float(parameters.get("tpsLambda", 0.0)),
        max_corr=int(parameters.get("tpsMaxCorr", 800)),
        seed=int(parameters.get("poseSeed", 0)),
        landmarksNode=srcLmNode,
      )
    slicer.util.updateMarkupsControlPointsFromArray(srcCorrNode, newCorr)

    self.last_opt_info = {
      "backend": POSE_EM_BACKEND,
      "decision": "pose_em_template_chosen",
      "norm_b": float(np.linalg.norm(result.coefficients)),
      "score": float(result.score),
      "score_margin": float(result.score_margin),
      "posterior_entropy": float(result.posterior_entropy),
      "effective_hypotheses": float(result.effective_hypotheses),
      "hypotheses_evaluated": int(result.hypotheses_evaluated),
      "hypotheses_refined": int(result.hypotheses_refined),
      "scale": float(result.scale),
      "pose_parallel": bool(result.final_parameters["pose_parallel"]),
      "size_ratio": float(
        np.linalg.norm(np.ptp(registeredCorr, axis=0)) /
        max(np.linalg.norm(np.ptp(target, axis=0)), np.finfo(float).eps)
      ),
    }
    return result

  def smoothPolyData(self, polyData, iterations=8, passBand=0.1):
      f = vtk.vtkWindowedSincPolyDataFilter()
      f.SetInputData(polyData)
      f.SetNumberOfIterations(iterations)
      f.SetPassBand(passBand)
      f.BoundarySmoothingOn()
      f.FeatureEdgeSmoothingOff()
      f.NormalizeCoordinatesOff()
      f.Update()
      out = vtk.vtkPolyData(); out.DeepCopy(f.GetOutput())
      return out

  def smoothModel(self, modelNode, iterations=8, passBand=0.1):
      modelNode.SetAndObservePolyData(self.smoothPolyData(modelNode.GetPolyData(), iterations=iterations, passBand=passBand))

  def savePolyDataVTP(self, polyData, outputPath):
      writer = vtk.vtkXMLPolyDataWriter()
      writer.SetFileName(outputPath)
      writer.SetInputData(polyData)
      if writer.Write() != 1:
          raise RuntimeError(f"Failed to save warped mesh: {outputPath}")

  def runDeformable(self, tableNode, sourceLM, scale, targetSLM, parameters, rigidMatrix=None, ssmData=None):
      import rustcpd
      if tableNode is None: raise ValueError("runDeformable requires tableNode to be set")

      # Keep SSM geometry consistent with the (already scaled) sourceLM
      ssmData = self._get_ssm_base(tableNode) if ssmData is None else ssmData
      M = ssmData["mean"].shape[0]
      if M != sourceLM.shape[0]:
          raise ValueError(f"SSM table has {M} points but sourceLM has {sourceLM.shape[0]}")

      modes = ssmData["U"]
      eig = ssmData["eig"]

      variance_keep = float(parameters.get("variance_keep", 0.95))
      total = eig.sum()
      if total <= 0:
          k = eig.size
      else:
          cum = np.cumsum(eig)
          k = int(np.searchsorted(cum, variance_keep * total) + 1)
          if k > eig.size: k = eig.size

      # keep the first k modes/eigenvalues, preserving original order
      modes = modes[:, :, :k]
      eigvals_eff = eig[:k]

      # Rigid rotation for the modes
      if rigidMatrix is None:
          if self.rigidTransformNode is None:
              raise ValueError("runDeformable requires either rigidMatrix or logic.rigidTransformNode to be set")
          mat = vtk.vtkMatrix4x4(); self.rigidTransformNode.GetMatrixTransformToParent(mat)
          T = vtk_mat_to_np(mat)
      else:
          T = as_np4x4(rigidMatrix)
      R = T[:3, :3]

      U_aligned = np.einsum('ij,pjk->pik', R, modes).reshape(3*M, -1)

      def _diag(P):
        if P.size==0: return 1.0
        bmin,bmax = P.min(0), P.max(0); return float(np.linalg.norm(bmax-bmin))

      d_tgt = _diag(np.asarray(targetSLM)); s20 = 20.0/max(d_tgt,1e-12)
      src_n = np.asarray(sourceLM)*s20; tgt_n = np.asarray(targetSLM)*s20

      cov = float(parameters.get("targetCoverage", 1.0))
      is_complete = abs(cov - 1.0) < 1e-6   # or: np.isclose(cov, 1.0)
      sparse_k = max(1, min(10, len(tgt_n)))

      pca = rustcpd.register_atlas(
          np.asarray(tgt_n),             # scaled target
          np.asarray(src_n),             # source is the SSM base
          U_aligned,
          eigvals_eff,                   # no external scaling/flooring
          lambda_regularization=float(parameters.get("lambda_reg", 0.01)),
          outlier_weight=float(parameters.get("w", 0.10)),
          tolerance=float(parameters.get("tolerance", 1e-6)),
          max_iterations=int(parameters.get("max_iterations", 250)),
          with_scale=is_complete,
          normalize=True,
          optimize_similarity=True,
          k=sparse_k,
          kdtree_radius_scale=10.0,
      )
      warped_landmarks = np.asarray(pca.points)

      if bool(parameters.get("skipFineCPD", False)):
        return warped_landmarks / s20

      final = rustcpd.register_deformable(
          np.asarray(tgt_n),
          warped_landmarks,
          beta=float(parameters.get("beta", 2.0)),
          alpha=float(parameters.get("alpha", 2.0)),
          low_rank=max(1, min(300, len(warped_landmarks))),
          tolerance=float(parameters.get("tolerance", 1e-6)),
          max_iterations=int(parameters.get("max_iterations", 250)),
          k=sparse_k,
      )
      warped_landmarks = np.asarray(final.points)
      return warped_landmarks/s20

  def _triangulate_polydata(self, pd):
    tf = vtk.vtkTriangleFilter()
    tf.SetInputData(pd)
    tf.PassLinesOff(); tf.PassVertsOff()
    tf.Update()
    return tf.GetOutput()

  def _cotangent_laplacian(self, V, F):
      from scipy import sparse

      # 1. Compute edge vectors for all faces at once
      # V has shape (N, 3), F has shape (M, 3)
      A = V[F[:, 0]]; B = V[F[:, 1]]; C = V[F[:, 2]]
      e0 = B - C; e1 = C - A; e2 = A - B

      # 2. Compute area/normals (cross product magnitude)
      # Using the fact that |cross(u,v)| = 2 * Area
      cross_prod = np.cross(e0, -e2)
      norm_n = np.linalg.norm(cross_prod, axis=1)

      # Filter degenerate triangles (zero area)
      valid = norm_n > 1e-12
      F = F[valid]; e0 = e0[valid]; e1 = e1[valid]; e2 = e2[valid]; norm_n = norm_n[valid]

      # 3. Compute Cotangents (dot / magnitude)
      # cot(angle) = dot(u, v) / |cross(u, v)|
      # We divide by 2*Area (norm_n) later, so we pre-multiply terms if needed
      # Actually simpler: cot(alpha) = (e1 . e2) / |e1 x e2| ... careful with signs/orientation
      # Standard formula: cot at vertex i (opposite edge e0) is -dot(e1, e2) / norm(cross(e1, e2))

      fact = 0.5 / norm_n
      c0 = -np.sum(e1 * e2, axis=1) * fact
      c1 = -np.sum(e2 * e0, axis=1) * fact
      c2 = -np.sum(e0 * e1, axis=1) * fact

      # 4. Construct Sparse Matrix
      # We have 3 cotangents per face. We need to assemble them into the NxN matrix.
      # Rows/Cols indices
      ii = F[:, [1, 2, 0]].ravel()
      jj = F[:, [2, 0, 1]].ravel()
      data = np.concatenate([c0, c1, c2]) # Corresponding cotangents

      # Use coo_matrix to handle duplicate entries (summing them automatically)
      # We need both (i,j) and (j,i) symmetric entries
      rows = np.concatenate([ii, jj])
      cols = np.concatenate([jj, ii])
      vals = np.concatenate([data, data]) * 0.5

      n = V.shape[0]
      L = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

      # Diagonal is negative sum of off-diagonals
      L.setdiag(-np.array(L.sum(axis=1)).ravel())

      # Mass matrix (lumped area)
      # Area per vertex is sum of 1/3 area of adjacent faces
      area_per_face = 0.5 * norm_n
      M_diag = np.zeros(n)
      np.add.at(M_diag, F, area_per_face[:, None] / 3.0)
      M = sparse.diags(np.maximum(M_diag, 1e-12))

      return L, M

  def warp_model_biharmonic(self, modelNode, src_corr, dst_corr, lam=1e4):
      """
      Solve (L^T M^{-1} L + lam*W) X = lam*W*target  for X (per coordinate),
      where W has 1's at constrained (nearest) vertices to src_corr.
      """
      from scipy.spatial import cKDTree
      from scipy.sparse import diags
      from scipy.sparse.linalg import splu
      import vtk.util.numpy_support as vtk_np


      # Triangulate and extract (V, F)
      tri = self._triangulate_polydata(modelNode.GetPolyData())
      V0 = vtk_np.vtk_to_numpy(tri.GetPoints().GetData()).astype(np.float64, copy=False)
      ca = vtk_np.vtk_to_numpy(tri.GetPolys().GetData())
      F = ca.reshape(-1, 4)[:, 1:4].astype(np.int64, copy=False)


      # Biharmonic operator
      L, M = self._cotangent_laplacian(V0, F)
      from scipy.sparse import diags
      Minv = diags(1.0 / np.maximum(M.diagonal(), 1e-12))
      A0 = (L.T @ Minv @ L).tocsr()

      # Map correspondences to nearest vertices (deduplicate)
      tree = cKDTree(V0)
      idx_raw = tree.query(np.asarray(src_corr, np.float64), k=1)[1]
      idx_unique, inv = np.unique(idx_raw, return_inverse=True)

      # Average multiple constraints that hit the same vertex
      target = np.zeros((idx_unique.size, 3), np.float64)
      np.add.at(target, inv, np.asarray(dst_corr, np.float64))
      counts = np.bincount(inv, minlength=idx_unique.size).astype(np.float64)
      target /= counts[:, None]

      # Assemble W and RHS
      n = V0.shape[0]
      Wd = np.zeros(n, dtype=np.float64); Wd[idx_unique] = 1.0
      A = (A0 + lam * diags(Wd)).tocsc()
      lu = splu(A)

      B = np.zeros((n, 3), np.float64)
      B[idx_unique, :] = target
      B *= lam

      # Solve for X (per coordinate)
      V1 = np.empty_like(V0)
      V1[:, 0] = lu.solve(B[:, 0])
      V1[:, 1] = lu.solve(B[:, 1])
      V1[:, 2] = lu.solve(B[:, 2])

      # Write back
      pts = modelNode.GetPolyData().GetPoints()
      pts.SetData(vtk_np.numpy_to_vtk(V1.astype(np.float32, copy=False), deep=1))
      pts.Modified(); modelNode.GetPolyData().Modified()

      return V0, V1, F  # handy if you want barycentric landmark warping

  def warp_markups_barycentric(self, markupsNode, V0, V1, F):
    """Project each point to the nearest triangle on (V0,F), get barycentric
    coords there, then carry to (V1,F)."""
    import vtk.util.numpy_support as vtk_np

    # Build a vtk polydata/locator for the pre-warp surface
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints(); pts.SetData(vtk_np.numpy_to_vtk(V0, deep=1))
    polys = vtk.vtkCellArray()
    for a, b, c in F:
        polys.InsertNextCell(3)
        polys.InsertCellPoint(int(a)); polys.InsertCellPoint(int(b)); polys.InsertCellPoint(int(c))
    pd.SetPoints(pts); pd.SetPolys(polys)
    locator = vtk.vtkStaticCellLocator(); locator.SetDataSet(pd); locator.BuildLocator()

    def barycentric(p, A, B, C):
        v0 = B - A; v1 = C - A; v2 = p - A
        d00 = v0.dot(v0); d01 = v0.dot(v1); d11 = v1.dot(v1)
        d20 = v2.dot(v0); d21 = v2.dot(v1)
        denom = d00 * d11 - d01 * d01
        if denom <= 1e-12: return 1.0, 0.0, 0.0
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return u, v, w

    P = slicer.util.arrayFromMarkupsControlPoints(markupsNode).astype(np.float64, copy=False)
    Pnew = np.empty_like(P)

    cid = vtk.reference(0); sid = vtk.reference(0); d2 = vtk.reference(0.0)
    for i, p in enumerate(P):
        cp = [0.0, 0.0, 0.0]
        locator.FindClosestPoint(p.tolist(), cp, cid, sid, d2)
        t = int(cid)
        a, b, c = F[t]
        u, v, w = barycentric(np.asarray(cp), V0[a], V0[b], V0[c])
        Pnew[i] = u * V1[a] + v * V1[b] + w * V1[c]

    with slicer.util.NodeModify(markupsNode):
        slicer.util.updateMarkupsControlPointsFromArray(markupsNode, Pnew.astype(np.float32, copy=False))

  # ---------- Stable TPS/RBF backend ----------
  def _tps_kernel_3d(self, r):
      # Classic 3D TPS radial kernel U(r) = r
      return r

  def _fps_indices(self, P, m, seed=0):
      """Farthest point sampling indices on P (N,3)."""
      P = np.asarray(P, float)
      n = P.shape[0]
      if m >= n: return np.arange(n, dtype=int)
      rng = np.random.default_rng(int(seed) & 0x7fffffff)
      # start farthest from mean
      start = int(np.argmax(np.linalg.norm(P - P.mean(0, keepdims=True), axis=1)))
      sel = [start]
      d = np.linalg.norm(P - P[start], axis=1)
      for _ in range(1, m):
          i = int(np.argmax(d))
          sel.append(i)
          d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
      return np.array(sel, dtype=int)

  def _tps_fit_3d(self, src, dst, lam=0.0, max_corr=None, seed=0):
      """
      Fit 3D TPS mapping f(x) = K(x,src) @ W + [1 x] @ A,
      solving block system [[K+λI, P],[P^T, 0]] [W; A] = [dst; 0].
      Returns (src_used, W (n×3), A (4×3)).
      """
      src = np.asarray(src, np.float64)
      dst = np.asarray(dst, np.float64)
      if src.shape != dst.shape or src.shape[1] != 3:
          raise ValueError("TPS fit expects src/dst with shape (N,3)")

      n = src.shape[0]
      if n < 4:
          # Fallback: affine fit
          A = self._fit_affine(src, dst)  # (4×3)
          return src, np.zeros((n,3), dtype=np.float64), A

      # Reduce constraints if needed
      if max_corr is not None and n > int(max_corr):
          idx = self._fps_indices(src, int(max_corr), seed=seed)
          src = src[idx]
          dst = dst[idx]
          n = src.shape[0]

      # Build K (n×n)
      # pairwise distances
      diff = src[:, None, :] - src[None, :, :]
      K = self._tps_kernel_3d(np.linalg.norm(diff, axis=2))
      if lam > 0:
          K = K + lam * np.eye(n, dtype=np.float64)

      # P: (n×4) with [1, x, y, z]
      P = np.hstack([np.ones((n,1), np.float64), src])

      # Assemble L and RHS
      L = np.zeros((n+4, n+4), dtype=np.float64)
      L[:n, :n] = K
      L[:n, n:] = P
      L[n:, :n] = P.T
      Y = np.zeros((n+4, 3), dtype=np.float64)
      Y[:n, :] = dst

      # Solve for [W; A]
      try:
          sol = np.linalg.solve(L, Y)
      except np.linalg.LinAlgError:
          # regularize more and retry
          L[:n, :n] += (1e-8 + lam) * np.eye(n, dtype=np.float64)
          sol = np.linalg.lstsq(L, Y, rcond=None)[0]

      W = sol[:n, :]   # (n×3)
      A = sol[n:, :]   # (4×3)
      return src, W, A

  def _tps_eval_3d(self, X, src, W, A, chunk=50000):
      """Apply fitted TPS to points X (M×3)."""
      X = np.asarray(X, np.float64)
      M = X.shape[0]
      out = np.empty((M, 3), dtype=np.float64)
      for s in range(0, M, chunk):
          e = min(M, s + chunk)
          Xi = X[s:e]
          # K(Xi, src): (m×n)
          diff = Xi[:, None, :] - src[None, :, :]
          K = self._tps_kernel_3d(np.linalg.norm(diff, axis=2))
          Pi = np.hstack([np.ones((Xi.shape[0], 1), np.float64), Xi])
          out[s:e] = K @ W + Pi @ A
      return out

  def warp_points_tps(self, points, src_corr, dst_corr, lam=0.0, max_corr=800, seed=0):
      src_used, W, A = self._tps_fit_3d(src_corr, dst_corr, lam=lam, max_corr=max_corr, seed=seed)
      return self._tps_eval_3d(points, src_used, W, A)

  def _fit_affine(self, P, Q):
      """Return 4×3 affine that maps P→Q in least squares: [P 1] A = Q."""
      P = np.asarray(P, np.float64); Q = np.asarray(Q, np.float64)
      X = np.hstack([P, np.ones((P.shape[0],1), np.float64)])  # (N×4)
      A, _, _, _ = np.linalg.lstsq(X, Q, rcond=None)           # (4×3)
      return A

  def _apply_affine(self, X, A):
      X = np.asarray(X, np.float64)
      return np.hstack([X, np.ones((X.shape[0],1), np.float64)]) @ A  # (N×3)

  def warp_model_tps(self, modelNode, src_corr, dst_corr, lam=0.0, max_corr=800, seed=0, landmarksNode=None):
      """
      Stable default warp: thin-plate spline in 3D with optional smoothing (lam)
      and constraint cap (max_corr). Returns (V0, V1).
      """
      # Fit
      src_used, W, A = self._tps_fit_3d(src_corr, dst_corr, lam=lam, max_corr=max_corr, seed=seed)

      # Model vertices
      V0 = vtk_np.vtk_to_numpy(modelNode.GetPolyData().GetPoints().GetData()).astype(np.float64, copy=False)
      V1 = self._tps_eval_3d(V0, src_used, W, A)

      # Write back
      pts = modelNode.GetPolyData().GetPoints()
      pts.SetData(vtk_np.numpy_to_vtk(V1.astype(np.float32, copy=False), deep=1))
      pts.Modified(); modelNode.GetPolyData().Modified()

      # Optionally landmarks too
      if landmarksNode is not None:
          P = slicer.util.arrayFromMarkupsControlPoints(landmarksNode).astype(np.float64, copy=False)
          P1 = self._tps_eval_3d(P, src_used, W, A)
          with slicer.util.NodeModify(landmarksNode):
              slicer.util.updateMarkupsControlPointsFromArray(landmarksNode, P1.astype(np.float32, copy=False))

      return V0, V1


  def convertMatrixToTransformNode(self, matrix, transformName):
    return make_transform_node(matrix, transformName)

  def convertPointsToVTK(self, points):
    array_vtk = vtk_np.numpy_to_vtk(points, deep=True, array_type=vtk.VTK_FLOAT)
    points_vtk = vtk.vtkPoints(); points_vtk.SetData(array_vtk)
    polydata_vtk = vtk.vtkPolyData(); polydata_vtk.SetPoints(points_vtk)
    return polydata_vtk

  def displayPointCloud(self, polydata, nodeName, nodeColor, frac=0.004, refNode=None):
    pd = vtk.vtkPolyData(); pd.DeepCopy(polydata)
    # radius from reference (target) bounds
    if refNode is not None:
        b = np.array(refNode.GetPolyData().GetBounds(), float).reshape(3,2)
        radius = 0.5 * np.max(b[:,1]-b[:,0]) * float(frac)
    else:
        pts = vtk_np.vtk_to_numpy(pd.GetPoints().GetData())
        d = np.linalg.norm(pts.max(0) - pts.min(0))
        radius = float(frac) * d

    sph = vtk.vtkSphereSource(); sph.SetRadius(radius); sph.SetThetaResolution(8); sph.SetPhiResolution(8)
    g = vtk.vtkGlyph3D(); g.SetInputData(pd); g.SetSourceConnection(sph.GetOutputPort()); g.ScalingOff(); g.Update()

    node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', nodeName); node.CreateDefaultDisplayNodes()
    node.SetAndObservePolyData(g.GetOutput()); node.GetDisplayNode().SetColor(nodeColor)
    return node


  def estimateTransform(self, sourcePoints, targetPoints, sourceFeatures, targetFeatures, voxelSize, parameters):
    import tiny3d as t3d
    t3d.utility.random.seed(int(42))
    distanceThreshold = voxelSize * parameters["distanceThreshold"]
    if not sourcePoints.has_normals():
      sourcePoints.estimate_normals(t3d.geometry.KDTreeSearchParamHybrid(radius=voxelSize * parameters["normalSearchRadius"], max_nn=30))
    if not targetPoints.has_normals():
      targetPoints.estimate_normals(t3d.geometry.KDTreeSearchParamHybrid(radius=voxelSize * parameters["normalSearchRadius"], max_nn=30))
    no_scaling = t3d.pipelines.registration.registration_ransac_based_on_feature_matching(
      sourcePoints, targetPoints, sourceFeatures, targetFeatures, True, distanceThreshold,
      t3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
      [t3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), t3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distanceThreshold)],
      t3d.pipelines.registration.RANSACConvergenceCriteria(parameters["maxRANSAC"], confidence=parameters["confidence"]))
    no_scaling_eval = t3d.pipelines.registration.evaluate_registration(targetPoints, sourcePoints, distanceThreshold, np.linalg.inv(no_scaling.transformation))
    best_result=no_scaling; fitness=(no_scaling.fitness + no_scaling_eval.fitness)/2
    count=0; maxAttempts=20
    try:
      while fitness < 0.99 and count < maxAttempts:
        result = t3d.pipelines.registration.registration_ransac_based_on_feature_matching(
          sourcePoints, targetPoints, sourceFeatures, targetFeatures, True, distanceThreshold,
          t3d.pipelines.registration.TransformationEstimationPointToPoint(True), 3,
          [t3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), t3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distanceThreshold)],
          t3d.pipelines.registration.RANSACConvergenceCriteria(parameters["maxRANSAC"], confidence=parameters["confidence"]))
        evaluation = t3d.pipelines.registration.evaluate_registration(targetPoints, sourcePoints, distanceThreshold, np.linalg.inv(result.transformation))
        mean_fitness = (result.fitness + evaluation.fitness)/2
        if mean_fitness > fitness: fitness = mean_fitness; best_result = result
        count += 1
    except: pass
    icp = t3d.pipelines.registration.registration_icp(
      sourcePoints, targetPoints, voxelSize * parameters["ICPDistanceThreshold"], best_result.transformation,
      t3d.pipelines.registration.TransformationEstimationPointToPlane())
    return icp.transformation

  def runSubsample(self, sourceNode, targetNode, skipScaling, parameters):
    import tiny3d as t3d
    t3d.utility.random.seed(int(42))
    src_pd = sourceNode.GetPolyData(); tgt_pd = targetNode.GetPolyData()
    sourcePoints_np = vtk_np.vtk_to_numpy(src_pd.GetPoints().GetData()); targetPoints_np = vtk_np.vtk_to_numpy(tgt_pd.GetPoints().GetData())
    source = t3d.geometry.PointCloud(); source.points = t3d.utility.Vector3dVector(sourcePoints_np)
    target = t3d.geometry.PointCloud(); target.points = t3d.utility.Vector3dVector(targetPoints_np)
    cov=float(parameters.get("targetCoverage",1.0))
    cov=float(np.clip(cov, 1e-3, 1.0))
    if skipScaling:
      size = np.linalg.norm(target.get_max_bound() - target.get_min_bound())
      size_eff = size / cov
      voxel_size = size_eff / (55 * max(float(parameters.get("pointDensity", 1.3)), 0.1))
      source_center = np.zeros(3); target_center = np.zeros(3); source_scaling = target_scaling = 1.0
    else:
      voxel_size = 1.0 / (55 * max(float(parameters.get("pointDensity", 1.3)), 0.1)); source_center = source.get_center(); target_center = target.get_center()
      tmp_src = copy.deepcopy(source).translate(-source_center); tmp_tgt = copy.deepcopy(target).translate(-target_center)
      sourceSize = np.linalg.norm(tmp_src.get_max_bound() - tmp_src.get_min_bound()); targetSize = np.linalg.norm(tmp_tgt.get_max_bound() - tmp_tgt.get_min_bound())
      source_scaling = 1.0 / sourceSize if sourceSize > 0 else 1.0; target_scaling = 1.0 / targetSize if targetSize > 0 else 1.0
      source.scale(source_scaling, center=source_center); target.scale(target_scaling, center=target_center)
    if "registrationPointCount" in parameters:
      voxel_size = target_count_voxel_size(
        np.asarray(target.points, dtype=float),
        int(parameters["registrationPointCount"]),
      )
      source_voxel_size = voxel_size
    else:
      source_voxel_size = voxel_size * cov
    source_down, source_fpfh = self.preprocess_point_cloud(source, source_voxel_size, parameters["normalSearchRadius"], parameters["FPFHSearchRadius"])
    target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size, parameters["normalSearchRadius"], parameters["FPFHSearchRadius"])
    if not skipScaling:
      src_pts = np.asarray(source_down.points); tgt_pts = np.asarray(target_down.points)
      src_pts = (src_pts - source_center) / source_scaling + source_center
      tgt_pts = (tgt_pts - target_center) / target_scaling + target_center
      source_down = t3d.geometry.PointCloud(); source_down.points = t3d.utility.Vector3dVector(src_pts)
      target_down = t3d.geometry.PointCloud(); target_down.points = t3d.utility.Vector3dVector(tgt_pts)
      voxel_size = voxel_size / target_scaling
    return (source, target, source_down, target_down, source_fpfh, target_fpfh, voxel_size)

  def propagateLandmarkTypes(self, sourceNode, targetNode):
    if sourceNode.GetNumberOfControlPoints() != targetNode.GetNumberOfControlPoints(): logging.warning("Source and target nodes have different number of landmarks"); return
    for i in range(sourceNode.GetNumberOfControlPoints()):
      if hasattr(sourceNode, 'GetNthControlPointDescription'):
        targetNode.SetNthControlPointDescription(i, sourceNode.GetNthControlPointDescription(i))
      if hasattr(sourceNode, 'GetNthControlPointLabel'):
        targetNode.SetNthControlPointLabel(i, sourceNode.GetNthControlPointLabel(i))

  def preprocess_point_cloud(self, pcd, voxel_size, radius_normal_factor, radius_feature_factor):
    import tiny3d as t3d
    t3d.utility.random.seed(int(42))
    pcd_down = pcd.voxel_down_sample(voxel_size)
    if len(pcd_down.points) == 0:
      raise RuntimeError(f"Downsampling produced 0 points at voxel={voxel_size:.4f}. Increase target registration points or verify model scale.")
    radius_normal = voxel_size * radius_normal_factor
    pcd_down.estimate_normals(t3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * radius_feature_factor
    pcd_fpfh = t3d.pipelines.registration.compute_fpfh_feature(pcd_down, t3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

  def runPointProjection(self, template, model, templateLandmarks, maxProjectionFactor):
      bb = np.array(model.GetPolyData().GetBounds(), float).reshape(3,2)
      maxProjection = np.linalg.norm(bb[:,1] - bb[:,0]) * maxProjectionFactor

      templatePoints = self.getFiducialPoints(templateLandmarks)
      projectedPointsPD = self.projectPointsPolydataCoherent(
          model.GetPolyData(), templatePoints, rayLength=maxProjection,
          k_neighbors=6, lambda_cohere=2.0, snap_to_surface=True
      )

      projectedLMNode = slicer.mrmlScene.AddNewNodeByClass(
          'vtkMRMLMarkupsFiducialNode', "Refined Predicted Landmarks"
      )
      with slicer.util.NodeModify(projectedLMNode):
          P = projectedPointsPD.GetPoints()
          for i in range(P.GetNumberOfPoints()):
              projectedLMNode.AddControlPoint(P.GetPoint(i))
          projectedLMNode.SetLocked(True)
          projectedLMNode.SetFixedNumberOfControlPoints(True)
      return projectedLMNode


  def getFiducialPoints(self,fiducialNode):
    points = vtk.vtkPoints()
    for i in range(fiducialNode.GetNumberOfControlPoints()): points.InsertNextPoint(fiducialNode.GetNthControlPointPosition(i))
    return points

  def projectPointsPolydataCoherent(self, targetPolydata, originalPoints, rayLength, k_neighbors=6, lambda_cohere=2.0, snap_to_surface=True, surface_tol=5e-4, outward_eps=1e-3, use_outer_surface=True):
      # --- target prep (single outer skin + normals + SDF) ---
      def _outer(pd):
          if not use_outer_surface: return pd
          tf=vtk.vtkTriangleFilter(); tf.SetInputData(pd); tf.Update(); tri=tf.GetOutput()
          b=np.array(tri.GetBounds(),float).reshape(3,2); c=b.mean(1); d=float(np.linalg.norm(b[:,1]-b[:,0]))+1e-12
          cf=vtk.vtkPolyDataConnectivityFilter(); cf.SetInputData(tri)
          cf.SetExtractionModeToClosestPointRegion(); cf.SetClosestPoint(float(c[0]),float(c[1]),float(c[2]+10*d)); cf.Update()
          out=vtk.vtkPolyData(); out.DeepCopy(cf.GetOutput()); return out
      tgt0=_outer(targetPolydata)
      nrm=vtk.vtkPolyDataNormals(); nrm.SetInputData(tgt0); nrm.SplittingOff(); nrm.ConsistencyOn(); nrm.AutoOrientNormalsOn()
      nrm.ComputeCellNormalsOn(); nrm.ComputePointNormalsOff(); nrm.Update(); tgt=nrm.GetOutput()
      cellLoc=vtk.vtkStaticCellLocator(); cellLoc.SetDataSet(tgt); cellLoc.BuildLocator()
      sdf=vtk.vtkImplicitPolyDataDistance(); sdf.SetInput(tgt)
      B=np.array(tgt.GetBounds(),float).reshape(3,2); diag=float(np.linalg.norm(B[:,1]-B[:,0]))+1e-12
      tol=float(surface_tol)*diag; eps=float(outward_eps)*diag; maxmove=float(rayLength)*1.0  # rayLength is already in “diag” units in your UI

      # --- data & masks ---
      N=originalPoints.GetNumberOfPoints()
      P0=np.array([originalPoints.GetPoint(i) for i in range(N)],float)
      d0=np.array([sdf.FunctionValue(P0[i]) for i in range(N)],float)  # signed distance
      fixed_mask=(d0>=0.0)&(np.abs(d0)<=tol)         # already on/near *exterior* surface => freeze
      move_mask=~fixed_mask                           # move the rest (outside but far, or inside)
      if not np.any(move_mask):
          out=vtk.vtkPolyData(); pts=vtk.vtkPoints()
          for i in range(N): pts.InsertNextPoint(P0[i]); out.SetPoints(pts); return out

      # --- anchors only for moving points (nearest surface + outward nudge if inside) ---
      A=np.empty((N,3),float); w=np.zeros((N,),float)
      cp=[0.0,0.0,0.0]; cid=vtk.mutable(0); sid=vtk.mutable(0); d2=vtk.mutable(0.0)
      for i in np.where(move_mask)[0]:
          cellLoc.FindClosestPoint(P0[i].tolist(),cp,cid,sid,d2)
          q=np.array(cp,float)
          if d0[i]<0.0:  # inside: push outward using SDF gradient at the closest point
              g=np.zeros(3); sdf.FunctionGradient(q,g); ng=np.linalg.norm(g)
              if ng>0: q=q+(eps*(g/ng))
          A[i]=q; w[i]=0.7
      A[fixed_mask]=P0[fixed_mask]  # irrelevant for fixed, but keeps arrays dense

      # --- coherent masked solve: (W_MM+λL_MM) X_M = W_MM A_M - λ L_MF X_F  with X_F=P0_F ---
      L,_,_ = self._knn_graph_laplacian(P0, k=k_neighbors, sigma=None)
      from scipy import sparse
      from scipy.sparse.linalg import spsolve
      idxF=np.where(fixed_mask)[0]; idxM=np.where(move_mask)[0]
      W = sparse.diags(w[idxM], 0, shape=(len(idxM),len(idxM)))
      L_MM=L[idxM[:,None], idxM]; L_MF=L[idxM[:,None], idxF]
      X = P0.copy()
      XF = P0[idxF]
      RHS = (W @ A[idxM])
      # solve per coordinate
      Msys = (W + float(lambda_cohere)*L_MM).tocsr()
      add  = (-float(lambda_cohere))*(L_MF @ XF)
      for c in range(3):
          X[idxM,c] = spsolve(Msys, RHS[:,c] + add[:,c])

      # --- clamp travel and enforce exterior at the end (moved points only) ---
      dvec=X-P0; dlen=np.linalg.norm(dvec,axis=1)
      over = (dlen>maxmove+1e-12)&move_mask
      if np.any(over): X[over]=P0[over]+dvec[over]*(maxmove/(dlen[over]+1e-12))[:,None]

      if snap_to_surface:
          for i in np.where(move_mask)[0]:
              cellLoc.FindClosestPoint(X[i].tolist(),cp,cid,sid,d2)
              q=np.array(cp,float)
              if sdf.FunctionValue(q)<0.0:
                  g=np.zeros(3); sdf.FunctionGradient(q,g); ng=np.linalg.norm(g)
                  if ng>0: q=q+(eps*(g/ng))
              X[i]=q

      pts=vtk.vtkPoints()
      for i in range(N): pts.InsertNextPoint(X[i].tolist())
      out=vtk.vtkPolyData(); out.SetPoints(pts); return out


  def _ssm_unpack(self, tableNode, drop_modes=None, ssmData=None):
    base = self._get_ssm_base(tableNode) if ssmData is None else ssmData
    mean = base["mean"]; U = base["U"]; eig = base["eig"]
    if drop_modes:
      keep = [i for i in range(U.shape[2]) if i not in set(drop_modes)]; U = U[:,:,keep]; eig = eig[keep]
    return mean, U, eig

  def initialize_template(self, tableNode, srcModelNode, srcCorrNode, srcLmNode, tgtModelNode, parameters,
                          k=10, span=2.5, optimizer="powell", max_evals=240,
                          eval_ransac_iters=50000, final_ransac_iters=200000, seed=0, ssmData=None):
      import numpy as np, tiny3d as t3d
      self.last_opt_info = None

      mean, U, eig = self._ssm_unpack(tableNode, ssmData=ssmData)
      k = int(min(k, U.shape[2])); k_eff = max(1, k)
      eig_k = eig[:k]; sqrt_eig_k = np.sqrt(eig_k)
      M = mean.shape[0]
      Uw_flat = (U[:, :, :k].reshape(-1, k) * sqrt_eig_k[None, :])
      mean_flat = mean.reshape(-1)
      def ssm_sample(b): return (mean_flat + Uw_flat @ np.asarray(b, float)).reshape(M, 3)
      def bbox_diag_np(P): P = np.asarray(P); return float(np.linalg.norm(P.max(0) - P.min(0)))

      # -- target prep (exactly as before) --
      oldCorr = slicer.util.arrayFromMarkupsControlPoints(srcCorrNode)
      tgt_np  = slicer.util.arrayFromModelPoints(tgtModelNode)
      tgt_pcd = t3d.geometry.PointCloud(); tgt_pcd.points = t3d.utility.Vector3dVector(tgt_np)
      s0 = bbox_diag_np(mean) / (bbox_diag_np(tgt_np) + 1e-12)
      tgt_scaled = t3d.geometry.PointCloud(tgt_pcd); tgt_scaled.scale(s0, center=tgt_scaled.get_center())
      size_scaled = float(np.linalg.norm(tgt_scaled.get_max_bound() - tgt_scaled.get_min_bound()))
      if "registrationPointCount" in parameters:
          voxel_f = target_count_voxel_size(
              np.asarray(tgt_scaled.points, dtype=float),
              int(parameters["registrationPointCount"]),
          )
      else:
          voxel_f = size_scaled / (25.0 * float(parameters.get("pointDensity", 1.0)))
      voxel_c = voxel_f * 1.5
      rn = int(parameters.get("normalSearchRadius", 2)); rf = int(parameters.get("FPFHSearchRadius", 5))
      tgt_down_f, tgt_fpfh_f = self.preprocess_point_cloud(tgt_scaled, voxel_f, rn, rf)
      tgt_down_c, tgt_fpfh_c = self.preprocess_point_cloud(tgt_scaled, voxel_c, max(1, rn // 2), max(2, rf // 2))
      dist_f = voxel_f * float(parameters.get("distanceThreshold", 3.0))
      dist_c = voxel_c * float(parameters.get("distanceThreshold", 3.0))

      # -- helpers shared by baseline & candidates --
      def _eval_pack(pack, iters, s):
          cand_down, cand_fpfh, td, tf, dth = pack
          r = t3d.pipelines.registration.registration_ransac_based_on_feature_matching(
              cand_down, td, cand_fpfh, tf, False, dth,
              t3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
              [t3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
              t3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dth)],
              t3d.pipelines.registration.RANSACConvergenceCriteria(int(iters), float(parameters.get("confidence", 0.95))))
          ev = t3d.pipelines.registration.evaluate_registration(cand_down, td, dth, r.transformation)
          size_td = float(np.linalg.norm(td.get_max_bound() - td.get_min_bound()))
          rmse_n = ev.inlier_rmse / (size_td + 1e-12)
          return r.transformation, float(ev.fitness), float(rmse_n)

      def _get_feat_from_points(pts_np, coarse):
          p = t3d.geometry.PointCloud(); p.points = t3d.utility.Vector3dVector(np.asarray(pts_np, float))
          if coarse:
              pd, f = self.preprocess_point_cloud(p, voxel_c, max(1, rn // 2), max(2, rf // 2))
              return (pd, f, tgt_down_c, tgt_fpfh_c, dist_c)
          pd, f = self.preprocess_point_cloud(p, voxel_f, rn, rf)
          return (pd, f, tgt_down_f, tgt_fpfh_f, dist_f)

      # -------------------------
      # 1) Baseline (original template) score
      # -------------------------
      w_rmse = float(parameters.get("w_rmse", 0.60))
      tau_fit  = float(parameters.get("opt_skip_if_fit_ge", 0.82))
      tau_rmse = float(parameters.get("opt_skip_if_rmse_le", 0.025))  # normalized

      src_pts_np = slicer.util.arrayFromModelPoints(srcModelNode)   # current template geometry
      base_pack  = _get_feat_from_points(src_pts_np, coarse=False)
      try:
          T_base, fit_base, rmse_base = _eval_pack(base_pack, final_ransac_iters, seed+101)
          score_base = fit_base - w_rmse * rmse_base
          best = {"score": score_base, "b": None, "fit": fit_base, "rmse": rmse_base, "T": T_base}
      except Exception:
          # Fall back if baseline eval failed
          best = {"score": -np.inf, "b": None, "fit": None, "rmse": None, "T": None}

      # Early exit if "already good"
      if (best["score"] > -np.inf) and (fit_base >= tau_fit) and (rmse_base <= tau_rmse):
          print(f"[opt] Skip: baseline good (fit={fit_base:.3f}, rmse_n={rmse_base:.3f})")
          # DO NOT warp template/correspondences; caller will apply T_base rigidly
          self.last_opt_info = {
              "decision": "baseline_good",
              "fit": float(fit_base),
              "rmse": float(rmse_base),
              "norm_b": 0.0
          }
          return np.zeros(k, float), T_base

      # -------------------------
      # 2) Candidate search (unchanged logic, but seeded by baseline)
      # -------------------------
      feat_cache = {}
      def get_feat(b, coarse):
          key = (tuple(np.round(np.asarray(b, float), 6)), bool(coarse))
          if key in feat_cache:
              return feat_cache[key]
          pts = ssm_sample(b)
          # optional candidate thinning for coarse stage
          max_pts_cand_coarse = int(parameters.get("maxFeatPtsCoarse", 1800))
          if coarse and (max_pts_cand_coarse > 0) and (len(pts) > max_pts_cand_coarse):
              rng = np.random.default_rng(seed)
              idx = rng.choice(len(pts), size=max_pts_cand_coarse, replace=False)
              pts = pts[idx]
          feat_cache[key] = _get_feat_from_points(pts, coarse)
          return feat_cache[key]

      def eval_bestof(b, iters, coarse, nrestarts):
          itp = max(4, int(iters // max(1, nrestarts)))
          best_loc = (-np.inf, None, None, None)
          for j in range(nrestarts):
              try:
                  T, fit, rmse = _eval_pack(get_feat(b, coarse), itp, seed+7919*j+(13 if coarse else 29))
                  sc = fit - w_rmse * rmse
                  if np.isfinite(sc) and sc > best_loc[0]: best_loc = (sc, T, fit, rmse)
              except Exception:
                  continue
          if best_loc[1] is None: raise RuntimeError("RANSAC failed")
          return best_loc[1], best_loc[2], best_loc[3]

      iL = 1.0 / (eig_k + 1e-12)
      rho = float(parameters.get("reg_strength", 0.10))
      w_reg = min(rho / k_eff, 0.25 / (k_eff * (span**2) + 1e-12))
      LARGE = 1e6; bound_margin = float(parameters.get("bound_margin", 1e-5))
      def clip_b(b): return np.clip(np.asarray(b, float), -span, span)

      cache={}
      def score_geom(b, iters, coarse, nrestarts):
          b = clip_b(b); key = (tuple(np.round(b,6)), bool(coarse), int(iters), int(nrestarts))
          if key in cache: return cache[key]["neg"]
          reg = float(b @ (iL * b))
          rmse_headroom = 0.05
          # upper bound on possible score for pruning
          ub = 1.0 - w_rmse*0.0 - w_reg*reg
          lb_best = best["score"] - (w_rmse * rmse_headroom)
          if best["score"] > -np.inf and ub <= lb_best - bound_margin:
              cache[key] = {"neg": LARGE + reg}; return cache[key]["neg"]
          if ssm_sample(b).shape != oldCorr.shape:
              cache[key] = {"neg": LARGE}; return cache[key]["neg"]
          try:
              T, fit, rmse = eval_bestof(b, iters, coarse, int(parameters.get("restarts_coarse" if coarse else "restarts_fine", 4)))
          except Exception:
              cache[key] = {"neg": LARGE}; return cache[key]["neg"]
          score = fit - w_rmse*rmse - w_reg*reg
          if np.isfinite(score) and score > best["score"]:
              best.update(score=score, b=b.copy(), fit=float(fit), rmse=float(rmse), T=T.copy())
          cache[key] = {"neg": -score}; return cache[key]["neg"]

      # candidate pool (as in your version; kept brief)
      try:
          from scipy.stats import qmc
          N = int(parameters.get("init_candidates", 192)); N = max(96, N)
          dpow = int(np.ceil(np.log2(max(2, N))))
          Sob = qmc.scale(qmc.Sobol(d=k, scramble=True, seed=seed).random_base2(dpow)[:N], -span, span)
      except Exception:
          rng0 = np.random.default_rng(seed); Sob = rng0.uniform(-span, span, size=(max(96, int(parameters.get("init_candidates", 384))), k))
      m = min(k, 10); axis=[]
      for a in (1.0, 0.5):
          A = np.zeros((m, k)); A[np.arange(m), np.arange(m)] =  a * span; axis.append(A)
          A = np.zeros((m, k)); A[np.arange(m), np.arange(m)] = -a * span; axis.append(A)
      Pool = np.unique(np.round(np.vstack([Sob, *axis]), 6), axis=0)

      # small coarse then fine scoring passes
      beam_B = int(parameters.get("beam_width", 24))
      def _embed_candidates(B, seed):
          B = np.asarray(B, float)
          Bz = (B - B.mean(0, keepdims=True)) / (B.std(0, keepdims=True) + 1e-9)
          try:
              import umap
              Z = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean",
                            random_state=int(seed)&0x7fffffff).fit_transform(Bz)
          except Exception:
              U2, S2, _ = np.linalg.svd(Bz, full_matrices=False); Z = U2[:, :2] * S2[:2]
          return np.asarray(Z, float)
      def _fps_2d(Z, k):
          n=Z.shape[0]
          if n==0: return np.zeros((0,), dtype=int)
          start=int(np.argmax(np.linalg.norm(Z-Z.mean(0,keepdims=True),axis=1)))
          sel=[start]; d=np.linalg.norm(Z-Z[start],axis=1)
          for _ in range(1, min(k,n)):
              i=int(np.argmax(d)); sel.append(i); d=np.minimum(d, np.linalg.norm(Z-Z[i],axis=1))
          return np.array(sel, int)
      Z = _embed_candidates(Pool, seed+23); C = Pool[_fps_2d(Z, min(beam_B, len(Pool)))]

      budgets = [0.02, 0.15]
      keeps   = [beam_B, max(12, beam_B // 2)]
      it = max(1, int(budgets[0] * eval_ransac_iters))
      C = sorted(C, key=lambda x: score_geom(x, it, True, int(parameters.get("restarts_coarse", 2))))[:min(keeps[0], len(C))]
      # quick geometric diversity
      def keep_diverse(C_sorted, keep):
          if not C_sorted: return []
          X = np.asarray(C_sorted, float); sel=[0]; d=np.linalg.norm(X-X[0],axis=1)
          for _ in range(1, min(keep, len(X))):
              i=int(np.argmax(d)); sel.append(i); d=np.minimum(d, np.linalg.norm(X-X[i],axis=1))
          return [X[i].tolist() for i in sel]
      C = keep_diverse(C, min(keeps[0], len(C)))

      it = max(1, int(budgets[1] * eval_ransac_iters))
      C = sorted(C, key=lambda x: score_geom(x, it, False, int(parameters.get("restarts_fine", 4))))[:min(keeps[1], len(C))]
      C = keep_diverse(C, min(keeps[1], len(C)))

      # local refinement
      if optimizer.lower() == "de":
          from scipy.optimize import differential_evolution
          bounds = [(-span, span)]*k
          res = differential_evolution(lambda x: score_geom(x, eval_ransac_iters, False, int(parameters.get("restarts_fine", 4))),
                                      bounds=bounds, strategy="best1bin", popsize=8, maxiter=max(1, max_evals//8),
                                      tol=1e-3, polish=False, seed=seed, updating="deferred", workers=1)
          b_star = np.clip(res.x, -span, span)
      else:
          from scipy.optimize import minimize
          start = best["b"] if best["b"] is not None else (np.asarray(C[0]) if len(C) else np.zeros(k))
          per = max(1, int(0.5 * max_evals))
          r = minimize(lambda x: score_geom(x, eval_ransac_iters, False, int(parameters.get("restarts_fine", 4))),
                      x0=np.clip(start, -span, span), method="Powell",
                      options={"maxfev": per, "xtol":1e-3, "ftol":1e-3, "disp":False})
          b_star = np.clip(getattr(r, "x", start), -span, span)

      _ = score_geom(b_star, final_ransac_iters, False, int(parameters.get("restarts_final", 8)))
      if best["score"] == -np.inf:
          raise RuntimeError("Optimization failed to find a valid candidate.")
      used_baseline = (best["b"] is None)

      # -------------------------
      # 3) Apply only if candidate beats baseline
      # -------------------------
      if not used_baseline:
          # warp to SSM sample (your existing backend choices)
          newCorr = ssm_sample(best["b"])
          use_bih = bool(parameters.get("useBiharmonic", False))
          if use_bih:
              lam = float(parameters.get("bih_lam_init", parameters.get("bih_lam", 1e4)))
              try:
                  V0, V1, F = self.warp_model_biharmonic(srcModelNode, oldCorr, newCorr, lam=lam)
                  if srcLmNode is not None: self.warp_markups_barycentric(srcLmNode, V0, V1, F)
              except Exception as e:
                  logging.warning(f"[opt] Biharmonic warp failed; falling back to TPS. Reason: {e}")
                  lam_tps  = float(parameters.get("tpsLambda", 0.0))
                  max_corr = int(parameters.get("tpsMaxCorr", 800))
                  self.warp_model_tps(srcModelNode, oldCorr, newCorr, lam=lam_tps, max_corr=max_corr, seed=0, landmarksNode=srcLmNode)
          else:
              lam_tps  = float(parameters.get("tpsLambda", 0.0))
              max_corr = int(parameters.get("tpsMaxCorr", 800))
              self.warp_model_tps(srcModelNode, oldCorr, newCorr, lam=lam_tps, max_corr=max_corr, seed=0, landmarksNode=srcLmNode)
          slicer.util.updateMarkupsControlPointsFromArray(srcCorrNode, newCorr)  # keep dense corr = exact SSM sample
          print(f"[opt] candidate beat baseline: ||b||={np.linalg.norm(best['b']):.3f} fit={best['fit']:.3f} rmse={best['rmse']:.3f}")
          self.last_opt_info = {
              "decision": "candidate_chosen",
              "fit": float(best["fit"]),
              "rmse": float(best["rmse"]),
              "norm_b": float(np.linalg.norm(best["b"]))
          }
          return best["b"], best["T"]

      # baseline wins → DO NOT warp; return zeros and the baseline rigid
      print(f"[opt] baseline kept: fit={best['fit']:.3f} rmse={best['rmse']:.3f}")
      self.last_opt_info = {
          "decision": "baseline_kept",
          "fit": float(best["fit"]),
          "rmse": float(best["rmse"]),
          "norm_b": 0.0
      }
      return np.zeros(k, float), best["T"]

  def _knn_graph_laplacian(self, X, k=6, sigma=None):
      # X: (N,3); returns L (N×N CSR), and degree-normalized weights W_ij
      from scipy.spatial import cKDTree
      from scipy import sparse
      X = np.asarray(X, float)
      N = X.shape[0]
      tree = cKDTree(X)
      d, idx = tree.query(X, k=k+1)          # first neighbor is self
      idx = idx[:, 1:]; d = d[:, 1:]
      if sigma is None:
          sigma = np.median(d[:, -1]) + 1e-12
      wij = np.exp(-(d**2) / (2.0 * sigma**2))
      # Build symmetric weight matrix
      rows = np.repeat(np.arange(N), k)
      cols = idx.ravel()
      data = wij.ravel()
      W = sparse.coo_matrix((data, (rows, cols)), shape=(N, N))
      # symmetrize
      W = (W + W.T) * 0.5
      # Laplacian
      deg = np.array(W.sum(axis=1)).ravel()
      L = sparse.diags(deg) - W
      return L.tocsr(), W.tocsr(), sigma


class MorphoWeaveLandmarkTransferTest(ScriptedLoadableModuleTest):
  def setUp(self): slicer.mrmlScene.Clear(0)
  def runTest(self): self.setUp(); self.test_MorphoWeaveLandmarkTransfer1()
  def test_MorphoWeaveLandmarkTransfer1(self):
    self.delayDisplay("Starting the test"); logic = MorphoWeaveLandmarkTransferLogic(); self.assertIsNotNone(logic); self.delayDisplay('Test passed!')
