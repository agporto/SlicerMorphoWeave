import os, json, logging, html, tempfile, numpy as np
import vtk, qt, ctk, slicer, vtk.util.numpy_support as vtk_np
from slicer.ScriptedLoadableModule import *
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.cluster.vq import kmeans2


def setWorkflowStatus(label, state, detail):
  title, color = {
    "needs_input": ("Needs input", "#c75b5b"),
    "ready": ("Ready", "#4f9d69"),
    "optional": ("Optional", "#c18b37"),
    "complete": ("Complete", "#4f9d69"),
    "warning": ("Finished with issues", "#c18b37"),
  }[state]
  safeDetail = html.escape(str(detail)).replace("\n", "<br>")
  label.setTextFormat(qt.Qt.RichText)
  label.setText(f'<span style="color:{color}; font-weight:600">{title}</span> — {safeDetail}')
  label.setAccessibleName(f"{title}: {detail}")


class MorphoWeaveSurfaceSegmentation(ScriptedLoadableModule):
  def __init__(self, parent):
    super().__init__(parent); p=parent
    p.title="Surface Segmentation"; p.categories=["MorphoWeave"]; p.dependencies=[]
    p.contributors=["Arthur Porto"]; p.helpText="Dataset-consistent segmentation from meshes + dense correspondences (.mrk.json)."; p.acknowledgementText=""

class MorphoWeaveSurfaceSegmentationWidget(ScriptedLoadableModuleWidget):
  def setup(self):
    super().setup(); self.logic=MorphoWeaveSurfaceSegmentationLogic(); self._buildUI()
  def _t(self, le):
    t = getattr(le, "text")
    return t().strip() if callable(t) else str(t).strip()

  def _spin(self, w):
    v = getattr(w, "value")
    return v() if callable(v) else v

  def _checked(self, cb):
    f = getattr(cb, "isChecked", None)
    return f() if callable(f) else bool(getattr(cb, "checked", False))

  def _browse(self, le, dirOnly=True):
    p=qt.QFileDialog.getExistingDirectory(self.parent,"Select folder") if dirOnly else qt.QFileDialog.getOpenFileName(self.parent,"Select file")[0]
    if p: le.setText(p)

  def _buildUI(self):
    root=ctk.ctkCollapsibleButton(); root.text="Surface Segmentation Workflow"; root.collapsed=False; self.layout.addWidget(root); f=qt.QFormLayout(root)
    info = qt.QLabel("Select paired mesh/.mrk.json folders, tune segmentation settings, then run.")
    info.setWordWrap(True)
    f.addRow(info)

    basic = ctk.ctkCollapsibleButton(); basic.text = "Required Inputs"; basic.collapsed = False
    fb = qt.QFormLayout(basic); f.addRow(basic)
    self.meshDir=qt.QLineEdit(); b1=qt.QToolButton(); b1.text="…"; b1.clicked.connect(lambda:self._browse(self.meshDir))
    h1=qt.QHBoxLayout(); h1.addWidget(self.meshDir); h1.addWidget(b1); fb.addRow("Meshes folder:",h1)
    self.mrkDir=qt.QLineEdit(); b2=qt.QToolButton(); b2.text="…"; b2.clicked.connect(lambda:self._browse(self.mrkDir))
    h2=qt.QHBoxLayout(); h2.addWidget(self.mrkDir); h2.addWidget(b2); fb.addRow("Dense Correspondences (.mrk.json) Folder:",h2)
    self.outDir=qt.QLineEdit(); b3=qt.QToolButton(); b3.text="…"; b3.clicked.connect(lambda:self._browse(self.outDir))
    h3=qt.QHBoxLayout(); h3.addWidget(self.outDir); h3.addWidget(b3); fb.addRow("Output folder:",h3)

    adv = ctk.ctkCollapsibleButton(); adv.text = "Advanced Tuning"; adv.collapsed = True
    fa = qt.QFormLayout(adv); f.addRow(adv)
    self.kSeg=qt.QSpinBox(); self.kSeg.setRange(2,64); self.kSeg.setValue(15); fa.addRow("# Segments (k):",self.kSeg)
    self.knn=qt.QSpinBox(); self.knn.setRange(4,128); self.knn.setValue(12); fa.addRow("kNN on loci:",self.knn)
    self.smooth=qt.QSpinBox(); self.smooth.setRange(0,10); self.smooth.setValue(1); fa.addRow("1-ring smoothing iters:",self.smooth)
    self.alpha=qt.QDoubleSpinBox(); self.alpha.setRange(0.0,1.0); self.alpha.setSingleStep(0.05); self.alpha.setValue(0.7); fa.addRow("Feature vs Spatial weight α:",self.alpha)
    self.autoTune = qt.QCheckBox("Auto-tune (k, α)"); self.autoTune.setChecked(False); fa.addRow(self.autoTune)
    self.kGrid = qt.QLineEdit("10,12,15,18"); fa.addRow("k grid:", self.kGrid)
    self.aGrid = qt.QLineEdit("0.5,0.7,0.9"); fa.addRow("α grid:", self.aGrid)
    self.saveVTP=qt.QCheckBox("Write .vtp with SegID & SegID_smooth"); self.saveVTP.setChecked(True); fa.addRow(self.saveVTP)
    self.savePLY=qt.QCheckBox("Write PLY segment interiors (mixed-label triangles excluded)")
    self.savePLY.setChecked(False); fa.addRow(self.savePLY)
    self.previewN=qt.QSpinBox(); self.previewN.setRange(0,20); self.previewN.setValue(3); fa.addRow("Preview first N in scene:",self.previewN)

    self.statusLbl = qt.QLabel("")
    self.statusLbl.setWordWrap(True)
    f.addRow(self.statusLbl)
    self.runBtn=qt.QPushButton("Run Segmentation"); self.runBtn.clicked.connect(self.onRun); f.addRow(self.runBtn)

    self.autoTune.toggled.connect(self._onAutoTuneChanged)
    self.meshDir.textChanged.connect(lambda *_: self._validateInputs())
    self.mrkDir.textChanged.connect(lambda *_: self._validateInputs())
    self.outDir.textChanged.connect(lambda *_: self._validateInputs())
    self._onAutoTuneChanged(self.autoTune.isChecked())
    self._validateInputs()

  def _onAutoTuneChanged(self, enabled):
    self.kGrid.setEnabled(bool(enabled))
    self.aGrid.setEnabled(bool(enabled))

  def _validateInputs(self):
    mesh_text, mrk_text, out_text = (self._t(w) for w in (self.meshDir, self.mrkDir, self.outDir))
    if not mesh_text or not mrk_text:
      self.runBtn.enabled = False
      setWorkflowStatus(self.statusLbl, "needs_input", "Select valid mesh and correspondence folders.")
      return
    dMeshes = os.path.abspath(os.path.expanduser(mesh_text))
    dMrk = os.path.abspath(os.path.expanduser(mrk_text))
    if not (os.path.isdir(dMeshes) and os.path.isdir(dMrk)):
      self.runBtn.enabled = False
      setWorkflowStatus(self.statusLbl, "needs_input", "Select valid mesh and correspondence folders.")
      return
    try:
      pairs = len(self.logic._pair_files(dMeshes, dMrk))
    except ValueError as exc:
      self.runBtn.enabled = False
      setWorkflowStatus(self.statusLbl, "needs_input", str(exc))
      return
    self.runBtn.enabled = pairs > 0 and bool(out_text)
    if pairs == 0:
      setWorkflowStatus(self.statusLbl, "needs_input", "No matching mesh and .mrk.json filenames were found.")
    elif not out_text:
      setWorkflowStatus(self.statusLbl, "needs_input", "Choose an output folder.")
    else:
      setWorkflowStatus(self.statusLbl, "ready", f"{pairs} paired specimens detected.")

  def onRun(self):
    if not all(self._t(w) for w in (self.meshDir, self.mrkDir, self.outDir)):
      slicer.util.errorDisplay("Select input folders and an explicit output folder."); return
    dMeshes=os.path.abspath(os.path.expanduser(self._t(self.meshDir)))
    dMrk=os.path.abspath(os.path.expanduser(self._t(self.mrkDir)))
    dOut=os.path.abspath(os.path.expanduser(self._t(self.outDir)))
    if not (os.path.isdir(dMeshes) and os.path.isdir(dMrk)):
      slicer.util.errorDisplay("Input folders not found."); return
    # Output creation is handled inside the guarded logic call below.

    # parse grids (robust to commas/semicolons/extra spaces)
    def _parse_ints(s):
      return [int(x) for x in s.replace(';',',').split(',') if x.strip()]
    def _parse_floats(s):
      return [float(x) for x in s.replace(';',',').split(',') if x.strip()]

    auto = self._checked(self.autoTune)
    try:
      k_grid = _parse_ints(self._t(self.kGrid)) if auto else None
      a_grid = _parse_floats(self._t(self.aGrid)) if auto else None
    except Exception:
      slicer.util.errorDisplay("Invalid auto-tune grid values. Use comma-separated numbers (example: k=10,12,15 and alpha=0.5,0.7,0.9).")
      return
    if auto and (not k_grid or not a_grid):
      slicer.util.errorDisplay("Auto-tune is enabled, but k/alpha grids are empty.")
      return

    with slicer.util.tryWithErrorDisplay("Surface Segmentation failed", waitCursor=True):
      res=self.logic.run(
        dMeshes,dMrk,dOut,
        k=int(self._spin(self.kSeg)), knn=int(self._spin(self.knn)),
        smooth_iters=int(self._spin(self.smooth)), alpha=float(self._spin(self.alpha)),
        write_vtp=self._checked(self.saveVTP), previewN=int(self._spin(self.previewN)),
        auto_tune=auto, k_grid=k_grid, alpha_grid=a_grid,
        write_ply_segments=self._checked(self.savePLY)
      )
      msg = (f"Surface Segmentation: {res['n_used']}/{res['n_pairs']} specimens used, "
             f"{res['n_skipped']} skipped, {res['n_export_failed']} export failures; "
             f"k={res['k']}, {res['wrote']} mesh files written")
      if 'Q' in res: msg += f", Q={res['Q']:.3f}"
      slicer.util.showStatusMessage(msg)
      issues = [f"{os.path.basename(f['mesh'])} ({f['stage']}): {f['error']}" for f in res['failures']]
      issues.extend(res['warnings'])
      state = "warning" if issues else "complete"
      detail = msg + f". Outputs: {dOut}"
      if issues:
        detail += "\n" + "\n".join(issues)
        slicer.util.warningDisplay(detail)
      setWorkflowStatus(self.statusLbl, state, detail)
      logging.info(res)


class MorphoWeaveSurfaceSegmentationLogic(ScriptedLoadableModuleLogic):
  def run(self, meshes_dir, mrk_dir, out_dir,
          k=15, knn=12, smooth_iters=1, alpha=0.7,
          write_vtp=True, previewN=3,
          auto_tune=False, k_grid=None, alpha_grid=None, write_ply_segments=False):

    if out_dir is None or not str(out_dir).strip():
      raise ValueError("Select an explicit output folder.")
    pairs = self._pair_files(meshes_dir, mrk_dir)
    if not pairs: raise RuntimeError("No mesh↔.mrk.json pairs found.")
    ok, M = self._validate_mrk_counts([m for _, m in pairs])
    if not ok: raise RuntimeError("Dense correspondences count mismatch.")
    if M is None or M < 3:
      raise ValueError("Surface segmentation requires at least 3 corresponding loci per specimen.")
    for k_value in (k_grid or [k]) if auto_tune else [k]:
      self._validate_cluster_count(M, k_value)
    for a_value in (alpha_grid or [alpha]) if auto_tune else [alpha]:
      if not np.isfinite(a_value) or not 0.0 <= a_value <= 1.0:
        raise ValueError("Feature weight alpha must be finite and between 0 and 1.")
    if int(knn) != knn or knn < 1:
      raise ValueError("kNN must be a positive integer.")
    os.makedirs(out_dir, exist_ok=True)
    self._clear_preview_nodes()
    feats = []; cache = {}
    used_pairs = []; failures = []; warnings = []

    # If you want to see mesh name inside lower helpers for logging:
    self._current_mesh = None

    for meshPath, mrkPath in pairs:
      self._current_mesh = meshPath
      try:
        V, F, pdW = self._load_mesh_VF_world(meshPath)
        cache[meshPath] = (V, F, pdW)

        P = self._load_mrk_positions_world(mrkPath)
        vidx = cKDTree(V).query(P, k=1)[1]

        F_locus = self._locus_geofeats(
          pdW, V, F, vidx,
          scales=(0,4,12),
          include_dihedral=True,
          include_normal_var=True,
          include_axis_orient=True,
          pool_hops=1,
          include_hks=False,   # HKS remains optional and disabled by default
          hks_k=30, hks_times=6
        )

        self._assert_finite("features", F_locus, meshPath)
        feats.append(F_locus)
        used_pairs.append((meshPath, mrkPath))

      except Exception as e:
        logging.warning(f"[MorphoWeaveSurfaceSegmentation] Skipping '{os.path.basename(meshPath)}' due to {type(e).__name__}: {e}")
        failures.append({"mesh": meshPath, "markups": mrkPath,
                         "stage": "feature extraction", "error": f"{type(e).__name__}: {e}"})
        continue

    self._current_mesh = None

    if not used_pairs:
      detail = "; ".join(f"{os.path.basename(f['mesh'])}: {f['error']}" for f in failures)
      raise RuntimeError("All meshes failed during feature extraction. " + detail)

    # Construct the spatial graph only from specimens actually used.
    M = feats[0].shape[0]
    loci_mean = self._mean_loci([m for _, m in used_pairs], M)
    A_geo = self._knn_graph(loci_mean, knn)

    # Ensure consistent feature width (optional HKS failures are zero-filled).
    Dmax = max(f.shape[1] for f in feats)
    if any(f.shape[1] != Dmax for f in feats):
      logging.warning("[MorphoWeaveSurfaceSegmentation] Feature dimension mismatch; padding to D=%d", Dmax)
      feats = [np.pad(f, ((0,0),(0, Dmax - f.shape[1])), mode='constant') for f in feats]

    feats = np.stack(feats, 0)  # S × M × D

    # choose/tune & cluster
    if auto_tune:
      if not k_grid: k_grid = [k]
      if not alpha_grid: alpha_grid = [alpha]
      k_best, alpha_best, Q_best, labels, W = self._choose_k_alpha(loci_mean, feats, A_geo, k_grid, alpha_grid)
      logging.info(f"[MorphoWeaveSurfaceSegmentation] Auto-tune chose k={k_best}, α={alpha_best:.3f}, Q={Q_best:.4f}")
      k, alpha, Q = k_best, alpha_best, Q_best
    else:
      W = self._loci_kernel_multi(loci_mean, feats, A_geo, alpha)
      labels = self._spectral_cluster(W, k)
      Q = self._modularity(W, labels)
      logging.info(f"[MorphoWeaveSurfaceSegmentation] Q={Q:.4f} (k={k}, α={alpha:.3f})")

    # Export exactly the cached world-coordinate geometry used for labeling.
    # Keep raw and smoothed point labels separate in the complete VTP surface.
    output_files = []; export_failed = set(); segment_reports = []
    for i, (meshPath, mrkPath) in enumerate(used_pairs):
      try:
        V, F, pdW = cache[meshPath]
        P = self._load_mrk_positions_world(mrkPath)
        labV_raw = self._project_labels(V, P, labels)
        labV_smooth = self._smooth_labels(V, F, labV_raw.copy(), smooth_iters)
      except Exception as exc:
        export_failed.add(meshPath)
        failures.append({"mesh": meshPath, "markups": mrkPath,
                         "stage": "label projection", "error": f"{type(exc).__name__}: {exc}"})
        continue
      if write_vtp:
        try:
          output_files.append(self._write_vtp_with_labels_world(
            meshPath, out_dir, labV_raw, labV_smooth, pdW=pdW))
        except Exception as exc:
          export_failed.add(meshPath)
          failures.append({"mesh": meshPath, "markups": mrkPath,
                           "stage": "VTP export", "error": f"{type(exc).__name__}: {exc}"})
      if write_ply_segments:
        try:
          report = self._write_ply_segments(pdW, out_dir, labV_smooth, os.path.basename(meshPath))
        except Exception as exc:
          export_failed.add(meshPath)
          failures.append({"mesh": meshPath, "markups": mrkPath,
                           "stage": "PLY export", "error": f"{type(exc).__name__}: {exc}"})
          report = None
        if report is not None:
          segment_reports.append(dict(report, mesh=meshPath))
          output_files.extend(report['files'])
          for error in report['failures']:
            export_failed.add(meshPath)
            failures.append({"mesh": meshPath, "markups": mrkPath,
                             "stage": "PLY export", "error": error})
          if report['excluded_cells'] or report['empty_segments']:
            warnings.append(
              f"{os.path.basename(meshPath)}: PLY interiors omit {report['excluded_cells']} "
              f"mixed/unlabeled cells; segments with no interior cells: {report['empty_segments']}. "
              "Use labeled VTP for the complete surface.")
      if i < previewN:
        try:
          self._preview_in_scene(n_name=os.path.splitext(os.path.basename(meshPath))[0], labV=labV_smooth, pdW=pdW)
        except Exception as exc:
          warnings.append(f"{os.path.basename(meshPath)}: preview failed: {exc}")

    lookup_files = self._save_labels_lookup(out_dir, labels)
    return {
      "n_pairs": len(pairs),
      "n_used": len(used_pairs),
      "n_skipped": len(pairs) - len(used_pairs),
      "n_export_failed": len(export_failed),
      "failures": failures,
      "warnings": warnings,
      "output_files": output_files,
      "lookup_files": lookup_files,
      "segment_reports": segment_reports,
      "k": int(k),
      "n_segments": int(len(np.unique(labels))),
      "wrote": len(output_files),
      "M": int(M),
      "Q": float(Q)
    }


  def _pair_files(self, meshes_dir, mrk_dir):
    """Pair case-insensitively, rejecting every ambiguous/output-colliding stem."""
    ex_mesh = {".ply", ".stl", ".obj", ".vtp", ".vtk"}
    def index_files(directory, is_supported, stem):
      by_base = {}
      for name in sorted(os.listdir(directory)):
        if not is_supported(name) or not os.path.isfile(os.path.join(directory, name)):
          continue
        key = stem(name).casefold()
        if key in by_base:
          raise ValueError(f"Duplicate specimen stem in '{directory}': "
                           f"'{os.path.basename(by_base[key])}' and '{name}'. "
                           "Keep only one mesh and one markup file per stem.")
        by_base[key] = os.path.join(directory, name)
      return by_base
    meshes = index_files(meshes_dir, lambda n: os.path.splitext(n)[1].lower() in ex_mesh,
                         lambda n: os.path.splitext(n)[0])
    markups = index_files(mrk_dir, lambda n: n.lower().endswith(".mrk.json"),
                          lambda n: n[:-len(".mrk.json")])
    return [(meshes[key], markups[key]) for key in sorted(meshes.keys() & markups.keys())]

  def _validate_mrk_counts(self, mrk_paths):
    c=None
    for p in mrk_paths:
      n=len(self._load_mrk_positions_world(p))
      if c is None: c=n
      elif n!=c: return False,None
    return True,c

  def _ids_of_class(self, className):
    col = slicer.mrmlScene.GetNodesByClass(className); col.UnRegister(None)
    return {col.GetItemAsObject(i).GetID() for i in range(col.GetNumberOfItems())}

  def _new_node_of_class(self, className, before_ids):
    col = slicer.mrmlScene.GetNodesByClass(className); col.UnRegister(None)
    for i in range(col.GetNumberOfItems()):
      n = col.GetItemAsObject(i)
      if n.GetID() not in before_ids:
        return n
    return None

  def _load_model_node(self, path):
    before = self._ids_of_class("vtkMRMLModelNode")
    node = None
    try:
      node = slicer.util.loadModel(path, properties={"coordinateSystem":"RAS"})
    except TypeError:  # very old Slicer: no properties kw
      ok = slicer.util.loadModel(path)
      if not ok: raise RuntimeError(f"Failed to load model: {path}")
    if node is True or node is False or node is None:
      node = self._new_node_of_class("vtkMRMLModelNode", before)
    if node is None: raise RuntimeError(f"Model loaded but not found in scene: {path}")
    return node

  def _load_markups_node(self, path):
    before = self._ids_of_class("vtkMRMLMarkupsNode")
    node = None
    try:
      node = slicer.util.loadMarkups(path, properties={"coordinateSystem":"RAS"})
    except TypeError:
      ok = slicer.util.loadMarkups(path)
      if not ok: raise RuntimeError(f"Failed to load markups: {path}")
    if node is True or node is False or node is None:
      node = self._new_node_of_class("vtkMRMLMarkupsNode", before)
    if node is None: raise RuntimeError(f"Markups loaded but not found in scene: {path}")
    return node

  def _polydata_world(self, modelNode, max_drop_frac=0.01):
    """
    Load polydata in world coords, clean, triangulate, and if any vertex
    is non-finite, drop those vertices and incident triangles.
    If the repair would remove > max_drop_frac of vertices, raise.
    """
    pd = modelNode.GetPolyData()

    # Apply world transform if any
    t = modelNode.GetParentTransformNode()
    if t:
      tf = vtk.vtkGeneralTransform()
      t.GetTransformToWorld(tf)
      f = vtk.vtkTransformPolyDataFilter()
      f.SetInputData(pd); f.SetTransform(tf); f.Update()
      pd = f.GetOutput()

    # Clean degenerates/duplicates then triangulate
    cl = vtk.vtkCleanPolyData()
    cl.SetInputData(pd)
    cl.ConvertStripsToPolysOn()
    cl.Update()
    pd = cl.GetOutput()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(pd)
    tri.PassLinesOff(); tri.PassVertsOff()
    tri.Update()
    pd = tri.GetOutput()

    # Check finiteness; attempt repair if needed
    V = vtk_np.vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
    if np.isfinite(V).all():
      return pd

    # --- repair: drop non-finite vertices & incident faces, then renumber ---
    keep = np.isfinite(V).all(axis=1)
    n_bad = int((~keep).sum()); n_tot = int(len(keep))
    drop_frac = n_bad / max(n_tot, 1)

    if drop_frac > max_drop_frac:
      raise RuntimeError(
        f"Mesh '{modelNode.GetName()}' has {n_bad}/{n_tot} non-finite vertices "
        f"({drop_frac:.2%} > {max_drop_frac:.2%}). Aborting to avoid heavy corruption."
      )

    # map old->new
    old2new = -np.ones(n_tot, dtype=np.int64)
    old2new[keep] = np.arange(keep.sum(), dtype=np.int64)

    # rebuild triangles using only kept points
    cell_ids = pd.GetPolys()
    idl = vtk.vtkIdList(); cell_ids.InitTraversal()
    Fnew = []
    while cell_ids.GetNextCell(idl):
      if idl.GetNumberOfIds() == 3:
        a, b, c = idl.GetId(0), idl.GetId(1), idl.GetId(2)
        if keep[a] and keep[b] and keep[c]:
          Fnew.append([old2new[a], old2new[b], old2new[c]])

    Fnew = np.asarray(Fnew, dtype=np.int32)
    if Fnew.size == 0:
      raise RuntimeError(
        f"Mesh '{modelNode.GetName()}' repair removed all triangles."
      )

    # build new polydata
    newPts = vtk.vtkPoints()
    Vkeep = V[keep]
    newPts.SetNumberOfPoints(Vkeep.shape[0])
    for i, p in enumerate(Vkeep):
      newPts.SetPoint(i, float(p[0]), float(p[1]), float(p[2]))

    newPolys = vtk.vtkCellArray()
    for a, b, c in Fnew:
      newPolys.InsertNextCell(3)
      newPolys.InsertCellPoint(int(a)); newPolys.InsertCellPoint(int(b)); newPolys.InsertCellPoint(int(c))

    newpd = vtk.vtkPolyData()
    newpd.SetPoints(newPts); newpd.SetPolys(newPolys)
    newpd.BuildCells(); newpd.BuildLinks()

    logging.warning(
      f"[MorphoWeaveSurfaceSegmentation] Repaired mesh '{modelNode.GetName()}': "
      f"removed {n_bad}/{n_tot} non-finite vertices ({drop_frac:.2%})."
    )
    return newpd



  def _load_mesh_VF_world(self, path):
    n=self._load_model_node(path); pdW=self._polydata_world(n)
    V=vtk_np.vtk_to_numpy(pdW.GetPoints().GetData()).astype(np.float64)
    c=pdW.GetPolys(); idl=vtk.vtkIdList(); c.InitTraversal(); F=[]
    while c.GetNextCell(idl):
      if idl.GetNumberOfIds()==3: F.append([idl.GetId(0),idl.GetId(1),idl.GetId(2)])
    F=np.asarray(F,np.int32); slicer.mrmlScene.RemoveNode(n); return V,F,pdW

  def _load_mrk_positions_world(self, path):
    n=self._load_markups_node(path); N=n.GetNumberOfControlPoints(); P=np.zeros((N,3),float)
    for i in range(N): x=[0,0,0]; n.GetNthControlPointPositionWorld(i,x); P[i]=x
    slicer.mrmlScene.RemoveNode(n); return P

  def _vertex_normals_polydata(self,pd):
    nn=vtk.vtkPolyDataNormals(); nn.SetInputData(pd); nn.SplittingOff(); nn.ConsistencyOn(); nn.Update()
    return vtk_np.vtk_to_numpy(nn.GetOutput().GetPointData().GetNormals()).astype(np.float64)

  def _stable_normals(self,pd):
    N=self._vertex_normals_polydata(pd); V=vtk_np.vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
    c=V.mean(0); flip=(np.einsum('ij,ij->i',N,V-c)<0); N[flip]*=-1.0; return N

  def _adj_from_faces(self,F,nv):
    I=np.concatenate([F[:,0],F[:,1],F[:,2],F[:,0],F[:,1],F[:,2]])
    J=np.concatenate([F[:,1],F[:,2],F[:,0],F[:,2],F[:,0],F[:,1]])
    A=coo_matrix((np.ones_like(I), (I,J)), shape=(nv,nv)).tocsr(); A=A.maximum(A.T); A.setdiag(0); A.eliminate_zeros(); return A

  def _umbrella_curv_proxy(self,V,A):
    deg=np.asarray(A.sum(1)).ravel(); L=diags(deg)-A; Hn=(L@V); H=np.linalg.norm(Hn,axis=1)/np.maximum(deg,1); return H

  def _knn_graph(self, P, knn):
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or not np.isfinite(P).all():
      raise ValueError("Locus coordinates must be a finite N-by-3 array.")
    if int(knn) != knn or knn < 1:
      raise ValueError("kNN must be a positive integer.")
    n = len(P)
    if n < 2:
      return coo_matrix((n, n)).tocsr()
    width = min(int(knn), n - 1)
    distances, neighbors = cKDTree(P).query(P, k=width + 1)
    # Remove the query vertex explicitly: coincident loci can change tie order.
    idx = np.empty((n, width), dtype=np.int64)
    d = np.empty((n, width), dtype=float)
    for row in range(n):
      keep = neighbors[row] != row
      idx[row] = neighbors[row][keep][:width]
      d[row] = distances[row][keep][:width]
    I = np.repeat(np.arange(n), width); J = idx.ravel()
    sig = np.median(d) + 1e-12
    weights = np.exp(-(d.ravel() ** 2) / (sig ** 2))
    A = coo_matrix((weights, (I, J)), shape=(n, n)).tocsr()
    A = A.maximum(A.T); A.setdiag(0); A.eliminate_zeros()
    return A

  def _loci_kernel_multi(self,loci, feats_stack, A_geo, alpha):
    """
    Self-tuning Gaussian on features + spatial term.
    feats_stack: S×M×D (already per-mesh z-scored in _locus_geofeats)
    """
    M = loci.shape[0]
    coo = A_geo.tocoo(); I, J = coo.row, coo.col
    und = I < J
    Iu, Ju = I[und], J[und]

    F = feats_stack  # S×M×D
    # robust edge feature distance (median across S)
    d2 = ((F[:, Iu, :] - F[:, Ju, :])**2).sum(axis=2)  # S×E
    df = np.sqrt(np.median(d2, axis=0) + 1e-12)        # E

    # per-node scales: sigma_i = median df over incident edges
    nodes = np.r_[Iu, Ju]; vals = np.r_[df, df]
    order = np.argsort(nodes); nodes = nodes[order]; vals = vals[order]
    uniq, counts = np.unique(nodes, return_counts=True)
    cuts = np.r_[0, np.cumsum(counts)]
    sigma = np.zeros(M, float)
    for u, s, e in zip(uniq, cuts[:-1], cuts[1:]):
      sigma[u] = np.median(vals[s:e]) if e > s else 1.0
    s_ij = sigma[Iu]*sigma[Ju] + 1e-12

    Wf = np.exp(-(df*df)/s_ij)

    # spatial (Euclidean on loci_mean) for gentle contiguity
    dx = np.linalg.norm(loci[Iu] - loci[Ju], axis=1)
    sx = np.median(dx) + 1e-12
    Wx = np.exp(-(dx*dx)/(sx*sx))

    We = alpha*Wf + (1.0 - alpha)*Wx
    # symmetrize
    Iall = np.r_[Iu, Ju]; Jall = np.r_[Ju, Iu]; Wall = np.r_[We, We]
    W = coo_matrix((Wall, (Iall, Jall)), shape=(M, M)).tocsr()
    W.setdiag(0); W.eliminate_zeros()
    return W

  def _validate_cluster_count(self, n, k):
    if n < 3:
      raise ValueError("Surface segmentation requires at least 3 corresponding loci per specimen.")
    if int(k) != k or not 2 <= k < n:
      raise ValueError(f"Number of segments must be an integer between 2 and {n - 1} for {n} loci.")

  def _spectral_cluster(self, W, k):
    n = W.shape[0]
    self._validate_cluster_count(n, k)
    if W.shape != (n, n):
      raise ValueError("The locus affinity matrix must be square.")
    W = W.tocsr()
    if not np.isfinite(W.data).all() or np.any(W.data < 0):
      raise ValueError("Locus affinities must be finite and nonnegative.")
    if (W - W.T).nnz and np.max(np.abs((W - W.T).data)) > 1e-10:
      raise ValueError("The locus affinity matrix must be symmetric.")
    degree = np.asarray(W.sum(1)).ravel()
    if np.any(degree <= 0):
      raise ValueError("Every locus needs a positive-weight graph neighbor; increase kNN or check coordinates.")
    invsqrt = diags(1.0 / np.sqrt(degree))
    L = diags(np.ones(n)) - invsqrt @ W @ invsqrt
    _, vecs = eigsh(L, k=int(k), which='SM', tol=1e-6)
    X = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    _, labels = kmeans2(X, int(k), minit='++', iter=60)
    return labels.astype(np.int32)

  def _project_labels(self,V,P_loci,labels):
    nn=cKDTree(P_loci).query(V,k=1)[1]; return labels[nn]

  def _smooth_labels(self,V,F,lab,iters):
    if iters<=0: return lab
    A=self._adj_from_faces(F,V.shape[0]); deg=np.asarray(A.sum(1)).ravel()
    for _ in range(iters):
      coo=A.tocoo(); I,J=coo.row,coo.col; K=lab.max()+1
      votes=np.zeros((V.shape[0],K),np.int32); np.add.at(votes,(I,lab[J]),1)
      maj=np.argmax(votes,1); lab=np.where(deg>0,maj,lab)
    return lab

  def _checked_polydata_write(self, writer, pd, out_path):
    """Write a new temporary file, then replace the destination only on success."""
    if pd.GetNumberOfPoints() == 0 or pd.GetNumberOfCells() == 0:
      raise RuntimeError(f"No surface data to write: {out_path}")
    fd, temporary = tempfile.mkstemp(prefix=".morphoweave-", suffix=os.path.splitext(out_path)[1],
                                     dir=os.path.dirname(os.path.abspath(out_path)))
    os.close(fd)
    try:
      writer.SetFileName(temporary); writer.SetInputData(pd)
      status = writer.Write()
      if status != 1 or writer.GetErrorCode() != 0 or not os.path.isfile(temporary) or os.path.getsize(temporary) == 0:
        raise IOError(f"Failed to write mesh: {out_path} (VTK error {writer.GetErrorCode()}).")
      os.replace(temporary, out_path)
      return out_path
    finally:
      if os.path.exists(temporary):
        os.remove(temporary)

  def _validate_vertex_labels(self, pd, labels):
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != pd.GetNumberOfPoints():
      raise ValueError("Exactly one label is required for each mesh vertex.")
    if (not np.isfinite(values).all() or np.any(values != np.floor(values))
        or np.any(values < -1) or np.any(values > np.iinfo(np.int32).max)):
      raise ValueError("Vertex labels must be finite integers from -1 through INT32_MAX.")
    return values.astype(np.int32)

  def _write_vtp_with_labels_world(self, meshPath, out_dir, labV_raw, labV_smooth, pdW=None):
    node = None
    try:
      if pdW is None:
        node = self._load_model_node(meshPath)
        pdW = self._polydata_world(node)
      pd = vtk.vtkPolyData(); pd.DeepCopy(pdW)
      for name, labels in (("SegID", labV_raw), ("SegID_smooth", labV_smooth)):
        values = self._validate_vertex_labels(pd, labels)
        arr = vtk_np.numpy_to_vtk(values, deep=True, array_type=vtk.VTK_INT)
        arr.SetName(name); pd.GetPointData().AddArray(arr)
      pd.GetPointData().SetActiveScalars("SegID_smooth")
      base = os.path.splitext(os.path.basename(meshPath))[0]
      out = os.path.join(out_dir, base + "_seg.vtp")
      return self._checked_polydata_write(vtk.vtkXMLPolyDataWriter(), pd, out)
    finally:
      if node is not None:
        slicer.mrmlScene.RemoveNode(node)

  def _write_ply_segments(self, pdW, out_dir, labels, meshBasename):
    """Optional eroded interiors, NOT an exhaustive partition of the surface.

    A cell is retained only when all its vertices have the same nonnegative
    label. Mixed-label cells are deliberately omitted and reported; complete
    geometry and both point-label arrays are available in the labeled VTP.
    """
    labels = self._validate_vertex_labels(pdW, labels)
    pd = vtk.vtkPolyData(); pd.DeepCopy(pdW)
    arr = vtk_np.numpy_to_vtk(labels, deep=True, array_type=vtk.VTK_INT)
    arr.SetName("SegID"); pd.GetPointData().SetScalars(arr)
    base = os.path.splitext(meshBasename)[0]
    report = {"files": [], "failures": [], "empty_segments": [],
              "total_cells": pd.GetNumberOfCells(), "interior_cells": 0,
              "written_cells": 0, "excluded_cells": 0}
    for seg in np.unique(labels):
      if seg < 0: continue
      t = vtk.vtkThreshold(); t.SetInputData(pd)
      t.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "SegID")
      t.SetLowerThreshold(float(seg) - 0.5); t.SetUpperThreshold(float(seg) + 0.5)
      t.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
      t.AllScalarsOn(); t.Update()
      g = vtk.vtkGeometryFilter(); g.SetInputConnection(t.GetOutputPort()); g.Update()
      count = g.GetOutput().GetNumberOfCells()
      report['interior_cells'] += count
      if count == 0:
        report['empty_segments'].append(int(seg))
        continue
      nf = vtk.vtkPolyDataNormals(); nf.SetInputConnection(g.GetOutputPort())
      nf.SplittingOff(); nf.ConsistencyOn(); nf.Update()
      writer = vtk.vtkPLYWriter(); writer.SetFileTypeToBinary()
      path = os.path.join(out_dir, f"{base}_seg_{int(seg):02d}.ply")
      try:
        report['files'].append(self._checked_polydata_write(writer, nf.GetOutput(), path))
        report['written_cells'] += count
      except Exception as exc:
        report['failures'].append(f"{os.path.basename(path)}: {type(exc).__name__}: {exc}")
    report['excluded_cells'] = report['total_cells'] - report['interior_cells']
    return report

  def _preview_in_scene(self, n_name, labV, pdW):
      pd = vtk.vtkPolyData(); pd.DeepCopy(pdW)
      nf = vtk.vtkPolyDataNormals(); nf.SetInputData(pd)
      nf.SplittingOff(); nf.ConsistencyOn(); nf.Update()
      pd = nf.GetOutput()
      a = vtk_np.numpy_to_vtk(labV, deep=True, array_type=vtk.VTK_INT); a.SetName("SegID_smooth")
      pd.GetPointData().AddArray(a); pd.GetPointData().SetActiveScalars("SegID_smooth")
      n = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
      n.SetName(f"{n_name}_seg_preview")
      n.SetAndObservePolyData(pd)
      n.CreateDefaultDisplayNodes()
      dn = n.GetDisplayNode()
      k = max(int(labV.max()+1), 1)
      ctn = self._ensure_color_table("MorphoWeaveSurfaceSegmentation_Colors", k)
      if hasattr(dn, "SetAndObserveColorNodeID"): dn.SetAndObserveColorNodeID(ctn.GetID())
      try:
          flag = getattr(dn, "ScalarRangeFlagUseColorNode", None)
          if flag is not None: dn.SetScalarRangeFlag(flag)
          else: dn.SetScalarRange(0.0, float(k-1)); dn.SetAutoScalarRange(False)
      except Exception:
          dn.SetScalarRange(0.0, float(k-1)); dn.SetAutoScalarRange(False)
      dn.SetScalarVisibility(True)

  def _clear_preview_nodes(self):
    nodes=slicer.mrmlScene.GetNodesByClass("vtkMRMLModelNode")
    for i in range(nodes.GetNumberOfItems()):
      node=nodes.GetItemAsObject(i)
      if node and node.GetName().endswith("_seg_preview"): slicer.mrmlScene.RemoveNode(node)

  def _save_labels_lookup(self,out_dir,labels):
    paths = [os.path.join(out_dir, "MorphoWeaveSurfaceSegmentation_locus_labels" + ext)
             for ext in (".npy", ".txt")]
    np.save(paths[0], labels)
    with open(paths[1], "w") as f: f.write("\n".join(map(str, labels.tolist())))
    return paths

  def _ensure_color_table(self, name, k):
    ctn = slicer.mrmlScene.GetFirstNodeByName(name)
    if ctn is None:
      ctn = slicer.vtkMRMLColorTableNode()
      ctn.SetTypeToUser()
      ctn.SetName(name)
      slicer.mrmlScene.AddNode(ctn)
    # make sure we have enough swatches
    k = max(int(k), 1)
    ctn.SetNumberOfColors(k)
    cols = self._glasbey(k)
    for i, (r, g, b) in enumerate(cols[:k]):
      ctn.SetColor(i, float(r), float(g), float(b), 1.0)
      try: ctn.SetColorName(i, f"Seg {i}")
      except AttributeError: pass  # older Slicer
    return ctn

  def _glasbey(self, n):
    base = [
      (0.0,0.0,0.0),(0.9,0.6,0.0),(0.35,0.7,0.9),(0.0,0.62,0.45),(0.8,0.47,0.65),
      (0.95,0.9,0.25),(0.0,0.45,0.7),(0.8,0.4,0.0),(0.36,0.2,0.8),(0.2,0.8,0.2),
      (0.8,0.2,0.2),(0.2,0.6,0.9),(0.9,0.3,0.5),(0.6,0.6,0.0),(0.0,0.7,0.7),
      (0.6,0.3,0.9),(0.9,0.6,0.6),(0.4,0.8,0.6),(0.6,0.4,0.2),(0.2,0.4,0.6),
      (0.9,0.8,0.4),(0.4,0.2,0.6),(0.2,0.8,0.8),(0.8,0.6,0.2),(0.6,0.2,0.4),
      (0.2,0.6,0.2),(0.6,0.2,0.8),(0.2,0.2,0.8),(0.8,0.2,0.6),(0.2,0.8,0.4),
      (0.4,0.6,0.8),(0.8,0.4,0.6)
    ]
    if n <= len(base): return base[:n]
    out = base[:]
    φ = 0.61803398875; h = 0.0
    while len(out) < n:
      h = (h + φ) % 1.0; s = 0.65; v = 0.95
      r,g,b = self._hsv2rgb(h,s,v); out.append((r,g,b))
    return out[:n]

  def _hsv2rgb(self, h, s, v):
    i = int(h*6.0); f = h*6.0 - i; i %= 6
    p = v*(1.0-s); q = v*(1.0-s*f); t = v*(1.0-s*(1.0-f))
    return [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][i]


  def _mean_loci(self, mrk_paths, M):
    S = len(mrk_paths)
    Q = np.empty((S, M, 3), float)
    for i, p in enumerate(mrk_paths):
      P = self._load_mrk_positions_world(p)
      if P.shape[0] != M:
        raise RuntimeError(f"Markups '{os.path.basename(p)}' has {P.shape[0]} points; expected {M}.")
      Q[i] = P
    return Q.mean(axis=0)

  def _vtk_curvature(self, pd, ctype="Mean"):
    curv = vtk.vtkCurvatures(); curv.SetInputData(pd)
    if ctype=="Mean": curv.SetCurvatureTypeToMean()
    elif ctype=="Gaussian": curv.SetCurvatureTypeToGaussian()
    else: raise ValueError("ctype must be Mean or Gaussian")
    curv.Update()
    a = curv.GetOutput().GetPointData().GetScalars()
    return vtk_np.vtk_to_numpy(a).astype(np.float64)

  def _principal_from_HK(self, H, K):
    # k1,k2 are solutions of t^2 - 2H t + K = 0
    disc = np.maximum(H*H - K, 0.0)
    r = np.sqrt(disc)
    k1 = H + r
    k2 = H - r
    # enforce |k1| >= |k2|
    swap = np.abs(k2) > np.abs(k1)
    k1[swap], k2[swap] = k2[swap], k1[swap]
    return k1, k2

  def _shape_index_curvedness(self, k1, k2):
    # The separate principal-curvature features retain magnitude ordering.
    # Shape index instead requires signed ordering: kmax - kmin >= 0.
    # atan2(0, 0) defines the flat-point convention S=0, C=0.
    kmax, kmin = np.maximum(k1, k2), np.minimum(k1, k2)
    S = (2.0 / np.pi) * np.arctan2(kmax + kmin, kmax - kmin)
    C = np.sqrt(0.5 * (kmax*kmax + kmin*kmin))
    return S, C

  def _smooth_polydata(self, pd, iters):
    if iters<=0: return pd
    f = vtk.vtkWindowedSincPolyDataFilter()
    f.SetInputData(pd); f.SetNumberOfIterations(int(iters))
    f.BoundarySmoothingOff(); f.FeatureEdgeSmoothingOff()
    f.NormalizeCoordinatesOn()   # mildly scale-invariant
    f.NonManifoldSmoothingOn(); f.Update()
    return f.GetOutput()

  def _face_normals(self, V, F):
    v0, v1, v2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    n = np.cross(v1-v0, v2-v0)
    l = np.linalg.norm(n, axis=1, keepdims=True)+1e-12
    return n/l

  def _dihedral_vertex(self, V, F):
    # per-edge dihedral, then average abs(dihedral) at incident vertices
    # build halfedge map
    from collections import defaultdict
    FN = self._face_normals(V, F)
    adj = defaultdict(list)
    for f,(a,b,c) in enumerate(F):
      adj[tuple(sorted((a,b)))].append(f)
      adj[tuple(sorted((b,c)))].append(f)
      adj[tuple(sorted((c,a)))].append(f)
    E_keys = list(adj.keys())
    dih = np.zeros(len(E_keys), float)
    for i,e in enumerate(E_keys):
      fs = adj[e]
      if len(fs)==2:
        f1,f2 = fs
        cosang = np.clip((FN[f1]*FN[f2]).sum(), -1.0, 1.0)
        dih[i] = np.arccos(cosang)  # [0,pi]
      else:
        dih[i] = 0.0
    # accumulate to vertices
    acc = np.zeros(V.shape[0], float); cnt = np.zeros(V.shape[0], float)
    for i,(a,b) in enumerate(E_keys):
      val = dih[i]
      acc[a]+=val; acc[b]+=val; cnt[a]+=1; cnt[b]+=1
    return acc/np.maximum(cnt,1.0)

  def _unique_edges_from_faces(self, F):
    # 3 edges per triangle, canonicalize (low, high), then unique
    E = np.vstack([F[:,[0,1]], F[:,[1,2]], F[:,[2,0]]]).astype(np.int64)
    E.sort(axis=1)
    return np.unique(E, axis=0)

  def _normal_variation(self, V, F, N):
    """
    Mean 1-ring normal deviation per vertex (radians).
    V: (nv,3), F: (nf,3), N: (nv,3) unit normals
    """
    E = self._unique_edges_from_faces(F)             # (m, 2)
    # angle between normals across each edge
    cosang = np.clip((N[E[:,0]] * N[E[:,1]]).sum(axis=1), -1.0, 1.0)
    ang = np.arccos(cosang)                          # (m,)
    acc = np.zeros(V.shape[0], dtype=np.float64)
    cnt = np.zeros(V.shape[0], dtype=np.float64)
    # accumulate to both end vertices
    np.add.at(acc, E[:,0], ang); np.add.at(acc, E[:,1], ang)
    np.add.at(cnt, E[:,0], 1.0); np.add.at(cnt, E[:,1], 1.0)
    return acc / np.maximum(cnt, 1.0)


  def _tooth_axis(self, V):
    # PCA axis for orientation (rough tooth axis)
    X = V - V.mean(0)
    C = X.T @ X
    w, U = np.linalg.eigh(C)
    axis = U[:, np.argmax(w)]
    axis = axis/np.linalg.norm(axis)
    # make it stable w.r.t. outward normals (optional flip)
    return axis

  def _locus_geofeats(self, pdW, V, F, vidx, scales=(0,4,12),
                      include_dihedral=True, include_normal_var=True, include_axis_orient=True,
                      pool_hops=1, include_hks=False, hks_k=30, hks_times=6):
    """
    Returns [M, D] per-locus features; robustly pooled over a small ring.
    """
    # stable per-vertex helpers (computed once)
    N0   = self._stable_normals(pdW)
    axis = self._tooth_axis(V) if include_axis_orient else None
    Dv   = self._dihedral_vertex(V,F) if include_dihedral else None
    NV   = self._normal_variation(V,F,N0) if include_normal_var else None
    A    = self._adj_from_faces(F, V.shape[0])

    # ----- core multiscale curvature block (S, C, k1, k2) -----
    feats_by_scale = []
    for s in scales:
      pdS = self._smooth_polydata(pdW, s)
      H   = self._vtk_curvature(pdS, "Mean")
      K   = self._vtk_curvature(pdS, "Gaussian")
      k1, k2 = self._principal_from_HK(H, K)
      S, C   = self._shape_index_curvedness(k1, k2)

      # pooled values at loci
      sc_feats = []
      for v0 in vidx:
        m = self._ring_mask(A, int(v0), hops=pool_hops)
        sc_feats.append([
          np.median(S[m]), np.median(C[m]),
          np.median(k1[m]), np.median(k2[m])
        ])
      feats_by_scale.append(np.asarray(sc_feats, float))  # [M,4]

    Fms = np.concatenate(feats_by_scale, axis=1)  # [M, 4*len(scales)]

    # ----- extras (ridge/valley & orientation), pooled -----
    extras = []
    if include_dihedral:
      x = []
      for v0 in vidx:
        m = self._ring_mask(A, int(v0), hops=pool_hops)
        x.append(np.median(Dv[m]))
      extras.append(np.asarray(x, float)[:, None])
    if include_normal_var:
      x = []
      for v0 in vidx:
        m = self._ring_mask(A, int(v0), hops=pool_hops)
        x.append(np.median(NV[m]))
      extras.append(np.asarray(x, float)[:, None])
    if include_axis_orient:
      x = []
      for v0 in vidx:
        m = self._ring_mask(A, int(v0), hops=pool_hops)
        x.append(np.mean(N0[m] @ axis))
      extras.append(np.asarray(x, float)[:, None])

    # ----- optional HKS on this mesh (intrinsic, robust) -----
# ----- optional HKS on this mesh (intrinsic, robust) -----
    # ----- optional HKS on this mesh -----
    if include_hks:
      H_med = None
      try:
        HKS = self._hks_at_vertices(V, F, k=hks_k, n_times=hks_times)  # [nv, t]
        X = []
        for v0 in vidx:
          m = self._ring_mask(A, int(v0), hops=pool_hops)
          X.append(np.median(HKS[m], axis=0))
        H_med = np.asarray(X, float)  # [M, hks_times]
      except Exception as e:
        logging.warning(f"[MorphoWeaveSurfaceSegmentation] HKS failed on '{os.path.basename(getattr(self, '_current_mesh', 'mesh'))}': {e} — filling zeros.")
        H_med = np.zeros((len(vidx), hks_times), dtype=float)
      extras.append(H_med)

    if extras:
      Fms = np.concatenate([Fms] + extras, axis=1)

    # per-mesh z-score across loci (keeps cross-mesh comparability)
    mu = Fms.mean(0, keepdims=True)
    sd = Fms.std(0, keepdims=True) + 1e-8
    return (Fms - mu) / sd


  def _ring_mask(self, A, seed, hops=1):
    mask = np.zeros(A.shape[0], dtype=bool)
    frontier = np.zeros_like(mask); frontier[seed] = True; mask[seed] = True
    for _ in range(hops):
      nxt = (A @ frontier.astype(np.uint8)) > 0
      nxt &= ~mask
      mask |= nxt
      frontier = nxt
    return mask

  def _cotangent_laplacian(self, V, F):
    # build cotangent L and barycentric M
    nv = V.shape[0]
    i, j, k = F[:,0], F[:,1], F[:,2]
    v0, v1, v2 = V[i], V[j], V[k]
    def cot(a, b, c):
      u, v = b - a, c - a
      cu = np.einsum('ij,ij->i', u, v)
      su = np.linalg.norm(np.cross(u, v), axis=1) + 1e-12
      return cu / su
    # Angle at i is opposite edge (j, k), and likewise cyclically.
    cotA = cot(v0, v1, v2)
    cotB = cot(v1, v2, v0)
    cotC = cot(v2, v0, v1)
    I = np.r_[i,j,j,k,k,i]
    J = np.r_[j,i,k,j,i,k]
    W = np.r_[cotC,cotC,cotA,cotA,cotB,cotB] * 0.5
    L = coo_matrix((W, (I, J)), shape=(nv, nv)).tocsr()
    L = diags(np.array(L.sum(1)).ravel()) - L  # L = D - W

    # barycentric (lumped) mass
    A2 = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) * 0.5
    Mdiag = np.zeros(nv, float)
    np.add.at(Mdiag, i, A2/3.0); np.add.at(Mdiag, j, A2/3.0); np.add.at(Mdiag, k, A2/3.0)
    Mdiag = np.maximum(Mdiag, 1e-12)   # avoid exact zeros
    M = diags(Mdiag)
    return L, M

  def _lb_eigens(self, V, F, k=30):
    if len(V) < 3 or len(F) == 0:
      raise ValueError("HKS requires at least 3 vertices and a triangle.")
    if int(k) != k or k < 2:
      raise ValueError("HKS eigenvector count must be an integer of at least 2.")
    L, M = self._cotangent_laplacian(V, F)
    # eigsh requires k < N, including on a single-triangle surface.
    kk = min(int(k), L.shape[0] - 1)
    vals, vecs = eigsh(L, k=kk, M=M, sigma=1e-8, which='LM', tol=1e-6)
    vals = np.maximum(vals, 1e-12)
    return vals, vecs

  def _hks_at_vertices(self, V, F, k=30, n_times=6):
    evals, evecs = self._lb_eigens(V, F, k=k)   # evals ~ [k]
    # pick times across decades, scaled by mean edge length^2
    E = np.vstack([F[:,[0,1]], F[:,[1,2]], F[:,[2,0]]])
    Ls = np.linalg.norm(V[E[:,0]] - V[E[:,1]], axis=1)
    Lm2 = (np.median(Ls)**2 + 1e-12)
    tmin, tmax = 1e-3*Lm2, 1e-1*Lm2
    ts = np.geomspace(tmin, tmax, num=n_times)
    Phi2 = evecs**2  # [nv,k]
    H = Phi2 @ np.exp(-np.outer(evals, ts))  # [nv, n_times]
    # per-mesh zscore across vertices (like other channels)
    mu = H.mean(0, keepdims=True); sd = H.std(0, keepdims=True) + 1e-8
    return (H - mu) / sd

  def _modularity(self, W, labels):
    """
    Newman-Girvan modularity Q for a weighted, undirected graph W and integer labels.
    """
    A = W.tocsr()
    two_m = A.sum()  # = 2m
    if two_m <= 0: return 0.0
    kdeg = np.asarray(A.sum(1)).ravel()
    Q = 0.0
    for c in np.unique(labels):
      idx = (labels == c)
      Ec = A[idx][:, idx].sum()          # total internal weight
      Kc = kdeg[idx].sum()
      Q += (Ec / two_m) - (Kc / two_m)**2
    return float(Q)

  def _choose_k_alpha(self, loci_mean, feats_stack, A_geo, k_grid, alpha_grid):
    """
    Tiny grid-search over (k, alpha). Returns (k_best, alpha_best, Q_best, labels, W_best).
    """
    best = None
    for a in alpha_grid:
      W = self._loci_kernel_multi(loci_mean, feats_stack, A_geo, a)
      for k in k_grid:
        labels = self._spectral_cluster(W, k)
        Q = self._modularity(W, labels)
        logging.info(f"[AutoTune] k={k} α={a:.3f} -> Q={Q:.4f}")
        if (best is None) or (Q > best[2]):
          best = (k, a, Q, labels, W)
    return best

  def _clean_feats(self, feats_stack):
    # feats_stack: S x M x D
    S, M, D = feats_stack.shape
    X = feats_stack.reshape(S*M, D)
    bad = ~np.isfinite(X)
    if bad.any():
      col_med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
      X[bad] = np.take(col_med, np.where(bad)[1])
      logging.info(f"[MorphoWeaveSurfaceSegmentation] Cleaned NaN/Inf in features.")
    return X.reshape(S, M, D)

  def _whiten_feats(self, feats_stack, eps=1e-6):
    """
    Population whitening across specimens×loci (Mahalanobis preconditioning).
    Returns whitened feats and transform (for provenance).
    """
    S, M, D = feats_stack.shape
    X = feats_stack.reshape(S*M, D)
    mu = X.mean(0)
    X0 = X - mu
    C = np.cov(X0, rowvar=False)
    evals, evecs = np.linalg.eigh(C)
    invsqrt = 1.0 / np.sqrt(np.maximum(evals, eps))
    A = (evecs * invsqrt) @ evecs.T        # D x D
    Y = (X0 @ A.T).reshape(S, M, D)
    return Y, {"whiten_mu": mu, "whiten_A": A}

  def _assert_finite(self, what, arr, path):
    if not np.isfinite(arr).all():
      bad = np.argwhere(~np.isfinite(arr))[:5].tolist()
      raise RuntimeError(
        f"Non-finite values in {what} of '{os.path.basename(path)}' "
        f"(first bad indices: {bad})."
      )


class MorphoWeaveSurfaceSegmentationTest(ScriptedLoadableModuleTest):
  def setUp(self):
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    self.setUp()
    self.assertIsNotNone(MorphoWeaveSurfaceSegmentationLogic())

