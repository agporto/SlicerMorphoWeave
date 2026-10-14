import os, json, shutil, logging, html, csv, tempfile
import numpy as np
from pathlib import Path
from vtk.util import numpy_support

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import connected_components

import slicer, qt, ctk, vtk
from slicer.ScriptedLoadableModule import *


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


#
# MorphoWeaveModelLibrary
#

class MorphoWeaveModelLibrary(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Model Library"
    self.parent.categories = ["MorphoWeave"]
    self.parent.dependencies = []
    self.parent.contributors = ["Arthur Porto"]
    self.parent.helpText = """
    Build a Statistical Shape Model (SSM) and visualize modes of variation on a template.
    Defaults to Local RBF (Gaussian KNN). Optionally use Biharmonic (experimental).
    """
    self.parent.acknowledgementText = ""

#
# MorphoWeaveModelLibraryLogic
#

class MorphoWeaveModelLibraryLogic(ScriptedLoadableModuleLogic):
  @staticmethod
  def build_ssm(shapes, variance_threshold=0.99, max_modes=None):
      """PCA with unit-length modes and covariance eigenvalues in length squared."""
      shapes = np.asarray(shapes, dtype=np.float64)
      if shapes.ndim != 3 or shapes.shape[0] == 0 or shapes.shape[1] == 0 or shapes.shape[2] != 3:
          raise ValueError("Shapes must have nonempty dimensions (specimens, points, 3).")
      if not np.isfinite(shapes).all():
          raise ValueError("Shapes contain non-finite coordinates.")
      if variance_threshold is not None and (not np.isfinite(variance_threshold) or not 0 < variance_threshold <= 1):
          raise ValueError("variance_threshold must be in (0, 1], or None.")
      if max_modes is not None and (isinstance(max_modes, (bool, np.bool_)) or not isinstance(max_modes, (int, np.integer)) or max_modes < 1):
          raise ValueError("max_modes must be a positive integer, or None.")
      n_shapes, n_pts, dim = shapes.shape
      X = shapes.reshape(n_shapes, -1)
      # Center differences rather than summing absolute coordinates: identical
      # inputs remain exactly zero, even when their decimal mean rounds.
      relative = X - X[0]
      offset = relative.mean(axis=0)
      mu = X[0] + offset
      Xc = relative - offset
      mean = mu.reshape(n_pts, dim).astype(np.float32)
      if not np.isfinite(mean).all() or not np.isfinite(Xc).all():
          raise ValueError("Coordinate magnitude exceeds the supported model precision.")
      n, D = Xc.shape
      if n < 2 or not np.any(Xc):
          return mean, np.zeros((n_pts, dim, 0), np.float32), np.zeros(0, np.float64), 0
      if D <= n * 2:
          _, singular_values, vt = np.linalg.svd(Xc, full_matrices=False)
          tolerance = singular_values[0] * max(n, D) * np.finfo(float).eps
          rank = min(n - 1, int(np.count_nonzero(singular_values > tolerance)))
          eigvals = singular_values[:rank] ** 2 / (n - 1)
          modes_flat = vt[:rank].T
      else:
          covariance = (Xc @ Xc.T) / (n - 1)
          evals, U = np.linalg.eigh(covariance)
          order = np.argsort(evals)[::-1]
          evals, U = evals[order], U[:, order]
          tolerance = max(float(evals[0]), 0.0) * max(n, D) * np.finfo(float).eps
          rank = min(n - 1, int(np.count_nonzero(evals > tolerance)))
          eigvals = evals[:rank]
          modes_flat = (Xc.T @ U[:, :rank]) / np.sqrt(eigvals * (n - 1))
      if not np.isfinite(eigvals).all() or np.any(eigvals <= 0):
          raise ValueError("PCA spectrum is outside supported numerical precision.")
      if not rank:
          return mean, np.zeros((n_pts, dim, 0), np.float32), np.zeros(0, np.float64), 0
      if max_modes is not None:
          num_modes = min(max_modes, rank)
      elif variance_threshold is None or variance_threshold == 1:
          num_modes = rank
      else:
          cumulative = np.cumsum(eigvals / eigvals.sum())
          num_modes = min(rank, int(np.searchsorted(cumulative, variance_threshold)) + 1)
      modes = modes_flat[:, :num_modes].reshape(n_pts, dim, num_modes).astype(np.float32)
      return mean, modes, eigvals[:num_modes].astype(np.float64, copy=False), num_modes

  def _norm_cs(self, cs):
    # Slicer legacy FCSV enum: 0 = RAS, 1 = LPS.
    tag = str(cs).strip().upper()
    if tag in ('RAS', '0'): return 'RAS'
    if tag in ('LPS', '1'): return 'LPS'
    raise ValueError(f"Unsupported coordinate system: {cs!r}; expected RAS/0 or LPS/1.")

  @staticmethod
  def _validate_points(pts, description):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
      raise ValueError(f"{description}: expected a nonempty array of 3D points, got {pts.shape}.")
    if not np.isfinite(pts).all():
      raise ValueError(f"{description}: non-finite landmark coordinates.")
    return pts

  def _read_fcsv(self, path, with_metadata=False):
    cs = 'RAS'; pts = []; labels = []; ids = []; columns = None
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
      for line_number, line in enumerate(f, 1):
        line = line.strip()
        if not line: continue
        if line.startswith('#'):
          key, separator, value = line[1:].partition('=')
          if separator and key.strip().lower() == 'coordinatesystem': cs = self._norm_cs(value)
          if separator and key.strip().lower() == 'columns':
            columns = [v.strip().lower() for v in next(csv.reader([value]))]
            if len(set(columns)) != len(columns) or not all(axis in columns for axis in ('x', 'y', 'z')):
              raise ValueError(f"{path}: invalid FCSV columns header.")
          continue
        try:
          parts = next(csv.reader([line], strict=True))
          if len(parts) < (len(columns) if columns is not None else 4):
            raise ValueError("incomplete point row")
          def field(name, default_index=None):
            index = columns.index(name) if columns is not None and name in columns else (default_index if columns is None else None)
            return parts[index].strip() if index is not None and index < len(parts) else ''
          pts.append([float(field(axis, index)) for axis, index in zip(('x', 'y', 'z'), (1, 2, 3))])
          status = field('positionstatus')
          if status and status.lower() != 'defined': raise ValueError(f"position is {status}")
          labels.append(field('label', 11)); ids.append(field('id', 0))
        except (ValueError, csv.Error, IndexError) as e:
          raise ValueError(f"{path}: invalid point on line {line_number}: {e}") from e
    pts = self._validate_points(pts, path)
    return (pts, cs, {'label': labels, 'id': ids}) if with_metadata else (pts, cs)

  def _read_json(self, path, with_metadata=False):
    with open(path, 'r', encoding='utf-8-sig') as f: data = json.load(f)
    markups = data.get('markups')
    if not isinstance(markups, list) or len(markups) != 1:
      raise ValueError(f"{path}: expected exactly one markup object.")
    markup = markups[0]
    cs = self._norm_cs(markup.get('coordinateSystem', 'RAS'))
    cps = markup.get('controlPoints') or []
    pts = []; labels = []; ids = []
    for i, cp in enumerate(cps):
      if str(cp.get('positionStatus', 'defined')).strip().lower() != 'defined':
        raise ValueError(f"{path}: control point {i + 1} does not have a defined position.")
      try: pts.append(cp['position'])
      except KeyError as e: raise ValueError(f"{path}: control point {i + 1} has no position.") from e
      labels.append(str(cp.get('label') or '').strip()); ids.append(str(cp.get('id') or '').strip())
    pts = self._validate_points(pts, path)
    return (pts, cs, {'label': labels, 'id': ids}) if with_metadata else (pts, cs)

  @staticmethod
  def _to_ras(pts, cs):
    if cs == 'LPS':
      pts = pts.copy(); pts[:, :2] *= -1
    return pts

  def _markups_data_ras(self, filepath):
    reader = self._read_fcsv if str(filepath).lower().endswith('.fcsv') else self._read_json
    pts, cs, metadata = reader(filepath, with_metadata=True)
    return self._to_ras(pts, cs), metadata

  def _markups_points_ras(self, filepath):
    return self._markups_data_ras(filepath)[0]

  def _markups_count(self, filepath): return int(self._markups_points_ras(filepath).shape[0])

  @staticmethod
  def _check_point_order(reference, current, description):
    # File/node-specific labels and IDs are not biological identities. Only a
    # complete unique identifier set shared by both files can verify ordering.
    for key in ('label', 'id'):
      a, b = reference.get(key, []), current.get(key, [])
      if a and b and all(a) and all(b) and len(set(a)) == len(a) and len(set(b)) == len(b):
        if set(a) == set(b) and a != b:
          raise ValueError(f"{description}: landmark {key} order differs; restore correspondence order before ingestion.")

  @staticmethod
  def _check_known_point_orders(metadata, known_orders, description):
    # Keep one reference for each comparable identifier set across the entire
    # cohort, not merely the first file (whose labels may be file-specific).
    for kind in ('label', 'id'):
      values = metadata.get(kind, [])
      if values and all(values) and len(set(values)) == len(values):
        key = (kind, frozenset(values))
        previous = known_orders.setdefault(key, tuple(values))
        if previous != tuple(values):
          raise ValueError(f"{description}: landmark {kind} order differs; restore correspondence order before ingestion.")

  @staticmethod
  def isSafeLibraryName(name):
    name = str(name or '')
    if not name or name != name.strip() or name in ('.', '..') or name.endswith('.'):
      return False
    if any(ord(char) < 32 or char in '<>:"/\\|?*' for char in name):
      return False
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
    return name.split('.', 1)[0].upper() not in reserved

  def _database_destination(self, dbPath, libraryRootPath=None):
    raw_path = os.fspath(dbPath)
    # Check before Path normalization removes '.', '..', or trailing separators.
    components = raw_path.replace('\\', '/').split('/')
    if not components[-1] or any(part in ('.', '..') for part in components):
      raise ValueError("Database destination must name a safe direct child of the library.")
    if not self.isSafeLibraryName(components[-1]):
      raise ValueError("Database name must be a safe single directory name.")
    requested = Path(os.path.abspath(raw_path))
    if not self.isSafeLibraryName(requested.name):
      raise ValueError("Database name must be a safe single directory name.")
    if libraryRootPath is not None and not os.fspath(libraryRootPath).strip():
      raise ValueError("Select a library root directory.")
    root = Path(libraryRootPath).absolute() if libraryRootPath is not None else requested.parent
    resolved_root = root.resolve()
    if root != resolved_root:
      raise ValueError("Choose the real library root directory, not a symlink or aliased path.")
    if requested.parent != root or requested.parent.resolve() != resolved_root or requested.resolve() == resolved_root:
      raise ValueError("Database destination must be a direct child of the selected library root.")
    if requested.is_symlink() or (requested.exists() and not requested.is_dir()):
      raise ValueError("Database destination must be a directory, not a symlink or file.")
    destination = requested.resolve()
    if destination.parent != resolved_root:
      raise ValueError("Database destination resolves outside the selected library root.")
    if destination.exists():
      entries = list(destination.iterdir())
      # A library root or arbitrary work folder is not an individual database.
      # Never recursively replace a folder containing child directories/aliases.
      if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("Refusing to replace a directory containing subdirectories or symlinks; select one database entry.")
      names = {entry.name for entry in entries}
      recognized = False
      if 'manifest.json' in names:
        try:
          with open(destination / 'manifest.json', encoding='utf-8') as stream: manifest = json.load(stream)
          files = manifest['files']
          recognized = all(isinstance(files.get(role), str) and self.isSafeLibraryName(files[role])
                           for role in ('model', 'dense', 'ssm'))
        except (OSError, ValueError, KeyError, TypeError): pass
      elif any(name.lower().endswith(('.ply', '.vtk', '.vtp', '.stl')) for name in names):
        for name in names:
          if not name.lower().endswith('.npz'): continue
          try:
            with np.load(destination / name, allow_pickle=False) as data:
              recognized = {'mean_shape', 'modes', 'eigenvalues'}.issubset(data.files)
          except (OSError, ValueError): pass
          if recognized: break
      if entries and not recognized:
        raise ValueError("Refusing to replace a nonempty directory that is not a Model Library database.")
    return destination

  @staticmethod
  def _validate_ssm(mean, modes, eigenvalues):
    if mean.ndim != 2 or mean.shape[1] != 3 or mean.shape[0] == 0:
      raise ValueError("SSM mean must contain a nonempty array of 3D points.")
    if modes.ndim != 3 or modes.shape[:2] != mean.shape or eigenvalues.ndim != 1 or modes.shape[2] != len(eigenvalues):
      raise ValueError("SSM mean, modes, and eigenvalue dimensions do not agree.")
    if not len(eigenvalues):
      raise ValueError("The population has no modes of variation. Supply at least two distinct, corresponding specimens.")
    if not all(np.isfinite(a).all() for a in (mean, modes, eigenvalues)) or np.any(eigenvalues <= 0):
      raise ValueError("SSM arrays must be finite and covariance eigenvalues must be positive.")

  @staticmethod
  def _validate_model_file(path):
    readers = {'.ply': vtk.vtkPLYReader, '.vtk': vtk.vtkPolyDataReader,
               '.vtp': vtk.vtkXMLPolyDataReader, '.stl': vtk.vtkSTLReader}
    reader_type = readers.get(Path(path).suffix.lower())
    if reader_type is None: raise ValueError(f"Unsupported template model format: {path}")
    reader = reader_type(); reader.SetFileName(str(path)); reader.Update()
    poly = reader.GetOutput()
    if reader.GetErrorCode() or poly is None or poly.GetNumberOfPoints() == 0 or poly.GetNumberOfCells() == 0:
      raise ValueError(f"Template model is empty or unreadable: {path}")
    if not np.isfinite(numpy_support.vtk_to_numpy(poly.GetPoints().GetData())).all():
      raise ValueError(f"Template model contains non-finite coordinates: {path}")

  def ingestSSMDatabase(self, dbPath, templateModelPath, templateLandmarksPath, sparseLandmarksPath, populationDir, progress_callback=None, libraryRootPath=None):
    def update_progress(p, msg):
      if progress_callback:
        try: progress_callback(p, msg)
        except Exception: logging.exception("Database progress callback failed")
    update_progress(0, "Preparing...")
    stage = None; backup = None
    try:
      # Validate the destination before any write, then again before the swap.
      destination = self._database_destination(dbPath, libraryRootPath)
      population_root = Path(populationDir).resolve()
      if population_root == destination or destination in population_root.parents:
        raise ValueError("Database destination must not contain the source population directory.")
      # Validate every input before creating or replacing the database entry.
      for source in (templateModelPath, templateLandmarksPath, sparseLandmarksPath):
        if not os.path.isfile(source): raise ValueError(f"Required template file not found: {source}")
      self._validate_model_file(templateModelPath)
      dense, dense_metadata = self._markups_data_ras(templateLandmarksPath)
      sparse = self._markups_points_ras(sparseLandmarksPath)
      landmark_files = sorted(f for f in os.listdir(populationDir) if f.lower().endswith(('.fcsv', '.json')))
      if not landmark_files: raise ValueError("No markups files found.")
      all_shapes = []; known_orders = {}
      self._check_known_point_orders(dense_metadata, known_orders, 'Dense template')
      for i, fname in enumerate(landmark_files):
        update_progress(5 + int(i / len(landmark_files) * 50), f"Loading: {fname}")
        points, metadata = self._markups_data_ras(os.path.join(populationDir, fname))
        if len(points) != len(dense):
          raise ValueError(f"{fname}: {len(points)} landmarks, but template dense correspondences contain {len(dense)}.")
        self._check_known_point_orders(metadata, known_orders, fname)
        all_shapes.append(points)
      update_progress(55, "Building Statistical Shape Model...")
      mean, modes, eigenvalues, num_modes = self.build_ssm(np.stack(all_shapes))
      self._validate_ssm(mean, modes, eigenvalues)
      update_progress(85, f"SSM built with {num_modes} modes.")

      destination.parent.mkdir(parents=True, exist_ok=True)
      stage = tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=str(destination.parent))
      modelFile = f"template_model{self._suffixes(templateModelPath)}"
      denseFile = f"dense_correspondences{self._suffixes(templateLandmarksPath)}"
      sparseFile = f"sparse_landmarks{self._suffixes(sparseLandmarksPath)}"
      for source, name in ((templateModelPath, modelFile), (templateLandmarksPath, denseFile), (sparseLandmarksPath, sparseFile)):
        shutil.copy2(source, os.path.join(stage, name))
      np.savez_compressed(os.path.join(stage, 'ssm_model.npz'), mean_shape=mean, modes=modes, eigenvalues=eigenvalues)
      manifest = {'version': 1, 'files': {'model': modelFile, 'dense': denseFile, 'sparse': sparseFile, 'ssm': 'ssm_model.npz'},
                  'counts': {'dense': len(dense), 'sparse': len(sparse)}}
      with open(os.path.join(stage, 'manifest.json'), 'w', encoding='utf-8') as f: json.dump(manifest, f, indent=2)
      # Read back the staged generation before publishing it.
      self._validate_model_file(os.path.join(stage, modelFile))
      for name, expected in ((denseFile, dense), (sparseFile, sparse)):
        if not np.array_equal(self._markups_points_ras(os.path.join(stage, name)), expected):
          raise ValueError(f"Staged template landmarks changed while saving: {name}")
      with np.load(os.path.join(stage, 'ssm_model.npz'), allow_pickle=False) as saved:
        self._validate_ssm(saved['mean_shape'], saved['modes'], saved['eigenvalues'])
      with open(os.path.join(stage, 'manifest.json'), encoding='utf-8') as f:
        if json.load(f) != manifest: raise ValueError("Staged manifest failed verification.")
      update_progress(95, "Publishing database...")
      if self._database_destination(dbPath, libraryRootPath) != destination:
        raise ValueError("Database destination changed while preparing the replacement.")
      if destination.exists():
        backup = tempfile.mkdtemp(prefix=f".{destination.name}.backup-", dir=str(destination.parent))
        os.rmdir(backup)
        os.replace(str(destination), backup)
      try:
        os.replace(stage, str(destination)); stage = None
      except Exception:
        if backup is not None:
          os.replace(backup, str(destination)); backup = None
        raise
      if backup is not None:
        try: shutil.rmtree(backup); backup = None
        except OSError: logging.warning("Database saved; previous generation retained at %s", backup)
      update_progress(100, "Finalizing...")
      return True, f"Database '{destination.name}' saved."
    except Exception as e:
      recovery = f" Previous generation is preserved at {backup}." if backup and os.path.exists(backup) else ''
      return False, f"Error creating database: {e}.{recovery}"
    finally:
      if stage and os.path.exists(stage): shutil.rmtree(stage, ignore_errors=True)

  def removeSSMDatabase(self, dbPath):
    try: shutil.rmtree(dbPath); return True, f"Database removed."
    except Exception as e: return False, f"Error removing database: {e}"

  def loadSSMDatabase(self, dbPath):
    try:
      dbName=Path(dbPath).name
      manifest_path=os.path.join(dbPath,"manifest.json")
      modelFile=lmFile=sparseFile=ssmFile=None
      if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as manifest_stream:
          man = json.load(manifest_stream)
        files=man.get("files",{})
        modelFile=os.path.join(dbPath,files.get("model",""))
        lmFile=os.path.join(dbPath,files.get("dense",""))
        sparseFile=os.path.join(dbPath,files.get("sparse",""))
        ssmFile=os.path.join(dbPath,files.get("ssm","ssm_model.npz"))
      else:
        files=sorted(os.listdir(dbPath))
        models=[f for f in files if f.lower().endswith((".ply",".vtk",".vtp",".stl"))]
        npzs=[f for f in files if f.lower().endswith(".npz")]
        markups=[f for f in files if f.lower().endswith((".fcsv",".json"))]
        modelFile=os.path.join(dbPath,models[0]) if models else None
        ssmFile=os.path.join(dbPath,"ssm_model.npz") if "ssm_model.npz" in npzs else (os.path.join(dbPath,npzs[0]) if npzs else None)
        if len(markups)==1:
          lmFile=os.path.join(dbPath,markups[0]); sparseFile=None
        elif len(markups)>=2:
          dense_hint=[f for f in markups if ("dense" in f.lower()) or ("corr" in f.lower())]
          sparse_hint=[f for f in markups if "sparse" in f.lower()]
          if dense_hint and sparse_hint:
            lmFile=os.path.join(dbPath,dense_hint[0]); sparseFile=os.path.join(dbPath,sparse_hint[0])
          else:
            counts=[(f,self._markups_count(os.path.join(dbPath,f))) for f in markups]
            counts.sort(key=lambda x:x[1],reverse=True)
            lmFile=os.path.join(dbPath,counts[0][0])
            sparseFile=os.path.join(dbPath,counts[1][0]) if len(counts)>1 else None

      if not (modelFile and ssmFile and lmFile):
        return False, "Database is incomplete."

      with np.load(ssmFile, allow_pickle=False) as ssm_data:
        mean_shape=ssm_data["mean_shape"]; modes=ssm_data["modes"]; eigenvalues=ssm_data["eigenvalues"]
      self._validate_ssm(mean_shape, modes, eigenvalues)
      needed_pts=int(mean_shape.shape[0])
      n_lm=self._markups_count(lmFile)
      n_sp=self._markups_count(sparseFile) if sparseFile else None
      # Legacy entries have no authoritative roles. Never override explicit
      # manifest provenance just because the sparse count happens to match.
      if not os.path.exists(manifest_path) and n_lm != needed_pts and n_sp == needed_pts:
        lmFile, sparseFile = sparseFile, lmFile
        n_lm, n_sp = n_sp, n_lm
      if n_lm != needed_pts:
        raise ValueError(f"Dense template has {n_lm} points; SSM requires {needed_pts}.")
      if os.path.exists(manifest_path):
        counts = man.get('counts', {})
        if counts.get('dense', n_lm) != n_lm or counts.get('sparse', n_sp) != n_sp:
          raise ValueError("Manifest landmark counts do not match template files.")
      self._validate_model_file(modelFile)

      modelNode=slicer.util.loadModel(modelFile)
      lmNode=slicer.util.loadMarkups(lmFile)
      sparseNode=slicer.util.loadMarkups(sparseFile) if sparseFile else None

      base_name=f"{dbName}_template"
      modelNode.SetName(base_name)
      lmNode.SetName(f"{base_name}_correspondences")
      if sparseNode: sparseNode.SetName(f"{base_name}_sparse_landmarks")

      mean_flat=mean_shape.reshape(-1,1)
      n_modes=modes.shape[2]
      modes_flat=modes.reshape(3*needed_pts,n_modes)
      table_array=np.hstack([mean_flat,modes_flat])
      tableNode=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", f"ssm_data_{dbName}")
      slicer.util.updateTableFromArray(tableNode, table_array.astype(np.float32,copy=False))
      t=tableNode.GetTable(); t.GetColumn(0).SetName("mean_shape")
      for i in range(n_modes): t.GetColumn(i+1).SetName(f"mode_{i}")
      tableNode.SetAttribute("ssm_eigenvalues", json.dumps(eigenvalues.tolist()))
      tableNode.SetAttribute("ssm_npoints", str(needed_pts))
      return True, f"SSM '{dbName}' loaded. Table Node has {n_modes} modes."
    except Exception as e:
      return False, f"Error loading database: {e}"

  def _suffixes(self, p):
    s=Path(p).suffixes
    return "".join(s) if s else Path(p).suffix


#
# MorphoWeaveModelLibraryWidget
#

class MorphoWeaveModelLibraryWidget(ScriptedLoadableModuleWidget):
  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    self.logic = MorphoWeaveModelLibraryLogic()
    self.ssm_data = {}             # persistent caches for visualization
    self.pcSliders = []            # list[ctk.ctkSliderWidget]
    self.methodCombo = None        # TPS / Biharmonic
    self.numPCSpin = None          # how many PC sliders to show

  # ---------------- UI ----------------

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # --- DB ---
    dbWidget = ctk.ctkCollapsibleButton(); dbWidget.text = "1) Database Library"; dbWidget.collapsed = False
    dbLayout = qt.QFormLayout(dbWidget); self.layout.addWidget(dbWidget)
    self.databasePathSelector = ctk.ctkPathLineEdit(); self.databasePathSelector.filters = ctk.ctkPathLineEdit.Dirs
    dbLayout.addRow("Database Location:", self.databasePathSelector)
    self.databaseListWidget = qt.QListWidget()
    self.loadDbButton = qt.QPushButton("Load Selected Database"); self.loadDbButton.setIcon(slicer.app.style().standardIcon(qt.QStyle.SP_ArrowUp)); self.loadDbButton.enabled = False
    self.removeDbButton = qt.QPushButton("Remove Selected Database"); self.removeDbButton.setIcon(slicer.app.style().standardIcon(qt.QStyle.SP_TrashIcon)); self.removeDbButton.enabled = False
    listButtonLayout = qt.QVBoxLayout(); listButtonLayout.addWidget(self.loadDbButton); listButtonLayout.addWidget(self.removeDbButton)
    listLayout = qt.QHBoxLayout(); listLayout.addWidget(self.databaseListWidget); listLayout.addLayout(listButtonLayout)
    dbLayout.addRow("Available Databases:", listLayout)

    ingestWidget = ctk.ctkCollapsibleButton(); ingestWidget.text = "2) Ingest New SSM"; ingestWidget.collapsed = True
    ingestLayout = qt.QFormLayout(ingestWidget); self.layout.addWidget(ingestWidget)
    ingestHelp = qt.QLabel("Use this section to create a new database from atlas artifacts and population correspondences.")
    ingestHelp.setWordWrap(True)
    ingestLayout.addRow(ingestHelp)
    self.ssmTemplateModelSelector = ctk.ctkPathLineEdit(); self.ssmTemplateModelSelector.filters = ctk.ctkPathLineEdit.Files; self.ssmTemplateModelSelector.nameFilters=["*.ply","*.vtk","*.vtp","*.stl"]
    self.ssmTemplateLandmarksSelector = ctk.ctkPathLineEdit(); self.ssmTemplateLandmarksSelector.filters = ctk.ctkPathLineEdit.Files; self.ssmTemplateLandmarksSelector.nameFilters=["*.json","*.fcsv"]
    self.ssmSparseLandmarksSelector = ctk.ctkPathLineEdit(); self.ssmSparseLandmarksSelector.filters = ctk.ctkPathLineEdit.Files; self.ssmSparseLandmarksSelector.nameFilters=["*.json","*.fcsv"]
    self.ssmPopulationDirSelector = ctk.ctkPathLineEdit(); self.ssmPopulationDirSelector.filters = ctk.ctkPathLineEdit.Dirs
    self.ssmNameEditor = qt.QLineEdit()
    ingestLayout.addRow("Template Model:", self.ssmTemplateModelSelector)
    ingestLayout.addRow("Dense Correspondences:", self.ssmTemplateLandmarksSelector)
    ingestLayout.addRow("Sparse Landmarks:", self.ssmSparseLandmarksSelector)
    ingestLayout.addRow("Population Correspondences Folder:", self.ssmPopulationDirSelector)
    ingestLayout.addRow("New Database Name:", self.ssmNameEditor)
    self.ingestStatusLabel = qt.QLabel("")
    self.ingestStatusLabel.setWordWrap(True)
    ingestLayout.addRow(self.ingestStatusLabel)
    self.ingestButton = qt.QPushButton("Ingest into Database"); self.ingestButton.enabled = False
    ingestLayout.addRow(self.ingestButton)

    # --- Visualization ---
    explorerWidget = ctk.ctkCollapsibleButton(); explorerWidget.text = "3) SSM Explorer"
    explorerWidget.collapsed = False
    explorerLayout = qt.QFormLayout(explorerWidget); self.layout.addWidget(explorerWidget)
    explorerLayout.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)
    explorerHelp = qt.QLabel(
      "PC sliders apply offsets around the loaded template correspondences.\n"
      "Use them to inspect variation while preserving the template alignment context."
    )
    explorerHelp.setWordWrap(True)
    explorerLayout.addRow(explorerHelp)

    self.explorerModelSelector = slicer.qMRMLNodeComboBox()
    self.explorerModelSelector.nodeTypes = ["vtkMRMLModelNode"]
    self.explorerModelSelector.selectNodeUponCreation = True
    self.explorerModelSelector.addEnabled = False
    self.explorerModelSelector.removeEnabled = False
    self.explorerModelSelector.noneEnabled = True
    self.explorerModelSelector.setMRMLScene(slicer.mrmlScene)
    self.explorerModelSelector.setToolTip("Select the surface model to deform.")
    explorerLayout.addRow("Model to Deform:", self.explorerModelSelector)

    self.explorerLandmarksSelector = slicer.qMRMLNodeComboBox()
    self.explorerLandmarksSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
    self.explorerLandmarksSelector.selectNodeUponCreation = True
    self.explorerLandmarksSelector.addEnabled = False
    self.explorerLandmarksSelector.removeEnabled = False
    self.explorerLandmarksSelector.noneEnabled = True
    self.explorerLandmarksSelector.setMRMLScene(slicer.mrmlScene)
    self.explorerLandmarksSelector.setToolTip("Select the dense correspondence landmarks that control deformation.")
    explorerLayout.addRow("Dense Correspondence Landmarks:", self.explorerLandmarksSelector)

    self.explorerTableSelector = slicer.qMRMLNodeComboBox()
    self.explorerTableSelector.nodeTypes = ["vtkMRMLTableNode"]
    self.explorerTableSelector.addAttribute("vtkMRMLTableNode", "ssm_eigenvalues")
    self.explorerTableSelector.selectNodeUponCreation = True
    self.explorerTableSelector.addEnabled = False
    self.explorerTableSelector.removeEnabled = False
    self.explorerTableSelector.noneEnabled = True
    self.explorerTableSelector.setMRMLScene(slicer.mrmlScene)
    self.explorerTableSelector.setToolTip("Select the SSM data table containing model data.")
    explorerLayout.addRow("SSM Data Table:", self.explorerTableSelector)

    self.methodCombo = qt.QComboBox()
    self.methodCombo.addItems(["Local RBF (Gaussian, fast)", "Biharmonic (experimental)"])
    self.methodCombo.setCurrentIndex(0)
    explorerLayout.addRow("Deformation Method:", self.methodCombo)

    self.numPCSpin = qt.QSpinBox(); self.numPCSpin.setMinimum(1); self.numPCSpin.setMaximum(1); self.numPCSpin.setValue(1)
    explorerLayout.addRow("Number of PC sliders:", self.numPCSpin)

    # Sliders area — larger and expanding
    self.slidersArea = qt.QScrollArea()
    self.slidersArea.setWidgetResizable(True)
    self.slidersArea.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
    self.slidersArea.setMinimumHeight(420)

    self.slidersWidget = qt.QWidget()
    self.slidersLayout = qt.QFormLayout(self.slidersWidget)
    self.slidersLayout.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)
    self.slidersArea.setWidget(self.slidersWidget)
    explorerLayout.addRow(self.slidersArea)

    self.resetButton = qt.QPushButton("Reset Explorer PCs")
    self.resetButton.toolTip = "Reset all sliders to zero and restore the loaded template baseline."
    self.resetButton.enabled = False
    explorerLayout.addRow(self.resetButton)

    self.progressBar = qt.QProgressBar()
    self.progressBar.setRange(0, 100)
    self.progressBar.setValue(0)
    self.progressBar.setFormat("%v% - Ready")
    self.layout.addWidget(self.progressBar)

    # -- Connections --
    self.databasePathSelector.currentPathChanged.connect(self.onDatabasePathChanged)
    self.databaseListWidget.currentItemChanged.connect(self.onDatabaseSelectionChanged)
    self.loadDbButton.connect('clicked(bool)', self.onLoadButton)
    self.removeDbButton.connect('clicked(bool)', self.onRemoveButton)
    self.ssmTemplateModelSelector.connect('validInputChanged(bool)', self.onSelect)
    self.ssmTemplateLandmarksSelector.connect('validInputChanged(bool)', self.onSelect)
    self.ssmSparseLandmarksSelector.connect('validInputChanged(bool)', self.onSelect)
    self.ssmPopulationDirSelector.connect('validInputChanged(bool)', self.onSelect)
    self.ingestButton.connect('clicked(bool)', self.onIngestButton)

    self.explorerModelSelector.currentNodeChanged.connect(self.onExplorerNodeSelected)
    self.explorerLandmarksSelector.currentNodeChanged.connect(self.onExplorerNodeSelected)
    self.explorerTableSelector.currentNodeChanged.connect(self.onExplorerNodeSelected)

    self.methodCombo.currentIndexChanged.connect(self.onMethodChanged)
    self.numPCSpin.valueChanged.connect(self.rebuildPCSliders)
    self.resetButton.connect('clicked(bool)', self.onResetAll)

    self.initializeDatabasePath()
    self.onSelect()
    self.layout.addStretch(1)

  def cleanup(self):
    for s in self.pcSliders:
      try: s.valueChanged.disconnect(self.onPCSlidersChanged)
      except: pass
    super().cleanup()

  # ------------- DB handlers -------------

  def initializeDatabasePath(self):
    settings = qt.QSettings()
    newKey = 'MorphoWeaveModelLibrary/databasePath'
    legacyKey = 'DATABASE/databasePath'
    defaultPath = os.path.join(Path.home(), "Documents", "MorphoWeaveModels")
    if settings.contains(newKey):
      dbPath = settings.value(newKey, defaultPath)
    elif settings.contains(legacyKey):
      dbPath = settings.value(legacyKey, defaultPath)
      settings.setValue(newKey, dbPath)
    else:
      dbPath = defaultPath
    self.databasePathSelector.setCurrentPath(dbPath)

  def onDatabasePathChanged(self, path):
    if os.path.isdir(path):
      settings = qt.QSettings()
      settings.setValue('MorphoWeaveModelLibrary/databasePath', path)
      self.refreshDatabaseList()

  def refreshDatabaseList(self):
    self.databaseListWidget.clear()
    dbPath = self.databasePathSelector.currentPath
    if not os.path.isdir(dbPath):
      try: os.makedirs(dbPath, exist_ok=True)
      except: pass
      return
    for itemName in sorted(os.listdir(dbPath)):
      if os.path.isdir(os.path.join(dbPath, itemName)):
        self.databaseListWidget.addItem(qt.QListWidgetItem(itemName))

  def onDatabaseSelectionChanged(self):
    isItemSelected = self.databaseListWidget.currentItem() is not None
    self.removeDbButton.enabled = isItemSelected
    self.loadDbButton.enabled = isItemSelected

  def onLoadButton(self):
      currentItem = self.databaseListWidget.currentItem()
      if not currentItem: return
      dbName = currentItem.text()
      dbPath = os.path.join(self.databasePathSelector.currentPath, dbName)

      before_model = set()
      before_lm = set()
      before_table = set()
      try:
        col = slicer.mrmlScene.GetNodesByClass("vtkMRMLModelNode"); col.UnRegister(None)
        before_model = {col.GetItemAsObject(i).GetID() for i in range(col.GetNumberOfItems())}
        col = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsFiducialNode"); col.UnRegister(None)
        before_lm = {col.GetItemAsObject(i).GetID() for i in range(col.GetNumberOfItems())}
        col = slicer.mrmlScene.GetNodesByClass("vtkMRMLTableNode"); col.UnRegister(None)
        before_table = {col.GetItemAsObject(i).GetID() for i in range(col.GetNumberOfItems())}
      except Exception:
        pass

      success, message = self.logic.loadSSMDatabase(dbPath)
      if success:
        slicer.util.showStatusMessage(f"✅ {message}", 3000)
        logging.info(message)

        # Force default to TPS for newly loaded databases
        b = qt.QSignalBlocker(self.methodCombo)
        try:
          self.methodCombo.setCurrentIndex(0)  # 0
        finally:
          del b

        # Auto-select nodes created by this load operation (avoid stale-name collisions)
        try:
          m = l = t = None
          col = slicer.mrmlScene.GetNodesByClass("vtkMRMLModelNode"); col.UnRegister(None)
          for i in range(col.GetNumberOfItems()):
            n = col.GetItemAsObject(i)
            if n.GetID() not in before_model:
              m = n
          col = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsFiducialNode"); col.UnRegister(None)
          for i in range(col.GetNumberOfItems()):
            n = col.GetItemAsObject(i)
            if n.GetID() not in before_lm and "correspondences" in (n.GetName() or "").lower():
              l = n
          col = slicer.mrmlScene.GetNodesByClass("vtkMRMLTableNode"); col.UnRegister(None)
          for i in range(col.GetNumberOfItems()):
            n = col.GetItemAsObject(i)
            if n.GetID() not in before_table:
              t = n
          if m is None:
            m = slicer.util.getNode(f"{dbName}_template")
          if l is None:
            l = slicer.util.getNode(f"{dbName}_template_correspondences")
          if t is None:
            t = slicer.util.getNode(f"ssm_data_{dbName}")
          self.explorerModelSelector.setCurrentNode(m)
          self.explorerLandmarksSelector.setCurrentNode(l)
          self.explorerTableSelector.setCurrentNode(t)
          sel_msg = (
            f"Explorer selection -> Model: {(m.GetName() if m else 'none')}, "
            f"Correspondences: {(l.GetName() if l else 'none')}, "
            f"Table: {(t.GetName() if t else 'none')}"
          )
          slicer.util.showStatusMessage(sel_msg, 5000)
          logging.info(sel_msg)
        except Exception as e:
          logging.warning(f"Auto-select failed: {e}")

        # Optional: keep this if you want an explicit rebuild after signals fire
        self.setupExplorerSliders()
      else:
        slicer.util.errorDisplay(f"Error loading database: {message}")


  def onRemoveButton(self):
    currentItem = self.databaseListWidget.currentItem()
    if not currentItem: return
    dbName = currentItem.text()
    if slicer.util.confirmOkCancelDisplay(f"Permanently delete the '{dbName}' database? This cannot be undone."):
      dbPath = os.path.join(self.databasePathSelector.currentPath, dbName)
      success, message = self.logic.removeSSMDatabase(dbPath)
      if success:
        slicer.util.showStatusMessage(f"✅ {message}", 3000)
        self.refreshDatabaseList()
      else:
        slicer.util.errorDisplay(message)

  def onSelect(self):
    ready = bool(
      self.ssmTemplateModelSelector.currentPath and
      self.ssmTemplateLandmarksSelector.currentPath and
      self.ssmSparseLandmarksSelector.currentPath and
      self.ssmPopulationDirSelector.currentPath
    )
    self.ingestButton.enabled = ready
    if ready:
      setWorkflowStatus(self.ingestStatusLabel, "ready", "Ingest inputs are complete.")
    else:
      setWorkflowStatus(self.ingestStatusLabel, "needs_input", "Select template model, dense and sparse landmarks, and population folder.")

  def onIngestButton(self):
    populationDir = self.ssmPopulationDirSelector.currentPath
    if not os.path.isdir(populationDir):
      slicer.util.errorDisplay(f"The specified population folder does not exist:\n\n{populationDir}")
      return

    dbName = self.ssmNameEditor.text or Path(self.ssmTemplateModelSelector.currentPath).stem
    dbRootPath = self.databasePathSelector.currentPath
    if not self.logic.isSafeLibraryName(dbName):
      slicer.util.errorDisplay("Choose a database name without path separators, reserved names, or '.'/'..'.")
      return
    db_path = os.path.join(dbRootPath, dbName)
    try:
      self.logic._database_destination(db_path, dbRootPath)
    except ValueError as error:
      slicer.util.errorDisplay(str(error))
      return

    if os.path.exists(db_path):
      if not slicer.util.confirmOkCancelDisplay(f"Database '{dbName}' already exists. Replace the entire entry, including any extra files in it?"):
        return

    self.ingestButton.enabled = False
    self.loadDbButton.enabled = False
    self.removeDbButton.enabled = False
    slicer.app.setOverrideCursor(qt.Qt.WaitCursor)

    def progress_callback(value, text):
      self.progressBar.setValue(value)
      self.progressBar.setFormat(f"{text} ({value}%)")
      slicer.app.processEvents()

    success, message = self.logic.ingestSSMDatabase(
      dbPath=db_path,
      templateModelPath=self.ssmTemplateModelSelector.currentPath,
      templateLandmarksPath=self.ssmTemplateLandmarksSelector.currentPath,
      sparseLandmarksPath=self.ssmSparseLandmarksSelector.currentPath,
      populationDir=populationDir,
      progress_callback=progress_callback,
      libraryRootPath=dbRootPath
    )

    self.progressBar.setValue(100)
    self.progressBar.setFormat("Complete")
    slicer.app.restoreOverrideCursor()
    self.ingestButton.enabled = True
    self.onDatabaseSelectionChanged()

    if success:
      slicer.util.showStatusMessage(f"✅ Success: {message}", 3000)
      self.refreshDatabaseList()
    else:
      slicer.util.errorDisplay(f"Failed to ingest SSM database:\n\n{message}")
      self.progressBar.setFormat("Failed")

  # ------------- Viz core -------------

  def _clearLayout(self, layout):
    if not layout: return
    while layout.count():
      child = layout.takeAt(0)
      if child.widget():
        child.widget().deleteLater()

  def _recreateSlidersPanel(self):
    """Replace the sliders panel widget completely (avoids duplicate visual copies)."""
    old = self.slidersArea.takeWidget()
    if old is not None:
      old.deleteLater()
    self.slidersWidget = qt.QWidget()
    self.slidersLayout = qt.QFormLayout(self.slidersWidget)
    self.slidersLayout.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)
    self.slidersLayout.setVerticalSpacing(12)
    self.slidersArea.setWidget(self.slidersWidget)

  def _methodIndex(self):
    # Slicer/Qt may expose currentIndex as a property (int) or as a callable
    try: return int(self.methodCombo.currentIndex)
    except TypeError: return int(self.methodCombo.currentIndex())

  def onExplorerNodeSelected(self):
    self.setupExplorerSliders()

  def onMethodChanged(self, *args):
    if not self.ssm_data: return
    ok = self.precomputeDeformer()
    if not ok:
      b = qt.QSignalBlocker(self.methodCombo); self.methodCombo.setCurrentIndex(0); del b
      slicer.util.errorDisplay("Biharmonic precompute failed; reverted to Local RBF.")
      self.precomputeDeformer()
    self.applyCurrentPCs()

  def setupExplorerSliders(self):
    # Disconnect any previous sliders and rebuild the panel fresh
    for s in self.pcSliders:
      try: s.valueChanged.disconnect(self.onPCSlidersChanged)
      except: pass
    self.pcSliders = []
    self.ssm_data = {}
    self.resetButton.enabled = False
    self._recreateSlidersPanel()

    modelNode = self.explorerModelSelector.currentNode()
    landmarkNode = self.explorerLandmarksSelector.currentNode()
    tableNode = self.explorerTableSelector.currentNode()

    if not all([modelNode, landmarkNode, tableNode]):
      return

    eigenvalues_str = tableNode.GetAttribute("ssm_eigenvalues")
    if not eigenvalues_str:
      return

    num_points_in_landmarks = landmarkNode.GetNumberOfControlPoints()
    num_rows_in_table = tableNode.GetTable().GetNumberOfRows()
    if num_rows_in_table != (num_points_in_landmarks * 3):
      slicer.util.errorDisplay("Mismatch between number of landmarks and data in SSM table.")
      return

    need=int(tableNode.GetAttribute("ssm_npoints") or 0)
    if need and need!=num_points_in_landmarks:
      slicer.util.errorDisplay("SSM table point count does not match landmark count.")
      return

    # cache geometry/SSM
    self.ssm_data['base_template_vertices'] = slicer.util.arrayFromModelPoints(modelNode).astype(np.float64, copy=True)
    self.ssm_data['template_vertices'] = self.ssm_data['base_template_vertices'].copy()
    self.ssm_data['original_points_RAS'] = slicer.util.arrayFromMarkupsControlPoints(landmarkNode).astype(np.float64, copy=True)

    mean_flat = slicer.util.arrayFromTableColumn(tableNode, "mean_shape")
    self.ssm_data['mean_RAS'] = mean_flat.reshape(num_points_in_landmarks, 3).astype(np.float64, copy=False)

    modes = []
    num_modes_from_table = tableNode.GetTable().GetNumberOfColumns() - 1
    if num_modes_from_table <= 0:
      slicer.util.infoDisplay("The loaded SSM has no modes of variation.")
      return

    for i in range(num_modes_from_table):
      mode_flat = slicer.util.arrayFromTableColumn(tableNode, f"mode_{i}")
      modes.append(mode_flat.reshape(num_points_in_landmarks, 3))
    self.ssm_data['modes_RAS'] = np.stack(modes, axis=-1).astype(np.float64, copy=False)

    self.ssm_data['eigenvalues'] = np.array(json.loads(eigenvalues_str))
    self.ssm_data['std_devs'] = np.sqrt(self.ssm_data['eigenvalues'])

    # precompute for current method
    ok = self.precomputeDeformer()
    if (not ok) and (self._methodIndex()==1):
      b = qt.QSignalBlocker(self.methodCombo); self.methodCombo.setCurrentIndex(0); del b
      slicer.util.errorDisplay("Biharmonic precompute failed; reverted to TPS.")
      self.precomputeDeformer()

    # build multi-PC sliders (1-indexed)
    self.numPCSpin.setMaximum(num_modes_from_table)
    self.numPCSpin.setValue(min(3, num_modes_from_table))
    self.rebuildPCSliders()

    self.resetButton.enabled = True
    self.applyCurrentPCs()

  def rebuildPCSliders(self):
    # Recreate the panel to guarantee a single fresh set of sliders
    self._recreateSlidersPanel()
    for s in self.pcSliders:
      try: s.valueChanged.disconnect(self.onPCSlidersChanged)
      except: pass
    self.pcSliders = []
    k = self.numPCSpin.value
    for i in range(k):
      sw = ctk.ctkSliderWidget()
      sw.minimum = -3; sw.maximum = 3; sw.value = 0; sw.singleStep = 0.1
      sw.setToolTip(f"PC{i+1} (±σ multiples)")
      self.slidersLayout.addRow(f"PC{i+1}:", sw)   # 1-indexed label
      sw.valueChanged.connect(self.onPCSlidersChanged)
      self.pcSliders.append(sw)
    # A new panel displays zero coefficients. Keep both geometry and landmarks
    # consistent immediately, rather than waiting for a slider event.
    self.resetGeometry()

  def onResetAll(self):
    # Reset sliders and geometry; avoid context-manager (not supported)
    for s in self.pcSliders:
      blocker = qt.QSignalBlocker(s); s.value = 0; del blocker
    self.resetGeometry()

  def resetGeometry(self):
    modelNode = self.explorerModelSelector.currentNode()
    if modelNode and 'template_vertices' in self.ssm_data:
      pd = modelNode.GetPolyData()
      pts = pd.GetPoints()
      original_points_vtk_array = numpy_support.numpy_to_vtk(self.ssm_data['template_vertices'].astype(np.float32, copy=False), deep=1)
      pts.SetData(original_points_vtk_array)
      pts.Modified(); pd.Modified(); modelNode.Modified()
    if self.explorerLandmarksSelector.currentNode() and 'original_points_RAS' in self.ssm_data:
      slicer.util.updateMarkupsControlPointsFromArray(self.explorerLandmarksSelector.currentNode(), self.ssm_data['original_points_RAS'])

  def onPCSlidersChanged(self, *args):
    self.applyCurrentPCs()

  def _currentPCWeights(self):
    if not self.ssm_data or not self.pcSliders: return None
    m = self.ssm_data['modes_RAS'].shape[-1]
    k = min(len(self.pcSliders), m, len(self.ssm_data['std_devs']))
    w = np.zeros(m, dtype=np.float64)
    # each slider is in units of σ for PC i (1-indexed)
    for i in range(k):
      w[i] = float(self.pcSliders[i].value) * float(self.ssm_data['std_devs'][i])
    return w

  def applyCurrentPCs(self):
    modelNode = self.explorerModelSelector.currentNode()
    landmarkNode = self.explorerLandmarksSelector.currentNode()
    if not (modelNode and landmarkNode and self.ssm_data):
      return
    w = self._currentPCWeights()
    if w is None:
      return
    modes = self.ssm_data['modes_RAS']
    # aggregate deformation across first k PCs shown
    k = min(len(self.pcSliders), modes.shape[-1])
    # Keep method switching visually stable: when all shown PC weights are zero,
    # preserve the exact template-centered baseline rather than re-solving with a
    # different backend (which can introduce tiny numerical drift).
    if k == 0 or np.all(w[:k] == 0.0):
      self.resetGeometry()
      return
    agg = np.tensordot(modes[...,:k], w[:k], axes=(2,0))   # (n_pts,3)
    # Template-centered deformation: slider weights are applied as offsets from
    # the loaded template landmarks, not from SSM mean landmarks.
    target = self.ssm_data['original_points_RAS'] + agg

    # Update landmarks
    slicer.util.updateMarkupsControlPointsFromArray(landmarkNode, target.astype(np.float64, copy=False))

    # Update mesh by selected method
    if self._methodIndex()==0:
      self._applyLocalRBF(target)
    else:
      ok = self._applyBiharmonic(target)
      if not ok:
        blocker = qt.QSignalBlocker(self.methodCombo); self.methodCombo.setCurrentIndex(0); del blocker
        self.precomputeDeformer()
        self._applyLocalRBF(target)

  # -------- Precomputes & solvers --------

  def precomputeDeformer(self):
      try:
          modelNode = self.explorerModelSelector.currentNode()
          if modelNode is None: return False

          V = self.ssm_data.get('base_template_vertices', None)
          if V is None:
              V = slicer.util.arrayFromModelPoints(modelNode).astype(np.float64, copy=True)
              self.ssm_data['base_template_vertices'] = V.copy()
          V = MorphoWeaveModelLibraryLogic._validate_points(V, "Template model")
          MorphoWeaveModelLibraryLogic._validate_points(self.ssm_data['original_points_RAS'], "Template landmarks")
          self.ssm_data['template_vertices'] = V.copy()

          if self._methodIndex()==0:
              lm = self.ssm_data['original_points_RAS'].astype(np.float64, copy=False)
              k = min(32, lm.shape[0])
              tree = cKDTree(lm)
              try: d, idx = tree.query(self.ssm_data['template_vertices'], k=k, workers=-1)
              except TypeError: d, idx = tree.query(V, k=k)
              if k == 1: d, idx = d[:, None], idx[:, None]
              h = np.percentile(d[:, -1], 75) + 1e-9
              w = np.exp(-(d*d)/(h*h))
              w /= w.sum(1, keepdims=True)
              self.ssm_data['interpolation_indices'] = idx.astype(np.int64, copy=False)
              self.ssm_data['interpolation_weights']  = w.astype(np.float64, copy=False)
              return True

          # Triangulate immutable rest geometry, with an explicit ID map even
          # when point counts happen to agree. Current display points may already
          # be deformed by another backend.
          rest = vtk.vtkPolyData(); rest.DeepCopy(modelNode.GetPolyData())
          if rest.GetNumberOfPoints() != len(V): raise ValueError("Template topology changed; reselect the model.")
          rest.GetPoints().SetData(numpy_support.numpy_to_vtk(V, deep=True))
          ids = numpy_support.numpy_to_vtk(np.arange(len(V), dtype=np.int64), deep=True)
          ids.SetName('MorphoWeaveOriginalPointId'); rest.GetPointData().AddArray(ids)
          triangle_filter = vtk.vtkTriangleFilter(); triangle_filter.SetInputData(rest)
          triangle_filter.PassLinesOff(); triangle_filter.PassVertsOff(); triangle_filter.Update()
          tri = triangle_filter.GetOutput()
          id_array = tri.GetPointData().GetArray('MorphoWeaveOriginalPointId')
          if id_array is None: raise ValueError("Triangulation did not preserve template point IDs.")
          orig_pid = numpy_support.vtk_to_numpy(id_array).astype(np.int64, copy=True)
          if len(np.unique(orig_pid)) != len(orig_pid) or np.any(orig_pid < 0) or np.any(orig_pid >= len(V)):
              raise ValueError("Invalid triangulated template point IDs.")
          Vtri = V[orig_pid].copy()
          ca = numpy_support.vtk_to_numpy(tri.GetPolys().GetData())
          F = ca.reshape(-1, 4)[:, 1:4].astype(np.int64, copy=False)
          if not len(F): raise ValueError("Biharmonic deformation requires a triangular surface.")
          n = Vtri.shape[0]
          L, mass = self._cotangent_laplacian(Vtri, F)
          Minv = diags(1.0 / np.maximum(mass, 1e-12))
          A0 = (L.T @ Minv @ L).tocsr()

          lm_base = self.ssm_data['original_points_RAS'].astype(np.float64, copy=False)
          kdt = cKDTree(Vtri); _, idx = kdt.query(lm_base, k=1)
          edges = np.concatenate((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]), axis=0)
          adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
          component_count, components = connected_components(adjacency, directed=False)
          if len(np.unique(components[idx])) != component_count:
              raise ValueError("Each disconnected template component needs a landmark constraint for biharmonic deformation.")

          # Repeated nearest-vertex constraints contribute a least-squares
          # average, not an order-dependent last-write-wins displacement.
          Wd = np.bincount(idx, minlength=n).astype(np.float64)
          lam = 1e4
          diag_max = float(A0.diagonal().max() or 1.0)
          if lam < 1e-6*diag_max or lam > 1e6*diag_max: lam = 1e-2*diag_max

          A = (A0 + lam*diags(Wd)).tocsc()
          lu = splu(A)

          self.ssm_data['bih_V'] = Vtri
          self.ssm_data['constrained_idx'] = idx.astype(np.int64, copy=False)
          self.ssm_data['bih_A_lu'] = lu
          self.ssm_data['bih_lam'] = lam
          self.ssm_data['bih_Wd'] = Wd
          self.ssm_data['faces'] = F
          self.ssm_data['bih_orig_pid'] = orig_pid
          return True
      except Exception as e:
          logging.error(f"Precompute failed: {e}")
          return False

  @staticmethod
  def _cotangent_laplacian(vertices, faces):
      """Symmetric (negative-semidefinite) stiffness and lumped vertex areas."""
      n = len(vertices); rows = []; columns = []; weights = []; mass = np.zeros(n)
      for i, j, k in faces:
          vi, vj, vk = vertices[i], vertices[j], vertices[k]
          double_area = np.linalg.norm(np.cross(vj - vi, vk - vi))
          if double_area <= 0: continue
          cot_i = np.dot(vj - vi, vk - vi) / double_area
          cot_j = np.dot(vi - vj, vk - vj) / double_area
          cot_k = np.dot(vi - vk, vj - vk) / double_area
          mass[[i, j, k]] += double_area / 6.0
          for a, b, cot in ((j, k, cot_i), (k, i, cot_j), (i, j, cot_k)):
              rows.extend((a, b)); columns.extend((b, a)); weights.extend((cot / 2.0, cot / 2.0))
      L = coo_matrix((weights, (rows, columns)), shape=(n, n)).tocsr()
      L.setdiag(-np.asarray(L.sum(axis=1)).ravel())
      return L, mass

  def _applyLocalRBF(self, target):
    modelNode = self.explorerModelSelector.currentNode()
    V0 = self.ssm_data['template_vertices']
    indices = self.ssm_data['interpolation_indices']
    weights = self.ssm_data['interpolation_weights']
    # total landmark delta across multiple PCs
    delta_landmarks = (target - self.ssm_data['original_points_RAS'])  # (nL,3)
    neighbor_deltas = delta_landmarks[indices]             # (nV,k,3)
    disp = np.sum(neighbor_deltas * weights[..., np.newaxis], axis=1)
    new_vertices = V0 + disp

    pd = modelNode.GetPolyData()
    points = pd.GetPoints()
    points.SetData(numpy_support.numpy_to_vtk(new_vertices.astype(np.float32, copy=False), deep=1))
    points.Modified(); pd.Modified(); modelNode.Modified()

  def _applyBiharmonic(self, target):
      try:
          modelNode = self.explorerModelSelector.currentNode()
          Vn = self.ssm_data['bih_V']; n = Vn.shape[0]
          idx = self.ssm_data['constrained_idx']; lu = self.ssm_data['bih_A_lu']; Wd = self.ssm_data['bih_Wd']; lam = self.ssm_data['bih_lam']
          delta = np.asarray(target, dtype=np.float64) - self.ssm_data['original_points_RAS']
          if delta.shape != (len(idx), 3) or not np.isfinite(delta).all():
              raise ValueError("Invalid target landmark displacements.")
          b = np.zeros((n,3), np.float64)
          np.add.at(b, idx, delta)
          # Minimize bending of displacements, not of absolute coordinates.
          # Therefore identity controls preserve all rest-shape detail.
          displacement = lu.solve(lam * b)
          X = Vn + displacement
          if not np.isfinite(X).all(): raise ValueError("Biharmonic solve produced non-finite coordinates.")

          pd = modelNode.GetPolyData(); pts = pd.GetPoints()
          orig_pid = self.ssm_data.get('bih_orig_pid', None)
          if orig_pid is not None:
              base = self.ssm_data.get('base_template_vertices', None)
              if base is None:
                  base = slicer.util.arrayFromModelPoints(modelNode).astype(np.float64, copy=True)
                  self.ssm_data['base_template_vertices'] = base.copy()
              base = np.array(base, dtype=np.float32, copy=True)
              base[orig_pid] = X.astype(np.float32, copy=False)
              arr = numpy_support.numpy_to_vtk(base, deep=1)
          else:
              arr = numpy_support.numpy_to_vtk(X.astype(np.float32, copy=False), deep=1)
          pts.SetData(arr); pts.Modified(); pd.Modified(); modelNode.Modified()
          return True
      except Exception as e:
          logging.error(f"Biharmonic solve failed: {e}")
          return False



# MorphoWeaveModelLibraryTest
#

class MorphoWeaveModelLibraryTest(ScriptedLoadableModuleTest):
  def setUp(self):
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    self.setUp()
    self.assertIsNotNone(MorphoWeaveModelLibraryLogic())
