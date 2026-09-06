"""Restoration integration tests: actual widget methods/core, simulated Slicer.

The registration backend is a deterministic test double, not native rustcpd.
VTK I/O, argument preparation, batch orchestration and the restored numerical
Python code are real. This does not certify anatomical/native fitting accuracy.
"""
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / 'Resources/Python/MorphoWeaveShapeCompletionCore.py'
BASE = ROOT / 'Resources/Python/MorphoWeaveShapeCompletionBase.py'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Signal:
    def __init__(self, owner):
        self.owner, self.callbacks = owner, []
    def connect(self, callback):
        self.callbacks.append(callback)
    def emit(self, *args):
        if not self.owner.blocked:
            for fn in self.callbacks:
                fn(*args)


class Control:
    Dirs, Files = 1, 2
    def __init__(self, text=''):
        self.text = text if isinstance(text, str) else ''
        self.value, self.checked, self.currentPath = 0, False, ''
        self.currentIndex, self.node, self.enabled, self.blocked = 0, None, True, False
        self.entries = []
        for s in ('currentNodeChanged','currentPathChanged','valueChanged','toggled',
                  'clicked','textChanged','currentIndexChanged'):
            setattr(self,s,Signal(self))
    def __getattr__(self, name):
        # Rendering-only Qt methods are deliberately not implemented here.
        if name.startswith(('set','add','insert')):
            return lambda *a, **k: None
        raise AttributeError(name)
    def currentNode(self): return self.node
    def setCurrentNode(self, node):
        if self.node is not node:
            self.node = node; self.currentNodeChanged.emit(node)
    def blockSignals(self, value):
        old = self.blocked; self.blocked = bool(value); return old
    def setValue(self, value):
        if self.value != value:
            self.value = value; self.valueChanged.emit(value)
    def setChecked(self, value):
        if self.checked != value:
            self.checked = bool(value); self.toggled.emit(value)
    def setCurrentIndex(self, value): self.currentIndex = value
    def setCurrentPath(self, value):
        if self.currentPath != str(value):
            self.currentPath = str(value); self.currentPathChanged.emit(self.currentPath)
    def setText(self, text): self.text = str(text)
    setPlainText = setText
    def toPlainText(self): return self.text
    def appendPlainText(self, text): self.text += '\n'+str(text)
    def clear(self): self.text = ''
    def setEnabled(self, value): self.enabled = bool(value)
    def addTab(self, tab, name): self.entries.append((tab, name))
    def insertTab(self, index, tab, name): self.entries.insert(index, (tab, name))


class Blocker:
    def __init__(self, owner):
        self.owner = owner; self.old = owner.blockSignals(True)
    def __del__(self): self.owner.blockSignals(self.old)


class Node:
    def __init__(self, name, points=None, labels=None, polydata=None):
        self.name = name; self.points = points; self.labels = labels
        self.polydata = polydata; self.identity = str(id(self))
    def GetID(self): return self.identity
    def GetName(self): return self.name
    def GetMTime(self): return 1
    def GetParentTransformNode(self): return None
    def GetPolyData(self): return self.polydata
    def GetSingletonTag(self): return None
    def GetAttribute(self, name):
        return str(len(self.points)) if name == 'ssm_npoints' else None
    def GetNumberOfControlPoints(self): return len(self.points)


def polydata(points):
    result = vtk.vtkPolyData(); p = vtk.vtkPoints()
    p.SetData(numpy_to_vtk(np.asarray(points,dtype=float),deep=True)); result.SetPoints(p)
    cells = vtk.vtkCellArray()
    for i in range(len(points)):
        cells.InsertNextCell(1); cells.InsertCellPoint(i)
    result.SetVerts(cells); return result


class Scene:
    def __init__(self): self.nodes=[]; self.folders=set(); self.counter=0
    def add(self,node): self.nodes.append(node); return node
    def GetNumberOfNodes(self): return len(self.nodes)
    def GetNthNode(self,i): return self.nodes[i]
    def GetNodeByID(self,i): return next((n for n in self.nodes if n.GetID()==i),None)
    def RemoveNode(self,n): self.nodes.remove(n)
    def GetFirstNodeByClass(self,c): return self
    def RemoveItem(self,f): self.folders.discard(f)
    def folder(self): self.counter+=1; self.folders.add(self.counter); return self.counter


class FragmentRestorationTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name); self.scene=Scene(); self.errors=[]
        scripted=types.ModuleType('slicer.ScriptedLoadableModule')
        class Module:
            def __init__(self,parent): self.parent=parent
        class Widget:
            def __init__(self,parent=None): self.parent=parent; self.layout=Control()
            def setup(self): pass
            def enter(self): pass
            def cleanup(self): pass
        scripted.ScriptedLoadableModule=Module
        scripted.ScriptedLoadableModuleWidget=Widget
        scripted.ScriptedLoadableModuleLogic=type('Logic',(),{})
        scripted.ScriptedLoadableModuleTest=unittest.TestCase
        q=types.ModuleType('qt'); c=types.ModuleType('ctk'); s=types.ModuleType('slicer'); s.__path__=[]
        for name in ('QTabWidget','QWidget','QFormLayout','QLabel','QCheckBox','QPushButton','QProgressBar',
                     'QPlainTextEdit','QLineEdit','QSpinBox','QHBoxLayout','QToolButton','QComboBox'):
            setattr(q,name,Control)
        q.QSignalBlocker=Blocker; q.Qt=types.SimpleNamespace(RichText=1,WaitCursor=2)
        q.QTimer=types.SimpleNamespace(singleShot=lambda *args:None)
        q.QApplication=types.SimpleNamespace(setOverrideCursor=lambda *args:None,restoreOverrideCursor=lambda:None)
        c.ctkCollapsibleButton=c.ctkDoubleSpinBox=c.ctkPathLineEdit=Control
        s.mrmlScene=self.scene; s.qMRMLNodeComboBox=Control
        s.app=types.SimpleNamespace(processEvents=lambda:None)
        s.vtkMRMLSubjectHierarchyNode=types.SimpleNamespace(GetSubjectHierarchyNode=lambda scene:scene)
        s.util=types.SimpleNamespace(getNodesByClass=lambda c:[],errorDisplay=self.errors.append,
                                     showStatusMessage=lambda *a:None,loadModel=self.load_model,loadMarkups=self.load_marks)
        packaging=types.ModuleType('slicer.packaging'); packaging.pip_ensure=Mock(); s.packaging=packaging
        modules={'slicer':s,'qt':q,'ctk':c,'slicer.ScriptedLoadableModule':scripted,'slicer.packaging':packaging,
                 'Resources':types.ModuleType('Resources'),'Resources.Python':types.ModuleType('Resources.Python')}
        self.patcher=patch.dict(sys.modules,modules); self.patcher.start(); self.addCleanup(self.patcher.stop)
        core=load('Resources.Python.MorphoWeaveShapeCompletionCore',CORE); self.core=core
        self.base=load('Resources.Python.MorphoWeaveShapeCompletionBase',BASE)
        self.runner=load('Resources.Python.MorphoWeaveCompletionBatch',ROOT/'Resources/Python/MorphoWeaveCompletionBatch.py')
        self.adapter=load('Resources.Python.MorphoWeaveShapeCompletionBatch',ROOT/'Resources/Python/MorphoWeaveShapeCompletionBatch.py')
        self.entry=load('MorphoWeaveShapeCompletion',ROOT/'MorphoWeaveShapeCompletion.py')
        fixture=load('fragment_core_test_fixture',ROOT/'Testing/Python/ShapeCompletionCoreTest.py')
        self.backend=fixture._FakeRustCPD(); self.backend.__version__='4.0.0'; self.backend.__file__=__file__
        sys.modules['rustcpd']=self.backend
        self.mean,self.modes,self.eigenvalues=fixture.synthetic_ssm(m=80)
        case=self; base=self.base
        class TestLogic(base.MorphoWeaveShapeCompletionLogic):
            def ssm_from_table(self,node): return case.mean.copy(),case.modes.copy(),case.eigenvalues.copy()
            def model_polydata_world(self,node): return node.GetPolyData()
            def model_points_world(self,node): return vtk_to_numpy(node.GetPolyData().GetPoints().GetData()).copy()
            def markups_labels_points_world(self,node): return node.labels,node.points.copy()
            def create_output_folder(self,*args): return case.scene.folder()
            def create_completion_outputs(self,**kwargs):
                result=kwargs['result']; self.last_result=result; self.last_diagnostics=kwargs['diagnostics']
                node=case.scene.add(Node('completion',polydata=polydata(result.completed_points)))
                return {'folder_item':self.create_output_folder(), 'model':node,'result':result}
            def save_completion_outputs(self,outputs,diagnostics,directory):
                directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
                writer=vtk.vtkXMLPolyDataWriter(); writer.SetInputData(outputs['model'].GetPolyData())
                writer.SetFileName(str(directory/'completed.vtp'))
                if writer.Write()!=1: raise IOError('VTK export failed')
                r=outputs['result']
                np.savez(directory/'results.npz',completed=r.completed_points,epistemic=r.epistemic_variance,
                         total=r.total_variance,scale=r.world_scale,rotation=r.world_rotation,
                         translation=r.world_translation,samples=np.asarray(r.samples))
                (directory/'diagnostics.json').write_text(json.dumps(diagnostics,sort_keys=True))
        class BatchLogic(self.adapter._BatchCompletionLogic,TestLogic): pass
        self.logic_type=TestLogic; self.batch_logic_type=BatchLogic
        replace=patch.object(self.adapter,'_BatchCompletionLogic',BatchLogic); replace.start(); self.addCleanup(replace.stop)
        self.widget=self.entry.MorphoWeaveShapeCompletionWidget(); self.widget.setup(); self.widget.logic=TestLogic()
        self.inputs=self.directory/'inputs'; self.inputs.mkdir(); self.marks=self.directory/'marks'; self.marks.mkdir()
        self.target=self.mean[40:] + [0.3,0.5,0.2]
        self.model=self.scene.add(Node('reference_template',polydata=polydata(self.mean)))
        labels=[f'L{i}' for i in range(4)]; self.ids=np.array([43,51,65,73])
        self.dense=self.scene.add(Node('reference_template_correspondences',self.mean.copy(),[f'D{i}' for i in range(80)]))
        self.sparse=self.scene.add(Node('reference_template_sparse_landmarks',self.mean[self.ids],labels))
        self.table=self.scene.add(Node('ssm_data_reference',self.mean))
        for attr,node in (('template_model_selector',self.model),('template_dense_selector',self.dense),
                          ('template_sparse_selector',self.sparse),('ssm_table_selector',self.table)):
            getattr(self.widget,attr).setCurrentNode(node)
        writer=vtk.vtkXMLPolyDataWriter(); writer.SetInputData(polydata(self.target)); writer.SetFileName(str(self.inputs/'fragment.vtp'))
        self.assertEqual(writer.Write(),1)
        (self.marks/'fragment.mrk.json').write_text(json.dumps({'points':(self.mean[self.ids]+[0.3,0.5,0.2]).tolist(),'labels':labels}))
        self.widget.target_model_selector.setCurrentNode(self.load_model(str(self.inputs/'fragment.vtp')))
        self.widget.coverage_spin.setValue(0.5)
        self.widget.batch_input.setCurrentPath(str(self.inputs)); self.widget.batch_landmarks.setCurrentPath(str(self.marks))
        self.widget.batch_output.setCurrentPath(str(self.directory/'batch'))
        self.widget.output_directory.setCurrentPath(str(self.directory/'single'))
        # Deliberately doubled backend; preflight itself is tested separately.
        self.widget._ensure_dependencies=lambda:True
        logger=patch.object(self.adapter.logging,'exception'); logger.start(); self.addCleanup(logger.stop)
        self.packaging=packaging

    def load_model(self,path):
        reader=vtk.vtkXMLPolyDataReader(); reader.SetFileName(str(path)); reader.Update()
        pd=vtk.vtkPolyData(); pd.DeepCopy(reader.GetOutput())
        return self.scene.add(Node(Path(path).stem,polydata=pd))
    def load_marks(self,path):
        data=json.loads(Path(path).read_text())
        return self.scene.add(Node(Path(path).stem,np.array(data['points']),data['labels']))
    def calls(self):
        return copy.deepcopy((self.backend.pose_call,self.backend.atlas_call,self.backend.fit.posterior_kwargs))
    def assert_same(self,a,b):
        if isinstance(a,np.ndarray): np.testing.assert_array_equal(a,b)
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for k in a:self.assert_same(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for left,right in zip(a,b):self.assert_same(left,right)
        else:self.assertEqual(a,b)
    def compare_paths(self):
        self.widget._run_completion_impl(); calls=self.calls()
        old_ids=[n.GetID() for n in self.scene.nodes]
        self.widget.on_run_batch(); self.assertEqual(self.errors,[])
        self.assertIn('1 success',self.widget.batch_status.text)
        self.assert_same(calls,self.calls())
        self.assertEqual([n.GetID() for n in self.scene.nodes],old_ids)
        left=np.load(self.directory/'single/results.npz'); right=np.load(self.directory/'batch/fragment/results.npz')
        for key in left.files:np.testing.assert_array_equal(left[key],right[key])
        self.assertEqual((self.directory/'single/diagnostics.json').read_bytes(),
                         (self.directory/'batch/fragment/diagnostics.json').read_bytes())
        for key in ('single/completed.vtp','batch/fragment/completed.vtp'):
            reader=vtk.vtkXMLPolyDataReader();reader.SetFileName(str(self.directory/key));reader.Update()
            self.assertEqual(reader.GetOutput().GetNumberOfPoints(),80)
        return calls

    def test_surface_fragment_single_and_batch_match(self):
        calls=self.compare_paths()
        self.assertEqual(calls[0][-1]['translation_anchor_count'],6)
        self.assertIsNotNone(calls[0][-1]['scale_bounds'])
        self.assertEqual(calls[0][-1]['scale_bounds'],calls[1][-1]['scale_bounds'])
        self.assertGreater(calls[1][-1]['sigma2'],0)
    def test_landmark_fragment_single_and_batch_match(self):
        self.widget.target_landmark_selector.setCurrentNode(self.load_marks(self.marks/'fragment.mrk.json'))
        self.widget.batch_use_landmarks.setChecked(True)
        calls=self.compare_paths()
        self.assertEqual(len(calls[0][-1]['landmark_indices']),4)
        self.assertEqual(calls[0][-1]['landmark_sigma'],calls[1][-1]['landmark_sigma'])
    def test_nondefault_fragment_options_are_not_dropped(self):
        self.widget.pose_anchor_count.setValue(9); self.widget.pose_adaptive_mixing.setValue(0.15)
        self.widget.pose_initial_sigma2.setValue(0.2); self.widget.pose_merge_tolerance.setValue(0.01)
        self.widget.free_scale_bounds.setValue(0.2); self.widget.posterior_samples.setValue(2)
        calls=self.compare_paths(); self.assertEqual(calls[0][-1]['translation_anchor_count'],9)
        self.assertEqual(calls[0][-1]['adaptive_mixing'],0.15); self.assertEqual(calls[1][-1]['adaptive_mixing'],0.15)
        self.assertEqual(calls[0][-1]['initial_sigma2'],0.2);self.assertEqual(calls[0][-1]['merge_tolerance'],0.01)
    def test_complete_coverage_disables_additional_seeds(self):
        self.widget.coverage_spin.setValue(1.0);calls=self.compare_paths()
        self.assertEqual(calls[0][-1]['translation_anchor_count'],1)
        self.assertIsNone(calls[0][-1]['scale_bounds'])
    def test_fixed_scale_matches_single_and_batch(self):
        self.widget.scale_policy_combo.setCurrentIndex(2);calls=self.compare_paths()
        self.assertFalse(calls[0][-1]['with_scale']);self.assertFalse(calls[1][-1]['with_scale'])
    def test_resume_does_not_fit_again(self):
        self.compare_paths()
        with patch.object(self.backend,'pose_initialize',side_effect=AssertionError('must not refit')):
            self.widget.on_run_batch()
        self.assertIn('1 skipped',self.widget.batch_status.text);self.assertEqual(self.errors,[])
    def test_changed_fragment_setting_cannot_reuse_old_output(self):
        self.compare_paths(); path=self.directory/'batch/fragment/results.npz'; old=path.read_bytes()
        self.widget.pose_anchor_count.setValue(7);self.widget.on_run_batch()
        self.assertIn('1 failed',self.widget.batch_status.text);self.assertEqual(old,path.read_bytes())
    def test_missing_paired_landmarks_do_not_fall_back_to_surface_only(self):
        (self.marks/'fragment.mrk.json').unlink();self.widget.batch_use_landmarks.setChecked(True)
        with patch.object(self.backend,'pose_initialize',side_effect=AssertionError('must not fit')):
            self.widget.on_run_batch()
        self.assertIn('1 failed',self.widget.batch_status.text);self.assertFalse((self.directory/'batch/fragment').exists())
    def test_cancel_before_fit_restores_state(self):
        self.widget._batch_progress_update=lambda *a:self.widget.on_cancel_batch()
        old=self.widget.logic;self.widget.on_run_batch()
        self.assertIn('1 cancelled',self.widget.batch_status.text);self.assertIs(self.widget.logic,old)
        self.assertIsNone(self.widget._workflow_busy)
    def test_auto_selection_mirrors_with_real_blocking_semantics(self):
        w=self.widget
        for attr in ('template_model_selector','template_dense_selector','template_sparse_selector','ssm_table_selector'):
            getattr(w,attr).setCurrentNode(None)
        w._latest_complete_ssm_set=lambda:{'model':self.model,'dense':self.dense,'sparse':self.sparse,'table':self.table}
        w._auto_select_canonical_ssm_set()
        self.assertIs(w.batch_template_model.currentNode(),self.model)
        self.assertIs(w.batch_template_dense.currentNode(),self.dense)
        self.assertIs(w.batch_template_sparse.currentNode(),self.sparse)
        self.assertIs(w.batch_ssm_table.currentNode(),self.table)
    def test_tabs_and_advanced_defaults(self):
        self.assertEqual([n for _,n in self.widget.tabs.entries],['Complete Shape','Batch','Calibration','Advanced'])
        settings=self.widget._read_settings()
        self.assertEqual(settings.translation_anchor_count,6);self.assertEqual(settings.residual_scale_policy,'auto')
        self.assertEqual(settings.free_scale_bounds_fraction,0.1);self.assertEqual(settings.atlas_sigma2_from_pose_factor,2.0)
        self.assertIsNone(settings.adaptive_mixing);self.assertIsNone(settings.initial_sigma2)
    def test_metadata_and_discovery(self):
        parent=types.SimpleNamespace();self.entry.MorphoWeaveShapeCompletion(parent)
        self.assertEqual(parent.title,'Shape Completion')
        tree=ast.parse((ROOT/'MorphoWeaveShapeCompletion.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MorphoWeaveShapeCompletion')
        self.assertIn('ScriptedLoadableModule',[n.id for n in cls.bases])
        self.assertFalse((ROOT/'__init__.py').exists())
    def test_wrong_loaded_backend_is_rejected_before_pip(self):
        self.backend.__version__='3.1.0';self.widget._deps_ready=False
        result=self.entry.MorphoWeaveShapeCompletionWidget._ensure_dependencies(self.widget)
        self.assertFalse(result);self.packaging.pip_ensure.assert_not_called()
        self.assertIn('restart',self.errors[0].lower())
    def test_all_fragment_api_parameters_are_checked(self):
        self.assertRaises(RuntimeError,self.entry.validate_completion_backend,self.backend)
        source=inspect.getsource(self.entry.validate_completion_backend)
        for name in ('translation_anchor_count','scale_bounds','sigma2','adaptive_mixing','initial_sigma2','merge_tolerance'):
            self.assertIn('"'+name+'"',source)
    def test_all_original_noninstaller_callables_are_preserved(self):
        manifest=json.loads((ROOT/'RESTORATION_PROVENANCE.json').read_text())
        found={}
        for node in ast.parse(BASE.read_text()).body:
            if isinstance(node,ast.FunctionDef): found[node.name]=node
            elif isinstance(node,ast.ClassDef):
                for method in node.body:
                    if isinstance(method,ast.FunctionDef): found[node.name+'.'+method.name]=method
        for name,expected in manifest['reference_callable_ast_sha256'].items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256(ast.dump(found[name],include_attributes=False).encode()).hexdigest(),expected)

    def test_batch_files_are_identical_to_branch(self):
        manifest=json.loads((ROOT/'RESTORATION_PROVENANCE.json').read_text())
        for relative,expected in manifest['branch_batch_blobs'].items():
            content=(ROOT/relative).read_bytes()
            actual=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
            self.assertEqual(actual,expected)

    def test_reference_core_hash_unchanged(self):
        expected='aa936d91e962a0c14b1a7a7737df6330ce672f1c99291f43bcffce24f183b274'
        self.assertEqual(hashlib.sha256(CORE.read_bytes()).hexdigest(),expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
