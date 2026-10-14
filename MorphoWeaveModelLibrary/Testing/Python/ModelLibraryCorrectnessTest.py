"""Real NumPy/SciPy/VTK regression tests; only Slicer/Qt scene and UI boundaries are doubled.

The production logic/widget classes are compiled verbatim from their source AST.
No numerical method is reimplemented or replaced. This suite does not exercise
live MRML observers, Qt signal ordering, or Slicer readers/writers.
Run: python MorphoWeaveModelLibrary/Testing/Python/ModelLibraryCorrectnessTest.py
"""
import ast
import csv
import json
import logging
import os
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import vtk
from vtk.util import numpy_support
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import connected_components

SOURCE = Path(__file__).resolve().parents[2] / 'MorphoWeaveModelLibrary.py'
SLICER = types.SimpleNamespace()
CTK = types.SimpleNamespace()
NAMESPACE = dict(np=np, vtk=vtk, numpy_support=numpy_support, cKDTree=cKDTree,
                 coo_matrix=coo_matrix, diags=diags, splu=splu, connected_components=connected_components, os=os, json=json,
                 shutil=shutil, logging=logging, csv=csv, tempfile=tempfile, Path=Path,
                 slicer=SLICER, qt=types.SimpleNamespace(), ctk=CTK,
                 ScriptedLoadableModuleLogic=object, ScriptedLoadableModuleWidget=object)
TREE = ast.parse(SOURCE.read_text(encoding='utf-8'), filename=str(SOURCE))
CLASSES = [node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name in
           ('MorphoWeaveModelLibraryLogic', 'MorphoWeaveModelLibraryWidget')]
exec(compile(ast.Module(body=CLASSES, type_ignores=[]), str(SOURCE), 'exec'), NAMESPACE)
Logic = NAMESPACE['MorphoWeaveModelLibraryLogic']
Widget = NAMESPACE['MorphoWeaveModelLibraryWidget']


def polydata(vertices, faces):
    points = vtk.vtkPoints(); points.SetData(numpy_support.numpy_to_vtk(np.asarray(vertices, float), deep=True))
    poly = vtk.vtkPolyData(); poly.SetPoints(points)
    cells = vtk.vtkCellArray()
    for face in faces:
        cells.InsertNextCell(len(face))
        for index in face: cells.InsertCellPoint(int(index))
    poly.SetPolys(cells)
    return poly


def coordinates(poly):
    return numpy_support.vtk_to_numpy(poly.GetPoints().GetData()).copy()


def write_markups(path, points, cs='RAS', labels=None, statuses=None):
    controls = [dict(position=list(map(float, point)), label=labels[i] if labels is not None else f'p{i}',
                     positionStatus=statuses[i] if statuses is not None else 'defined')
                for i, point in enumerate(points)]
    Path(path).write_text(json.dumps({'markups': [dict(coordinateSystem=cs, controlPoints=controls)]}))


def write_fcsv(path, points, cs='RAS', labels=None):
    with open(path, 'w', newline='') as stream:
        stream.write(f'# CoordinateSystem = {cs}\n# columns = id,x,y,z,label\n')
        writer = csv.writer(stream)
        for i, point in enumerate(points): writer.writerow([i, *point, labels[i] if labels is not None else f'p{i}'])


class Node:
    """Slicer MRML node boundary only; geometry remains real vtkPolyData."""
    def __init__(self, poly=None, points=None): self.poly = poly; self.points = points; self.name = None
    def GetPolyData(self): return self.poly
    def Modified(self): pass
    def SetName(self, name): self.name = name


class TableNode(Node):
    def __init__(self): super().__init__(); self.attributes = {}; self.columns = {}
    def GetTable(self): return self
    def GetColumn(self, index): return types.SimpleNamespace(SetName=lambda name: self.columns.update({index: name}))
    def SetAttribute(self, key, value): self.attributes[key] = value


class Signal:
    def connect(self, _): pass
    def disconnect(self, _): pass


class Slider:
    def __init__(self): self.valueChanged = Signal(); self.value = 0
    def setToolTip(self, _): pass


def explorer(poly, controls, method=1):
    widget = Widget.__new__(Widget)
    widget.model = Node(poly)
    widget.landmarks = Node(points=np.array(controls, float))
    widget.explorerModelSelector = types.SimpleNamespace(currentNode=lambda: widget.model)
    widget.explorerLandmarksSelector = types.SimpleNamespace(currentNode=lambda: widget.landmarks)
    widget.methodCombo = types.SimpleNamespace(currentIndex=method)
    widget.ssm_data = dict(base_template_vertices=coordinates(poly), original_points_RAS=np.array(controls, float))
    return widget


class ModelLibraryCorrectnessTest(unittest.TestCase):
    def setUp(self):
        self.logic = Logic()
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.population = self.root / 'population'; self.population.mkdir()
        self.points = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 3]], float)
        self.faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        self.model = self.root / 'template.ply'
        writer = vtk.vtkPLYWriter(); writer.SetFileName(str(self.model)); writer.SetInputData(polydata(self.points, self.faces))
        self.assertEqual(writer.Write(), 1)
        self.dense = self.root / 'dense.mrk.json'; self.sparse = self.root / 'sparse.mrk.json'
        write_markups(self.dense, self.points); write_markups(self.sparse, self.points[:3])
        write_markups(self.population / 'a.mrk.json', self.points)
        write_markups(self.population / 'b.mrk.json', self.points + [1, 0, 0])
        self.database = self.root / 'database'
        SLICER.util = types.SimpleNamespace(
            updateMarkupsControlPointsFromArray=lambda node, array: setattr(node, 'points', np.array(array)),
            arrayFromModelPoints=lambda node: coordinates(node.poly))

    def ingest(self):
        return self.logic.ingestSSMDatabase(str(self.database), str(self.model), str(self.dense), str(self.sparse), str(self.population))

    def snapshot(self):
        return {p.relative_to(self.database).as_posix(): p.read_bytes() for p in self.database.rglob('*') if p.is_file()}

    def test_coordinate_tags_and_equivalent_populations(self):
        ras = self.points + [10, 20, 30]
        lps = ras * [-1, -1, 1]
        for tag in (0, ' 0\n', 'ras', ' RAS '): self.assertEqual(self.logic._norm_cs(tag), 'RAS')
        for tag in (1, ' 1\n', 'lps', ' LPS '): self.assertEqual(self.logic._norm_cs(tag), 'LPS')
        for tag in (2, 'LPS-ish', '', None):
            with self.subTest(tag=tag), self.assertRaises(ValueError): self.logic._norm_cs(tag)
        for suffix, writer in (('.fcsv', write_fcsv), ('.mrk.json', write_markups)):
            for tag, values in (('0', ras), ('1', lps), ('RAS', ras), ('LPS', lps)):
                path = self.root / ('coordinates' + suffix); writer(path, values, tag)
                np.testing.assert_array_equal(self.logic._markups_points_ras(path), ras)
        source = np.stack([ras, ras + [0.2, 0, 0]])
        parsed = []
        for i, specimen in enumerate(source):
            path = self.root / f'converted{i}.fcsv'; write_fcsv(path, specimen * [-1, -1, 1], ' 1 ')
            parsed.append(self.logic._markups_points_ras(path))
        for a, b in zip(self.logic.build_ssm(source)[:3], self.logic.build_ssm(np.stack(parsed))[:3]):
            np.testing.assert_array_equal(a, b)

    def test_malformed_rows_never_reduce_cohort_or_shift_correspondence(self):
        for p in self.population.iterdir(): p.unlink()
        for specimen, bad_row in (('a', 1), ('b', 2)):
            rows = [[i, *point] for i, point in enumerate(self.points)]
            rows[bad_row][1] = 'not-a-coordinate'
            (self.population / f'{specimen}.fcsv').write_text('# CoordinateSystem = 0\n' + '\n'.join(','.join(map(str, row)) for row in rows))
        ok, message = self.ingest()
        self.assertFalse(ok); self.assertIn('invalid point on line', message); self.assertFalse(self.database.exists())

    def test_empty_short_and_nonfinite_inputs_fail_closed(self):
        for content in ('# CoordinateSystem = 0\n', '0,1,2\n', '0,NaN,2,3\n', '0,inf,2,3\n'):
            path = self.root / 'bad.fcsv'; path.write_text(content)
            with self.subTest(content=content), self.assertRaises(ValueError): self.logic._markups_points_ras(path)
        for bad_points in ([], [[0, 1]], [[0, 1, float('inf')]]):
            write_markups(self.dense, bad_points)
            self.assertFalse(self.ingest()[0]); self.assertFalse(self.database.exists())

    def test_undefined_json_position_rejected(self):
        write_markups(self.dense, self.points, statuses=['defined', 'undefined', 'defined', 'defined'])
        ok, message = self.ingest()
        self.assertFalse(ok); self.assertIn('defined position', message); self.assertFalse(self.database.exists())

    def test_shared_labels_or_ids_cannot_silently_reorder(self):
        write_markups(self.population / 'b.mrk.json', self.points[[0, 2, 1, 3]], labels=['p0', 'p2', 'p1', 'p3'])
        ok, message = self.ingest(); self.assertFalse(ok); self.assertIn('order differs', message)
        # IDs are checked independently, even if labels are file-specific.
        with self.assertRaises(ValueError):
            self.logic._check_point_order({'id': ['1', '2']}, {'id': ['2', '1']}, 'specimen')
        self.logic._check_point_order({'label': ['A-1', 'A-2']}, {'label': ['B-1', 'B-2']}, 'specimen')

    def test_later_specimens_with_shared_identifier_sets_are_compared(self):
        for kind in ('label', 'id'):
            with self.subTest(kind=kind):
                write_markups(self.dense, self.points, labels=[f'template-{i}' for i in range(4)])
                write_markups(self.population / 'a.mrk.json', self.points, labels=[f'a-{i}' for i in range(4)])
                write_markups(self.population / 'b.mrk.json', self.points + [1, 0, 0], labels=[f'b-{i}' for i in range(4)])
                write_markups(self.population / 'c.mrk.json', self.points[[0, 2, 1, 3]] + [2, 0, 0], labels=[f'c-{i}' for i in range(4)])
                for name, order in (('b', [0, 1, 2, 3]), ('c', [0, 2, 1, 3])):
                    path = self.population / f'{name}.mrk.json'; data = json.loads(path.read_text())
                    for cp, i in zip(data['markups'][0]['controlPoints'], order): cp[kind] = f'p{i}'
                    path.write_text(json.dumps(data))
                ok, message = self.ingest()
                self.assertFalse(ok); self.assertIn(f'{kind} order differs', message)
                self.assertFalse(self.database.exists())
        # Unrelated complete sets must still be accepted, not guessed equivalent.
        for name in ('a', 'b', 'c'):
            write_markups(self.population / f'{name}.mrk.json', self.points + [ord(name) - ord('a'), 0, 0], labels=[f'{name}-{i}' for i in range(4)])
        self.assertTrue(self.ingest()[0])

    def test_unsafe_database_names_are_rejected_before_ui_overwrite(self):
        library = self.root / 'library'; library.mkdir()
        sibling = library / 'other_entry'; sibling.mkdir(); sentinel = sibling / 'model.npz'; sentinel.write_bytes(b'old and unrelated')
        SLICER.util.errorDisplay = mock.Mock()
        SLICER.util.confirmOkCancelDisplay = mock.Mock(side_effect=AssertionError('unsafe overwrite prompt'))
        widget = Widget.__new__(Widget); widget.logic = self.logic
        widget.ssmPopulationDirSelector = types.SimpleNamespace(currentPath=str(self.population))
        widget.ssmTemplateModelSelector = types.SimpleNamespace(currentPath=str(self.model))
        widget.databasePathSelector = types.SimpleNamespace(currentPath=str(library))
        for name in ('.', '..', '../outside', str(self.root / 'absolute'), 'nested/name', 'nested\\name', ' ', 'CON', 'trailing.'):
            with self.subTest(name=name):
                widget.ssmNameEditor = types.SimpleNamespace(text=name)
                widget.onIngestButton()
                self.assertTrue(sentinel.exists()); self.assertEqual(sentinel.read_bytes(), b'old and unrelated')
                SLICER.util.confirmOkCancelDisplay.assert_not_called()
        for name in ('mouse_atlas', 'Mouse Atlas 2026'): self.assertTrue(self.logic.isSafeLibraryName(name))
        self.assertFalse(self.logic.isSafeLibraryName(''))

    def test_direct_api_cannot_replace_library_or_population(self):
        library = self.root / 'library'; library.mkdir()
        sibling = library / 'other_entry'; sibling.mkdir(); sentinel = sibling / 'model.npz'; sentinel.write_bytes(b'keep')
        previous = self.database
        for unsafe in (str(library) + '/.', str(library) + '/..', str(library), str(self.population), str(self.root)):
            with self.subTest(destination=unsafe):
                self.database = unsafe
                self.assertFalse(self.ingest()[0]); self.assertEqual(sentinel.read_bytes(), b'keep')
                self.assertTrue((self.population / 'a.mrk.json').exists())
        self.database = previous
        with self.assertRaises(ValueError): self.logic._database_destination(self.root / 'outside', library)
        with self.assertRaises(ValueError): self.logic._database_destination(library / 'nested' / 'child', library)

    def test_unrelated_manifest_is_not_database_authority(self):
        self.database.mkdir()
        manifest = self.database / 'manifest.json'; manifest.write_text('{"project": "unrelated"}')
        sentinel = self.database / 'important.txt'; sentinel.write_text('preserve')
        self.assertFalse(self.ingest()[0]); self.assertEqual(sentinel.read_text(), 'preserve')
        self.assertEqual(json.loads(manifest.read_text()), {'project': 'unrelated'})

    def test_symlink_and_ancestor_alias_destinations_are_rejected(self):
        library = self.root / 'library'; library.mkdir()
        alias = self.root / 'alias'; alias.symlink_to(library, target_is_directory=True)
        target_alias = library / 'entry'; target_alias.symlink_to(library, target_is_directory=True)
        for destination, root in ((target_alias, library), (alias / 'entry2', alias), (alias / 'entry3', None),
                                  (library / 'entry' / 'entry4', library)):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                self.logic._database_destination(destination, root)
        self.assertTrue(target_alias.is_symlink()); self.assertTrue(alias.is_symlink())

    def test_backup_cleanup_failure_preserves_published_and_old_generations(self):
        self.assertTrue(self.ingest()[0]); old = self.snapshot()
        write_markups(self.population / 'b.mrk.json', self.points + [3, 0, 0])
        original_rmtree = shutil.rmtree
        def fail_backup(path, *args, **kwargs):
            if '.backup-' in str(path): raise OSError('injected backup cleanup failure')
            return original_rmtree(path, *args, **kwargs)
        with mock.patch.object(shutil, 'rmtree', side_effect=fail_backup), self.assertLogs(level='WARNING'):
            self.assertTrue(self.ingest()[0])
        backups = list(self.root.glob('.database.backup-*')); self.assertEqual(len(backups), 1)
        retained = {p.relative_to(backups[0]).as_posix(): p.read_bytes() for p in backups[0].rglob('*') if p.is_file()}
        self.assertEqual(retained, old)
        with np.load(self.database / 'ssm_model.npz') as data:
            np.testing.assert_allclose(data['mean_shape'], self.points + [1.5, 0, 0])

    def test_dense_count_and_invalid_mesh_rejected_before_persistence(self):
        write_markups(self.dense, self.points[:3])
        ok, message = self.ingest(); self.assertFalse(ok); self.assertIn('template dense', message)
        self.assertFalse(self.database.exists())
        write_markups(self.dense, self.points); self.model.write_text('not a mesh')
        self.assertFalse(self.ingest()[0]); self.assertFalse(self.database.exists())

    def test_singleton_and_identical_cohorts_are_actionably_rejected(self):
        (self.population / 'b.mrk.json').unlink()
        ok, message = self.ingest(); self.assertFalse(ok); self.assertIn('at least two distinct', message)
        write_markups(self.population / 'b.mrk.json', self.points)
        ok, message = self.ingest(); self.assertFalse(ok); self.assertIn('no modes of variation', message)
        self.assertFalse(self.database.exists())
        # All-zero coordinates are valid positions, not grounds to skip a file.
        write_markups(self.population / 'a.mrk.json', np.zeros_like(self.points))
        self.assertTrue(self.ingest()[0])

    def test_failed_replacement_preserves_old_database_at_every_write_step(self):
        self.assertTrue(self.ingest()[0]); old = self.snapshot()
        write_markups(self.population / 'b.mrk.json', self.points + [3, 0, 0])
        original_copy = shutil.copy2; original_replace = os.replace
        cases = [(shutil, 'copy2', index, original_copy) for index in (1, 2, 3)]
        cases += [(np, 'savez_compressed', 1, np.savez_compressed), (json, 'dump', 1, json.dump)]
        cases += [(os, 'replace', index, original_replace) for index in (1, 2)]
        for module, name, fail_at, original in cases:
            calls = [0]
            def injected(*args, **kwargs):
                calls[0] += 1
                if calls[0] == fail_at: raise OSError(f'injected {name} failure')
                return original(*args, **kwargs)
            with self.subTest(step=name, index=fail_at), mock.patch.object(module, name, side_effect=injected):
                ok, message = self.ingest()
            self.assertFalse(ok, message); self.assertEqual(self.snapshot(), old)
            self.assertFalse(list(self.root.glob('.database.stage-*')))
        # The historical missing sparse source must not touch any old bytes.
        self.sparse.unlink(); self.assertFalse(self.ingest()[0]); self.assertEqual(self.snapshot(), old)

    def test_successful_replacement_publishes_one_complete_generation(self):
        self.assertTrue(self.ingest()[0])
        write_markups(self.population / 'b.mrk.json', self.points + [3, 0, 0])
        self.assertTrue(self.ingest()[0])
        with np.load(self.database / 'ssm_model.npz') as data:
            np.testing.assert_allclose(data['mean_shape'], self.points + [1.5, 0, 0])
        self.assertFalse(list(self.root.glob('.database.*')))

    def test_corrupt_staged_npz_does_not_replace_database(self):
        self.assertTrue(self.ingest()[0]); old = self.snapshot()
        def corrupt(path, **kwargs): Path(path).write_bytes(b'corrupt')
        with mock.patch.object(np, 'savez_compressed', side_effect=corrupt): self.assertFalse(self.ingest()[0])
        self.assertEqual(self.snapshot(), old)

    def test_load_validates_before_scene_mutation_and_preserves_manifest_roles(self):
        self.assertTrue(self.ingest()[0])
        SLICER.util.loadModel = mock.Mock(side_effect=AssertionError('scene mutation before validation'))
        # Sparse happens to match the SSM count, but must not override manifest roles.
        manifest = json.loads((self.database / 'manifest.json').read_text())
        write_markups(self.database / manifest['files']['dense'], self.points[:3])
        write_markups(self.database / manifest['files']['sparse'], self.points)
        ok, message = self.logic.loadSSMDatabase(str(self.database))
        self.assertFalse(ok); self.assertIn('Dense template', message); SLICER.util.loadModel.assert_not_called()
        write_markups(self.database / manifest['files']['dense'], self.points)
        np.savez(self.database / 'ssm_model.npz', mean_shape=self.points, modes=np.zeros((4, 3, 0)), eigenvalues=[])
        ok, message = self.logic.loadSSMDatabase(str(self.database))
        self.assertFalse(ok); self.assertIn('no modes of variation', message); SLICER.util.loadModel.assert_not_called()

    def test_invalid_ssm_arrays_rejected_before_scene_mutation(self):
        self.assertTrue(self.ingest()[0])
        SLICER.util.loadModel = mock.Mock(side_effect=AssertionError('scene mutation before validation'))
        valid_modes = np.ones((4, 3, 1)) / np.sqrt(12)
        for mean, modes, eig in ((self.points, valid_modes, [-1]),
                                 (self.points, valid_modes, [np.inf]),
                                 (self.points, np.ones((3, 3, 1)), [1]),
                                 (self.points * np.nan, valid_modes, [1])):
            np.savez(self.database / 'ssm_model.npz', mean_shape=mean, modes=modes, eigenvalues=eig)
            self.assertFalse(self.logic.loadSSMDatabase(str(self.database))[0])
            SLICER.util.loadModel.assert_not_called()

    def test_actual_loader_roundtrip_with_explicit_scene_boundaries(self):
        self.assertTrue(self.ingest()[0]); table = TableNode()
        SLICER.util.loadModel = lambda _: Node(polydata(self.points, self.faces))
        SLICER.util.loadMarkups = lambda path: Node(points=self.logic._markups_points_ras(path))
        SLICER.util.updateTableFromArray = lambda node, array: setattr(node, 'array', array.copy())
        SLICER.mrmlScene = types.SimpleNamespace(AddNewNodeByClass=lambda *args: table)
        ok, message = self.logic.loadSSMDatabase(str(self.database)); self.assertTrue(ok, message)
        self.assertEqual(table.array.shape, (12, 2)); self.assertEqual(table.columns, {0: 'mean_shape', 1: 'mode_0'})
        self.assertEqual(table.attributes['ssm_npoints'], '4')
        self.assertGreater(json.loads(table.attributes['ssm_eigenvalues'])[0], 0)

    def test_pca_full_retention_numerical_rank_and_units_both_branches(self):
        rng = np.random.default_rng(42)
        for n, points in ((9, 2), (3, 20)):
            shapes = rng.normal(size=(n, points, 3))
            for scale in (1e-9, 1, 1e9):
                for options in ({'variance_threshold': 1}, {'variance_threshold': None}, {'max_modes': 100}):
                    mean, modes, eig, count = self.logic.build_ssm(shapes * scale, **options)
                    flat = modes.reshape(points * 3, count).astype(float)
                    self.assertEqual(count, min(n - 1, points * 3)); self.assertTrue(np.all(eig > 0))
                    np.testing.assert_allclose(flat.T @ flat, np.eye(count), atol=2e-7)
                    covariance = np.cov((shapes * scale).reshape(n, -1), rowvar=False)
                    np.testing.assert_allclose(flat @ np.diag(eig / scale**2) @ flat.T, covariance / scale**2, atol=2e-7, rtol=2e-6)
        coefficients = np.array([-2, -1, 0, 1, 2], float)
        for points in (1, 20):
            direction = rng.normal(size=(points, 3))
            _, modes, eig, count = self.logic.build_ssm(coefficients[:, None, None] * direction, variance_threshold=1)
            self.assertEqual(count, 1)
            self.assertAlmostEqual(eig[0], np.var(coefficients, ddof=1) * np.sum(direction**2), places=10)

    def test_pca_invalid_arguments_and_arrays(self):
        shapes = np.stack([self.points, self.points + 1])
        for options in ({'max_modes': 0}, {'max_modes': 1.5}, {'max_modes': True}, {'variance_threshold': 0},
                        {'variance_threshold': 1.1}, {'variance_threshold': float('nan')}):
            with self.subTest(options=options), self.assertRaises(ValueError): self.logic.build_ssm(shapes, **options)
        for invalid in (np.empty((0, 4, 3)), np.zeros((2, 4, 2)), np.full((2, 4, 3), np.nan)):
            with self.assertRaises(ValueError): self.logic.build_ssm(invalid)

    def sphere_widget(self):
        source = vtk.vtkSphereSource(); source.SetRadius(1); source.SetThetaResolution(12); source.SetPhiResolution(10); source.Update()
        poly = vtk.vtkPolyData(); poly.DeepCopy(source.GetOutput()); vertices = coordinates(poly)
        ids = np.unique([np.argmax(vertices[:, axis] * sign) for axis in range(3) for sign in (-1, 1)])
        return explorer(poly, vertices[ids]), vertices

    def test_biharmonic_identity_tiny_translation_and_rest_detail(self):
        widget, vertices = self.sphere_widget(); self.assertTrue(widget.precomputeDeformer())
        controls = widget.ssm_data['original_points_RAS']
        for delta in ([0, 0, 0], [0, 0, 1e-5], [0.2, -0.3, 0.4]):
            self.assertTrue(widget._applyBiharmonic(controls + delta))
            np.testing.assert_allclose(coordinates(widget.model.poly), vertices + delta, atol=7e-8, rtol=1e-6)

    def test_cotangents_match_analytic_triangle_and_orientation(self):
        vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]], float)
        expected = np.array([[-1.25, .25, 1], [.25, -.25, 0], [1, 0, -1]])
        for faces in ([[0, 1, 2]], [[2, 1, 0]], [[1, 2, 0]]):
            matrix, mass = Widget._cotangent_laplacian(vertices, np.array(faces))
            np.testing.assert_allclose(matrix.toarray(), expected, atol=1e-15)
            np.testing.assert_allclose(mass, np.ones(3) / 3, atol=1e-15)

    def test_biharmonic_winding_and_vertex_id_invariance(self):
        vertices = np.vstack([self.points, [1, 1, 1]])
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 4], [1, 4, 3], [2, 3, 4]])
        outputs = []
        for order, triangles in ((np.arange(5), faces), (np.arange(5), faces[:, ::-1]), (np.array([4, 2, 0, 3, 1]), faces)):
            inverse = np.argsort(order)
            widget = explorer(polydata(vertices[order], inverse[triangles]), vertices[:4])
            self.assertTrue(widget.precomputeDeformer())
            self.assertTrue(widget._applyBiharmonic(vertices[:4] + [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, .2]]))
            outputs.append(coordinates(widget.model.poly)[inverse])
        for output in outputs[1:]: np.testing.assert_allclose(output, outputs[0], atol=2e-7)

    def test_backend_switch_uses_immutable_rest_geometry(self):
        widget, vertices = self.sphere_widget(); controls = widget.ssm_data['original_points_RAS']
        widget.methodCombo.currentIndex = 0; self.assertTrue(widget.precomputeDeformer())
        widget._applyLocalRBF(controls + [1, 0, 0])
        widget.methodCombo.currentIndex = 1; self.assertTrue(widget.precomputeDeformer())
        self.assertTrue(widget._applyBiharmonic(controls + [0, .2, 0]))
        np.testing.assert_allclose(coordinates(widget.model.poly), vertices + [0, .2, 0], atol=1e-7)
        widget.methodCombo.currentIndex = 0; self.assertTrue(widget.precomputeDeformer())
        widget._applyLocalRBF(controls)
        np.testing.assert_array_equal(coordinates(widget.model.poly), vertices)

    def test_duplicate_snapped_constraints_are_order_independent(self):
        controls = np.vstack([self.points, self.points[0] + [0, 0, .001]])
        targets = controls + [[0, 0, .1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, .3]]
        outputs = []
        for order in (np.arange(5), np.array([4, 1, 2, 3, 0])):
            widget = explorer(polydata(self.points, self.faces), controls[order])
            self.assertTrue(widget.precomputeDeformer()); self.assertTrue(widget._applyBiharmonic(targets[order]))
            outputs.append(coordinates(widget.model.poly))
        np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-7)

    def test_slider_rebuild_immediately_resets_geometry_and_landmarks(self):
        widget = explorer(polydata(self.points, self.faces), self.points, method=0)
        self.assertTrue(widget.precomputeDeformer()); widget._applyLocalRBF(self.points + 1)
        widget.landmarks.points = self.points + 1
        widget.ssm_data.update(modes_RAS=np.ones((4, 3, 2)), std_devs=np.ones(2))
        widget.pcSliders = [Slider()]; widget.pcSliders[0].value = 1
        widget.numPCSpin = types.SimpleNamespace(value=2)
        widget._recreateSlidersPanel = lambda: None
        widget.slidersLayout = types.SimpleNamespace(addRow=lambda *args: None)
        CTK.ctkSliderWidget = Slider
        widget.rebuildPCSliders()
        np.testing.assert_array_equal(widget._currentPCWeights(), [0, 0])
        np.testing.assert_array_equal(coordinates(widget.model.poly), self.points)
        np.testing.assert_array_equal(widget.landmarks.points, self.points)

    def test_unconstrained_disconnected_surface_fails_closed(self):
        vertices = np.vstack([self.points, self.points + [10, 0, 0]])
        faces = np.vstack([self.faces, self.faces + 4])
        widget = explorer(polydata(vertices, faces), self.points)
        with self.assertLogs(level='ERROR') as messages:
            self.assertFalse(widget.precomputeDeformer())
        self.assertIn('disconnected template component', '\n'.join(messages.output))
        np.testing.assert_array_equal(coordinates(widget.model.poly), vertices)

    def test_single_control_local_smoother_keeps_query_dimensions(self):
        widget = explorer(polydata(self.points, self.faces), self.points[:1], method=0)
        self.assertTrue(widget.precomputeDeformer()); widget._applyLocalRBF(self.points[:1] + [1, 2, 3])
        np.testing.assert_allclose(coordinates(widget.model.poly), self.points + [1, 2, 3])


if __name__ == '__main__':
    unittest.main(verbosity=2)
