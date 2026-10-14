"""Load the production logic without Qt/Slicer; do not replace geometry methods.

Tests use real NumPy, SciPy and VTK, substituting only MRML/application boundaries.
This is deliberately separate from live-Slicer integration coverage.
"""
import ast
import contextlib
import copy
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import vtk
import vtk.util.numpy_support as vtk_np

MODULE_DIR = Path(__file__).resolve().parents[2]
SOURCE_PATH = MODULE_DIR / "MorphoWeaveLandmarkTransfer.py"


def polydata(vertices, faces):
    pd = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(vtk_np.numpy_to_vtk(np.asarray(vertices, dtype=float), deep=True))
    pd.SetPoints(points)
    cells = vtk.vtkCellArray()
    for face in faces:
        cells.InsertNextCell(len(face))
        for index in face:
            cells.InsertCellPoint(int(index))
    pd.SetPolys(cells)
    return pd


class Node:
    def __init__(self, pd=None, points=None):
        self.pd = vtk.vtkPolyData() if pd is None else pd
        self.points = np.zeros((0, 3)) if points is None else np.asarray(points, float).copy()

    def SetAttribute(self, *args):
        pass

    def GetDisplayNode(self):
        return None

    def GetPolyData(self):
        return self.pd

    def SetAndObservePolyData(self, pd):
        self.pd = pd

    def AddControlPoint(self, point):
        self.points = np.vstack([self.points, point])

    def GetNumberOfControlPoints(self):
        return len(self.points)

    def GetNthControlPointPosition(self, i):
        return self.points[i]

    def SetLocked(self, value):
        pass

    def SetFixedNumberOfControlPoints(self, value):
        pass


class Scene:
    def __init__(self):
        self.nodes = []
        self.states = []

    def AddNewNodeByClass(self, *args):
        node = Node()
        self.nodes.append(node)
        return node

    def RemoveNode(self, node):
        self.nodes.remove(node)

    def StartState(self, state):
        self.states.append(state)

    def EndState(self, state):
        self.states.remove(state)


class App:
    def __init__(self):
        self.paused = False

    def isRenderPaused(self):
        return self.paused

    def setRenderPaused(self, value):
        self.paused = value


def slicer_boundary():
    return SimpleNamespace(
        mrmlScene=Scene(), app=App(), vtkMRMLScene=SimpleNamespace(BatchProcessState=1),
        util=SimpleNamespace(
            arrayFromMarkupsControlPoints=lambda node: node.points.copy(),
            updateMarkupsControlPointsFromArray=lambda node, points: setattr(node, "points", points.copy()),
            NodeModify=lambda node: contextlib.nullcontext(),
        ),
    )


def load_logic(slicer=None):
    if slicer is None:
        slicer = slicer_boundary()
    namespace = dict(np=np, vtk=vtk, vtk_np=vtk_np, os=os, json=json, logging=logging,
                     copy=copy, slicer=slicer, ScriptedLoadableModuleLogic=object,
                     pose_em_enabled=lambda parameters, skip_optimization=False:
                         not skip_optimization and parameters.get("optimizationBackend") == "pose_em")
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) or
            isinstance(node, ast.ClassDef) and node.name == "MorphoWeaveLandmarkTransferLogic"]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace["MorphoWeaveLandmarkTransferLogic"](), namespace
