"""Slicer entry point for single and batch shape completion.

The original single-specimen/calibration implementation is retained unchanged in
MorphoWeaveShapeCompletionBase. Both tabs use its one completion pipeline.
"""
# Preserve the module's existing public helpers and logic for Slicer scripts.
from Resources.Python.MorphoWeaveShapeCompletionBase import *  # noqa: F401,F403
from Resources.Python.MorphoWeaveShapeCompletionBase import (
    MorphoWeaveShapeCompletion as _CompletionModule,
    MorphoWeaveShapeCompletionWidget as _SingleCompletionWidget,
    MorphoWeaveShapeCompletionTest as _CompletionTest,
)
from Resources.Python.MorphoWeaveShapeCompletionBatch import ShapeCompletionBatchMixin


class MorphoWeaveShapeCompletion(_CompletionModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.helpText += (
            " Use the Batch tab to process a directory of fragments with the "
            "same model and settings, optional paired landmarks, and resumable exports."
        )


class MorphoWeaveShapeCompletionWidget(ShapeCompletionBatchMixin, _SingleCompletionWidget):
    """Complete Shape, Batch, Calibration and Advanced in one module."""


class MorphoWeaveShapeCompletionTest(_CompletionTest):
    pass
