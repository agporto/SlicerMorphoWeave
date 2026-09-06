#-----------------------------------------------------------------------------
set(MODULE_NAME MorphoWeaveShapeCompletion)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  README.md
  TUTORIAL.md
  BATCH.md
  Resources/Icons/${MODULE_NAME}.png
  Resources/Python/MorphoWeaveShapeCompletionCore.py
  Resources/Python/MorphoWeaveShapeCompletionBase.py
  Resources/Python/MorphoWeaveShapeCompletionBatch.py
  Resources/Python/MorphoWeaveCompletionBatch.py
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)
  slicer_add_python_unittest(SCRIPT Testing/Python/ShapeCompletionCoreTest.py)
  slicer_add_python_unittest(SCRIPT Testing/Python/ShapeCompletionModuleSourceTest.py)
  slicer_add_python_unittest(SCRIPT Testing/Python/BatchRunnerTest.py)
  slicer_add_python_unittest(SCRIPT Testing/Python/BatchTabTest.py)
endif()
