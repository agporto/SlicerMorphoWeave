set(MODULE_NAME MorphoWeaveBatchShapeCompletion)

set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  README.md
  Resources/Python/MorphoWeaveCompletionBatch.py
  )

slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT Testing/Python/BatchRunnerTest.py)
  slicer_add_python_unittest(SCRIPT Testing/Python/BatchSlicerAdapterTest.py)
endif()
