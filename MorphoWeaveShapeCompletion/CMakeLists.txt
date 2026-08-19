#-----------------------------------------------------------------------------
set(MODULE_NAME MorphoWeaveShapeCompletion)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  README.md
  TUTORIAL.md
  Resources/Icons/${MODULE_NAME}.png
  Resources/Python/__init__.py
  Resources/Python/MorphoWeaveShapeCompletionCore.py
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
endif()
