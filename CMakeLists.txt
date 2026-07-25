cmake_minimum_required(VERSION 3.5)

project(MorphoWeave)

# Extension metadata displayed in the Slicer Extensions Manager.
set(EXTENSION_HOMEPAGE "https://github.com/agporto/SlicerMorphoWeave")
set(EXTENSION_CATEGORY "Registration")
set(EXTENSION_CONTRIBUTORS "Arthur Porto (Florida Museum of Natural History, University of Florida)")
set(EXTENSION_DESCRIPTION "MorphoWeave provides workflows for constructing 3D anatomical atlases and statistical shape models and for automatically transferring landmarks to new surface models. It combines rigid surface registration, shape-model-guided deformable registration, and optional surface projection.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/agporto/SlicerMorphoWeave/main/logo.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/agporto/SlicerMorphoWeave/main/tutorial/images/20.png")
set(EXTENSION_DEPENDS "")

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

add_subdirectory(MorphoWeaveModelLibrary)
add_subdirectory(MorphoWeaveAtlasBuilder)
add_subdirectory(MorphoWeaveLandmarkTransfer)
add_subdirectory(MorphoWeaveSurfaceSegmentation)

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
