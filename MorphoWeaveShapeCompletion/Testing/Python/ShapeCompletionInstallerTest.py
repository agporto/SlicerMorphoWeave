import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


MODULE_DIR = Path(__file__).resolve().parents[2]
PATCH_ROOT = MODULE_DIR.parent
ROOT_INSTALLER_PATH = PATCH_ROOT / "install_into_checkout.py"
INSTALLER_PATH = ROOT_INSTALLER_PATH


def _load_installer():
    if not INSTALLER_PATH.is_file():
        return None
    spec = importlib.util.spec_from_file_location("shape_completion_installer", INSTALLER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


installer = _load_installer()


@unittest.skipUnless(installer is not None, "package-level installer is not present")
class ShapeCompletionInstallerTest(unittest.TestCase):
    def test_root_cmake_update_is_idempotent_and_ordered(self):
        source = "\n".join((
            "add_subdirectory(MorphoWeaveModelLibrary)",
            "add_subdirectory(MorphoWeaveLandmarkTransfer)",
            "add_subdirectory(MorphoWeaveSurfaceSegmentation)",
            "",
        ))
        updated = installer.add_root_cmake(source)
        self.assertEqual(updated.count("add_subdirectory(MorphoWeaveShapeCompletion)"), 1)
        self.assertLess(
            updated.index("add_subdirectory(MorphoWeaveLandmarkTransfer)"),
            updated.index("add_subdirectory(MorphoWeaveShapeCompletion)"),
        )
        self.assertEqual(installer.add_root_cmake(updated), updated)

    def test_root_readme_update_is_idempotent(self):
        source = """# MorphoWeave

- target-specific template optimization before registration; and
- correspondence-guided segmentation of surface models.

## Workflow
3. **Landmark Transfer** transfers landmarks to individual specimens or batches using rigid registration, SSM-guided CPD, optional fine deformation, and surface projection.
4. **Surface Segmentation** uses dense correspondences to divide homologous surface regions consistently across specimens.

### Surface Segmentation
Existing section.
"""
        updated = installer.update_root_readme(source)
        self.assertEqual(updated.count("### Shape Completion"), 1)
        self.assertEqual(updated.count("**Shape Completion** completes partial target surfaces"), 1)
        self.assertIn("5. **Surface Segmentation**", updated)
        self.assertEqual(installer.update_root_readme(updated), updated)

    def test_tutorial_inserts_description_and_full_section_once(self):
        source = """# Tutorial

* Continuous optimization of the template (pose + shape) before large batch runs

`MorphoWeave` organizes functionality into four scripted modules: `Atlas Builder`, `Model Library`, `Landmark Transfer`, and `Surface Segmentation`.

#### Surface Segmentation
Description.

## Surface Segmentation
Instructions.
"""
        updated = installer.update_tutorial(source)
        self.assertEqual(updated.count("\n#### Shape Completion\n"), 1)
        self.assertEqual(updated.count("\n## Shape Completion\n"), 1)
        self.assertIn("five scripted modules", updated)
        self.assertEqual(installer.update_tutorial(updated), updated)

    def test_installer_dry_run_install_refusal_and_force_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "SlicerMorphoWeave"
            (checkout / "tutorial").mkdir(parents=True)
            (checkout / "CMakeLists.txt").write_text(
                "add_subdirectory(MorphoWeaveLandmarkTransfer)\n"
                "add_subdirectory(MorphoWeaveSurfaceSegmentation)\n",
                encoding="utf-8",
            )
            (checkout / "README.md").write_text(
                "# MorphoWeave\n\n"
                "- target-specific template optimization before registration;\n\n"
                "3. **Landmark Transfer** transfers landmarks to individual specimens or batches using rigid registration, SSM-guided CPD, optional fine deformation, and surface projection.\n"
                "4. **Surface Segmentation** uses dense correspondences consistently.\n\n"
                "### Surface Segmentation\n",
                encoding="utf-8",
            )
            (checkout / "tutorial" / "README.md").write_text(
                "# Tutorial\n\n"
                "* Continuous optimization of the template (pose + shape) before large batch runs\n\n"
                "`MorphoWeave` organizes functionality into four scripted modules: `Atlas Builder`, `Model Library`, `Landmark Transfer`, and `Surface Segmentation`.\n\n"
                "#### Surface Segmentation\n\n## Surface Segmentation\n",
                encoding="utf-8",
            )

            dry = subprocess.run(
                [sys.executable, str(INSTALLER_PATH), str(checkout), "--dry-run"],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(dry.returncode, 0, dry.stderr)
            self.assertFalse((checkout / "MorphoWeaveShapeCompletion").exists())

            first = subprocess.run(
                [sys.executable, str(INSTALLER_PATH), str(checkout)],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertTrue((checkout / "MorphoWeaveShapeCompletion" / "MorphoWeaveShapeCompletion.py").exists())

            refused = subprocess.run(
                [sys.executable, str(INSTALLER_PATH), str(checkout)],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(refused.returncode, 2)

            marker = checkout / "MorphoWeaveShapeCompletion" / "local-marker.txt"
            marker.write_text("replace me", encoding="utf-8")
            forced = subprocess.run(
                [sys.executable, str(INSTALLER_PATH), str(checkout), "--force"],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(forced.returncode, 0, forced.stderr)
            self.assertFalse(marker.exists())
            self.assertTrue(
                (checkout / ".shape-completion-backup" / "MorphoWeaveShapeCompletion" / "local-marker.txt").exists()
            )

            cmake = (checkout / "CMakeLists.txt").read_text(encoding="utf-8")
            readme = (checkout / "README.md").read_text(encoding="utf-8")
            tutorial = (checkout / "tutorial" / "README.md").read_text(encoding="utf-8")
            self.assertEqual(cmake.count("add_subdirectory(MorphoWeaveShapeCompletion)"), 1)
            self.assertEqual(readme.count("### Shape Completion"), 1)
            self.assertEqual(tutorial.count("\n#### Shape Completion\n"), 1)
            self.assertEqual(tutorial.count("\n## Shape Completion\n"), 1)


if __name__ == "__main__":
    unittest.main()
