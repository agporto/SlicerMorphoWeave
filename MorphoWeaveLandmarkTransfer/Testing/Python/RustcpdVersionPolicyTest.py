"""Dependency-policy regression tests, without Slicer or native registration.

Execute the real installer method extracted from the module, substituting only
package-management and import operations. No packages are installed by tests.
"""
import ast
import importlib
import logging
from pathlib import Path
import re
import sys
import types
import unittest
from unittest.mock import Mock, patch

MODULE_DIR = Path(__file__).resolve().parents[2]
REPO = MODULE_DIR.parent
PIN = "rustcpd==4.0.0"


class RustcpdVersionPolicyTest(unittest.TestCase):
    def setUp(self):
        path = MODULE_DIR / "MorphoWeaveLandmarkTransfer.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "MorphoWeaveLandmarkTransferWidget")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_ensure_dependencies")
        namespace = {"logging": logging}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        self.ensure = lambda: namespace["_ensure_dependencies"](types.SimpleNamespace())
        self.backend = types.SimpleNamespace(__version__="4.0.0")
        self.slicer = types.ModuleType("slicer")
        self.slicer.__path__ = []
        self.slicer.util = types.SimpleNamespace(
            infoDisplay=Mock(), errorDisplay=Mock(), showStatusMessage=Mock(),
            confirmOkCancelDisplay=Mock(return_value=True),
        )
        self.packaging = types.ModuleType("slicer.packaging")
        self.packaging.pip_check = Mock(return_value=False)
        self.packaging.pip_ensure = Mock()
        self.packaging.pip_uninstall = Mock()
        self.slicer.packaging = self.packaging
        modules = patch.dict(sys.modules, {"slicer": self.slicer, "slicer.packaging": self.packaging})
        modules.start()
        self.addCleanup(modules.stop)
        for name in list(sys.modules):
            if name in ("rustcpd", "tiny3d") or name.startswith(("rustcpd.", "tiny3d.")):
                sys.modules.pop(name)
        importer = patch.object(importlib, "import_module", side_effect=lambda name:
                                self.backend if name == "rustcpd" else types.SimpleNamespace())
        self.importer = importer.start()
        self.addCleanup(importer.stop)
        logger = patch.object(logging, "exception")
        logger.start()
        self.addCleanup(logger.stop)

    def test_installer_uses_exact_pin(self):
        self.assertTrue(self.ensure())
        self.packaging.pip_ensure.assert_called_once_with(
            ["tiny3d-rs>=2.1,<3", PIN], prompt_install=True, requester="Landmark Transfer")

    def test_matching_loaded_release_is_accepted(self):
        sys.modules["rustcpd"] = self.backend
        self.assertTrue(self.ensure())
        self.slicer.util.infoDisplay.assert_not_called()

    def test_different_loaded_releases_are_rejected_before_any_package_changes(self):
        for version in ("3.1.0", "4.0.1", "4.0.0rc1", "4.0.0+local", None):
            with self.subTest(version=version):
                sys.modules["rustcpd"] = types.SimpleNamespace(__version__=version)
                self.assertFalse(self.ensure())
        self.packaging.pip_check.assert_not_called()
        self.packaging.pip_ensure.assert_not_called()
        self.packaging.pip_uninstall.assert_not_called()
        self.assertIn("restart Slicer", self.slicer.util.infoDisplay.call_args.args[0])

    def test_missing_loaded_version_is_rejected(self):
        sys.modules["rustcpd"] = types.SimpleNamespace()
        self.assertFalse(self.ensure())
        self.packaging.pip_ensure.assert_not_called()

    def test_wrong_version_returned_by_import_is_rejected(self):
        self.backend.__version__ = "3.1.0"
        self.assertFalse(self.ensure())
        self.assertIn(PIN, self.slicer.util.errorDisplay.call_args.args[0])
        self.assertIn("restart Slicer", self.slicer.util.errorDisplay.call_args.args[0])

    def test_unknown_version_returned_by_import_is_rejected(self):
        del self.backend.__version__
        self.assertFalse(self.ensure())
        self.assertIn(PIN, self.slicer.util.errorDisplay.call_args.args[0])

    def test_declined_install_does_not_report_exception(self):
        self.packaging.pip_ensure.side_effect = RuntimeError("User declined package installation")
        self.assertFalse(self.ensure())
        self.slicer.util.errorDisplay.assert_not_called()

    def test_install_failure_can_be_retried(self):
        self.packaging.pip_ensure.side_effect = RuntimeError("network failure")
        self.assertFalse(self.ensure())
        self.packaging.pip_ensure.side_effect = None
        self.assertTrue(self.ensure())

    def test_loaded_legacy_tiny3d_still_requires_restart(self):
        self.packaging.pip_check.return_value = True
        sys.modules["tiny3d"] = types.SimpleNamespace()
        self.assertFalse(self.ensure())
        self.packaging.pip_uninstall.assert_not_called()
        self.packaging.pip_ensure.assert_not_called()

    def test_legacy_migration_confirmation_uses_exact_pin(self):
        self.packaging.pip_check.return_value = True
        self.assertTrue(self.ensure())
        self.assertIn(PIN, self.slicer.util.confirmOkCancelDisplay.call_args.args[0])
        self.packaging.pip_uninstall.assert_called_once_with(["tiny3d", "tiny3d-rs"])
        self.packaging.pip_ensure.assert_called_once_with(
            ["tiny3d-rs>=2.1,<3", PIN], prompt_install=False, requester="Landmark Transfer")

    def test_declined_legacy_migration_does_not_change_packages(self):
        self.packaging.pip_check.return_value = True
        self.slicer.util.confirmOkCancelDisplay.return_value = False
        self.assertFalse(self.ensure())
        self.packaging.pip_uninstall.assert_not_called()
        self.packaging.pip_ensure.assert_not_called()

    def test_package_dependency_declarations_have_one_pin(self):
        requirements = []
        for path in REPO.rglob("*"):
            if not path.is_file() or path.suffix not in (".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".cfg", ".ini"):
                continue
            if any(part.startswith(".") for part in path.relative_to(REPO).parts):
                continue
            for requirement in re.findall(r"rustcpd[=<>!~]+[0-9][0-9.,<>=!~]*", path.read_text(encoding="utf-8")):
                requirements.append((path.relative_to(REPO).as_posix(), requirement))
        self.assertTrue(requirements)
        for path, requirement in requirements:
            with self.subTest(path=path):
                self.assertEqual(requirement.rstrip(".,"), PIN)


if __name__ == "__main__":
    unittest.main()
