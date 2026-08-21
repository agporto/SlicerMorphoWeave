import ast
import unittest
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[2]
MODULE = MODULE_DIR / "MorphoWeaveSurfaceSegmentation.py"
CMAKE = MODULE_DIR / "CMakeLists.txt"
README = MODULE_DIR / "README.md"


class SurfaceSegmentationSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MODULE.read_text(encoding="utf-8")
        ast.parse(cls.source)

    def test_hdmseg_replaces_the_legacy_numerical_core(self):
        self.assertIn('HDMSEG_REQUIREMENT = "hdmseg>=0.2,<0.3"', self.source)
        self.assertIn("slicer.packaging.pip_ensure(", self.source)
        self.assertIn('requester="Surface Segmentation"', self.source)
        self.assertIn("result = hdmseg.segment(", self.source)
        for legacy in (
            "scipy.sparse.linalg",
            "scipy.cluster.vq",
            "def _locus_geofeats",
            "def _spectral_cluster",
            "Feature vs Spatial weight",
            "alpha_grid",
        ):
            self.assertNotIn(legacy, self.source)

    def test_reference_and_model_selection_are_wired(self):
        self.assertIn("reference=np.ascontiguousarray(reference", self.source)
        self.assertIn('"stability"', self.source)
        self.assertIn('"modularity"', self.source)
        self.assertIn('"eigengap"', self.source)
        self.assertIn("atlas_dense_correspondences.mrk.json", self.source)
        self.assertIn("reference_graph_components", self.source)

    def test_outputs_and_mesh_projection_remain_available(self):
        for name in (
            "MorphoWeaveSurfaceSegmentation_locus_labels.npy",
            "MorphoWeaveSurfaceSegmentation_locus_labels.csv",
            "MorphoWeaveSurfaceSegmentation_embedding.npy",
            "MorphoWeaveSurfaceSegmentation_eigenvalues.csv",
            "MorphoWeaveSurfaceSegmentation_summary.json",
        ):
            self.assertIn(name, self.source)
        self.assertIn("SegID_smooth", self.source)
        self.assertIn("def _writeVtp", self.source)
        self.assertIn("def _writePlys", self.source)

    def test_module_documentation_and_source_test_are_registered(self):
        cmake = CMAKE.read_text(encoding="utf-8")
        self.assertIn("${MODULE_NAME}.py", cmake)
        self.assertIn("Testing/Python/SurfaceSegmentationSourceTest.py", cmake)
        self.assertTrue(README.is_file())


if __name__ == "__main__":
    unittest.main()
