import ast
import re
import unittest
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[2]
MODULE = MODULE_DIR / "Resources/Python/MorphoWeaveShapeCompletionBase.py"
CORE = MODULE_DIR / "Resources" / "Python" / "MorphoWeaveShapeCompletionCore.py"


class ShapeCompletionModuleSourceTest(unittest.TestCase):
    def test_python_sources_parse(self):
        ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))
        ast.parse(CORE.read_text(encoding="utf-8"), filename=str(CORE))

    def test_pipeline_is_pose_then_pure_atlas_then_posterior(self):
        source = CORE.read_text(encoding="utf-8")
        pose = source.index("rustcpd_module.pose_initialize")
        atlas = source.index("rustcpd_module.register_atlas")
        posterior = source.index("fit.posterior")
        self.assertLess(pose, atlas)
        self.assertLess(atlas, posterior)
        self.assertNotIn("register_deformable", source)

    def test_ui_has_coverage_landmarks_calibration_and_scope_warning(self):
        source = MODULE.read_text(encoding="utf-8")
        for phrase in (
            "Target coverage:",
            "recommended below 0.75 target coverage",
            "Use matched landmarks in pose search",
            "Optional Calibration Profile",
            "Complete mesh directory:",
            "Paired dense truth directory (optional):",
            "conditional on the selected pose",
        ):
            self.assertIn(phrase, source)

    def test_new_native_api_is_required(self):
        source = MODULE.read_text(encoding="utf-8")
        self.assertIn('"refine_landmark_sigma"', source)
        self.assertIn('"landmark_sigma"', source)
        self.assertIn('"ShapePosterior"', source)
        self.assertIn('"AtlasResult"', source)
        self.assertIn('"posterior"', source)
        self.assertIn('"rustcpd==4.0.0"', source)
        for name in (
            '"translation_anchor_count"',
            '"scale_bounds"',
            '"adaptive_mixing"',
            '"initial_sigma2"',
            '"merge_tolerance"',
            '"translation_anchors_used"',
            '"winner_support"',
            '"mixing_weights"',
        ):
            self.assertIn(name, source)

    def test_fragments_default_to_pinned_scale_and_seeding_ui(self):
        source = MODULE.read_text(encoding="utf-8")
        # "Always free" as the default silently disables translation seeding.
        self.assertIn("self.scale_policy_combo.setCurrentIndex(0)", source)
        self.assertIn("Partial-Target Pose Seeding", source)
        self.assertIn("pose_seeding_warning", source)
        core = CORE.read_text(encoding="utf-8")
        self.assertIn("def effective_anchor_count", core)
        self.assertIn("def scale_bounds", core)


    def test_calibration_fragments_come_from_complete_mesh_surface(self):
        source = MODULE.read_text(encoding="utf-8")
        calibration = source.split("    def _run_calibration_impl", 1)[1].split(
            "    def _update_calibration_progress", 1
        )[0]
        self.assertIn("mask = contiguous_plane_mask(posed_surface", calibration)
        self.assertIn("fragment = posed_surface[mask.indices]", calibration)
        self.assertNotIn("fragment = posed_truth", calibration)
        self.assertNotIn("fragment = posed_full", calibration)
        self.assertIn("nearest_truth, errors_all = nearest_indices", calibration)
        self.assertIn("~mask.contains(posed_surface[nearest_truth])", calibration)

    def test_calibrated_uncertainty_names_make_scope_explicit(self):
        source = MODULE.read_text(encoding="utf-8")
        self.assertIn("CompletionCalibratedPointwiseRadius", source)
        self.assertIn("CompletionCalibratedSimultaneousRadius", source)
        self.assertIn("marginal Euclidean-error coverage", source)

    def test_calibration_requires_loaded_ssm_inputs(self):
        source = MODULE.read_text(encoding="utf-8")
        validation = source.split("    def _validate_calibration_inputs", 1)[1].split(
            "    def _parse_coverages", 1
        )[0]
        self.assertIn("self.ssm_table_selector.currentNode()", validation)
        self.assertIn("self.template_dense_selector.currentNode()", validation)
        self.assertIn("self.template_sparse_selector.currentNode()", validation)

    def test_ssm_template_identity_is_checked_beyond_point_count(self):
        source = MODULE.read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("indexed_correspondence_compatibility("), 2)
        core = CORE.read_text(encoding="utf-8")
        self.assertIn("index by index", core)

    def test_output_preserves_markup_metadata_and_exports_samples(self):
        source = MODULE.read_text(encoding="utf-8")
        self.assertIn("def copy_markups_metadata", source)
        self.assertIn("self.copy_markups_metadata(template_dense_node, dense_node)", source)
        self.assertIn("self.copy_markups_metadata(template_sparse_node, sparse_node)", source)
        self.assertIn('"latent_surface"', source)
        self.assertIn('"latent_dense"', source)
        self.assertIn('f"{base}_{sample_kind}_{index:02d}.vtp"', source)
        self.assertIn("latent coefficient posterior", source)

    def test_full_resolution_transfer_uses_exact_pose_and_cached_sparse_residual(self):
        source = MODULE.read_text(encoding="utf-8")
        output = source.split("    def create_completion_outputs", 1)[1].split(
            "    def point_cloud_model", 1
        )[0]
        self.assertIn("def _get_or_build_transfer_operator", source)
        self.assertIn("posed_vertices = apply_similarity(", output)
        self.assertIn("posed_dense = apply_similarity(", output)
        self.assertIn("dense_residual =", output)
        self.assertIn("full_fields = transfer.apply(dense_fields)", output)
        self.assertIn("transfer_operator_cache_hit", output)
        self.assertNotIn("local_displacement_interpolate(", output)
        self.assertNotIn("local_scalar_interpolate(", output)

    def test_large_sample_surfaces_are_opt_in(self):
        source = MODULE.read_text(encoding="utf-8")
        self.assertIn("Create full-resolution sample surfaces (memory intensive)", source)
        self.assertIn("self.full_resolution_samples_check.checked = False", source)
        self.assertIn("LatentShapeSampleDense_", source)
        self.assertIn("LatentShapeSampleSurface_", source)

    def test_transfer_operator_supports_components_anchors_and_chunks(self):
        source = CORE.read_text(encoding="utf-8")
        for phrase in (
            "class LocalTransferOperator",
            "def build_local_transfer_operator",
            "exact_source_vertex_indices",
            "query_groups",
            "source_groups",
            "chunk_size",
            "csr_matrix",
            "def pose_residual_interpolate",
        ):
            self.assertIn(phrase, source)

    def test_save_failures_are_checked(self):
        source = MODULE.read_text(encoding="utf-8")
        self.assertIn("def save_node_checked", source)
        save_method = source.split("    def save_completion_outputs", 1)[1].split(
            "    def list_model_files", 1
        )[0]
        self.assertIn("self.save_node_checked(", save_method)
        self.assertNotIn("slicer.util.saveNode(", save_method)

    def test_native_result_fields_used_by_diagnostics_are_preflighted(self):
        source = MODULE.read_text(encoding="utf-8")
        for phrase in (
            '"PoseInitialization"',
            '"score_margin"',
            '"landmark_rms"',
            '"coefficient_covariance"',
            '"noise_variance"',
            '"discrepancy_variance"',
        ):
            self.assertIn(phrase, source)

    def test_surface_normals_are_recomputed_and_input_target_is_not_restyled(self):
        source = MODULE.read_text(encoding="utf-8")
        self.assertIn("def recompute_polydata_normals", source)
        self.assertGreaterEqual(source.count("self.recompute_polydata_normals("), 1)
        output = source.split("    def create_completion_outputs", 1)[1].split(
            "    def point_cloud_model", 1
        )[0]
        self.assertIn("recompute_normals=True", output)
        self.assertIn("_FragmentReference", output)
        self.assertNotIn("target_node.GetDisplayNode().SetColor", output)

    def test_full_resolution_transfer_is_cached_pose_plus_residual(self):
        source = MODULE.read_text(encoding="utf-8")
        output = source.split("    def create_completion_outputs", 1)[1].split(
            "    def point_cloud_model", 1
        )[0]
        self.assertIn("self._get_or_build_transfer_operator", output)
        self.assertIn("posed_vertices = apply_similarity", output)
        self.assertIn("posed_dense = apply_similarity", output)
        self.assertIn("dense_residual", output)
        self.assertIn("transfer.apply(dense_fields)", output)
        self.assertIn("exact_similarity_plus_cached_sparse_local_residual", output)
        self.assertNotIn("local_displacement_interpolate(", output)

    def test_transfer_supports_exact_vertex_ids_components_and_chunking(self):
        source = MODULE.read_text(encoding="utf-8")
        core = CORE.read_text(encoding="utf-8")
        for phrase in (
            "MorphoWeave.TemplateVertexIndices",
            "polydata_point_component_ids",
            "transfer_operator_chunk_size",
            "transfer_component_fallback_vertices",
            "transfer_mean_support_radius",
        ):
            self.assertIn(phrase, source)
        self.assertIn("class LocalTransferOperator", core)
        self.assertIn("component_fallback_query_count", core)
        self.assertIn("for start in range(0, query_count, chunk_size)", core)

    def test_large_sample_surfaces_are_explicit_opt_in(self):
        source = MODULE.read_text(encoding="utf-8")
        self.assertIn("Create full-resolution sample surfaces (memory intensive)", source)
        self.assertIn("self.full_resolution_samples_check.checked = False", source)
        self.assertIn("dense_latent_point_cloud", source)
        self.assertIn("latent coefficient posterior", source)

    def test_ctk_double_spinbox_suffixes_use_pythonqt_properties(self):
        source = MODULE.read_text(encoding="utf-8")
        self.assertNotIn(".setSuffix(", source)
        self.assertIn('self.calibration_success_threshold.suffix = " %"', source)
        self.assertIn('self.landmark_sigma_percent.suffix = " %"', source)

    def test_enter_is_guarded_when_setup_did_not_finish(self):
        source = MODULE.read_text(encoding="utf-8")
        self.assertIn("self._ui_ready = False", source)
        setup = source.split("    def setup(self):", 1)[1].split(
            "    def _preflight_dependencies", 1
        )[0]
        self.assertIn("self._ui_ready = True", setup)
        enter = source.split("    def enter(self):", 1)[1].split(
            "    def cleanup", 1
        )[0]
        self.assertIn("if not self._ui_ready:", enter)

    def test_module_tutorial_is_packaged(self):
        tutorial = MODULE_DIR / "TUTORIAL.md"
        self.assertTrue(tutorial.is_file())
        text = tutorial.read_text(encoding="utf-8")
        self.assertIn("exact-pose-plus-residual", text)
        self.assertIn("1,000,000 vertices", text)
        self.assertIn("Calibration tutorial", text)

    def test_cmake_resources_are_module_local_and_exist(self):
        cmake = (MODULE_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
        match = re.search(
            r"set\(MODULE_PYTHON_RESOURCES(?P<body>.*?)\n\s*\)",
            cmake,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        resources = [
            line.strip()
            for line in match.group("body").splitlines()
            if line.strip()
        ]
        self.assertNotIn("install_into_checkout.py", resources)
        self.assertFalse((MODULE_DIR / "install_into_checkout.py").exists())
        self.assertNotIn("ShapeCompletionInstallerTest.py", cmake)
        for relative_path in resources:
            relative_path = relative_path.replace("${MODULE_NAME}", "MorphoWeaveShapeCompletion")
            self.assertTrue(
                (MODULE_DIR / relative_path).is_file(),
                f"CMake resource does not exist: {relative_path}",
            )

    def test_calibration_signature_includes_order_fallback_semantics(self):
        source = MODULE.read_text(encoding="utf-8")
        self.assertIn('values["allow_order_fallback"]', source)
        self.assertIn("current_settings=self._settings_snapshot(settings)", source)

    def test_auto_selection_does_not_replace_targets(self):
        source = MODULE.read_text(encoding="utf-8")
        resolver = source.split("    def _auto_select_canonical_ssm_set", 1)[1].split(
            "    def _validate_complete_inputs", 1
        )[0]
        self.assertNotIn("target_model_selector", resolver)
        self.assertNotIn("target_landmark_selector", resolver)


if __name__ == "__main__":
    unittest.main()
