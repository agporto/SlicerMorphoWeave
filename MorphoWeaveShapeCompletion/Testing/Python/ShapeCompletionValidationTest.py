"""Contract tests only; fake adapter tests are not scientific-accuracy evidence."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import ShapeCompletionValidation as validation


class ValidationScaffoldTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model, cases = validation.synthetic_cases()
        self.case = cases[1]
        np.savez(self.root / "model.npz", mean=self.model[0], modes=self.model[1], eigenvalues=self.model[2])
        self.write_specimen("a.npz", 0)
        self.write_specimen("b.npz", 1)
        self.manifest = {"version": 1, "units": "mm", "model_npz": "model.npz", "ssm_training_ids": ["train-001"],
                         "specimens": [self.item("heldout-001", "a.npz", "tuning"),
                                       self.item("heldout-002", "b.npz", "test")]}

    def write_specimen(self, name, offset):
        np.savez(self.root / name, points=self.case["points"] + offset,
                 complete_dense=self.case["complete_dense"] + offset,
                 observed_mask=self.case["observed_mask"])

    @staticmethod
    def item(identity, filename, split):
        return {"specimen_id": identity, "npz": filename, "split": split,
                "coverage": .33, "dense_correspondence_verified": True}

    def load(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(self.manifest))
        return validation.validate_manifest(path)

    def test_manifest_records_hashes_and_unverified_external_status(self):
        model, cases, settings, provenance = self.load()
        np.testing.assert_array_equal(model[0], self.model[0])
        self.assertEqual(cases[0]["external_status"], "unverified")
        self.assertEqual(settings["atlas_lambda"], 5.)
        self.assertEqual(len(provenance["manifest_sha256"]), 64)
        self.assertEqual(len(provenance["specimens"][0]["truth_sha256"]), 64)

    def test_rejects_training_overlap_and_duplicate_specimen_ids(self):
        self.manifest["specimens"][1]["specimen_id"] = "train-001"
        with self.assertRaisesRegex(ValueError, "leakage"):
            self.load()
        self.manifest["specimens"][1]["specimen_id"] = "heldout-001"
        with self.assertRaisesRegex(ValueError, "duplicate specimen"):
            self.load()

    def test_rejects_aliased_files_and_repacked_truth(self):
        self.manifest["specimens"][1]["npz"] = "a.npz"
        with self.assertRaisesRegex(ValueError, "Duplicate specimen file"):
            self.load()
        self.manifest["specimens"][1]["npz"] = "b.npz"
        np.savez_compressed(self.root / "b.npz", points=self.case["points"] + .1,
                            complete_dense=self.case["complete_dense"], observed_mask=self.case["observed_mask"])
        with self.assertRaisesRegex(ValueError, "complete truth content"):
            self.load()

    def test_rejects_missing_training_provenance_bad_split_and_unverified_correspondence(self):
        for key, value, message in (("split", "validation", "Split must"),
                                     ("dense_correspondence_verified", False, "correspondence"),
                                     ("external_status", "certified", "external_status")):
            with self.subTest(key=key):
                original = dict(self.manifest["specimens"][0])
                self.manifest["specimens"][0][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    self.load()
                self.manifest["specimens"][0] = original
        self.manifest["ssm_training_ids"] = []
        with self.assertRaisesRegex(ValueError, "training_ids"):
            self.load()

    def test_rejects_invalid_truth_dimensions_or_mask(self):
        for truth, mask in ((self.case["complete_dense"][:-1], self.case["observed_mask"]),
                            (self.case["complete_dense"], self.case["observed_mask"].astype(int))):
            np.savez(self.root / "a.npz", points=self.case["points"], complete_dense=truth, observed_mask=mask)
            with self.assertRaisesRegex(ValueError, "Invalid points"):
                self.load()

    def test_fragment_requires_withheld_truth_loci(self):
        np.savez(self.root / "a.npz", points=self.case["points"], complete_dense=self.case["complete_dense"],
                 observed_mask=np.ones(len(self.case["complete_dense"]), dtype=bool))
        with self.assertRaisesRegex(ValueError, "unobserved dense truth loci"):
            self.load()
        self.manifest["specimens"][0]["coverage"] = 1.0
        self.load()  # Complete cases have no missing-region error by definition.

    def test_frozen_synthetic_cases_and_positive_prior(self):
        model, cases = validation.synthetic_cases()
        np.testing.assert_array_equal(model[0], self.model[0])
        self.assertEqual([case["case_id"] for case in cases],
                         ["complete", "proximal_third", "middle_third", "distal_third", "proximal_coverage_misspecified"])
        self.assertTrue(all(len(case["points"]) >= 20 for case in cases))
        self.assertEqual(asdict(validation.fixed_settings())["seed"], 0)
        with self.assertRaises(ValueError):
            validation.fixed_settings(atlas_lambda=0.)

    def test_state_adapter_replaces_only_initialization_kwargs(self):
        state = SimpleNamespace(sigma2=.25)
        backend = SimpleNamespace(pose_initialize=mock.Mock(return_value=SimpleNamespace(state=state)),
                                  register_atlas=mock.Mock(return_value="fit"))
        initialization = dict(sigma2=9., initial_coefficients=[1], initial_rotation="rotation",
                              initial_scale=2., initial_translation=[0, 0, 0], lambda_regularization=5.)
        current = validation.HandoffAdapter(backend, False)
        current.pose_initialize("source")
        current.register_atlas("target", **initialization)
        self.assertEqual(backend.register_atlas.call_args.kwargs, initialization)
        experimental = validation.HandoffAdapter(backend, True)
        experimental.pose_initialize("source")
        experimental.register_atlas("target", **initialization)
        self.assertEqual(backend.register_atlas.call_args.kwargs,
                         {"initial_state": state, "lambda_regularization": 5.})
        self.assertEqual(experimental.initial_sigma2, .25)
        self.assertEqual(initialization["sigma2"], 9.)

    def test_native_absence_fails_instead_of_skipping_or_faking(self):
        with mock.patch.object(validation.importlib, "import_module", side_effect=ImportError):
            with self.assertRaisesRegex(RuntimeError, "no fake-backend accuracy"):
                validation.native_backend()

    def test_rejects_python_fake_backend(self):
        fake = SimpleNamespace(__version__="4.0.0", __file__="fake.py")
        with mock.patch.object(validation.importlib, "import_module", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "unmodified native"):
                validation.native_backend()

    def test_runtime_failure_is_recorded_not_mislabeled_returned(self):
        backend = SimpleNamespace(pose_initialize=mock.Mock(side_effect=RuntimeError("native failure")))
        row = validation.evaluate_case(self.model, self.case, validation.fixed_settings(), backend, False)
        self.assertEqual(row["status"], "failed")
        self.assertIn("native failure", row["error"])
        self.assertNotIn("rms", row)
        self.assertGreaterEqual(row["elapsed_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
