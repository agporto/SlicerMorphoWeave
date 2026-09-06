"""Run with ordinary Python; no Slicer or native registration backend required."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

MODULE_DIR = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "MorphoWeaveCompletionBatch", MODULE_DIR / "Resources/Python/MorphoWeaveCompletionBatch.py")
batch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = batch
SPEC.loader.exec_module(batch)


class BatchRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.outputs = self.root / "outputs"
        self.context = {"settings": {"seed": 17, "coverage": 0.75}, "model": "test-model"}
        self.calls = []
        self.logging = patch.object(batch.logging, "exception")
        self.logging.start()
        self.addCleanup(self.logging.stop)
        self.addCleanup(self.temporary.cleanup)

    def mesh(self, name="a.ply", content="mesh"):
        path = self.inputs / name
        path.write_text(content, encoding="utf-8")
        return path

    def processor(self, specimen, output):
        self.calls.append(specimen.name)
        (output / "shape.vtp").write_text("completed " + specimen.mesh.read_text())
        (output / "diagnostics.json").write_text(json.dumps(self.context))

    def run_batch(self, processor=None, **kwargs):
        return batch.run_batch(
            batch.discover_specimens(self.inputs), self.outputs,
            processor or self.processor, context=self.context, **kwargs)

    def summary(self):
        with open(self.outputs / "batch_summary.csv", newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_discovers_all_formats_sorted_and_ignores_directories(self):
        for name in ("b.STL", "A.ply", "c.VTP", "d.vtk", "e.obj", "ignore.txt"):
            self.mesh(name)
        (self.inputs / "directory.ply").mkdir()
        self.assertEqual([s.name for s in batch.discover_specimens(self.inputs)], ["A", "b", "c", "d", "e"])

    def test_pairs_case_insensitive_complete_markup_suffix(self):
        self.mesh("Bone.PLY")
        marks = self.root / "landmarks"
        marks.mkdir()
        (marks / "bONE.MRK.JSON").write_text("{}")
        specimen = batch.discover_specimens(self.inputs, marks)[0]
        self.assertEqual(specimen.name, "Bone")
        self.assertEqual(specimen.landmarks.name, "bONE.MRK.JSON")

    def test_rejects_duplicate_mesh_stems(self):
        self.mesh("Bone.ply")
        self.mesh("bone.stl")
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            batch.discover_specimens(self.inputs)

    def test_rejects_duplicate_landmark_stems(self):
        self.mesh()
        marks = self.root / "landmarks"
        marks.mkdir()
        (marks / "a.fcsv").write_text("x")
        (marks / "A.mrk.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            batch.discover_specimens(self.inputs, marks)

    def test_missing_and_empty_input_directories(self):
        with self.assertRaises(ValueError):
            batch.discover_specimens(self.root / "missing")
        with self.assertRaises(ValueError):
            batch.discover_specimens(self.inputs)

    def test_success_manifest_and_quoted_summary(self):
        self.mesh('bone, "one".ply')
        rows = self.run_batch()
        self.assertEqual(rows[0]["status"], "success")
        self.assertEqual(self.summary(), rows)
        manifest = json.loads((self.outputs / rows[0]["specimen"] / batch.MANIFEST_NAME).read_text())
        self.assertEqual(manifest["context"], self.context)
        self.assertEqual(len(manifest["artifacts"]), 2)
        self.assertEqual(manifest["status"], "success")
        self.assertFalse((self.outputs / ".morphoweave_batch.lock").exists())

    def test_failure_isolation_and_partial_output_cleanup(self):
        self.mesh("a.ply")
        self.mesh("b.ply")
        def process(specimen, output):
            self.processor(specimen, output)
            if specimen.name == "a":
                raise RuntimeError("deliberate fit failure")
        rows = self.run_batch(process)
        self.assertEqual([r["status"] for r in rows], ["failed", "success"])
        self.assertIn("deliberate fit failure", rows[0]["error"])
        self.assertFalse((self.outputs / "a").exists())
        self.assertEqual(list(self.outputs.glob(".*.tmp-*")), [])
        self.assertEqual(self.summary(), rows)

    def test_missing_landmarks_fail_one_specimen_without_unassisted_fallback(self):
        self.mesh("a.ply")
        self.mesh("b.ply")
        marks = self.root / "landmarks"
        marks.mkdir()
        (marks / "b.fcsv").write_text("landmarks")
        rows = batch.run_batch(batch.discover_specimens(self.inputs, marks), self.outputs,
                               self.processor, context=self.context, require_landmarks=True)
        self.assertEqual([r["status"] for r in rows], ["failed", "success"])
        self.assertEqual(self.calls, ["b"])

    def test_verified_resume_does_not_run_processor(self):
        self.mesh()
        self.run_batch()
        self.calls.clear()
        rows = self.run_batch()
        self.assertEqual(rows[0]["status"], "skipped")
        self.assertEqual(self.calls, [])

    def test_changed_settings_do_not_reuse_or_overwrite_results(self):
        self.mesh()
        self.run_batch()
        artifact = self.outputs / "a" / "shape.vtp"
        original = artifact.read_bytes()
        self.context["settings"]["coverage"] = 0.5
        rows = self.run_batch()
        self.assertEqual(rows[0]["status"], "failed")
        self.assertEqual(artifact.read_bytes(), original)

    def test_same_size_changed_mesh_is_not_skipped(self):
        path = self.mesh(content="aaaa")
        self.run_batch()
        path.write_text("bbbb")
        self.assertEqual(self.run_batch()[0]["status"], "failed")

    def test_changed_landmarks_are_not_skipped(self):
        self.mesh()
        marks = self.root / "marks"
        marks.mkdir()
        path = marks / "a.fcsv"
        path.write_text("111")
        specimens = batch.discover_specimens(self.inputs, marks)
        batch.run_batch(specimens, self.outputs, self.processor, context=self.context, require_landmarks=True)
        path.write_text("222")
        rows = batch.run_batch(specimens, self.outputs, self.processor, context=self.context, require_landmarks=True)
        self.assertEqual(rows[0]["status"], "failed")

    def test_missing_artifact_is_not_a_completed_specimen(self):
        self.mesh()
        self.run_batch()
        (self.outputs / "a" / "shape.vtp").unlink()
        self.assertEqual(self.run_batch()[0]["status"], "failed")

    def test_same_size_corrupt_output_is_not_skipped(self):
        self.mesh()
        self.run_batch()
        path = self.outputs / "a" / "shape.vtp"
        path.write_bytes(b"!" * path.stat().st_size)
        self.assertEqual(self.run_batch()[0]["status"], "failed")

    def test_cancel_before_any_specimen(self):
        self.mesh("a.ply")
        self.mesh("b.ply")
        rows = self.run_batch(should_cancel=lambda: True)
        self.assertEqual([r["status"] for r in rows], ["cancelled", "cancelled"])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.summary(), rows)

    def test_cancel_after_first_specimen_can_resume(self):
        self.mesh("a.ply")
        self.mesh("b.ply")
        cancel = []
        def progress(row, completed, total):
            if row["status"] == "success":
                cancel.append(True)
        rows = self.run_batch(should_cancel=lambda: bool(cancel), progress=progress)
        self.assertEqual([r["status"] for r in rows], ["success", "cancelled"])
        self.calls.clear()
        rows = self.run_batch()
        self.assertEqual([r["status"] for r in rows], ["skipped", "success"])
        self.assertEqual(self.calls, ["b"])

    def test_cancellation_delivered_by_running_progress_does_not_start_fit(self):
        self.mesh()
        cancel = []
        rows = self.run_batch(should_cancel=lambda: bool(cancel),
                              progress=lambda *args: cancel.append(True))
        self.assertEqual(rows[0]["status"], "cancelled")
        self.assertEqual(self.calls, [])

    def test_export_must_produce_files(self):
        self.mesh()
        rows = self.run_batch(lambda specimen, output: None)
        self.assertEqual(rows[0]["status"], "failed")
        self.assertIn("no output files", rows[0]["error"])

    def test_zero_length_output_is_failure(self):
        self.mesh()
        rows = self.run_batch(lambda specimen, output: (output / "empty.vtp").touch())
        self.assertEqual(rows[0]["status"], "failed")

    def test_changed_input_during_fit_is_failure(self):
        self.mesh()
        def process(specimen, output):
            self.processor(specimen, output)
            specimen.mesh.write_text("edited while fitting")
        self.assertEqual(self.run_batch(process)[0]["status"], "failed")
        self.assertFalse((self.outputs / "a").exists())

    def test_context_is_frozen_for_entire_run(self):
        self.mesh()
        def process(specimen, output):
            self.context["settings"]["seed"] = 99
            self.processor(specimen, output)
        self.run_batch(process)
        manifest = json.loads((self.outputs / "a" / batch.MANIFEST_NAME).read_text())
        self.assertEqual(manifest["context"]["settings"]["seed"], 17)

    def test_rejects_unsafe_manual_names_before_writing(self):
        path = self.mesh()
        for name in ("..", "../escape", "nested/path", "nested\\path"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                batch.run_batch([batch.BatchSpecimen(name, path)], self.outputs,
                                self.processor, context=self.context)
        self.assertFalse(self.outputs.exists())

    def test_rejects_input_directory_as_output(self):
        self.mesh()
        with self.assertRaisesRegex(ValueError, "must be different"):
            batch.run_batch(batch.discover_specimens(self.inputs), self.inputs,
                            self.processor, context=self.context)

    def test_lock_prevents_concurrent_writes(self):
        self.mesh()
        self.outputs.mkdir()
        lock = self.outputs / ".morphoweave_batch.lock"
        lock.write_text("existing process")
        with self.assertRaisesRegex(RuntimeError, "Another batch"):
            self.run_batch()
        self.assertEqual(lock.read_text(), "existing process")

    def test_lock_released_after_progress_exception(self):
        self.mesh()
        def progress(*args):
            raise RuntimeError("UI failed")
        with self.assertRaisesRegex(RuntimeError, "UI failed"):
            self.run_batch(progress=progress)
        self.assertFalse((self.outputs / ".morphoweave_batch.lock").exists())

    def test_unrelated_existing_output_directory_preserved(self):
        self.mesh()
        existing = self.outputs / "a"
        existing.mkdir(parents=True)
        (existing / "important.txt").write_text("preserve me")
        self.assertEqual(self.run_batch()[0]["status"], "failed")
        self.assertEqual((existing / "important.txt").read_text(), "preserve me")

    def test_export_symlink_is_rejected(self):
        source = self.mesh()
        def process(specimen, output):
            (output / "shape.vtp").symlink_to(source)
        self.assertEqual(self.run_batch(process)[0]["status"], "failed")
        self.assertEqual(source.read_text(), "mesh")

    def test_manifest_path_traversal_does_not_resume(self):
        self.mesh()
        self.run_batch()
        path = self.outputs / "a" / batch.MANIFEST_NAME
        manifest = json.loads(path.read_text())
        manifest["artifacts"][0]["path"] = "../../inputs/a.ply"
        path.write_text(json.dumps(manifest))
        self.assertEqual(self.run_batch()[0]["status"], "failed")

    def test_resume_false_does_not_overwrite(self):
        self.mesh()
        self.run_batch()
        self.assertEqual(self.run_batch(resume=False)[0]["status"], "failed")

    def test_processor_cannot_write_reserved_manifest(self):
        self.mesh()
        def process(specimen, output):
            (output / batch.MANIFEST_NAME).write_text("{}")
        self.assertEqual(self.run_batch(process)[0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
