from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

from kwm_av.cli import fetch
from kwm_av.weights import inspect_bundle, read_manifest


class BundleTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.relpath = "transformer/model.safetensors"
        self.payload = b"weights"
        self.files = {
            self.relpath: {"bytes": len(self.payload), "sha256": hashlib.sha256(self.payload).hexdigest()},
        }
        self.asset = self.root / self.relpath
        self.asset.parent.mkdir()
        self.asset.write_bytes(self.payload)

    def manifest(self, *, model="base-av", files=None):
        (self.root / "bundle.json").write_text(json.dumps({
            "format_version": 1, "model": model, "files": self.files if files is None else files,
        }))

    def test_reads_and_verifies_bundle(self):
        self.manifest()
        bundle = inspect_bundle(self.root, verify=True)
        self.assertEqual(bundle.model, "base-av")
        self.assertEqual(set(bundle.files), {self.relpath})

    def test_rejects_same_size_corruption_with_hashes_enabled(self):
        self.manifest()
        self.asset.write_bytes(b"WEIGHTS")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            inspect_bundle(self.root, verify=True)

    def test_detects_missing_file_without_hashes(self):
        self.manifest()
        self.asset.unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            inspect_bundle(self.root)

    def test_rejects_unsafe_manifest_paths(self):
        entry = self.files[self.relpath]
        for path in ("/model.pt", "../model.pt", "dir/../model.pt", "dir//model.pt", "model?.pt"):
            with self.subTest(path=path):
                self.manifest(files={path: entry})
                with self.assertRaisesRegex(ValueError, "path"):
                    read_manifest(self.root)

    def test_rejects_reserved_manifest_path_as_payload(self):
        self.manifest(files={"bundle.json": self.files[self.relpath]})
        with self.assertRaisesRegex(ValueError, "path"):
            read_manifest(self.root)

    def test_rejects_wrong_model_bundle(self):
        self.manifest(model="ucpe")
        with self.assertRaisesRegex(ValueError, "Expected ar"):
            inspect_bundle(self.root, expected_model="ar")

    def test_example_bundle_names_follow_the_release_labels(self):
        for name in ("ar", "f2lf"):
            self.manifest(model=name)
            self.assertEqual(inspect_bundle(self.root, expected_model=name).model, name)

    def test_existing_autoregressive_manifest_is_still_readable(self):
        self.manifest(model="piflow")
        self.assertEqual(inspect_bundle(self.root, expected_model="ar").model, "ar")

    def test_fetch_uses_the_same_pinned_revision_for_manifest_and_payload(self):
        self.manifest()
        revision = "a" * 40
        downloads = []

        def record_download(**options):
            downloads.append(options)

        hub = SimpleNamespace(snapshot_download=record_download)
        with patch.dict(sys.modules, {"huggingface_hub": hub}), redirect_stdout(StringIO()):
            fetch("example/base", revision, self.root)

        expected = {"repo_id": "example/base", "revision": revision, "local_dir": self.root}
        self.assertEqual(downloads, [
            {**expected, "allow_patterns": ["bundle.json"]},
            {**expected, "allow_patterns": [self.relpath]},
        ])

    def test_fetch_rejects_a_mutable_revision(self):
        with self.assertRaisesRegex(ValueError, "40-character"):
            fetch("example/base", "main", self.root)

    def test_fetch_downloads_one_bundle_from_a_shared_repo(self):
        self.manifest(model="ar")
        (self.root / "ar").mkdir()
        (self.root / "bundle.json").rename(self.root / "ar/bundle.json")
        self.asset.parent.rename(self.root / "ar/transformer")
        downloads = []
        hub = SimpleNamespace(snapshot_download=lambda **options: downloads.append(options))

        with patch.dict(sys.modules, {"huggingface_hub": hub}), redirect_stdout(StringIO()):
            fetch("example/shared", "a" * 40, self.root, model="ar")

        common = {"repo_id": "example/shared", "revision": "a" * 40, "local_dir": self.root}
        self.assertEqual(downloads, [
            {**common, "allow_patterns": ["ar/bundle.json"]},
            {**common, "allow_patterns": [f"ar/{self.relpath}"]},
        ])


if __name__ == "__main__":
    unittest.main()
