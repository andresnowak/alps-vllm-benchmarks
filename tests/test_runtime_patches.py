from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("runtime_patches", ROOT / "scripts" / "runtime-patches.py")
assert SPEC and SPEC.loader
runtime_patches = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_patches)

ORIGINAL = "def answer():\n    return 41\n"
PATCHED = "def answer():\n    return 42\n"
# A format-patch as upstream writes it: repository paths, plus a test file outside the package.
PATCH = """From 0000000000000000000000000000000000000000 Mon Sep 17 00:00:00 2001
From: Someone <someone@example.com>
Subject: [PATCH] Fix the answer
 (#1)

---
diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,2 +1,2 @@
 def answer():
-    return 41
+    return 42
diff --git a/tests/test_mod.py b/tests/test_mod.py
new file mode 100644
--- /dev/null
+++ b/tests/test_mod.py
@@ -0,0 +1 @@
+assert True
"""


class RuntimePatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        base = Path(self.temporary.name)
        self.site = base / "site"
        (self.site / "pkg").mkdir(parents=True)
        (self.site / "pkg" / "__init__.py").write_text("")
        (self.site / "pkg" / "mod.py").write_text(ORIGINAL)
        self.patches = base / "patches"
        (self.patches / "pkg").mkdir(parents=True)
        (self.patches / "pkg" / "0001-fix.patch").write_text(PATCH)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def apply(self) -> dict:
        return runtime_patches.apply_patch("pkg/0001-fix.patch", self.patches, {"pkg": self.site})

    def test_applies_package_files_and_skips_the_rest(self) -> None:
        record = self.apply()

        self.assertEqual(record["status"], "applied")
        self.assertEqual(record["subject"], "Fix the answer (#1)")
        self.assertEqual(record["files"], ["pkg/mod.py"])
        self.assertEqual((self.site / "pkg" / "mod.py").read_text(), PATCHED)
        self.assertFalse((self.site / "tests").exists())

    def test_a_patch_already_in_the_runtime_is_left_alone(self) -> None:
        (self.site / "pkg" / "mod.py").write_text(PATCHED)

        record = self.apply()

        self.assertEqual(record["status"], "already-present")
        self.assertEqual((self.site / "pkg" / "mod.py").read_text(), PATCHED)

    def test_a_conflicting_patch_stops_with_the_git_reason(self) -> None:
        (self.site / "pkg" / "mod.py").write_text("def answer():\n    return 7\n")

        with self.assertRaisesRegex(runtime_patches.PatchError, "does not apply to pkg"):
            self.apply()

    def test_names_outside_the_patch_directory_are_rejected(self) -> None:
        with self.assertRaisesRegex(runtime_patches.PatchError, "not a file under"):
            runtime_patches.apply_patch("../etc/passwd", self.patches, {"pkg": self.site})

    def test_finds_an_installed_package_without_importing_it(self) -> None:
        with patch.object(sys, "path", [str(self.site), *sys.path]):
            self.assertEqual(runtime_patches.site_root("pkg"), self.site.resolve())
        with self.assertRaisesRegex(runtime_patches.PatchError, "not installed"):
            runtime_patches.site_root("no_such_package_here")

    def test_uenv_runs_record_the_skip(self) -> None:
        status = Path(self.temporary.name) / "status.json"
        environment = {**os.environ, "RUNTIME_BACKEND": "uenv", "RUNTIME_PATCHES": "pkg/0001-fix.patch"}
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "runtime-patches.py"), "--status", str(status)],
            env=environment, capture_output=True, text=True, check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("uenv", json.loads(status.read_text())["skipped"])

    def test_the_shipped_vllm_patches_name_only_vllm_files(self) -> None:
        for path in sorted((ROOT / "patches" / "vllm").glob("*.patch")):
            files = [line.split()[2][2:] for line in path.read_text().splitlines() if line.startswith("diff --git ")]
            self.assertTrue(files, path.name)
            self.assertTrue(all(name.startswith("vllm/") for name in files), (path.name, files))


if __name__ == "__main__":
    unittest.main()
