#!/usr/bin/env python3
"""Apply git patches to the image's Python packages at runtime, inside each node's container.

RUNTIME_PATCHES names patch files under patches/, separated by spaces, applied in order. The
directory of a patch names the package it targets (patches/vllm/... patches vllm), and its paths
are that project's own repository paths (a/vllm/...), so `git format-patch` output from upstream
applies as is; files outside the package, such as tests, are skipped.

The container layer is writable and private to the job, so the image itself never changes. Each
patch is applied when it applies cleanly, skipped when the image already contains it (a later
release), and otherwise the node stops before vLLM starts. The status file records every patch
with its hash and result for the run's provenance.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from script_utils import atomic_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "patches"


class PatchError(RuntimeError):
    pass


# The directory holding a top-level package, found without importing it.
def site_root(package: str) -> Path:
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise PatchError(f"package {package!r} is not installed in this runtime")
    return Path(next(iter(spec.submodule_search_locations))).resolve().parent


def _git_apply(root: Path, patch: Path, package: str, *flags: str) -> subprocess.CompletedProcess:
    # Outside a repository git apply patches the working tree like patch(1); the ceiling keeps it
    # from finding a repository above the package and resolving paths against that instead.
    environment = {**os.environ, "GIT_CEILING_DIRECTORIES": str(root.parent)}
    return subprocess.run(
        ["git", "apply", "-p1", f"--include={package}/*", *flags, str(patch)],
        cwd=root, env=environment, capture_output=True, text=True, check=False,
    )


def _subject(patch: Path) -> str:
    lines = patch.read_text(errors="replace").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Subject:"):
            text = line.removeprefix("Subject:").strip()
            for more in lines[index + 1:]:
                if not more.startswith(" "):
                    break
                text += more
            return text.removeprefix("[PATCH]").strip()
    return ""


# Apply one patch, or confirm that the runtime already contains it.
def apply_patch(name: str, patch_dir: Path = PATCH_DIR, roots: dict[str, Path] | None = None) -> dict[str, Any]:
    patch = (patch_dir / name).resolve()
    if not patch.is_relative_to(patch_dir.resolve()) or not patch.is_file():
        raise PatchError(f"runtime patch {name!r} is not a file under {patch_dir}")
    package = Path(name).parts[0]
    root = (roots or {}).get(package) or site_root(package)
    record = {"name": name, "package": package, "subject": _subject(patch), "sha256": file_sha256(patch)}

    numstat = _git_apply(root, patch, package, "--numstat")
    files = [line.split("\t", 2)[2] for line in numstat.stdout.splitlines() if line.count("\t") >= 2]
    if not files:
        raise PatchError(f"{name}: no file of package {package!r} in the patch")
    record["files"] = files

    if _git_apply(root, patch, package, "--check").returncode == 0:
        result = _git_apply(root, patch, package)
        if result.returncode != 0:
            raise PatchError(f"{name}: git apply failed after a clean check:\n{result.stderr.strip()}")
        record["status"] = "applied"
    elif _git_apply(root, patch, package, "--check", "--reverse").returncode == 0:
        record["status"] = "already-present"
    else:
        detail = _git_apply(root, patch, package, "--check", "-v").stderr.strip()
        raise PatchError(
            f"{name} does not apply to {package} in this runtime and is not already in it:\n{detail}"
        )
    record["file_sha256"] = {path: file_sha256(root / path) for path in files}
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--status", type=Path, required=True, help="JSON status file to write")
    parser.add_argument("patches", nargs="*", help="patch files under patches/ (default: $RUNTIME_PATCHES)")
    args = parser.parse_args()
    names = args.patches or os.getenv("RUNTIME_PATCHES", "").split()
    status: dict[str, Any] = {"patches": []}
    if names and os.getenv("RUNTIME_BACKEND") == "uenv":
        # The patches target the container image's packages; a uenv's are read-only.
        status["skipped"] = "runtime is a uenv, not the container image"
        atomic_json(args.status, status)
        print(f"runtime patches not applied (uenv runtime): {' '.join(names)}", file=sys.stderr)
        return
    try:
        for name in names:
            record = apply_patch(name)
            status["patches"].append(record)
            print(f"runtime patch {record['status']}: {name} ({record['subject']})")
    except PatchError as error:
        status["error"] = str(error)
        atomic_json(args.status, status)
        print(f"ERROR: {error}", file=sys.stderr)
        print("Fix or drop the patch; RUNTIME_PATCHES= (empty) runs the image unpatched.", file=sys.stderr)
        sys.exit(2)
    atomic_json(args.status, status)


if __name__ == "__main__":
    main()
