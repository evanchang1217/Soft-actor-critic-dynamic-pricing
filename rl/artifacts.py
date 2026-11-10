"""Small, inspectable run artifacts and source/environment provenance."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def new_run_directory(path):
    path = Path(path)
    # Refuse accidental result overwrites, including reruns with a different config.
    path.mkdir(parents=True, exist_ok=False)
    return path


def manifest(kind):
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    source_hash = hashlib.sha256()
    for folder in ("rl", "experiments", "integrations", "myapp", "project"):
        for path in sorted((root / folder).rglob("*.py")):
            source_hash.update(path.relative_to(root).as_posix().encode())
            source_hash.update(path.read_bytes())
    packages = {}
    for package in ("torch", "numpy", "gymnasium", "matplotlib", "Django", "requests"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    status = git("status", "--porcelain")
    return {"kind": kind, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status),
            "python_source_sha256": source_hash.hexdigest()}
