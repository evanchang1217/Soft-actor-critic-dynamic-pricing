"""Verify saved research with its hashed source snapshot, or replay diagnostics.

The current checkout can evolve without rewriting old provenance. Models and
result records remain read-only; reports and replay outputs go to a fresh path.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import zipfile

from .records import with_evidence
from rl.artifacts import new_run_directory, write_json


def extract_snapshot(record, destination):
    hashes = json.loads((record / "research-source-hashes.json").read_text())
    with zipfile.ZipFile(record / "research-source.zip") as archive:
        for name, digest in hashes.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                raise ValueError("Source snapshot contains an invalid path")
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Source snapshot hash mismatch: {name}")
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return hashes


@with_evidence(whole_results=True)
def reproduce(record, output, mode="verify", results_root=None):
    record = Path(record).resolve()
    repo = Path(__file__).resolve().parents[1]
    if mode not in ("verify", "replay"):
        raise ValueError("Choose verify or replay")
    verifier = next((record / p for p in ("source/verify.py", "source/verify_results.py")
                     if (record / p).is_file()), None)
    if mode == "verify" and verifier is None:
        raise ValueError("This record does not include a verification script")
    if mode == "replay" and not (record / "frozen-selection.json").is_file():
        raise ValueError("Snapshot replay needs a frozen decision record; use verify for other records")
    entrypoint = record / "replay-entrypoint.json"
    module = json.loads(entrypoint.read_text())["module"] if entrypoint.is_file() else "experiments.feedback_diagnostics"
    if module not in ("experiments.feedback_diagnostics", "experiments.robustness"):
        raise ValueError("Unsupported snapshot replay entry point")
    output = new_run_directory(Path(output).resolve())
    with tempfile.TemporaryDirectory(prefix="sac-research-source-") as folder:
        archived = Path(folder)
        hashes = extract_snapshot(record, archived)
        # The source runs in isolation; relative curated data/model paths retain
        # their repository layout. Archived evaluators only write to output.
        (archived / "results").symlink_to(results_root or repo / "results", target_is_directory=True)
        env = {**os.environ, "PYTHONOPTIMIZE": "0", "PYTHONPATH": str(archived), "MPLBACKEND": "Agg",
               "MPLCONFIGDIR": str(archived / ".matplotlib")}
        if mode == "verify":
            # Keep four parent levels for the diagnostic verifier's repo lookup.
            script = archived / "verification/a/b/c/verify.py"
            script.parent.mkdir(parents=True)
            shutil.copy2(verifier, script)
            driver = archived / "verify_driver.py"
            driver.write_text('''import pathlib, runpy, sys
from unittest.mock import patch
script, record, output = map(pathlib.Path, sys.argv[1:])
write = pathlib.Path.write_text
def capture(path, data, *args, **kwargs):
    if path.resolve() != record / "verification.json":
        raise RuntimeError("Verifier attempted an unexpected write")
    return write(output / "verification.json", data, *args, **kwargs)
sys.argv = [str(script), str(record)]
with patch.object(pathlib.Path, "write_text", capture):
    runpy.run_path(str(script), run_name="__main__")
''')
            command = [sys.executable, str(driver), str(script), str(record), str(output)]
        else:
            for name in ("config.json", "frozen-selection.json", "research-source-hashes.json"):
                shutil.copy2(record / name, output / name)
            if entrypoint.is_file():
                shutil.copy2(entrypoint,output / entrypoint.name)
            command = [sys.executable, "-m", module,
                       "--phase", "test", "--out", str(output)]
        subprocess.run(command, cwd=archived, env=env, check=True)
        current_matches = all((repo / name).is_file()
                              and hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest
                              for name, digest in hashes.items())
        write_json(output / "reproduction.json", {
            "record": str(record), "mode": mode, "source": "verified archived snapshot",
            "snapshot_sha256": hashlib.sha256((record / "research-source.zip").read_bytes()).hexdigest(),
            "current_checkout_matches_snapshot": current_matches,
            "original_record_modified": False,
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("verify", "replay"), default="verify")
    parser.add_argument('--archive')
    args = parser.parse_args()
    reproduce(args.record, args.out, args.mode, archive=args.archive)


if __name__ == "__main__":
    main()
