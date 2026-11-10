"""Checks grouped by maintained responsibility."""
import hashlib
import json
from pathlib import Path
import zipfile
import pytest
from experiments.reproduce import extract_snapshot, reproduce
import io
import tarfile
from experiments import records
from scripts import research_archive


def snapshot(path, name="sentinel.txt", content=b"archived", digest=None):
    with zipfile.ZipFile(path / "research-source.zip", "w") as archive:
        archive.writestr(name, content)
    (path / "research-source-hashes.json").write_text(json.dumps({
        name: digest or hashlib.sha256(content).hexdigest()}))


@pytest.mark.parametrize("invalid", ["../escape.txt", "/tmp/escape.txt"])
def test_snapshot_rejects_paths_outside_the_source_directory(tmp_path, invalid):
    snapshot(tmp_path, invalid)
    with pytest.raises(ValueError, match="invalid path"):
        extract_snapshot(tmp_path, tmp_path / "source")


def test_snapshot_rejects_changed_source(tmp_path):
    snapshot(tmp_path, digest="0"*64)
    with pytest.raises(ValueError, match="hash mismatch"):
        extract_snapshot(tmp_path, tmp_path / "source")


def test_verifier_uses_archived_source_and_preserves_original_report(tmp_path):
    record = tmp_path / "record"
    record.mkdir()
    snapshot(record)
    (record / "verification.json").write_text('{"original": true}\n')
    source = record / "source"
    source.mkdir()
    (source / "verify.py").write_text('''from pathlib import Path
import json, sys
assert Path("sentinel.txt").read_text() == "archived"
report = {"archived_source_used": True}
(Path(sys.argv[1]) / "verification.json").write_text(json.dumps(report))
''')
    reproduce(record, tmp_path / "verified")
    assert json.loads((record / "verification.json").read_text()) == {"original": True}
    assert json.loads((tmp_path / "verified/verification.json").read_text()) == {"archived_source_used": True}
    metadata = json.loads((tmp_path / "verified/reproduction.json").read_text())
    assert metadata["current_checkout_matches_snapshot"] is False
    assert metadata["original_record_modified"] is False
    with pytest.raises(FileExistsError):
        reproduce(record, tmp_path / "verified")


def test_replay_dispatches_the_recorded_robustness_entrypoint(tmp_path):
    record = tmp_path/"record"
    record.mkdir()
    content = {"experiments/__init__.py":b"", "sentinel.txt":b"archived",
        "experiments/robustness.py":b'''import argparse
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--phase")
p.add_argument("--out")
a=p.parse_args()
assert a.phase=="test" and Path("sentinel.txt").read_text()=="archived"
(Path(a.out)/"replayed.txt").write_text("robustness")
'''}
    with zipfile.ZipFile(record/"research-source.zip","w") as archive:
        for name,data in content.items():archive.writestr(name,data)
    (record/"research-source-hashes.json").write_text(json.dumps({n:hashlib.sha256(d).hexdigest() for n,d in content.items()}))
    for name in ("config.json","frozen-selection.json"):(record/name).write_text('{}')
    (record/"replay-entrypoint.json").write_text(json.dumps({"module":"experiments.robustness"}))
    reproduce(record,tmp_path/"replayed","replay")
    assert (tmp_path/"replayed/replayed.txt").read_text()=="robustness"


def collection(tmp_path, monkeypatch):
    repo = tmp_path/'repo'; root = repo/'results/diagnostics/example'; root.mkdir(parents=True)
    (root/'README.md').write_text('Original report\n')
    (root/'summary.csv').write_text('reward\n-1\n')
    (root/'steps.csv.gz').write_bytes(b'large evidence stand-in')
    catalog = repo/'results/archive-manifest.json'; bundle = tmp_path/'evidence.tar.gz'
    monkeypatch.setattr(records, 'CATALOG', catalog)
    monkeypatch.setattr(research_archive, 'CATALOG', catalog)
    monkeypatch.setattr(research_archive, 'REPO', repo)
    research_archive.pack(bundle)
    return repo, root, bundle


def test_detailed_inputs_are_archived_but_summary_tables_remain():
    assert not research_archive.kept('results/confirmation/2026-11-05-robustness/summary.json')
    assert research_archive.kept('results/confirmation/2026-11-05-robustness/summary.csv')
    base = research_archive.BEHAVIOR + 'models/original/seed-0/'
    assert not research_archive.kept(base + 'training.csv')
    assert research_archive.kept(base + 'policy.pt')
    assert not research_archive.kept(base + 'outcome.json')
    assert research_archive.kept(research_archive.BEHAVIOR + 'validation.csv')
    assert not research_archive.kept('results/diagnostics/2026-11-09-model-error-study/calibration/replicate-0.csv')
    assert not research_archive.kept('results/diagnostics/2026-11-05-sac-feedback/local-gold-response.csv')
    assert research_archive.kept(research_archive.BEHAVIOR + 'models.json')
    assert not research_archive.kept('results/diagnostics/2026-11-09-compute-tradeoff/frontier.csv')


def test_pack_prune_and_materialize_preserve_original_evidence(tmp_path, monkeypatch):
    repo, root, bundle = collection(tmp_path, monkeypatch)
    original = (root/'steps.csv.gz').read_bytes()
    research_archive.prune(bundle)
    assert (root/'summary.csv').exists() and not (root/'steps.csv.gz').exists()
    with pytest.raises(ValueError, match='thin checkout'):
        research_archive.pack(tmp_path/'incomplete.tar.gz')
    # Checkout prose can evolve without rewriting a frozen report's provenance.
    (root/'README.md').write_text('Presentation-only revision\n')
    with records.materialize_record(root, bundle) as evidence:
        assert evidence.materialized and evidence.record != root
        assert (evidence.record/'steps.csv.gz').read_bytes() == original
        assert (evidence.record/'README.md').read_text() == 'Original report\n'
    assert (root/'README.md').read_text() == 'Presentation-only revision\n'
    out = tmp_path/'restored'; records.restore(bundle, out)
    assert (out/'results/diagnostics/example/steps.csv.gz').read_bytes() == original
    with pytest.raises(FileExistsError): records.restore(bundle, out)


def test_prune_refuses_changed_data_and_missing_archive_is_actionable(tmp_path, monkeypatch):
    repo, root, bundle = collection(tmp_path, monkeypatch)
    (root/'steps.csv.gz').write_bytes(b'new experiment data')
    with pytest.raises(ValueError, match='changed since packing'): research_archive.prune(bundle)
    assert (root/'steps.csv.gz').read_bytes() == b'new experiment data'
    monkeypatch.setenv('SAC_RESEARCH_ARCHIVE', str(tmp_path/'missing.tar.gz'))
    with pytest.raises(FileNotFoundError, match='--archive'):
        with records.materialize_record(root): pass


@pytest.mark.parametrize('kind', ['traversal', 'symlink', 'wrong-content', 'unknown', 'duplicate'])
def test_restore_rejects_invalid_archives_without_exposing_partial_output(tmp_path, monkeypatch, kind):
    repo, root, bundle = collection(tmp_path, monkeypatch)
    catalog = records.load_catalog(); name = 'results/diagnostics/example/steps.csv.gz'
    if kind == 'traversal': name = '../outside'
    if kind == 'unknown': name = 'results/unlisted.txt'
    content = b'X' * catalog['files']['results/diagnostics/example/steps.csv.gz']['bytes']
    with tarfile.open(bundle, 'w:gz') as tar:
        member = tarfile.TarInfo(name); member.size = len(content)
        if kind == 'symlink': member.type = tarfile.SYMTYPE; member.linkname = '/tmp/outside'; member.size = 0
        tar.addfile(member, io.BytesIO(content) if member.isfile() else None)
        if kind == 'duplicate': tar.addfile(member, io.BytesIO(content))
    catalog['archive'].update(bytes=bundle.stat().st_size, sha256=records.digest(bundle))
    records.CATALOG.write_text(json.dumps(catalog))
    out = tmp_path/'restored'
    with pytest.raises(ValueError): records.restore(bundle, out)
    assert not out.exists() and not (tmp_path/'outside').exists()


def test_source_manifest_supports_frozen_and_current_runtime_but_rejects_other_versions():
    legacy = records.LEGACY_SHARED | {'rl/sac.py', 'experiments/stationary_study.py'}
    records.validate_source_manifest(dict.fromkeys(legacy), legacy)
    records.validate_source_manifest(dict.fromkeys((legacy-records.LEGACY_SHARED)|records.CURRENT_SHARED), legacy)
    with pytest.raises(ValueError): records.validate_source_manifest({'unexpected.py':''}, legacy)


def test_retired_runner_executes_hash_checked_source_with_persistent_output(tmp_path):
    from scripts.reproduce import run_archived
    record=tmp_path/'record';record.mkdir()
    frozen=b'''import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('phase');p.add_argument('--out');a=p.parse_args()
assert a.phase=='init' and Path('sentinel.txt').read_text()=='frozen'
out=Path(a.out);out.mkdir();(out/'proof.txt').write_text('executed archived source')
'''
    files={'experiments/__init__.py':b'', 'experiments/behavior_audit.py':frozen,'sentinel.txt':b'frozen'}
    with zipfile.ZipFile(record/'research-source.zip','w') as bundle:
        for name,content in files.items():bundle.writestr(name,content)
    (record/'research-source-hashes.json').write_text(json.dumps({n:hashlib.sha256(c).hexdigest() for n,c in files.items()}))
    output=tmp_path/'new-study'
    run_archived(record,'behavior',['--','init','--out',str(output)])
    assert (output/'proof.txt').read_text()=='executed archived source'


def test_retired_runner_rejects_temporary_output_paths(tmp_path):
    from scripts.reproduce import run_archived
    with pytest.raises(ValueError,match='absolute --out'):
        run_archived(tmp_path,'behavior',['init','--out','relative-output'])
