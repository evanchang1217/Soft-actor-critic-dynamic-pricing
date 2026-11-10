"""Materialize byte-verified evidence without putting large artifacts in Git."""
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile

CATALOG = Path(__file__).resolve().parents[1] / 'results/archive-manifest.json'
LEGACY_SHARED = {'experiments/'+n+'.py' for n in (
    'feature_market', 'feature_screen', 'inventory_market', 'inventory_study',
    'advantage_market', 'advantage_study', 'advantage_anchor',
    'advantage_ledger_probe', 'stationary_market')}
CURRENT_SHARED = {'experiments/common.py', 'experiments/market/__init__.py',
    'experiments/market/economy.py', 'experiments/market/inventory.py',
    'experiments/market/controllers.py', 'experiments/market/continuing.py'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def valid_path(name):
    p = PurePosixPath(name)
    if (p.is_absolute() or '..' in p.parts or '\\' in name
            or len(p.parts) < 2 or p.parts[0] != 'results' or str(p) != name):
        raise ValueError(f'Invalid evidence path: {name}')
    return p


def load_catalog(path=None):
    path = Path(path or CATALOG)
    catalog = json.loads(path.read_text())
    if catalog.get('format') != 'sac-pricing-evidence-v1':
        raise ValueError('Unsupported evidence catalog')
    archive = catalog['archive']
    if Path(archive['filename']).name != archive['filename'] or not re.fullmatch('[a-f0-9]{64}', archive['sha256']):
        raise ValueError('Invalid archive metadata')
    for name, meta in catalog['files'].items():
        valid_path(name)
        if (type(meta['bytes']) is not int or meta['bytes'] < 0
                or not re.fullmatch('[a-f0-9]{64}', meta['sha256'])):
            raise ValueError('Invalid evidence size')
    for record in catalog['records']:
        valid_path(record)
    return catalog


def archive_path(catalog, explicit=None):
    supplied = explicit or os.environ.get('SAC_RESEARCH_ARCHIVE')
    path = (Path(supplied).expanduser() if supplied else
            CATALOG.parents[1].parent / 'research-archives' / catalog['archive']['filename'])
    if not path.is_file():
        raise FileNotFoundError(
            'Full evidence is archived. Supply --archive /path/to/'
            + catalog['archive']['filename']
            + ' or set SAC_RESEARCH_ARCHIVE. See docs/experiments.md#full-evidence-archive.')
    if path.stat().st_size != catalog['archive']['bytes'] or digest(path) != catalog['archive']['sha256']:
        raise ValueError('Evidence archive checksum mismatch')
    return path


def extract_evidence(catalog, archive, destination=None, prefix=None):
    """Reject unknown/link/traversal entries; stream only exact manifest sizes."""
    expected = {n for n in catalog['files'] if prefix is None or n.startswith(prefix.rstrip('/')+'/')}
    if not expected:
        raise ValueError('Record is absent from the evidence catalog')
    seen = set(); encountered = set()
    with tarfile.open(archive, 'r|gz') as bundle:
        for member in bundle:
            name = str(valid_path(member.name))
            if name not in catalog['files'] or not member.isfile() or name in encountered:
                raise ValueError(f'Unexpected evidence member: {name}')
            encountered.add(name)
            meta = catalog['files'][name]
            if member.size != meta['bytes']:
                raise ValueError(f'Evidence size mismatch: {name}')
            if name not in expected:
                continue
            seen.add(name); h = hashlib.sha256(); size = 0
            target = Path(destination)/name if destination is not None else None
            if target is not None:
                if not target.resolve().is_relative_to(Path(destination).resolve()):
                    raise ValueError('Evidence target escapes the output directory')
                target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source:
                sink = target.open('wb') if target is not None else None
                try:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        size += len(block); h.update(block)
                        if sink is not None:
                            sink.write(block)
                finally:
                    if sink is not None:
                        sink.close()
            if size != meta['bytes'] or h.hexdigest() != meta['sha256']:
                raise ValueError(f'Evidence file checksum mismatch: {name}')
    if seen != expected:
        raise ValueError('Evidence archive is missing declared files')
    return len(seen)


@dataclass(frozen=True)
class Evidence:
    record: Path
    results: Path
    materialized: bool = False


@contextmanager
def materialize_record(record, archive=None, *, whole_results=False):
    record = Path(record).resolve(); repo = CATALOG.parents[1]
    try:
        relative = record.relative_to(repo).as_posix()
    except ValueError:
        relative = ''
    catalog = load_catalog() if CATALOG.exists() and relative else None
    owned = ({n:m for n,m in catalog['files'].items() if n.startswith(relative+'/')}
             if catalog and relative in catalog['records'] else {})
    if not owned or all(m['kept_in_checkout'] for m in owned.values()):
        results = next((p for p in record.parents if p.name == 'results'), repo/'results')
        yield Evidence(record, results)
        return
    bundle = archive_path(catalog, archive)
    with tempfile.TemporaryDirectory(prefix='pricing-evidence-') as folder:
        root = Path(folder)
        extract_evidence(catalog, bundle, root, None if whole_results else relative)
        yield Evidence(root/relative, root/'results', True)


def with_evidence(function=None, *, whole_results=False):
    def decorate(func):
        @wraps(func)
        def wrapped(record, *args, archive=None, **kwargs):
            bound = inspect.signature(func).bind_partial(record, *args, **kwargs)
            with materialize_record(record, archive, whole_results=whole_results) as evidence:
                if evidence.materialized and any(bound.arguments.get(k) for k in ('save', 'save_verification')):
                    raise ValueError('Restore evidence to runs/ before saving a new verification; archived records stay frozen')
                if whole_results:
                    kwargs['results_root'] = evidence.results
                return func(evidence.record, *args, **kwargs)
        return wrapped
    return decorate(function) if function is not None else decorate


def validate_source_manifest(hashes, legacy):
    current = (set(legacy) - LEGACY_SHARED) | CURRENT_SHARED
    if set(hashes) not in (set(legacy), current, current - {'experiments/stationary_study.py'}):
        raise ValueError('Unexpected archived source manifest')


def restore(archive, output, *, catalog_path=None):
    catalog = load_catalog(catalog_path); bundle = archive_path(catalog, archive)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a fresh evidence output directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pricing-evidence-', dir=output.parent) as folder:
        staged = Path(folder)/'restored'; staged.mkdir()
        count = extract_evidence(catalog, bundle, staged)
        shutil.copyfile(catalog_path or CATALOG, staged/'results/archive-manifest.json')
        staged.rename(output)
    return count
