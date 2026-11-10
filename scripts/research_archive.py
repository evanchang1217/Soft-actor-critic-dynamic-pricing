"""Pack, verify or restore the complete evidence collection outside the checkout."""
import argparse
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.records import CATALOG, digest, load_catalog, archive_path, extract_evidence, restore

REPO = CATALOG.parents[1]
BEHAVIOR = 'results/confirmation/2026-11-06-behavior-audit/'
# The public checkout keeps the central tables, protocols and demo inputs.
# Complete records, figures and frozen sources remain in the evidence archive.
CORE_RESULTS = {
    'confirmation/2026-11-05-robustness': {
        'protocol.json', 'config.json', 'frozen-selection.json', 'summary.csv',
        'differences.csv', 'generalization.png',
    },
    'confirmation/2026-11-06-behavior-audit': {
        'protocol.json', 'frozen-selection.json', 'validation.csv',
        'research-source-hashes.json', 'models.json', 'test-episodes.csv',
        'report/summary.csv', 'report/behavior-comparisons.png',
    },
    'confirmation/2026-11-07-mechanism-audit': {
        'protocol.json', 'report/summary.csv', 'report/differences.csv',
        'report/factorial.csv', 'report/mechanism-comparison.png',
    },
    'confirmation/2026-11-09-stationary-retraining': {
        'protocol.json', 'summary.csv', 'contrasts.csv', 'boundaries.csv',
        'retraining-boundaries.png',
    },
    'diagnostics/2026-11-05-sac-feedback': set(),
    'diagnostics/2026-11-09-adaptive-planning': {
        'protocol.json', 'summary.csv', 'contrasts.csv', 'adaptive-planning.png',
    },
    'diagnostics/2026-11-09-compute-tradeoff': {
        'protocol.json', 'summary.csv', 'contrasts.csv', 'common-costs.csv',
        'compute-tradeoff-zoom.png',
    },
    'diagnostics/2026-11-09-model-error-study': {
        'protocol.json', 'summary.csv', 'contrasts.csv', 'model-error.png',
    },
    'reevaluation/2026-10-28-review': set(),
    'reevaluation/2026-11-04-balanced-control': {
        'config.json', 'summary.csv', 'differences.csv', 'comparison.png',
    },
    'original': set(),
}


def kept(name):
    path = Path(name)
    for record, files in CORE_RESULTS.items():
        prefix = 'results/' + record + '/'
        if name.startswith(prefix):
            relative = name[len(prefix):]
            if prefix == BEHAVIOR and relative.startswith(('models/original/', 'models/inventory/')):
                return Path(relative).name == 'policy.pt'
            return relative in files
    # New or synthetic records keep small tables and presentation files by default.
    if path.suffix in ('.md', '.png', '.svg', '.pdf'):
        return True
    if any(part in path.parts for part in ('source', 'models', 'parent', 'training-replay', 'steps-parts', 'evaluation-cache')):
        return False
    if path.suffix == '.json':
        return True
    if path.suffix == '.csv' and (REPO/path).stat().st_size <= 250_000:
        return True
    return False


def write_catalog(catalog, path):
    """Keep the inventory readable without expanding each file across five lines."""
    sections = []
    for key, value in catalog.items():
        if key == 'files':
            entries = [f'    {json.dumps(name)}: {json.dumps(meta)}'
                       for name, meta in value.items()]
            rendered = '{\n' + ',\n'.join(entries) + '\n  }'
        else:
            rendered = json.dumps(value, indent=2).replace('\n', '\n  ')
        sections.append(f'  {json.dumps(key)}: {rendered}')
    Path(path).write_text('{\n' + ',\n'.join(sections) + '\n}\n')


def pack(output):
    output = Path(output).resolve()
    if output.is_relative_to(REPO) or output.exists():
        raise ValueError('Choose a new archive path outside the repository')
    if CATALOG.exists():
        previous = load_catalog()
        if any(not (REPO/name).is_file() for name in previous['files']):
            raise ValueError('Cannot repack a thin checkout; complete evidence must be restored first')
    files = {}; reconstructible = []
    for path in sorted((REPO/'results').rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc' or path == CATALOG:
            continue
        name = path.relative_to(REPO).as_posix()
        if path.name == 'steps.csv.gz' and (path.parent/'trace-parts.json').exists():
            trace = json.loads((path.parent/'trace-parts.json').read_text())
            if digest(path) != trace['sha256']:
                raise ValueError('Reconstructible full trace checksum mismatch')
            reconstructible.append(dict(path=name, manifest=str((path.parent/'trace-parts.json').relative_to(REPO)),
                                        sha256=trace['sha256'], bytes=trace['size']))
            continue
        files[name] = dict(bytes=path.stat().st_size, sha256=digest(path), kept_in_checkout=kept(name))
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open('xb') as stream:
            with tarfile.open(fileobj=stream, mode='w:gz', compresslevel=1) as bundle:
                for name in files:
                    path = REPO/name
                    if path.is_symlink():
                        raise ValueError('Evidence symlinks are not supported')
                    bundle.add(path, arcname=name, recursive=False)
        records = [str(p.relative_to(REPO)) for bucket in (REPO/'results').iterdir() if bucket.is_dir()
                   for p in ([bucket] if bucket.name == 'original' else bucket.iterdir()) if p.is_dir()]
        catalog = dict(format='sac-pricing-evidence-v1', created='2026-11-10',
            availability='Local archive prepared; no public download has been published.',
            archive=dict(filename=output.name, bytes=output.stat().st_size, sha256=digest(output)),
            records=sorted(records), files=files, reconstructible=reconstructible)
        extract_evidence(catalog, output)
        write_catalog(catalog, CATALOG)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    print(json.dumps(dict(archive=str(output), files=len(files),
        archive_bytes=catalog['archive']['bytes'], kept_bytes=sum(m['bytes'] for m in files.values() if m['kept_in_checkout'])), indent=2))


def prune(archive):
    catalog = load_catalog(); bundle = archive_path(catalog, archive)
    extract_evidence(catalog, bundle)
    # Check the whole deletion set before deleting anything.
    removing = {n:m for n,m in catalog['files'].items() if not m['kept_in_checkout']}
    for name, meta in removing.items():
        path = REPO/name
        if path.exists() and digest(path) != meta['sha256']:
            raise ValueError(f'Checkout changed since packing: {name}')
    for meta in catalog['reconstructible']:
        path = REPO/meta['path']
        if path.exists() and digest(path) != meta['sha256']:
            raise ValueError('Full trace changed since packing')
    for name in removing:
        (REPO/name).unlink(missing_ok=True)
    for meta in catalog['reconstructible']:
        path = REPO/meta['path']
        if path.exists():
            path.unlink()
    for path in sorted((REPO/'results').rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    print(f'Archived {len(removing)} evidence files; summaries and demo dependencies remain in the checkout')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['pack', 'verify', 'restore', 'prune'])
    p.add_argument('--archive'); p.add_argument('--out')
    a = p.parse_args()
    if a.action == 'pack':
        if not a.archive: p.error('pack requires --archive')
        pack(a.archive)
    elif a.action == 'prune':
        prune(a.archive)
    elif a.action == 'restore':
        if not a.out: p.error('restore requires --out')
        print(f'Restored {restore(a.archive, a.out)} byte-verified evidence files to {a.out}')
    else:
        c = load_catalog(); b = archive_path(c, a.archive)
        print(f'Verified {extract_evidence(c, b)} evidence files and the complete archive checksum')
