"""One entry point for byte-verified historical study reproduction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from uuid import uuid4

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.records import with_evidence,validate_source_manifest,LEGACY_SHARED
from experiments.common import require
from experiments.reproduce import extract_snapshot,reproduce as reproduce_main
from rl.artifacts import new_run_directory,write_json

STUDIES = {
 'objective':'reevaluation/2026-11-04-balanced-control',
 'main':'confirmation/2026-11-05-robustness',
 'behavior':'confirmation/2026-11-06-behavior-audit',
 'mechanism':'confirmation/2026-11-07-mechanism-audit',
 'stationary':'confirmation/2026-11-09-stationary-retraining',
 'model-error':'diagnostics/2026-11-09-model-error-study',
 'adaptive':'diagnostics/2026-11-09-adaptive-planning',
 'compute':'diagnostics/2026-11-09-compute-tradeoff',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def hydrate_trace(root):
    root=Path(root)
    if not (root/'trace-parts.json').exists():return
    p=json.loads((root/'trace-parts.json').read_text())
    require(p['format']=='lossless-compressed-byte-parts-v1' and p['file']=='steps.csv.gz',
            'Reconstructed trace metadata or checksum mismatch')
    if (root/'steps.csv.gz').exists():
        require(hashlib.sha256((root/'steps.csv.gz').read_bytes()).hexdigest()==p['sha256'],
                'Saved evidence hash mismatch')
        return
    full=hashlib.sha256();size=0
    with (root/'steps.csv.gz').open('wb') as stream:
        for part in p['parts']:
            if not re.fullmatch(r'steps-parts/part-[0-9]{5}\.bin',part['path']):raise ValueError('Unexpected trace part path')
            block=(root/part['path']).read_bytes()
            if len(block)!=part['size'] or hashlib.sha256(block).hexdigest()!=part['sha256']:raise ValueError('Trace part mismatch')
            stream.write(block);full.update(block);size+=len(block)
    require(full.hexdigest()==p['sha256'] and size==p['size'], 'Saved evidence hash mismatch')

@with_evidence
def reproduce_stationary(record):
    record=Path(record).resolve();hashes=json.loads((record/'source-hashes.json').read_text())
    expected={'experiments/advantage_market.py','experiments/advantage_study.py','experiments/inventory_market.py','experiments/inventory_study.py',
        'experiments/feature_market.py','experiments/feature_screen.py','rl/sac.py','rl/config.py','rl/environment.py','rl/rewards.py','rl/artifacts.py',
        'experiments/advantage_anchor.py','experiments/advantage_ledger_probe.py','experiments/stationary_market.py','experiments/market_planner.py','experiments/stationary_study.py'}
    validate_source_manifest(hashes, expected)
    with tempfile.TemporaryDirectory(prefix='stationary-retraining-archive-') as temporary:
        root=Path(temporary);code=root/'code'
        for name,digest in hashes.items():
            source=record/'source'/name
            if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:raise ValueError(f'Archived source mismatch: {name}')
            target=code/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        for package in ('experiments','rl'):(code/package/'__init__.py').touch()
        shutil.copytree(record,root/'record',ignore=lambda path,names:['steps.csv.gz'] if Path(path)==record and (record/'trace-parts.json').exists() else [])
        hydrate_trace(root/'record')
        subprocess.run([sys.executable,'-m','experiments.stationary_study','--phase','verify','--out',str(root/'record')],cwd=code,env=dict(os.environ, PYTHONOPTIMIZE="0",PYTHONPATH=str(code)),check=True)
        if (record/'report-provenance.json').exists():
            provenance=json.loads((record/'report-provenance.json').read_text())
            if hashlib.sha256((record/'report-source.py').read_bytes()).hexdigest()!=provenance['source_sha256']:raise ValueError('Report source mismatch')
            shutil.copyfile(record/'report-source.py',code/'experiments/stationary_report.py')
            subprocess.run([sys.executable,'-m','experiments.stationary_report','--phase','verify','--out',str(root/'record')],cwd=code,env=dict(os.environ, PYTHONOPTIMIZE="0",PYTHONPATH=str(code)),check=True)
        supplement=record/'planning-extra'
        if (supplement/'hashes.json').exists():
            plan=json.loads((supplement/'plan.json').read_text())
            if hashlib.sha256((supplement/'source.py').read_bytes()).hexdigest()!=plan['source_sha256']:raise ValueError('Supplement source mismatch')
            shutil.copyfile(supplement/'source.py',code/'experiments/stationary_planning_extra.py')
            subprocess.run([sys.executable,'-m','experiments.stationary_planning_extra','--phase','verify','--parent',str(root/'record')],cwd=code,env=dict(os.environ, PYTHONOPTIMIZE="0",PYTHONPATH=str(code)),check=True)
        if (record/'timing-provenance.json').exists():
            p=json.loads((record/'timing-provenance.json').read_text())
            for name,key in [('timing-source.py','source_sha256'),('decision-states.json','input_states_sha256'),('models.json','models_sha256'),('selection.json','selection_sha256'),('decision-cpu-time.csv','output_sha256')]:
                if hashlib.sha256((record/name).read_bytes()).hexdigest()!=p[key]:raise ValueError('Timing provenance mismatch')
            print('Timing provenance hashes checked; machine timings are not rerun',flush=True)

@with_evidence
def reproduce_model_error(record, save_verification=False):
    record = Path(record).resolve()
    hashes = json.loads((record/'source-hashes.json').read_text())
    expected = {
        'experiments/inventory_market.py', 'experiments/inventory_study.py',
        'experiments/feature_market.py', 'experiments/feature_screen.py',
        'experiments/advantage_market.py', 'experiments/advantage_study.py',
        'experiments/advantage_anchor.py', 'experiments/advantage_ledger_probe.py',
        'experiments/stationary_market.py', 'experiments/stationary_study.py',
        'experiments/market_planner.py', 'experiments/model_error_study.py',
        'rl/sac.py', 'rl/config.py', 'rl/environment.py', 'rl/rewards.py', 'rl/artifacts.py',
    }
    validate_source_manifest(hashes, expected)
    with tempfile.TemporaryDirectory(prefix='model-error-archive-') as temporary:
        root = Path(temporary); code = root/'code'
        for name, digest in hashes.items():
            source = record/'source'/name
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError('Archived source mismatch: ' + name)
            target = code/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for package in ('experiments', 'rl'): (code/package/'__init__.py').touch()
        shutil.copytree(record, root/'record')
        subprocess.run([sys.executable, '-m', 'experiments.model_error_study', '--phase', 'verify',
                        '--out', str(root/'record')], cwd=code,
                       env=dict(os.environ, PYTHONOPTIMIZE="0", PYTHONPATH=str(code)+os.pathsep+os.environ.get('PYTHONPATH', '')), check=True)
        prediction = root/'record'/'prediction-audit'
        if prediction.exists():
            saved = json.loads((prediction/'hashes.json').read_text())
            for name, digest in saved.items():
                if hashlib.sha256((prediction/name).read_bytes()).hexdigest() != digest:
                    raise ValueError('Prediction audit hash mismatch: ' + name)
            shutil.copyfile(prediction/'source.py', code/'experiments/model_error_prediction.py')
            # Regenerate only inside this disposable copy; retain the original record.
            shutil.rmtree(prediction)
            subprocess.run([sys.executable, '-m', 'experiments.model_error_prediction', '--record', str(root/'record')],
                           cwd=code, env=dict(os.environ, PYTHONOPTIMIZE="0", PYTHONPATH=str(code)+os.pathsep+os.environ.get('PYTHONPATH', '')), check=True)
            for name, digest in saved.items():
                if hashlib.sha256((prediction/name).read_bytes()).hexdigest() != digest:
                    raise ValueError('Prediction audit regeneration mismatch: ' + name)
            print('Prediction audit logs and all 60 result rows regenerated', flush=True)
        if (record/'render-provenance.json').exists():
            render = json.loads((record/'render-provenance.json').read_text())
            for name, digest in {**render['inputs'], **render['outputs']}.items():
                if hashlib.sha256((record/name).read_bytes()).hexdigest() != digest:
                    raise ValueError('Rendered report hash mismatch: ' + name)
            (code/'scripts').mkdir(); (code/'scripts/__init__.py').touch()
            shutil.copyfile(record/'report-render-source.py', code/'scripts/report_model_error_study.py')
            helper = 'experiments.common' if 'experiments/common.py' in hashes else 'experiments.feature_screen'
            snippet = ('from pathlib import Path; '
                f'from {helper} import csv_read,compare_tables,aggregate; '
                'from scripts.report_model_error_study import planner_gaps; '
                'r=Path(__import__("sys").argv[1]); '
                'compare_tables(csv_read(r/"planner-model-gap.csv"),planner_gaps(r)); '
                'compare_tables(csv_read(r/"per-model.csv"),aggregate(csv_read(r/"episodes.csv"),'
                '["case","method","size","replicate","model_seed","window"]))')
            subprocess.run([sys.executable, '-c', snippet, str(root/'record')], cwd=code,
                           env=dict(os.environ, PYTHONOPTIMIZE="0", PYTHONPATH=str(code)+os.pathsep+os.environ.get('PYTHONPATH', '')), check=True)
            print('Rendered hashes checked; per-model means and planner-model gaps regenerated', flush=True)
        result = json.loads((root/'record'/'verification.json').read_text())
        if save_verification:
            shutil.copyfile(root/'record'/'verification.json', record/'verification.json')
            shutil.copyfile(__file__, record/'reproduce-source.py')
            archive = dict(execution='exact archived source in disposable copy', result=result,
                source_manifest_sha256=hashlib.sha256((record/'source-hashes.json').read_bytes()).hexdigest(),
                data_manifest_sha256=hashlib.sha256((record/'data-hashes.json').read_bytes()).hexdigest(),
                wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                prediction_audit_regenerated=(record/'prediction-audit').exists(),
                rendered_hashes_checked=(record/'render-provenance.json').exists())
            (record/'archive-verification.json').write_text(json.dumps(archive, indent=2)+'\n')
        return result

@with_evidence
def reproduce_adaptive(record, workers=1, save=False):
    record = Path(record).resolve(); hashes = json.loads((record/'source-hashes.json').read_text())
    expected = {'experiments/inventory_market.py', 'experiments/inventory_study.py',
        'experiments/feature_market.py', 'experiments/feature_screen.py', 'experiments/advantage_market.py',
        'experiments/advantage_study.py', 'experiments/advantage_anchor.py', 'experiments/advantage_ledger_probe.py',
        'experiments/stationary_market.py', 'experiments/stationary_study.py', 'experiments/market_planner.py',
        'experiments/model_error_study.py', 'experiments/adaptive_planning.py',
        'rl/sac.py', 'rl/config.py', 'rl/environment.py', 'rl/rewards.py', 'rl/artifacts.py'}
    validate_source_manifest(hashes, expected)
    with tempfile.TemporaryDirectory(prefix='adaptive-planning-archive-') as temporary:
        root = Path(temporary); code = root/'code'
        for name, digest in hashes.items():
            source = record/'source'/name
            require(hashlib.sha256(source.read_bytes()).hexdigest() == digest,
                    'Saved evidence hash mismatch')
            target = code/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
        for package in ('experiments', 'rl'): (code/package/'__init__.py').touch()
        shutil.copytree(record, root/'record')
        subprocess.run([sys.executable, '-m', 'experiments.adaptive_planning', '--phase', 'verify', '--workers', str(workers),
            '--out', str(root/'record')], cwd=code,
            env=dict(os.environ, PYTHONOPTIMIZE="0", PYTHONPATH=str(code)+os.pathsep+os.environ.get('PYTHONPATH', '')), check=True)
        if (record/'render-provenance.json').exists():
            provenance = json.loads((record/'render-provenance.json').read_text())
            for name, digest in {**provenance['inputs'], **provenance['outputs']}.items():
                require(hashlib.sha256((record/name).read_bytes()).hexdigest() == digest,
                        'Saved evidence hash mismatch')
            (code/'scripts').mkdir(); (code/'scripts/__init__.py').touch()
            shutil.copyfile(record/'render-source.py', code/'scripts/report_adaptive_planning.py')
            helper = 'experiments.common' if 'experiments/common.py' in hashes else 'experiments.feature_screen'
            snippet = ('from pathlib import Path; from scripts.report_adaptive_planning import auxiliary_tables; '
                f'from {helper} import csv_read,compare_tables; '
                'r=Path(__import__("sys").argv[1]); h,c=auxiliary_tables(r); '
                'compare_tables(csv_read(r/"estimate-quality.csv"),h); '
                'compare_tables(csv_read(r/"cpu-summary.csv"),c)')
            subprocess.run([sys.executable, '-c', snippet, str(root/'record')], cwd=code,
                env=dict(os.environ, PYTHONOPTIMIZE="0", PYTHONPATH=str(code)+os.pathsep+os.environ.get('PYTHONPATH', '')), check=True)
            print('Rendered hashes checked; parameter/timing summaries reconstructed (timings not rerun)', flush=True)
        result = json.loads((root/'record/verification.json').read_text())
        output = dict(result=result, execution='exact archived source and copied parent models',
            source_manifest_sha256=hashlib.sha256((record/'source-hashes.json').read_bytes()).hexdigest(),
            data_manifest_sha256=hashlib.sha256((record/'data-hashes.json').read_bytes()).hexdigest(),
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if save:
            shutil.copyfile(root/'record/verification.json', record/'verification.json')
            shutil.copyfile(__file__, record/'reproduce-source.py')
            (record/'archive-verification.json').write_text(json.dumps(output, indent=2)+'\n')
        return output

@with_evidence
def reproduce_compute(record,workers=1,save=False):
    record = Path(record).resolve(); hashes = json.loads((record/'source-hashes.json').read_text())
    expected = {'experiments/inventory_market.py','experiments/inventory_study.py','experiments/feature_market.py',
        'experiments/feature_screen.py','experiments/advantage_market.py','experiments/advantage_study.py',
        'experiments/advantage_anchor.py','experiments/advantage_ledger_probe.py','experiments/stationary_market.py',
        'experiments/stationary_study.py','experiments/market_planner.py','experiments/model_error_study.py',
        'experiments/adaptive_planning.py','experiments/compute_tradeoff.py','rl/sac.py','rl/config.py',
        'rl/environment.py','rl/rewards.py','rl/artifacts.py'}
    validate_source_manifest(hashes, expected)
    with tempfile.TemporaryDirectory(prefix='compute-tradeoff-archive-') as temporary:
        root = Path(temporary); code = root/'code'
        for name,digest in hashes.items():
            source = record/'source'/name
            require(hashlib.sha256(source.read_bytes()).hexdigest() == digest,
                    'Saved evidence hash mismatch')
            target = code/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,target)
        for package in ('experiments','rl'): (code/package/'__init__.py').touch()
        shutil.copytree(record,root/'record')
        env = dict(os.environ, PYTHONOPTIMIZE="0",PYTHONPATH=str(code)+os.pathsep+os.environ.get('PYTHONPATH',''))
        subprocess.run([sys.executable,'-m','experiments.compute_tradeoff','--phase','verify','--workers',str(workers),
            '--out',str(root/'record')],cwd=code,env=env,check=True)
        if (record/'render-provenance.json').exists():
            p = json.loads((record/'render-provenance.json').read_text())
            for name,digest in {**p['inputs'],**p['outputs']}.items():
                require(hashlib.sha256((record/name).read_bytes()).hexdigest() == digest,
                        'Saved evidence hash mismatch')
            print('Report input/source/output hashes checked; CPU times are not rerun',flush=True)
        result = json.loads((root/'record/verification.json').read_text())
        output = dict(result=result,execution='exact archived sources and copied original/replayed model artifacts',
            source_manifest_sha256=hashlib.sha256((record/'source-hashes.json').read_bytes()).hexdigest(),
            data_manifest_sha256=hashlib.sha256((record/'data-hashes.json').read_bytes()).hexdigest(),
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            timing_remeasurement=False,training_rerun=False)
        if save:
            shutil.copyfile(root/'record/verification.json',record/'verification.json')
            shutil.copyfile(__file__,record/'reproduce-source.py')
            (record/'archive-verification.json').write_text(json.dumps(output,indent=2)+'\n')
        return output

@with_evidence
def reproduce_audit(record, output, kind):
    record=Path(record).resolve();output=new_run_directory(Path(output).resolve())
    meta=json.loads((record/'report/provenance.json').read_text());report=record/'report/report-source.py'
    if sha(report)!=meta['script_sha256']:raise ValueError('Archived report source mismatch')
    for name,digest in meta['outputs'].items():
        if Path(name).name!=name or sha(record/'report'/name)!=digest:raise ValueError('Archived report artifact mismatch')
    files=('test-episodes.csv','players.csv.gz','traces.csv.gz') if kind=='behavior' else ('test-episodes.csv','players.json.gz','traces.json.gz')
    before={n:sha(record/n) for n in files}
    with tempfile.TemporaryDirectory(prefix='audit-reproduction-') as folder:
        source=Path(folder);extract_snapshot(record,source)
        shutil.copyfile(report,source/f'experiments/{kind}_report.py')
        staged=source/'record';staged.mkdir()
        for path in record.iterdir():
            if path.name not in ('verification.json','test-jobs'):(staged/path.name).symlink_to(path,target_is_directory=path.is_dir())
        env=dict(os.environ, PYTHONOPTIMIZE="0",PYTHONPATH=str(source),MPLCONFIGDIR=str(source/'.matplotlib'))
        subprocess.run([sys.executable,'-m',f'experiments.{kind}_audit','verify','--out',str(staged)],cwd=source,env=env,check=True)
        subprocess.run([sys.executable,'-m',f'experiments.{kind}_report',str(staged),'--check'],cwd=source,env=env,check=True)
        shutil.copyfile(staged/'verification.json',output/'verification.json')
    if before!={n:sha(record/n) for n in files}:raise RuntimeError('Evidence was modified')
    write_json(output/'reproduction.json',dict(record=str(record),source='hash-checked frozen study/report',original_data_unchanged=True))
    return json.loads((output/'verification.json').read_text())


def reproduce_study(study, record=None, output=None, workers=1, archive=None, mode='verify'):
    record=record or Path('results')/STUDIES[study]
    output=Path(output or f'runs/reproduction-{study}-{uuid4().hex[:8]}')
    if study in ('main','objective'):
        reproduce_main(record,output,mode,archive=archive)
        return json.loads((output/'verification.json').read_text()) if mode=='verify' else {'full_replay_output':str(output)}
    if mode!='verify':raise ValueError('Full replay mode applies to the original main study')
    if study in ('behavior','mechanism'):return reproduce_audit(record,output,study,archive=archive)
    if study=='stationary':return reproduce_stationary(record,archive=archive)
    if study=='model-error':return reproduce_model_error(record,archive=archive)
    fn={'adaptive':reproduce_adaptive,'compute':reproduce_compute}[study]
    return fn(record,workers=workers,archive=archive)['result']


@with_evidence(whole_results=True)
def run_archived(record, study, arguments, results_root=None):
    if study not in ('behavior','mechanism','stationary'):raise ValueError('Use current runners for maintained studies')
    values=list(arguments)
    if values and values[0]=='--':values.pop(0)
    output=next((values[i+1] for i,x in enumerate(values[:-1]) if x=='--out'),None)
    if output is None or not Path(output).is_absolute():raise ValueError('Archived runs require an absolute --out path outside the temporary source tree')
    requested=Path(output).resolve();repo=Path(__file__).resolve().parents[1]
    if requested.is_relative_to(repo/'results'):raise ValueError('Write archived runs under runs/, not curated results/')
    with tempfile.TemporaryDirectory(prefix='frozen-study-') as folder:
        code=Path(folder);record=Path(record)
        if (record/'research-source.zip').exists():extract_snapshot(record,code)
        else:
            hashes=json.loads((record/'source-hashes.json').read_text())
            expected=LEGACY_SHARED|{f'rl/{n}.py' for n in ('sac','config','environment','rewards','artifacts')}|{'experiments/market_planner.py','experiments/stationary_study.py'}
            validate_source_manifest(hashes,expected)
            for name,digest in hashes.items():
                if sha(record/'source'/name)!=digest:raise ValueError('Frozen source mismatch')
                target=code/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(record/'source'/name,target)
        for pkg in ('experiments','rl'):(code/pkg).mkdir(exist_ok=True);(code/pkg/'__init__.py').touch()
        configs=code/'experiments/configs';configs.mkdir(exist_ok=True)
        if study=='stationary':shutil.copyfile(record/'protocol.json',configs/'stationary-retraining.json')
        (code/'results').symlink_to(results_root,target_is_directory=True)
        module='stationary_study' if study=='stationary' else study+'_audit'
        subprocess.run([sys.executable,'-m','experiments.'+module,*values],cwd=code,
            env=dict(os.environ, PYTHONOPTIMIZE="0",PYTHONPATH=str(code)+os.pathsep+os.environ.get('PYTHONPATH','')),check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',choices=STUDIES,required=True)
    p.add_argument('--record');p.add_argument('--out');p.add_argument('--archive')
    p.add_argument('--workers',type=int,default=1);p.add_argument('--mode',choices=['verify','replay'],default='verify')
    p.add_argument('--run',action='store_true',help='Run retired study code from its verified snapshot; pass frozen CLI arguments after --')
    p.add_argument('arguments',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if a.workers<1:p.error('--workers must be positive')
    if a.run:run_archived(a.record or Path('results')/STUDIES[a.study],a.study,a.arguments,archive=a.archive)
    else:
        if a.arguments:p.error('Extra arguments require --run')
        result=reproduce_study(a.study,a.record,a.out,a.workers,a.archive,a.mode)
        if result is not None:print(json.dumps(result,indent=2))
