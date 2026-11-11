"""Render maintained study reports from complete records under runs/."""
import argparse
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
try:
    from experiments.common import read,sha,csv_read,csv_write,aggregate
except ModuleNotFoundError as exc:
    if exc.name != 'experiments.common':
        raise
    # Report verification also runs inside exact pre-refactor source snapshots.
    from experiments.feature_screen import read,sha,csv_read,csv_write,aggregate
from rl.artifacts import write_json

def record_dimensions(root, protocol):
    rows = csv_read(root/'episodes.csv')
    keys = ('case', 'method', 'size', 'replicate', 'model_seed', 'market_seed')
    episodes = len({tuple(row.get(key, '') for key in keys) for row in rows})
    inherited = (root/'parent/models.json').exists()
    models = read(root/('parent/models.json' if inherited else 'models.json'))
    models = [model for model in models
              if model['replicate'] in protocol['calibration_replicates']
              and model['seed'] in protocol['training_seeds']
              and (not inherited or model['size'] == protocol['calibration_size'])]
    return dict(datasets=len(protocol['calibration_replicates']),
                seeds=len(protocol['training_seeds']), markets=len(protocol['evaluation_seeds']),
                cases=len(protocol['cases']), models=len(models),
                training_steps=sum(model['steps'] for model in models), episodes=episodes,
                steps=episodes*protocol['evaluation_steps'], summaries=len(rows),
                refits=len(csv_read(root/'updates.csv')) if (root/'updates.csv').exists() else 0)

def planner_gaps(root):
    p = read(root/'protocol.json'); rows = csv_read(root/'episodes.csv')
    lookup = {(r['case'], r['method'], int(r['size']), int(r['replicate']), int(r['market_seed'])):r
              for r in rows if r['window'] == 'later' and r['method'] != 'fitted_sac'}
    reps = p['calibration_replicates']; markets = p['evaluation_seeds']; rng = np.random.default_rng(20261130)
    n = p['bootstrap_resamples']; ri = rng.integers(len(reps), size=(n, len(reps))); mi = rng.integers(len(markets), size=(n, len(markets)))
    output = []
    for case in p['cases']:
        for size in p['calibration_sizes']:
            for suffix in ['8']+(['64'] if size == max(p['calibration_sizes']) and case in ('nominal', 'combined') else []):
                for metric in ('mean_reward', 'mean_money_error', 'mean_fulfillment'):
                    delta = np.array([[float(lookup[case, 'fitted_saa'+suffix, size, r, m][metric])-float(lookup[case, 'oracle_saa'+suffix, 0, -1, m][metric]) for m in markets] for r in reps])
                    draws = delta[ri[:, :, None], mi[:, None, :]].mean((1, 2)); low, high = np.quantile(draws, [.025, .975])
                    output.append(dict(case=case, size=size, method='fitted_saa'+suffix, baseline='oracle_saa'+suffix,
                                       metric=metric, difference=float(delta.mean()), lower=float(low), upper=float(high)))
    return output

def auxiliary_tables(root):
    p = read(root/'protocol.json'); updates = csv_read(root/'updates.csv'); times = csv_read(root/'timings.csv')
    histories = []; costs = []
    for case in p['cases']:
        for method in ('online8', 'online64'):
            selected = [r for r in updates if r['case'] == case and r['method'] == method]
            if not selected: continue
            for step in sorted({int(r['after_step']) for r in selected}):
                rows = [r for r in selected if int(r['after_step']) == step]
                record = dict(case=case, method=method, after_step=step, trajectories=len(rows))
                for key in ('base_slope', 'reservation', 'switch_probability'):
                    values = np.array([float(r[key]) for r in rows])
                    record.update({f'mean_{key}':float(values.mean()), f'std_{key}':float(values.std()),
                        f'mean_absolute_error_{key}':float(np.abs(values-p['cases'][case][key]).mean())})
                record.update(slope_acceptance=float(np.mean([float(r['slope_accepted']) for r in rows])),
                    reservation_acceptance=float(np.mean([float(r['reservation_accepted']) for r in rows])))
                histories.append(record)
    for method in dict.fromkeys(r['method'] for r in times):
        rows = [r for r in times if r['method'] == method]
        per_step = np.array([1000*float(r['decision_cpu_seconds'])/float(r['decisions']) for r in rows])
        refits = np.array([1000*float(r['refit_cpu_seconds'])/float(r['decisions']) for r in rows])
        costs.append(dict(method=method, episodes=len(rows), mean_decision_cpu_ms=float(per_step.mean()),
            median_episode_decision_cpu_ms=float(np.median(per_step)), mean_refit_cpu_ms_per_period=float(refits.mean()),
            mean_total_cpu_ms_per_period=float((per_step+refits).mean())))
    return histories, costs

def render_model_error(root):
    root = Path(root); p = read(root/'protocol.json'); fits = read(root/'fits.json')
    d = record_dimensions(root, p)
    summaries = csv_read(root/'summary.csv'); contrasts = csv_read(root/'contrasts.csv')
    degradation = csv_read(root/'degradation.csv'); largest = max(p['calibration_sizes'])
    def summary(case, method, size=largest):
        return next(r for r in summaries if r['case'] == case and r['method'] == method and r['window'] == 'later'
                    and int(r['size']) == (size if method.startswith('fitted') else 0))
    def contrast(case, base, size=largest):
        return next(r for r in contrasts if r['case'] == case and int(r['size']) == size
                    and r['baseline'] == base and r['metric'] == 'mean_reward')
    def number(r, key, multiplier=1): return f'{float(r[key])*multiplier:.4f}'
    primary = read(root/'primary.json')
    lines = [
        '# Shared-data model estimation and frozen-controller transfer', '',
        'This is a new exploratory synthetic study. Earlier data, code and conclusions remain separately archived.', '',
        '## Findings and scope', '',
        'SAC and planning share the same public market records and estimated simulator. SAC is not given a correct simulator while planning receives a deliberately wrong one.',
        'SAC additionally generates explicitly budgeted synthetic training interactions. Planning can sample the same fitted model; correct-parameter planners remain references.', '',
        f'The declared primary comparison uses the larger calibration dataset, nominal market and later reward: SAC minus SAA8 = {float(primary["difference"]):+.5f},',
        f'exploratory 95% interval [{float(primary["lower"]):+.5f}, {float(primary["upper"]):+.5f}].',
        'Nominal reward and robustness to model error are distinct questions. Prediction quality, degradation under unseen parameters and stronger-planner comparisons are reported below.', '',
        '## Frozen design and information', '',
        '- Aggregate two-good NPC shop, shared instantaneous production, six-period warehouses, no expiry; same physical equations/reward as the preceding stationary study.',
        f'- {d["datasets"]} independent exploratory logging datasets; nested '+ '/'.join(map(str, p['calibration_sizes'])) +' public transitions per dataset. Bounded random target prices change every eight periods in 96-period logging episodes.',
        '- Estimate only demand slope, production reservation price and recipe-switch probability. Functional equations, production capacity, demand-arrival/noise laws, cash/stock constraints and all other constants are shared correct prior knowledge.',
        '- Fit public realized supply/sales with censoring and cash constraints. Do not read rejected demand, latent productivity/preference or true parameters. Forward least squares uses conditional-median noise, not an exact likelihood.',
        f'- Each fitted model trains {d["seeds"]} SAC initialization seeds, {p["training_episodes"]*p["training_steps_per_episode"]:,} synthetic interactions each, stationary inputs from the first replay sample. No checkpoint/test selection or online updates.',
        f'- Main planner (`SAA8` method ID): 82 first moves, hold/feedback continuation, horizon {p["planner"]["horizon"]}, {p["planner"]["particles"]} sampled futures. Supplement (`SAA64` ID): horizon {p["planner_supplement"]["horizon"]}, {p["planner_supplement"]["particles"]} futures on nominal/combined cases at the larger data size.',
        '- Oracle-labeled planners know the correct structural parameters, but never actual future shocks. Fitted controllers remain frozen in every case. PI gains are frozen from the earlier study, not retuned on these tests.',
        f'- {d["cases"]} cases, {d["markets"]} fresh paired evaluation markets, {p["evaluation_steps"]} periods each. Primary endpoint is periods {p["early_window"]+1}–{p["evaluation_steps"]}; all first/full windows are retained. Market shifts apply from test period one.',
        '- Shared logged data and prior knowledge do not imply matched compute: SAC has extra offline simulation/training; planning spends compute online. This entire SAC pipeline depends on a fitted simulator despite its model-free policy-learning algorithm.', '',
        '## Parameter estimates', '',
        '| Transitions | Dataset | Slope (true 5) | Reservation (true 7.5) | Switch probability (true .06) |',
        '| --- | --- | --- | --- | --- |',
    ]
    for f in fits:
        v = f['parameters']
        lines.append(f'| {f["size"]} | {f["replicate"]} | {v["base_slope"]:.5f} | {v["reservation"]:.5f} | {v["switch_probability"]:.5f} |')
    if (root/'prediction-audit/prediction.csv').exists():
        audit = csv_read(root/'prediction-audit/prediction.csv')
        lines += ['', '### Independent prediction audit', '',
            'Independent exploratory logs provide public states never used for fitting, tuning, training or test selection. Structural RMSE compares the estimated model with the correct conditional-median equations at the same states, including joint supply and sales clipping. It is neither realized-noise error nor a long-run control-value error.', '',
            '| Case | Transitions | Supply structural RMSE (units/good/period) | Sales structural RMSE | Switch probability absolute error |',
            '| --- | --- | --- | --- | --- |']
        for case in p['cases']:
            for size in p['calibration_sizes']:
                group = [r for r in audit if r['case'] == case and int(r['size']) == size]
                means = [np.mean([float(r[m]) for r in group]) for m in ('supply_structural_rmse', 'sales_structural_rmse', 'switch_probability_absolute_error')]
                lines.append(f'| {case} | {size} | {means[0]:.5f} | {means[1]:.5f} | {means[2]:.5f} |')
    lines += ['', 'The estimate tables and prediction audit quantify model error within the known synthetic equation family; they do not establish real-player identifiability.', '',
              f'## Later-window results at {largest} calibration transitions', '',
              '| Case | FeedbackController | Reward ↑ | Task completion % ↑ | Mean currency error % ↓ |',
              '| --- | --- | --- | --- | --- |']
    for case in p['cases']:
        methods = ['fitted_sac', 'fitted_saa8', 'oracle_saa8', 'pi']
        if case in ('nominal', 'combined'): methods += ['fitted_saa64', 'oracle_saa64']
        for method in methods:
            r = summary(case, method)
            lines.append(f'| {case} | {method} | {number(r,"mean_reward")} | {number(r,"mean_fulfillment",100)} | {number(r,"mean_money_error",100)} |')
    gaps = planner_gaps(root); csv_write(root/'planner-model-gap.csv', gaps)
    lines += ['', '### Impact of fitting within the same planner', '',
        'Secondary paired comparison at the same actual market: fitted minus correct-parameter planning. Negative reward differences indicate worse realized control with the fitted model. Correct parameters are not an upper bound for a finite-search planner, so differences need not always be negative. Intervals cross calibration and market seeds only; they contain no fictitious planner training replicates.', '',
        '| Transitions | Case | Planner | Reward difference [exploratory 95% interval] |',
        '| --- | --- | --- | --- |']
    for r in gaps:
        if r['metric'] == 'mean_reward':
            lines.append(f'| {r["size"]} | {r["case"]} | {r["method"]} | {r["difference"]:+.5f} [{r["lower"]:+.5f}, {r["upper"]:+.5f}] |')
    lines += ['', '## Paired reward contrasts', '',
              '| Transitions | Case | SAC minus | Difference | Exploratory 95% interval |',
              '| --- | --- | --- | --- | --- |']
    for r in contrasts:
        if r['metric'] != 'mean_reward': continue
        lines.append(f'| {r["size"]} | {r["case"]} | {r["baseline"]} | {float(r["difference"]):+.5f} | [{float(r["lower"]):+.5f}, {float(r["upper"]):+.5f}] |')
    lines += ['', '## Change from nominal: differential robustness', '',
              'Negative change is degradation. The final column is (SAC change − fitted SAA8 change); positive favors SAC retaining performance. It measures transfer under the specified shift, rather than robustness to every possible model error.', '',
              '| Transitions | Shift | SAC change | Fitted SAA8 change | Difference [95% interval] |',
              '| --- | --- | --- | --- | --- |']
    for r in degradation:
        lines.append(f'| {r["size"]} | {r["case"]} | {float(r["sac_change"]):+.5f} | {float(r["planner_change"]):+.5f} | {float(r["difference"]):+.5f} [{float(r["lower"]):+.5f}, {float(r["upper"]):+.5f}] |')
    lines += ['', '## Limits and interpretation', '',
        f'Crossed bootstrap resamples {d["datasets"]} calibration datasets, {d["seeds"]} common initialization seeds and {d["markets"]} evaluation markets. Planners are broadcast over initialization seeds, not counted as extra independent runs. Intervals are exploratory and pointwise without multiplicity correction; do not pool them with the previous confirmation.', '',
        'An absolute SAC reward advantage does not establish smaller sensitivity to model error. Compare changes from nominal and fitted-versus-oracle planning. These reward changes include changed task difficulty, so differential degradation is a performance measure, not an isolated causal mechanism. Increasing logged data does not guarantee monotonic RL improvement under finite training.', '',
        'This is model estimation within a correct functional family, not structural model misspecification, online adaptation, real-player calibration, or a proof of universal RL superiority. All baseline scores, seed failures and transfer cases are retained.', '',
        '## Reproduction and artifacts', '',
        'Use a working Python environment with the repository requirements:', '',
        '```sh',
        'python -m experiments.model_error_study --phase init --out /tmp/new-model-error-study',
        'python -m experiments.model_error_study --phase calibrate --out /tmp/new-model-error-study',
        'python -m experiments.model_error_study --phase train --out /tmp/new-model-error-study --workers 6',
        'python -m experiments.model_error_study --phase evaluate --out /tmp/new-model-error-study --workers 6',
        'python -m experiments.model_error_study --phase report --out /tmp/new-model-error-study',
        'python -m experiments.model_error_study --phase verify --out /tmp/new-model-error-study',
        'python scripts/reproduce.py --study model-error --record results/diagnostics/2026-11-09-model-error-study',
        '```', '',
        f'Frozen protocols, source manifests, calibration logs, parameter fits, {d["models"]} final checkpoints and training curves, compressed traces and {d["summaries"]:,} window summaries are retained. See `verification.json` and `archive-verification.json` for actual verification status; `manifest.json` records the execution environment.', '',
        '![Model estimation and transfer](model-error.png)', '',
    ]
    (root/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    rows = csv_read(root/'episodes.csv')
    csv_write(root/'per-model.csv', aggregate(rows, ['case', 'method', 'size', 'replicate', 'model_seed', 'window']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained')
    cases = list(p['cases']); x = np.arange(len(cases))
    for method, color, label in [('fitted_sac', '#0072B2', 'SAC: fitted simulator'), ('fitted_saa8', '#D55E00', 'SAA8: fitted model'), ('oracle_saa8', '#009E73', 'SAA8: correct parameters'), ('pi', '#777777', 'PI')]:
        axes[0].plot(x, [float(summary(c, method)['mean_reward']) for c in cases], 'o-', label=label, color=color)
    axes[0].set_title('Shared data; frozen controllers'); axes[0].set_ylabel('Later reward (higher is better)')
    axes[0].set_xticks(x, cases, rotation=35, ha='right'); axes[0].legend(fontsize=8)
    for size, offset, color in [(p['calibration_sizes'][0], -.06, '#E69F00'), (largest, .06, '#0072B2')]:
        chosen = [contrast(c, 'fitted_saa8', size) for c in cases]
        y = np.array([float(r['difference']) for r in chosen]); low = np.array([float(r['lower']) for r in chosen]); high = np.array([float(r['upper']) for r in chosen])
        axes[1].errorbar(x+offset, y, yerr=[y-low, high-y], fmt='o', capsize=3, color=color, label=f'{size} logged transitions')
    axes[1].axhline(0, color='black', linewidth=.8); axes[1].set_title('SAC minus fitted SAA8')
    axes[1].set_ylabel('Reward difference, exploratory 95% CI'); axes[1].set_xticks(x, cases, rotation=35, ha='right'); axes[1].legend(fontsize=8)
    chosen = [r for r in degradation if int(r['size']) == largest]
    y = np.array([float(r['difference']) for r in chosen]); low = np.array([float(r['lower']) for r in chosen]); high = np.array([float(r['upper']) for r in chosen]); xx = np.arange(len(chosen))
    axes[2].errorbar(xx, y, yerr=[y-low, high-y], fmt='o', capsize=3, color='#CC79A7')
    axes[2].axhline(0, color='black', linewidth=.8); axes[2].set_title('Relative retention under market shifts')
    axes[2].set_ylabel('(SAC shift change) - (SAA8 shift change)'); axes[2].set_xticks(xx, [r['case'] for r in chosen], rotation=35, ha='right')
    fig.savefig(root/'model-error.png', dpi=150); fig.savefig(root/'model-error.pdf'); plt.close(fig)
    shutil.copyfile(__file__, root/'report-render-source.py')
    names = ['README.md', 'per-model.csv', 'planner-model-gap.csv', 'model-error.png', 'model-error.pdf', 'report-render-source.py']
    inputs = ['protocol.json', 'fits.json', 'episodes.csv', 'summary.csv', 'contrasts.csv', 'degradation.csv', 'primary.json']
    if (root/'prediction-audit/prediction.csv').exists(): inputs.append('prediction-audit/prediction.csv')
    write_json(root/'render-provenance.json', dict(inputs={n: sha(root/n) for n in inputs}, outputs={n: sha(root/n) for n in names}))

def render_adaptive(root):
    root = Path(root); p = read(root/'protocol.json'); summary = csv_read(root/'summary.csv'); contrasts = csv_read(root/'contrasts.csv')
    d = record_dimensions(root, p)
    first_update = ((p['minimum_observations']+p['update_interval']-1)//p['update_interval'])*p['update_interval']
    histories, costs = auxiliary_tables(root)
    csv_write(root/'estimate-quality.csv', histories); csv_write(root/'cpu-summary.csv', costs)
    def row(case, method, window='later'):
        return next(r for r in summary if r['case'] == case and r['method'] == method and r['window'] == window)
    def contrast(case, method, baseline, window='later'):
        return next(r for r in contrasts if r['case'] == case and r['method'] == method and r['baseline'] == baseline and r['metric'] == 'mean_reward' and r['window'] == window)
    primary = read(root/'primary.json'); sac64 = contrast('combined', 'sac', 'online64')
    classification = 'positive interval' if float(sac64['lower']) > 0 else 'negative interval' if float(sac64['upper']) < 0 else 'superiority not established'
    explanation = {'positive interval':'the exploratory interval supports higher SAC reward',
        'negative interval':'the exploratory interval supports lower SAC reward',
        'superiority not established':'the interval crosses zero, so SAC superiority is not established'}[classification]
    assessment = dict(primary=primary, sac_minus_online64_combined=sac64, classification=classification,
                      scope=f'Exploratory pointwise intervals, {d["datasets"]} calibration datasets, {d["seeds"]} initialization seeds and {d["markets"]} fresh markets.')
    write_json(root/'assessment.json', assessment)
    lines = ['# Periodic model refitting versus frozen planning', '',
        '## Findings and scope', '',
        f'The market, reward and inherited SAC policies are fixed. Planning updates its model first after {first_update} periods, then every {p["update_interval"]}, with at most {p["rolling_window"]} observations. Insufficiently identified parameters retain previous values. True parameters and future outcomes are unavailable.', '',
        f'Under the combined shift, later reward for online SAA8 minus frozen SAA8 is **{float(primary["difference"]):+.5f}**, exploratory 95% interval **[{float(primary["lower"]):+.5f}, {float(primary["upper"]):+.5f}]**.', '',
        f'For the same case, SAC minus online SAA64 is **{float(sac64["difference"]):+.5f}**, interval **[{float(sac64["lower"]):+.5f}, {float(sac64["upper"]):+.5f}]**; {explanation}.', '',
        'This compares particular complete control pipelines, not the limits of all RL. Earlier frozen-model findings remain valid within their original conditions.', '',
        '## Frozen design', '',
        f'- Aggregate two-good market, 82 first moves, two continuations; main horizon {p["planner"]["horizon"]}, {p["planner"]["particles"]} sampled futures.',
        f'- {d["datasets"]} calibration datasets of {p["calibration_size"]} transitions and {d["models"]} inherited SAC policies; no retraining or checkpoint selection.',
        f'- {d["markets"]} paired evaluation markets, {d["cases"]} cases, {p["evaluation_steps"]} periods each. Parameters differ from calibration from period one; task recipes switch.',
        f'- Main planner IDs retain the `8` suffix; supplement IDs retain `64` and use horizon {p["planner_supplement"]["horizon"]}, {p["planner_supplement"]["particles"]} futures on '+', '.join(p['supplement_cases'])+'. Frozen PI is retained.',
        f'- Public transactions feed the censored estimator. A {p["minimum_identifying_points"]}-point identifiability gate can retain prior behavioral parameters; all estimates and used-data indices are saved.',
        f'- Model updates preserve integral memory and private forecast RNG. Online/frozen planners match before the first update; refitting can change actions from period {first_update+1}.',
        f'- Windows: 1–{p["minimum_observations"]} pre-update, {p["minimum_observations"]+1}–{p["early_window"]} adaptation, {p["early_window"]+1}–{p["evaluation_steps"]} later, first {p["early_window"]}, and full. This record has {d["episodes"]:,} episodes, {d["steps"]:,} physical steps, {d["summaries"]:,} window summaries and {d["refits"]:,} refits.',
        f'- Online planning retains a {p["rolling_window"]}-period estimation window; SAC uses fixed weights and two-frame inputs. Memory and online computation differ.',
        f'- The {d["models"]} selected policies originally used {d["training_steps"]:,} synthetic interactions. Additional training here is zero; refitting/planning compute is recorded separately.', '',
        '## Later-window results', '',
        '| Market | Method | Reward ↑ | Task completion % ↑ | Currency error % ↓ |',
        '| --- | --- | ---: | ---: | ---: |']
    for case in p['cases']:
        methods = ['sac', 'frozen8', 'online8', 'oracle8', 'pi']
        if case in p['supplement_cases']: methods += ['frozen64', 'online64', 'oracle64']
        for method in methods:
            r = row(case, method)
            lines.append(f'| {case} | {method} | {float(r["mean_reward"]):.5f} | {100*float(r["mean_fulfillment"]):.3f} | {100*float(r["mean_money_error"]):.3f} |')
    lines += ['', '## Reward comparisons and adaptation windows', '',
        '| Market | Window | Method − baseline | Difference | Exploratory 95% interval |',
        '| --- | --- | --- | ---: | --- |']
    for r in contrasts:
        if r['metric'] != 'mean_reward' or r['window'] == 'pre_update': continue
        lines.append(f'| {r["case"]} | {r["window"]} | {r["method"]} − {r["baseline"]} | {float(r["difference"]):+.5f} | [{float(r["lower"]):+.5f}, {float(r["upper"]):+.5f}] |')
    lines += ['', f'Crossed intervals resample {d["datasets"]} calibration datasets, {d["seeds"]} common SAC initialization seeds and {d["markets"]} markets. Planner copies over initialization seeds are not independent samples. These pointwise intervals are exploratory, without multiplicity adjustment or pooling with earlier studies.', '',
        '## Learned parameter histories', '',
        'The plotted bands show observed trajectory spread, not confidence bands. Ground-truth parameters enter analysis only. `estimate-quality.csv` retains every update-time mean, spread, absolute parameter error and acceptance rate; `updates.csv` retains every original update.', '',
        '| Market | Method | After period | Estimated slope | Estimated reservation | Estimated switch probability |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    shown_steps = {p['minimum_observations'], p['early_window'], p['evaluation_steps']}
    for r in histories:
        if r['after_step'] in shown_steps:
            lines.append(f'| {r["case"]} | {r["method"]} | {r["after_step"]} | {r["mean_base_slope"]:.4f} | {r["mean_reservation"]:.4f} | {r["mean_switch_probability"]:.4f} |')
    lines += ['', '## Computation', '',
        'Descriptive process CPU cost during this execution, including encoding/decision overhead. Refitting is amortized over all periods. These are neither production latency nor a total-compute-matched benchmark; timings are hash-checked rather than rerun during deterministic replay.', '',
        '| Method | Mean decision CPU ms/period | Mean refit CPU ms/period | Total CPU ms/period |',
        '| --- | ---: | ---: | ---: |']
    for r in costs:
        lines.append(f'| {r["method"]} | {r["mean_decision_cpu_ms"]:.4f} | {r["mean_refit_cpu_ms_per_period"]:.4f} | {r["mean_total_cpu_ms_per_period"]:.4f} |')
    lines += ['', '## Evidence scope and reproduction', '',
        'Higher score after refitting tests a concrete frozen-model failure explanation. A finite planner with correct parameters is a reference, not an optimal bound. This study does not establish an inherent ordering of RL versus rules, equal-memory learning capability, or performance under unknown functional forms.', '',
        'All behavior is synthetic. Parameters change relative to calibration at deployment, not midway through an episode. Frozen SAC does not update weights online. No human-behavior calibration, welfare, retention or real-game inflation effect is established.', '',
        'See the [full protocol](../../../docs/protocols.md#periodic-online-refitting), `protocol.json`, `source-hashes.json`, copied parent models and manifests, all compressed traces, windows, update logs, contrasts, costs, and verification reports. Earlier records are unchanged.', '',
        '```sh',
        'python -m experiments.adaptive_planning --phase init --out /tmp/new-adaptive-record',
        'python -m experiments.adaptive_planning --phase run --out /tmp/new-adaptive-record --workers 6',
        'python -m experiments.adaptive_planning --phase report --out /tmp/new-adaptive-record',
        'python scripts/report.py --study adaptive --record /tmp/new-adaptive-record',
        'python scripts/reproduce.py --study adaptive --record /tmp/new-adaptive-record --workers 6',
        '```', '',
        '![Online model refitting and controller comparison](adaptive-planning.png)', '']
    (root/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained'); cases = list(p['cases']); x = np.arange(len(cases))
    for method, color, label in [('sac', '#0072B2', 'SAC, frozen policy'), ('frozen8', '#D55E00', 'SAA8, frozen model'), ('online8', '#009E73', 'SAA8, online refitting'), ('oracle8', '#CC79A7', 'SAA8, correct parameters'), ('pi', '#777777', 'PI')]:
        axes[0].plot(x, [float(row(c, method)['mean_reward']) for c in cases], 'o-', color=color, label=label)
    axes[0].set_xticks(x, cases, rotation=35, ha='right'); axes[0].set_ylabel('Later reward, higher is better'); axes[0].set_title('Fresh markets, identical inherited SAC weights'); axes[0].legend(fontsize=7)
    baselines = ['frozen8', 'online8', 'frozen64', 'online64']; selected = [contrast('combined', 'sac', b) for b in baselines]
    y = np.array([float(r['difference']) for r in selected]); low = np.array([float(r['lower']) for r in selected]); high = np.array([float(r['upper']) for r in selected])
    axes[1].errorbar(np.arange(4), y, yerr=[y-low, high-y], fmt='o', capsize=4); axes[1].axhline(0, color='black', linewidth=.8)
    axes[1].set_xticks(np.arange(4), baselines, rotation=35, ha='right'); axes[1].set_ylabel('SAC minus planner, exploratory 95% CI'); axes[1].set_title('Combined shift: surviving SAC advantage')
    for case, color in [('nominal', '#0072B2'), ('combined', '#D55E00')]:
        data = [r for r in histories if r['case'] == case and r['method'] == 'online8']
        xx = np.array([r['after_step'] for r in data]); yy = np.array([r['mean_reservation'] for r in data]); sd = np.array([r['std_reservation'] for r in data])
        axes[2].plot(xx, yy, color=color, label=case+' estimate ± trajectory SD'); axes[2].fill_between(xx, yy-sd, yy+sd, color=color, alpha=.15)
        axes[2].axhline(p['cases'][case]['reservation'], color=color, linestyle='--', linewidth=.8)
    axes[2].set_xlabel('Observed period, update applied afterward'); axes[2].set_ylabel('Production reservation estimate'); axes[2].set_title('Causal on-policy identification'); axes[2].legend(fontsize=7)
    fig.savefig(root/'adaptive-planning.png', dpi=150); fig.savefig(root/'adaptive-planning.pdf'); plt.close(fig)
    shutil.copyfile(__file__, root/'render-source.py')
    names = ['README.md', 'estimate-quality.csv', 'cpu-summary.csv', 'assessment.json', 'adaptive-planning.png', 'adaptive-planning.pdf', 'render-source.py']
    inputs = ['protocol.json', 'episodes.csv', 'summary.csv', 'contrasts.csv', 'primary.json', 'updates.csv', 'timings.csv']
    write_json(root/'render-provenance.json', dict(inputs={n:sha(root/n) for n in inputs}, outputs={n:sha(root/n) for n in names}))

def render_compute(root):
    root = Path(root); p = read(root/'protocol.json')
    d = record_dimensions(root, p)
    quality = csv_read(root/'frontier.csv'); costs = csv_read(root/'common-costs.csv')
    contrasts = csv_read(root/'contrasts.csv'); crossovers = csv_read(root/'break-even.csv')
    training = csv_read(root/'training-cost.csv'); samples = csv_read(root/'decision-samples.csv')
    median_training = float(np.median([float(r['cpu_seconds']) for r in training]))
    methods = [r['method'] for r in costs]; names = {x['method']:f'H{x["horizon"]}/F{x["particles"]}' for x in p['planners']}
    names.update(sac='SAC',pi='PI')
    def score(case,method): return next(r for r in quality if r['case'] == case and r['method'] == method)
    def gap(case,method):
        return next(r for r in contrasts if r['case'] == case and r['baseline'] == method and r['window'] == 'later' and r['metric'] == 'mean_reward')
    def number_or_dash(value,places=1): return '—' if value == '' else f'{float(value):,.{places}f}'
    lines = ['# Control-quality / CPU trade-off and training-cost remeasurement', '',
        '## Findings and scope', '',
        f'{len(p["planners"])} planning budgets are evaluated with the same market, reward, online estimator and {d["models"]} fixed SAC policies. PI remains included: faster inference than heavy planning does not make SAC the fastest controller.', '',
        'Quality uses fresh paired markets. CPU is measured separately on common public states and histories; these probes do not imply identical policy trajectories.', '',
        f'Historical training duration was not recorded. The {len(training)} exact-setting timed replays measure a median of **{median_training:.2f} CPU seconds per policy** on the current host. Replayed weights and curves match the originals; evaluated policies are unchanged. These are remeasurements, not historical durations.', '',
        'CPU amortization concerns cumulative computation, not equal control quality or real-game benefit. Online fitting assumes the known equation family. No real-player data calibrate this study.', '',
        '## Quality and common-state cost', '',
        f'Mean online CPU includes decision plus amortized observe/refit. Higher reward is better. All {len(p["planners"])} planning budgets update parameters identically; only horizon/scenario count differs.', '',
        '| Method | Online CPU ms/period | Nominal reward | Combined-shift reward |',
        '| --- | ---: | ---: | ---: |']
    for c in costs:
        method = c['method']
        lines.append(f'| {names[method]} | {float(c["mean_total_cpu_ms"]):.4f} | {float(score("nominal",method)["mean_reward"]):.5f} | {float(score("combined",method)["mean_reward"]):.5f} |')
    lines += ['', 'These means do not establish a winner by themselves. The paired exploratory intervals below concern reward differences; a zero-crossing interval is not an equivalence test.', '',
        '| Case | SAC minus controller | Difference | Exploratory 95% interval |',
        '| --- | --- | ---: | --- |']
    for r in contrasts:
        if r['window'] != 'later' or r['metric'] != 'mean_reward': continue
        lines.append(f'| {r["case"]} | {names[r["baseline"]]} | {float(r["difference"]):+.5f} | [{float(r["lower"]):+.5f}, {float(r["upper"]):+.5f}] |')
    lines += ['', '## Service, currency and initial CPU cost', '',
        f'For one new deployed SAC policy, the following arithmetic adds the median **remeasured** training CPU to {p["evaluation_steps"]} online decisions. Other methods have no policy-training term here. Shared initial model-estimation/data-collection costs, engineering, searches and failed training are excluded. This is not total research cost.', '',
        '| Method | Estimated CPU seconds for initial training + '+str(p['evaluation_steps'])+' decisions | Combined completion % | Combined currency error % |',
        '| --- | ---: | ---: | ---: |']
    for c in costs:
        method = c['method']; q = score('combined',method)
        total = p['evaluation_steps']*float(c['mean_total_cpu_ms'])/1000+(median_training if method == 'sac' else 0.)
        lines.append(f'| {names[method]} | {total:.3f} | {100*float(q["mean_fulfillment"]):.3f} | {100*float(q["mean_money_error"]):.3f} |')
    lines += ['', '## Conditional CPU break-even per deployed policy', '',
        'Assume the same policy is reused without retraining and its control quality is acceptable. Divide one-policy training CPU by per-decision CPU saving. No CPU crossover exists against a faster controller. The cadences below are hypothetical, not recommendations about game price-update frequency.', '',
        '| Compared with | Decisions to CPU break-even, approximately | Days at one decision/minute | Days at one decision/hour |',
        '| --- | ---: | ---: | ---: |']
    for r in crossovers:
        lines.append(f'| {names[r["baseline"]]} | {number_or_dash(r["decisions_to_cpu_break_even"],0)} | {number_or_dash(r["days_if_one_decision_per_minute"])} | {number_or_dash(r["days_if_one_decision_per_hour"])} |')
    lines += ['', 'There is no monetary/energy/welfare inference from this CPU arithmetic. Timing variation, different implementations/hardware and future retraining change the crossover. A smaller number does not compensate automatically for worse control quality.', '',
        '## Training replays and provenance', '',
        '| Calibration / seed | Interactions | Updates | CPU seconds | Wall seconds | Original policy/curve match |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for r in training:
        lines.append(f'| {r["replicate"]} / {r["seed"]} | {r["steps"]} | {r["updates"]} | {float(r["cpu_seconds"]):.3f} | {float(r["wall_seconds"]):.3f} | {r["identical_policy"]} / {r["identical_curve"]} |')
    lines += ['',
        f'These timed replays consume {sum(int(r["steps"]) for r in training):,} extra simulated interactions and {sum(int(r["updates"]) for r in training):,} updates. They are reproduction/cost measurements, not additional performance training or selection. CPU and wall durations are measured on the current shared host; wall time is affected by parallel work. Imports are excluded, full training-function initialization/simulation/learning/saving are included.', '',
        '## Frozen evaluation and profiling design', '',
        f'- Quality: {d["episodes"]} fresh-market episodes, {d["cases"]} cases, {d["markets"]} paired markets, {d["datasets"]} fitted calibration models and {d["models"]} inherited SAC policies. Later reward uses periods {p["early_window"]+1}–{p["evaluation_steps"]}; other windows remain available. No prior test markets are pooled.',
        f'- Every planner uses 82 first actions and hold/feedback continuations. The known equation family and noise laws are shared prior knowledge. Estimation uses a {p["minimum_observations"]}-observation minimum, {p["update_interval"]}-period update interval, {p["rolling_window"]}-period memory and {p["minimum_identifying_points"]}-point gate. SAC retains two-frame inputs and fixed weights.',
        f'- Speed: {len(read(root/"common-states.json"))} common PI-visited states/histories, one warmup and {p["timing_repetitions"]} measured calls per method/model/state; {len(samples):,} retained decision samples. Counterfactual planner integral memory is zero and its private forecast RNG is fresh. Preparation/loading/construction are outside the timer; encoding/public-state reconstruction/search/inference inside `act` are included.',
        '- The identical observation/refit function is timed once across common histories. Update and ordinary-observation samples are combined at the declared 1/'+str(p['update_interval'])+' update rate; they are not equally weighted despite balanced audit positions. All observation samples and actual-rollout costs remain visible.',
        '- Benchmarking follows completion of the heavy evaluation/training phases and uses one Torch thread. Local process CPU is not production latency or a total-compute-matched comparison. Private planner forecast seed is fixed as in the prior study; rankings are conditional on the tested implementation.',
        '- Paired intervals resample calibration datasets, common SAC initialization seeds and evaluation markets. Planner/PI copies over SAC seeds are not extra independent samples. Few groups and many comparisons make intervals exploratory and pointwise; there is no multiplicity correction or declared noninferiority margin.',
        '- Point-estimate nondominated flags in `frontier.csv` are descriptive. No tested-budget winner is selected and presented as a separate confirmation.',
        '- Parameters differ from calibration starting at deployment and stay fixed within each episode. Task recipes remain observable. This is not unknown within-episode law detection, online SAC learning, real-player calibration or novel SAC.', '',
        '## Reproduction', '',
        'See the [full protocol](../../../docs/protocols.md#quality-and-computation), complete traces, parent/checkpoint hashes, update logs, benchmark states/samples, timing-replay artifacts and saved verification reports.', '',
        '```sh',
        'python -m experiments.compute_tradeoff --phase init --out /tmp/new-compute-record',
        'python -m experiments.compute_tradeoff --phase run --out /tmp/new-compute-record --workers 6',
        'python -m experiments.compute_tradeoff --phase training-cost --out /tmp/new-compute-record --workers 3',
        'python -m experiments.compute_tradeoff --phase benchmark --out /tmp/new-compute-record',
        'python -m experiments.compute_tradeoff --phase analyze --out /tmp/new-compute-record',
        'python scripts/report.py --study compute --record /tmp/new-compute-record',
        'python scripts/reproduce.py --study compute --record /tmp/new-compute-record --workers 6',
        '```', '',
        f'The archive verifies stored timings and arithmetic; it does not rerun noisy timing or the {len(training)} full trainings. It reconstructs common audit states, selected timing actions, replayed weights/curves, summaries/intervals and a fixed subset of complete trajectories.', '',
        '![Quality / online CPU and conditional amortization](compute-tradeoff.png)', '']
    (root/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(16,5),layout='constrained')
    colors = ['#777777','#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9','#A6761D']
    cost_by_method = {r['method']:float(r['mean_total_cpu_ms']) for r in costs}
    for ax,case in zip(axes[:2],('nominal','combined')):
        ax.axhline(0,color='black',linewidth=.8)
        for i,method in enumerate(methods):
            x = cost_by_method[method]
            if method == 'sac':
                ax.plot(x,0,'*',markersize=13,color=colors[i]); y = 0
            else:
                r = gap(case,method); y = -float(r['difference']); lo = -float(r['upper']); hi = -float(r['lower'])
                ax.errorbar(x,y,yerr=[[y-lo],[hi-y]],fmt='o',capsize=3,color=colors[i])
            ax.annotate(names[method],(x,y),xytext=(4,5 if i%2 else -12),textcoords='offset points',fontsize=7)
        ax.set_xscale('log'); ax.set_xlabel('Common-state mean online process CPU (ms/period)')
        ax.set_ylabel('FeedbackController reward minus SAC, exploratory 95% CI'); ax.set_title(case+' market: higher is better')
    rows = [r for r in crossovers if r['decisions_to_cpu_break_even'] != '']
    labels = [names[r['baseline']] for r in rows]; values = [float(r['decisions_to_cpu_break_even']) for r in rows]
    axes[2].barh(np.arange(len(rows)),values,color='#0072B2'); axes[2].set_yticks(np.arange(len(rows)),labels)
    axes[2].set_xscale('log'); axes[2].set_xlabel('Decisions to repay one-policy training CPU')
    axes[2].set_title('CPU crossover, conditional on acceptable quality')
    if any(r['baseline'] == 'pi' and r['decisions_to_cpu_break_even'] == '' for r in crossovers):
        axes[2].text(.02,.02,'PI is faster than SAC: no CPU crossover',transform=axes[2].transAxes,fontsize=7)
    fig.savefig(root/'compute-tradeoff.png',dpi=150); fig.savefig(root/'compute-tradeoff.pdf'); plt.close(fig)
    shutil.copyfile(__file__,root/'render-source.py')
    names = ['README.md','compute-tradeoff.png','compute-tradeoff.pdf','render-source.py']
    inputs = ['protocol.json','episodes.csv','frontier.csv','common-costs.csv','contrasts.csv','break-even.csv','training-cost.csv','common-states.json','decision-samples.csv']
    write_json(root/'render-provenance.json',dict(inputs={n:sha(root/n) for n in inputs},outputs={n:sha(root/n) for n in names}))

def render_zoom(root):
    root = Path(root); costs = csv_read(root/'common-costs.csv'); rows = csv_read(root/'contrasts.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    labels = dict(pi='PI',sac='SAC',h4_f2='H4/F2',h4_f8='H4/F8',h12_f2='H12/F2',h12_f8='H12/F8',h12_f64='H12/F64')
    colors = ['#777777','#0072B2','#009E73','#CC79A7','#D55E00','#56B4E9','#A6761D']
    for ax,case in zip(axes,('nominal','combined')):
        ax.axhline(0,color='black',linewidth=.8)
        for i,(method,label) in enumerate(labels.items()):
            x = float(next(r['mean_total_cpu_ms'] for r in costs if r['method'] == method))
            if method == 'sac': y = 0.; ax.plot(x,y,'*',color=colors[i],markersize=13)
            else:
                r = next(r for r in rows if r['case'] == case and r['baseline'] == method and r['window'] == 'later' and r['metric'] == 'mean_reward')
                y = -float(r['difference']); lower = -float(r['upper']); upper = -float(r['lower'])
                ax.errorbar(x,y,yerr=[[y-lower],[upper-y]],fmt='o',capsize=4,color=colors[i])
            offset = (-47,10) if method == 'h12_f64' else (5,9 if method in ('sac','h12_f2') else -14)
            ax.annotate(label,(x,y),xytext=offset,textcoords='offset points',fontsize=8)
        ax.set_xscale('log'); ax.set_xlabel('Common-state mean online process CPU, ms/period')
        ax.set_ylabel('Controller reward minus SAC, exploratory 95% CI')
        ax.set_title(case+' market; higher is better')
    fig.suptitle('Zoom excluding H1/F2 only; all candidates retained in the full figure and tables',fontsize=10)
    fig.savefig(root/'compute-tradeoff-zoom.png',dpi=160); fig.savefig(root/'compute-tradeoff-zoom.pdf'); plt.close(fig)
    shutil.copyfile(__file__,root/'zoom-source.py')
    write_json(root/'zoom-provenance.json',dict(scope='Plot-only post-analysis zoom; no new experiments, selection or statistical changes.',
        inputs={n:sha(root/n) for n in ('common-costs.csv','contrasts.csv')},
        outputs={n:sha(root/n) for n in ('compute-tradeoff-zoom.png','compute-tradeoff-zoom.pdf','zoom-source.py')}))

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',choices=['model-error','adaptive','compute','compute-zoom'],required=True)
    p.add_argument('--record',required=True,type=Path)
    a=p.parse_args()
    if (a.record/'source').exists():
        {'model-error':render_model_error,'adaptive':render_adaptive,'compute':render_compute,'compute-zoom':render_zoom}[a.study](a.record)
    else:
        p.error('Restore a complete record under runs/ before rendering a report')
