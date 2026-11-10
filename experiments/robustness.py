"""Five-seed response robustness and validation-frozen currency/activity trade-offs."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, replace
import hashlib
import itertools
import inspect
import json
from multiprocessing import get_context
from pathlib import Path
import shutil

import numpy as np
import torch

from rl.artifacts import manifest, new_run_directory, write_json
from rl.config import MarketConfig
from rl.environment import MultiStoreEnv
from rl.sac import SACAgent, verify_checkpoint_updates
from .balanced_control import load_config
from .baselines import CurrencyFeedbackPolicy
from .evaluate import write_csv
from .evaluate import BlendPolicy, METRICS, control_rollout as rollout, control_summary as summarize, paired_interval
from .common import snapshot_research


def read_csv(path):
    rows = list(csv.DictReader(Path(path).open()))
    for row in rows:
        row["training_seed"] = int(row["training_seed"]) if row["training_seed"] else ""
        row["evaluation_seed"] = int(row["evaluation_seed"])
        for metric in METRICS:
            row[metric] = float(row[metric])
    return rows


def validate_protocol(config):
    market,sac,base = load_config(config["base_config"])
    if config["training_steps"] != 30000 or config["training_seeds"] != [0,1,2,3,4]:
        raise ValueError("Use the declared five-seed 30k comparison")
    used = {s+ep*10000 for s in config["training_seeds"] for ep in range(301)}
    for name in ("validation_seeds","evaluation_seeds"):
        seeds = config[name]
        if (len(seeds)<2 or len(set(seeds))!=len(seeds)
                or any(type(s) is not int or s<0 for s in seeds) or set(seeds)&used):
            raise ValueError("Invalid or overlapping market seeds")
    if set(config["validation_seeds"])&set(config["evaluation_seeds"]):
        raise ValueError("Validation/test leakage")
    permitted = {"price_sensitivity_low","price_sensitivity_high","shock_multiplier"}
    for name in ("validation_scenarios","test_scenarios"):
        scenarios = config[name]
        if len({s["name"] for s in scenarios}) != len(scenarios):
            raise ValueError("Duplicate scenarios")
        for scenario in scenarios:
            if set(scenario["changes"])-permitted:
                raise ValueError("Scenario changes observation/action semantics")
            replace(market,**scenario["changes"])
    weights = config["feedback_weights"]
    if sorted(set(weights))!=weights or weights[0]!=0 or weights[-1]!=1:
        raise ValueError("Blend grid must increase from zero to one")
    if not all(0<w<1 for w in weights[1:-1]):
        raise ValueError("Invalid blend weights")
    if any(not 0<c<1 for c in config["currency_caps"]) or config["matching_tolerance"]<=0:
        raise ValueError("Invalid currency constraints")
    return market,sac,base


def initialize(root,config_path):
    source = Path(config_path)
    config = json.loads(source.read_text())
    validate_protocol(config)
    collisions = []
    chosen = set(config["validation_seeds"]+config["evaluation_seeds"])
    for path in Path("results").rglob("*.json"):
        try:
            value = json.loads(path.read_text())
        except (ValueError,UnicodeDecodeError):
            continue
        if isinstance(value,dict) and any(chosen & {s for s in value.get(k,[]) if type(s) is int}
                                          for k in ("validation_seeds","evaluation_seeds") if isinstance(value.get(k,[]),list)):
            collisions.append(str(path))
    new_run_directory(root)
    write_json(root/"protocol.json",config)
    write_json(root/"protocol-record.json",{"input_path":str(source),"input_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "test_episodes_seen":0,"prior_protocol_market_seed_collisions":collisions,
        "manifest":manifest("pretraining-robustness-design")})


def stage_models(root,config,market,sac):
    expected = replace(sac,episodes=300)
    records = []
    for family in ("nominal","randomized"):
        for seed in config["training_seeds"]:
            if family=="nominal":
                source = Path(config["nominal_extension_root"] if seed in config["new_nominal_seeds"] else config["prior_budget_root"])/f"seed-{seed}"
                policy = source/"steps-30000/policy.pt"
            else:
                source = Path(config["randomized_training_root"])/f"seed-{seed}"
                policy = source/"policy.pt"
            folder = new_run_directory(root/"models"/family/f"seed-{seed}")
            shutil.copy2(policy,folder/"policy.pt")
            if family=="nominal":
                shutil.copy2(source/"steps-10000/policy.pt",folder/"reference-10k-policy.pt")
            for name in ("training.csv","config.json","manifest.json","completion.json","outcome.json",
                         "checkpoints.json","training-source-hashes.json"):
                if (source/name).is_file():
                    shutil.copy2(source/name,folder/name)
            agent,meta = SACAgent.load(folder/"policy.pt",MultiStoreEnv(market))
            verification = verify_checkpoint_updates(meta,folder/"policy.pt")
            if (meta["seed"]!=seed or meta["trained_steps"]!=config["training_steps"]
                    or verification["gradient_updates"]!=29001 or agent.config!=expected):
                raise ValueError("Wrong model seed, budget, updates or hyperparameters")
            if family=="randomized" and meta.get("training_distribution")!=config["response_distribution"]:
                raise ValueError("Missing randomized-training provenance")
            if family=="nominal":
                checks = json.loads((source/"checkpoints.json").read_text())
                initial = next(r for r in checks if r["steps"]==10000)
                if not initial["matches_published_policy"] or not initial["matches_published_history"]:
                    raise ValueError("The nominal extension does not reproduce original 10k training")
            records.append({"family":family,"training_seed":seed,"steps":30000,"updates":29001,
                "path":str(folder/"policy.pt"),"source_path":str(policy),
                "sha256":hashlib.sha256((folder/"policy.pt").read_bytes()).hexdigest(),
                "training_distribution":meta.get("training_distribution","nominal")})
    write_json(root/"models.json",records)
    return records


def feedback_specs(base):
    return [{"method":f"feedback_{i}","family":"feedback","parameters":
             {"buy_fraction":b,"sell_fraction":s,"gain":g}}
            for i,(b,s,g) in enumerate(itertools.product(base["target_buy_fractions"],
                base["target_sell_fractions"],base["currency_gains"]))]


def evaluate_job(market_values,spec,model,scenarios,seeds):
    torch.set_num_threads(1)
    market = MarketConfig.from_dict(market_values)
    agent = SACAgent.load(model["path"])[0] if model else None
    rows = []
    for scenario in scenarios:
        env = MultiStoreEnv(replace(market,**scenario["changes"]))
        feedback = CurrencyFeedbackPolicy(env,**spec["parameters"])
        if agent is not None:
            agent.env = env
        policy = BlendPolicy(agent,feedback,spec["weight"]) if agent is not None else feedback
        for seed in seeds:
            rows.append({"scenario":scenario["name"],"method":spec["method"],
                         "training_seed":model["training_seed"] if model else "",
                         "evaluation_seed":seed,**rollout(env,policy,seed)})
        env.close()
    return rows


def evaluate_grid(market,specs,models,scenarios,seeds,workers):
    rows = []
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context("spawn")) as pool:
        jobs = []
        for spec in specs:
            draws = [r for r in models if r["family"]==spec["family"]] if spec["family"]!="feedback" else [None]
            for model in draws:
                jobs.append(pool.submit(evaluate_job,market.to_dict(),spec,model,scenarios,seeds))
        for i,future in enumerate(as_completed(jobs),1):
            rows.extend(future.result())
            if i%10==0 or i==len(jobs):
                print(f"Evaluated {i}/{len(jobs)} policy/model jobs",flush=True)
    si = {s["name"]:i for i,s in enumerate(scenarios)}
    mi = {s["method"]:i for i,s in enumerate(specs)}
    return sorted(rows,key=lambda r:(si[r["scenario"]],mi[r["method"]],str(r["training_seed"]),r["evaluation_seed"]))


def mixture_rows(rows,scenarios):
    names = {s["name"] for s in scenarios}
    groups = {}
    for row in rows:
        key = (row["method"],row["training_seed"],row["evaluation_seed"])
        groups.setdefault(key,[]).append(row)
    combined = []
    for (method,seed,market),group in groups.items():
        if len(group)!=len(names) or {r["scenario"] for r in group}!=names:
            raise ValueError("Validation mixture has missing/duplicate scenarios")
        combined.append({"scenario":"validation_mixture","method":method,"training_seed":seed,
                         "evaluation_seed":market,**{m:float(np.mean([r[m] for r in group])) for m in METRICS}})
    return combined


def select_operating_points(summaries,specs,feedback_method,caps,tolerance):
    stats = {r["method"]:r for r in summaries}
    points,matches = [],[]
    feedback = [s for s in specs if s["family"]=="feedback"]
    for family in ("feedback","nominal","randomized"):
        candidates = [s for s in specs if s["family"]==family]
        if family!="feedback":
            candidates += [{"method":feedback_method,"family":family,"weight":1.}]
        for cap in caps:
            feasible = [s for s in candidates if stats[s["method"]]["mean_gold_deviation_mean"]<=cap]
            choices = feasible or candidates
            selected = (max(choices,key=lambda s:(stats[s["method"]]["total_volume_mean"],
                                                  -stats[s["method"]]["mean_item_price_change_mean"])) if feasible
                        else min(choices,key=lambda s:(stats[s["method"]]["mean_gold_deviation_mean"],
                                                       -stats[s["method"]]["total_volume_mean"])))
            value = stats[selected["method"]]
            point = {"family":family,"cap":cap,"method":selected["method"],
                     "weight":selected.get("weight"),"validation_feasible":bool(feasible),
                     "validation_error":value["mean_gold_deviation_mean"],
                     "validation_volume":value["total_volume_mean"]}
            points.append(point)
            if family!="feedback":
                comparable = [s for s in feedback if abs(stats[s["method"]]["mean_gold_deviation_mean"]-
                                                        point["validation_error"])<=tolerance]
                partner = max(comparable,key=lambda s:(stats[s["method"]]["total_volume_mean"],
                                                       -stats[s["method"]]["mean_item_price_change_mean"])) if comparable else None
                matches.append({**point,"feedback_method":partner["method"] if partner else None,
                    "validation_error_gap":point["validation_error"]-stats[partner["method"]]["mean_gold_deviation_mean"] if partner else None})
    return points,matches


def pareto_front(points):
    def dominates(a,b):
        av = (a["mean_gold_deviation_mean"],-a["total_volume_mean"],a["mean_item_price_change_mean"])
        bv = (b["mean_gold_deviation_mean"],-b["total_volume_mean"],b["mean_item_price_change_mean"])
        return all(x<=y for x,y in zip(av,bv)) and any(x<y for x,y in zip(av,bv))
    return [r for r in points if not any(dominates(other,r) for other in points if other is not r)]


def feedback_cache_signature(config,market,specs):
    digest = hashlib.sha256()
    for name in ("rl/environment.py","rl/sac.py","rl/config.py","rl/rewards.py","experiments/evaluate.py"):
        digest.update(Path(name).read_bytes())
    digest.update(inspect.getsource(evaluate_job).encode())
    digest.update(inspect.getsource(evaluate_grid).encode())
    return {"config":config,"market":market.to_dict(),"specs":specs,"evaluation_source":digest.hexdigest()}


def feedback_validation(root,workers):
    config = json.loads((root/"protocol.json").read_text())
    market,_,base = validate_protocol(config)
    specs = feedback_specs(base)
    if (root/"validation-feedback.csv").exists():
        raise FileExistsError("Feedback validation already exists")
    rows = evaluate_grid(market,specs,[],config["validation_scenarios"],config["validation_seeds"],workers)
    write_csv(root/"validation-feedback.csv",rows)
    metadata = feedback_cache_signature(config,market,specs)
    metadata["data_sha256"] = hashlib.sha256((root/"validation-feedback.csv").read_bytes()).hexdigest()
    write_json(root/"validation-feedback-metadata.json",metadata)


def validate(root,workers):
    config = json.loads((root/"protocol.json").read_text())
    recorded = json.loads((root/"protocol-record.json").read_text())
    if hashlib.sha256(Path(recorded.get("input_path","experiments/configs/robustness.json")).read_bytes()).hexdigest()!=recorded["input_sha256"]:
        raise ValueError("The declared protocol changed after it was recorded")
    market,sac,base = validate_protocol(config)
    if (root/"frozen-selection.json").exists():
        raise FileExistsError("The validation decision is already frozen")
    models = stage_models(root,config,market,sac)
    write_json(root/"config.json",config)
    feedback = feedback_specs(base)
    if not (root/"validation-feedback.csv").exists():
        feedback_validation(root,workers)
    metadata = json.loads((root/"validation-feedback-metadata.json").read_text())
    digest = metadata.pop("data_sha256")
    if metadata!=feedback_cache_signature(config,market,feedback) or digest!=hashlib.sha256((root/"validation-feedback.csv").read_bytes()).hexdigest():
        raise ValueError("Cached feedback validation source/config/data differs")
    baseline_rows = read_csv(root/"validation-feedback.csv")
    stats = summarize(mixture_rows(baseline_rows,config["validation_scenarios"]))
    chosen = max(stats,key=lambda r:r["common_return_mean"])["method"]
    parameters = next(s["parameters"] for s in feedback if s["method"]==chosen)
    learned = [{"method":f"{family}_w{weight:g}","family":family,"weight":weight,"parameters":parameters}
               for family in ("nominal","randomized") for weight in config["feedback_weights"] if weight<1]
    rows = baseline_rows+evaluate_grid(market,learned,models,config["validation_scenarios"],config["validation_seeds"],workers)
    write_csv(root/"validation-episodes.csv",rows)
    summaries = summarize(mixture_rows(rows,config["validation_scenarios"]))
    write_csv(root/"validation-summary.csv",summaries)
    specs = feedback+learned
    points,matches = select_operating_points(summaries,specs,chosen,config["currency_caps"],config["matching_tolerance"])
    decision = {"test_episodes_seen":0,"feedback_method":chosen,"feedback_parameters":parameters,
                "operating_points":points,"matched_comparisons":matches,"candidate_specs":specs,"models":models,
                "selection":"Equal-weight validation mixture only; volume-maximizing points under fixed error caps"}
    snapshot_research(root)
    write_json(root/"replay-entrypoint.json",{"module":"experiments.robustness"})
    write_json(root/"frozen-selection.json",decision)
    print(f"Frozen feedback {chosen}: {parameters}; {len(points)} operating points, {len(matches)} matches",flush=True)


def test(root,workers):
    config = json.loads((root/"config.json").read_text())
    market,_,_ = validate_protocol(config)
    decision = json.loads((root/"frozen-selection.json").read_text())
    if decision["test_episodes_seen"]!=0 or (root/"test-episodes.csv").exists():
        raise FileExistsError("Refuse to overwrite a frozen test evaluation")
    for name,digest in json.loads((root/"research-source-hashes.json").read_text()).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise ValueError("Use the archived source snapshot after maintenance")
    rows = evaluate_grid(market,decision["candidate_specs"],decision["models"],config["test_scenarios"],
                         config["evaluation_seeds"],workers)
    write_csv(root/"test-episodes.csv",rows)
    summaries = summarize(rows)
    write_csv(root/"summary.csv",summaries)
    write_json(root/"summary.json",summaries)
    differences,rng = [],np.random.default_rng(config["bootstrap_seed"])
    for scenario in config["test_scenarios"]:
        sc = scenario["name"]
        for proposed,reference in (("nominal_w0",decision["feedback_method"]),
                                   ("randomized_w0",decision["feedback_method"]),("randomized_w0","nominal_w0")):
            a = [r for r in rows if r["scenario"]==sc and r["method"]==proposed]
            b = [r for r in rows if r["scenario"]==sc and r["method"]==reference]
            for metric in ("common_return","mean_gold_deviation","total_volume","mean_item_price_change"):
                differences.append({"scenario":sc,"comparison":proposed+"_minus_"+reference,
                    **paired_interval(a,b,metric,rng,config["bootstrap_resamples"])})
    write_csv(root/"differences.csv",differences)
    matched = []
    lookup = {(r["scenario"],r["method"]):r for r in summaries}
    for point in decision["matched_comparisons"]:
        for scenario in config["test_scenarios"]:
            sc = scenario["name"]
            a = lookup[(sc,point["method"])]
            partner = point["feedback_method"]
            b = lookup[(sc,partner)] if partner else None
            gap = a["mean_gold_deviation_mean"]-b["mean_gold_deviation_mean"] if b else None
            common = {"scenario":sc,"family":point["family"],"cap":point["cap"],"method":point["method"],
                "feedback_method":partner,"learned_action_used":point["weight"]!=1,
                "validation_feasible":point["validation_feasible"],"test_constraint_met":a["mean_gold_deviation_mean"]<=point["cap"],
                "test_error_gap":gap,"test_within_tolerance":abs(gap)<=config["matching_tolerance"] if gap is not None else False}
            for metric in ("total_volume","mean_item_price_change"):
                interval = (paired_interval([r for r in rows if r["scenario"]==sc and r["method"]==point["method"]],
                    [r for r in rows if r["scenario"]==sc and r["method"]==partner],metric,rng,config["bootstrap_resamples"])
                    if partner else {"metric":metric,"difference":None,"exploratory_95_low":None,"exploratory_95_high":None})
                matched.append({**common,**interval})
    write_csv(root/"matched-comparisons.csv",matched)
    operating = []
    for point in decision["operating_points"]:
        for scenario in config["test_scenarios"]:
            value = lookup[(scenario["name"],point["method"])]
            operating.append({"scenario":scenario["name"],**point,
                "learned_action_used":point["family"]!="feedback" and point["weight"]!=1,
                "test_constraint_met":value["mean_gold_deviation_mean"]<=point["cap"],
                **{m:value[m+"_mean"] for m in METRICS}})
    write_csv(root/"operating-points.csv",operating)
    fronts = [{"scenario":sc,"method":r["method"]} for sc in (s["name"] for s in config["test_scenarios"])
              for r in pareto_front([r for r in summaries if r["scenario"]==sc])]
    write_csv(root/"pareto-fronts.csv",fronts)
    subgroups = []
    for scenario in config["test_scenarios"]:
        for label,seeds in (("prior_three",[0,1,2]),("added_two",[3,4])):
            selected = [r for r in rows if r["scenario"]==scenario["name"] and r["method"]=="nominal_w0" and r["training_seed"] in seeds]
            subgroups.extend({**r,"model_subset":label} for r in summarize(selected))
    write_csv(root/"confirmation-subsets.csv",subgroups)
    model_rows = []
    for family in ("nominal","randomized"):
        for seed in config["training_seeds"]:
            for scenario in config["test_scenarios"]:
                group = [r for r in rows if r["method"]==family+"_w0" and r["training_seed"]==seed
                         and r["scenario"]==scenario["name"]]
                model_rows.append({"scenario":scenario["name"],"family":family,"training_seed":seed,
                                   **{m:float(np.mean([r[m] for r in group])) for m in METRICS}})
    write_csv(root/"per-model.csv",model_rows)
    plot(root,summaries,decision)
    write_json(root/"completion.json",{"test_episodes":len(rows),"market_steps":len(rows)*market.max_steps,
        "models":len(decision["models"]),"summary_rows":len(summaries),"test_scenarios":len(config["test_scenarios"]),
        "interpretation":"Exploratory follow-up; new initial markets; covered and unseen response centers reported separately"})


def plot(root,summaries,decision):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = [(decision["feedback_method"],"Feedback","#8a929b"),("nominal_w0","Nominal SAC","#5672ac"),
               ("randomized_w0","Randomized SAC","#237d65")]
    scenarios = list(dict.fromkeys(r["scenario"] for r in summaries))
    fig,axes = plt.subplots(1,2,figsize=(13,5.2),layout="constrained")
    for index,(method,label,color) in enumerate(methods):
        rows = [next(r for r in summaries if r["scenario"]==s and r["method"]==method) for s in scenarios]
        for ax,metric,factor in ((axes[0],"common_return",1),(axes[1],"mean_gold_deviation",100)):
            ax.bar(np.arange(len(scenarios))+(index-1)*.25,[r[metric+"_mean"]*factor for r in rows],width=.25,
                   label=label,color=color)
    axes[0].set(title="Common balanced score — higher",ylabel="Mean return")
    axes[1].set(title="Mean absolute currency deviation — lower",ylabel="Percent")
    for ax in axes:
        ax.set_xticks(range(len(scenarios)),[s.replace("_"," ") for s in scenarios],rotation=35,ha="right")
        ax.grid(axis="y",alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Five seeds per SAC training condition; 30k steps; point estimates")
    fig.savefig(root/"generalization.png",dpi=170)
    plt.close(fig)
    centers = []
    for model in decision["models"]:
        if model["family"]=="randomized":
            with (Path(model["path"]).parent/"training.csv").open() as stream:
                centers.extend(float(r["response_center"]) for r in csv.DictReader(stream))
    fig,ax = plt.subplots(figsize=(8,3.5),layout="constrained")
    ax.hist(centers,bins=20,color="#237d65",alpha=.75)
    for center in (.6,.8,1.,1.2,1.4):
        ax.axvline(center,color="#b15b4e" if center in (.6,1.4) else "#697a8a",linestyle="--",linewidth=1)
    ax.set(xlabel="Episode response center (player sensitivities are center ± 0.3)",ylabel="Training episodes",
           title="Randomized training covers centers 0.8–1.2; tests also include 0.6 and 1.4")
    fig.savefig(root/"training-response-centers.png",dpi=170)
    plt.close(fig)
    specs = decision["candidate_specs"]
    for scenario in scenarios:
        fig,axes = plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
        for family,label,color,marker in (("feedback","Feedback grid","#929ba5","s"),
                                          ("nominal","Nominal SAC blends","#5672ac","o"),
                                          ("randomized","Randomized SAC blends","#237d65","^")):
            selected = [r for r in summaries if r["scenario"]==scenario and
                        next(s["family"] for s in specs if s["method"]==r["method"])==family]
            for ax,metric,factor in ((axes[0],"total_volume",.001),(axes[1],"mean_item_price_change",100)):
                ax.scatter([100*r["mean_gold_deviation_mean"] for r in selected],
                           [factor*r[metric+"_mean"] for r in selected],label=label,color=color,marker=marker,
                           alpha=.65 if family=="feedback" else .95,s=30 if family=="feedback" else 65)
        for ax in axes:
            ax.set(xlabel="Mean absolute currency deviation (%) — lower")
            ax.grid(alpha=.2)
        axes[0].set(ylabel="Units traded (thousands) — higher")
        axes[1].set(ylabel="Mean per-item relative price change (%) — lower")
        axes[0].legend(fontsize=8)
        fig.suptitle(scenario.replace("_"," ")+": fixed candidate grid, descriptive test trade-offs")
        fig.savefig(root/f"tradeoffs-{scenario}.png",dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("init","feedback","validation","test"),required=True)
    parser.add_argument("--config",default="experiments/configs/robustness.json")
    parser.add_argument("--out",required=True)
    parser.add_argument("--workers",type=int,default=3)
    args = parser.parse_args()
    if not 1<=args.workers<=3:
        parser.error("Use 1–3 workers")
    if args.phase=="init":
        initialize(Path(args.out),args.config)
    elif args.phase=="feedback":
        feedback_validation(Path(args.out),args.workers)
    elif args.phase=="validation":
        validate(Path(args.out),args.workers)
    else:
        test(Path(args.out),args.workers)


if __name__ == "__main__":
    main()
