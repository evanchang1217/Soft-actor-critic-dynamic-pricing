# Running and interpreting experiments

All commands run from the repository root after installing dependencies. No Django settings or web server are required. Runs are CPU-only by default and output directories must be new, so older evidence is never silently overwritten.

This page documents the original four-method maintenance pipeline. For the stronger comparison with validation-tuned simple rules, fixed-fee SAC, fresh test markets, reward auditing, and explicit market shifts, use [the follow-up protocol](review-protocol.md).

## Complete comparison

```bash
# A short pipeline check: 3 training seeds, 4 episodes each, 5 held-out markets.
python -m experiments.run --config experiments/configs/quickstart.json --out runs/quickstart

# A larger starting protocol: 5 training seeds, 100 episodes each, 10 held-out markets.
python -m experiments.run --config experiments/configs/default.json --out runs/default
```

The quickstart is intentionally too short to establish learning performance. The larger protocol is a starting configuration, not a guarantee of convergence or the original experiment settings. Inspect training curves and repeat with a sufficient budget before drawing conclusions.

The suite trains independent SAC policies and evaluates their deterministic mean actions on held-out markets. Fixed/random/adaptive policies are evaluated once per held-out market, rather than duplicating their observations for every SAC training seed. Evaluation never updates policy weights or normalization.

## Separate training, evaluation, and plotting

```bash
python -m experiments.train --config experiments/configs/default.json --seed 0 --out runs/model-0

python -m experiments.evaluate --config experiments/configs/default.json \
  --checkpoints runs/model-0/policy.pt --out runs/evaluation-0

# Run only inexpensive baselines, with no trained checkpoint.
python -m experiments.evaluate --config experiments/configs/default.json \
  --methods fixed random adaptive --out runs/baselines

# Rebuild the exact figures from saved step CSVs; no model or training needed.
python -m experiments.plot_results runs/evaluation-0
```

Pass multiple model paths after `--checkpoints` to compare several SAC training seeds. Checkpoint market configuration must match the evaluation configuration, including horizon and shock settings. To study distribution shifts, introduce an explicit protocol change rather than silently evaluating a mismatched model.

Models are only pooled when they share the same fee-control mode, SAC training settings, training-step budget, and known update count. A price-only model is labeled `sac_fixed_fee` in results and figures; do not mix it with fee-controlling SAC in this four-method entry point. Use `experiments.review` to compare those variants explicitly. Per-checkpoint settings, update counts, and verification sources are saved in the evaluation configuration.

New checkpoints record the actual number of gradient updates. Evaluation rejects known zero-update models: interacting with the market does not by itself train the policy. Older checkpoints lack that field; keep their matching `outcome.json` alongside `policy.pt` so the update count can be verified. If an older checkpoint has no update evidence, it is rejected unless `--allow-unverified-checkpoint` is explicitly supplied; that choice produces a warning and marks the model unverified in the results. This override never accepts a checkpoint known to have zero updates.

## Baseline definitions

These are new, readable baselines inspired by the project problem. The poster's exact Random/Adaptive implementations were not recovered.

| Method | Decision at each market step |
| --- | --- |
| Fixed | Zero action; hold each episode's initial sampled prices and fee |
| Random | Independently uniform actions within each dimension's declared bounds; policy RNG is separate from the market RNG |
| Adaptive | After the first step, use previous directional participation: `buy_delta = -gain*price_offset*(sellers-N/2)/(N/2)` and `sell_delta = gain*price_offset*(buyers-N/2)/(N/2)`, clipped to action bounds; hold fee fixed |
| SAC | Deterministic mean of the trained tanh-Gaussian policy, scaled per dimension; controls prices and fee |

`adaptive_gain` defaults to `0.5` and is recorded in the protocol. Adaptive does not have the same fee-control flexibility as SAC/random; this comparison alone cannot isolate whether gains come from learned pricing or fee control. The separate [follow-up protocol](review-protocol.md) implements a genuinely price-only SAC model and tuned simple controls to examine that limitation.

## Seeds and comparable conditions

`training_seeds` control network initialization and independent replay/exploration streams. Training episode `e` resets the market with `seed + 10000*e` (one-based episodes). Every policy is evaluated on the same `evaluation_seeds`, with the same horizon, initial distributions, resource shock, and reward.

Protocol validation rejects evaluation seeds used in training. Checkpoint metadata is checked as well. Resource randomness is separate from policy and trading randomness. Same-seed initial markets and resource paths match across methods; policy-dependent trade draws are not guaranteed to be common-random-number paired.

Evaluation seeds are for reporting. If you tune hyperparameters against them, reserve a separate final test set before making performance claims.

## Artifacts

```text
runs/example/
  config.json              # Fully resolved protocol
  manifest.json            # Timestamp, versions, Git revision/dirty state, source digest
  training.png             # Across-seed learning curves
  seed-0/
    config.json
    manifest.json
    training.csv           # Per-episode returns, market metrics, update counts/losses
    outcome.json
    policy.pt              # Inference weights and compatible environment settings
  seed-1/ ...
  evaluation/
    config.json            # Methods and input checkpoint hashes
    manifest.json
    episodes.csv           # One held-out rollout per row
    steps.csv              # Every market step, including all reward components
    summary.csv
    summary.json
    comparison.png
    comparison.svg
```

`manifest.json` fingerprints Python source files as well as the Git state, so an uncommitted checkout is not falsely described as an exact clean revision. Keep the source checkout and its configuration with a published run. Generated `runs/` directories are ignored by Git; intentionally publish selected artifacts under `results/` with a provenance note.

## Aggregation and interpretation

- **SAC:** average held-out markets within each training seed, then report the mean and sample standard deviation across independently trained seeds.
- **Baselines:** report the mean and sample standard deviation across held-out market seeds.
- `n_units`, `n_episodes`, and `aggregation_unit` identify those different sources of variability in the summary. They are not interchangeable confidence intervals. With fewer than two units, sample SD is unavailable: it is `null` in JSON, blank in CSV, and labeled unavailable in command-line output.
- Comparison plots use the same hierarchy at each timestep. Shading is ±1 sample SD, not a confidence band or significance test. It does not separately quantify SAC scenario variance.
- `cpi_std` is within-episode price-index variability. The other summary SD fields describe across-unit variation. Fixed prices have zero within-episode price variability by construction; that does not establish good economic control.
- Review returns alongside total/final currency, volume, and participation. The reward is mostly driven by trade-target deviations with default weights.
- Include the number of steps and updates: a checkpoint saved before learning starts is a pipeline artifact, not evidence of a trained strategy.

The repository includes original poster figures and separately labeled new runs. Never relabel a new run as an original capstone result, and never infer raw historical measurements by digitizing a figure without explicitly describing that process.
