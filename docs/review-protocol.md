# Stronger comparison and reward audit

The first reference run established a working evaluation pipeline. It did not establish that SAC was necessary or that the reward meaningfully controlled inflation. A follow-up protocol adds simple tuned controls, a fee-control ablation, fresh test markets, and explicitly specified distribution shifts. The original poster and first reference run remain unchanged.

## Run the complete protocol

```bash
# Self-contained: train both SAC variants, select simple baselines, then evaluate.
python -m experiments.review --config experiments/configs/review.json --out runs/review

# Reuse the five compatible full-control models from the first reference run.
# This still trains five independent fixed-fee models with the same step budget.
python -m experiments.review --config experiments/configs/review.json \
  --full-checkpoints results/reevaluation/2026-10-28-reference/seed-*/policy.pt \
  --out runs/review-with-reused-models

# A small end-to-end software check, not performance evidence.
python -m experiments.review --config experiments/configs/review-smoke.json --out runs/review-smoke
```

Wildcard expansion above is for Bash/Zsh. PowerShell users should list checkpoint paths explicitly. Existing output directories are refused.

## Separate training, selection, and reporting

The full protocol uses:

- Training initialization seeds `0–4`, with the same 100 episodes / 10,000 steps per model and the same SAC hyperparameters as the first reference run.
- **Validation markets `2000–2009`** to choose a simple adaptive gain and constant target prices. Selection uses mean nominal validation return; ties use the first candidate in the recorded grid.
- **Fresh reporting markets `3000–3019`**, disjoint from validation and all training resets. These were not used by the earlier exploratory comparison or review diagnostic.

The hypothesis and candidate grid were motivated by the earlier review. This is a transparent follow-up, not a preregistered study or an independent external replication. Test outputs must not be used to choose candidates or silently revise the protocol. If further tuning is performed, use another final test set.

`baseline_selection.json` records every candidate's mean validation return and the selected parameters. `validation_rollouts.csv` preserves each candidate/market return. The selection function receives validation seeds only, not reporting seeds.

## Seven strategies

| Method | Decision and fee control |
| --- | --- |
| Fixed | Keep initially sampled prices and fee |
| Random | Uniform bounded changes to prices and fee |
| Adaptive | Original participation rule with gain `0.5`; hold fee |
| Tuned Adaptive | Same rule, gain chosen from `[0.1, 0.25, 0.5, 1.0]` on validation markets; hold fee |
| Target-price rule | Move toward constant buy/sell targets, respecting the same per-step action bounds; hold fee |
| SAC, fixed fee | A separately trained price-only SAC policy; fee action is always zero |
| SAC, price + fee | The full policy, with the same architecture/hyperparameters/budget as the first reference run |

The target-price rule chooses one buy fraction from `[0.5, 0.75, 1.0]` and one sell fraction from `[0.0, 0.25, 0.5]`. Fractions apply to every product's own configured interval:

```text
target_buy  = buy_low  + buy_fraction  * (buy_high  - buy_low)
target_sell = sell_low + sell_fraction * (sell_high - sell_low)
buy_action  = clip(target_buy / observed_buy_price - 1, -price_offset, +price_offset)
sell_action = clip(target_sell / observed_sell_price - 1, -price_offset, +price_offset)
fee_action  = 0
```

Prices still start from the same seeded initial market. The rule cannot jump directly to its targets or inspect future market behavior. Unlike the original Fixed baseline, it is allowed to approach a useful fixed price level.

For **fixed-fee SAC**, the actor output, critic action input, replay action, and entropy target all have `2P` dimensions. There is no unused stochastic fee coordinate contributing entropy. Environment actions append a zero fee delta, including during random warmup. The full policy uses `2P+1` dimensions. Both variants share the remaining model settings and training budget. Changing the available action naturally changes network parameter count and entropy dimensionality; this is the intended action-space ablation.

## Frozen strategies under market shifts

All models are trained in the nominal market. Baseline parameters are selected once on nominal validation markets and reused without retuning. Every strategy is then evaluated on the same twenty seeds in each scenario:

| Scenario | Change from training |
| --- | --- |
| Nominal | Resource shock at step 50 |
| No shock | Disable resource boost |
| Early shock | Move shock to step 25 |
| Late shock | Move shock to step 75 |
| Higher response | Sample player price-response multipliers from `[0.9, 1.5]` instead of `[0.7, 1.3]` |

The response multiplier scales simulated participation probabilities; it is not a fitted economic elasticity estimate. These scenarios test limited specified shifts, not all possible economic changes.

The runner only permits changes to shock timing and response-multiplier bounds. Products, observation encoding, price/action bounds, reward, and horizon stay fixed. Models retain their training-time encoder; no normalization is fitted on shifted test data. The strict general checkpoint/config validation in `experiments.evaluate` remains in place.

## Comparable variability estimates

The original four-method report used across-model SD for SAC and across-market SD for baselines. The follow-up reports both sources separately:

- `*_mean`: average over test markets, averaging trained models within each market first.
- `*_scenario_std`: sample SD across the twenty test-market means, using the **same unit for every method**. This is the uncertainty displayed with the main table/trajectory curves.
- `*_model_std`: sample SD of each model's scenario-averaged result; unavailable (`null` in JSON, blank in CSV) for baselines or a single trained model. Scenario SD is similarly unavailable with only one test market.

The archived follow-up summary retains its original zero placeholders for unavailable baseline model SDs. Those placeholders do not represent measured zero variation.
- `n_test_markets`, `n_training_seeds`, and `n_episodes` record the actual replication counts.

Neither SD is a confidence interval or a significance test. Averaging models within a market is statistical aggregation, not a policy ensemble executing averaged actions. In a single deployment, one model would act. The outputs do not quantify every source of uncertainty or prove that observed differences generalize.

## Reward audit

The legacy reward remains unchanged. `reward_components.csv` and `.json` report the mean weighted penalty and percentage contribution of count, volume, price-change, and money-change terms for every method/scenario. Each logged reward is checked against the sum of its configured components before analysis is written.

To audit any existing evaluation without modifying its files:

```bash
python -m experiments.analyze_reward \
  results/reevaluation/2026-10-28-reference/evaluation \
  --out runs/reference-reward-audit
```

When trade targets dominate the realized penalty, a higher return mainly supports better performance on those targets. It does not establish currency stability or reduced real-game inflation. Changing reward weights requires a separately named experiment; this protocol does not silently replace the original objective.

## Output and provenance

The follow-up output includes the resolved protocol, startup source archive and per-file hashes, package/source manifest, validation trials and selection, both sets of inference checkpoints with training logs, step/episode data, summaries, reward decomposition, and nominal/shift figures.

```text
config.json, manifest.json
research-source.zip, research-source-hashes.json
baseline_selection.json, validation_rollouts.csv
models.json
models/sac/seed-*/...
models/sac_fixed_fee/seed-*/...
steps.csv, episodes.csv
summary.csv, summary.json
reward_components.csv, reward_components.json
comparison.png, comparison.svg
robustness.png, robustness.svg
```

Local steps remain uncompressed. Published follow-ups may store `steps.csv.gz`; plotting and reward-audit commands read it transparently. Regenerate the follow-up figures with `python -m experiments.plot_review <directory>`.

Checkpoint metadata is validated against the expected seed set, fee-control mode, hyperparameters, full training budget, nominal market configuration, and evidence of at least one gradient update before use. New checkpoints record the update count. Older checkpoints can use their matching `outcome.json` as evidence; the published model directories include it. If neither record is available, `--allow-unverified-checkpoint` requires an explicit choice and records the model as unverified. It does not permit a checkpoint known to have zero updates.

Reusing full-control models preserves the first run's weights; their input paths, hashes, and update-verification source are recorded. The new source snapshot is taken **before** selection/training, and source hashes are checked again at completion.
