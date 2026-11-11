# RL baseline comparison

This supplementary comparison uses the original three-good synthetic market:
200 players, 100 periods and a fixed 5% fee. Five training seeds per RL method
receive **30,000 interactions each**, with width-128 networks. Evaluation uses
20 market seeds per case, shared by all methods. The 25 trained policies and
two feedback controls produce 1,620 evaluation episodes.

The common score combines trading targets, currency tracking and price stability;
higher is better. Scores are episode totals and use the original-market objective.

| Method | Default market ↑ | Weaker price response ↑ | Reduced supply ↑ |
| --- | ---: | ---: | ---: |
| SAC, repository implementation | **−30.81** | −39.22 | **−30.99** |
| SAC, Stable-Baselines3 implementation | −32.60 | −40.08 | −32.69 |
| PI | −33.67 | **−35.37** | −33.73 |
| P | −36.01 | −46.13 | −36.38 |
| TD3 | −36.24 | −46.09 | −36.94 |
| PPO | −37.36 | −51.11 | −36.97 |
| DQN, discretized actions | −67.84 | −62.73 | −68.34 |

Both SAC rows use the same algorithm. The repository implementation is in
[`rl/sac.py`](../../rl/sac.py); the second uses Stable-Baselines3 2.7.1 as an
implementation cross-check. Their default-case difference is +1.78 with an
exploratory 95% interval of [−0.33, 4.13], so superiority over SB3 SAC is not established.

## Default-market differences

Repository SAC minus each comparator, using crossed model/market bootstrap
intervals. These are exploratory 95% intervals without multiplicity adjustment.

| Comparator | SAC score difference ↑ | Interval |
| --- | ---: | --- |
| PI | +2.85 | [1.55, 4.05] |
| P | +5.19 | [3.89, 6.40] |
| TD3 | +5.42 | [3.15, 7.78] |
| PPO | +6.54 | [5.09, 8.66] |
| DQN | +37.03 | [25.88, 50.92] |

SAC leads under these default-market presets and budget; PI leads under weaker
price responses. The results do not establish a universal algorithm ranking.

## Settings and evidence

- This is a retrospective extension of the earlier behavioral study, reusing its
  market, SAC settings, five training seeds and 20 previously used test markets.
  The repository SAC policies reproduce the historical policy tensors and default
  mean score. This is not a fresh confirmatory test.
- P/PI parameters remain frozen from earlier validation. RL presets are fixed
  without matched hyperparameter searches. Equal interactions do not imply equal
  compute, gradient updates or convergence.
- SAC, TD3 and PPO use six continuous price controls. DQN uses 729 joint actions
  from a three-move grid per control, so its action resolution differs. PPO uses
  1,000-step rollouts and 125-sample minibatches to meet the exact budget.

[Configuration](../../experiments/configs/rl-baselines.json) ·
[Runner](../../experiments/rl_baselines.py) ·
[Reproduction instructions](../../docs/experiments.md#optional-rl-baseline-comparison) ·
[All metrics](summary.csv) · [All paired differences](contrasts.csv).

The [evidence index](evidence.json) records source/data checksums and the separate
full-record archive, which includes all policies, training logs, episode metrics
and frozen source. Download it from the [evidence release](https://github.com/evanchang1217/Soft-actor-critic-dynamic-pricing/releases/tag/pricing-evidence-2026-11-11).
