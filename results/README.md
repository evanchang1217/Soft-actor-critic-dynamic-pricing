# Results

Start with the question you want to answer:

| Question | Guide |
| --- | --- |
| What did the main SAC and feedback experiment show? | [Main experiment](main/) |
| How does SAC compare with TD3, PPO and DQN at the same interaction budget? | [RL baseline comparison](rl-baselines/) |
| When did SAC fail, and what are the simulation limits? | [Limitations and failure cases](limitations/) |
| How does SAC compare with PI and stronger planning? | [Planning and computation](planning/) |

The original individual-player economy and the later aggregate task economy use
different rewards and populations; their scores cannot be pooled. The dated
`confirmation/`, `diagnostics/` and `reevaluation/` directories retain key
protocols, result tables, figures and saved-demo inputs. Their paths also identify
frozen records used by the reproduction tools.

## Complete evidence

Detailed per-run data, model-training records, audit inputs, source snapshots and
additional checkpoints are in a separate checksum-verified archive, available
in the [evidence release](https://github.com/evanchang1217/Soft-actor-critic-dynamic-pricing/releases/tag/pricing-evidence-2026-11-11). Checkout tables and demo models
remain available. [Inventory and hashes](archive-manifest.json) ·
[Archive access and restoration](../docs/experiments.md#full-evidence-archive).

The later RL baseline comparison has a separate evidence bundle listed in its
[evidence index](rl-baselines/evidence.json); it is not in the historical archive.

SAC is an existing algorithm. These are bounded empirical comparisons in synthetic
markets, not evidence of real-player welfare or a universal advantage over
rule-based methods. [Research scope](../docs/methodology.md#research-scope).
