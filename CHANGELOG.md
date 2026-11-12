# Project changes

## November 2026 — research follow-up

- Studied currency-tracking objectives, equal-budget controls and five-seed
  original-market robustness. Added behavioral and persistent-market analyses,
  including cases where SAC loses to feedback.
- Tested observation representations and transfer in a separate aggregate market,
  then compared SAC, PI and planners under model error, online refitting and
  different computation budgets.
- Consolidated study runners, documentation and tests. Kept summaries, key
  figures and the saved-policy demo in the checkout; detailed evidence and frozen
  sources are in a checksum-verified [evidence release](https://github.com/evanchang1217/Soft-actor-critic-dynamic-pricing/releases/tag/pricing-evidence-2026-11-11).

See the [research results](results/README.md), [methods](docs/methodology.md) and
[reproduction guide](docs/experiments.md) for study scope and limitations.

## October 2026 — reproducible baseline

- Refactored the shared SAC and market code and added held-out evaluation,
  feedback baselines, automated checks and the original Django demo.
