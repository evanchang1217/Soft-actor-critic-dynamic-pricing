# Documentation

Start with the [project overview](../README.md), then choose the detail you need:

| Question | Guide |
| --- | --- |
| How do I run the demo or connect Minecraft status reads? | [Demo and integration](demo.md) |
| How do the markets, rewards and controllers work? | [Methods and research scope](methodology.md) |
| How do I verify results or run an experiment? | [Experiments and reproduction](experiments.md) |
| What was fixed for each study? | [Study protocols](protocols.md) |
| What do the experiments show? | [Results](../results/README.md) |
| Who did the work and how does it relate to the original poster? | [Project history and authorship](project.md) |

## Architecture

`rl/` contains the original market and shared SAC implementation.
`experiments/market/` contains reusable task-economy physics and controllers;
other modules in `experiments/` run and evaluate studies. `myapp/` hosts the Django
training and saved-policy demo. [Reproduction scripts](../scripts/) use frozen
source and evidence when verifying completed studies.

The saved demo depends on small policies and reference data included in the
checkout. Full study verification uses the published [evidence archive](experiments.md#full-evidence-archive).
Keep new outputs and restored data under
ignored `runs/`. Original poster and figure provenance are in [Poster](../Poster/)
and [assets](assets/README.md).

## Validation

Run the maintained tests and basic application checks with Python 3.10–3.12:

```bash
python -m pytest -q
python manage.py check
node --check statics/js/new_training.js
node --check statics/js/comparison.js
```

Tests cover the market and SAC core, frozen-reference behavior, saved-demo
comparisons, archive safety and maintained study runners. Selected historical
replays verify saved study statistics; they do not validate real-player behavior
or repeat all training. See the [experiment guide](experiments.md) for scope and
archive requirements.

Frozen numerical references use dtype-aware rounding bounds for continuous
values. Fixture hashes and exogenous task counts remain exact. CI also checks
the reference with optional NumPy SIMD kernels disabled.
