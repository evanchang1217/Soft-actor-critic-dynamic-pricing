# Results and provenance

| Directory | Evidence |
| --- | --- |
| [`original/`](original/) | Directly extracted original poster comparisons and the old standalone demo reward image |
| [`reevaluation/2026-10-28-reference/`](reevaluation/2026-10-28-reference/) | Initial exploratory four-method comparison: 5 training seeds / 10 held-out markets, with raw data and models |
| [`reevaluation/2026-10-28-review/`](reevaluation/2026-10-28-review/) | Follow-up: validation-selected simple controls, true fixed-fee SAC, 20 fresh test markets, 5 specified scenarios, reward audit, 10 inference models, and compressed raw steps |

New evaluations must be stored separately from historical artifacts. Each published run should include its resolved settings, manifests, raw step/episode data, training logs, and an explanation of what conclusions the run can support. Local experiment outputs and model checkpoints are saved under ignored `runs/` directories.

See [the experiment protocol](../docs/experiments.md) for commands and aggregation definitions. A smooth learning curve or a short smoke run does not establish superiority over a baseline.
