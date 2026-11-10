# Study protocols

This page consolidates the study-specific narrative guides. The linked resolved
protocols, interpretation plans, frozen source and complete comparisons define
exact settings. Follow-ups were informed by earlier exploration; they are not
externally preregistered. Seeds and comparisons were fixed within their declared
runs. Scores from the two market families are not pooled.

[Execution and archive access](experiments.md) · [Equations and scope](methodology.md).

| Study | Resolved inputs |
| --- | --- |
| Objective design | [Config](../results/reevaluation/2026-11-04-balanced-control/config.json) |
| Original-market robustness | [Protocol](../results/confirmation/2026-11-05-robustness/protocol.json) |
| Behavioral assumptions | [Protocol](../results/confirmation/2026-11-06-behavior-audit/protocol.json) |
| Independent mechanisms | [Protocol](../results/confirmation/2026-11-07-mechanism-audit/protocol.json) |
| Representation/transfer | [Protocol](../results/confirmation/2026-11-09-stationary-retraining/protocol.json) |
| Shared-data estimation | [Protocol](../results/diagnostics/2026-11-09-model-error-study/protocol.json) |
| Online refitting | [Protocol](../results/diagnostics/2026-11-09-adaptive-planning/protocol.json) |
| Quality/CPU | [Protocol](../results/diagnostics/2026-11-09-compute-tradeoff/protocol.json) |

## Objective design

Compare legacy, zero-currency and balanced objectives with five seeds and the
same 10,000-interaction budget. The individual-player market uses three goods,
200 players, 100 periods and fixed 5% fees. Fixed-fee actors remove the fee action
from learning and append zero during execution. All methods receive a declared
common balanced evaluation score; native returns are not mixed.

Ten validation markets select controller settings before forty held-out markets
and declared response/resource scenarios. Currency tracking uses deviation from
initial total gold; trading targets and price-change penalties retain explicit
weights. Validation tuning is separate from test evaluation. Report every
objective, tuned comparator and trade-off, not only the improved currency metric.
The 88.9% currency-error reduction is versus legacy SAC, not versus all feedback.

## Original-market robustness

Five policies per nominal/randomized SAC condition train for 30,000 interactions.
Twenty validation markets under three response conditions fix feedback/blend
operating points before forty new markets under six scenarios. Episodes retain
three goods, 200 players, 100 periods and fixed fees. Price-response multipliers
are simulator parameters, not estimated economic elasticities.

Report common reward, currency error, units and per-item sell-price changes.
A score-selected controller and activity/error-matched operating points answer
different questions. The action blend is a separately labeled inference rule.
Crossed bootstrap intervals use training-model and paired-market units; few seeds,
selection uncertainty and unadjusted multiple comparisons limit conclusions.
Nominal score gains do not establish currency, welfare or universal dominance.

## Behavioral assumptions

A 2x2 model audit crosses participation/quantity coupling with an inventory-demand
bundle. That bundle adds 5% stochastic consumption and purchase participation
scaled by `20/(20+inventory)`. Separated quantity removes the repeated response
multiplier from conditional Poisson quantities. These are illustrative rules.

Five 30k policies per model compare with independently validation-tuned P/PI.
Ten original/randomized models are reused; fifteen further models train in the
other behavior cells. Structural transfer distinguishes frozen original controls
from target-tuned controls. Persistent evaluations use 100/500/1,000 periods and
declared shock variations without resetting holdings mid-market.

Report score, currency, trading, both price sides and individual-balance metrics.
Low balances and Gini are diagnostics, not calibrated welfare. Intervals resample
models and markets, never individual players. Satiation-related ranking reversals
remain visible. Selected archived verification replays 72 episodes.

## Independent mechanisms

Cross consumption off/on with satiation off/on while preserving the original
conditional-quantity rule. Consumption removes 5% of held units before mining;
satiation multiplies purchase participation once by `20/(20+inventory)`.

Neither/both reuse ten policies; the single-mechanism cells train ten more at the
same five-seed 30k budget. Each cell tunes P and PI independently on ten fresh
validation markets and tests twenty new markets. Report within-cell gaps,
conditional mechanism effects and interactions, with all reused models declared.

At this budget both satiation cells favor feedback. Consumption alone retains a
SAC/P gain with uncertain SAC/PI ranking. The factorial identifies sensitivity
of this simulator comparison, not whether insufficient learning or information
causes the loss. Frozen verification reconstructs tables and replays 28 episodes.

## Representation and transfer

The separate aggregate economy has two complementary goods, public recipe
changes, pooled cash, shared production and persistent storage. It uses its own
service/currency/holding/price objective and a two-frame public observation.

Five new paired seeds train raw and stationary-input SAC for 24,576 interactions
each, retaining 12,288/24,576 checkpoints. Translation removes historical ledger
growth cues from inputs while keeping actual economics, money and reward fixed.
Deployment-only translation is a separate comparator. The primary endpoint is
stationary minus raw later reward, periods 97–384, on eight held-out markets;
four other markets tune baselines.

Nine fixed regimes vary recipe-switch probability, warehouse size and production
capacity. Policies transfer without retraining. Compare P/PI, fixed anchors,
recipe-aware control and finite planners. Planners search 82 first actions with
hold/feedback continuations; nominal supplements increase sampled futures or
horizon. No planner is an optimal upper bound. Scarce-production failures remain.
Matched-state diagnostics do not establish uniquely better learned anticipation.

## Shared-data model estimation

SAC and planning receive the same three logging datasets and fitted simulator.
Nested samples of 48/384 public transitions fit demand slope, production
reservation price and recipe-switch probability. Functional equations, noise
laws and other constants are shared prior knowledge. Forward least squares
accounts for inventory/cash censoring and is not an exact likelihood.

Each fitted model trains three SAC seeds for 24,576 simulated interactions:
eighteen policies in total. Extra simulated SAC data are counted. Frozen fitted
controllers transfer to five cases: nominal, stronger price response, higher
production reservation, faster recipe switching and their combination. Parameters
change from evaluation start, not at an unknown point within the episode.

Four fresh 384-period markets retain later reward, service and currency error.
The declared primary nominal/larger-data SAC–SAA8 interval crosses zero.
Production shifts favor SAC over stale-model planning; correct-parameter planning
and PI constrain broader claims. SAA64 supplements remain on nominal/combined
cases. All fits, models, cases and pointwise exploratory intervals are retained.

## Periodic online refitting

Reuse all nine larger-data SAC policies with fixed weights and no new performance
training. The same five cases use four new markets. Planning starts parameter
updates after 32 observations, then every 16 periods, retaining at most 64.
Identifiability gates require enough nonbinding trade observations; insufficiently
identified production/demand parameters keep their previous values. Only public
past transitions enter estimates; no actual future draws or true parameters do.

Freeze the 82-action menu, continuations, objective and H12/F8 planning; retain
H12/F64 supplements. Compare pre-update, adaptation, later and full windows.
Online fitting repairs much of stale-model loss. Sustained SAC superiority over
the updated strong planner is not established. Early advantages depend on the
chosen estimation warmup. Planner memory and model-family knowledge differ from
SAC's two-frame input and offline training, so this is a pipeline comparison.

## Quality and computation

Nine unchanged policies, two cases and four new paired markets yield 224 episodes.
All six online-planning budgets remain: H1/F2, H4/F2, H4/F8, H12/F2, H12/F8 and
H12/F64. Every planner uses the same periodic estimator. PI remains the cheap
reference; SAC is not presented as the fastest controller.

Quality comes from full independent trajectories. CPU is measured separately on
24 common PI-visited states/histories, with one Torch thread, one warmup and three
measurements. The 2,016 decision and 216 observation samples include update/non-
update positions. Observe/refit cost is amortized at the actual 1/16 update rate,
not the balanced audit sampling ratio. Implementation and hardware are recorded.

Three exact-setting timed training replays retain identical original policy
weights and curves. Their 73,728 interactions are reproduction/cost work, not
performance improvement or policy selection. Original historical timing was
unavailable. CPU break-even divides one-policy training by per-decision savings,
conditional on acceptable quality, reuse and no retraining. PI has no crossover
because it is faster. This is not monetary ROI or real-game benefit.

Later reward favors SAC over nominal PI and some limited-budget planners.
Strong-planner score rankings remain uncertain, and PI/strong planning have better
combined-shift means. Report service/currency trade-offs, all windows, every
budget and uncertainty. Point-estimate frontiers do not prove dominance or
statistical equivalence. The common-state probe is not a shared policy trajectory.
