# Main experiment: original market

The NPC shop changes market-wide prices for three goods while 200 simulated
players trade. The objective combines trade, currency tracking and price stability.
The simulation is not calibrated to real players.

Five SAC policies per condition received 30,000 interactions each. A feedback
controller was chosen from 36 candidates on validation markets; each controller
was then evaluated on 40 held-out market seeds per scenario.

| Controller | Common score ↑ | Currency error ↓ | Units traded | Sell-price change ↓ |
| --- | ---: | ---: | ---: | ---: |
| Validation-selected feedback | −36.01 | 3.39% | 183,370 | 2.15% |
| SAC, default market training | −30.70 | 2.91% | 227,090 | 1.40% |
| SAC, varied player-response training | −34.54 | 3.52% | 257,145 | 1.18% |

The default-trained SAC common-score difference over feedback is **+5.30** with an
exploratory 95% interval of **[3.88, 6.62]**; the currency-error difference
interval includes zero. These scores evaluate the stated simulation objectives,
not real-player welfare.

## Reward-design ablation

This is a separate experiment with **10,000 training interactions per SAC policy**,
five training seeds per objective and 40 held-out markets per scenario. The table
shows the default market. Both rows use SAC; the training objective changes.

| SAC training objective | Mean currency error ↓ | Units traded |
| --- | ---: | ---: |
| Legacy reward | 44.74% | 227,576 |
| Revised currency-tracking reward | 4.96% | 219,047 |

Currency error is the mean absolute deviation of total player currency from its
initial value, expressed as a fraction of that initial value. From unrounded means,
`1 - revised_error / legacy_error` is **88.9%**, and
`revised_units / legacy_units` is **96.3%**. Both percentages use **legacy SAC**
as the reference. They measure the effect of reward design, not SAC's improvement
over feedback. At this 10k budget, the currency-feedback baseline has a better
mean common score than revised SAC. The 30k comparison above is a different study.

## Scenario results

![Performance across response and resource scenarios](../confirmation/2026-11-05-robustness/generalization.png)

[Scenario summary](../confirmation/2026-11-05-robustness/summary.csv) ·
[Paired differences](../confirmation/2026-11-05-robustness/differences.csv) ·
[Objective ablation](../reevaluation/2026-11-04-balanced-control/summary.csv) ·
[Methods](../../docs/methodology.md) · [Full evidence archive](../../docs/experiments.md#full-evidence-archive)
