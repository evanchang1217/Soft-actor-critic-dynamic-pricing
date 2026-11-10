# Planning and computation

A separate two-good task economy tests persistent inventories, shared production
and changing complementary demand. Its reward differs from the original market.
Nine fixed SAC policies are compared with PI and six online-planning budgets on
four fresh markets in each of two cases. H is planning horizon and F is the number
of sampled futures.

Rewards below are means over evaluation periods **97–384**, after the initial
96-period window. CPU is measured separately on common public market states and
histories: decision time plus amortized observation/model-refitting time. It
excludes policy loading, initial model fitting and SAC's offline training.

| Method | Default market reward ↑ | Changed market reward ↑ | Online CPU, ms/period ↓ |
| --- | ---: | ---: | ---: |
| PI | −0.3572 | −0.3757 | 0.13 |
| SAC | −0.3203 | −0.4088 | 0.32 |
| Planner, H4/F2 | −0.3671 | −0.7205 | 6.89 |
| Planner, H12/F2 | −0.3643 | −0.4098 | 13.49 |
| Planner, H12/F64 | −0.3307 | −0.3701 | 105.70 |

SAC outperforms PI in the default case and the H4/F2 planner in both cases under
exploratory paired intervals. Superiority over the strongest planner is **not
established**; PI and strong planning have better mean reward in the changed
market. Three exact-setting replays measured about 399 CPU seconds per SAC training;
PI is faster online, so inference speed alone is not total cost.

The changed market alters demand and production assumptions from the beginning
of evaluation; it is not an unannounced mid-episode change. Timings are local measurements, not
universal benchmarks. [All six budgets](../diagnostics/2026-11-09-compute-tradeoff/summary.csv) ·
[Paired intervals](../diagnostics/2026-11-09-compute-tradeoff/contrasts.csv) ·
[Model-estimation results](../diagnostics/2026-11-09-model-error-study/summary.csv) ·
[Online refitting results](../diagnostics/2026-11-09-adaptive-planning/summary.csv).
