# Project history and presentation

## Authorship

The original capstone was an individual project under faculty supervision at
National Cheng Kung University. The author designed the pricing formulation and
reward and developed the simulator, SAC system and web demonstration. The author
reports second place among 87 projects; the award concerns that capstone.

Later maintenance and evaluation used AI assistance for portions of coding and
analysis. The contributions use the existing SAC algorithm. They center on
modeling, system implementation, objective design and empirical evaluation.

The [pre-maintenance implementation](https://github.com/evanchang1217/Soft-actor-critic-dynamic-pricing/tree/e092c8684b2e42dd0277d053d38824a194e51aac)
and [original poster](../Poster/capstone-poster.pdf) are historical evidence, separate
from subsequent experiments with their own protocols and source snapshots.

| Stage | Maintained evidence |
| --- | --- |
| Original capstone | [Poster supplement](project.md#original-poster), original figures, server-integration workflow |
| Objective and original-market comparison | [Reward ablation](protocols.md#objective-design), [five-seed comparison](protocols.md#original-market-robustness) |
| Behavioral limits | [Behavior](protocols.md#behavioral-assumptions) and [independent mechanisms](protocols.md#independent-mechanisms), including ranking reversals |
| Representation and transfer | [New-seed retraining](protocols.md#representation-and-transfer), nine regimes and stronger planning |
| Model knowledge and computation | [Estimation](protocols.md#shared-data-model-estimation), [online refitting](protocols.md#periodic-online-refitting), [quality/CPU study](protocols.md#quality-and-computation) |

Earlier prototypes and diagnostics informed these studies. Repository curation
on 2026-11-10 removed superseded standalone experiments and redundant narrative,
while retaining the declared main comparisons, boundary cases and reproducibility
artifacts. A subsequent small refactor extracted common market/controller code
and moved full evidence to a separate verified archive; the checkout keeps
summaries and runnable demo dependencies. [Result scope and limitations](../results/limitations/).

The released Minecraft client reads market/player status. The original server
plugin, scheduled price write-back code, historical dataset/fitting pipeline and
deployed checkpoint are not included. Limited game-server testing is
self-reported without detailed field efficacy measurements. Quantitative
conclusions here concern synthetic markets.

## Original poster

The [original capstone poster](../Poster/capstone-poster.pdf)
records the individual project's pricing system, integration workflow, and historical
experiments. Follow-up work adds reproducible objective ablations, tuned feedback
baselines, and five-seed evaluation. The current descriptive title is **Multi-Product
Game-Market Pricing: SAC and Feedback Control**. The original poster remains unchanged.

**Current contributions:** a multi-product pricing system, tested control-objective
design, and empirical robustness/trade-off evidence. In nominal simulated markets,
five-seed SAC exceeds validation-selected feedback on the common score by +5.30,
exploratory 95% interval [3.88, 6.62]. Currency-control superiority is not universal.
[Five-seed record](../results/main/).

| Historical poster account | Supported scope for the current presentation |
| --- | --- |
| Lower CPI indicates inflation control and stable currency value | The released CPI is a policy-controlled shop-price index. Its decrease alone cannot establish purchasing-power or real-inflation improvement. Current evaluation uses currency-setpoint error, trading, and price changes. |
| Higher reward than random/rule strategies demonstrates effective market regulation | Reward measures chosen targets. The legacy reward is dominated by trade terms; follow-up ablations, stronger feedback, and held-out tests expose both improvements and losses. |
| Logistic regression / Apriori behavior modeling from historical records | Original data and fitting/mining pipelines are not released. Reproducible results use heuristic player behavior; optional item associations are manually supplied. |
| Daily scheduled SAC prices are written back to the game server | The poster describes the project-level workflow. Released code provides status reads; the full write-back script, plugin, and deployed checkpoint are absent. Limited game testing is author-reported without detailed efficacy measurements. |

The exact original experiment settings and raw logs have not been recovered. Follow-up
records have their own dates, configurations, and source snapshots; they do not replace
or reconstruct historical measurements. [Original poster figures](../Poster/capstone-poster.pdf).
The award concerns the original capstone, not the subsequent evaluations.

A concise account is: "The original project built the pricing system and integration
workflow. Subsequent ablations and stronger comparisons established its effective
conditions and limits. Quantitative conclusions concern control in simulated markets."
Original contributions, later work, and AI assistance are distinguished in the
[implementation history](project.md#authorship).

[Research brief](project-summary.pdf) · [Methods](methodology.md).

## Dated main-study brief

The following marked source generates the dated two-page original-market PDF.
Later results are presented in the root README and result index.

<!-- brief:start -->
# Multi-Product Game-Market Pricing: SAC and Feedback Control

**Individual undergraduate capstone | Computer Science, National Cheng Kung University**

**Author: evanchang1217 (GitHub). Original capstone: 2nd place among 87 projects.**

## Research question and contributions

How can an NPC shop adjust multiple buy/sell prices while balancing trading targets,
player-currency tracking, and price stability? This project builds a continuous-control
system and evaluates Soft Actor-Critic (SAC) against validation-tuned feedback rules.
Quantitative evidence comes from a heuristic market simulator.

1. **Build a reproducible pricing system.** Model inventories, budgets, and stochastic
   trades; integrate SAC, a Django demonstration, and Minecraft status API reads.
   The maintained CLI and demo share one core and save configurations, policies, and logs.
2. **Design and test the control objective.** Audit the weak realized currency penalty,
   introduce explicit currency-setpoint tracking, and compare legacy, zero-currency,
   and revised objectives at equal training budgets.
3. **Measure controller trade-offs.** Compare five seeds per SAC condition with
   36 feedback candidates across six scenarios, separating validation from testing
   and retaining unfavorable outcomes and paired uncertainty estimates.

**[Objective ablation](../results/main/):** At equal
10k budgets, revised SAC lowers currency error **88.9% versus legacy SAC**, retaining
**96.3%** of units. Feedback still scores better.

The award concerns the original individual capstone. Follow-up work is separately
dated and used AI assistance for portions of coding and evaluation. The contribution
uses the existing SAC algorithm. [Authorship and history](project.md#authorship).

<!-- pagebreak -->

## Main comparison: five seeds, fixed 30k budget

Each SAC model completes 30,000 interactions. Twenty validation markets under three
response conditions select feedback parameters before testing on 40 new initial
markets under six scenarios. Episodes use 200 players, three goods, 100 steps, and
a fixed 5% fee. The table below reports nominal test markets.

| Method | Common score ↑ | Currency error ↓ | Units traded | Sell-price change ↓ |
| --- | ---: | ---: | ---: | ---: |
| Feedback (validation-tuned) | −36.01 | 3.39% | 183,370 | 2.15% |
| SAC (nominal training) | −30.70 | 2.91% | 227,090 | 1.40% |
| SAC (response-randomized) | −34.54 | 3.52% | 257,145 | 1.18% |

**Supported advantage.** Nominal SAC improves the common score over the score-selected
feedback rule by **+5.30**, exploratory 95% interval **[3.88, 6.62]**. Its lower mean
currency error does not establish an advantage: that difference interval includes
zero. Differences use unrounded values; displayed means need not subtract exactly.

**What the score means.** All methods receive the same negative sum of trading-target,
aggregate shop-price-index-change, and currency-deviation penalties over 100 steps.
Less negative is better. Trading penalties track specified targets; a higher score
need not mean more trading or greater welfare.

## Robustness and the cost of learned pricing

- **Training distribution:** response randomization reduces higher-response SAC
  currency deviation from **9.06% to 3.74%**, versus feedback's **4.75%**, while reducing
  nominal score. Outside-center lower response remains a failure: **6.69%** deviation
  versus feedback's **2.02%**.
- **Stronger activity comparator:** in a validation-fixed, approximately error-matched
  nominal pair, randomized SAC and volume-oriented feedback have **3.52% / 3.28%** error,
  **257,145 / 324,326** units, and **1.18% / 2.14%** sell-price changes. SAC is smoother
  and trades less; the chosen score advantage does not imply dominance over feedback.

## Scope and reproducibility

Currency error averages `abs(total_gold / initial_total_gold - 1)`. Sell-price change
averages absolute relative changes across goods and steps, excluding shop buy prices.
Policies execute separately and are averaged for reporting, not as an action ensemble.

The main-study simulator lacks real-player calibration and a consumption/utility model.
A later [behavior audit](protocols.md#behavioral-assumptions) adds illustrative consumption/satiation,
under which the equal-budget SAC score advantage reverses. Original-market 1,000-step
tests retain score gains but expose currency/distribution trade-offs. Currency is a
chosen setpoint; these simulations do not establish welfare or real-game inflation
control. Limited game testing is author-reported; public API code reads status.
Five seeds and unadjusted intervals support exploratory conclusions.

Configurations, models, data, source snapshots, and verification code are retained.
Main-study verification reconstructs statistics and replays 66 selected episodes; full replay is
separate. [Results](../results/main/) ·
[Methods](methodology.md) · [Reproduction](experiments.md) · [Poster supplement](project.md#original-poster).

<!-- brief:end -->
