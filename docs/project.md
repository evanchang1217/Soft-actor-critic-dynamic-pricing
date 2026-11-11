# Project history

## Authorship

The original capstone was an individual project under faculty supervision at
National Cheng Kung University. The author designed the pricing formulation and
reward and developed the simulator, SAC system and web demonstration. The author
reports second place among 87 projects; the award concerns that capstone.

The research contributions center on market modeling, system implementation,
objective design and empirical evaluation using the established SAC algorithm.
AI tools assisted portions of subsequent coding and analysis.

The [pre-maintenance implementation](https://github.com/evanchang1217/Soft-actor-critic-dynamic-pricing/tree/e092c8684b2e42dd0277d053d38824a194e51aac)
and [original poster](../Poster/capstone-poster.pdf) are historical evidence, separate
from subsequent experiments with their own protocols and source snapshots.

| Stage | Maintained evidence |
| --- | --- |
| Original capstone | [Original poster](project.md#original-poster), pricing system and server-integration workflow |
| Objective and original-market comparison | [Reward ablation](protocols.md#objective-design), [five-seed comparison](protocols.md#original-market-robustness) |
| Behavioral limits | [Behavior](protocols.md#behavioral-assumptions) and [independent mechanisms](protocols.md#independent-mechanisms), including ranking reversals |
| Representation and transfer | [New-seed retraining](protocols.md#representation-and-transfer), nine regimes and stronger planning |
| Model knowledge and computation | [Estimation](protocols.md#shared-data-model-estimation), [online refitting](protocols.md#periodic-online-refitting), [quality/CPU study](protocols.md#quality-and-computation) |

The current checkout contains the shared runtime, result summaries and demo
models. Full experimental records and historical source snapshots are preserved
in the [evidence archive](experiments.md#full-evidence-archive).

## Original poster

The [original capstone poster](../Poster/capstone-poster.pdf)
records the original pricing system, game-integration workflow and experiments.
The repository has since added a revised reward, tuned feedback baselines,
multi-seed evaluation, input-representation studies and planning comparisons.
The poster is retained as the original capstone document; the
[current result guides](../results/README.md) describe the subsequent evaluations.

The versions differ in the following ways:

| Topic | Current implementation and evaluation |
| --- | --- |
| Evaluation metrics | Currency-setpoint error, traded units, price changes and a common control score. The legacy `CPI` field is a shop-price index; results here concern simulated market control rather than measured real-game inflation. |
| Reward and comparisons | An explicit currency-tracking reward is tested against legacy SAC at equal training budgets. Separate held-out studies compare SAC with tuned feedback and online planning. |
| Player behavior | Reproducible experiments use documented synthetic behavior and optional manually supplied item associations. The historical data and logistic-regression/Apriori fitting pipeline described in the poster are outside the public release. |
| Game integration | The public Minecraft client reads market/player status. The poster's complete server-side price-update workflow, plugin and deployed checkpoint are outside this release; reported game testing has no accompanying field-effect measurements. |

Follow-up records include their own settings, logs and source snapshots. They
evaluate the maintained system rather than reconstructing the poster's original
measurements, whose exact settings and raw logs are unavailable. The reported
capstone award applies to the original project.

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

The award concerns the original individual capstone. Follow-up experiments are
separately dated. The contributions are market modeling, objective design and
empirical evaluation using the established SAC algorithm.
[Project history](project.md#authorship).

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
