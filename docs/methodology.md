# Market model and SAC implementation

The maintained implementation separates market transitions, control objectives,
controllers and evaluation. The [project history](project.md#authorship) records
its relationship to the original capstone; [study protocols](experiments.md)
define the exact settings behind each result.

## Which version is being measured?

| Entry point | Objective and execution |
| --- | --- |
| Main research studies | Explicit `balanced_v1`, fixed 5% fee, saved-policy evaluation; 10k reward ablations and 30k controller comparisons |
| Saved-model web demo | Frozen 30k SAC and tuned P/PI from the behavioral audit; no training |
| Web training form | Balanced objective by default, 10k training; optional legacy objective |
| Generic four-method CLI | Legacy defaults for compatibility with earlier experiments |
| Aggregate-market follow-up | Separate two-good task economy; representation, transfer and online-planning studies |

SAC has one shared implementation in `rl/sac.py`. Response-randomized training
changes the training distribution; its checkpoint `market` retains nominal
inference semantics, while `training_distribution` and episode logs record the
sampled training domains. The historical 50/50 SAC/feedback blend is a separate
inference controller.

## Controllers

| Controller | Decision rule |
| --- | --- |
| SAC | Learns bounded continuous price adjustments from market interactions; evaluation uses deterministic policy actions. |
| P | Corrects target price fractions in proportion to the observed total-currency error. |
| PI | Adds accumulated currency error to the proportional correction; the integral is clamped to ±0.5 and reset each episode. |

P/PI parameters are selected on validation markets before testing. All methods
in these studies use the same fixed fee and price-action limits. They observe
aggregate market information; P/PI use the total-currency component.

## Market mechanics

Each episode samples player price/fee sensitivities and initial shop prices, resets inventories to zero, and gives every player `initial_gold`. All strategies use the same configuration and initial market seed. One step:

1. Apply bounded, multiplicative shop buy/sell price changes and an additive fee change.
2. Generate resources and add them to player inventories.
3. Sample player sales to the shop, limited by inventory; credit net sale proceeds.
4. Sample player purchases from the shop, limited by their joint budget across products.
5. Record trade counts, units, net currency sink, shop price index, and player gold; calculate reward.

The probability functions retain the demo's price-position, sensitivity, fee, inventory, and budget factors. The sampling combines a Bernoulli participation mask with Poisson quantities. Association rules are optional **manually supplied** purchase relationships. There is no released logistic-regression fitting or Apriori mining pipeline.

Default player price-response multipliers are sampled from `[0.7, 1.3]`; configurable bounds support explicitly identified response-shift tests. These multipliers scale participation probabilities and are not estimated economic elasticities. In the original environment, players have no item-consumption or utility model: purchases add to inventory and mining supplies more resources. This is a reduced market-control simulator, not a validated complete game economy.

Price bounds must be strictly positive with `buy_low < buy_high <= sell_low < sell_high`. The new spread restriction prevents immediate risk-free resale to the shop. Example product values are illustrative and were not recovered from the poster's experiment settings.

Resources follow a bounded, multiplicative noisy process, with floating-point rates rounded only when resources enter inventories. `shock_step` applies a one-time multiplier **inside every episode**, before that step's noise and cap. Reset restores the baseline rate. A separate resource RNG keeps this exogenous process identical for a given market seed regardless of policy actions. Trade trajectories can still differ because decision probabilities and Poisson draws depend on policy-induced states.

## State and action layout

For `P` products, observations have `6P + 4` features:

```text
[buy_prices(P), sell_prices(P),
 shop_buy_counts(P), shop_sell_counts(P),
 shop_buy_volumes(P), shop_sell_volumes(P),
 net_sink, shop_price_index, total_player_gold, fee_rate]
```

Shop buy counts are players **selling to** the shop; shop sell counts are players **buying from** it. Counts are per product; their sum is not a count of unique players.

The action has `2P + 1` features, **interleaved**, unlike the state:

```text
[buy_offset_0, sell_offset_0, ..., buy_offset_P-1, sell_offset_P-1, fee_delta]
```

Default price offsets are bounded by ±5%, and fee changes by ±0.01. The environment rejects out-of-range or non-finite actions. Prices and fee levels are then constrained to their configured bounds.

The raw observation space admits signed sinks and unbounded trade volumes/gold. `env.encode()` uses fixed, documented semantic scales: affine transforms for prices, counts, index, and fees; `2*tanh(volume / (players*avg_qty))-1` for volumes; `tanh(sink / initial_total_gold)` for sink; `2*money/(money+initial_total_gold)-1` for money. Encoded features are bounded by `[-1, 1]`. No normalization statistics are fitted on evaluation data.

Inventory distributions, player sensitivities, and the resource process are not fully observable in this aggregate state. This is a partially observed control problem; the feed-forward policy does not recover a complete market state.

## Reward and reported indicators

The main studies use `balanced_v1`. For `P` products, `N` players, average
quantity `q`, initial total gold `M0`, counts `C_b,C_s` and volumes `V_b,V_s`:

```text
L_count  = sum(|C_b - N/2| + |C_s - N/2|) / (N/2)
L_volume = sum(|V_b - N*q/2| + |V_s - N*q/2|) / (N*q/2)
L_cpi    = |index_t - index_(t-1)| / index_(t-1)

balanced_reward = -(
    0.5  * L_count / (2P)
  + 0.5  * L_volume / (2P)
  + 0.05 * L_cpi / price_offset
  + 1.0  * |money_t / M0 - 1|
)
```

Trade terms average across product/direction slots; price changes scale with the
per-step action bound. Currency tracking penalizes both depletion and accumulation.
The implementation saves weighted component losses so score differences can be
separated into trading, price and currency contributions. [Objective ablation](protocols.md#objective-design).

<details>
<summary>Preserved legacy reward</summary>

```text
L_money = |money_t - money_(t-1)| / (N * money_(t-1))
reward  = -(L_count + L_volume + 0.5*L_cpi + 0.5*L_money)
```

The legacy currency term measures one-step change with an extra player-count
divisor. `balanced_v1` replaces it with episode-initial setpoint tracking.
Returns under different objectives require a declared common evaluation score.

</details>

The quantity called CPI in the legacy code is `mean(shop_sell_price / sell_lower_bound)`. It is an unweighted shop-price index with a configurable lower-bound reference, **not** an independently measured consumer basket or currency-value estimate. The policy directly controls its numerator. `L_cpi` penalizes changes, including decreases; it does not explicitly target a low index. Lower or more stable values alone do not prove inflation control.

The later diagnostic/robustness field `mean_item_price_change` is
`mean_t mean_product abs(sell_price_t / sell_price_(t-1) - 1)` using executed
shop **sell** prices after clipping. It excludes shop buy prices and differs
from `mean_price_change`, which averages the aggregate index's relative changes.
The first step is measured against the episode's initial prices. New overview
labels explicitly say sell-price changes; archived figure labels and CSV field
names are preserved. Neither indicator includes player-to-player market prices.

`sink = purchases_paid_by_players - proceeds_paid_to_players`. Positive sink removes player currency; negative sink injects it. The accounting identity `delta(total_player_gold) = -sink` is tested. Participation is the fraction of `2*N*P` possible directional player-product trade slots that traded. It is not the percentage of unique active players.

## Interpreting the chosen targets

These settings define the control task and diagnostics. They are not fitted to
observed player preferences or established economic optima.

| Setting | Role in this study | Interpretation |
| --- | --- | --- |
| Trade counts of `N/2` per product and direction; volumes of `N*q/2` | Retained from the legacy formulation to provide explicit trading targets and continuity across reward ablations | The 50% count target refers to directional player-product slots. Its attainability depends on the behavior model; it is not a measured ideal participation rate. |
| Initial total player gold `M0` | A fixed reference for comparing currency tracking in a fixed-population simulation | Penalizes depletion and accumulation relative to the start; does not define an optimal money supply. |
| Gold below `0.2 * initial_gold` | A lower-tail diagnostic in the behavioral audit | The 20% threshold describes low simulated balances, not a calibrated poverty or player-retention boundary. |
| Balanced reward weights | Specify the relative penalties on trading targets, price-index changes and currency deviation | Method rankings apply to this chosen objective; component metrics show the associated trade-offs. |

The [behavioral audit](protocols.md#behavioral-assumptions) extends the original mechanics with
illustrative consumption and inventory-dependent purchase participation. That
study combines those two mechanisms into one factor. The later
[independent mechanism study](protocols.md#independent-mechanisms) separates their switches while
holding the original participation/quantity rule fixed.

## SAC conventions

- Twin Q-networks, target Q-networks, replay sampling, reparameterized Gaussian policy, and automatic entropy temperature.
- Policy samples, replay actions, critic inputs, and entropy are all in normalized `[-1,1]` action coordinates. Each dimension is scaled to its own environment bounds only when executing the action.
- The tanh log-Jacobian uses a numerically stable expression. Because entropy is defined in normalized coordinates, there is no physical-action affine-scale term. Full-control SAC uses target entropy `-(2P+1)`. The separately trained fixed-fee ablation removes the fee dimension from actor/critics/replay and uses `-2P`, appending zero fee delta only for environment execution, including warmup.
- Observation encoding is identical for training and inference.
- Market episodes end by **time-limit truncation**, not a modeled terminal market event. Critic targets bootstrap through that truncation. This is the continuing-control interpretation of the demo, not a finite-horizon liquidation task.
- CPU, one PyTorch thread, explicit initialization/exploration/replay/market seeds. Identical software/hardware can reproduce small runs; cross-platform bitwise identity is not promised.
- New `policy.pt` files store weights, market settings, normalization convention, seed, step count, and actual gradient-update count for inference. They do not contain replay/optimizer/RNG state for exact training resumption. Older artifacts can use the accompanying `outcome.json` to verify update counts.


## Aggregate task economy — separate follow-up

This model represents pooled demand and production, rather than individual
players. It is not a calibration or a numerical continuation of the original
three-good market. All prices are market-wide per good.

Two complementary goods fulfill tasks with recipes 2:1 or 1:2. A public current
recipe follows a Markov process (nominal switch probability 0.06 per period).
Eight tasks arrive on average per period. Nominal shared production capacity is
36 units; bids control participation and softmax allocation across goods.
Purchases depend on asks through a sigmoid willingness function, household
holdings, available stock and finite pooled cash. Preference and productivity
noise and Poisson task arrivals are generated independently of the policy.

The nominal retained setting has instantaneous delivery, no goods expiry,
warehouses of 72 units per good, initial stock 24 per good, and initial household
and shop money of 6,000 each. Completing a task issues four currency units.
The simulator records goods conservation and the identity
`household money + shop treasury = initial combined cash + issued task rewards`.
These are specified behavioral rules, not observations of human players.

Actions adjust two bids and two asks by at most 5% per period. Bids remain in
[3,12], asks in [6,16], with a minimum 0.5 spread. The observation stacks two
public frames (140 inputs), including prices, flows, stocks, household holdings,
task/recipe state and cash ledgers. No true future shocks enter a controller.

The nominal immediate-service objective is:

```text
reward = -(
    abs(household_money / initial_household_money - 1) / 0.1
  + unfulfilled_tasks / 8
  + 0.01 * (shop_goods + household_goods) / (18 * 8)
  + 0.03 * mean(abs(log(price / previous_price))) / (-log(0.95))
)
```

Task cohorts, any retention/backlog terms and regime-specific parameters are
resolved in each saved protocol. Currency and service are chosen control
objectives; neither defines player welfare or an optimal real-game economy.

### Representation correction

Raw treasury and cumulative issued currency can grow beyond the 96-period
training range in continuing markets. Stationary-input SAC translates these two
input fields in each frame using the actual household money:

```text
translated_treasury = max(0, initial_combined_cash - household_money)
translated_issued  = household_money + translated_treasury - initial_combined_cash
```

This changes encoded inputs from the first training sample, preserving actual
market state, money, actions and reward. It removes historical ledger-growth
cues; it can also discard meaningful liquidity information in another economy.
Five new paired seeds and nine transfer regimes test the benefit and limits.
[Retraining protocol](protocols.md#representation-and-transfer).

### Feedback and planning

Aggregate P/PI convert public currency/inventory errors into target prices with
fixed gains; PI also accumulates bounded error. These controls differ from the
original market's implementation and must be identified with their study.

The stronger planner evaluates 82 first actions: all 81 combinations of
{-5%,0,+5%} across four prices, plus a continuous feedback proposal. For each
first action it simulates hold-price and feedback continuations, averages sampled
future outcomes, selects the better continuation, and chooses the best first
move. It executes only that move and plans again next period. Forecasts use
private random draws, never the actual future trajectory. This finite search
is not an optimal-control solution or a search over every future action sequence.

Online planning fits demand slope, production reservation price and recipe
switch probability from public transactions. It starts after 32 observations,
updates every 16 periods, and uses at most the last 64. Identifiability gates
retain previous estimates when realized trade is insufficient. The functional
form, noise laws and other constants are shared prior knowledge. SAC has a
two-frame input and extra offline simulator training; evaluation weights are
fixed. This is not equal-memory or model-free-data-access equivalence.
[Estimator](protocols.md#shared-data-model-estimation) · [Online updates](protocols.md#periodic-online-refitting) ·
[Common-state CPU and training-cost measurement](protocols.md#quality-and-computation).

## Implementation provenance

The shared `rl/sac.py` implementation is unchanged by repository curation.
The aggregate runtime lives in `experiments/market/`; shared record statistics
live in `experiments/common.py`. Historical experiment runners are no longer
runtime dependencies. Their exact source snapshots remain in the separate
evidence archive and are verified independently of the current checkout.
[Authorship and implementation history](project.md#authorship).

## Research scope

This is a synthetic control testbed with system and empirical contributions:
objective ablations, multi-seed feedback comparisons, a representation correction,
and measured quality/online-compute trade-offs. The studies use the established
SAC algorithm and evaluate market design and controller behavior.

The original-market score advantage reverses under tested satiation mechanisms.
Aggregate-market transfer fails under scarce production. Periodic model refitting
repairs much of stale-model planning loss; sustained SAC superiority over the
strong planner is not established. A learned policy is itself a rule: these
results cannot establish that all rule-based methods fail.

No real-player transactions, fitted preferences or field outcomes calibrate these
markets. Currency tracking, service and price stability are chosen targets, not
measurements of welfare, retention or real inflation. Current recipes are public;
transfer parameters change from evaluation start. Frozen SAC weights respond to
observations without online weight updates. Planning knows the structural family
and uses longer estimation history; SAC consumes additional offline simulator
interactions. Those information and compute differences remain explicit.

### Related work


| Prior work | Relationship |
| --- | --- |
| Haarnoja et al., [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) | The learning algorithm is established; no new SAC update is claimed. |
| Rana and Oliveira, [Interdependent-product RL pricing](https://www.sciencedirect.com/science/article/pii/S095741741400400X) | Multi-product coordination with RL is established; adding multiple goods alone is not novelty. Publisher abstract consulted. |
| Yavuz and Kaya, [RL pricing and inventory for perishables](https://www.sciencedirect.com/science/article/pii/S1568494624006380) | SAC and richer inventory/pricing mechanisms have prior work. Publisher abstract consulted. |
| Henderson et al., [Deep Reinforcement Learning That Matters](https://ojs.aaai.org/index.php/AAAI/article/view/11694) | Seed variability and reproducibility motivate explicit protocols and saved evidence. |
| Agarwal et al., [Statistical reliability of deep RL evaluation](https://arxiv.org/abs/2108.13264) | Few training runs limit ranking precision; intervals and unfavorable comparisons must accompany means. |

This is a targeted positioning summary, not an exhaustive novelty review.
Publication-level originality has not been established. The project can be
presented as an empirical systems/control study using an existing RL method.
