# Market model and SAC implementation

This release derives its simulator and learning formulation from the **published Django demo**, not from a recovered original experiment bundle. It provides a new evaluation protocol. The original poster remains an independent historical artifact.

## Market mechanics

Each episode samples player price/fee sensitivities and initial shop prices, resets inventories to zero, and gives every player `initial_gold`. All strategies use the same configuration and initial market seed. One step:

1. Apply bounded, multiplicative shop buy/sell price changes and an additive fee change.
2. Generate resources and add them to player inventories.
3. Sample player sales to the shop, limited by inventory; credit net sale proceeds.
4. Sample player purchases from the shop, limited by their joint budget across products.
5. Record trade counts, units, net currency sink, shop price index, and player gold; calculate reward.

The probability functions retain the demo's price-position, sensitivity, fee, inventory, and budget factors. The sampling combines a Bernoulli participation mask with Poisson quantities. Association rules are optional **manually supplied** purchase relationships. There is no released logistic-regression fitting or Apriori mining pipeline.

Default player price-response multipliers are sampled from `[0.7, 1.3]`; configurable bounds support explicitly identified response-shift tests. These multipliers scale participation probabilities and are not estimated economic elasticities. Players have no item-consumption or utility model: purchases add to inventory and mining supplies more resources. This is a reduced market-control simulator, not a validated complete game economy.

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

Let `N` be player count, `q` average quantity, `C_b,C_s` player-product trade counts and `V_b,V_s` volumes:

```text
L_count  = sum(|C_b - N/2| + |C_s - N/2|) / (N/2)
L_volume = sum(|V_b - N*q/2| + |V_s - N*q/2|) / (N*q/2)
L_cpi    = |index_t - index_(t-1)| / index_(t-1)
L_money  = |money_t - money_(t-1)| / (N * money_(t-1))
reward   = -(L_count + L_volume + 0.5*L_cpi + 0.5*L_money)
```

Weights are configurable, but the defaults retain the demo's formula, including the additional `N` divisor in `L_money`. This makes that term relatively small and deserves sensitivity analysis. All four component losses are saved in evaluation `steps.csv`.

The quantity called CPI in the legacy code is `mean(shop_sell_price / sell_lower_bound)`. It is an unweighted shop-price index with a configurable lower-bound reference, **not** an independently measured consumer basket or currency-value estimate. The policy directly controls its numerator. `L_cpi` penalizes changes, including decreases; it does not explicitly target a low index. Lower or more stable values alone do not prove inflation control.

`sink = purchases_paid_by_players - proceeds_paid_to_players`. Positive sink removes player currency; negative sink injects it. The accounting identity `delta(total_player_gold) = -sink` is tested. Participation is the fraction of `2*N*P` possible directional player-product trade slots that traded. It is not the percentage of unique active players.

## SAC conventions

- Twin Q-networks, target Q-networks, replay sampling, reparameterized Gaussian policy, and automatic entropy temperature.
- Policy samples, replay actions, critic inputs, and entropy are all in normalized `[-1,1]` action coordinates. Each dimension is scaled to its own environment bounds only when executing the action.
- The tanh log-Jacobian uses a numerically stable expression. Because entropy is defined in normalized coordinates, there is no physical-action affine-scale term. Full-control SAC uses target entropy `-(2P+1)`. The separately trained fixed-fee ablation removes the fee dimension from actor/critics/replay and uses `-2P`, appending zero fee delta only for environment execution, including warmup.
- Observation encoding is identical for training and inference.
- Market episodes end by **time-limit truncation**, not a modeled terminal market event. Critic targets bootstrap through that truncation. This is the continuing-control interpretation of the demo, not a finite-horizon liquidation task.
- CPU, one PyTorch thread, explicit initialization/exploration/replay/market seeds. Identical software/hardware can reproduce small runs; cross-platform bitwise identity is not promised.
- New `policy.pt` files store weights, market settings, normalization convention, seed, step count, and actual gradient-update count for inference. They do not contain replay/optimizer/RNG state for exact training resumption. Older artifacts can use the accompanying `outcome.json` to verify update counts.

## Changes from the released demo

The historical implementation is preserved in [the pre-refactor source](https://github.com/evanchang1217/Soft-actor-critic-dynamic-pricing/blob/e092c8684b2e42dd0277d053d38824a194e51aac/myapp/sac_model/sac_component.py).

| Area | This release |
| --- | --- |
| Reset | Returns real prices, initial currency, index, and fee; previous CPI starts at the actual initial index |
| Configuration | Validated per-environment configuration replaces shared mutable globals |
| Actions and entropy | Per-dimension scaling, normalized critic/entropy convention, fee included in entropy target |
| Observation bounds | Allows negative sinks; correct fee bounds; explicit nonlinear scaling for unbounded features |
| Resource changes | Observable, seeded within-episode shock; no reset-erased boost; avoids repeated integer truncation bias |
| Hyperparameters | Configurable new CPU-oriented settings, recorded with every run |
| Evaluation | New explicit baselines, held-out seeds, frozen-model inference, CSVs and figures |
| UI | Uses the same core, reserves jobs before startup, stops jobs, and saves per-run artifacts |

These changes can materially alter results. The new evaluation is not an exact reproduction of the original poster curves.

The [follow-up protocol](review-protocol.md) retains the legacy reward while auditing its realized component scales, selecting simple baseline parameters on separate validation markets, and testing fixed-fee SAC and explicit market shifts. It does not revise reward weights to improve reported results.

## References

- Haarnoja et al., [Soft Actor-Critic](https://arxiv.org/abs/1801.01290), 2018.
- [Spinning Up: SAC equations and algorithm](https://spinningup.openai.com/en/latest/algorithms/sac.html).
- [Gymnasium: custom environment contract](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).
- [Stable-Baselines3: independent evaluation and repeated runs](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).
