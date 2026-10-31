# Running the interactive demo

Install dependencies as described in the root README, then:

```bash
cp .env.example .env
python manage.py migrate
python manage.py runserver --noreload
```

Open <http://127.0.0.1:8000/>. The page uses local CSS/JavaScript and canvas charts; no CDN, Minecraft service, or GPU is required.

1. Choose SAC, Fixed, Random, or Adaptive.
2. Set the seed, episodes, market steps, players, price ranges, and optional resource shock.
3. Start a run. SAC learns; baselines run independent simulations without learning.
4. Inspect episode return, mean shop-price index, final gold/player, and final-step shop statistics.
5. Download the latest completed run's CSV/configuration, provenance manifest, and SAC inference model. Stop retains completed episodes and writes an explicit partial outcome for SAC.

With very short SAC runs, the replay buffer may not reach the batch size and no learning occurs. The status message explicitly reports when there were no parameter updates; inspect `updates` in the CSV/outcome before interpreting the model. The saved zero-update checkpoint is available for inspection but is rejected by evaluation. A one-strategy web run is exploratory; use `experiments.run` to compare methods under a recorded protocol.

## Products and relationships

Shop buy prices are paid to players, and shop sell prices are charged to them. All bounds must be positive and satisfy `buy_low < buy_high <= sell_low < sell_high`.

Optional manually configured association rules use zero-based product indices:

```json
[
  {"antecedent": [0], "consequent": 1, "type": "complementary", "confidence": 0.7},
  {"antecedent": [1], "consequent": 2, "type": "substitute", "probability": 0.5}
]
```

These rules influence sampled purchase probabilities. They are not automatically fitted or mined from historical data.

## Runtime and storage

The demo reserves one job at a time per Django process and runs it in a background thread. Use a **single local process with `--noreload`**: auto-reload interrupts jobs; multiple WSGI processes would have separate in-memory job states. This local demonstration is not a persistent multiuser job service.

The example `.env` enables `DEBUG=True` for local static files. Existing shell environment variables take precedence over `.env`; if you already export `DEBUG=False`, unset it or set it to `True` for this local development command.

Limits: 1–200 episodes, 1–500 steps/episode, at most 20,000 total steps, 2–1,000 players, and 1–12 products. Use the CLI for larger research batches.

Outputs are stored in `runs/demo/<run-id>/`. Reloading the page does not erase results. The status/download interface serves the latest run in the current process; older artifacts remain on disk but are not rediscovered after a restart.

`/server_status/` is the independent optional Minecraft API page. It defaults to labeled synthetic data; see [the integration guide](minecraft-api.md) for live connections and implementation scope.
