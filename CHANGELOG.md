# Maintenance record

## 0.2.0 — 2026-10-28

This release builds on the published capstone demo. It does not replace or relabel the original poster results.

- Extracted the configurable market, SAC implementation, observation encoding, and training loop into a shared `rl/` package.
- Corrected reset observations, initial CPI reference, price/fee action handling, observation bounds, and resource-shock timing; documented additional modeling changes.
- Added four explicit pricing policies, held-out evaluation, multiple training seeds, inference checkpoints, raw CSVs, summaries, figures, and provenance manifests.
- Rebuilt the local Django interface around the shared implementation, with prefilled settings, episode charts, stopping, downloads, job reservation, and validation.
- Preserved the existing Minecraft status reads behind a standalone client, added sample mode, and documented the missing original write-back/deployment components.
- Extracted original poster figures, retained the original standalone SAC demo plot, and separated historical evidence from new evaluation.
- Added environment/SAC/protocol/API/demo tests, a tested dependency snapshot, and a GitHub Actions test/smoke workflow.
- Added a separate review follow-up: validation-selected adaptive/constant-target rules, true price-only SAC ablation, fresh reporting markets, specified market shifts, realized reward-component analysis, common market-level variability estimates, and startup research-source snapshots.
- Made zero-update SAC runs visible in the demo and rejected them during evaluation. Evaluation now records update provenance, labels fixed-fee models correctly, and refuses to pool checkpoints with incompatible SAC modes, training settings, or budgets.
- Represented sample standard deviations as unavailable when fewer than two independent units exist, instead of reporting misleading zeroes.
- Removed disconnected templates, unused vendored Bootstrap assets, an unrelated DQN image, and legacy SQLite monkeypatching.

The prior demo source remains available in Git history at `e092c8684b2e42dd0277d053d38824a194e51aac`.
