# Experiments and reproduction

Install dependencies from the [README](../README.md). All new outputs belong
under ignored `runs/`. Keep original-market and aggregate-market rewards separate.
[Methods](methodology.md) · [Fixed study designs](protocols.md).

## Verify saved studies

One command selects the corresponding frozen source and saved evidence:

```bash
python scripts/reproduce.py --study main
python scripts/reproduce.py --study compute --workers 4
```

| `--study` | Evidence |
| --- | --- |
| `objective` | Equal-budget reward ablation |
| `main` | Five-seed original-market robustness |
| `behavior` | Behavioral assumptions and persistent markets |
| `mechanism` | Independent consumption and satiation |
| `stationary` | Representation retraining and transfer |
| `model-error` | Shared-data estimation and frozen transfer |
| `adaptive` | Periodic online refitting |
| `compute` | Control quality and CPU costs |

Use `--record` for another complete record, `--archive` for an explicit evidence
bundle and `--out` for a fresh verification-output directory where applicable.
Original-market selected verification replays 66 episodes; behavior/mechanism
replay 72/28. `--study main --mode replay --out runs/full-main-replay` is a
separate complete test replay. Aggregate verifiers reconstruct declared tables,
check hashes and replay fixed subsets. Timings are checked as stored measurements,
not expected to reproduce identical noisy values. No verifier retrains SAC.

Source/model/data checksums and stored-table reconstruction are distinct from
numerical replay. PyTorch does not guarantee identical results across platforms
([reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)).
Small inference differences can change discrete trades, so strict historical
replay may require the original numerical environment. CI checks current demo
behavior against the source-verified evaluator on the same platform, while
retaining exact source/model and saved-result checks.

## Full evidence archive

An archive is a compressed, byte-preserved copy of the source, parameters, models
and data behind the saved results. The checkout keeps summaries, figures and ten
policies needed by the demo. Full evidence is in
**sac-pricing-evidence-2026-11-10.tar.gz**, with the complete inventory and SHA-256
in [archive-manifest.json](../results/archive-manifest.json).

**Availability:** the archive exists locally outside the repo; no public download
has been published. The demo and core tests work from a clean clone. Full study
verification requires obtaining the archive from the project owner.
Detailed per-run results, audit inputs, and compute-study state and timing
samples are in the archive; checkout result summaries, figures and demo inputs
remain available. Start with the [result guides](../results/README.md).

```bash
export SAC_RESEARCH_ARCHIVE=/path/to/sac-pricing-evidence-2026-11-10.tar.gz
python scripts/research_archive.py verify
python scripts/research_archive.py restore --out runs/evidence
```

The tools also find the named archive in `../research-archives/` in the owner's
workspace. The manifest contains no absolute machine-specific path. Verification
checks the archive checksum and each extracted file's size/hash, rejects unsafe
paths or links, and uses a temporary copy. Stationary trace byte parts reconstruct
the exact full compressed trace when needed. Frozen originals are verified even
when checkout Markdown links have been simplified.

Restoration creates a fresh `runs/evidence/results/` tree. Do not restore over the
checkout or overwrite curated records. Source snapshots retain their historical
filenames and equations; they are independent of current package organization.
The earlier cleanup backup also contains superseded exploration outside the
maintained evidence collection. Git history has not been rewritten.

### Publishing the evidence

After committing and pushing the corresponding code, create a
[GitHub Release](https://github.com/evanchang1217/Soft-actor-critic-dynamic-pricing/releases/new)
for that code version. Attach the named `.tar.gz` archive and a `SHA256SUMS` file
containing its checksum from the manifest; use a Release asset rather than adding
the archive to Git. Download the published asset and check its checksum, then
replace the local-only availability notices with its verified download link.
Until that upload is complete, the archive is not publicly available.

## Run maintained experiments

```bash
# Short installation/pipeline checks, not convergence evidence.
python -m experiments.run --config experiments/configs/quickstart.json --out runs/quickstart
python -m experiments.balanced_control \
  --config experiments/configs/balanced-control-smoke.json --out runs/objective-smoke
```

Current research runners are `balanced_control`, `robust_training`, `robustness`,
`model_error_study`, `adaptive_planning` and `compute_tradeoff`. Their remaining
configs live under `experiments/configs/`. Assign fresh test seeds for changed
designs and retain source/model metadata. Fitted-model follow-ups accept a restored
model-error record with `--parent`; it includes the initial logging data and models.
For original-market reruns, point reference-model paths in a copied config at the
restored tree. SAC's additional simulated training interactions count as resources.

## Run a completed study from frozen source

Behavior, mechanism and stationary-retraining runners are preserved in their
evidence snapshots. They remain executable without live copies in `experiments/`:

```bash
# Supply a copied protocol with fresh held-out seeds for a changed study.
python scripts/reproduce.py --study behavior --run -- \
  init --config "$PWD/runs/behavior-input.json" --out "$PWD/runs/behavior-new"
python scripts/reproduce.py --study behavior --run -- \
  train --out "$PWD/runs/behavior-new" --workers 4
```

Arguments after `--` go to the frozen CLI. Use absolute config/output paths so
new results survive deletion of the temporary source directory. Mechanism uses
the same phase syntax; stationary uses `--phase init`, `--phase train`, etc.
These optional runs can be expensive. Verification above reuses saved models.

## Figures, reports and record interpretation

`report.py` consolidates model-error, adaptive, compute and compute-zoom rendering.
Render only a complete restored/new record under `runs/`:

```bash
python scripts/report.py --study compute --record runs/evidence/results/diagnostics/2026-11-09-compute-tradeoff
python scripts/render_summary.py --out runs/research-brief
```

`plot_application.py` regenerates the original-market overview from saved tables.
One-time price audits, old timing tools and trace chunk creation are preserved
with historical evidence or the cleanup backup, rather than daily utilities.

Protocols and frozen choices separate validation from testing. Policy/curve
metadata record interactions and updates. Score components and service/currency
metrics expose trade-offs. Means summarize separate policies, not an action
ensemble. Intervals resample independent model/calibration/market units, not
individual steps or players. Exploratory intervals are not universal rankings
or tests of equivalence. [All results](../results/README.md).
