"""Protocol validation and reporting shared by the CLI entry points."""

from dataclasses import asdict
import json
from pathlib import Path

from rl.config import MarketConfig, SACConfig, integer


def load_protocol(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return resolve_protocol(raw)


def resolve_protocol(raw):
    """Resolve an in-memory base protocol as well as a JSON file."""
    unknown = set(raw) - {"description", "market", "sac", "training_seeds", "evaluation_seeds", "adaptive_gain"}
    if unknown:
        raise ValueError(f"Unknown protocol fields: {sorted(unknown)}")
    market = MarketConfig.from_dict(raw.get("market", {}))
    sac = SACConfig(**raw.get("sac", {}))
    train_seeds = raw.get("training_seeds", [0, 1, 2])
    eval_seeds = raw.get("evaluation_seeds", [100, 101, 102, 103, 104])
    for name, seeds in (("training_seeds", train_seeds), ("evaluation_seeds", eval_seeds)):
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"{name} must be a nonempty list")
        for seed in seeds:
            integer(seed, name, 0)
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{name} must be unique")
    # Training resets derive distinct episode seeds from the initialization seed.
    training_market_seeds = {s + ep * 10_000 for s in train_seeds for ep in range(1, sac.episodes + 1)}
    if set(eval_seeds) & (set(train_seeds) | training_market_seeds):
        raise ValueError("Evaluation seeds must be held out from training")
    gain = raw.get("adaptive_gain", 0.5)
    if not isinstance(gain, (int, float)) or not 0 < gain <= 1:
        raise ValueError("adaptive_gain must be in (0, 1]")
    resolved = {"description": raw.get("description", ""), "market": market.to_dict(),
                "sac": asdict(sac), "training_seeds": train_seeds,
                "evaluation_seeds": eval_seeds, "adaptive_gain": gain}
    return market, sac, resolved


def run_main(main):
    """Keep expected configuration and filesystem errors readable in the CLI."""
    try:
        main()
    except (ValueError, TypeError, KeyError, OSError) as error:
        raise SystemExit(f"Error: {error}") from error
