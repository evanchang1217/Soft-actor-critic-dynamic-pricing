"""One local training job per Django process. Use the CLI for research batches."""

import csv
from copy import deepcopy
from dataclasses import asdict
import logging
from pathlib import Path
import threading
from uuid import uuid4

from experiments.baselines import Baseline
from rl.artifacts import manifest, new_run_directory, write_json
from rl.environment import MultiStoreEnv
from rl.training import episode_summary, train


logger = logging.getLogger(__name__)


class DemoRunner:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.output = None
        self.status = {"running": False, "state": "idle", "method": "sac", "message": "Ready to run",
                       "current_episode": 0, "total_episodes": 0, "history": [], "latest": None,
                       "run_id": None, "artifacts": [], "configuration": None}

    def snapshot(self):
        with self.lock:
            return deepcopy(self.status)

    def start(self, market, sac, seed, method, root):
        with self.lock:
            if self.status["running"]:
                return False
            run_id = uuid4().hex
            self.output = Path(root) / run_id
            self.stop_event.clear()
            self.status = {"running": True, "state": "running", "method": method,
                           "message": "Training SAC" if method == "sac" else f"Simulating {method} pricing",
                           "current_episode": 0, "total_episodes": sac.episodes, "history": [],
                           "latest": None, "run_id": run_id, "artifacts": [],
                           "configuration": {"market": market.to_dict(), "sac": asdict(sac), "seed": seed}}
        worker = threading.Thread(target=self._run, args=(market, sac, seed, method), daemon=True)
        try:
            worker.start()
        except RuntimeError:
            with self.lock:
                self.status.update(running=False, state="failed", message="Could not start worker")
            raise
        return True

    def cancel(self):
        with self.lock:
            if self.status["running"]:
                self.stop_event.set()
                self.status["message"] = "Stopping; completed episodes will be retained"

    def _progress(self, row, info):
        with self.lock:
            self.status["history"].append(row)
            self.status["current_episode"] = row["episode"]
            self.status["latest"] = info

    def _run(self, market, sac, seed, method):
        try:
            if method == "sac":
                result = train(market, sac, seed, self.output, self._progress, self.stop_event.is_set)
                cancelled = result["cancelled"]
                if result["updates"] == 0:
                    message = ("Stopped before any SAC parameter updates." if cancelled else
                               "Run complete, but SAC had no parameter updates. Increase episodes or steps.")
                else:
                    message = "Stopped" if cancelled else "Run complete"
            else:
                cancelled = self._simulate(market, sac.episodes, seed, method)
                message = "Stopped" if cancelled else "Run complete"
            with self.lock:
                self.status.update(state="cancelled" if cancelled else "completed",
                                   message=message,
                                   artifacts=[name for name in ("training.csv", "policy.pt", "config.json", "manifest.json", "outcome.json")
                                              if (self.output / name).is_file()])
        except Exception:
            logger.exception("Demo run failed")
            with self.lock:
                self.status.update(state="failed", message="Run failed. See the Django console for details.")
        finally:
            with self.lock:
                self.status["running"] = False

    def _simulate(self, market, episodes, seed, method):
        new_run_directory(self.output)
        write_json(self.output / "config.json", {"market": market.to_dict(), "episodes": episodes, "seed": seed, "method": method})
        write_json(self.output / "manifest.json", manifest("demo-baseline"))
        env = MultiStoreEnv(market)
        with (self.output / "training.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = None
            for episode in range(1, episodes + 1):
                observation, _ = env.reset(seed=seed + episode * 10_000)
                policy = Baseline(method, env, seed=seed + episode)
                infos, rewards = [], []
                for _ in range(market.max_steps):
                    if self.stop_event.is_set():
                        env.close()
                        return True
                    observation, reward, terminated, truncated, info = env.step(policy.act(observation))
                    infos.append(info)
                    rewards.append(reward)
                    if terminated or truncated:
                        break
                row = {"episode": episode, **episode_summary(infos, rewards)}
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                self._progress(row, info)
        env.close()
        return False

    def artifact(self, run_id, name):
        with self.lock:
            if (self.status["running"] or run_id != self.status["run_id"]
                    or name not in self.status["artifacts"]):
                return None
            return self.output / name


runner = DemoRunner()
