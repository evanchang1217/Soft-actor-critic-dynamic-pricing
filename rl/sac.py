"""SAC in normalized action coordinates; environment scaling is explicit."""

from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import MarketConfig, SACConfig
from .environment import MultiStoreEnv


def mlp(input_dim, output_dim, hidden):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, output_dim))


def verify_checkpoint_updates(checkpoint, path, *, allow_unverified=False):
    """Require evidence of learning; older training runs recorded it in outcome.json."""
    steps = checkpoint.get("trained_steps")
    if type(steps) is not int or steps <= 0:
        raise ValueError("Cannot evaluate a checkpoint without training steps")
    updates = checkpoint.get("gradient_updates")
    source = "checkpoint"
    if updates is None:
        outcome_path = Path(path).parent / "outcome.json"
        if outcome_path.is_file():
            try:
                outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise ValueError("Cannot read the legacy checkpoint's outcome.json") from error
            if not isinstance(outcome, dict) or type(outcome.get("steps")) is not int or outcome["steps"] != steps:
                raise ValueError("Legacy checkpoint and outcome.json have inconsistent training steps")
            updates = outcome.get("updates")
            source = "legacy_outcome"
        elif allow_unverified:
            warnings.warn("Checkpoint has no gradient-update record; evaluating it as explicitly unverified.",
                          RuntimeWarning, stacklevel=2)
            return {"gradient_updates": None, "update_verification": "unverified"}
        else:
            raise ValueError("Checkpoint has no gradient-update record. Include its outcome.json or explicitly "
                             "use --allow-unverified-checkpoint to evaluate an unverified checkpoint")
    if type(updates) is not int or updates < 0:
        raise ValueError("Checkpoint gradient-update count must be a nonnegative integer")
    if updates == 0:
        raise ValueError("SAC checkpoint has no gradient updates; increase the training budget before evaluation")
    return {"gradient_updates": updates, "update_verification": source}


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.net = mlp(state_dim, 2 * action_dim, hidden)

    def forward(self, state):
        mean, log_std = self.net(state).chunk(2, dim=-1)
        return mean, log_std.clamp(-20, 2)

    def sample(self, state):
        mean, log_std = self(state)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw = distribution.rsample()
        action = raw.tanh()
        # Stable tanh Jacobian. This density is in [-1, 1] coordinates, as are
        # critic inputs and target entropy (-action_dim). No physical-scale term.
        log_prob = distribution.log_prob(raw) - 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        return action, log_prob.sum(-1, keepdim=True)


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.net = mlp(state_dim + action_dim, 1, hidden)

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=-1))


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, seed):
        self.rng = np.random.default_rng(seed)
        self.capacity, self.position, self.size = capacity, 0, 0
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.terminated = np.empty((capacity, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, terminated):
        i = self.position
        self.states[i], self.actions[i] = state, action
        self.rewards[i], self.next_states[i], self.terminated[i] = reward, next_state, terminated
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, size, device):
        indices = self.rng.choice(self.size, size, replace=False)
        return [torch.as_tensor(array[indices], device=device)
                for array in (self.states, self.actions, self.rewards, self.next_states, self.terminated)]

    def __len__(self):
        return self.size


class SACAgent:
    def __init__(self, env: MultiStoreEnv, config: SACConfig, device="cpu"):
        self.env, self.config = env, config
        self.device = torch.device(device)
        state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0] - int(not config.control_fee)
        action_dim = self.action_dim
        self.policy = GaussianPolicy(state_dim, action_dim, config.hidden).to(self.device)
        self.q1 = QNet(state_dim, action_dim, config.hidden).to(self.device)
        self.q2 = QNet(state_dim, action_dim, config.hidden).to(self.device)
        self.target_q1, self.target_q2 = deepcopy(self.q1), deepcopy(self.q2)
        for target in (self.target_q1, self.target_q2):
            target.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            [*self.q1.parameters(), *self.q2.parameters()], lr=config.learning_rate)
        self.log_alpha = torch.tensor(math.log(config.initial_alpha), device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)
        self.target_entropy = -float(action_dim)
        low = env.action_space.low[:action_dim]
        high = env.action_space.high[:action_dim]
        self.scale = (high - low) / 2
        self.bias = (high + low) / 2

    def normalize_action(self, action):
        return np.asarray((action[:self.action_dim] - self.bias) / self.scale, dtype=np.float32)

    @torch.no_grad()
    def act(self, observation, deterministic=False):
        state = torch.as_tensor(self.env.encode(observation), device=self.device).unsqueeze(0)
        if deterministic:
            mean, _ = self.policy(state)
            normalized = mean.tanh()
        else:
            normalized, _ = self.policy.sample(state)
        action = normalized[0].cpu().numpy() * self.scale + self.bias
        if not self.config.control_fee:
            action = np.r_[action, 0.0]
        return np.clip(action, self.env.action_space.low, self.env.action_space.high).astype(np.float32)

    def update(self, replay):
        c = self.config
        state, action, reward, next_state, terminated = replay.sample(c.batch_size, self.device)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(next_state)
            next_q = torch.minimum(self.target_q1(next_state, next_action), self.target_q2(next_state, next_action))
            target = reward + c.gamma * (1 - terminated) * (next_q - alpha * next_log_prob)
        critic_loss = F.mse_loss(self.q1(state, action), target) + F.mse_loss(self.q2(state, action), target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        for critic in (self.q1, self.q2):
            critic.requires_grad_(False)
        policy_action, log_prob = self.policy.sample(state)
        policy_q = torch.minimum(self.q1(state, policy_action), self.q2(state, policy_action))
        actor_loss = (alpha * log_prob - policy_q).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        for critic in (self.q1, self.q2):
            critic.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            for critic, target_critic in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
                for parameter, target_parameter in zip(critic.parameters(), target_critic.parameters()):
                    target_parameter.lerp_(parameter, c.tau)
        metrics = {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
                   "alpha": self.log_alpha.exp().item()}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Non-finite SAC update; inspect configuration and training data")
        return metrics

    def save(self, path, *, seed, trained_steps, gradient_updates=None):
        """Inference checkpoint, not a training-resumption checkpoint."""
        if gradient_updates is not None and (type(gradient_updates) is not int or gradient_updates < 0):
            raise ValueError("gradient_updates must be a nonnegative integer or None")
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        torch.save({
            "format_version": 1,
            "normalization": "semantic-v1",
            "action_coordinates": "normalized-v1",
            "market": self.env.config.to_dict(),
            "sac": asdict(self.config),
            "seed": int(seed), "trained_steps": int(trained_steps),
            "gradient_updates": gradient_updates,
            "policy": self.policy.state_dict(),
        }, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, env=None, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if (checkpoint.get("format_version") != 1 or checkpoint.get("normalization") != "semantic-v1"
                or checkpoint.get("action_coordinates") != "normalized-v1"):
            raise ValueError("Unsupported checkpoint format")
        market = MarketConfig.from_dict(checkpoint["market"])
        if env is None:
            env = MultiStoreEnv(market)
        elif env.config.to_dict() != market.to_dict():
            raise ValueError("Checkpoint market configuration differs from evaluation configuration")
        agent = cls(env, SACConfig(**checkpoint["sac"]), device)
        agent.policy.load_state_dict(checkpoint["policy"])
        agent.policy.eval()
        return agent, checkpoint
