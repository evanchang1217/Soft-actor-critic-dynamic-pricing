"""Continuing-market input representation and transfer regimes."""
from dataclasses import dataclass,asdict
import numpy as np

from .economy import TaskConfig
from .inventory import InventoryMarket,InventoryParameters



@dataclass(frozen=True)
class Regime:
    switch_probability:float=.06
    buffer_periods:float=6.
    capacity_ratio:float=1.5

    def __post_init__(self):
        if not np.isfinite(self.switch_probability) or not 0<=self.switch_probability<=.5:raise ValueError('Invalid switch probability')
        InventoryParameters(self.buffer_periods,self.capacity_ratio)


class StationaryMarket(InventoryMarket):
    def __init__(self,config=None,regime=None,representation='raw'):
        if representation not in ('raw','stationary'):raise ValueError('Invalid representation')
        self.regime=regime or Regime();self.representation=representation
        super().__init__(config,InventoryParameters(self.regime.buffer_periods,self.regime.capacity_ratio),case='capacity')

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed,options=options)
        if seed is None:raise ValueError('Research requires explicit seed')
        c=self.config;g=np.random.default_rng(np.random.SeedSequence([int(seed),5813]))
        # Consume the exact original draws so every regime retains matched shocks.
        g.poisson(c.mean_tasks,c.max_steps+1);g.uniform(.94,1.06,(c.max_steps,2));g.normal(0,.06,(c.max_steps,2))
        phase=int(g.integers(2));schedule=[]
        for u in g.uniform(size=c.max_steps+1):
            if u<self.regime.switch_probability:phase=1-phase
            schedule.append([2.,1.] if phase else [1.,2.])
        self.env.recipe_schedule=np.asarray(schedule);self.env.recipes[:]=schedule[0]
        self.env.frames.clear();self.env.frames.extend(self.env.frame().copy() for _ in range(c.history))
        self.frames.clear();self.frames.extend(self.frame().copy() for _ in range(c.history))
        return self.observe(),{}

    def encode(self,observation):
        x=np.asarray(observation,dtype=np.float32)
        if self.representation=='stationary':
            # Support both one observation and batched observations without touching
            # physical state. Replay stores the same encoded state used by actor.
            x=x.copy();f=x.reshape(*x.shape[:-1],self.config.history,self.frame_size)
            i=6+4*self.config.lag+2+3+6+2+2;money=f[...,i]
            treasury=np.maximum(0.,self.config.initial_money+self.config.initial_treasury-money)
            f[...,i+2]=treasury;f[...,i+3]=money+treasury-self.config.initial_money-self.config.initial_treasury
        return super().encode(x)

    def public_model(self,frame,slope=None):
        model=super().public_model(frame,slope);model.__class__=StationaryMarket
        model.regime=self.regime;model.representation='raw'
        persistence=(1-2*self.regime.switch_probability)**np.arange(self.config.max_steps+1)
        model.env.recipe_schedule[:]=1.5+(model.env.recipes[0]-1.5)*persistence[:,None]
        return model


def translate(env,obs):
    x=obs.copy().reshape(env.config.history,env.frame_size)
    money_index=6+4*env.config.lag+2+3+6+2+2
    # Represent the same player balance in a hypothetical ledger with historical
    # minted accumulation removed, retaining nonnegative treasury and identity.
    money=x[:,money_index];treasury=np.maximum(0.,env.config.initial_money+env.config.initial_treasury-money)
    x[:,money_index+2]=treasury
    x[:,money_index+3]=money+treasury-env.config.initial_money-env.config.initial_treasury
    return x.ravel()
