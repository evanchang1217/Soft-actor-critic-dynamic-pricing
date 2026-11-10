"""Storage/capacity and FIFO perishability on the archived aggregate market.

Wrapper preserves the original market implementation and old result sources.
Every method sees the same shop/household age bins; prices remain per good,
not individualized or separately set for every age. Shelf age starts on delivery.
"""
from collections import deque
from dataclasses import asdict,dataclass,replace
import math

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

from .economy import TaskConfig,TaskMarket


@dataclass(frozen=True)
class InventoryParameters:
    buffer_periods:float=6.
    capacity_ratio:float=1.5
    shelf_life:int=0
    max_age:int=6
    waste_weight:float=.25

    def __post_init__(self):
        for name in ('buffer_periods','capacity_ratio','waste_weight'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError('Invalid inventory scale')
        if type(self.max_age) is not int or self.max_age<=0:raise ValueError('Invalid age dimension')
        if type(self.shelf_life) is not int or not 0<=self.shelf_life<=self.max_age:raise ValueError('Invalid shelf life')


def remove_fifo(bins,quantity):
    """Mutate age bins; return the removed per-age/per-good mass."""
    remaining=np.asarray(quantity,dtype=float).copy();taken=np.zeros_like(bins)
    if np.any(remaining < -1e-7) or np.any(remaining>bins.sum(0)+1e-7):raise AssertionError('FIFO request exceeds holdings')
    remaining=np.maximum(remaining,0.)
    for age in range(len(bins)-1,-1,-1):
        used=np.minimum(bins[age],remaining);bins[age]-=used;taken[age]=used;remaining-=used
    if np.any(remaining>1e-7):raise AssertionError('Unfilled FIFO transfer')
    return taken


def age_and_expire(bins,life):
    waste=np.zeros(2)
    if life:
        waste=bins[life-1].copy();bins[life-1]=0.
    following=np.zeros_like(bins);following[1:]=bins[:-1]
    if not life:following[-1]+=bins[-1]
    return following,waste


class InventoryMarket(gym.Wrapper):
    def __init__(self,config=None,parameters=None,case='delay_capacity'):
        c=config or TaskConfig();self.parameters=parameters or InventoryParameters()
        physical=replace(c,production_capacity=3*c.mean_tasks*self.parameters.capacity_ratio)
        super().__init__(TaskMarket(physical,case))
        self.env.stock_cap=self.env.recipe*c.mean_tasks*self.parameters.buffer_periods
        self.base_size=self.env.frame_size;self.frame_size=self.base_size+4*self.parameters.max_age
        self.observation_space=Box(-np.inf,np.inf,(self.frame_size*c.history,),np.float32)
        self.frames=deque(maxlen=c.history)
        self.shop_ages=np.zeros((self.parameters.max_age,2));self.house_ages=np.zeros_like(self.shop_ages)

    @property
    def config(self):return self.env.config
    @property
    def features(self):return self.env.features
    @property
    def recipe(self):return self.env.recipe
    @property
    def target_stock(self):return self.env.target_stock
    @property
    def stock_cap(self):return self.env.stock_cap

    def frame(self):return np.r_[self.env.frame(),self.shop_ages.ravel(),self.house_ages.ravel()].astype(np.float32)
    def observe(self):return np.concatenate(self.frames)

    def unpack(self,frame):
        x=np.asarray(frame);state=self.env.unpack(x[:self.base_size]);n=2*self.parameters.max_age
        if len(x)!=self.frame_size:raise ValueError('Inventory frame dimension mismatch')
        state['shop_ages']=x[self.base_size:self.base_size+n].reshape(-1,2).astype(float).copy()
        state['house_ages']=x[self.base_size+n:].reshape(-1,2).astype(float).copy()
        return state

    def encode(self,observation):
        x=np.asarray(observation,dtype=np.float32)
        if x.shape[-1]!=self.frame_size*self.config.history:raise ValueError('Inventory observation dimension mismatch')
        f=x.reshape(*x.shape[:-1],self.config.history,self.frame_size)
        base=f[...,:self.base_size].copy().reshape(*x.shape[:-1],self.config.history*self.base_size)
        encoded=self.env.encode(base).reshape(*x.shape[:-1],self.config.history,self.base_size)
        ages=f[...,self.base_size:].copy().reshape(*x.shape[:-1],self.config.history,2,self.parameters.max_age,2)
        ages/=self.target_stock
        return np.concatenate([encoded,ages.reshape(*x.shape[:-1],self.config.history,-1)],axis=-1).reshape(x.shape)

    def reset(self,*,seed=None,options=None):
        self.env.reset(seed=seed,options=options)
        self.env.stock=np.minimum(self.env.target_stock,self.stock_cap).copy()
        self.shop_ages.fill(0.);self.house_ages.fill(0.);self.shop_ages[0]=self.env.stock
        self.env.frames.clear();self.env.frames.extend(self.env.frame().copy() for _ in range(self.config.history))
        self.frames.clear();self.frames.extend(self.frame().copy() for _ in range(self.config.history))
        return self.observe(),{}

    def public_model(self,frame,slope=None):
        model=InventoryMarket(self.config,self.parameters,self.env.case);model.reset(seed=0)
        state=self.unpack(frame)
        for name,value in state.items():
            if name in ('shop_ages','house_ages'):setattr(model,name,value)
            else:setattr(model.env,name,value)
        # Public float32 age totals may differ slightly from rounded totals.
        totals=model.shop_ages.sum(0);factor=np.minimum(1.,model.stock_cap/np.maximum(totals,1e-12))
        model.shop_ages*=factor;model.env.stock=model.shop_ages.sum(0);model.env.household=model.house_ages.sum(0)
        model.env.treasury=max(model.env.treasury,float(model.env.commitments.sum()))
        model.env.minted=model.env.money+model.env.treasury-self.config.initial_money-self.config.initial_treasury
        model.env.slope=self.config.base_slope if slope is None else float(slope)
        model.env.arrivals.fill(self.config.mean_tasks);model.env.productivity.fill(1.);model.env.preference.fill(0.)
        persistence=(1-2*.06)**np.arange(self.config.max_steps+1)
        model.env.recipe_schedule[:]=1.5+(model.env.recipes[0]-1.5)*persistence[:,None]
        model.frames.clear();model.frames.extend(model.frame().copy() for _ in range(self.config.history))
        return model

    def step(self,action):
        house_before=self.env.household.copy()
        _,reward,term,trunc,info=self.env.step(action)
        self.shop_ages[0]+=self.env.last_supply
        transfer=remove_fifo(self.shop_ages,self.env.last_sales);self.house_ages+=transfer
        consumed=house_before+self.env.last_sales-self.env.household
        remove_fifo(self.house_ages,consumed)
        if (np.max(abs(self.shop_ages.sum(0)-self.env.stock))>1e-7
                or np.max(abs(self.house_ages.sum(0)-self.env.household))>1e-7):raise AssertionError('Age totals do not match holdings')
        before=self.env.stock+self.env.household+self.env.pipeline.sum(0)
        # Keep the holding-cost denominator fixed across storage capacities.
        holding=self.config.inventory_weight*float((self.env.stock+self.env.household).sum())/(18*self.config.mean_tasks)
        reward+=info['inventory_cost']-holding;info['inventory_cost']=holding
        self.shop_ages,shop_waste=age_and_expire(self.shop_ages,self.parameters.shelf_life)
        self.house_ages,house_waste=age_and_expire(self.house_ages,self.parameters.shelf_life)
        waste=shop_waste+house_waste;self.env.stock-=shop_waste;self.env.household-=house_waste
        residual=self.env.stock+self.env.household+self.env.pipeline.sum(0)-before+waste
        if np.max(abs(residual))>1e-7:raise AssertionError('Expiry goods ledger failure')
        waste_cost=self.parameters.waste_weight*float(waste.sum())/(3*self.config.mean_tasks)
        reward-=waste_cost
        info.update(reward=reward,waste=float(waste.sum()),shop_waste=float(shop_waste.sum()),house_waste=float(house_waste.sum()),
            waste_cost=waste_cost,stock=float(self.env.stock.sum()),household=float(self.env.household.sum()),
            goods_residual=max(info['goods_residual'],float(np.max(abs(residual)))))
        self.env.frames[-1]=self.env.frame();self.frames.append(self.frame())
        return self.observe(),reward,term,trunc,info



VARIANTS = {
    'shared_switch': (True, True),
    'separate_switch': (False, True),
    'shared_fixed': (True, False),
    'separate_fixed': (False, False),
}


class FactorialMarket(InventoryMarket):
    def __init__(self, config=None, variant='shared_switch'):
        if variant not in VARIANTS:
            raise ValueError('Unknown ablation')
        self.variant = variant
        shared, _ = VARIANTS[variant]
        super().__init__(config, InventoryParameters(), 'capacity' if shared else 'base')

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if not VARIANTS[self.variant][1]:
            # Preserve all RNG draws and total ingredient demand, remove mix changes.
            self.env.recipe_schedule[:] = 1.5
            self.env.recipes[:] = 1.5
            self.env.frames.clear()
            self.env.frames.extend(self.env.frame().copy() for _ in range(self.config.history))
            self.frames.clear()
            self.frames.extend(self.frame().copy() for _ in range(self.config.history))
        return self.observe(), {}

    def public_model(self, frame, slope=None):
        model = super().public_model(frame, slope)
        # Same wrapper state; attach only public experiment settings, no live RNG.
        model.__class__ = FactorialMarket
        model.variant = self.variant
        if not VARIANTS[self.variant][1]:
            model.env.recipe_schedule[:] = 1.5
        return model
