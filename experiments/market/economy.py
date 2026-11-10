"""Aggregate fluid task economy for a controlled five-mechanism screen.

This is a separate, uncalibrated reduction: fractional quantities represent
population mass, not fractional purchases by individual players. All controls
are market-wide bids/asks. Every public frame exposes queues, commitments,
household holdings and task ages; only the identification variant hides slope.
"""
from collections import deque
from dataclasses import asdict,dataclass
import math

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

CASES={
    'base':(), 'delay':('delay',), 'capacity':('capacity',),
    'habit':('habit',), 'waiting':('waiting',), 'identification':('identification',),
    'delay_capacity':('delay','capacity'), 'habit_waiting':('habit','waiting')}


@dataclass(frozen=True)
class TaskConfig:
    max_steps:int=96
    history:int=2
    lag:int=3
    initial_money:float=6000.
    initial_treasury:float=6000.
    mean_tasks:float=8.
    production_capacity:float=36.
    task_payment:float=4.
    initial_bid:float=8.5
    initial_ask:float=10.
    bid_low:float=3.
    bid_high:float=12.
    ask_low:float=6.
    ask_high:float=16.
    spread:float=.5
    price_step:float=.05
    reservation:float=7.5
    production_temperature:float=1.
    allocation_temperature:float=1.5
    base_slope:float=5.
    weak_slope:float=2.
    strong_slope:float=12.
    reference_rate:float=.15
    money_scale:float=.1
    price_weight:float=.03
    backlog_weight:float=.1
    inventory_weight:float=.01

    def __post_init__(self):
        for name in ('max_steps','history','lag'):
            if type(getattr(self,name)) is not int or getattr(self,name)<=0:raise ValueError('Positive integer dimensions required')
        for name,value in asdict(self).items():
            if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or value<=0:
                raise ValueError(f'Invalid positive parameter: {name}')
        if not (0<self.price_step<1 and 0<self.reference_rate<1 and self.max_steps>self.lag):raise ValueError('Invalid rates/horizon')
        if not (self.bid_low<self.initial_bid<self.bid_high and self.ask_low<self.initial_ask<self.ask_high
                and self.bid_low+self.spread<=self.ask_low and self.bid_high+self.spread<=self.ask_high
                and self.initial_bid+self.spread<=self.initial_ask):raise ValueError('Invalid prices/spread')
        if self.weak_slope>=self.strong_slope:raise ValueError('Invalid hidden slope support')


class TaskMarket(gym.Env):
    """Frame layout is fixed across all eight variants; no feature/mode labels."""
    metadata={'render_modes':[]}

    def __init__(self,config=None,case='base'):
        self.config=config or TaskConfig()
        if case not in CASES:raise ValueError('Unknown feature case')
        self.case=case;self.features=set(CASES[case]);self.recipe=np.array([1.5,1.5])
        self.target_stock=self.recipe*self.config.mean_tasks*2
        self.stock_cap=self.target_stock*3
        self.action_space=Box(-self.config.price_step,self.config.price_step,(4,),np.float32)
        # prices4, stocks2, pipeline units2L, pipeline cash2L, household2,
        # task ages3, recipes6, reference2, trend2, money/delta2,
        # treasury/minted2, supply2, sales2,
        # accepted2, eligible2, completed1.
        self.frame_size=34+4*self.config.lag
        self.observation_space=Box(-np.inf,np.inf,(self.frame_size*self.config.history,),np.float32)
        self.frames=deque(maxlen=self.config.history);self.ready=False

    def frame(self):
        return np.r_[self.prices,self.stock,self.pipeline.ravel(),self.commitments.ravel(),
            self.household,self.tasks,self.recipes.ravel(),self.reference,self.trend,self.money,self.delta,self.treasury,self.minted,
            self.last_supply,self.last_sales,self.last_accepted,self.last_eligible,self.last_completed].astype(np.float32)

    def unpack(self,frame):
        x=np.asarray(frame,dtype=float);i=6;l=self.config.lag
        result=dict(prices=x[:4].copy(),stock=x[4:6].copy())
        for name in ('pipeline','commitments'):
            result[name]=x[i:i+2*l].reshape(l,2).copy();i+=2*l
        for name,n in [('household',2),('tasks',3)]:
            result[name]=x[i:i+n].copy();i+=n
        result['recipes']=x[i:i+6].reshape(3,2).copy();i+=6
        for name in ('reference','trend'):
            result[name]=x[i:i+2].copy();i+=2
        result['money'],result['delta']=x[i:i+2];i+=2
        result['treasury'],result['minted']=x[i:i+2];i+=2
        for name in ('last_supply','last_sales','last_accepted','last_eligible'):
            result[name]=x[i:i+2].copy();i+=2
        result['last_completed']=x[i];i+=1
        if i!=self.frame_size or len(x)!=i:raise ValueError('Frame dimension mismatch')
        return result

    def observe(self):
        return np.concatenate(self.frames)

    def encode(self,observation):
        x=np.asarray(observation,dtype=np.float32)
        if x.shape[-1]!=self.frame_size*self.config.history:raise ValueError('Observation dimension mismatch')
        f=x.reshape(*x.shape[:-1],self.config.history,self.frame_size).copy();c=self.config;l=c.lag
        f[...,:4]=(f[...,:4]-np.array([c.initial_bid]*2+[c.initial_ask]*2))/5
        f[...,4:6]/=self.stock_cap
        f[...,6:6+2*l]/=c.production_capacity
        f[...,6+2*l:6+4*l]/=c.production_capacity*c.initial_bid
        i=6+4*l
        f[...,i:i+2]/=self.target_stock;i+=2
        f[...,i:i+3]/=c.mean_tasks;i+=3
        f[...,i:i+6]/=2.;i+=6
        f[...,i:i+2]=(f[...,i:i+2]-c.initial_ask)/5;i+=2
        f[...,i:i+2]/=c.price_step;i+=2
        f[...,i]=np.arcsinh((f[...,i]/c.initial_money-1)/c.money_scale)/3;i+=1
        f[...,i]=np.arcsinh(f[...,i]/(c.mean_tasks*c.initial_ask));i+=1
        f[...,i:i+2]=np.arcsinh(f[...,i:i+2]/c.initial_treasury);i+=2
        for _ in range(4):f[...,i:i+2]/=self.recipe*c.mean_tasks;i+=2
        f[...,i]/=c.mean_tasks
        return f.reshape(x.shape)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if options:raise ValueError('No reset options')
        c=self.config;generator=np.random.default_rng(np.random.SeedSequence([int(seed),5813])) if seed is not None else self.np_random
        self.arrivals=generator.poisson(c.mean_tasks,c.max_steps+1).astype(float)
        self.productivity=generator.uniform(.94,1.06,(c.max_steps,2))
        self.preference=generator.normal(0,.06,(c.max_steps,2))
        phase=int(generator.integers(2));recipe_schedule=[]
        for switch in generator.uniform(size=c.max_steps+1):
            if switch<.06:phase=1-phase
            recipe_schedule.append([2.,1.] if phase else [1.,2.])
        self.recipe_schedule=np.asarray(recipe_schedule)
        mode=(np.random.default_rng(np.random.SeedSequence([int(seed),905])).integers(2)
              if seed is not None else self.np_random.integers(2))
        self.slope=(c.strong_slope if mode else c.weak_slope) if 'identification' in self.features else c.base_slope
        self.prices=np.array([c.initial_bid]*2+[c.initial_ask]*2)
        self.stock=self.target_stock.copy();self.pipeline=np.zeros((c.lag,2));self.commitments=np.zeros((c.lag,2))
        self.household=np.zeros(2);self.tasks=np.array([self.arrivals[0],0.,0.])
        self.recipes=np.tile(self.recipe_schedule[0],(3,1))
        self.reference=np.full(2,c.initial_ask);self.trend=np.zeros(2)
        self.money=c.initial_money;self.treasury=c.initial_treasury;self.delta=0.;self.minted=0.
        self.last_supply=np.zeros(2);self.last_sales=np.zeros(2);self.last_accepted=np.zeros(2);self.last_eligible=np.zeros(2);self.last_completed=0.
        self.elapsed_steps=0;self.ready=True;self.frames.clear();self.frames.extend(self.frame().copy() for _ in range(c.history))
        return self.observe(),{}

    def supply(self,bids,productivity=None):
        c=self.config;w=self.recipe/self.recipe.sum();noise=np.ones(2) if productivity is None else productivity
        sigmoid=lambda z:1/(1+np.exp(np.clip(-z,-50,50)))
        if 'capacity' in self.features:
            z=(np.asarray(bids)-c.reservation)/c.allocation_temperature+np.log(w);a=np.exp(z-z.max());a/=a.sum()
            participation=sigmoid((float(w@bids)-c.reservation)/c.production_temperature)
            return c.production_capacity*participation*a*noise
        return c.production_capacity*w*sigmoid((np.asarray(bids)-c.reservation)/c.production_temperature)*noise

    def acceptance(self,asks,preference=None):
        c=self.config;z=math.log(3)-self.slope*np.log(np.asarray(asks)/c.initial_ask)
        if 'habit' in self.features:
            ratio=np.log(self.reference/asks);z+=6*ratio*np.where(ratio<0,2.,1.)
        if preference is not None:z+=preference
        return 1/(1+np.exp(np.clip(-z,-50,50)))

    def public_model(self,frame,slope=None):
        """Construct a fresh model only from public inputs, never actual future RNG."""
        model=TaskMarket(self.config,self.case);model.reset(seed=0)
        for name,value in self.unpack(frame).items():setattr(model,name,value)
        model.slope=self.config.base_slope if slope is None else float(slope)
        # Reconcile only float32 public-ledger rounding, without hidden inputs.
        model.treasury=max(model.treasury,float(model.commitments.sum()))
        model.minted=model.money+model.treasury-self.config.initial_money-self.config.initial_treasury
        model.arrivals.fill(self.config.mean_tasks);model.productivity.fill(1.);model.preference.fill(0.)
        persistence=(1-2*.06)**np.arange(self.config.max_steps+1)
        model.recipe_schedule[:]=1.5+(model.recipes[0]-1.5)*persistence[:,None]
        model.frames.clear();model.frames.extend(model.frame().copy() for _ in range(self.config.history))
        return model

    def step(self,action):
        if not self.ready or self.elapsed_steps>=self.config.max_steps:raise RuntimeError('Reset before stepping')
        action=np.asarray(action,dtype=np.float32)
        if not self.action_space.contains(action):raise ValueError('Invalid price action')
        c=self.config;t=self.elapsed_steps;old_prices=self.prices.copy();old_money=self.money
        goods_before=self.stock+self.household+self.pipeline.sum(0)
        self.prices=np.clip(old_prices*(1+action),[c.bid_low]*2+[c.ask_low]*2,[c.bid_high]*2+[c.ask_high]*2)
        floor=np.maximum(c.bid_low,old_prices[:2]*(1-c.price_step))
        self.prices[2:]=np.maximum(self.prices[2:],floor+c.spread)
        self.prices[:2]=np.minimum(self.prices[:2],self.prices[2:]-c.spread)
        due=self.pipeline[0].copy();payout=float(self.commitments[0].sum())
        self.pipeline[:-1]=self.pipeline[1:].copy();self.pipeline[-1]=0
        self.commitments[:-1]=self.commitments[1:].copy();self.commitments[-1]=0
        self.stock+=due;self.money+=payout;self.treasury-=payout
        started=np.minimum(self.supply(self.prices[:2],self.productivity[t]),np.maximum(0,self.stock_cap-self.stock-self.pipeline.sum(0)))
        cost=float(started@self.prices[:2]);free=max(0.,self.treasury-self.commitments.sum())
        if cost>free:started*=free/max(cost,1e-12)
        contract=started*self.prices[:2]
        if 'delay' in self.features:
            self.pipeline[-1]+=started;self.commitments[-1]+=contract
        else:
            self.stock+=started;payout+=float(contract.sum());self.money+=float(contract.sum());self.treasury-=float(contract.sum())
        buffer=np.zeros(2);timing=1.
        if 'waiting' in self.features:
            buffer=c.mean_tasks*2*self.recipes[0]*np.clip(np.log(self.reference/self.prices[2:])/.15,0,1)
            urgent=self.tasks[-1]/max(self.tasks.sum(),1e-12)
            timing=urgent+(1-urgent)/(1+np.exp(np.clip(-(2*self.trend+.005)/.015,-50,50)))
        eligible=np.maximum(0,(self.tasks[:,None]*self.recipes).sum(0)+buffer-self.household)
        accepted=eligible*self.acceptance(self.prices[2:],self.preference[t])*timing
        bought=np.minimum(accepted,self.stock);purchase=float(bought@self.prices[2:])
        if purchase>self.money:bought*=max(0,self.money)/max(purchase,1e-12)
        purchase=float(bought@self.prices[2:]);self.money-=purchase;self.treasury+=purchase
        self.stock-=bought;self.household+=bought
        completed=0.;consumed=np.zeros(2)
        for age in (2,1,0):
            used=min(float(self.tasks[age]),float(np.min(self.household/self.recipes[age])))
            used=max(0.,used);goods=used*self.recipes[age]
            self.household-=goods;consumed+=goods;self.tasks[age]-=used;completed+=used
        expired=float(self.tasks[-1] if 'waiting' in self.features else self.tasks.sum())
        retained=self.tasks[:2].copy() if 'waiting' in self.features else np.zeros(2)
        mint=completed*c.task_payment;self.money+=mint;self.minted+=mint
        self.delta=self.money-old_money
        change=float(np.abs(np.log(self.prices/old_prices)).mean())
        money_error=abs(self.money/c.initial_money-1)
        costs=dict(money=money_error/c.money_scale,service=(expired+c.backlog_weight*retained.sum())/c.mean_tasks,
            inventory=c.inventory_weight*float((self.stock+self.household).sum()/self.stock_cap.sum()),
            price=c.price_weight*change/-math.log1p(-c.price_step))
        self.trend=.5*self.trend+.5*np.log(self.prices[2:]/old_prices[2:])
        self.reference=np.exp((1-c.reference_rate)*np.log(self.reference)+c.reference_rate*np.log(self.prices[2:]))
        self.elapsed_steps+=1;self.tasks=np.r_[self.arrivals[self.elapsed_steps],retained]
        self.recipes=np.vstack([self.recipe_schedule[self.elapsed_steps],self.recipes[:2]])
        self.last_supply=due+started if 'delay' not in self.features else due
        self.last_sales=bought;self.last_accepted=accepted;self.last_eligible=eligible;self.last_completed=completed
        residual=self.stock+self.household+self.pipeline.sum(0)-goods_before-started+consumed
        cash_residual=self.money+self.treasury-c.initial_money-c.initial_treasury-self.minted
        if (np.any(np.abs(residual)>1e-7) or abs(cash_residual)>1e-7 or min(self.money,self.treasury)<-1e-8
                or np.any(self.stock>self.stock_cap+1e-8) or self.commitments.sum()>self.treasury+1e-7):raise AssertionError('Ledger/capacity failure')
        self.frames.append(self.frame())
        info=dict(reward=-sum(costs.values()),money_error=money_error,completed=completed,expired=expired,
            arrivals=float(self.arrivals[t]),pending_tasks=float(retained.sum()),household=float(self.household.sum()),
            stock=float(self.stock.sum()),pending_goods=float(self.pipeline.sum()),pending_cash=float(self.commitments.sum()),
            payout=payout,purchase=purchase,mint=mint,money=self.money,treasury=self.treasury,price_change=change,
            money_cost=costs['money'],service_cost=costs['service'],inventory_cost=costs['inventory'],price_cost=costs['price'],
            cash_residual=cash_residual,goods_residual=float(np.max(np.abs(residual))),hidden_slope=self.slope)
        return self.observe(),info['reward'],False,self.elapsed_steps>=c.max_steps,info
