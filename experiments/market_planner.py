"""Batched expected-model/SAA price planning for instantaneous nonperishables.

All forecast randomness is private, independent of actual future market draws.
Full Cartesian first-action menu; hold or joint-PI continuation. No oracle label.
"""
from itertools import product
import numpy as np

from .market.controllers import JointController


class MarketPlanner(JointController):
    def __init__(self,env,horizon=4,particles=1,forecast_seed=9017):
        super().__init__(env,'joint',gain=3.,stock_gain=.5,integral_gain=.03,allocation_gain=1.)
        self.horizon=horizon;self.particles=particles
        self.rng=np.random.default_rng(forecast_seed)

    def forecast(self,state,particles=None,horizon=None):
        f=particles or self.particles;h=horizon or self.horizon;c=self.env.config
        demand=np.full((f,h),c.mean_tasks);recipes=np.tile(state['recipes'][0],(f,h,1)).astype(float)
        productivity=np.ones((f,h,2));preference=np.zeros((f,h,2))
        if f==1:
            power=(1-2*self.env.regime.switch_probability)**np.arange(h)
            recipes[:]=1.5+(state['recipes'][0]-1.5)*power[:,None]
        else:
            demand=self.rng.poisson(c.mean_tasks,(f,h)).astype(float)
            productivity=self.rng.uniform(.94,1.06,(f,h,2));preference=self.rng.normal(0,.06,(f,h,2))
            phase=np.full(f,state['recipes'][0,0]>1.5)
            switches=self.rng.uniform(size=(f,h))<self.env.regime.switch_probability
            for t in range(1,h):
                phase^=switches[:,t];recipes[:,t,0]=np.where(phase,2.,1.);recipes[:,t,1]=3-recipes[:,t,0]
        demand[:,0]=state['tasks'][0]
        return dict(demand=demand,recipes=recipes,productivity=productivity,preference=preference)

    def score_actions(self,state,actions,forecast,return_paths=False):
        c=self.env.config;actions=np.asarray(actions,dtype=np.float32)
        n=len(actions);f,h=forecast['demand'].shape
        shape=(n,2,f);prices=np.broadcast_to(state['prices'],shape+(4,)).copy()
        stock=np.broadcast_to(state['stock'],shape+(2,)).copy();house=np.broadcast_to(state['household'],shape+(2,)).copy()
        money=np.full(shape,state['money']);treasury=np.full(shape,state['treasury']);integral=np.full(shape,self.integral);value=np.zeros(shape)
        paths=[];sigmoid=lambda z:1/(1+np.exp(np.clip(-z,-50,50)))
        for t in range(h):
            recipe=forecast['recipes'][:,t];tasks=forecast['demand'][:,t]
            if t==0:move=np.broadcast_to(actions[:,None,None,:],shape+(4,))
            else:
                error=money/c.initial_money-1;integral=np.clip(integral+self.integral_gain*error,-.5,.5)
                target=np.minimum(c.mean_tasks*2*recipe,self.env.stock_cap*.75)
                se=stock/target-1
                bid=c.initial_bid*np.exp(np.clip(-3*error[...,None]-integral[...,None]-.5*se,-2,2))
                diff=c.allocation_temperature*np.log(np.maximum(recipe/recipe.mean(-1,keepdims=True),.1));diff-=diff.mean(-1,keepdims=True)
                bid+=diff
                ask=c.initial_ask*np.exp(np.clip(.75*error[...,None]-.175*se,-2,2))
                move=np.clip(np.concatenate([bid,ask],-1)/prices-1,-c.price_step,c.price_step).astype(np.float32)
                move[:,0]=0.  # held-price continuation; second branch is feedback.
            old=prices.copy();prices=np.clip(old*(1+move),[c.bid_low]*2+[c.ask_low]*2,[c.bid_high]*2+[c.ask_high]*2)
            floor=np.maximum(c.bid_low,old[...,:2]*(1-c.price_step))
            prices[...,2:]=np.maximum(prices[...,2:],floor+c.spread)
            prices[...,:2]=np.minimum(prices[...,:2],prices[...,2:]-c.spread)
            z=(prices[...,:2]-c.reservation)/c.allocation_temperature+np.log(.5)
            allocation=np.exp(z-z.max(-1,keepdims=True));allocation/=allocation.sum(-1,keepdims=True)
            participation=sigmoid((prices[...,:2].mean(-1)-c.reservation)/c.production_temperature)
            supply=c.production_capacity*participation[...,None]*allocation*forecast['productivity'][:,t]
            supply=np.minimum(supply,np.maximum(0,self.env.stock_cap-stock))
            cost=(supply*prices[...,:2]).sum(-1);supply*=np.minimum(1,np.maximum(0,treasury)/np.maximum(cost,1e-12))[...,None]
            payout=(supply*prices[...,:2]).sum(-1);stock+=supply;money+=payout;treasury-=payout
            eligible=np.maximum(0,tasks[...,None]*recipe-house)
            acceptance=sigmoid(np.log(3)-c.base_slope*np.log(prices[...,2:]/c.initial_ask)+forecast['preference'][:,t])
            bought=np.minimum(eligible*acceptance,stock);cost=(bought*prices[...,2:]).sum(-1)
            bought*=np.minimum(1,np.maximum(0,money)/np.maximum(cost,1e-12))[...,None]
            purchase=(bought*prices[...,2:]).sum(-1);stock-=bought;house+=bought;money-=purchase;treasury+=purchase
            complete=np.maximum(0,np.minimum(tasks,np.min(house/recipe,axis=-1)));house-=complete[...,None]*recipe
            money+=complete*c.task_payment
            change=np.abs(np.log(prices/old)).mean(-1)
            reward=-(np.abs(money/c.initial_money-1)/c.money_scale+(tasks-complete)/c.mean_tasks
                +c.inventory_weight*(stock+house).sum(-1)/(18*c.mean_tasks)+c.price_weight*change/-np.log1p(-c.price_step))
            value+=.97**t*reward
            if return_paths:paths.append(dict(prices=prices.copy(),stock=stock.copy(),household=house.copy(),money=money.copy(),treasury=treasury.copy(),reward=reward.copy(),completed=complete.copy()))
        scores=value.mean(-1).max(-1)
        return (scores,paths) if return_paths else scores

    def act(self,observation):
        state=self.env.public_model(np.asarray(observation)[-self.env.frame_size:]).env
        # Reconstructing via the public model reconciles float32 ledger rounding.
        state=dict(prices=state.prices,stock=state.stock,household=state.household,money=state.money,treasury=state.treasury,
                   tasks=state.tasks,recipes=state.recipes,pipeline=state.pipeline,commitments=state.commitments)
        self.integral=float(np.clip(self.integral+self.integral_gain*(state['money']/self.env.config.initial_money-1),-.5,.5))
        menu=np.array(list(product((-1,0,1),repeat=4)),np.float32)*np.float32(self.env.config.price_step)
        menu=np.vstack([menu,self.proposal(state,True)])
        score=self.score_actions(state,menu,self.forecast(state))
        return menu[int(np.argmax(score))].copy()
