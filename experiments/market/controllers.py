"""Feedback, inventory-aware, joint-price and anchored PI controllers."""
from copy import deepcopy
import numpy as np

class FeedbackController:
    def __init__(self,env,method,gain=1.,stock_gain=.5,integral_gain=0.,lookahead=4):
        if method not in ('fixed','p','pi','predictive','adaptive','oracle'):raise ValueError('Unknown controller')
        self.env=env;self.method=method;self.gain=gain;self.stock_gain=stock_gain;self.integral_gain=integral_gain
        self.lookahead=lookahead;self.integral=0.;self.calls=0;self.samples=[]

    def slope_estimate(self,state):
        c=self.env.config
        for price,accepted,eligible in zip(state['prices'][2:],state['last_accepted'],state['last_eligible']):
            if eligible>1e-5:
                ratio=np.clip(accepted/eligible,1e-5,1-1e-5)
                self.samples.append((np.log(price/c.initial_ask),np.log(ratio/(1-ratio))-np.log(3)))
        self.samples=self.samples[-32:]
        if not self.samples:return c.base_slope
        x,y=np.asarray(self.samples).T
        return float(np.clip(-(x@y)/(x@x+.002),1.,20.))

    def proposal(self,state,predictive=False):
        c=self.env.config;error=state['money']/c.initial_money-1
        inventory=state['stock'].copy()
        if predictive:
            inventory+=state['pipeline'].sum(0)
            error+=state['commitments'].sum()/c.initial_money
        average_recipe=state['tasks']@state['recipes']/max(float(state['tasks'].sum()),1e-12) if state['tasks'].sum()>0 else state['recipes'][0]
        target=c.mean_tasks*2*average_recipe
        stock_error=inventory/target-1
        bid=c.initial_bid*np.exp(np.clip(-self.gain*error-self.integral-self.stock_gain*stock_error,-2,2))
        ask=c.initial_ask*np.exp(np.clip(.25*self.gain*error-.35*self.stock_gain*stock_error,-2,2))
        return np.clip(np.r_[bid,ask]/state['prices']-1,-c.price_step,c.price_step).astype(np.float32)

    def act(self,observation):
        c=self.env.config;frame=np.asarray(observation)[-self.env.frame_size:];state=self.env.unpack(frame)
        self.integral=float(np.clip(self.integral+self.integral_gain*(state['money']/c.initial_money-1),-.5,.5))
        self.calls+=1
        if self.method=='fixed':return np.zeros(4,np.float32)
        if self.method in ('p','pi'):return self.proposal(state)
        slope=c.base_slope
        if self.method=='adaptive' and 'identification' in self.env.features:
            slope=self.slope_estimate(state)
            if self.calls<=4:
                action=self.proposal(state,True);action[2:]=(.02 if self.calls%2 else -.02)*np.array([1.,-1.]);return action
        elif self.method=='oracle':slope=self.env.slope
        model=self.env.public_model(frame,slope)
        candidates=[np.zeros(4,np.float32),self.proposal(state,True)]
        for i in range(4):
            for sign in (-1,1):
                a=np.zeros(4,np.float32);a[i]=sign*c.price_step;candidates.append(a)
        for i in (0,2):
            for sign in (-1,1):
                a=np.zeros(4,np.float32);a[i:i+2]=sign*c.price_step;candidates.append(a)
        values=[]
        for action in candidates:
            fork=deepcopy(model);value=0.
            for step in range(self.lookahead):
                _,reward,_,_,_=fork.step(action if step==0 else np.zeros(4,np.float32));value+=.97**step*reward
            values.append(value)
        return candidates[int(np.argmax(values))].copy()


class InventoryController(FeedbackController):
    def __init__(self,env,method,age_gain=0.,**kwargs):
        self.clearance=method=='clearance';self.age_gain=age_gain
        super().__init__(env,'pi' if self.clearance else method,**kwargs)

    def proposal(self,state,predictive=False):
        c=self.env.config;error=state['money']/c.initial_money-1;inventory=state['stock'].copy()
        if predictive:inventory+=state['pipeline'].sum(0);error+=state['commitments'].sum()/c.initial_money
        average=state['tasks']@state['recipes']/max(float(state['tasks'].sum()),1e-12) if state['tasks'].sum()>0 else state['recipes'][0]
        target=np.minimum(c.mean_tasks*2*average,self.env.stock_cap*.75)
        stock_error=inventory/target-1;pressure=np.zeros(2);life=self.env.parameters.shelf_life
        if self.clearance and life:
            pressure=state['shop_ages'][life-1]/target
            if life>1:pressure+=.5*state['shop_ages'][life-2]/target
        bid=c.initial_bid*np.exp(np.clip(-self.gain*error-self.integral-self.stock_gain*stock_error-.5*self.age_gain*pressure,-2,2))
        ask=c.initial_ask*np.exp(np.clip(.25*self.gain*error-.35*self.stock_gain*stock_error-self.age_gain*pressure,-2,2))
        return np.clip(np.r_[bid,ask]/state['prices']-1,-c.price_step,c.price_step).astype(np.float32)


class JointController(InventoryController):
    def __init__(self, env, method, allocation_gain=1., **kwargs):
        self.requested_method = method
        self.allocation_gain = allocation_gain
        super().__init__(env, 'pi' if method in ('joint', 'rollout') else method, **kwargs)

    def proposal(self, state, predictive=False):
        action = super().proposal(state, predictive)
        if self.requested_method not in ('joint', 'rollout'):
            return action
        c = self.env.config
        recipe = (state['tasks'] @ state['recipes'] / max(state['tasks'].sum(), 1e-12)
                  if state['tasks'].sum() else state['recipes'][0])
        # Analytic softmax allocation feedforward. Relative bids steer shared
        # capacity toward the currently required recipe; PI still controls cash.
        differential = c.allocation_temperature * np.log(np.maximum(recipe / recipe.mean(), .1))
        differential -= differential.mean()
        error = state['money'] / c.initial_money - 1
        stock = state['stock'] + (state['pipeline'].sum(0) if predictive else 0)
        target = np.minimum(c.mean_tasks * 2 * recipe, self.env.stock_cap * .75)
        prices = c.initial_bid * np.exp(np.clip(-self.gain * error - self.integral
                                                - self.stock_gain * (stock / target - 1), -2, 2))
        desired = prices + self.allocation_gain * differential
        action[:2] = np.clip(desired / state['prices'][:2] - 1, -c.price_step, c.price_step)
        return action.astype(np.float32)

    def act(self, observation):
        if self.requested_method != 'rollout':
            return super().act(observation)
        c = self.env.config
        frame = np.asarray(observation)[-self.env.frame_size:]
        state = self.env.unpack(frame)
        self.integral = float(np.clip(self.integral + self.integral_gain *
                                      (state['money'] / c.initial_money - 1), -.5, .5))
        model = self.env.public_model(frame)
        candidates = [np.zeros(4, np.float32), self.proposal(state, True)]
        # Original coordinate and same-side actions, plus coordinated bid/ask
        # and differential moves missing from the older predictor's menu.
        directions = list(np.eye(4))
        directions += [np.array(v) for v in ((1,1,0,0),(0,0,1,1),(1,1,1,1),
                       (1,1,-1,-1),(1,-1,0,0),(0,0,1,-1),(1,-1,1,-1),(1,-1,-1,1))]
        for direction in directions:
            for sign in (-1, 1):
                candidates.append((sign * c.price_step * direction).astype(np.float32))
        values = []
        for action in candidates:
            fork = deepcopy(model)
            # Compare two declared continuation policies rather than assuming
            # that the chosen first move will be followed by frozen prices.
            for follow_feedback in (False, True):
                trial = deepcopy(fork)
                feedback = JointController(trial, 'joint', gain=self.gain,
                    stock_gain=self.stock_gain, integral_gain=self.integral_gain,
                    allocation_gain=self.allocation_gain)
                feedback.integral = self.integral
                value = 0.
                for t in range(self.lookahead):
                    move = action if t == 0 else feedback.act(trial.observe()) if follow_feedback else np.zeros(4, np.float32)
                    _, reward, _, _, _ = trial.step(move)
                    value += .97 ** t * reward
                values.append((value, action))
        return max(values, key=lambda pair: pair[0])[1].copy()


class AnchoredPI(JointController):
    def __init__(self,env,bid_anchor,ask_anchor):
        super().__init__(env,'pi',gain=3.,stock_gain=0.,integral_gain=.06)
        self.bid_anchor=bid_anchor;self.ask_anchor=ask_anchor

    def proposal(self,state,predictive=False):
        c=self.env.config;error=state['money']/c.initial_money-1
        bids=np.full(2,self.bid_anchor*np.exp(np.clip(-3*error-self.integral,-2,2)))
        asks=np.full(2,self.ask_anchor*np.exp(np.clip(.75*error,-2,2)))
        return np.clip(np.r_[bids,asks]/state['prices']-1,-c.price_step,c.price_step).astype(np.float32)
