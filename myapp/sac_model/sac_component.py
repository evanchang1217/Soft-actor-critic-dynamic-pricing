#sac_model/sac_component.py

import time
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import matplotlib.pyplot as plt
import random
import os
from django.conf import settings
from gym.spaces import Box
"""
Soft Actor‑Critic (SAC) implementation for the multi‑product pricing environment.
整體流程：
1.  Twin Q‑networks + target Q‑networks (soft update)
2.  Gaussian policy network (re‑parameterization‑trick + Tanh) → 連續動作再縮放到 [-0.1,0.1]
3.  自動調節熵係數 α（可學也可固定）
4.  Off‑policy replay buffer，迷你 batch 更新
"""

# ────────────────────────── 超參數 ──────────────────────────
NUM_PLAYERS  =200
MIN_OFFSET    = -0.05
MAX_OFFSET    =  0.05
DROP_BOOST_EP = 50        # every N episodes drop ×2
FEE_MIN, FEE_MAX       = 0.01, 0.2      # auction fee bounds

AVG_QTY       = 5                # 假設每位願意交易玩家平均 5 單位
GAMMA         = 0.99
TAU           = 0.005          # soft‑update 係數
LR_ACTOR      = LR_CRITIC = LR_ALPHA = 3e-4
BATCH_SIZE    = 512
WARMUP_STEPS  = 1000

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

buy_price_bounds = []
sell_price_bounds = []

import numpy as np
import torch
import gym
from gym.spaces import Box

# 假設以下常數已在其他地方定義：
# NUM_PLAYERS, NUM_PRODUCTS, AVG_QTY, BUY_LOW, BUY_HIGH, SELL_LOW, SELL_HIGH
# MIN_OFFSET, MAX_OFFSET, FEE_MIN, FEE_MAX

def batch_decide_sell(inv, price_sens, fee_sens, base_supply, shop_buy, fee):
    """
    inv:            (NUM_PLAYERS, NUM_PRODUCTS) int
    price_sens:     (NUM_PLAYERS,) float
    fee_sens:       (NUM_PLAYERS,) float
    shop_buy:       (NUM_PRODUCTS,) float
    fee:            float
    回傳 qty_sell:  (NUM_PLAYERS, NUM_PRODUCTS) int
    """
    net = shop_buy * (1 - fee)  # (P,)
    price_pos = np.clip((net - BUY_LOW) / (BUY_HIGH - BUY_LOW), 0, 1)  # (P,)

    sell_prob = price_pos[None, :] * price_sens[:, None]  # (N,P)
    sell_prob /= (1 + fee_sens[:, None] * fee)
    sell_prob = np.clip(sell_prob, 0, 1)

    Ki = base_supply * 2
    inv = inv.astype(np.float32)
    inv_ratio = inv / (inv + Ki)  # (N,P)

    mask = np.random.rand(NUM_PLAYERS, NUM_PRODUCTS) < sell_prob
    lam = base_supply * sell_prob * inv_ratio
    qty = np.random.poisson(lam) * mask
    qty = np.minimum(qty, inv).astype(np.int32)

    return qty

def batch_decide_buy(gold, price_sens, fee_sens, base_demand, inv, prices, fee, rules_list=None):
    """
    gold:          (NUM_PLAYERS,) float
    price_sens:    (NUM_PLAYERS,) float
    fee_sens:      (NUM_PLAYERS,) float
    base_demand:   float
    inv:           (NUM_PLAYERS, NUM_PRODUCTS) int (unused here)
    prices:        (NUM_PRODUCTS,) float
    fee:           float
    回傳 qty_buy:  (NUM_PLAYERS, NUM_PRODUCTS) int
    """
    cost_unit = prices * (1 + fee)  # (P,)
    price_pos = np.clip(1 - (cost_unit - SELL_LOW) / (SELL_HIGH - SELL_LOW), 0, 1)

    buy_prob = price_pos[None, :] * price_sens[:, None]  # (N,P)
    buy_prob /= (1 + fee_sens[:, None] * fee)
    buy_prob = np.clip(buy_prob, 0, 1)
    if rules_list:
        for rule in rules_list:
            ant_idx = np.array([int(x) for x in rule["antecedent"]], dtype=np.int32)
            con_idx = int(rule["consequent"])

            # 伯努利抽樣：哪些玩家決定要買 antecedent 那幾樣
            mask_ante = np.random.rand(gold.shape[0], ant_idx.size) < buy_prob[:, ant_idx]
            buy_all_a = mask_ante.all(axis=1)    # (N,) 布林陣列

            if not np.any(buy_all_a):
                continue

            if rule["type"] == "complementary":
                conf = float(rule["confidence"])
                buy_prob[buy_all_a, con_idx] = np.maximum(
                    buy_prob[buy_all_a, con_idx],
                    conf
                )
            else:  # substitute
                prob = float(rule["probability"])
                buy_prob[buy_all_a, con_idx] *= (1.0 - prob)
    Kb = base_demand * cost_unit
    gold = gold.astype(np.float32)
    budget_ratio = gold[:, None] / (gold[:, None] + Kb[None, :]+ 1e-8)
    budget_ratio = np.clip(budget_ratio, 0, 1)

    mask = np.random.rand(NUM_PLAYERS, NUM_PRODUCTS) < (buy_prob * budget_ratio)
    lam = base_demand * buy_prob * budget_ratio
    qty = np.random.poisson(lam) * mask

    # cost constraint
    total_cost = (qty * cost_unit[None, :]).sum(axis=1)
    scale = np.minimum(1.0, gold / (total_cost + 1e-8))
    qty = np.floor(qty * scale[:, None]).astype(np.int32)

    return qty

# ────────────────────────── 環境 ──────────────────────────
class MultiStoreEnv(gym.Env):
    def __init__(self, association_rules=None):
        super().__init__()
        # 玩家屬性向量化
        self.association_rules = association_rules or []
        self.player_gold        = np.full((NUM_PLAYERS,), 1000.0, dtype=np.float32)
        self.player_inv         = np.zeros((NUM_PLAYERS, NUM_PRODUCTS), dtype=np.int32)
        self.price_sensitivity  = np.random.uniform(0.7, 1.3, size=(NUM_PLAYERS,)).astype(np.float32)
        self.fee_sensitivity    = np.random.uniform(0.5, 1.5, size=(NUM_PLAYERS,)).astype(np.float32)
        self.drop_rate          = np.full((NUM_PRODUCTS,), AVG_QTY, dtype=np.int32)
        self.store_buy_prices   = BUY_LOW.copy()
        self.store_sell_prices  = SELL_HIGH.copy()
        self.fee_rate           = 0.05
        self.pre_cpi            = None
        self.pre_money          = None

        low = np.concatenate([
            BUY_LOW, SELL_LOW,
            np.zeros(NUM_PRODUCTS*4),
            [0,0,0,0.1]
        ]).astype(np.float32)
        high = np.concatenate([
            BUY_HIGH, SELL_HIGH,
            np.full(NUM_PRODUCTS*4, NUM_PLAYERS*AVG_QTY),
            [1e9, 10, 1e9, 0.15]
        ]).astype(np.float32)

        self.observation_space = Box(low, high, dtype=np.float32)
        self.action_space = Box(
            low  = np.concatenate([np.full(NUM_PRODUCTS*2, MIN_OFFSET), [-0.01]]),
            high = np.concatenate([np.full(NUM_PRODUCTS*2, MAX_OFFSET), [ 0.01]]),
            dtype=np.float32
        )
        self.reset()

    def _mine(self):
        # 在每次礦產生成前，做一次隨機波動（例如 ±20% 內隨機）
        noise = np.random.normal(loc=1.0, scale=0.2, size=(NUM_PRODUCTS,))
        self.drop_rate = np.clip(self.drop_rate * noise, 1, AVG_QTY*5).astype(np.int32)
        self.player_inv += self.drop_rate[None, :]

    def step(self, action):
        # 解包動作
        po, fee_d = action[:-1], float(action[-1])
        self.fee_rate = np.clip(self.fee_rate + fee_d, FEE_MIN, FEE_MAX)
        self.store_buy_prices  = np.clip(self.store_buy_prices  * (1+po[0::2]), BUY_LOW,  BUY_HIGH)
        self.store_sell_prices = np.clip(self.store_sell_prices * (1+po[1::2]), SELL_LOW, SELL_HIGH)

        # 礦產事件
        self._mine()

        # 向量化決策
        qty_sell = batch_decide_sell(
            self.player_inv,
            self.price_sensitivity,
            self.fee_sensitivity,
            AVG_QTY,
            self.store_buy_prices,
            self.fee_rate
        )
        self.player_inv -= qty_sell
        player_rev = (qty_sell * self.store_buy_prices[None, :] * (1 - self.fee_rate)).sum(axis=1)
        self.player_gold += player_rev
        payout = player_rev.sum()

        store_buy_cnt = (qty_sell > 0).sum(axis=0).astype(np.float32)
        store_buy_vol = qty_sell.sum(axis=0).astype(np.float32)

        qty_buy = batch_decide_buy(
            self.player_gold,
            self.price_sensitivity,
            self.fee_sensitivity,
            AVG_QTY,
            self.player_inv,
            self.store_sell_prices,
            self.fee_rate,
            self.association_rules
        )
        cost = (qty_buy * self.store_sell_prices[None, :]).sum(axis=1) * (1 + self.fee_rate)
        self.player_gold -= cost
        self.player_inv  += qty_buy
        revenue = cost.sum()

        store_sell_cnt = (qty_buy > 0).sum(axis=0).astype(np.float32)
        store_sell_vol = qty_buy.sum(axis=0).astype(np.float32)

        sink = revenue - payout
        money = self.player_gold.sum()
        cpi   = (self.store_sell_prices / SELL_LOW).mean()
        loss_cpi = abs(cpi-self.pre_cpi)/self.pre_cpi
        loss_ms  = abs(money-self.pre_money)/(NUM_PLAYERS*self.pre_money)
        self.pre_cpi = cpi
        self.pre_money  = money

        # 計算 reward
        tgt_cnt = NUM_PLAYERS / 2
        tgt_vol = tgt_cnt * AVG_QTY
        loss_cnt = ((abs(store_buy_cnt - tgt_cnt) + abs(store_sell_cnt - tgt_cnt)) / tgt_cnt).sum()
        loss_vol = ((abs(store_buy_vol - tgt_vol) + abs(store_sell_vol - tgt_vol)) / tgt_vol).sum()
        reward   = -(loss_cnt+loss_vol+0.5*loss_cpi+0.5*loss_ms)

        self.state = np.concatenate([
            self.store_buy_prices, self.store_sell_prices,
            store_buy_cnt, store_sell_cnt,
            store_buy_vol, store_sell_vol,
            [sink, cpi, money, self.fee_rate]
        ]).astype(np.float32)

        return self.state.copy(), float(reward), False, {}

    def reset(self):
        self.player_gold[:]       = 1000.0
        self.player_inv[:]        = 0
        self.drop_rate[:]         = AVG_QTY
        self.store_buy_prices[:]  = np.random.uniform(BUY_LOW,  BUY_HIGH)
        self.store_sell_prices[:] = np.random.uniform(SELL_LOW, SELL_HIGH)
        self.fee_rate             = 0.05
        self.state                = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.pre_cpi              = 1.0
        self.pre_money            = self.player_gold.sum()
        return self.state.copy()

    def boost_drop(self):
        self.drop_rate *= 2

    def normalize(self, x):
        low  = torch.tensor(self.observation_space.low , dtype=torch.float32, device=x.device if torch.is_tensor(x) else DEVICE)
        high = torch.tensor(self.observation_space.high, dtype=torch.float32, device=x.device if torch.is_tensor(x) else DEVICE)
        return (x - low) / (high - low) * 2.0 - 1.0


# ──────────────────────── ReplayBuffer ────────────────────────
class ReplayBuffer:
    def __init__(self, cap=200_000):
        self.cap, self.buf, self.pos = cap, [], 0
    def push(self, *exp):
        if len(self.buf) < self.cap: self.buf.append(None)
        self.buf[self.pos] = exp
        self.pos = (self.pos + 1) % self.cap
    def sample(self, n):
        idx = np.random.choice(len(self.buf), n, replace=False)
        s,a,r,s2,d = zip(*[self.buf[i] for i in idx])
        return (np.stack(s), np.stack(a), np.float32(r).reshape(-1,1),
                np.stack(s2), np.float32(d).reshape(-1,1))
    def __len__(self): return len(self.buf)

# ──────────────────────── SAC 網路 ──────────────────────────
class GaussianPolicy(nn.Module):
    def __init__(self, s_dim, a_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu      = nn.Linear(hidden, a_dim)
        self.log_std = nn.Linear(hidden, a_dim)
        self.min_log_std = -20
        self.max_log_std =  2

    def forward(self, s):
        h = self.net(s)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(self.min_log_std, self.max_log_std)
        return mu, log_std

    def sample(self, s):
        mu, log_std = self(s)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()                    # reparameterization
        a = torch.tanh(x)                     # squash to [-1,1]
        a_scaled = a * MAX_OFFSET            # 再縮放到 ±0.1
        logp = dist.log_prob(x) - torch.log(1 - a.pow(2) + 1e-6)
        return a_scaled, logp.sum(dim=-1, keepdim=True)

class QNet(nn.Module):
    def __init__(self, s_dim, a_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),       nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, s, a):
        return self.net(torch.cat([s,a], dim=-1))

# ────────────────────────── Utils ──────────────────────────
@torch.no_grad()
def soft_update(src, tgt, tau):
    for p, tp in zip(src.parameters(), tgt.parameters()):
        tp.data.mul_(1 - tau).add_(tau * p.data)

# ────────────────────────── Main ──────────────────────────

def run_sac_training(num_episodes,max_steps,num_products,progress_callback=None,association_rules=None):


    global BUY_LOW, BUY_HIGH, SELL_LOW, SELL_HIGH
    global NUM_PRODUCTS
    global TARGET_ENTROPY 
    global STEPS_EP
    STEPS_EP=max_steps
    NUM_PRODUCTS=num_products
    TARGET_ENTROPY = -NUM_PRODUCTS*2    # 每維動作 -1 (~ld 0.5) 可按需微調
    BUY_LOW   = [0.0] * NUM_PRODUCTS
    BUY_HIGH  = [0.0] * NUM_PRODUCTS
    SELL_LOW  = [0.0] * NUM_PRODUCTS
    SELL_HIGH = [0.0] * NUM_PRODUCTS
    # 先用 list comprehension 拆出上下限
    BUY_LOW  = [low  for (low,  high) in buy_price_bounds]
    BUY_HIGH = [high for (low,  high) in buy_price_bounds]
    SELL_LOW  = [low  for (low,  high) in sell_price_bounds]
    SELL_HIGH = [high for (low,  high) in sell_price_bounds]


    BUY_LOW   = np.array(BUY_LOW,   dtype=np.float32)
    BUY_HIGH  = np.array(BUY_HIGH,  dtype=np.float32)
    SELL_LOW  = np.array(SELL_LOW,  dtype=np.float32)
    SELL_HIGH = np.array(SELL_HIGH, dtype=np.float32)

    env = MultiStoreEnv(association_rules = association_rules)
    STATE_DIM  = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.shape[0]

    policy     = GaussianPolicy(STATE_DIM, ACTION_DIM).to(DEVICE)
    q1         = QNet(STATE_DIM, ACTION_DIM).to(DEVICE)
    q2         = QNet(STATE_DIM, ACTION_DIM).to(DEVICE)
    q1_targ    = deepcopy(q1).to(DEVICE)
    q2_targ    = deepcopy(q2).to(DEVICE)

    opt_policy = optim.Adam(
        policy.parameters(),
        lr=LR_ACTOR
    )
    opt_q1     = optim.Adam(
        q1.parameters(),
        lr=LR_CRITIC
    )
    opt_q2     = optim.Adam(
        q2.parameters(),
        lr=LR_CRITIC
    )

    # 自動熵調節
    log_alpha  = torch.zeros(1, requires_grad=True, device=DEVICE)
    opt_alpha  = optim.Adam([log_alpha], LR_ALPHA)

    def alpha():
        return log_alpha.exp()

    buf = ReplayBuffer()

    # warm‑up
    s = env.reset()
    for _ in range(WARMUP_STEPS):
        a = env.action_space.sample()
        s2, r, d, _ = env.step(a)
        buf.push(s, a, r, s2, d)
        s = s2


    UPDATE_FREQ = 2   # 每隔几次 Critic 更新，再更新一次 Actor + α

    # 在训练循环最外层前面，初始化一个计数器
    update_it = 0

    # ─── training ───
    fig, ax = plt.subplots()
    R_hist, eps = [], []
    for ep in range(1, num_episodes + 1):
        if ep and ep%DROP_BOOST_EP==0: env.boost_drop()
        s = env.reset();
        ep_ret = 0
            # 为每个商品创建单独的计数器
        buy_counts = np.zeros(NUM_PRODUCTS, dtype=np.float32)
        sell_counts = np.zeros(NUM_PRODUCTS, dtype=np.float32)
        buy_vol = np.zeros(NUM_PRODUCTS, dtype=np.float32)
        sell_vol = np.zeros(NUM_PRODUCTS, dtype=np.float32)
        fee_rate = np.zeros(1, dtype=np.float32)
        player_gold = np.zeros(1, dtype=np.float32)
        cpi = np.zeros(1, dtype=np.float32)

        for t in range(max_steps):
            st = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            st_norm = env.normalize(st)                   # <- 这一行
            with torch.no_grad():
                a_env, _ = policy.sample(st_norm)   # 直接 [-0.02,0.02]
            a_env = a_env.squeeze().cpu().numpy()

            # 3) 與環境互動
            s2, r, d, _ = env.step(a_env)
            buf.push(s, a_env, r, s2, d)
            s, ep_ret = s2, ep_ret + r

            # ---------- 累加人數 & 數量 ----------
            N = NUM_PRODUCTS
            buy_counts  += s[2*N : 3*N]          
            sell_counts += s[3*N : 4*N]          
            buy_vol  += s[4*N : 5*N]          
            sell_vol += s[5*N : 6*N]          
            fee_rate += s[-1]
            player_gold += s[-2]
            cpi   += s[-3]
            
            # -------------------------------------
            # —— parameter updates ——
            if len(buf) >= BATCH_SIZE:
                update_it += 1
                bs, ba, br, bs2, bd = [torch.tensor(x, dtype=torch.float32, device=DEVICE)
                                        for x in buf.sample(BATCH_SIZE)]

                bs  = env.normalize(bs)
                bs2 = env.normalize(bs2)
                # 1) Q‑network update
                with torch.no_grad():
                    a2, logp2 = policy.sample(bs2)
                    min_q_targ = torch.min(q1_targ(bs2, a2), q2_targ(bs2, a2))
                    y = br + GAMMA * (1 - bd) * (min_q_targ - alpha() * logp2)
                q1_val, q2_val = q1(bs, ba), q2(bs, ba)
                loss_q = nn.functional.mse_loss(q1_val, y) + nn.functional.mse_loss(q2_val, y)
                opt_q1.zero_grad(); opt_q2.zero_grad(); loss_q.backward();
                opt_q1.step();       opt_q2.step()

                # Actor + α 延迟更新
                if update_it % UPDATE_FREQ == 0:
                    # 2) Policy update
                    a_pi, logp_pi = policy.sample(bs)
                    min_q_pi = torch.min(q1(bs, a_pi), q2(bs, a_pi))
                    loss_pi = (alpha() * logp_pi - min_q_pi).mean()
                    opt_policy.zero_grad(); loss_pi.backward(); opt_policy.step()

                    # 3) Entropy coefficient α update (automatic)
                    loss_alpha = -(log_alpha * (logp_pi + TARGET_ENTROPY).detach()).mean()
                    opt_alpha.zero_grad(); loss_alpha.backward(); opt_alpha.step()

                    # 4) Soft update of target networks
                    soft_update(q1, q1_targ, TAU)
                    soft_update(q2, q2_targ, TAU)



        total_avg_buy = ( buy_counts / STEPS_EP ).sum()
        total_avg_sell = ( sell_counts / STEPS_EP ).sum()
        fee_rate_avg = float(fee_rate / STEPS_EP)
        player_gold = float(player_gold / STEPS_EP)
        cpi = float(cpi / STEPS_EP)
        print(f"Ep {ep:4d} | AvgReward {ep_ret:9.2f} | StoreTotalAvgBuyCnt {total_avg_buy:.1f} | StoreTotalAvgSellCnt {total_avg_sell:.1f} | AvgTaxRate {fee_rate_avg:.1f}")
        print(f"     CPI {cpi:.3f} | PlayerGold {player_gold:.1f}")
        # 打印每种商品的统计信息
        print("Product-specific statistics:")
        for i in range(NUM_PRODUCTS):
            print(f"  Product {i+1}: StoreAvgBuy={(buy_counts / STEPS_EP)[i]:.1f}, StoreAvgSell={(sell_counts / STEPS_EP)[i]:.1f}, \
            StoreAvgBuyVol={(buy_vol / STEPS_EP)[i]:.1f}, StoreAvgSellVol={(sell_vol / STEPS_EP)[i]:.1f}")
        print() # 空行分隔每个episode的输出



        # —— logging ——
        R_hist.append(ep_ret); eps.append(ep)
        # 1）畫圖
        ax.clear()
        ax.plot(eps, R_hist)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.grid(True)

        # 2）存檔到 static 資料夾
        plot_path = os.path.join(
            settings.BASE_DIR,
            'statics', 'images',
            'sac_reward.png'         # 你自由命名
        )
        fig.savefig(plot_path, bbox_inches='tight')

        if progress_callback:
            avg_buy_prices  = [int(x) for x in env.store_buy_prices]   # 每個商品最後一刻的價格
            avg_sell_prices = [int(x) for x in env.store_sell_prices]

            # ⇩ 先把每種商品的統計打包成 list[dict]，之後前端好解析
            per_product_stats = [
                {
                    "product_id" : i + 1,
                    "avg_buy_cnt": float((buy_counts  / STEPS_EP)[i]),
                    "avg_sell_cnt":float((sell_counts / STEPS_EP)[i]),
                    "avg_buy_vol": float((buy_vol     / STEPS_EP)[i]),
                    "avg_sell_vol":float((sell_vol    / STEPS_EP)[i]),
                }
                for i in range(NUM_PRODUCTS)
            ]

            progress_callback(
                current_episode = ep,
                total_episodes  = num_episodes,
                avg_buy_price   = avg_buy_prices,
                avg_sell_price  = avg_sell_prices,
                reward          = round(float(ep_ret), 2),
                cpi             = float(cpi), 
                player_gold_avg = float(player_gold),      # 人均金幣
                per_product_stats = per_product_stats       # 逐商品統計
            )
        time.sleep(0.05)
        plt.pause(0.001)




