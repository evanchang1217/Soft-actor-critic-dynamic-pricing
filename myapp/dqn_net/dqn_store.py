
#dqn_store
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from django.conf import settings

# ================ 全域變數設定 ================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 由外部設定的買賣價格上下限 (在 start_training_view 填寫後保存)
buy_price_bounds = []
sell_price_bounds = []

# 動作調整參數設定 (離散化選項數)
num_action_options = 5  
minimum_offset = -0.05
maximum_offset = 0.05
offset_range_per_step = (maximum_offset - minimum_offset) / (num_action_options - 1)
action_offsets = [minimum_offset + offset * offset_range_per_step for offset in range(num_action_options)]

# 存放訓練好的 policy net（拆成買與賣兩個網路，供推論使用）
trained_policy_net_buy = None
trained_policy_net_sell = None

# ================ 自定義 Gym 環境 ================
class StoreEnv(gym.Env):
    def __init__(self, num_products, num_players=500):
        super(StoreEnv, self).__init__()
        self.num_products = num_products
        self.num_players = num_players
        self.state = None
        self.reset()

    def _calculate_probability(self, price, low, high, is_buying):
        price = np.clip(price, low, high)
        if is_buying:
            return max(0, min(1, (price - low) / (high - low)))
        else:
            return max(0, min(1, 1 - (price - low) / (high - low)))

    def step(self, actions):
        next_state = self.state.copy()
        total_buy_count = []
        total_sell_count = []

        for i in range(self.num_products):
            # 由兩個獨立網路決定的動作合併：
            # buy_action 為 actions[i] // num_action_options，
            # sell_action 為 actions[i] % num_action_options
            action = (actions[i] // num_action_options, actions[i] % num_action_options)
            buy_adj = action_offsets[action[0]]
            sell_adj = action_offsets[action[1]]

            bp_low, bp_high = buy_price_bounds[i]
            sp_low, sp_high = sell_price_bounds[i]

            # 更新買/賣價格
            next_state[i, 0] = np.clip(self.state[i, 0] + self.state[i, 0] * buy_adj, bp_low, bp_high)
            next_state[i, 1] = np.clip(self.state[i, 1] + self.state[i, 1] * sell_adj, sp_low, sp_high)

            buy_prob = self._calculate_probability(next_state[i, 0], bp_low, bp_high, True)
            sell_prob = self._calculate_probability(next_state[i, 1], sp_low, sp_high, False)

            buy_count = sum(np.random.rand() < buy_prob for _ in range(self.num_players))
            sell_count = sum(np.random.rand() < sell_prob for _ in range(self.num_players))

            total_buy_count.append(buy_count)
            total_sell_count.append(sell_count)
            next_state[i, 2] = buy_count
            next_state[i, 3] = sell_count

        # 原本的獎勵是綜合買與賣，但這裡我們拆分後會分別計算（環境回傳的 rewards 不再直接使用）
        rewards = [
            -abs(b - self.num_players / 2) - abs(s - self.num_players / 2)
            for b, s in zip(total_buy_count, total_sell_count)
        ]
        done = False
        info = {}
        self.state = next_state.copy()
        return next_state, rewards, done, info

    def reset(self):
        state_list = []
        for i in range(self.num_products):
            bp_low, bp_high = buy_price_bounds[i]
            sp_low, sp_high = sell_price_bounds[i]
            buy = random.randint(int(bp_low), int(bp_high))
            sell = random.randint(int(sp_low), int(sp_high))
            state_list.append([buy, sell, 0, 0])
        self.state = np.array(state_list)
        return self.state

# ================ DQN 網路架構 (參數化輸出維度) ================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

# ================ 主要訓練函式 (拆分買與賣網路) ================
def run_dqn_training(num_episodes, max_steps, batch_size, num_products, progress_callback=None):
    """
    拆分買與賣兩個 DQN 網路的訓練，訓練完成後分別存到全域變數 trained_policy_net_buy 與 trained_policy_net_sell
    """
    global trained_policy_net_buy, trained_policy_net_sell

    env = StoreEnv(num_products=num_products)
    # state: [buy, sell, buy_count, sell_count] + one-hot(product) => input_dim
    input_dim = 4 + num_products

    # 建立獨立的網路：買的與賣的 (輸出維度均為 num_action_options，即 5)
    policy_net_buy = DQN(input_dim, num_action_options).to(device)
    target_net_buy = DQN(input_dim, num_action_options).to(device)
    target_net_buy.load_state_dict(policy_net_buy.state_dict())
    target_net_buy.eval()

    policy_net_sell = DQN(input_dim, num_action_options).to(device)
    target_net_sell = DQN(input_dim, num_action_options).to(device)
    target_net_sell.load_state_dict(policy_net_sell.state_dict())
    target_net_sell.eval()

    optimizer_buy = optim.AdamW(policy_net_buy.parameters(), lr=0.001)
    optimizer_sell = optim.AdamW(policy_net_sell.parameters(), lr=0.001)

    # -------------- Replay Buffer (分別使用) --------------
    class ReplayBuffer:
        def __init__(self, capacity):
            self.capacity = capacity
            self.buffer = deque(maxlen=capacity)
        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))
        def sample(self, batch_size):
            return random.sample(self.buffer, batch_size)
        def __len__(self):
            return len(self.buffer)

    replay_buffer_buy = ReplayBuffer(10000)
    replay_buffer_sell = ReplayBuffer(10000)

    # -------------- 一些訓練參數 --------------
    gamma = 0.99
    epsilon = 0.99
    epsilon_decay = 0.99
    min_epsilon = 0.1
    target_update = 10

    # 用於視覺化 Reward 的紀錄（分別記錄買與賣）
    rewards_history_buy = [[] for _ in range(num_products)]
    rewards_history_sell = [[] for _ in range(num_products)]
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    # -------------- 選擇動作函式 (買與賣分開) --------------
    def select_buy_action(state, product_idx, epsilon):
        product_one_hot = np.zeros(num_products)
        product_one_hot[product_idx] = 1
        combined_state = np.concatenate((state, product_one_hot))
        if random.random() < epsilon:
            return random.randint(0, num_action_options - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(combined_state, dtype=torch.float32).to(device)
                return policy_net_buy(state_tensor).argmax().item()

    def select_sell_action(state, product_idx, epsilon):
        product_one_hot = np.zeros(num_products)
        product_one_hot[product_idx] = 1
        combined_state = np.concatenate((state, product_one_hot))
        if random.random() < epsilon:
            return random.randint(0, num_action_options - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(combined_state, dtype=torch.float32).to(device)
                return policy_net_sell(state_tensor).argmax().item()

    # -------------- 優化模型函式 (共用一個函式，只是參數不同) --------------
    def optimize_model(replay_buffer, policy_net, target_net, optimizer):
        if len(replay_buffer) < batch_size:
            return
        transitions = replay_buffer.sample(batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        batch_state = torch.tensor(np.array(batch_state), dtype=torch.float32).to(device)
        batch_action = torch.tensor(np.array(batch_action), dtype=torch.int64).unsqueeze(1).to(device)
        batch_reward = torch.tensor(np.array(batch_reward), dtype=torch.float32).unsqueeze(1).to(device)
        batch_next_state = torch.tensor(np.array(batch_next_state), dtype=torch.float32).to(device)
        batch_done = torch.tensor(np.array(batch_done), dtype=torch.float32).unsqueeze(1).to(device)

        current_q = policy_net(batch_state).gather(1, batch_action)
        with torch.no_grad():
            next_q = target_net(batch_next_state).max(1)[0].unsqueeze(1)
        expected_q = batch_reward + gamma * next_q * (1 - batch_done)

        loss = F.smooth_l1_loss(current_q, expected_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("開始 DQN 訓練")
    for episode in range(1, num_episodes + 1):
        total_rewards_buy = [0] * num_products
        total_rewards_sell = [0] * num_products
        states = env.reset()

        # 執行一個 episode
        for t in range(max_steps):
            buy_actions = []
            sell_actions = []
            combined_actions = []
            for i in range(num_products):
                b_action = select_buy_action(states[i], i, epsilon)
                s_action = select_sell_action(states[i], i, epsilon)
                buy_actions.append(b_action)
                sell_actions.append(s_action)
                # 將兩個動作合併成一個數值傳給環境
                combined_actions.append(b_action * num_action_options + s_action)

            next_states, rewards, done, _ = env.step(combined_actions)

            for i in range(num_products):
                product_one_hot = np.zeros(num_products)
                product_one_hot[i] = 1
                combined_state = np.concatenate((states[i], product_one_hot))
                combined_next_state = np.concatenate((next_states[i], product_one_hot))
                # 分別計算買與賣的獎勵（例如：以買/賣人數與 num_players/2 的差距）
                buy_reward = -abs(next_states[i, 2] - env.num_players / 2)
                sell_reward = -abs(next_states[i, 3] - env.num_players / 2)
                replay_buffer_buy.push(combined_state, buy_actions[i], buy_reward, combined_next_state, done)
                replay_buffer_sell.push(combined_state, sell_actions[i], sell_reward, combined_next_state, done)
                total_rewards_buy[i] += buy_reward
                total_rewards_sell[i] += sell_reward

            states = next_states

        # 優化買與賣網路
        optimize_model(replay_buffer_buy, policy_net_buy, target_net_buy, optimizer_buy)
        optimize_model(replay_buffer_sell, policy_net_sell, target_net_sell, optimizer_sell)

        # 更新 reward history
        for i in range(num_products):
            rewards_history_buy[i].append(total_rewards_buy[i])
            rewards_history_sell[i].append(total_rewards_sell[i])

        # 更新 target_net 每 target_update 個 episode
        if episode % target_update == 0:
            target_net_buy.load_state_dict(policy_net_buy.state_dict())
            target_net_sell.load_state_dict(policy_net_sell.state_dict())

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # 更新繪圖
        ax.clear()
        for i in range(num_products):
            ax.plot(range(len(rewards_history_buy[i])), rewards_history_buy[i],
                    label=f'Product {i} Buy Reward', linestyle='--')
            ax.plot(range(len(rewards_history_sell[i])), rewards_history_sell[i],
                    label=f'Product {i} Sell Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        plot_path = os.path.join(settings.BASE_DIR, 'statics', 'images', 'dqn_reward.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.draw()
        plt.pause(0.01)

        if progress_callback:
            avg_buy_prices = [env.state[i, 0] for i in range(num_products)]
            avg_sell_prices = [env.state[i, 1] for i in range(num_products)]
            progress_callback(
                episode=episode,
                total_episodes=num_episodes,
                avg_buy_price=[int(price) for price in avg_buy_prices],
                avg_sell_price=[int(price) for price in avg_sell_prices],
                reward=int(sum(total_rewards_buy) + sum(total_rewards_sell))
            )

        time.sleep(0.05)

    trained_policy_net_buy = policy_net_buy
    trained_policy_net_sell = policy_net_sell
    print("訓練完成，policy_net_buy 與 policy_net_sell 已儲存")
    plt.ioff()

# ================ 單步推論函式 (拆分買與賣) ================
def inference_one_step(
    product_idx: int,
    current_buy_price: float,
    current_sell_price: float,
    buy_count: float,
    sell_count: float
):
    """
    針對單一商品的單步推論，分別透過買與賣的網路取得動作，
    回傳 (new_buy_price, new_sell_price)。
    """
    global trained_policy_net_buy, trained_policy_net_sell
    if trained_policy_net_buy is None or trained_policy_net_sell is None:
        print("模型尚未訓練好，無法推論！")
        return current_buy_price, current_sell_price

    num_products = len(buy_price_bounds)
    if product_idx < 0 or product_idx >= num_products:
        print(f"product_idx {product_idx} 超出範圍!")
        return current_buy_price, current_sell_price

    product_one_hot = np.zeros(num_products, dtype=float)
    product_one_hot[product_idx] = 1.0

    state_np = np.array([
        current_buy_price,
        current_sell_price,
        buy_count,
        sell_count
    ], dtype=float)
    combined_state = np.concatenate((state_np, product_one_hot))
    state_tensor = torch.tensor(combined_state, dtype=torch.float32).to(device)

    with torch.no_grad():
        buy_q_values = trained_policy_net_buy(state_tensor)
        sell_q_values = trained_policy_net_sell(state_tensor)
        buy_action = buy_q_values.argmax().item()    # 0 ~ 4
        sell_action = sell_q_values.argmax().item()    # 0 ~ 4

    bp_low, bp_high = buy_price_bounds[product_idx]
    sp_low, sp_high = sell_price_bounds[product_idx]

    new_buy_price = np.clip(current_buy_price + current_buy_price * action_offsets[buy_action], bp_low, bp_high)
    new_sell_price = np.clip(current_sell_price + current_sell_price * action_offsets[sell_action], sp_low, sp_high)

    return float(new_buy_price), float(new_sell_price)

# ================ 多商品一次推論函式 (新增) ================
def inference_multiple_products(products_info):
    """
    接收一個商品陣列，每個商品包含:
      {
        "product_idx": int,
        "current_buy_price": float,
        "current_sell_price": float,
        "buy_count": float,
        "sell_count": float
      }
    回傳一個 list，每一項是 {
        "product_idx": ...,
        "new_buy_price": ...,
        "new_sell_price": ...
    }
    """
    results = []
    for item in products_info:
        p_idx = int(item["product_idx"])
        cbp = float(item["current_buy_price"])
        csp = float(item["current_sell_price"])
        bc = float(item["buy_count"])
        sc = float(item["sell_count"])
        nbp, nsp = inference_one_step(p_idx, cbp, csp, bc, sc)
        results.append({
            "product_idx": p_idx,
            "new_buy_price": nbp,
            "new_sell_price": nsp
        })
    return results









'''
# store_env_dqn.py

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from django.conf import settings
#from myapp.cgan_module.cgan_buy_training import GeneratorCGANBuy
#from myapp.cgan_module.cgan_sell_training import GeneratorCGAN



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================ 全域變數設定 ================
buy_price_bounds = []
sell_price_bounds = []

num_action_options = 5  
minimum_offset = -0.05
maximum_offset = 0.05
offset_range_per_step = (maximum_offset - minimum_offset) / (num_action_options - 1)
action_offsets = [minimum_offset + offset * offset_range_per_step for offset in range(num_action_options)]


trained_policy_net_buy = None
trained_policy_net_sell = None

# --- cGAN 全域變數（賣） ---
z_dim_cgan = 8
trained_cgan_generator_sell = None  # 指向「賣」的 cGAN

# --- cGAN 全域變數（買） ---
z_dim_cgan_buy = 8
trained_cgan_generator_buy = None   # 指向「買」的 cGAN

# 假設商品數量 = 5 (跟你的 cGAN 訓練一致)
cgan_num_items_sell = 5
cgan_num_items_buy = 5

# ============ 若不想在此檔案內定義 Generator，改在 cgan_xx_training.py 也可以 ================
# 這裡示範只放載入函式

def load_trained_cgan_generator_sell(model_path="cgan_generator_sell.pth"):
    """
    載入「賣」的 cGAN 生成器
    """
    global trained_cgan_generator_sell
    
    if trained_cgan_generator_sell is None:
        gen = GeneratorCGAN().to(device)
        gen.load_state_dict(torch.load(model_path, map_location=device))
        gen.eval()
        trained_cgan_generator_sell = gen
    return trained_cgan_generator_sell

def load_trained_cgan_generator_buy(model_path="cgan_generator_buy.pth"):
    """
    載入「買」的 cGAN 生成器
    """
    global trained_cgan_generator_buy
    
    if trained_cgan_generator_buy is None:
        gen = GeneratorCGANBuy().to(device)
        gen.load_state_dict(torch.load(model_path, map_location=device))
        gen.eval()
        trained_cgan_generator_buy = gen
    return trained_cgan_generator_buy

def cgan_compute_sell_counts(sell_prices, num_players=500):
    """
    使用「賣」的 cGAN 生成籃，然後 sum 得到賣出數
    """
    generator_cgan = load_trained_cgan_generator_sell()
    sell_prices_batch = np.tile(sell_prices, (num_players, 1)).astype(np.float32)
    z = torch.randn(num_players, z_dim_cgan).to(device)
    prices_tensor = torch.tensor(sell_prices_batch, dtype=torch.float32).to(device)
    with torch.no_grad():
        fake_basket = generator_cgan(z, prices_tensor)
        basket_binary = (fake_basket > 0.5).float()
    sell_count_array = basket_binary.sum(dim=0).cpu().numpy().astype(int)
    return sell_count_array

def cgan_compute_buy_counts(buy_prices, num_players=500):
    """
    使用「買」的 cGAN 生成籃，然後 sum 得到買入數
    """
    generator_buy = load_trained_cgan_generator_buy()
    buy_prices_batch = np.tile(buy_prices, (num_players, 1)).astype(np.float32)
    z = torch.randn(num_players, z_dim_cgan_buy).to(device)
    prices_tensor = torch.tensor(buy_prices_batch, dtype=torch.float32).to(device)
    with torch.no_grad():
        fake_basket = generator_buy(z, prices_tensor)
        basket_binary = (fake_basket > 0.5).float()
    buy_count_array = basket_binary.sum(dim=0).cpu().numpy().astype(int)
    return buy_count_array


# ================ 自定義 Gym 環境 ================
class StoreEnv(gym.Env):
    def __init__(self, num_products, num_players=500):
        super(StoreEnv, self).__init__()
        self.num_products = num_products
        self.num_players = num_players
        self.state = None
        self.reset()

    def step(self, actions):
        """
        actions[i] -> 分為 buy_action 與 sell_action
        然後透過 cGAN 計算買 / 賣人數
        """
        next_state = self.state.copy()
        updated_buy_prices = []
        updated_sell_prices = []

        # 1) 更新所有產品的買 / 賣價格
        for i in range(self.num_products):
            buy_act = actions[i] // num_action_options
            sell_act = actions[i] % num_action_options

            buy_adj = action_offsets[buy_act]
            sell_adj = action_offsets[sell_act]

            bp_low, bp_high = buy_price_bounds[i]
            sp_low, sp_high = sell_price_bounds[i]

            new_buy_price = np.clip(self.state[i, 0] + self.state[i, 0] * buy_adj, bp_low, bp_high)
            new_sell_price = np.clip(self.state[i, 1] + self.state[i, 1] * sell_adj, sp_low, sp_high)

            next_state[i, 0] = new_buy_price
            next_state[i, 1] = new_sell_price

            updated_buy_prices.append(new_buy_price)
            updated_sell_prices.append(new_sell_price)

        # 2) 呼叫 cGAN：計算買 / 賣數量
        if self.num_products == 5:
            buy_counts_array = cgan_compute_buy_counts(np.array(updated_buy_prices), self.num_players)
            sell_counts_array = cgan_compute_sell_counts(np.array(updated_sell_prices), self.num_players)
        else:
            raise ValueError("目前示範只能處理 num_products=5")

        # 3) 更新 buy_count, sell_count
        total_buy_count = []
        total_sell_count = []
        for i in range(self.num_products):
            bc = buy_counts_array[i]
            sc = sell_counts_array[i]
            total_buy_count.append(bc)
            total_sell_count.append(sc)
            next_state[i, 2] = bc
            next_state[i, 3] = sc

        # 4) 計算 reward
        rewards = [
            -abs(b - self.num_players / 2) - abs(s - self.num_players / 2)
            for b, s in zip(total_buy_count, total_sell_count)
        ]
        done = False
        info = {}
        self.state = next_state.copy()
        return next_state, rewards, done, info

    def reset(self):
        state_list = []
        for i in range(self.num_products):
            bp_low, bp_high = buy_price_bounds[i]
            sp_low, sp_high = sell_price_bounds[i]
            b_init = random.randint(int(bp_low), int(bp_high))
            s_init = random.randint(int(sp_low), int(sp_high))
            state_list.append([b_init, s_init, 0, 0])
        self.state = np.array(state_list)
        return self.state

# ================ DQN 網路架構 ================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

# ================ 主要訓練與推論函式 ================
def run_dqn_training(num_episodes, max_steps, batch_size, num_products, progress_callback=None):
    global trained_policy_net_buy, trained_policy_net_sell

    env = StoreEnv(num_products=num_products)
    input_dim = 4 + num_products

    policy_net_buy = DQN(input_dim, num_action_options).to(device)
    policy_net_sell = DQN(input_dim, num_action_options).to(device)

    target_net_buy = DQN(input_dim, num_action_options).to(device)
    target_net_sell = DQN(input_dim, num_action_options).to(device)

    target_net_buy.load_state_dict(policy_net_buy.state_dict())
    target_net_sell.load_state_dict(policy_net_sell.state_dict())
    target_net_buy.eval()
    target_net_sell.eval()

    optimizer_buy = optim.AdamW(policy_net_buy.parameters(), lr=0.001)
    optimizer_sell = optim.AdamW(policy_net_sell.parameters(), lr=0.001)

    # ---- Replay Buffer ----
    class ReplayBuffer:
        def __init__(self, capacity):
            self.capacity = capacity
            self.buffer = deque(maxlen=capacity)
        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))
        def sample(self, bsz):
            return random.sample(self.buffer, bsz)
        def __len__(self):
            return len(self.buffer)

    replay_buffer_buy = ReplayBuffer(10000)
    replay_buffer_sell = ReplayBuffer(10000)

    gamma = 0.99
    epsilon = 0.99
    epsilon_decay = 0.99
    min_epsilon = 0.1
    target_update = 10

    rewards_history_buy = [[] for _ in range(num_products)]
    rewards_history_sell = [[] for _ in range(num_products)]
    plt.ion()
    fig, ax = plt.subplots(figsize=(10,6))

    def select_buy_action(state, product_idx, eps):
        product_one_hot = np.zeros(num_products)
        product_one_hot[product_idx] = 1
        combined_state = np.concatenate((state, product_one_hot))
        if random.random() < eps:
            return random.randint(0, num_action_options-1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(combined_state, dtype=torch.float32).to(device)
                return policy_net_buy(state_tensor).argmax().item()

    def select_sell_action(state, product_idx, eps):
        product_one_hot = np.zeros(num_products)
        product_one_hot[product_idx] = 1
        combined_state = np.concatenate((state, product_one_hot))
        if random.random() < eps:
            return random.randint(0, num_action_options-1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(combined_state, dtype=torch.float32).to(device)
                return policy_net_sell(state_tensor).argmax().item()

    def optimize_model(replay_buffer, policy_net, target_net, optimizer):
        if len(replay_buffer) < batch_size:
            return
        transitions = replay_buffer.sample(batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        batch_state = torch.tensor(np.array(batch_state), dtype=torch.float32).to(device)
        batch_action = torch.tensor(np.array(batch_action), dtype=torch.int64).unsqueeze(1).to(device)
        batch_reward = torch.tensor(np.array(batch_reward), dtype=torch.float32).unsqueeze(1).to(device)
        batch_next_state = torch.tensor(np.array(batch_next_state), dtype=torch.float32).to(device)
        batch_done = torch.tensor(np.array(batch_done), dtype=torch.float32).unsqueeze(1).to(device)

        current_q = policy_net(batch_state).gather(1, batch_action)
        with torch.no_grad():
            next_q = target_net(batch_next_state).max(1)[0].unsqueeze(1)
        expected_q = batch_reward + gamma * next_q * (1 - batch_done)

        loss = F.smooth_l1_loss(current_q, expected_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("開始 DQN 訓練")
    for episode in range(1, num_episodes+1):
        total_rewards_buy = [0]*num_products
        total_rewards_sell = [0]*num_products
        states = env.reset()

        for t in range(max_steps):
            buy_actions = []
            sell_actions = []
            combined_actions = []
            for i in range(num_products):
                b_act = select_buy_action(states[i], i, epsilon)
                s_act = select_sell_action(states[i], i, epsilon)
                buy_actions.append(b_act)
                sell_actions.append(s_act)
                combined_actions.append(b_act * num_action_options + s_act)

            next_states, rewards, done, _ = env.step(combined_actions)

            for i in range(num_products):
                product_one_hot = np.zeros(num_products)
                product_one_hot[i] = 1
                combined_state = np.concatenate((states[i], product_one_hot))
                combined_next_state = np.concatenate((next_states[i], product_one_hot))

                buy_reward = -abs(next_states[i, 2] - env.num_players/2)
                sell_reward = -abs(next_states[i, 3] - env.num_players/2)

                replay_buffer_buy.push(combined_state, buy_actions[i], buy_reward, combined_next_state, done)
                replay_buffer_sell.push(combined_state, sell_actions[i], sell_reward, combined_next_state, done)
                total_rewards_buy[i] += buy_reward
                total_rewards_sell[i] += sell_reward

            states = next_states

        optimize_model(replay_buffer_buy, policy_net_buy, target_net_buy, optimizer_buy)
        optimize_model(replay_buffer_sell, policy_net_sell, target_net_sell, optimizer_sell)

        for i in range(num_products):
            rewards_history_buy[i].append(total_rewards_buy[i])
            rewards_history_sell[i].append(total_rewards_sell[i])

        if episode % target_update == 0:
            target_net_buy.load_state_dict(policy_net_buy.state_dict())
            target_net_sell.load_state_dict(policy_net_sell.state_dict())

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # -- 即時繪圖 --
        ax.clear()
        for i in range(num_products):
            ax.plot(range(len(rewards_history_buy[i])), rewards_history_buy[i],
                    label=f'Product {i} Buy Reward', linestyle='--')
            ax.plot(range(len(rewards_history_sell[i])), rewards_history_sell[i],
                    label=f'Product {i} Sell Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        plot_path = os.path.join(settings.BASE_DIR, 'statics', 'images', 'dqn_reward.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.draw()
        plt.pause(0.01)

        if progress_callback:
            avg_buy_prices = [env.state[i, 0] for i in range(num_products)]
            avg_sell_prices = [env.state[i, 1] for i in range(num_products)]
            progress_callback(
                episode=episode,
                total_episodes=num_episodes,
                avg_buy_price=[int(x) for x in avg_buy_prices],
                avg_sell_price=[int(x) for x in avg_sell_prices],
                reward=int(sum(total_rewards_buy) + sum(total_rewards_sell))
            )

        time.sleep(0.01)

    trained_policy_net_buy = policy_net_buy
    trained_policy_net_sell = policy_net_sell
    print("訓練完成，policy_net_buy 與 policy_net_sell 已儲存")
    plt.ioff()

def inference_one_step(product_idx, current_buy_price, current_sell_price, buy_count, sell_count):
    global trained_policy_net_buy, trained_policy_net_sell
    if trained_policy_net_buy is None or trained_policy_net_sell is None:
        print("模型尚未訓練好，無法推論！")
        return current_buy_price, current_sell_price

    num_products = len(buy_price_bounds)
    if product_idx<0 or product_idx>=num_products:
        print(f"product_idx {product_idx} 超出範圍！")
        return current_buy_price, current_sell_price

    product_one_hot = np.zeros(num_products)
    product_one_hot[product_idx] = 1
    state_np = np.array([current_buy_price, current_sell_price, buy_count, sell_count])
    combined_state = np.concatenate((state_np, product_one_hot))
    state_tensor = torch.tensor(combined_state, dtype=torch.float32).to(device)

    with torch.no_grad():
        buy_q = trained_policy_net_buy(state_tensor)
        sell_q = trained_policy_net_sell(state_tensor)
        buy_action = buy_q.argmax().item()
        sell_action = sell_q.argmax().item()

    bp_low, bp_high = buy_price_bounds[product_idx]
    sp_low, sp_high = sell_price_bounds[product_idx]

    new_buy_price = np.clip(current_buy_price + current_buy_price * action_offsets[buy_action], bp_low, bp_high)
    new_sell_price = np.clip(current_sell_price + current_sell_price * action_offsets[sell_action], sp_low, sp_high)

    return float(new_buy_price), float(new_sell_price)

def inference_multiple_products(products_info):
    results = []
    for item in products_info:
        p_idx = item["product_idx"]
        cbp = item["current_buy_price"]
        csp = item["current_sell_price"]
        bc = item["buy_count"]
        sc = item["sell_count"]
        nbp, nsp = inference_one_step(p_idx, cbp, csp, bc, sc)
        results.append({
            "product_idx": p_idx,
            "new_buy_price": nbp,
            "new_sell_price": nsp
        })
    return results

# (檔案結束)
'''



    #優化dqn#輸入今天購買數據輸出隔天調整#buycal#apriory




