#cgan
import csv
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data as Data
from mlxtend.frequent_patterns import apriori, association_rules

# ====== 基本設定 ======
num_items = 5
num_records = 500
csv_filename = "C:\\python\\project\\cgangenplayerbuy\\my_data.csv"

df = pd.read_csv(csv_filename)
price_cols = [f'price_{i}' for i in range(num_items)]
basket_cols = [f'商品{chr(ord("A")+i)}' for i in range(num_items)]
df_price = df[price_cols]
df_basket = df[basket_cols].astype(bool)

# ====== (3) Apriori ======
frequent_itemsets = apriori(df_basket, min_support=0.4, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
rules = rules.reset_index(drop=True)
print("[Apriori 規則]")
print(rules[['antecedents','consequents','support','confidence']])

# 新增：定義可微分的 Apriori 懲罰函式
def compute_apriori_penalty(gen_basket, rules_df):
    # gen_basket 的 shape: [batch_size, num_items]
    total_penalty = 0.0
    for _, rule in rules_df.iterrows():
        # 將規則中的 antecedents 與 consequents 轉為 list (例如：{'商品A', '商品B'} -> ['商品A', '商品B'])
        antecedent_items = list(rule['antecedents'])
        consequent_items = list(rule['consequents'])
        # 將商品名稱轉成 index (假設 '商品A' 對應 index 0, '商品B' 對應 index 1, ...)
        antecedent_indices = [ord(item[-1]) - ord('A') for item in antecedent_items]#item[-1] 取的是字串的最後一個字符。 
        consequent_indices = [ord(item[-1]) - ord('A') for item in consequent_items]#這部分遍歷 antecedent_items 清單中的每個元素，每個 item 代表一個商品名稱，例如 "商品A"。
        
        # 改進：使用支持度和置信度加權懲罰
        conf_weight = rule['confidence']
        supp_weight = rule['support']

        # 計算 antecedents 與 consequents 的滿足程度 (連續機率乘積)
        antecedent_prob = torch.prod(gen_basket[:, antecedent_indices], dim=1)
        consequent_prob = torch.prod(gen_basket[:, consequent_indices], dim=1)
        
        # 當 antecedent 的機率高但 consequent 低，表示違反規則：懲罰值 = antecedent_prob * (1 - consequent_prob)


        threshold=0.25
        eps = 1e-6
        mask = (antecedent_prob > threshold).float()

        log_penalty = -torch.log(consequent_prob + eps)
        log_penalty = torch.clamp(log_penalty, max=5.0)  # <-- clip

        # 改成 mean 而不是 sum
        rule_penalty = torch.mean(mask * antecedent_prob * log_penalty) * conf_weight * supp_weight
        total_penalty += rule_penalty
    return total_penalty

# ====== (4) Conditional GAN ======
z_dim = 8

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + num_items, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),

            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),

            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(32),

            nn.Linear(32, num_items),
            nn.Sigmoid()
        )
    
    def forward(self, z, prices):
        x = torch.cat([z, prices], dim=1)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_items * 2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, basket, prices):
        x = torch.cat([basket, prices], dim=1)
        return self.net(x)

G, D = Generator(), Discriminator()
opt_G = optim.Adam(G.parameters(), lr=0.0005, betas=(0.5, 0.999))  # 標準GAN的推薦參數
opt_D = optim.Adam(D.parameters(), lr=0.00015, betas=(0.5, 0.999))  # 使D學習慢一點
bce = nn.BCELoss()

# ====== (5) Dataset ======
all_price_np  = df_price.values.astype('float32')
all_basket_np = df_basket.values.astype('float32')

class PriceBasketDataset(Data.Dataset):
    def __init__(self):
        self.price = all_price_np
        self.basket = all_basket_np
    def __len__(self):
        return len(self.price)
    def __getitem__(self, idx):
        return self.price[idx], self.basket[idx]

loader = Data.DataLoader(PriceBasketDataset(), batch_size=64, shuffle=True)

# ====== (6) Log ======
log = {'epoch':[], 'd_loss':[], 'g_loss':[], 'penalty':[]}

# ====== (6.5) 初始化圖表 ======
plt.ion()  # 開啟互動模式
fig, ax = plt.subplots(figsize=(10,6))

# ====== (7) Training ======
print("✅ CGAN Training")
best_g_loss = float('inf')
patience = 30
counter = 0

for epoch in range(400):
    for prices, baskets in loader:
        batch_size = prices.shape[0]
        z = torch.randn(batch_size, z_dim)
        
        d_steps = 1
        if epoch < 50 or epoch % 50 == 0:
            d_steps = 3

        for _ in range(d_steps):
            fake_basket = G(z, prices).detach()
            d_real = D(baskets, prices)
            d_fake = D(fake_basket, prices)
            real_labels = torch.ones_like(d_real) * 0.9  
            d_loss_real = bce(d_real, real_labels)
            d_loss_fake = bce(d_fake, torch.zeros_like(d_fake))
            d_loss = d_loss_real + d_loss_fake
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

        z2 = torch.randn(batch_size, z_dim)
        gen_basket = G(z2, prices)
        d_out = D(gen_basket, prices)
        apriori_penalty = compute_apriori_penalty(gen_basket, rules)
        


        if epoch > 150:
            lambda_penalty = 0.3
        elif epoch > 100:
            lambda_penalty = 0.1 + 0.1 * (epoch-100) / 25
        else:
            lambda_penalty = 0.1


        g_loss = bce(d_out, torch.ones_like(d_out)) + lambda_penalty * apriori_penalty
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

    # ====== log ======
    log['epoch'].append(epoch+1)
    log['d_loss'].append(d_loss.item())
    log['g_loss'].append(g_loss.item())
    log['penalty'].append(apriori_penalty.item() if isinstance(apriori_penalty, torch.Tensor) else apriori_penalty)

    # ====== ✅ 即時繪圖 ======
    if (epoch+1) % 1 == 0:   # 每個 epoch 都更新圖
        plt.cla()
        plt.plot(log['epoch'], log['d_loss'], label='D Loss')
        plt.plot(log['epoch'], log['g_loss'], label='G Loss')
        plt.plot(log['epoch'], log['penalty'], label='Apriori Penalty')
        plt.xlabel('Epoch')
        plt.ylabel('Loss / Penalty')
        plt.legend()
        plt.grid(True)
        plt.pause(0.01)   # 必須有這行才能即時刷新

    if (epoch+1) % 50 == 0:
        print(f"[Epoch {epoch+1}] D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}, Penalty={apriori_penalty.item():.4f}")

    if (epoch+1) % 10 == 0:
        if g_loss.item() < best_g_loss:
            best_g_loss = g_loss.item()
            counter = 0
            torch.save(G.state_dict(), 'best_generator.pth')
        else:
            counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# ====== 關閉互動 ======
plt.ioff()
plt.savefig("training_curve.png")
plt.show()
