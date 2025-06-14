# cgan_sell_training.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import torch.utils.data as Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_dim_cgan = 8
cgan_num_items = 5  # 假設商品數量=5

# 這裡定義「賣」的 Generator 與 Discriminator
class GeneratorCGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim_cgan + cgan_num_items, 128),
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

            nn.Linear(32, cgan_num_items),
            nn.Sigmoid()
        )
    
    def forward(self, z, prices):
        x = torch.cat([z, prices], dim=1)
        return self.net(x)

class DiscriminatorCGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cgan_num_items*2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, basket, prices):
        x = torch.cat([basket, prices], dim=1)
        return self.net(x)

def train_cgan_for_selling(csv_filename_sell="my_data_sell.csv"):
    """
    訓練「賣」的 cGAN（含 Apriori 懲罰），最終存成 cgan_generator.pth
    """
    # 1) 載入資料
    df = pd.read_csv(csv_filename_sell)
    price_cols = [f'price_{i}' for i in range(cgan_num_items)]
    basket_cols = [f'商品{chr(ord("A")+i)}' for i in range(cgan_num_items)]
    df_price = df[price_cols]
    df_basket = df[basket_cols].astype(bool)

    # 2) Apriori
    frequent_itemsets = apriori(df_basket, min_support=0.4, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
    rules = rules.reset_index(drop=True)
    print("[Apriori 規則]")
    print(rules[['antecedents','consequents','support','confidence']])

    def compute_apriori_penalty(gen_basket, rules_df):
        total_penalty = 0.0
        for _, rule in rules_df.iterrows():
            antecedent_items = list(rule['antecedents'])
            consequent_items = list(rule['consequents'])
            antecedent_indices = [ord(x[-1]) - ord('A') for x in antecedent_items]
            consequent_indices = [ord(x[-1]) - ord('A') for x in consequent_items]

            conf_weight = rule['confidence']
            supp_weight = rule['support']

            antecedent_prob = torch.prod(gen_basket[:, antecedent_indices], dim=1)
            consequent_prob = torch.prod(gen_basket[:, consequent_indices], dim=1)

            threshold = 0.25
            eps = 1e-6
            mask = (antecedent_prob > threshold).float()
            log_penalty = -torch.log(consequent_prob + eps)
            log_penalty = torch.clamp(log_penalty, max=5.0)

            rule_penalty = torch.mean(mask * antecedent_prob * log_penalty) * conf_weight * supp_weight
            total_penalty += rule_penalty
        return total_penalty

    # 3) Dataset
    all_price_np = df_price.values.astype('float32')
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

    # 4) 建立 G, D
    G = GeneratorCGAN().to(device)
    D = DiscriminatorCGAN().to(device)

    opt_G = optim.Adam(G.parameters(), lr=0.0005, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=0.00015, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    log = {'epoch':[], 'd_loss':[], 'g_loss':[], 'penalty':[]}
    plt.ion()
    fig, ax = plt.subplots(figsize=(10,6))

    best_g_loss = float('inf')
    patience = 30
    counter = 0
    max_epoch = 400

    print("✅ 開始訓練 (賣) cGAN ...")
    for epoch in range(max_epoch):
        for prices, baskets in loader:
            batch_size = prices.shape[0]
            z = torch.randn(batch_size, z_dim_cgan).to(device)
            prices = prices.to(device)
            baskets = baskets.to(device)

            d_steps = 1
            if epoch<50 or epoch%50==0:
                d_steps = 3

            for _ in range(d_steps):
                fake_basket = G(z, prices).detach()
                d_real = D(baskets, prices)
                d_fake = D(fake_basket, prices)
                real_labels = torch.ones_like(d_real)*0.9
                d_loss_real = bce(d_real, real_labels)
                d_loss_fake = bce(d_fake, torch.zeros_like(d_fake))
                d_loss = d_loss_real + d_loss_fake

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

            z2 = torch.randn(batch_size, z_dim_cgan).to(device)
            gen_basket = G(z2, prices)
            d_out = D(gen_basket, prices)
            apriori_penalty = compute_apriori_penalty(gen_basket, rules)

            if epoch>150:
                lambda_penalty = 0.3
            elif epoch>100:
                lambda_penalty = 0.1 + 0.1*(epoch-100)/25
            else:
                lambda_penalty = 0.1

            g_loss = bce(d_out, torch.ones_like(d_out)) + lambda_penalty*apriori_penalty

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

        log['epoch'].append(epoch+1)
        log['d_loss'].append(d_loss.item())
        log['g_loss'].append(g_loss.item())
        log['penalty'].append(apriori_penalty.item() if isinstance(apriori_penalty, torch.Tensor) else apriori_penalty)

        if (epoch+1)%1 == 0:
            plt.cla()
            plt.plot(log['epoch'], log['d_loss'], label='D Loss')
            plt.plot(log['epoch'], log['g_loss'], label='G Loss')
            plt.plot(log['epoch'], log['penalty'], label='Apriori Penalty')
            plt.xlabel('Epoch')
            plt.ylabel('Loss / Penalty')
            plt.legend()
            plt.grid(True)
            plt.pause(0.01)

        if (epoch+1)%50 == 0:
            print(f"[Epoch {epoch+1}] D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}, penalty={apriori_penalty.item():.4f}")

        # Early stopping
        if (epoch+1)%10 == 0:
            if g_loss.item()<best_g_loss:
                best_g_loss = g_loss.item()
                counter=0
                torch.save(G.state_dict(), 'cgan_generator_sell.pth')
            else:
                counter+=1
            if counter>=patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    plt.ioff()
    plt.savefig("training_curve_sell.png")
    plt.show()
    print("✅ (賣) cGAN 訓練完畢，已儲存 Generator 權重到 cgan_generator.pth")

# (檔案結束)
