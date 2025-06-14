# cgan_buy_training.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import torch.utils.data as Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_dim_cgan_buy = 8
cgan_num_items_buy = 5

class GeneratorCGANBuy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim_cgan_buy + cgan_num_items_buy, 128),
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

            nn.Linear(32, cgan_num_items_buy),
            nn.Sigmoid()
        )
    
    def forward(self, z, prices):
        x = torch.cat([z, prices], dim=1)
        return self.net(x)

class DiscriminatorCGANBuy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cgan_num_items_buy*2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, basket, prices):
        x = torch.cat([basket, prices], dim=1)
        return self.net(x)

def train_cgan_for_buying(csv_filename_buy="my_data_buy.csv"):
    """
    訓練「買」的 cGAN，不使用 apriori 懲罰。
    最後存成 cgan_generator_buy.pth
    """
    df = pd.read_csv(csv_filename_buy)
    price_cols = [f'price_{i}' for i in range(cgan_num_items_buy)]
    basket_cols = [f'商品{chr(ord("A")+i)}' for i in range(cgan_num_items_buy)]
    df_price = df[price_cols]
    df_basket = df[basket_cols].astype(bool)

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

    G = GeneratorCGANBuy().to(device)
    D = DiscriminatorCGANBuy().to(device)

    opt_G = optim.Adam(G.parameters(), lr=0.0005, betas=(0.5,0.999))
    opt_D = optim.Adam(D.parameters(), lr=0.00015, betas=(0.5,0.999))
    bce = nn.BCELoss()

    log = {'epoch':[], 'd_loss':[], 'g_loss':[]}
    plt.ion()
    fig, ax = plt.subplots(figsize=(10,6))

    best_g_loss = float('inf')
    patience = 30
    counter = 0
    max_epoch = 400

    print("✅ 開始訓練 (買) cGAN (無 Apriori) ...")
    for epoch in range(max_epoch):
        for prices, baskets in loader:
            bsz = prices.shape[0]
            z = torch.randn(bsz, z_dim_cgan_buy).to(device)
            prices = prices.to(device)
            baskets = baskets.to(device)

            # 可能 d 多跑幾次
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

            z2 = torch.randn(bsz, z_dim_cgan_buy).to(device)
            gen_basket = G(z2, prices)
            d_out = D(gen_basket, prices)
            g_loss = bce(d_out, torch.ones_like(d_out))

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

        log['epoch'].append(epoch+1)
        log['d_loss'].append(d_loss.item())
        log['g_loss'].append(g_loss.item())

        if (epoch+1)%1==0:
            plt.cla()
            plt.plot(log['epoch'], log['d_loss'], label='D Loss')
            plt.plot(log['epoch'], log['g_loss'], label='G Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.pause(0.01)

        if (epoch+1)%50==0:
            print(f"[Epoch {epoch+1}] D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")

        if (epoch+1)%10==0:
            if g_loss.item()<best_g_loss:
                best_g_loss = g_loss.item()
                counter=0
                torch.save(G.state_dict(), 'cgan_generator_buy.pth')
            else:
                counter+=1
            if counter>=patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    plt.ioff()
    plt.savefig("training_curve_buy.png")
    plt.show()
    print("✅ (買) cGAN 訓練完畢，已儲存 Generator 權重到 cgan_generator_buy.pth")

# (檔案結束)
