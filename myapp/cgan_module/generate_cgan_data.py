# cgan_module/generate_cgan_data.py

import csv
import numpy as np
import pandas as pd
# 注意：從上層匯入 dqn_store
from ..dqn_net import dqn_store

def generate_data_for_buy(csv_filename_buy="my_data_buy.csv", num_records=500):
    """
    依照 dqn_store.buy_price_bounds 生成一份 CSV，供 cGAN (Buy) 訓練。
    """
    num_items = len(dqn_store.buy_price_bounds)
    all_data = []

    for _ in range(num_records):
        # 在 [0.1, 1.0] 範圍隨機生成比例
        prices_offset = np.round(np.random.uniform(0.1, 1.0, size=num_items), 2)

        # 從 dqn_store.buy_price_bounds 中分別取出 low/high
        bp_lows  = np.array([pair[0] for pair in dqn_store.buy_price_bounds])
        bp_highs = np.array([pair[1] for pair in dqn_store.buy_price_bounds])

        # 計算商品價格 = (high - low) * offset + low
        prices = (bp_highs - bp_lows) * prices_offset + bp_lows

        # 簡單假設 offset 越大，玩家越有可能購賣
        buy_prob = np.clip(prices_offset, 0, 1)
        basket = (np.random.rand(num_items) < buy_prob).astype(int)

        row = list(prices) + list(basket)
        all_data.append(row)

    # 建立 CSV
    header = [f'price_{k}' for k in range(num_items)] + [f'商品{chr(ord("A")+k)}' for k in range(num_items)]
    with open(csv_filename_buy, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in all_data:
            writer.writerow(row)

    print(f"[generate_data_for_buy] 產生 {csv_filename_buy} 共{num_records}筆, {num_items}商品.")


def generate_data_for_sell(csv_filename_sell="my_data_sell.csv", num_records=500):
    """
    依照 dqn_store.sell_price_bounds 生成一份 CSV，供 cGAN (Sell) 訓練。
    """
    num_items = len(dqn_store.sell_price_bounds)
    all_data = []

    for _ in range(num_records):
        prices_offset = np.round(np.random.uniform(0.1, 1.0, size=num_items), 2)
        sp_lows  = np.array([pair[0] for pair in dqn_store.sell_price_bounds])
        sp_highs = np.array([pair[1] for pair in dqn_store.sell_price_bounds])

        prices = (sp_highs - sp_lows) * prices_offset + sp_lows

        # 模擬：offset 越小，價格越低，玩家買的意願也可能越高

        sell_prob = np.clip( 1.2 - prices_offset, 0, 1)
        basket = (np.random.rand(num_items) < sell_prob).astype(int)

        row = list(prices) + list(basket)
        all_data.append(row)

    header = [f'price_{k}' for k in range(num_items)] + [f'商品{chr(ord("A")+k)}' for k in range(num_items)]
    with open(csv_filename_sell, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in all_data:
            writer.writerow(row)

    print(f"[generate_data_for_sell] 產生 {csv_filename_sell} 共{num_records}筆, {num_items}商品.")