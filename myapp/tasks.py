# tasks.py

from .sac_model.sac_component import buy_price_bounds, sell_price_bounds
from .sac_model.sac_component import run_sac_training
import threading

# SAC 訓練狀態（與 DQN TRAINING_STATUS 分開）
TRAINING_STATUS_SAC = {
    "running": False,               # SAC 是否正在跑
    "current_episode": 0,           # 当前跑到第几回合
    "total_episodes": 0,            # 一共要跑多少回合
    "avg_buy_price": [],            # 每个商品的即时平均买价
    "avg_sell_price": [],           # 每个商品的即时平均卖价
    "reward": 0.0,                  # 本回合的 reward（或累积 reward，可自行定义）
    "cpi": 0.0,                     # 本回合统计的 CPI
    "player_gold_avg": 0.0,         # 平均玩家金钱
    "per_product_stats": [],        # 每个商品的统计（人数与量）
    "message": "",                  # 状态消息，比如 “SAC Episode 3/100”
    "final_buy_bounds": buy_price_bounds,   # 初始的买价上下限列表
    "final_sell_bounds": sell_price_bounds, # 初始的卖价上下限列表
}

# —— 新增全域變數：用來暫存「前端傳過來的互補/替代品規則」 —— 
ASSOCIATION_RULES = []


def _sac_progress_callback(**payload):
    """
    SAC 在每个 episode 结束后会调用这个回调，把当下的统计信息打包成 kwargs 传进来，
    payload 里必须包含以下字段（名字要和 TRAINING_STATUS_SAC 保持一致）：
      - current_episode
      - total_episodes
      - avg_buy_price        （长度 = num_products）
      - avg_sell_price       （长度 = num_products）
      - reward
      - cpi
      - player_gold_avg
      - per_product_stats    （list of dict，见下方举例）
    """
    # 1) 先把 payload 里所有同名字段都更新到 TRAINING_STATUS_SAC
    TRAINING_STATUS_SAC.update(payload)

    # 2) 再单独更新 message 为 “SAC Episode X/总共 Y”
    TRAINING_STATUS_SAC["message"] = (
        f"SAC Episode {payload.get('current_episode')}/{payload.get('total_episodes')}"
    )

    # 3)（可选）在后端控制台输出一行 debug
    print(f"[SAC DEBUG] {TRAINING_STATUS_SAC['message']}")


def start_background_sac_training(num_episodes, max_steps, product_count, rules_list=None):
    """
    在后台启动一个线程，执行 run_sac_training(...)。每个 episode 结束之后，
    run_sac_training 内部会调用 _sac_progress_callback(...)，它又会把所有字段 update 到 TRAINING_STATUS_SAC。

    新增参数：
      - rules_list: list of dict
          前端输入的互补品/替代品规则，每一条规则格式类似：
            {
              "antecedent": ["0", "1"],      # 前置項的商品索引列表 (字串形式)
              "consequent": "2",             # 結論項的商品索引 (字串)
              "type": "complementary",       # "complementary" 或 "substitute"
              "confidence": 0.6,             # 互補品才有
              "support": 0.7,                # 共用
              "probability": 0.65            # 替代品才有
            }
    """
    def _bg():
        # 标记 “正在跑 SAC”
        print("→ 進入 _bg thread") 
        TRAINING_STATUS_SAC["running"] = True
        TRAINING_STATUS_SAC["message"] = "SAC 訓練啟動"

        # —— 把 rules_list 写入全域，以备 run_sac_training 內部或環境使用 —— 
        global ASSOCIATION_RULES
        ASSOCIATION_RULES.clear()
        if rules_list:
            # 深複製一份，確保原列表不被外部意外修改
            import copy
            ASSOCIATION_RULES = copy.deepcopy(rules_list)

        try:
            # 把 rules_list（即 ASSOCIATION_RULES）傳給 run_sac_training
            run_sac_training(
                num_episodes=num_episodes,
                max_steps=max_steps,
                num_products=product_count,
                progress_callback=_sac_progress_callback,
                association_rules=ASSOCIATION_RULES  # 新增這行
            )
            # 一旦 run_sac_training 返回（训练结束），就修改状态
            TRAINING_STATUS_SAC["message"] = "SAC 訓練完成"
        except Exception as e:
            # 如果过程中出现异常，也要把状态更新，让前端能看到错误
            TRAINING_STATUS_SAC["message"] = f"SAC 失敗：{e}"
        finally:
            TRAINING_STATUS_SAC["running"] = False

    threading.Thread(target=_bg, daemon=True).start()


'''
import threading
from .dqn_net import dqn_store

TRAINING_STATUS = {
    "running": False,
    "current_episode": 0,
    "total_episodes": 0,
    "avg_buy_price": [],
    "avg_sell_price": [],
    "reward": 0.0,
    "message": "",
    "final_buy_bounds": dqn_store.buy_price_bounds,
    "final_sell_bounds": dqn_store.sell_price_bounds
}

def _progress_callback(episode, total_episodes, avg_buy_price, avg_sell_price, reward):
    TRAINING_STATUS["current_episode"] = episode
    TRAINING_STATUS["total_episodes"] = total_episodes
    TRAINING_STATUS["avg_buy_price"] = avg_buy_price
    TRAINING_STATUS["avg_sell_price"] = avg_sell_price
    TRAINING_STATUS["reward"] = reward
    TRAINING_STATUS["message"] = f"Episode {episode} / {total_episodes}"
    TRAINING_STATUS["final_buy_bounds"] = dqn_store.buy_price_bounds
    TRAINING_STATUS["final_sell_bounds"] = dqn_store.sell_price_bounds
    print(f"🔴 [DEBUG] Updated TRAINING_STATUS: {TRAINING_STATUS}")

def _background_training(num_episodes, max_steps, product_count):
    TRAINING_STATUS["running"] = True
    TRAINING_STATUS["message"] = "Training started"
    TRAINING_STATUS["current_episode"] = 0
    TRAINING_STATUS["avg_buy_price"] = []
    TRAINING_STATUS["avg_sell_price"] = []
    TRAINING_STATUS["reward"] = 0.0

    dqn_store.run_dqn_training(
        num_episodes=num_episodes,
        max_steps=max_steps,
        batch_size=256,
        num_products=product_count,
        progress_callback=_progress_callback
    )

    TRAINING_STATUS["running"] = False
    TRAINING_STATUS["message"] = "Training complete!"

def start_background_training(num_episodes, max_steps, product_count):
    t = threading.Thread(
        target=_background_training,
        args=(num_episodes, max_steps, product_count)
    )
    t.start()
'''