import threading
from copy import deepcopy

from .sac_model.sac_component import buy_price_bounds, run_sac_training, sell_price_bounds

TRAINING_STATUS_SAC = {
    "running": False,
    "current_episode": 0,
    "total_episodes": 0,
    "avg_buy_price": [],
    "avg_sell_price": [],
    "reward": 0.0,
    "cpi": 0.0,
    "player_gold_avg": 0.0,
    "per_product_stats": [],
    "message": "",
    "final_buy_bounds": buy_price_bounds,
    "final_sell_bounds": sell_price_bounds,
}

ASSOCIATION_RULES = []


def _sac_progress_callback(**payload):
    TRAINING_STATUS_SAC.update(payload)
    TRAINING_STATUS_SAC["message"] = (
        f"SAC Episode {payload.get('current_episode')}/{payload.get('total_episodes')}"
    )


def start_background_sac_training(num_episodes, max_steps, product_count, rules_list=None):
    def _bg():
        TRAINING_STATUS_SAC["running"] = True
        TRAINING_STATUS_SAC["message"] = "SAC training is running."

        global ASSOCIATION_RULES
        ASSOCIATION_RULES.clear()
        if rules_list:
            ASSOCIATION_RULES = deepcopy(rules_list)

        try:
            run_sac_training(
                num_episodes=num_episodes,
                max_steps=max_steps,
                num_products=product_count,
                progress_callback=_sac_progress_callback,
                association_rules=ASSOCIATION_RULES,
            )
            TRAINING_STATUS_SAC["message"] = "SAC training finished."
        except Exception as exc:
            TRAINING_STATUS_SAC["message"] = f"SAC training failed: {exc}"
        finally:
            TRAINING_STATUS_SAC["running"] = False

    threading.Thread(target=_bg, daemon=True).start()
