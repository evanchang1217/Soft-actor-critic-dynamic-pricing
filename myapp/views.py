import os

import requests
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render

from .sac_model import sac_component
from .tasks import ASSOCIATION_RULES, TRAINING_STATUS_SAC, start_background_sac_training


def start_training_view(request):
    plot_path = os.path.join(settings.BASE_DIR, "statics", "images", "sac_reward.png")
    if os.path.exists(plot_path):
        os.remove(plot_path)

    message = None
    if request.method == "POST":
        try:
            num_episodes = int(request.POST.get("num_episodes", 100))
            max_steps = int(request.POST.get("max_steps", 50))
            product_count = int(request.POST.get("product_count", 3))

            new_buy_bounds = []
            new_sell_bounds = []
            for i in range(product_count):
                buy_low = float(request.POST.get(f"buy_low_{i}"))
                buy_high = float(request.POST.get(f"buy_high_{i}"))
                sell_low = float(request.POST.get(f"sell_low_{i}"))
                sell_high = float(request.POST.get(f"sell_high_{i}"))
                new_buy_bounds.append((buy_low, buy_high))
                new_sell_bounds.append((sell_low, sell_high))

            sac_component.buy_price_bounds[:] = new_buy_bounds
            sac_component.sell_price_bounds[:] = new_sell_bounds

            ASSOCIATION_RULES.clear()
            rules_count = int(request.POST.get("rules_count", 0))
            for i in range(rules_count):
                sup_list = request.POST.getlist(f"support_{i}")
                conf_list = request.POST.getlist(f"confidence_{i}")
                prob_list = request.POST.getlist(f"probability_{i}")

                sup_str = next((value for value in sup_list if value.strip()), "")
                conf_str = next((value for value in conf_list if value.strip()), "")
                prob_str = next((value for value in prob_list if value.strip()), "")

                antecedent = [
                    value.strip()
                    for value in request.POST.get(f"antecedent_{i}", "").split(",")
                    if value.strip()
                ]
                consequent = request.POST.get(f"consequent_{i}", "").strip()
                rule_type = request.POST.get(f"rule_type_{i}")

                if sup_str == "":
                    raise ValueError(f"Missing support value for rule {i + 1}.")

                rule = {
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "type": rule_type,
                    "support": float(sup_str),
                }

                if rule_type == "complementary":
                    if conf_str == "":
                        raise ValueError(f"Missing confidence value for rule {i + 1}.")
                    rule["confidence"] = float(conf_str)
                else:
                    if prob_str == "":
                        raise ValueError(f"Missing probability value for rule {i + 1}.")
                    rule["probability"] = float(prob_str)

                ASSOCIATION_RULES.append(rule)

            start_background_sac_training(
                num_episodes=num_episodes,
                max_steps=max_steps,
                product_count=product_count,
                rules_list=ASSOCIATION_RULES,
            )
            message = "SAC training has started."
        except Exception as exc:
            message = f"Failed to start SAC training: {exc}"

    return render(request, "new_training.html", {"message": message})


def training_status_api(request):
    response = JsonResponse(TRAINING_STATUS_SAC)
    response["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


def server_status_view(request):
    market_api_url = "http://127.0.0.1:4567/api/market/status"
    player_api_url = "http://127.0.0.1:4567/api/player/status"

    market_data = []
    player_data = {}
    sell_prices = []
    buy_prices = []
    error_message = None

    headers = {"ngrok-skip-browser-warning": "true"}

    try:
        market_resp = requests.get(market_api_url, headers=headers, timeout=5)
        market_resp.raise_for_status()
        raw_market = market_resp.json()

        for item in raw_market:
            new_item = {
                "player": item.get("player", ""),
                "type": item.get("type", ""),
                "prices": item.get("prices", []),
            }

            product_list = []
            for name, qty in item.get("products", {}).items():
                display_name = name.replace("_", " ").title()
                product_list.append((display_name, qty))
            new_item["products"] = product_list

            market_data.append(new_item)

        if raw_market:
            product_keys = list(raw_market[0].get("products", {}).keys())
        else:
            product_keys = []

        sell_entry = next((entry for entry in raw_market if entry.get("type") == "sell"), None)
        buy_entry = next((entry for entry in raw_market if entry.get("type") == "buy"), None)

        if sell_entry:
            for idx, key in enumerate(product_keys):
                display_name = key.replace("_", " ").title()
                price_val = (
                    sell_entry.get("prices", [])[idx]
                    if idx < len(sell_entry.get("prices", []))
                    else 0
                )
                sell_prices.append({"name": display_name, "price": price_val})

        if buy_entry:
            for idx, key in enumerate(product_keys):
                display_name = key.replace("_", " ").title()
                price_val = (
                    buy_entry.get("prices", [])[idx]
                    if idx < len(buy_entry.get("prices", []))
                    else 0
                )
                buy_prices.append({"name": display_name, "price": price_val})

        player_resp = requests.get(player_api_url, headers=headers, timeout=5)
        player_resp.raise_for_status()
        raw_player = player_resp.json()

        player_data = {
            "onlinePlayers": raw_player.get("onlinePlayers", 0),
            "loggedInToday": raw_player.get("loggedInToday", 0),
        }

    except requests.Timeout:
        error_message = "Request to the local market API timed out."
    except ValueError as exc:
        error_message = f"Failed to decode JSON response: {exc}"
    except requests.RequestException as exc:
        error_message = f"Failed to connect to the local market API: {exc}"

    context = {
        "market_data": market_data,
        "player_data": player_data,
        "sell_prices": sell_prices,
        "buy_prices": buy_prices,
        "error_message": error_message,
    }
    return render(request, "server_status.html", context)
