# views.py
from django.shortcuts import render
from django.http import JsonResponse
import os
from django.conf import settings

# 「必須」匯入這兩個：TRAINING_STATUS_SAC 與 start_background_sac_training
from .tasks import TRAINING_STATUS_SAC, ASSOCIATION_RULES, start_background_sac_training
from .sac_model import sac_component


def start_training_view(request):
    # 每次點「開始訓練」前，把舊的 sac_reward.png 刪掉，避免瀏覽器快取
    print(1)
    plot_path = os.path.join(settings.BASE_DIR, 'statics', 'images', 'sac_reward.png')
    if os.path.exists(plot_path):
        os.remove(plot_path)

    message = None
    if request.method == "POST":
        print("a")
        try:
            print("a")
            # 解析基本訓練參數
            num_episodes  = int(request.POST.get("num_episodes", 100))
            max_steps     = int(request.POST.get("max_steps", 50))
            product_count = int(request.POST.get("product_count", 3))

            # 1) 解析「商品上下限」並更新 sac_component 裡的全域 buy_price_bounds / sell_price_bounds
            new_buy_bounds  = []
            new_sell_bounds = []
            print("a")
            for i in range(product_count):
                buy_low   = float(request.POST.get(f"buy_low_{i}"))
                buy_high  = float(request.POST.get(f"buy_high_{i}"))
                sell_low  = float(request.POST.get(f"sell_low_{i}"))
                sell_high = float(request.POST.get(f"sell_high_{i}"))
                new_buy_bounds.append((buy_low, buy_high))
                new_sell_bounds.append((sell_low, sell_high))

            sac_component.buy_price_bounds[:]  = new_buy_bounds
            sac_component.sell_price_bounds[:] = new_sell_bounds
            print("b")
            # 2) 解析「關聯規則」
            #    先清空全域 ASSOCIATION_RULES
            ASSOCIATION_RULES.clear()

            # 從 POST 取 rules_count，如果沒有，預設為 0
            rules_count = int(request.POST.get("rules_count", 0))
            for i in range(rules_count):
                '''
                keys = [f"support_{i}", f"confidence_{i}", f"probability_{i}"]
                for k in keys:
                    # 先印出每個 key 真正從 POST 拿到的值
                    print(f"[DEBUG] {k} = '{request.POST.get(k, '')}'")
                # 如果某欄位空字串，就馬上拋錯或者做預設
                for k in keys:
                    v = request.POST.get(k, "").strip()
                    if v == "":
                        raise ValueError(f"欄位「{k}」沒有填值，請確認「Support/Confidence/Probability」都有輸入。")
                '''


                sup_list  = request.POST.getlist(f"support_{i}")
                conf_list = request.POST.getlist(f"confidence_{i}")
                prob_list = request.POST.getlist(f"probability_{i}")

                # 只取「第一個非空值」，後面那個空字串就丟掉
                sup_str  = next((v for v in sup_list  if v.strip()), "")
                conf_str = next((v for v in conf_list if v.strip()), "")
                prob_str = next((v for v in prob_list if v.strip()), "")

                print(f"[DEBUG] support_{i} = '{sup_str}'")
                print(f"[DEBUG] confidence_{i} = '{conf_str}'")
                print(f"[DEBUG] probability_{i} = '{prob_str}'")

                # -------- 基本欄位 --------
                antecedent = [
                    x.strip() for x in
                    request.POST.get(f"antecedent_{i}", "").split(",") if x.strip()
                ]
                consequent = request.POST.get(f"consequent_{i}", "").strip()
                rule_type  = request.POST.get(f"rule_type_{i}")

                if sup_str == "":
                    raise ValueError(f"欄位 support_{i} 沒有填值！")

                rule = {
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "type":       rule_type,
                    "support":    float(sup_str)
                }

                if rule_type == "complementary":
                    if conf_str == "":
                        raise ValueError(f"欄位 confidence_{i} 沒有填值！")
                    rule["confidence"] = float(conf_str)
                else:  # substitute
                    if prob_str == "":
                        raise ValueError(f"欄位 probability_{i} 沒有填值！")
                    rule["probability"] = float(prob_str)

                ASSOCIATION_RULES.append(rule)


            print("[DEBUG] 在啟動背景訓練前，buy_price_bounds:", sac_component.buy_price_bounds)
            # 3) 啟動 SAC 訓練背景執行緒，把剛剛解析的 ASSOCIATION_RULES 一併傳進去
            start_background_sac_training(
                num_episodes=num_episodes,
                max_steps=max_steps,
                product_count=product_count,
                rules_list=ASSOCIATION_RULES
            )
            print("[DEBUG] 已經呼叫 start_background_sac_training")

            message = "完成：已開始 SAC 背景訓練。"


        except Exception as e:
            message = f"輸入錯誤或訓練失敗：{e}"
        for i in range(product_count):
            # —— 把這 4 個欄位先印出來 ——  
            debug_keys = [
                f"buy_low_{i}",  f"buy_high_{i}",
                f"sell_low_{i}", f"sell_high_{i}"
            ]
            for k in debug_keys:
                v = request.POST.get(k, "")
                print(f"[DEBUG] {k} = '{v}'")

            # 任何空值就直接丟更清楚的錯
            for k in debug_keys:
                v = request.POST.get(k, "").strip()
                if v == "":
                    raise ValueError(f"欄位「{k}」沒有填值！")

            # —— 接下來才做 float 轉換 ——  
            buy_low  = float(request.POST[f"buy_low_{i}"])
            buy_high = float(request.POST[f"buy_high_{i}"])
            sell_low = float(request.POST[f"sell_low_{i}"])
            sell_high= float(request.POST[f"sell_high_{i}"])


    return render(request, "new_training.html", {"message": message})


def training_status_api(request):
    # 前端會每秒 fetch() 這個 API，拿最新的 TRAINING_STATUS_SAC
    response = JsonResponse(TRAINING_STATUS_SAC)
    response["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response
'''
def market_status_view(request):
    api_url = "https://4cd5-59-125-237-249.ngrok-free.app/api/market/status"
    market_data = None
    error_message = None

    try:
        # 向 API 發送請求
        response = requests.get(api_url)
        response.raise_for_status()  # 如果有錯誤，會拋出異常
        market_data = response.json()  # 將回應轉為 JSON 格式
    except requests.RequestException as e:
        error_message = f"無法取得資料：{e}"

    # 將資料傳遞給模板
    return render(request, "market_status.html", {
        "market_data": market_data,
        "error_message": error_message
    })
'''
# myapp/views.py
from django.shortcuts import render
import requests

def server_status_view(request):
    market_api_url  = "http://127.0.0.1:4567/api/market/status"
    player_api_url  = "http://127.0.0.1:4567/api/player/status"

    market_data     = []     # 保留每個玩家的買賣卡片資料
    player_data     = {}     
    sell_prices     = []     # 伺服器當天的「全部品項出售價」
    buy_prices      = []     # 伺服器當天的「全部品項採購價」
    error_message   = None

    headers = {
        "ngrok-skip-browser-warning": "true"
    }

    try:
        # 1) 先拿到市場原始資料
        market_resp = requests.get(market_api_url, headers=headers, timeout=5)
        market_resp.raise_for_status()
        raw_market = market_resp.json()
        # raw_market 範例：
        # [
        #   {
        #     "player": "Evzen11111",
        #     "type": "sell",
        #     "prices": [25.0, 350.0, 20.0, … ],
        #     "products": {"APPLE":0, "BLAZE_ROD":0, … }
        #   },
        #   { "player": "Evzen11111", "type": "buy", "prices": […], "products": {…} },
        #   { … },
        #   …
        # ]

        # －－－－－－－－－－－－－－－－－－－－－－－－－－
        # （一）先把每個玩家的「買賣卡片資料」轉成前端要用的格式
        # －－－－－－－－－－－－－－－－－－－－－－－－－－
        market_data = []
        for item in raw_market:
            # 直接把 player、prices、type 拿過來，products 改成 list of tuples (顯示名稱, 數量)
            new_item = {
                "player": item.get("player", ""),
                "type": item.get("type", ""),        # "sell" 或 "buy"
                "prices": item.get("prices", []),
            }
            # 把 products 裡的 key 由 "IRON_INGOT" 換成 "Iron Ingot"，並保留數量
            product_list = []
            for name, qty in item.get("products", {}).items():
                display_name = name.replace("_", " ").title()
                product_list.append((display_name, qty))
            new_item["products"] = product_list

            market_data.append(new_item)

        # －－－－－－－－－－－－－－－－－－－－－－－－－－
        # （二）從 raw_market 裡找出第一組 type="sell" 和 type="buy" 的「全品項價格陣列」
        # －－－－－－－－－－－－－－－－－－－－－－－－－－
        # 先取出「產品名稱順序」（假設每個物件裡 products dict keys 的順序都一致）
        if raw_market:
            product_keys = list(raw_market[0].get("products", {}).keys())
        else:
            product_keys = []

        # 找到第一個 type="sell" 的 entry
        sell_entry = next((x for x in raw_market if x.get("type") == "sell"), None)
        # 找到第一個 type="buy" 的 entry
        buy_entry  = next((x for x in raw_market if x.get("type") == "buy"), None)

        if sell_entry:
            for idx, key in enumerate(product_keys):
                # 把 "IRON_INGOT" 換成 "Iron Ingot"
                display_name = key.replace("_", " ").title()
                price_val   = sell_entry.get("prices", [])[idx] if idx < len(sell_entry.get("prices", [])) else 0
                sell_prices.append({
                    "name": display_name,
                    "price": price_val
                })

        if buy_entry:
            for idx, key in enumerate(product_keys):
                display_name = key.replace("_", " ").title()
                price_val    = buy_entry.get("prices", [])[idx] if idx < len(buy_entry.get("prices", [])) else 0
                buy_prices.append({
                    "name": display_name,
                    "price": price_val
                })

        # －－－－－－－－－－－－－－－－－－－－－－－－－－
        # （三）拿玩家人數狀態
        # －－－－－－－－－－－－－－－－－－－－－－－－－－
        player_resp = requests.get(player_api_url, headers=headers, timeout=5)
        player_resp.raise_for_status()
        raw_player = player_resp.json()
        # raw_player 範例：{"onlinePlayers":0,"loggedInToday":1}

        player_data = {
            "onlinePlayers": raw_player.get("onlinePlayers", 0),
            "loggedInToday": raw_player.get("loggedInToday", 0),
            "loggedInTodayPlus2": int(raw_player.get("loggedInToday", 0)) + 2
        }

    except requests.Timeout:
        error_message = "請求超時，伺服器可能忙碌中"
    except ValueError as e:
        error_message = f"JSON 解析錯誤：{e}"
    except requests.RequestException as e:
        error_message = f"無法取得資料：{e}"

    context = {
        "market_data":   market_data,
        "player_data":   player_data,
        "sell_prices":   sell_prices,
        "buy_prices":    buy_prices,
        "error_message": error_message,
    }
    return render(request, "server_status.html", context)






