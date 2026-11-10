"""Django adapters for the shared research code and optional game API."""

import json

from django.conf import settings
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_POST, require_http_methods

from experiments.baselines import METHODS
from integrations.minecraft import MinecraftAPIError, MinecraftClient, sample_status
from rl.config import MarketConfig, ProductConfig, SACConfig
from .tasks import runner


REWARD_PROFILES = {
    "balanced_v1": {"reward_count": 0.5, "reward_volume": 0.5, "reward_cpi": 0.05, "reward_money": 1.0},
    "legacy": {"reward_count": 1.0, "reward_volume": 1.0, "reward_cpi": 0.5, "reward_money": 0.5},
}


@require_http_methods(["GET", "POST"])
def start_training_view(request):
    error, status = None, 200
    market = MarketConfig(reward_mode="balanced_v1", **REWARD_PROFILES["balanced_v1"])
    defaults = {"method": "sac", "reward_mode": "balanced_v1", "episodes": 100, "max_steps": 100,
                "num_players": 200, "seed": 0, "shock_step": 50}
    previous = runner.snapshot()
    if previous.get("configuration"):
        configuration = previous["configuration"]
        market = MarketConfig.from_dict(configuration["market"])
        defaults = {"method": previous["method"], "reward_mode": market.reward_mode,
                    "episodes": configuration["sac"]["episodes"],
                    "max_steps": market.max_steps, "num_players": market.num_players,
                    "seed": configuration["seed"], "shock_step": market.shock_step or ""}
    products = [p.__dict__ for p in market.products]
    rules = json.dumps(list(market.association_rules))
    if request.method == "POST":
        # Keep the submitted values even when parsing or validation fails.
        defaults = {"method": request.POST.get("method", "sac"),
                    "reward_mode": request.POST.get("reward_mode", "balanced_v1"),
                    "episodes": request.POST.get("episodes", "100"),
                    "max_steps": request.POST.get("max_steps", "100"),
                    "num_players": request.POST.get("num_players", "200"),
                    "seed": request.POST.get("seed", "0"),
                    "shock_step": request.POST.get("shock_step", "50")}
        rules = request.POST.get("association_rules", "[]")
        try:
            method = defaults["method"]
            if method not in METHODS:
                raise ValueError("Choose a supported pricing method")
            reward_mode = defaults["reward_mode"]
            if reward_mode not in REWARD_PROFILES:
                raise ValueError("Choose Balanced currency + trade or Legacy trade-target reward")
            count = int(request.POST.get("product_count", "3"))
            if not 1 <= count <= 12:
                raise ValueError("The web demo supports 1–12 products")
            price_keys = ("buy_low", "buy_high", "sell_low", "sell_high")
            products = [{"name": request.POST.get(f"name_{index}", f"Product {index + 1}"),
                         **{key: request.POST.get(f"{key}_{index}", "") for key in price_keys}}
                        for index in range(count)]
            episodes = int(defaults["episodes"])
            steps = int(defaults["max_steps"])
            players = int(defaults["num_players"])
            seed = int(defaults["seed"])
            association_rules = json.loads(rules)
            if not isinstance(association_rules, list) or len(association_rules) > 30:
                raise ValueError("Association rules must be a JSON list with at most 30 entries")
            shock = defaults["shock_step"].strip()
            if not (1 <= episodes <= 200 and 1 <= steps <= 500 and episodes * steps <= 20_000):
                raise ValueError("Use 1–200 episodes, 1–500 steps, and at most 20,000 total steps. Use the CLI for larger runs.")
            if not 2 <= players <= 1000 or not 0 <= seed <= 2**31 - 1:
                raise ValueError("Use 2–1,000 players and a seed between 0 and 2,147,483,647")
            validated_products = tuple(ProductConfig(name=p["name"], **{key: float(p[key]) for key in price_keys})
                                       for p in products)
            market = MarketConfig(products=validated_products, num_players=players,
                                  max_steps=steps, shock_step=int(shock) if shock else None,
                                  association_rules=tuple(association_rules), reward_mode=reward_mode,
                                  **REWARD_PROFILES[reward_mode])
            # Match the study at the default 10,000-step budget. Shorter runs
            # reduce warmup for demonstration; they are not the research protocol.
            sac = SACConfig(episodes=episodes, hidden=128, batch_size=128, control_fee=False,
                            warmup_steps=min(1000, episodes * steps // 4))
            if not runner.start(market, sac, seed, method, settings.DEMO_RUNS_DIR):
                error, status = "A training run or saved-model comparison is active. Wait for completion or stop the training run.", 409
            else:
                return redirect("home")
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            error, status = str(exc), 400
    return render(request, "new_training.html", {"defaults": defaults, "products": products,
                                                "association_rules": rules, "error": error,
                                                "methods": METHODS}, status=status)


@require_GET
def training_status_api(request):
    response = JsonResponse(runner.snapshot())
    response["Cache-Control"] = "no-store"
    return response


@require_POST
def stop_training(request):
    runner.cancel()
    return JsonResponse({"message": "Stop requested"})


@require_GET
def training_artifact(request, run_id, name):
    path = runner.artifact(run_id, name)
    if path is None or not path.is_file():
        raise Http404("Artifact is not available")
    return FileResponse(path.open("rb"), as_attachment=True, filename=name)


@require_GET
def server_status_view(request):
    mode = settings.MINECRAFT_MODE
    context = {"mode": mode, "error": None, "market": [], "players": {}}
    try:
        if mode == "sample":
            data = sample_status()
        elif mode == "live":
            data = MinecraftClient(settings.MINECRAFT_API_BASE_URL, settings.MINECRAFT_API_TIMEOUT).status()
        else:
            raise ValueError("MINECRAFT_MODE must be sample or live")
        context["players"] = data["players"]
        context["market"] = [
            {"player": entry["player"], "type": entry["type"],
             "products": [{"name": name.replace("_", " ").title(), "quantity": quantity, "price": price}
                          for (name, quantity), price in zip(entry["products"].items(), entry["prices"])]}
            for entry in data["market"]]
    except (MinecraftAPIError, ValueError) as exc:
        context["error"] = str(exc)
    return render(request, "server_status.html", context)
