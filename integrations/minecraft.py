"""Read-only client for the custom game-server API used by the original demo."""

import json
import math
from pathlib import Path
from urllib.parse import urlparse

import requests


class MinecraftAPIError(RuntimeError):
    pass


def validate_status(market, players):
    if not isinstance(market, list) or not isinstance(players, dict):
        raise MinecraftAPIError("Expected market list and player-status object")
    for entry in market:
        if (not isinstance(entry, dict) or not isinstance(entry.get("player"), str)
                or entry.get("type") not in ("buy", "sell")
                or not isinstance(entry.get("products"), dict)
                or not isinstance(entry.get("prices"), list)):
            raise MinecraftAPIError("Market entries require player, type, products, and prices")
        if len(entry["prices"]) != len(entry["products"]):
            raise MinecraftAPIError("Each product must have a corresponding price")
        if any(not isinstance(name, str) for name in entry["products"]):
            raise MinecraftAPIError("Product names must be strings")
        for value in [*entry["products"].values(), *entry["prices"]]:
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise MinecraftAPIError("Prices and quantities must be finite and nonnegative")
    for name in ("onlinePlayers", "loggedInToday"):
        if type(players.get(name)) is not int or players[name] < 0:
            raise MinecraftAPIError(f"{name} must be a nonnegative integer")
    return {"market": market, "players": players}


class MinecraftClient:
    def __init__(self, base_url="http://127.0.0.1:4567", timeout=5.0):
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("Minecraft API base URL must be http(s)")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("API timeout must be positive")
        self.base_url, self.timeout = base_url.rstrip("/"), timeout

    def _get(self, endpoint):
        try:
            response = requests.get(self.base_url + endpoint, timeout=self.timeout,
                                    headers={"ngrok-skip-browser-warning": "true"})
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            raise MinecraftAPIError("Could not read the game API; check the service, URL, and response format") from error

    def status(self):
        return validate_status(self._get("/api/market/status"), self._get("/api/player/status"))


def sample_status():
    data = json.loads((Path(__file__).parent / "fixtures" / "market_status.json").read_text(encoding="utf-8"))
    return validate_status(data["market"], data["players"])
