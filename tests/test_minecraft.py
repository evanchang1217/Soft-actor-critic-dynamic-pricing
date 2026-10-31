from unittest.mock import Mock, patch

import pytest
import requests

from integrations.minecraft import MinecraftAPIError, MinecraftClient, sample_status, validate_status


def test_sample_fixture_and_live_api_contract():
    sample = sample_status()
    with patch("integrations.minecraft.requests.get") as get:
        market, players = Mock(), Mock()
        market.json.return_value = sample["market"]
        players.json.return_value = sample["players"]
        get.side_effect = [market, players]
        assert MinecraftClient("http://127.0.0.1:4567/", timeout=2).status() == sample
        assert get.call_args_list[0].args[0] == "http://127.0.0.1:4567/api/market/status"
        assert get.call_args_list[1].args[0] == "http://127.0.0.1:4567/api/player/status"
        assert get.call_args_list[0].kwargs["timeout"] == 2


def test_api_failures_and_invalid_response_are_clear():
    with patch("integrations.minecraft.requests.get", side_effect=requests.Timeout):
        with pytest.raises(MinecraftAPIError, match="Could not read"):
            MinecraftClient().status()
    with pytest.raises(MinecraftAPIError):
        validate_status({"unexpected": "object"}, {})
    sample = sample_status()
    sample["market"][0]["prices"] = [float("nan")] * 3
    with pytest.raises(MinecraftAPIError, match="finite"):
        validate_status(sample["market"], sample["players"])
