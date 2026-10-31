from unittest.mock import patch

import pytest

from myapp.tasks import DemoRunner
from rl.config import MarketConfig, SACConfig


def valid_form():
    return {"method": "fixed", "episodes": "2", "max_steps": "4", "num_players": "10", "seed": "1",
            "product_count": "1", "name_0": "Iron", "buy_low_0": "10", "buy_high_0": "20",
            "sell_low_0": "25", "sell_high_0": "50", "association_rules": "[]", "shock_step": "2"}


def test_pages_render_and_sample_mode_makes_no_network_calls(client, settings):
    settings.MINECRAFT_MODE = "sample"
    with patch("integrations.minecraft.requests.get") as get:
        response = client.get("/")
        assert response.status_code == 200
        assert b"Configure a run" in response.content
        response = client.get("/server_status/")
        assert response.status_code == 200
        assert b"SAMPLE DATA" in response.content
        assert b"Iron Ingot" in response.content
        get.assert_not_called()
    assert client.get("/training_status_api/")["Cache-Control"] == "no-store"


def test_invalid_input_is_rejected_before_worker_starts(client):
    form = valid_form()
    form["sell_low_0"] = "0"
    with patch("myapp.views.runner.start") as start:
        assert client.post("/", form).status_code == 400
        start.assert_not_called()
    form = valid_form()
    form["episodes"] = "200"
    form["max_steps"] = "500"
    assert client.post("/", form).status_code == 400


@pytest.mark.parametrize("invalid_field,invalid_value", [
    ("association_rules", "["),
    ("buy_high_0", "5"),
    ("max_steps", "not-a-number"),
])
def test_invalid_form_preserves_submitted_configuration(client, invalid_field, invalid_value):
    form = valid_form()
    form.update({"method": "adaptive", "episodes": "7", "max_steps": "9", "num_players": "11",
                 "seed": "42", "shock_step": "", "name_0": "Custom product"})
    form[invalid_field] = invalid_value
    with patch("myapp.views.runner.start") as start:
        response = client.post("/", form)
    assert response.status_code == 400
    start.assert_not_called()
    assert response.context["defaults"] == {key: form[key] for key in
        ("method", "episodes", "max_steps", "num_players", "seed", "shock_step")}
    assert response.context["products"][0]["name"] == "Custom product"
    assert response.context["products"][0]["buy_high"] == form["buy_high_0"]
    assert response.context["association_rules"] == form["association_rules"]


def test_submit_and_conflicting_run(client):
    with patch("myapp.views.runner.start", return_value=True) as start:
        response = client.post("/", valid_form())
        assert response.status_code == 302
        assert start.call_args.args[0].num_players == 10
    with patch("myapp.views.runner.start", return_value=False):
        assert client.post("/", valid_form()).status_code == 409


def test_csrf_is_required_for_start_and_stop():
    from django.test import Client
    client = Client(enforce_csrf_checks=True)
    assert client.post("/", valid_form()).status_code == 403
    assert client.post("/stop_training/").status_code == 403
    assert client.get("/stop_training/").status_code == 405


def test_job_reservation_prevents_concurrent_start_and_get_does_not_erase_state(tmp_path):
    runner = DemoRunner()
    with patch("myapp.tasks.threading.Thread"):
        assert runner.start(MarketConfig(), SACConfig(), 0, "sac", tmp_path)
        assert not runner.start(MarketConfig(), SACConfig(), 0, "sac", tmp_path)
    runner.cancel()
    assert runner.stop_event.is_set()
    assert runner.snapshot()["running"]


@pytest.mark.parametrize("method", ["fixed", "random", "adaptive", "sac"])
def test_demo_worker_completes_and_exposes_only_known_artifacts(tmp_path, method):
    # Run worker synchronously so failures are visible to the test.
    runner = DemoRunner()
    market = MarketConfig(num_players=8, max_steps=5, shock_step=3)
    config = SACConfig(episodes=2, hidden=16, batch_size=4, warmup_steps=4, replay_size=30)
    with patch("myapp.tasks.threading.Thread"):
        runner.start(market, config, 0, method, tmp_path)
    runner._run(market, config, 0, method)
    status = runner.snapshot()
    assert status["state"] == "completed"
    assert not status["running"]
    assert len(status["history"]) == 2
    assert runner.artifact(status["run_id"], "training.csv").is_file()
    assert runner.artifact(status["run_id"], "../../.env") is None
    assert runner.artifact("other-run", "training.csv") is None


def test_short_sac_demo_reports_that_no_learning_occurred(tmp_path):
    runner = DemoRunner()
    market = MarketConfig(num_players=8, max_steps=3, shock_step=None)
    config = SACConfig(episodes=1, hidden=16, batch_size=8, warmup_steps=10, replay_size=30)
    with patch("myapp.tasks.threading.Thread"):
        runner.start(market, config, 0, "sac", tmp_path)
    runner._run(market, config, 0, "sac")
    status = runner.snapshot()
    assert status["state"] == "completed"
    assert "no parameter updates" in status["message"]
    assert status["history"][-1]["updates"] == 0


def test_live_mode_reports_error_without_using_sample_data(client, settings):
    from integrations.minecraft import MinecraftAPIError
    settings.MINECRAFT_MODE = "live"
    with patch("myapp.views.MinecraftClient.status", side_effect=MinecraftAPIError("Service offline")):
        response = client.get("/server_status/")
        assert b"Service offline" in response.content
        assert b"SAMPLE DATA" not in response.content
