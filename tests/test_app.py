"""Checks grouped by maintained responsibility."""
import ast
import hashlib
from dataclasses import asdict, replace
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from myapp.tasks import DemoRunner
from myapp.views import REWARD_PROFILES
from rl.config import MarketConfig, SACConfig
from unittest.mock import Mock, patch
import requests
from integrations.minecraft import MinecraftAPIError, MinecraftClient, sample_status, validate_status
import csv
from django.test import Client
from myapp.comparison import compare,RECORD
from rl.sac import SACAgent
from experiments.behavior_market import BehaviorMarket
from experiments.behavior_runtime import behavior_for, episode, make_policy


def function_fingerprint(node):
    # The recorded fingerprints use Python 3.10's AST schema. Python 3.12
    # adds an empty type_params field to ordinary function/class definitions.
    for child in ast.walk(node):
        if getattr(child, 'type_params', None) == []:
            child._fields = tuple(field for field in child._fields if field != 'type_params')
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def test_saved_demo_runtime_matches_original_function_fingerprints():
    metadata=json.loads(Path('myapp/comparison-runtime.json').read_text())
    recorded=json.loads(Path(RECORD,'research-source-hashes.json').read_text())
    assert metadata['original_source_sha256']==recorded['experiments/behavior_audit.py']
    source=Path('experiments/behavior_runtime.py').read_text()
    assert hashlib.sha256(source.encode()).hexdigest()==metadata['runtime_sha256']
    functions={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    for name,expected in metadata['original_function_ast_sha256'].items():
        assert function_fingerprint(functions[name])==expected


def test_function_fingerprint_normalizes_only_empty_type_parameters():
    node = ast.parse('def f(x): return x + 1').body[0]
    expected = function_fingerprint(node)
    node._fields = (*node._fields, 'type_params')
    node.type_params = []
    assert function_fingerprint(node) == expected
    node._fields = (*node._fields, 'type_params')
    node.type_params = [ast.Name(id='T', ctx=ast.Load())]
    assert function_fingerprint(node) != expected
    changed = ast.parse('def f(x): return x + 2').body[0]
    assert function_fingerprint(changed) != expected


def valid_form():
    return {"method": "fixed", "reward_mode": "balanced_v1", "episodes": "2", "max_steps": "4", "num_players": "10", "seed": "1",
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


def test_default_form_submits_the_published_balanced_training_settings(client):
    response = client.get("/")
    defaults = response.context["defaults"]
    assert defaults["reward_mode"] == "balanced_v1"
    assert b"Final gold deviation" in response.content
    assert b"Legacy trade-target reward" in response.content
    form = {key: str(value) for key, value in defaults.items()}
    products = response.context["products"]
    form.update(product_count=str(len(products)), association_rules="[]")
    for index, product in enumerate(products):
        form.update({f"{key}_{index}": str(value) for key, value in product.items()})
    protocol = json.loads(Path("experiments/configs/balanced-control.json").read_text())
    with patch("myapp.views.runner.start", return_value=True) as start:
        assert client.post("/", form).status_code == 302
    market, sac, seed, method, _ = start.call_args.args
    assert market.to_dict() == MarketConfig.from_dict(protocol["market"]).to_dict()
    assert asdict(sac) == asdict(SACConfig(**protocol["sac"]))
    assert seed == 0 and method == "sac"


@pytest.mark.parametrize("reward_mode", ["balanced_v1", "legacy"])
def test_objective_selection_survives_submission_and_reload(client, reward_mode):
    form = valid_form()
    form.update(method="sac", reward_mode=reward_mode)
    with patch("myapp.views.runner.start", return_value=True) as start:
        assert client.post("/", form).status_code == 302
    market, sac, seed, method, _ = start.call_args.args
    assert market.reward_mode == reward_mode
    assert {key: getattr(market, key) for key in REWARD_PROFILES[reward_mode]} == REWARD_PROFILES[reward_mode]
    assert sac.control_fee is False
    previous = {"method": method, "configuration": {"market": market.to_dict(), "sac": asdict(sac), "seed": seed}}
    with patch("myapp.views.runner.snapshot", return_value=previous):
        response = client.get("/")
    assert response.context["defaults"]["reward_mode"] == reward_mode


def test_unknown_objective_cannot_start_a_job(client):
    form = valid_form()
    form["reward_mode"] = "unsupported"
    with patch("myapp.views.runner.start") as start:
        assert client.post("/", form).status_code == 400
    start.assert_not_called()


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
    form.update({"method": "adaptive", "reward_mode": "legacy", "episodes": "7", "max_steps": "9", "num_players": "11",
                 "seed": "42", "shock_step": "", "name_0": "Custom product"})
    form[invalid_field] = invalid_value
    with patch("myapp.views.runner.start") as start:
        response = client.post("/", form)
    assert response.status_code == 400
    start.assert_not_called()
    assert response.context["defaults"] == {key: form[key] for key in
        ("method", "reward_mode", "episodes", "max_steps", "num_players", "seed", "shock_step")}
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
@pytest.mark.parametrize("reward_mode", ["balanced_v1", "legacy"])
def test_demo_worker_completes_and_exposes_only_known_artifacts(tmp_path, method, reward_mode):
    # Run worker synchronously so failures are visible to the test.
    runner = DemoRunner()
    market = MarketConfig(num_players=8, max_steps=5, shock_step=3,
                          reward_mode=reward_mode, **REWARD_PROFILES[reward_mode])
    config = SACConfig(episodes=2, hidden=16, batch_size=4, warmup_steps=4, replay_size=30, control_fee=False)
    with patch("myapp.tasks.threading.Thread"):
        runner.start(market, config, 0, method, tmp_path)
    runner._run(market, config, 0, method)
    status = runner.snapshot()
    assert status["state"] == "completed"
    assert not status["running"]
    assert len(status["history"]) == 2
    assert status["configuration"]["market"]["reward_mode"] == reward_mode
    assert status["latest"]["fee_rate"] == pytest.approx(0.05, abs=1e-8)
    saved = json.loads(runner.artifact(status["run_id"], "config.json").read_text())
    assert saved["market"] == market.to_dict()
    assert saved["sac"]["control_fee"] is False
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


@pytest.mark.parametrize('scenario,suite,target,record_scenario',[('original','behavior','original','nominal'),('inventory','behavior','inventory','nominal'),('persistent','stress','original','long_1000')])
def test_saved_comparison_matches_frozen_evaluator_without_updates(monkeypatch,scenario,suite,target,record_scenario):
    def forbidden(*args,**kwargs):raise AssertionError('Demo must not train')
    monkeypatch.setattr(SACAgent,'update',forbidden)
    data=compare(scenario,43000010,0)
    root = Path(RECORD)
    protocol = json.loads((root/'protocol.json').read_text())
    frozen = json.loads((root/'frozen-selection.json').read_text())
    models = json.loads((root/'models.json').read_text())
    model = next(row for row in models if row['family'] == target and row['training_seed'] == 0)
    market = MarketConfig.from_dict(protocol['market'])
    if scenario == 'persistent':
        market = replace(market, max_steps=1000)
    specs = [dict(kind='sac', method=f'sac_{target}', path=str(root/model['path'])),
             {**frozen['choices'][target]['feedback'], 'method':'matched_feedback'},
             {**frozen['choices'][target]['pi'], 'method':'matched_pi'}]
    with (root/'report/summary.csv').open() as f:
        saved_averages = list(csv.DictReader(f))
    for controller,spec in zip(data['controllers'],specs):
        # Compare the adapter to the hash/fingerprint-pinned original evaluator
        # on this platform. CPU kernels can change tiny inference values and
        # consequently discrete trades; historical rollouts are not bitwise portable.
        env = BehaviorMarket(market, behavior_for(protocol, target))
        expected, _, _ = episode(env, make_policy(env, spec), 43000010,
                                 protocol['low_gold_fraction'])
        assert controller['summary'] == pytest.approx(expected, rel=1e-10, abs=1e-10)
        saved = next(row for row in saved_averages if row['suite'] == suite
                     and row['target'] == target and row['scenario'] == record_scenario
                     and row['method'] == spec['method'])
        # Displayed historical study averages remain exactly the saved data.
        assert controller['batch'] == {key:float(saved[key]) for key in controller['batch']}
        assert len(controller['steps'])==data['steps']
        assert all(f'buy_{i}' in controller['steps'][0] and f'sell_{i}' in controller['steps'][0] for i in range(3))


def test_comparison_routes_and_allowlisted_inputs():
    c=Client()
    assert c.get('/comparison/').status_code==200
    assert b'Saved models' in c.get('/').content
    for params in ({'scenario':'../../etc/passwd'},{'market':'1'},{'model':'999'},{'market':'nan'}):
        assert c.get('/comparison/api/',params).status_code==400
    assert c.post('/comparison/api/').status_code==405
    response=c.get('/comparison/api/')
    assert response.status_code==200 and len(response.json()['controllers'])==3
    assert response['Cache-Control']=='no-store'


def test_comparison_and_training_share_execution_slot(monkeypatch,tmp_path):
    from unittest.mock import patch
    from myapp import comparison
    from myapp.tasks import DemoRunner
    from rl.config import MarketConfig,SACConfig
    runner=DemoRunner();monkeypatch.setattr(comparison,'runner',runner)
    market=MarketConfig(num_players=4,max_steps=2,shock_step=None)
    config=SACConfig(episodes=1)
    with runner.compute_lock:
        assert not runner.start(market,config,0,'fixed',tmp_path)
    with patch('myapp.tasks.threading.Thread'):
        assert runner.start(market,config,0,'fixed',tmp_path)
    with pytest.raises(RuntimeError,match='active'):comparison.compare('original',43000010,0)
    runner._run(market,config,0,'fixed')
    assert not runner.compute_lock.locked()
    assert runner.snapshot()['state']=='completed'


def test_failed_worker_start_releases_comparison_slot(tmp_path):
    from unittest.mock import patch
    from myapp.tasks import DemoRunner
    from rl.config import MarketConfig,SACConfig
    runner=DemoRunner()
    with patch('myapp.tasks.threading.Thread.start',side_effect=RuntimeError('start failed')):
        with pytest.raises(RuntimeError):runner.start(MarketConfig(),SACConfig(),0,'sac',tmp_path)
    assert not runner.compute_lock.locked()
    assert not runner.snapshot()['running']
