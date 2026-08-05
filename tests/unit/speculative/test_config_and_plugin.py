import textwrap

from py_inference_scheduler.speculative import vllm_plugin
from py_inference_scheduler.speculative.config import (
    ENABLE_ENV_VAR,
    DASConfig,
    das_enabled,
    load_das_config,
)
from py_inference_scheduler.speculative.drafter_state import (
    DASDrafterState,
    get_active_state,
    register_active_state,
)


def test_das_disabled_by_default(monkeypatch):
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    assert not das_enabled()
    monkeypatch.setenv(ENABLE_ENV_VAR, "0")
    assert not das_enabled()
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    assert das_enabled()


def test_defaults_without_config_path(monkeypatch):
    monkeypatch.delenv("DAS_CONFIG_PATH", raising=False)
    cfg = load_das_config()
    assert cfg == DASConfig()


def test_load_from_yaml(tmp_path):
    path = tmp_path / "das.yaml"
    path.write_text(
        textwrap.dedent(
            """
            das:
              window_iterations: 32
              fresh_iterations: 4
              budgets: {long: 16, medium: 4, short: 0}
              classifier: {short_max_tokens: 128, long_min_tokens: 2048}
              service: {max_total_tokens: 1000, max_problems: 10, max_seqs_per_problem: 5}
              push: {batch_size: 8, flush_interval_s: 0.1, queue_capacity: 100}
              poll_interval_s: 1.5
            """
        )
    )
    cfg = load_das_config(str(path))
    assert cfg.window_iterations == 32
    assert cfg.fresh_iterations == 4
    assert cfg.budgets.long == 16
    assert cfg.classifier.long_min_tokens == 2048
    assert cfg.limits.max_seqs_per_problem == 5
    assert cfg.push.batch_size == 8
    assert cfg.poll_interval_s == 1.5


def test_bad_config_falls_back_to_defaults(tmp_path):
    path = tmp_path / "das.yaml"
    path.write_text("das: [not, a, mapping]")
    assert load_das_config(str(path)) == DASConfig()
    assert load_das_config("/nonexistent/das.yaml") == DASConfig()


def test_plugin_register_noop_when_disabled(monkeypatch):
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    vllm_plugin.register()  # must not raise, must not import vllm


def test_plugin_register_survives_missing_vllm(monkeypatch):
    # vllm is not installed in the dev env: register() must swallow the
    # ImportError rather than break engine (or test-process) startup.
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    vllm_plugin.register()


def test_active_state_registry():
    state = DASDrafterState()
    register_active_state(state)
    assert get_active_state() is state
    register_active_state(None)
    assert get_active_state() is None
