from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_benchmark_module() -> Any:
    module_path = Path(__file__).parent / "benchmarks" / "benchmark_miner_challenge.py"
    spec = importlib.util.spec_from_file_location("benchmark_miner_challenge_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_result(utterance_id: str) -> dict[str, Any]:
    return {
        "utterance_id": utterance_id,
        "validator_score": {
            "accuracy": 0.9,
            "accuracy_pass": True,
            "score": 0.8,
            "speech_rate": {"penalty": 1.0},
            "stt_text": "ok",
            "gt_text": "ok",
        },
    }


def test_validator_random_utterances_runs_one_request_per_utterance(monkeypatch, tmp_path) -> None:
    module = _load_benchmark_module()
    calls: list[dict[str, Any]] = []

    def fake_run_benchmark(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return _fake_result(str(kwargs["utterance_id"]))

    monkeypatch.setattr(module._BENCHMARK, "run_benchmark", fake_run_benchmark)

    result = module.run_challenge_benchmark(
        base_url="http://miner",
        sample_path=tmp_path / "sample.json",
        out_dir=tmp_path / "out",
        utterance_ids=["0", "1", "2", "3", "4"],
        skip_validator_score=False,
        accuracy_only=True,
        validator_mode=True,
        concurrent_validators=3,
        validator_utterance_mode="random",
        validator_random_seed=7,
    )

    assert len(calls) == 5
    assert {call["utterance_id"] for call in calls} == {"0", "1", "2", "3", "4"}
    assert all(call["validator_mode"] is True for call in calls)
    assert all(call["concurrent_validators"] == 1 for call in calls)
    assert result["validator_utterance_mode"] == "random"
    assert result["validator_random_seed"] == 7
    assert result["concurrent_validators"] == 3
    assert result["utterance_count"] == 5


def test_validator_same_utterance_mode_preserves_existing_duplicate_behavior(
    monkeypatch, tmp_path
) -> None:
    module = _load_benchmark_module()
    calls: list[dict[str, Any]] = []

    def fake_run_benchmark(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return _fake_result(str(kwargs["utterance_id"]))

    monkeypatch.setattr(module._BENCHMARK, "run_benchmark", fake_run_benchmark)

    result = module.run_challenge_benchmark(
        base_url="http://miner",
        sample_path=tmp_path / "sample.json",
        out_dir=tmp_path / "out",
        utterance_ids=["0", "1"],
        skip_validator_score=False,
        accuracy_only=True,
        validator_mode=True,
        concurrent_validators=3,
        validator_utterance_mode="same",
    )

    assert [call["utterance_id"] for call in calls] == ["0", "1"]
    assert [call["concurrent_validators"] for call in calls] == [3, 3]
    assert result["validator_utterance_mode"] == "same"
