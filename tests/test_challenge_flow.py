from __future__ import annotations

import io
from typing import Any

import pytest

from babelbit.benchmarks import challenge_flow as module
from babelbit.benchmarks.challenge_flow import ARENA_STAGE, QUALIFYING_STAGE


def test_arena_stage_available_skips_not_started() -> None:
    assert module.arena_stage_available({"arena_status": "Not Started"}) is False
    assert module.arena_stage_available({"arena_status": "Completed"}) is True
    assert module.arena_stage_available({"arena_status": "Running"}) is True


def test_compare_benchmark_to_winner_reports_delta() -> None:
    comparison = module.compare_benchmark_to_winner(
        accuracy_summary={
            "mean_accuracy": 0.896,
            "mean_production_score": 0.93,
            "production_pass_rate": 1.0,
        },
        winner_reference={"miner_hotkey": "hk-winner", "mean_accuracy": 0.7767},
    )
    assert comparison["winner_available"] is True
    assert comparison["beat_winner_on_accuracy"] is True
    assert comparison["accuracy_delta"] == pytest.approx(0.1193, abs=1e-4)


def test_compare_benchmark_to_winner_without_reference() -> None:
    comparison = module.compare_benchmark_to_winner(
        accuracy_summary={"mean_accuracy": 0.5, "mean_production_score": 0.0},
        winner_reference=None,
    )
    assert comparison["winner_available"] is False
    assert comparison["our_mean_accuracy"] == 0.5


def test_extract_utterance_metrics_reads_validator_score() -> None:
    metrics = module.extract_utterance_metrics(
        {
            "utterance_id": "3",
            "wall_latency_sec": 12.5,
            "source_duration_sec": 4.2,
            "first_output_frame": 8,
            "out_eos": True,
            "validator_score": {
                "accuracy": 0.91,
                "accuracy_pass": True,
                "score": 0.95,
                "latency": {
                    "score": 0.95,
                    "overshoot_sec": 0.1,
                    "completion_sec": 4.3,
                },
                "speech_rate": {"penalty": 0.0},
                "stt_text": "hello world",
            },
        }
    )
    assert metrics["scored"] is True
    assert metrics["accuracy"] == 0.91
    assert metrics["production_score"] == 0.95
    assert metrics["overshoot_sec"] == 0.1


def test_format_utterance_log_line_is_compact() -> None:
    line = module.format_utterance_log_line(
        {
            "utterance_id": "0",
            "scored": True,
            "accuracy": 0.896,
            "production_score": 0.93,
            "wall_latency_sec": 11.2,
            "overshoot_sec": 0.0,
        }
    )
    assert "u0" in line
    assert "acc=0.896" in line
    assert "PASS" in line


def test_build_stage_result_includes_comparison() -> None:
    stage_result = module.build_stage_result(
        stage=QUALIFYING_STAGE,
        status="completed",
        data_source="cache",
        winner_reference={"mean_accuracy": 0.77},
        benchmark_result={
            "accuracy_summary": {
                "mean_accuracy": 0.88,
                "mean_production_score": 0.9,
                "production_pass_rate": 1.0,
                "scored_utterance_count": 2,
            },
            "utterances": [],
        },
    )
    assert stage_result["comparison"]["beat_winner_on_accuracy"] is True


def test_summarize_flow_result_detects_beat_all() -> None:
    overall = module.summarize_flow_result(
        {
            "qualifying": {
                "status": "completed",
                "comparison": {"winner_available": True, "beat_winner_on_accuracy": True},
            },
            "arena": {
                "status": "completed",
                "comparison": {"winner_available": True, "beat_winner_on_accuracy": True},
            },
        }
    )
    assert overall["beat_winner_all_completed_stages"] is True


def test_fetch_winner_reference_merges_summary_and_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module,
        "fetch_dialogue_summary",
        lambda *_args, **_kwargs: [
            {
                "miner_hotkey": "hk-1",
                "mean_accuracy": 0.8,
                "mean_u_best": 0.8,
                "utterance_count": 10,
            }
        ],
    )
    monkeypatch.setattr(
        module,
        "fetch_submission_leader",
        lambda *_args, **_kwargs: {
            "miner_hotkey": "hk-1",
            "score_overall": 0.8,
            "is_winner": 1,
        },
    )
    winner = module.fetch_winner_reference("challenge-1", stage=ARENA_STAGE)
    assert winner is not None
    assert winner["miner_hotkey"] == "hk-1"
    assert winner["mean_accuracy"] == 0.8
    assert winner["is_winner"] is True


class _LoggerStream(io.StringIO):
    pass


def test_benchmark_flow_logger_writes_log_file(tmp_path: Any) -> None:
    log_path = tmp_path / "flow.log"
    logger = module.BenchmarkFlowLogger(log_file=log_path)
    logger.header(
        challenge_uid="challenge-1",
        challenge_detail={"epoch": 1, "qualifying_status": "Completed"},
        base_url="http://127.0.0.1:8000",
    )
    written = logger.write_log_file()
    assert written == log_path
    assert "BABELBIT CHALLENGE FLOW BENCHMARK" in log_path.read_text(encoding="utf-8")
