"""Qualifying → arena benchmark flow with winner comparison and readable logging."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

from babelbit.benchmarks.api_challenge import (
    ApiChallengeError,
    dialogue_scores_fetch_limit,
    fetch_challenge_detail,
    fetch_dialogue_scores,
    fetch_dialogue_summary,
    fetch_submission_leader,
    fetch_top_miner_hotkey,
    legacy_cache_dir_for_challenge,
    load_cached_challenge,
    prepare_fixtures_from_dialogue_scores,
)
from babelbit.benchmarks.miner_test_data import (
    api_challenge_benchmark_dir,
    summarize_accuracy_results,
)

UtteranceCallback = Callable[[str, str, dict[str, Any], int, int], None]


@dataclass(frozen=True)
class StageFlow:
    name: str
    dialogue_stage: str
    submissions_stage: str
    label: str


QUALIFYING_STAGE = StageFlow(
    name="qualifying",
    dialogue_stage="main",
    submissions_stage="qualifying",
    label="Qualifying",
)
ARENA_STAGE = StageFlow(
    name="arena",
    dialogue_stage="arena",
    submissions_stage="arena",
    label="Arena",
)

FLOW_STAGES: tuple[StageFlow, ...] = (QUALIFYING_STAGE, ARENA_STAGE)

ARENA_SKIP_STATUSES = frozenset({"not started", "pending", ""})


def stage_cache_dir(challenge_cache_dir: Path, *, stage_name: str) -> Path:
    return challenge_cache_dir / "stages" / stage_name


def benchmark_out_dir(workspace_root: Path, *, challenge_uid: str, stage_name: str) -> Path:
    return api_challenge_benchmark_dir(workspace_root, challenge_uid=challenge_uid) / stage_name


def flow_log_path(workspace_root: Path, *, challenge_uid: str) -> Path:
    return api_challenge_benchmark_dir(workspace_root, challenge_uid=challenge_uid) / "flow.log"


def flow_result_path(workspace_root: Path, *, challenge_uid: str) -> Path:
    return api_challenge_benchmark_dir(workspace_root, challenge_uid=challenge_uid) / "flow_result.json"


def arena_stage_available(challenge_detail: dict[str, Any]) -> bool:
    status = str(challenge_detail.get("arena_status") or "").strip().lower()
    return status not in ARENA_SKIP_STATUSES


def _load_cached_stage_fixture(
    *,
    challenge_cache_dir: Path,
    legacy_challenge_cache_dir: Path,
    stage_name: str,
) -> Path | None:
    cached = load_cached_challenge(stage_cache_dir(challenge_cache_dir, stage_name=stage_name))
    if cached is not None:
        return cached
    cached = load_cached_challenge(stage_cache_dir(legacy_challenge_cache_dir, stage_name=stage_name))
    if cached is not None:
        return cached
    if stage_name != "qualifying":
        return None
    for root in (challenge_cache_dir, legacy_challenge_cache_dir):
        cached = load_cached_challenge(root)
        if cached is not None:
            return cached
    return None


def resolve_stage_sample_path(
    *,
    challenge_uid: str,
    stage: StageFlow,
    challenge_cache_dir: Path,
    legacy_challenge_cache_dir: Path,
    use_cache: bool,
    miner_hotkey: str | None,
    max_utterances: int | None,
    source_language: str,
    target_language: str,
    on_fixture_progress: Any | None = None,
) -> tuple[Path, str]:
    cache_dir = stage_cache_dir(challenge_cache_dir, stage_name=stage.name)
    if use_cache:
        cached = _load_cached_stage_fixture(
            challenge_cache_dir=challenge_cache_dir,
            legacy_challenge_cache_dir=legacy_challenge_cache_dir,
            stage_name=stage.name,
        )
        if cached is not None:
            derived_from = json.loads(cached.read_text(encoding="utf-8")).get("derived_from", "cache")
            return cached, str(derived_from)

    hotkey = miner_hotkey or fetch_top_miner_hotkey(challenge_uid, stage=stage.submissions_stage)
    items = fetch_dialogue_scores(
        challenge_uid,
        stage=stage.dialogue_stage,
        limit=dialogue_scores_fetch_limit(max_utterances),
        miner_hotkey=hotkey,
    )
    if not items:
        raise ApiChallengeError(
            f"No dialogue-scores for {challenge_uid} stage={stage.dialogue_stage!r}"
        )
    sample_path = prepare_fixtures_from_dialogue_scores(
        challenge_uid=challenge_uid,
        items=items,
        out_dir=cache_dir,
        source_language=source_language,
        target_language=target_language,
        max_utterances=max_utterances,
        on_progress=on_fixture_progress,
    )
    derived_from = f"results.babelbit.ai/dialogue-scores({stage.dialogue_stage})+deepinfra-tts"
    return sample_path, derived_from


def fetch_winner_reference(
    challenge_uid: str,
    *,
    stage: StageFlow,
) -> dict[str, Any] | None:
    summary_items = fetch_dialogue_summary(
        challenge_uid,
        stage=stage.dialogue_stage,
        limit=1,
    )
    submission = fetch_submission_leader(challenge_uid, stage=stage.submissions_stage)
    if not summary_items and submission is None:
        return None

    summary = summary_items[0] if summary_items else {}
    hotkey = str(
        summary.get("miner_hotkey")
        or (submission or {}).get("miner_hotkey")
        or ""
    ).strip()
    mean_accuracy = _coerce_float(summary.get("mean_accuracy"))
    if mean_accuracy is None and submission is not None:
        mean_accuracy = _coerce_float(submission.get("score_overall"))
    if mean_accuracy is None:
        return None

    return {
        "stage": stage.name,
        "dialogue_stage": stage.dialogue_stage,
        "submissions_stage": stage.submissions_stage,
        "miner_hotkey": hotkey or None,
        "miner_uid": summary.get("miner_uid") or (submission or {}).get("miner_uid"),
        "mean_accuracy": round(mean_accuracy, 6),
        "mean_u_best": _round_or_none(summary.get("mean_u_best")),
        "mean_u_best_early": _round_or_none(summary.get("mean_u_best_early")),
        "utterance_count": summary.get("utterance_count"),
        "dialogue_count": summary.get("dialogue_count"),
        "submission_score_overall": _round_or_none((submission or {}).get("score_overall")),
        "submission_score_main": _round_or_none((submission or {}).get("score_main")),
        "submission_score_arena": _round_or_none((submission or {}).get("score_arena")),
        "is_winner": bool((submission or {}).get("is_winner")),
        "is_arena_winner": bool((submission or {}).get("is_arena_winner")),
        "last_scored_at": summary.get("last_scored_at"),
    }


def compare_benchmark_to_winner(
    *,
    accuracy_summary: dict[str, Any],
    winner_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    our_accuracy = float(accuracy_summary.get("mean_accuracy") or 0.0)
    our_production = float(accuracy_summary.get("mean_production_score") or 0.0)
    our_pass_rate = float(accuracy_summary.get("production_pass_rate") or 0.0)

    if winner_reference is None:
        return {
            "winner_available": False,
            "our_mean_accuracy": round(our_accuracy, 6),
            "our_mean_production_score": round(our_production, 6),
            "our_production_pass_rate": round(our_pass_rate, 6),
        }

    winner_accuracy = float(winner_reference.get("mean_accuracy") or 0.0)
    accuracy_delta = our_accuracy - winner_accuracy
    return {
        "winner_available": True,
        "winner_hotkey": winner_reference.get("miner_hotkey"),
        "winner_mean_accuracy": winner_accuracy,
        "our_mean_accuracy": round(our_accuracy, 6),
        "our_mean_production_score": round(our_production, 6),
        "our_production_pass_rate": round(our_pass_rate, 6),
        "accuracy_delta": round(accuracy_delta, 6),
        "beat_winner_on_accuracy": our_accuracy > winner_accuracy,
        "tie_winner_on_accuracy": abs(accuracy_delta) < 1e-6,
    }


def extract_utterance_metrics(result: dict[str, Any]) -> dict[str, Any]:
    validator_score = result.get("validator_score")
    if not isinstance(validator_score, dict):
        return {
            "utterance_id": result.get("utterance_id"),
            "scored": False,
            "wall_latency_sec": result.get("wall_latency_sec"),
            "out_eos": result.get("out_eos"),
        }

    latency = validator_score.get("latency") if isinstance(validator_score.get("latency"), dict) else {}
    speech_rate = (
        validator_score.get("speech_rate")
        if isinstance(validator_score.get("speech_rate"), dict)
        else {}
    )
    return {
        "utterance_id": result.get("utterance_id"),
        "scored": "error" not in validator_score,
        "accuracy": validator_score.get("accuracy"),
        "accuracy_pass": validator_score.get("accuracy_pass"),
        "production_score": validator_score.get("score"),
        "latency_score": latency.get("score"),
        "overshoot_sec": latency.get("overshoot_sec"),
        "completion_sec": latency.get("completion_sec"),
        "source_duration_sec": result.get("source_duration_sec"),
        "wall_latency_sec": result.get("wall_latency_sec"),
        "first_output_frame": result.get("first_output_frame"),
        "speech_rate_penalty": speech_rate.get("penalty"),
        "out_eos": result.get("out_eos"),
        "stt_text": validator_score.get("stt_text"),
        "error": validator_score.get("error"),
    }


def format_utterance_log_line(metrics: dict[str, Any]) -> str:
    utterance_id = metrics.get("utterance_id", "?")
    if not metrics.get("scored"):
        wall = metrics.get("wall_latency_sec")
        eos = "EOS" if metrics.get("out_eos") else "NO_EOS"
        return f"  u{utterance_id:<4} unscored | wall={wall}s | {eos}"

    accuracy = float(metrics.get("accuracy") or 0.0)
    production = float(metrics.get("production_score") or 0.0)
    overshoot = metrics.get("overshoot_sec")
    wall = metrics.get("wall_latency_sec")
    passed = "PASS" if production > 0.0 else "FAIL"
    return (
        f"  u{utterance_id:<4} acc={accuracy:.3f} prod={production:.3f} "
        f"wall={wall}s overshoot={overshoot}s | {passed}"
    )


class BenchmarkFlowLogger:
    def __init__(self, *, stream: TextIO | None = None, log_file: Path | None = None) -> None:
        self._stream = stream or sys.stdout
        self._log_file = log_file
        self._lines: list[str] = []

    def _emit(self, line: str = "") -> None:
        self._lines.append(line)
        print(line, file=self._stream, flush=True)

    def header(
        self,
        *,
        challenge_uid: str,
        challenge_detail: dict[str, Any],
        base_url: str,
    ) -> None:
        self._emit("=" * 80)
        self._emit("BABELBIT CHALLENGE FLOW BENCHMARK")
        self._emit("=" * 80)
        self._emit(f"Challenge : {challenge_uid} (epoch {challenge_detail.get('epoch')})")
        self._emit(f"Miner URL : {base_url}")
        self._emit(
            "Stages    : "
            f"qualifying={challenge_detail.get('qualifying_status')} | "
            f"arena={challenge_detail.get('arena_status')}"
        )
        winner_hotkey = challenge_detail.get("winner_hotkey")
        if winner_hotkey:
            self._emit(f"Winner HK : {winner_hotkey}")
        self._emit(f"Started   : {datetime.now(timezone.utc).isoformat()}")
        self._emit("")

    def stage_start(
        self,
        *,
        stage: StageFlow,
        stage_index: int,
        stage_count: int,
        winner_reference: dict[str, Any] | None,
        utterance_count: int,
        data_source: str,
    ) -> None:
        self._emit("-" * 80)
        self._emit(
            f"STAGE {stage_index}/{stage_count}: {stage.label.upper()} "
            f"(dialogue={stage.dialogue_stage}, submissions={stage.submissions_stage})"
        )
        self._emit("-" * 80)
        self._emit(f"Data source : {data_source}")
        self._emit(f"Utterances  : {utterance_count}")
        if winner_reference is None:
            self._emit("Winner ref  : unavailable")
        else:
            hotkey = winner_reference.get("miner_hotkey") or "unknown"
            self._emit(
                "Winner ref  : "
                f"{hotkey} | mean_accuracy={winner_reference.get('mean_accuracy')} "
                f"| utterances={winner_reference.get('utterance_count')}"
            )
        self._emit("")

    def fixture_progress(self, message: str) -> None:
        self._emit(f"  [fixtures] {message}")

    def utterance(self, metrics: dict[str, Any]) -> None:
        self._emit(format_utterance_log_line(metrics))

    def stage_summary(
        self,
        *,
        stage: StageFlow,
        accuracy_summary: dict[str, Any],
        comparison: dict[str, Any],
    ) -> None:
        self._emit("")
        self._emit(f"{stage.label} summary:")
        self._emit(
            f"  mean_accuracy       : {accuracy_summary.get('mean_accuracy')} "
            f"(min={accuracy_summary.get('min_accuracy')}, max={accuracy_summary.get('max_accuracy')})"
        )
        self._emit(f"  mean_production     : {accuracy_summary.get('mean_production_score')}")
        self._emit(f"  production_pass_rate: {accuracy_summary.get('production_pass_rate')}")
        self._emit(f"  scored_utterances   : {accuracy_summary.get('scored_utterance_count')}")

        if not comparison.get("winner_available"):
            self._emit("  vs winner           : no winner reference")
            self._emit("")
            return

        accuracy_delta = float(comparison.get("accuracy_delta") or 0.0)
        sign = "+" if accuracy_delta >= 0 else ""
        verdict = "BEAT" if comparison.get("beat_winner_on_accuracy") else "BEHIND"
        if comparison.get("tie_winner_on_accuracy"):
            verdict = "TIE"
        self._emit(
            f"  vs winner accuracy  : {sign}{accuracy_delta:.6f} ({verdict}) "
            f"[winner={comparison.get('winner_mean_accuracy')}]"
        )
        self._emit("")

    def stage_skipped(self, *, stage: StageFlow, reason: str) -> None:
        self._emit("-" * 80)
        self._emit(f"STAGE SKIPPED: {stage.label.upper()} — {reason}")
        self._emit("")

    def final_report(self, flow_result: dict[str, Any]) -> None:
        self._emit("=" * 80)
        self._emit("FINAL REPORT")
        self._emit("=" * 80)
        for stage_name, stage_result in (flow_result.get("stages") or {}).items():
            status = stage_result.get("status")
            self._emit(f"{stage_name:<12}: {status}")
            if status != "completed":
                continue
            comparison = stage_result.get("comparison") or {}
            if comparison.get("winner_available"):
                delta = comparison.get("accuracy_delta")
                verdict = "BEAT" if comparison.get("beat_winner_on_accuracy") else "BEHIND"
                if comparison.get("tie_winner_on_accuracy"):
                    verdict = "TIE"
                self._emit(
                    f"  accuracy {comparison.get('our_mean_accuracy')} "
                    f"vs winner {comparison.get('winner_mean_accuracy')} "
                    f"(delta {delta}, {verdict})"
                )
        self._emit("=" * 80)

    def write_log_file(self) -> Path | None:
        if self._log_file is None:
            return None
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        self._log_file.write_text("\n".join(self._lines) + "\n", encoding="utf-8")
        return self._log_file


def build_stage_result(
    *,
    stage: StageFlow,
    status: str,
    reason: str | None = None,
    data_source: str | None = None,
    winner_reference: dict[str, Any] | None = None,
    benchmark_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stage": stage.name,
        "dialogue_stage": stage.dialogue_stage,
        "submissions_stage": stage.submissions_stage,
        "status": status,
    }
    if reason:
        payload["reason"] = reason
    if data_source:
        payload["data_source"] = data_source
    if winner_reference is not None:
        payload["winner_reference"] = winner_reference
    if benchmark_result is None:
        return payload

    accuracy_summary = benchmark_result.get("accuracy_summary")
    if not isinstance(accuracy_summary, dict):
        accuracy_summary = summarize_accuracy_results(benchmark_result.get("utterances") or [])

    comparison = compare_benchmark_to_winner(
        accuracy_summary=accuracy_summary,
        winner_reference=winner_reference,
    )
    payload["benchmark"] = benchmark_result
    payload["accuracy_summary"] = accuracy_summary
    payload["comparison"] = comparison
    return payload


def summarize_flow_result(stage_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    completed = {
        name: result
        for name, result in stage_results.items()
        if result.get("status") == "completed"
    }
    all_beat = all(
        bool((result.get("comparison") or {}).get("beat_winner_on_accuracy"))
        for result in completed.values()
        if (result.get("comparison") or {}).get("winner_available")
    )
    any_behind = any(
        not bool((result.get("comparison") or {}).get("beat_winner_on_accuracy"))
        and not bool((result.get("comparison") or {}).get("tie_winner_on_accuracy"))
        for result in completed.values()
        if (result.get("comparison") or {}).get("winner_available")
    )
    return {
        "completed_stage_count": len(completed),
        "beat_winner_all_completed_stages": all_beat if completed else False,
        "behind_winner_any_stage": any_behind,
    }


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_or_none(value: Any) -> float | None:
    parsed = _coerce_float(value)
    if parsed is None:
        return None
    return round(parsed, 6)
