#!/usr/bin/env python3
"""Benchmark a running miner against the latest production challenge from Babelbit APIs."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from babelbit.benchmarks.api_challenge import (
    ApiChallengeError,
    cache_dir_for_challenge,
    dialogue_scores_fetch_limit,
    download_challenge_from_engine,
    fetch_challenge_detail,
    fetch_dialogue_scores,
    fetch_dialogue_summary,
    fetch_top_miner_hotkey,
    get_challenge_uid,
    get_latest_challenge,
    prepare_challenge_fixtures,
    prepare_fixtures_from_dialogue_scores,
    resolve_cached_challenge_path,
)
from babelbit.benchmarks.challenge_flow import (
    QUALIFYING_STAGE,
    BenchmarkFlowLogger,
    arena_stage_available,
    benchmark_out_dir,
    compare_benchmark_to_winner,
    extract_utterance_metrics,
    fetch_winner_reference,
    flow_log_path,
)
from babelbit.benchmarks.miner_test_data import (
    api_challenge_benchmark_dir,
    list_challenge_utterance_ids,
    workspace_root_from,
)

_BENCHMARK_MODULE_PATH = Path(__file__).resolve().with_name("benchmark_miner_challenge.py")
_SPEC = importlib.util.spec_from_file_location("benchmark_miner_challenge", _BENCHMARK_MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_CHALLENGE_BENCHMARK = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CHALLENGE_BENCHMARK
_SPEC.loader.exec_module(_CHALLENGE_BENCHMARK)
_SAMPLE_BENCHMARK = _CHALLENGE_BENCHMARK._BENCHMARK

_FLOW_MODULE_PATH = Path(__file__).resolve().with_name("benchmark_challenge_flow.py")
_FLOW_SPEC = importlib.util.spec_from_file_location("benchmark_challenge_flow", _FLOW_MODULE_PATH)
assert _FLOW_SPEC is not None and _FLOW_SPEC.loader is not None
_FLOW_BENCHMARK = importlib.util.module_from_spec(_FLOW_SPEC)
sys.modules[_FLOW_SPEC.name] = _FLOW_BENCHMARK
_FLOW_SPEC.loader.exec_module(_FLOW_BENCHMARK)


def subnet_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_miner_env_path() -> Path:
    return workspace_root_from(subnet_root()) / "babelbit_miner/.env"


def _print_challenge_context(
    *,
    challenge_uid: str,
    challenge_detail: dict[str, Any],
    dialogue_summary: list[dict[str, Any]],
    data_source: str,
) -> None:
    print(
        json.dumps(
            {
                "challenge_uid": challenge_uid,
                "epoch": challenge_detail.get("epoch"),
                "qualifying_status": challenge_detail.get("qualifying_status"),
                "arena_status": challenge_detail.get("arena_status"),
                "miner_count": challenge_detail.get("miner_count"),
                "data_source": data_source,
                "top_main_miner": dialogue_summary[0] if dialogue_summary else None,
            },
            indent=2,
        )
    )


async def _resolve_sample_path_from_engine(
    *,
    resolved_uid: str,
    utterance_engine_url: str,
    wallet_cold: str,
    wallet_hot: str,
    max_utterances: int | None,
    cache_dir: Path,
) -> Path:
    fetched_uid, start_data, payloads, transcription_metadata = await download_challenge_from_engine(
        utterance_engine_url=utterance_engine_url,
        wallet_cold=wallet_cold,
        wallet_hot=wallet_hot,
        max_utterances=max_utterances,
    )
    if fetched_uid != resolved_uid:
        print(
            json.dumps(
                {
                    "warning": "utterance_engine_challenge_mismatch",
                    "results_api_uid": resolved_uid,
                    "utterance_engine_uid": fetched_uid,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        resolved_uid = fetched_uid
        cache_dir = cache_dir.parent / resolved_uid
    return prepare_challenge_fixtures(
        challenge_uid=resolved_uid,
        start_data=start_data,
        payloads=payloads,
        transcription_metadata=transcription_metadata,
        out_dir=cache_dir,
    ), resolved_uid


def _resolve_sample_path_from_dialogue_scores(
    *,
    resolved_uid: str,
    cache_dir: Path,
    miner_hotkey: str | None,
    max_utterances: int | None,
    source_language: str,
    target_language: str,
) -> Path:
    hotkey = miner_hotkey or fetch_top_miner_hotkey(resolved_uid)
    items = fetch_dialogue_scores(
        resolved_uid,
        stage="main",
        limit=dialogue_scores_fetch_limit(max_utterances),
        miner_hotkey=hotkey,
    )
    return prepare_fixtures_from_dialogue_scores(
        challenge_uid=resolved_uid,
        items=items,
        out_dir=cache_dir,
        source_language=source_language,
        target_language=target_language,
        max_utterances=max_utterances,
    )


async def _resolve_sample_path(
    *,
    challenge_uid: str | None,
    status: str,
    data_source: str,
    utterance_engine_url: str,
    wallet_cold: str,
    wallet_hot: str,
    miner_hotkey: str | None,
    max_utterances: int | None,
    use_cache: bool,
    workspace_root: Path,
    source_language: str,
    target_language: str,
) -> tuple[str, Path, str]:
    if challenge_uid:
        challenge_detail = fetch_challenge_detail(challenge_uid)
        resolved_uid = get_challenge_uid(challenge_detail)
    else:
        challenge = get_latest_challenge(status=status)
        resolved_uid = get_challenge_uid(challenge)
        challenge_detail = fetch_challenge_detail(resolved_uid)

    cache_dir = cache_dir_for_challenge(workspace_root, challenge_uid=resolved_uid)
    if use_cache:
        cached = resolve_cached_challenge_path(
            workspace_root,
            challenge_uid=resolved_uid,
            stage_name="qualifying",
        )
        if cached is not None:
            derived_from = json.loads(cached.read_text(encoding="utf-8")).get("derived_from", "cache")
            return resolved_uid, cached, str(derived_from)

    if data_source in {"utterance-engine", "auto"}:
        try:
            sample_path, resolved_uid = await _resolve_sample_path_from_engine(
                resolved_uid=resolved_uid,
                utterance_engine_url=utterance_engine_url,
                wallet_cold=wallet_cold,
                wallet_hot=wallet_hot,
                max_utterances=max_utterances,
                cache_dir=cache_dir,
            )
            selected_source = "api.babelbit.ai/source-audio"
            _print_challenge_context(
                challenge_uid=resolved_uid,
                challenge_detail=challenge_detail,
                dialogue_summary=fetch_dialogue_summary(resolved_uid, limit=1),
                data_source=selected_source,
            )
            return resolved_uid, sample_path, selected_source
        except Exception as exc:
            from babelbit.utils.utterance_auth import UtteranceAuthError

            if data_source == "utterance-engine" or not isinstance(exc, UtteranceAuthError):
                raise
            print(
                "warning: utterance engine unavailable; falling back to dialogue-scores",
                file=sys.stderr,
            )

    sample_path = _resolve_sample_path_from_dialogue_scores(
        resolved_uid=resolved_uid,
        cache_dir=cache_dir,
        miner_hotkey=miner_hotkey,
        max_utterances=max_utterances,
        source_language=source_language,
        target_language=target_language,
    )
    selected_source = "results.babelbit.ai/dialogue-scores+deepinfra-tts"
    _print_challenge_context(
        challenge_uid=resolved_uid,
        challenge_detail=challenge_detail,
        dialogue_summary=fetch_dialogue_summary(resolved_uid, limit=1),
        data_source=selected_source,
    )
    return resolved_uid, sample_path, selected_source


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark miner against latest production challenge fetched from Babelbit APIs."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--challenge-uid", default=None)
    parser.add_argument("--status", choices=["completed", "live"], default="completed")
    parser.add_argument(
        "--data-source",
        choices=["auto", "utterance-engine", "dialogue-scores"],
        default="auto",
    )
    parser.add_argument(
        "--utterance-engine-url",
        default=os.getenv("BB_UTTERANCE_ENGINE_URL", "https://api.babelbit.ai"),
    )
    parser.add_argument("--wallet-cold", default=os.getenv("BITTENSOR_WALLET_COLD", "default"))
    parser.add_argument("--wallet-hot", default=os.getenv("BITTENSOR_WALLET_HOT", "default"))
    parser.add_argument("--miner-hotkey", default=None)
    parser.add_argument("--source-language", default=os.getenv("DEEPINFRA_SOURCE_LANGUAGE", "fr"))
    parser.add_argument("--target-language", default=os.getenv("DEEPINFRA_TARGET_LANGUAGE", "en"))
    parser.add_argument("--miner-env", type=Path, default=None)
    parser.add_argument("--max-utterances", type=int, default=None, help="Limit utterances (default: all).")
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Refetch challenge fixtures instead of using miner-test-data cache.",
    )
    parser.add_argument("--accuracy-only", action="store_true", default=True)
    parser.add_argument(
        "--full-flow",
        action="store_true",
        help="Run qualifying then arena with winner comparison and detailed logs.",
    )
    parser.add_argument("--skip-arena", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()

    workspace_root = workspace_root_from(subnet_root())
    miner_env = args.miner_env or default_miner_env_path()
    _SAMPLE_BENCHMARK._load_env(miner_env)

    if args.full_flow:
        log_file = args.log_file
        if log_file is None and args.challenge_uid:
            log_file = flow_log_path(workspace_root, challenge_uid=args.challenge_uid)
        logger = BenchmarkFlowLogger(log_file=log_file)
        try:
            flow_result = asyncio.run(
                _FLOW_BENCHMARK.run_challenge_flow(
                    base_url=args.base_url,
                    challenge_uid=args.challenge_uid,
                    status=args.status,
                    data_source=args.data_source,
                    utterance_engine_url=args.utterance_engine_url.rstrip("/"),
                    wallet_cold=args.wallet_cold,
                    wallet_hot=args.wallet_hot,
                    miner_hotkey=args.miner_hotkey,
                    max_utterances=args.max_utterances,
                    min_words=args.min_words,
                    use_cache=not args.no_cache,
                    workspace_root=workspace_root,
                    source_language=args.source_language,
                    target_language=args.target_language,
                    logger=logger,
                    skip_arena=args.skip_arena,
                )
            )
        except ApiChallengeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        logger.write_log_file()
        print(json.dumps(flow_result["overall"], indent=2))
        return 0

    try:
        challenge_uid, sample_path, _data_source = asyncio.run(
            _resolve_sample_path(
                challenge_uid=args.challenge_uid,
                status=args.status,
                data_source=args.data_source,
                utterance_engine_url=args.utterance_engine_url.rstrip("/"),
                wallet_cold=args.wallet_cold,
                wallet_hot=args.wallet_hot,
                miner_hotkey=args.miner_hotkey,
                max_utterances=args.max_utterances,
                use_cache=not args.no_cache,
                workspace_root=workspace_root,
                source_language=args.source_language,
                target_language=args.target_language,
            )
        )
    except ApiChallengeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    utterance_ids = list_challenge_utterance_ids(
        sample_path,
        max_utterances=args.max_utterances,
        min_words=args.min_words,
    )
    if not utterance_ids:
        print(f"error: no scorable utterances in {sample_path}", file=sys.stderr)
        return 1

    legacy_benchmark_dir = api_challenge_benchmark_dir(workspace_root, challenge_uid=challenge_uid)
    out_dir = benchmark_out_dir(workspace_root, challenge_uid=challenge_uid, stage_name="qualifying")
    if legacy_benchmark_dir.is_dir() and not out_dir.is_dir():
        legacy_u_dirs = [path for path in legacy_benchmark_dir.glob("u*") if path.is_dir()]
        if legacy_u_dirs and not any(out_dir.glob("u*")):
            out_dir = legacy_benchmark_dir

    challenge_detail = fetch_challenge_detail(challenge_uid)
    winner_reference = fetch_winner_reference(challenge_uid, stage=QUALIFYING_STAGE)
    logger = BenchmarkFlowLogger(log_file=args.log_file)
    logger.header(
        challenge_uid=challenge_uid,
        challenge_detail=challenge_detail,
        base_url=args.base_url,
    )
    logger.stage_start(
        stage=QUALIFYING_STAGE,
        stage_index=1,
        stage_count=1,
        winner_reference=winner_reference,
        utterance_count=len(utterance_ids),
        data_source=_data_source,
    )

    def _on_utterance(
        _stage_name: str,
        _utterance_id: str,
        utterance_result: dict[str, Any],
        _index: int,
        _total: int,
    ) -> None:
        logger.utterance(extract_utterance_metrics(utterance_result))

    result = _CHALLENGE_BENCHMARK.run_challenge_benchmark(
        base_url=args.base_url,
        sample_path=sample_path,
        out_dir=out_dir,
        utterance_ids=utterance_ids,
        skip_validator_score=False,
        accuracy_only=args.accuracy_only,
        on_utterance_complete=_on_utterance,
        stage_name="qualifying",
    )
    comparison = compare_benchmark_to_winner(
        accuracy_summary=result["accuracy_summary"],
        winner_reference=winner_reference,
    )
    logger.stage_summary(
        stage=QUALIFYING_STAGE,
        accuracy_summary=result["accuracy_summary"],
        comparison=comparison,
    )
    logger.write_log_file()
    print(json.dumps({"accuracy_summary": result["accuracy_summary"], "comparison": comparison}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
