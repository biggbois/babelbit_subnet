#!/usr/bin/env python3
"""Benchmark miner through qualifying → arena with winner comparison."""

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
    download_challenge_from_engine,
    fetch_challenge_detail,
    get_challenge_uid,
    get_latest_challenge,
    legacy_cache_dir_for_challenge,
    prepare_challenge_fixtures,
    resolve_cached_challenge_path,
)
from babelbit.benchmarks.challenge_flow import (
    ARENA_STAGE,
    QUALIFYING_STAGE,
    BenchmarkFlowLogger,
    _load_cached_stage_fixture,
    arena_stage_available,
    benchmark_out_dir,
    build_stage_result,
    extract_utterance_metrics,
    fetch_winner_reference,
    flow_log_path,
    flow_result_path,
    resolve_stage_sample_path,
    summarize_flow_result,
)
from babelbit.benchmarks.miner_test_data import list_challenge_utterance_ids, workspace_root_from
from babelbit.benchmarks.s2s_client import S2sClientConfig

_BENCHMARK_MODULE_PATH = Path(__file__).resolve().with_name("benchmark_miner_challenge.py")
_SPEC = importlib.util.spec_from_file_location("benchmark_miner_challenge", _BENCHMARK_MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_CHALLENGE_BENCHMARK = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CHALLENGE_BENCHMARK
_SPEC.loader.exec_module(_CHALLENGE_BENCHMARK)
_SAMPLE_BENCHMARK = _CHALLENGE_BENCHMARK._BENCHMARK


def subnet_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_miner_env_path() -> Path:
    return workspace_root_from(subnet_root()) / "babelbit_miner/.env"


async def _resolve_qualifying_from_engine(
    *,
    challenge_uid: str,
    utterance_engine_url: str,
    wallet_cold: str,
    wallet_hot: str,
    max_utterances: int | None,
    cache_dir: Path,
) -> tuple[Path, str, str]:
    fetched_uid, start_data, payloads, transcription_metadata = await download_challenge_from_engine(
        utterance_engine_url=utterance_engine_url,
        wallet_cold=wallet_cold,
        wallet_hot=wallet_hot,
        max_utterances=max_utterances,
    )
    resolved_uid = challenge_uid
    out_dir = cache_dir / "stages" / "qualifying"
    if fetched_uid != challenge_uid:
        print(
            json.dumps(
                {
                    "warning": "utterance_engine_challenge_mismatch",
                    "results_api_uid": challenge_uid,
                    "utterance_engine_uid": fetched_uid,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        resolved_uid = fetched_uid
        out_dir = cache_dir.parent / resolved_uid / "stages" / "qualifying"
    sample_path = prepare_challenge_fixtures(
        challenge_uid=resolved_uid,
        start_data=start_data,
        payloads=payloads,
        transcription_metadata=transcription_metadata,
        out_dir=out_dir,
    )
    return sample_path, "api.babelbit.ai/source-audio", resolved_uid


def _run_stage_benchmark(
    *,
    base_url: str,
    stage: Any,
    sample_path: Path,
    workspace_root: Path,
    challenge_uid: str,
    utterance_ids: list[str],
    logger: BenchmarkFlowLogger,
    winner_reference: dict[str, Any] | None,
    data_source: str,
    stage_index: int,
    stage_count: int,
    validator_mode: bool = False,
    concurrent_validators: int = 1,
    validator_utterance_mode: str = "same",
    validator_random_seed: int = 59,
    s2s_config: S2sClientConfig | None = None,
) -> dict[str, Any]:
    out_dir = benchmark_out_dir(workspace_root, challenge_uid=challenge_uid, stage_name=stage.name)
    logger.stage_start(
        stage=stage,
        stage_index=stage_index,
        stage_count=stage_count,
        winner_reference=winner_reference,
        utterance_count=len(utterance_ids),
        data_source=data_source,
    )

    def _on_utterance(
        _stage_name: str,
        _utterance_id: str,
        result: dict[str, Any],
        _index: int,
        _total: int,
    ) -> None:
        logger.utterance(extract_utterance_metrics(result))

    benchmark_result = _CHALLENGE_BENCHMARK.run_challenge_benchmark(
        base_url=base_url,
        sample_path=sample_path,
        out_dir=out_dir,
        utterance_ids=utterance_ids,
        skip_validator_score=False,
        accuracy_only=True,
        on_utterance_complete=_on_utterance,
        stage_name=stage.name,
        validator_mode=validator_mode,
        concurrent_validators=concurrent_validators,
        validator_utterance_mode=validator_utterance_mode,
        validator_random_seed=validator_random_seed,
        s2s_config=s2s_config,
    )
    stage_result = build_stage_result(
        stage=stage,
        status="completed",
        data_source=data_source,
        winner_reference=winner_reference,
        benchmark_result=benchmark_result,
    )
    logger.stage_summary(
        stage=stage,
        accuracy_summary=stage_result["accuracy_summary"],
        comparison=stage_result["comparison"],
    )
    return stage_result


async def run_challenge_flow(
    *,
    base_url: str,
    challenge_uid: str | None,
    status: str,
    data_source: str,
    utterance_engine_url: str,
    wallet_cold: str,
    wallet_hot: str,
    miner_hotkey: str | None,
    max_utterances: int | None,
    min_words: int,
    use_cache: bool,
    workspace_root: Path,
    source_language: str,
    target_language: str,
    logger: BenchmarkFlowLogger,
    skip_arena: bool,
    validator_mode: bool = False,
    concurrent_validators: int = 1,
    validator_utterance_mode: str = "same",
    validator_random_seed: int = 59,
    s2s_config: S2sClientConfig | None = None,
) -> dict[str, Any]:
    if challenge_uid:
        challenge_detail = fetch_challenge_detail(challenge_uid)
        resolved_uid = get_challenge_uid(challenge_detail)
    else:
        challenge = get_latest_challenge(status=status)
        resolved_uid = get_challenge_uid(challenge)
        challenge_detail = fetch_challenge_detail(resolved_uid)

    challenge_cache_dir = cache_dir_for_challenge(workspace_root, challenge_uid=resolved_uid)
    legacy_challenge_cache_dir = legacy_cache_dir_for_challenge(
        workspace_root, challenge_uid=resolved_uid
    )
    logger.header(
        challenge_uid=resolved_uid,
        challenge_detail=challenge_detail,
        base_url=base_url,
    )

    stages_to_run: list[Any] = [QUALIFYING_STAGE]
    if not skip_arena and arena_stage_available(challenge_detail):
        stages_to_run.append(ARENA_STAGE)
    elif not skip_arena:
        logger.stage_skipped(stage=ARENA_STAGE, reason=f"arena_status={challenge_detail.get('arena_status')}")

    stage_results: dict[str, dict[str, Any]] = {}
    if not skip_arena and not arena_stage_available(challenge_detail):
        stage_results["arena"] = build_stage_result(
            stage=ARENA_STAGE,
            status="skipped",
            reason=f"arena_status={challenge_detail.get('arena_status')}",
        )

    for stage_index, stage in enumerate(stages_to_run, start=1):
        winner_reference = fetch_winner_reference(resolved_uid, stage=stage)
        data_source_label = ""
        sample_path: Path | None = None

        if stage.name == "qualifying" and data_source in {"utterance-engine", "auto"}:
            cached = None
            if use_cache:
                cached = _load_cached_stage_fixture(
                    challenge_cache_dir=challenge_cache_dir,
                    legacy_challenge_cache_dir=legacy_challenge_cache_dir,
                    stage_name="qualifying",
                )
            if cached is not None:
                sample_path = cached
                data_source_label = str(
                    json.loads(cached.read_text(encoding="utf-8")).get("derived_from", "cache")
                )
            else:
                try:
                    sample_path, data_source_label, resolved_uid = await _resolve_qualifying_from_engine(
                        challenge_uid=resolved_uid,
                        utterance_engine_url=utterance_engine_url,
                        wallet_cold=wallet_cold,
                        wallet_hot=wallet_hot,
                        max_utterances=max_utterances,
                        cache_dir=challenge_cache_dir,
                    )
                    challenge_cache_dir = cache_dir_for_challenge(
                        workspace_root, challenge_uid=resolved_uid
                    )
                    legacy_challenge_cache_dir = legacy_cache_dir_for_challenge(
                        workspace_root, challenge_uid=resolved_uid
                    )
                except Exception as exc:
                    from babelbit.utils.utterance_auth import UtteranceAuthError

                    if data_source == "utterance-engine" or not isinstance(exc, UtteranceAuthError):
                        raise
                    print(
                        "warning: utterance engine unavailable; falling back to dialogue-scores",
                        file=sys.stderr,
                    )

        if sample_path is None:
            try:
                sample_path, data_source_label = resolve_stage_sample_path(
                    challenge_uid=resolved_uid,
                    stage=stage,
                    challenge_cache_dir=challenge_cache_dir,
                    legacy_challenge_cache_dir=legacy_challenge_cache_dir,
                    use_cache=use_cache,
                    miner_hotkey=miner_hotkey,
                    max_utterances=max_utterances,
                    source_language=source_language,
                    target_language=target_language,
                    on_fixture_progress=logger.fixture_progress,
                )
            except ApiChallengeError as exc:
                if stage.name == "arena":
                    logger.stage_skipped(stage=stage, reason=str(exc))
                    stage_results[stage.name] = build_stage_result(
                        stage=stage,
                        status="skipped",
                        reason=str(exc),
                        winner_reference=winner_reference,
                    )
                    continue
                raise

        utterance_ids = list_challenge_utterance_ids(
            sample_path,
            max_utterances=max_utterances,
            min_words=min_words,
        )
        if not utterance_ids:
            stage_results[stage.name] = build_stage_result(
                stage=stage,
                status="skipped",
                reason=f"no scorable utterances in {sample_path}",
                winner_reference=winner_reference,
            )
            logger.stage_skipped(stage=stage, reason="no scorable utterances")
            continue

        stage_results[stage.name] = _run_stage_benchmark(
            base_url=base_url,
            stage=stage,
            sample_path=sample_path,
            workspace_root=workspace_root,
            challenge_uid=resolved_uid,
            utterance_ids=utterance_ids,
            logger=logger,
            winner_reference=winner_reference,
            data_source=data_source_label,
            stage_index=stage_index,
            stage_count=len(stages_to_run),
            validator_mode=validator_mode,
            concurrent_validators=concurrent_validators,
            validator_utterance_mode=validator_utterance_mode,
            validator_random_seed=validator_random_seed,
            s2s_config=s2s_config,
        )

    flow_result = {
        "challenge_uid": resolved_uid,
        "challenge_detail": {
            "epoch": challenge_detail.get("epoch"),
            "qualifying_status": challenge_detail.get("qualifying_status"),
            "arena_status": challenge_detail.get("arena_status"),
            "winner_hotkey": challenge_detail.get("winner_hotkey"),
            "arena_score": challenge_detail.get("arena_score"),
        },
        "stages": stage_results,
        "overall": summarize_flow_result(stage_results),
    }
    result_path = flow_result_path(workspace_root, challenge_uid=resolved_uid)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(flow_result, indent=2) + "\n", encoding="utf-8")
    flow_result["flow_result_path"] = str(result_path)
    logger.final_report(flow_result)
    return flow_result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark miner through qualifying and arena with winner comparison."
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
    parser.add_argument("--skip-arena", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument(
        "--validator-mode",
        action="store_true",
        help="Use validator S2S envelope (3s chunk timeout, drain loop, concurrent validators).",
    )
    parser.add_argument(
        "--concurrent-validators",
        type=int,
        default=1,
        help="Parallel validator sessions per utterance (default 1; production-like=3).",
    )
    parser.add_argument(
        "--validator-random-utterances",
        action="store_true",
        help=(
            "In validator mode, run each concurrency wave with random distinct utterances "
            "instead of duplicating the same utterance across validators."
        ),
    )
    parser.add_argument(
        "--validator-random-seed",
        type=int,
        default=59,
        help="Seed for --validator-random-utterances shuffling.",
    )
    parser.add_argument(
        "--chunk-timeout-sec",
        type=float,
        default=3.0,
        help="Per-frame timeout in validator mode (BB_S2S_CHUNK_TIMEOUT_SEC).",
    )
    args = parser.parse_args()

    workspace_root = workspace_root_from(subnet_root())
    miner_env = args.miner_env or default_miner_env_path()
    _SAMPLE_BENCHMARK._load_env(miner_env)

    log_file = args.log_file
    if log_file is None and args.challenge_uid:
        log_file = flow_log_path(workspace_root, challenge_uid=args.challenge_uid)
    logger = BenchmarkFlowLogger(log_file=log_file)
    s2s_config = S2sClientConfig(
        chunk_timeout_sec=max(0.001, args.chunk_timeout_sec),
        drain_timeout_sec=30.0,
        drain_max_requests=64,
        final_drain_min_timeout_sec=10.0,
        pace_realtime=bool(args.validator_mode),
    )

    try:
        flow_result = asyncio.run(
            run_challenge_flow(
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
                validator_mode=args.validator_mode,
                concurrent_validators=max(1, args.concurrent_validators),
                validator_utterance_mode=(
                    "random" if args.validator_random_utterances else "same"
                ),
                validator_random_seed=args.validator_random_seed,
                s2s_config=s2s_config,
            )
        )
    except ApiChallengeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    logger.write_log_file()
    print(json.dumps(flow_result["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
