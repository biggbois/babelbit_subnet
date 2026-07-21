#!/usr/bin/env python3
"""Run a production-like challenge benchmark across many utterances."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from babelbit.benchmarks.miner_test_data import (
    list_challenge_utterance_ids,
    locale_sample_path,
    summarize_accuracy_results,
    workspace_root_from,
)
from babelbit.benchmarks.s2s_client import S2sClientConfig

_BENCHMARK_MODULE_PATH = Path(__file__).resolve().with_name("benchmark_miner_sample.py")
_SPEC = importlib.util.spec_from_file_location("benchmark_miner_sample", _BENCHMARK_MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BENCHMARK
_SPEC.loader.exec_module(_BENCHMARK)


def subnet_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_miner_env_path() -> Path:
    return workspace_root_from(subnet_root()) / "babelbit_miner/.env"


def default_output_dir(*, locale: str, sample_stem: str) -> Path:
    return subnet_root() / "benchmark_outputs" / f"challenge_{locale}" / sample_stem


def run_challenge_benchmark(
    *,
    base_url: str,
    sample_path: Path,
    out_dir: Path,
    utterance_ids: list[str],
    skip_validator_score: bool,
    accuracy_only: bool,
    on_utterance_complete: Any | None = None,
    stage_name: str | None = None,
    validator_mode: bool = False,
    concurrent_validators: int = 1,
    validator_utterance_mode: str = "same",
    validator_random_seed: int = 59,
    s2s_config: S2sClientConfig | None = None,
) -> dict[str, Any]:
    utterance_results: list[dict[str, Any]] = []
    total = len(utterance_ids)
    mode = str(validator_utterance_mode or "same").strip().lower()
    if mode not in {"same", "random"}:
        raise ValueError("validator_utterance_mode must be 'same' or 'random'")

    def _record_result(
        *,
        utterance_id: str,
        result: dict[str, Any],
        index: int,
        total_count: int,
    ) -> None:
        validator_score = result.get("validator_score")
        if isinstance(validator_score, dict):
            result["accuracy_focus"] = {
                "accuracy": validator_score.get("accuracy"),
                "accuracy_pass": validator_score.get("accuracy_pass"),
                "speech_rate_penalty": (validator_score.get("speech_rate") or {}).get("penalty"),
                "production_score": validator_score.get("score"),
                "stt_text": validator_score.get("stt_text"),
                "gt_text": validator_score.get("gt_text"),
            }
        utterance_results.append(result)
        if on_utterance_complete is not None:
            on_utterance_complete(
                stage_name or "challenge",
                utterance_id,
                result,
                index,
                total_count,
            )

    if validator_mode and mode == "random" and concurrent_validators > 1:
        shuffled = list(utterance_ids)
        random.Random(int(validator_random_seed)).shuffle(shuffled)
        result_by_index: dict[int, dict[str, Any]] = {}
        total_count = len(shuffled)
        for wave_start in range(0, total_count, concurrent_validators):
            wave_ids = shuffled[wave_start : wave_start + concurrent_validators]
            with ThreadPoolExecutor(max_workers=len(wave_ids)) as pool:
                futures = {
                    pool.submit(
                        _BENCHMARK.run_benchmark,
                        base_url=base_url,
                        sample_path=sample_path,
                        out_dir=out_dir / f"wave{wave_start // concurrent_validators}" / f"u{uid}",
                        skip_validator_score=skip_validator_score,
                        utterance_id=uid,
                        validator_mode=True,
                        concurrent_validators=1,
                        s2s_config=s2s_config,
                    ): wave_start + idx
                    for idx, uid in enumerate(wave_ids)
                }
                for future in as_completed(futures):
                    result_by_index[futures[future]] = future.result()

        for index in sorted(result_by_index):
            result = result_by_index[index]
            utterance_id = str(result.get("utterance_id") or shuffled[index])
            _record_result(
                utterance_id=utterance_id,
                result=result,
                index=index + 1,
                total_count=total_count,
            )
        summary = summarize_accuracy_results(utterance_results)
        challenge_result = {
            "sample": str(sample_path),
            "stage": stage_name,
            "utterance_ids": shuffled,
            "utterance_count": len(shuffled),
            "validator_mode": validator_mode,
            "concurrent_validators": concurrent_validators,
            "validator_utterance_mode": mode,
            "validator_random_seed": int(validator_random_seed),
            "accuracy_summary": summary,
            "utterances": utterance_results,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "challenge_result.json").write_text(
            json.dumps(challenge_result, indent=2) + "\n",
            encoding="utf-8",
        )
        return challenge_result

    for index, utterance_id in enumerate(utterance_ids, start=1):
        result = _BENCHMARK.run_benchmark(
            base_url=base_url,
            sample_path=sample_path,
            out_dir=out_dir / f"u{utterance_id}",
            skip_validator_score=skip_validator_score,
            utterance_id=utterance_id,
            validator_mode=validator_mode,
            concurrent_validators=concurrent_validators,
            s2s_config=s2s_config,
        )
        _record_result(
            utterance_id=utterance_id,
            result=result,
            index=index,
            total_count=total,
        )

    summary = summarize_accuracy_results(utterance_results)
    challenge_result = {
        "sample": str(sample_path),
        "stage": stage_name,
        "utterance_ids": utterance_ids,
        "utterance_count": len(utterance_ids),
        "validator_mode": validator_mode,
        "concurrent_validators": concurrent_validators,
        "validator_utterance_mode": mode,
        "validator_random_seed": int(validator_random_seed),
        "accuracy_summary": summary,
        "utterances": utterance_results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "challenge_result.json").write_text(
        json.dumps(challenge_result, indent=2) + "\n",
        encoding="utf-8",
    )
    return challenge_result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark all scorable utterances in one miner-test-data challenge."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--locale", choices=["en", "fr", "de"], default="fr")
    parser.add_argument("--sample", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--miner-env", type=Path, default=None)
    parser.add_argument("--max-utterances", type=int, default=None, help="Limit utterances (default: all).")
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--skip-validator-score", action="store_true")
    parser.add_argument(
        "--accuracy-only",
        action="store_true",
        help="Include per-utterance accuracy_focus fields and challenge accuracy summary.",
    )
    parser.add_argument("--validator-mode", action="store_true")
    parser.add_argument("--concurrent-validators", type=int, default=1)
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
    parser.add_argument("--chunk-timeout-sec", type=float, default=3.0)
    args = parser.parse_args()

    workspace_root = workspace_root_from(subnet_root())
    miner_env = args.miner_env or default_miner_env_path()
    _BENCHMARK._load_env(miner_env)

    if args.sample is None:
        sample_path = locale_sample_path(workspace_root, locale=args.locale)
    else:
        sample_path = args.sample
        if not sample_path.is_file():
            sample_path = locale_sample_path(
                workspace_root,
                locale=args.locale,
                relative_en_path=str(args.sample),
            )

    utterance_ids = list_challenge_utterance_ids(
        sample_path,
        max_utterances=args.max_utterances,
        min_words=args.min_words,
    )
    if not utterance_ids:
        raise RuntimeError(f"No scorable utterances found in {sample_path}")

    out_dir = args.out_dir or default_output_dir(
        locale=args.locale,
        sample_stem=sample_path.stem,
    )
    result = run_challenge_benchmark(
        base_url=args.base_url,
        sample_path=sample_path,
        out_dir=out_dir,
        utterance_ids=utterance_ids,
        skip_validator_score=args.skip_validator_score,
        accuracy_only=args.accuracy_only or True,
        validator_mode=args.validator_mode,
        concurrent_validators=max(1, args.concurrent_validators),
        validator_utterance_mode="random" if args.validator_random_utterances else "same",
        validator_random_seed=args.validator_random_seed,
        s2s_config=S2sClientConfig(chunk_timeout_sec=max(0.001, args.chunk_timeout_sec)),
    )
    print(json.dumps(result["accuracy_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
