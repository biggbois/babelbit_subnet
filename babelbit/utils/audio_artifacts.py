from __future__ import annotations

import json
import tarfile
from logging import getLogger
from pathlib import Path

from babelbit.schemas.audio_prediction import (
    BBAudioChallengeResult,
    BBAudioUtteranceResult,
)
from babelbit.utils.file_handling import normalize_challenge_type
from babelbit.utils.miner_registry import Miner

logger = getLogger(__name__)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _serialize_optional_text(value: str | None) -> str | None:
    text = str(value or "").strip()
    return text or None


def _build_challenge_summary(challenge_result: BBAudioChallengeResult) -> dict:
    utterance_scores = [
        float(utterance.score) for utterance in challenge_result.utterances
    ]
    accuracies = [
        float(utterance.accuracy) for utterance in challenge_result.utterances
    ]
    completion_seconds = [
        float(utterance.effective_completion_sec)
        for utterance in challenge_result.utterances
    ]
    return {
        "average_U_best_early": round(_mean(utterance_scores), 6),
        "accuracy": round(_mean(accuracies), 6),
        "effective_completion_sec": round(_mean(completion_seconds), 4),
        "completed_utterances": sum(
            1 for utterance in challenge_result.utterances if utterance.completed
        ),
        "total_utterances": len(challenge_result.utterances),
    }


def _build_challenge_run_utterance_entry(
    utterance_result: BBAudioUtteranceResult,
) -> dict:
    entry = {
        "utterance_uid": utterance_result.utterance_id,
        "utterance_number": utterance_result.utterance_index,
        "reference_text": utterance_result.reference_text,
        "transcript": _serialize_optional_text(utterance_result.transcript_text),
        "score": utterance_result.score,
        "accuracy": utterance_result.accuracy,
        "effective_completion_sec": utterance_result.effective_completion_sec,
        "source_duration_sec": utterance_result.source_duration_sec,
        "predicted_duration_sec": utterance_result.predicted_duration_sec,
        "frame_count_in": utterance_result.frame_count_in,
        "frame_count_out": utterance_result.frame_count_out,
        "predicted_num_bytes": utterance_result.predicted_num_bytes,
        "score_breakdown": utterance_result.score_breakdown,
        "score_is_fallback": utterance_result.score_is_fallback,
        "score_method": utterance_result.score_method,
        "completed": utterance_result.completed,
    }
    if utterance_result.error:
        entry["error"] = utterance_result.error
    return entry


def _build_challenge_score_utterance_entry(
    challenge_uid: str,
    utterance_result: BBAudioUtteranceResult,
) -> dict:
    return {
        "utterance_uid": utterance_result.utterance_id
        or f"{challenge_uid}:{utterance_result.utterance_index}",
        "utterance_score": utterance_result.score,
        "utterance_index": utterance_result.utterance_index,
        "score_is_fallback": utterance_result.score_is_fallback,
        "score_method": utterance_result.score_method,
    }


def save_audio_run_log(
    challenge_result: BBAudioChallengeResult,
    *,
    output_dir: str = "logs",
) -> tuple[Path, dict]:
    base_dir = (
        Path(output_dir)
        / "s2s"
        / challenge_result.challenge_uid
        / f"miner_{challenge_result.miner_uid}__hk_{challenge_result.miner_hotkey}"
    )
    source_dir = base_dir / "source"
    predicted_dir = base_dir / "predicted"
    source_dir.mkdir(parents=True, exist_ok=True)
    predicted_dir.mkdir(parents=True, exist_ok=True)

    utterance_entries = []
    for utterance in challenge_result.utterances:
        utterance_filename = f"utt_{utterance.utterance_index:04d}.wav"
        source_relpath = f"source/{utterance_filename}"
        predicted_relpath = None

        source_path = source_dir / utterance_filename
        if utterance.source_audio_bytes:
            source_path.write_bytes(utterance.source_audio_bytes)
        if utterance.predicted_audio_bytes:
            predicted_relpath = f"predicted/{utterance_filename}"
            predicted_path = predicted_dir / utterance_filename
            predicted_path.write_bytes(utterance.predicted_audio_bytes)

        utterance.source_audio_path = source_relpath
        utterance.predicted_audio_path = predicted_relpath
        utterance_entries.append(
            utterance.model_dump(
                exclude={"source_audio_bytes", "predicted_audio_bytes"}
            )
        )

    log_doc = {
        "challenge_uid": challenge_result.challenge_uid,
        "challenge_type": challenge_result.challenge_type,
        "miner_uid": challenge_result.miner_uid,
        "miner_hotkey": challenge_result.miner_hotkey,
        "protocol": challenge_result.protocol,
        "score_is_fallback": challenge_result.score_is_fallback,
        "score_method": challenge_result.score_method,
        "completed": challenge_result.completed,
        "score": challenge_result.score,
        "utterances": utterance_entries,
    }
    if challenge_result.error:
        log_doc["error"] = challenge_result.error

    log_path = base_dir / "run.json"
    log_path.write_text(json.dumps(log_doc, indent=2), encoding="utf-8")
    logger.info("Saved S2S audio run log: %s", log_path)
    return log_path, log_doc


def save_audio_artifact_bundle(
    challenge_result: BBAudioChallengeResult,
    *,
    output_dir: str = "logs",
) -> Path:
    base_dir = (
        Path(output_dir)
        / "s2s"
        / challenge_result.challenge_uid
        / f"miner_{challenge_result.miner_uid}__hk_{challenge_result.miner_hotkey}"
    )
    tar_path = base_dir / "audio.tar"
    with tarfile.open(tar_path, "w") as tar_file:
        for relative_dir in ("source", "predicted"):
            directory = base_dir / relative_dir
            if not directory.exists():
                continue
            for file_path in sorted(directory.glob("*.wav")):
                tar_file.add(file_path, arcname=str(file_path.relative_to(base_dir)))
    logger.info("Saved S2S audio bundle: %s", tar_path)
    return tar_path


def create_audio_challenge_run_data(
    *,
    miner: Miner,
    challenge_uid: str,
    challenge_type: str,
    challenge_result: BBAudioChallengeResult,
    log_file_path: str,
) -> dict:
    return {
        "log_file": log_file_path,
        "challenge_uid": challenge_uid,
        "challenge_type": normalize_challenge_type(challenge_type, default="main"),
        "miner_uid": miner.uid,
        "miner_hotkey": miner.hotkey,
        "utterances": [
            _build_challenge_run_utterance_entry(utterance_result)
            for utterance_result in challenge_result.utterances
        ],
        "challenge_summary": _build_challenge_summary(challenge_result),
        "protocol": "s2s_audio_v1",
        "score_is_fallback": challenge_result.score_is_fallback,
        "score_method": challenge_result.score_method,
        "completed": challenge_result.completed,
        **({"error": challenge_result.error} if challenge_result.error else {}),
    }


def create_audio_challenge_score_data(
    *,
    miner: Miner,
    challenge_uid: str,
    challenge_type: str,
    challenge_result: BBAudioChallengeResult,
    log_file_path: str,
) -> dict:
    challenge_mean_u = round(
        _mean([float(utterance.score) for utterance in challenge_result.utterances]),
        6,
    )
    return {
        "run_file": log_file_path,
        "challenge_uid": challenge_uid,
        "challenge_type": normalize_challenge_type(challenge_type, default="main"),
        "miner_uid": miner.uid,
        "miner_hotkey": miner.hotkey,
        "utterances": [
            _build_challenge_score_utterance_entry(challenge_uid, utterance)
            for utterance in challenge_result.utterances
        ],
        "challenge_mean_U": challenge_mean_u,
        "challenge_summary": _build_challenge_summary(challenge_result),
        "protocol": "s2s_audio_v1",
        "score_is_fallback": challenge_result.score_is_fallback,
        "score_method": challenge_result.score_method,
        "completed": challenge_result.completed,
        **({"error": challenge_result.error} if challenge_result.error else {}),
    }


def create_audio_dialogue_score_file_data(
    *,
    miner: Miner,
    challenge_uid: str,
    challenge_type: str,
    utterance_result: BBAudioUtteranceResult,
    log_file_path: str,
) -> dict:
    legacy_challenge_result = BBAudioChallengeResult(
        challenge_uid=challenge_uid,
        challenge_type=challenge_type,
        miner_uid=miner.uid,
        miner_hotkey=miner.hotkey,
        utterances=[utterance_result],
        completed=utterance_result.completed,
        score=utterance_result.score,
        score_is_fallback=utterance_result.score_is_fallback,
        score_method=utterance_result.score_method,
    )
    return create_audio_challenge_run_data(
        miner=miner,
        challenge_uid=challenge_uid,
        challenge_type=challenge_type,
        challenge_result=legacy_challenge_result,
        log_file_path=log_file_path,
    )


def create_audio_challenge_summary_data(
    *,
    miner: Miner,
    challenge_uid: str,
    challenge_type: str,
    challenge_result: BBAudioChallengeResult,
    log_file_path: str,
) -> dict:
    return create_audio_challenge_score_data(
        miner=miner,
        challenge_uid=challenge_uid,
        challenge_type=challenge_type,
        challenge_result=challenge_result,
        log_file_path=log_file_path,
    )


__all__ = [
    "create_audio_challenge_run_data",
    "create_audio_challenge_score_data",
    "create_audio_challenge_summary_data",
    "create_audio_dialogue_score_file_data",
    "save_audio_artifact_bundle",
    "save_audio_run_log",
]
