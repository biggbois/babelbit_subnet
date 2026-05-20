import json
import os
from datetime import datetime
from logging import getLogger
from typing import Optional, Set, Tuple

logger = getLogger(__name__)


_CHALLENGE_TYPE_ALIASES = {
    "round1": "main",
    "round2": "arena",
}


def _resolve_timestamp(timestamp: Optional[str] = None) -> str:
    if isinstance(timestamp, str) and timestamp.strip():
        return timestamp.strip()
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_challenge_type(
    challenge_type: object, default: Optional[str] = None
) -> Optional[str]:
    if isinstance(challenge_type, str):
        normalized = challenge_type.strip().lower()
        if normalized:
            return _CHALLENGE_TYPE_ALIASES.get(normalized, normalized)
    return default


def save_challenge_run_file(
    run_data: dict, output_dir: str = "scores", timestamp: Optional[str] = None
):
    os.makedirs(output_dir, exist_ok=True)

    timestamp_value = _resolve_timestamp(timestamp)
    miner_uid = run_data.get("miner_uid", "unknown")
    challenge_uid = run_data.get("challenge_uid", "unknown")
    challenge_type = normalize_challenge_type(
        run_data.get("challenge_type"), default="main"
    )
    run_data["challenge_type"] = challenge_type

    filename = (
        f"challenge_run_{challenge_uid}_type_{challenge_type}"
        f"_miner_{miner_uid}_run_{timestamp_value}.json"
    )
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as file_handle:
        json.dump(run_data, file_handle, indent=2)

    logger.info("Saved challenge run file: %s", filepath)
    return filepath


def save_challenge_score_file(
    summary_data: dict, output_dir: str = "scores", timestamp: Optional[str] = None
):
    os.makedirs(output_dir, exist_ok=True)

    timestamp_value = _resolve_timestamp(timestamp)
    challenge_uid = summary_data.get("challenge_uid", "unknown")
    miner_uid = summary_data.get("miner_uid", "unknown")
    challenge_type = normalize_challenge_type(
        summary_data.get("challenge_type"), default="main"
    )
    summary_data["challenge_type"] = challenge_type

    filename = f"challenge_score_{challenge_uid}_type_{challenge_type}_miner_{miner_uid}_score_{timestamp_value}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as file_handle:
        json.dump(summary_data, file_handle, indent=2)

    logger.info("Saved challenge score file: %s", filepath)
    return filepath


def save_dialogue_score_file(
    score_data: dict, output_dir: str = "scores", timestamp: Optional[str] = None
):
    return save_challenge_run_file(
        score_data, output_dir=output_dir, timestamp=timestamp
    )


def save_challenge_summary_file(
    summary_data: dict, output_dir: str = "scores", timestamp: Optional[str] = None
):
    return save_challenge_score_file(
        summary_data, output_dir=output_dir, timestamp=timestamp
    )


def get_processed_miners_for_challenge(
    output_dir: str,
    challenge_uid: str,
    challenge_type: Optional[str] = None,
) -> Set[Tuple[int, str]]:
    processed: Set[Tuple[int, str]] = set()
    if not os.path.isdir(output_dir):
        return processed

    normalized_requested_type = normalize_challenge_type(challenge_type)

    for fname in os.listdir(output_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(output_dir, fname)
        try:
            with open(fpath, "r") as file_handle:
                data = json.load(file_handle)
            if (
                isinstance(data, dict)
                and data.get("challenge_uid") == challenge_uid
                and "miner_uid" in data
                and "miner_hotkey" in data
            ):
                if normalized_requested_type is not None:
                    normalized_type = normalize_challenge_type(
                        data.get("challenge_type"), default="main"
                    )
                    if normalized_type != normalized_requested_type:
                        continue
                try:
                    miner_uid = int(data["miner_uid"])
                    miner_hotkey = str(data["miner_hotkey"])
                    processed.add((miner_uid, miner_hotkey))
                except Exception:
                    continue
        except Exception:
            continue

    if processed:
        if challenge_type is None:
            logger.info(
                "Detected %d already processed miners for challenge %s",
                len(processed),
                challenge_uid,
            )
        else:
            logger.info(
                "Detected %d already processed miners for challenge %s (type=%s)",
                len(processed),
                challenge_uid,
                challenge_type,
            )
    return processed
