from __future__ import annotations

import json
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from babelbit.scoring.reference_metadata import resolve_audio_reference_metadata

_SOUND_EFFECT_RE = re.compile(r"^\(SOUNDBITE", re.IGNORECASE)


@dataclass(frozen=True)
class ChallengeUtteranceRef:
    utterance_id: str
    dialogue_index: int
    utterance_index: int
    reference_text: str


@dataclass(frozen=True)
class MinerTestUtterance:
    challenge_uid: str
    utterance_id: str
    source_language: str
    target_language: str
    source_text: str
    reference_text: str
    challenge_doc: dict[str, Any]
    sample_path: Path
    source_audio_path: Path
    dialogue_index: int
    utterance_index: int


def workspace_root_from(subnet_root: Path) -> Path:
    return subnet_root.parent


def miner_test_data_root(workspace_root: Path) -> Path:
    return workspace_root / "miner-test-data"


def api_challenge_fixtures_dir(workspace_root: Path, *, challenge_uid: str) -> Path:
    return miner_test_data_root(workspace_root) / "api_challenges" / challenge_uid


def legacy_api_challenge_fixtures_dir(workspace_root: Path, *, challenge_uid: str) -> Path:
    return workspace_root / "babelbit_subnet" / "benchmark_outputs" / "api_challenges" / challenge_uid


def api_challenge_benchmark_dir(workspace_root: Path, *, challenge_uid: str) -> Path:
    return legacy_api_challenge_fixtures_dir(workspace_root, challenge_uid=challenge_uid) / "benchmark"


def locale_sample_path(
    workspace_root: Path,
    *,
    locale: str,
    relative_en_path: str = "en/npr/01/en-npr-001481.json",
) -> Path:
    rel = Path(relative_en_path)
    if rel.parts and rel.parts[0] == "en":
        rel = Path(*rel.parts[1:])
    if locale == "en":
        return miner_test_data_root(workspace_root) / "en" / rel
    return miner_test_data_root(workspace_root) / locale / rel


def source_audio_sidecar_path(sample_path: Path, *, utterance_id: str = "0") -> Path:
    return sample_path.with_name(f"{sample_path.stem}.u{utterance_id}.source.wav")


def wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def estimate_min_source_duration_sec(text: str, *, max_wps: float = 3.5) -> float:
    words = len(str(text).split())
    return max(1.0, words / max_wps)


def _normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", str(word).lower())


def transcript_word_recall(asr_text: str, reference_text: str) -> float:
    ref_words = {_normalize_word(word) for word in reference_text.split()}
    ref_words.discard("")
    if not ref_words:
        return 1.0
    asr_words = {_normalize_word(word) for word in asr_text.split()}
    return len(ref_words & asr_words) / len(ref_words)


def validate_source_audio_asr_roundtrip(
    source_text: str,
    *,
    asr_text: str,
    min_recall: float = 0.82,
) -> tuple[bool, str]:
    recall = transcript_word_recall(asr_text, source_text)
    if recall >= min_recall:
        return True, ""
    return (
        False,
        (
            f"source audio ASR roundtrip recall too low: {recall:.3f} < {min_recall:.3f} "
            f"for transcript {asr_text!r}"
        ),
    )


def normalize_locale_list(
    locales: list[str] | None,
    *,
    default: tuple[str, ...] = ("en", "fr", "de"),
) -> list[str]:
    ordered = list(locales) if locales else list(default)
    seen: set[str] = set()
    out: list[str] = []
    for locale in ordered:
        if locale not in seen:
            seen.add(locale)
            out.append(locale)
    return out


def validate_source_audio_duration(
    source_text: str,
    source_audio_path: Path,
    *,
    max_wps: float = 3.5,
    tolerance: float = 0.85,
) -> tuple[bool, str]:
    if not source_audio_path.is_file():
        return False, f"missing source audio: {source_audio_path}"
    duration_sec = wav_duration_sec(source_audio_path)
    min_duration_sec = estimate_min_source_duration_sec(source_text, max_wps=max_wps)
    if duration_sec < min_duration_sec * tolerance:
        return (
            False,
            (
                f"source audio too short: {duration_sec:.2f}s < "
                f"{min_duration_sec:.2f}s expected for {len(source_text.split())} words "
                f"({source_audio_path})"
            ),
        )
    return True, ""


def is_scorable_utterance(text: str, *, min_words: int = 4) -> bool:
    clean = str(text).strip()
    if not clean:
        return False
    if _SOUND_EFFECT_RE.match(clean):
        return False
    if clean in {"Right.", "OK.", "Thank you.", "Thanks.", "Yes.", "No."}:
        return False
    return len(clean.split()) >= min_words


def iter_en_challenge_utterances(
    en_doc: dict[str, Any],
    *,
    min_words: int = 4,
) -> list[ChallengeUtteranceRef]:
    dialogues = en_doc.get("dialogues")
    if not isinstance(dialogues, list):
        raise ValueError("challenge document is missing dialogues")

    refs: list[ChallengeUtteranceRef] = []
    flat_id = 0
    for dialogue_index, dialogue in enumerate(dialogues):
        if not isinstance(dialogue, dict):
            continue
        utterances = dialogue.get("utterances")
        if not isinstance(utterances, list):
            continue
        for utterance_index, raw in enumerate(utterances):
            text = str(raw).strip()
            if not is_scorable_utterance(text, min_words=min_words):
                continue
            refs.append(
                ChallengeUtteranceRef(
                    utterance_id=str(flat_id),
                    dialogue_index=dialogue_index,
                    utterance_index=utterance_index,
                    reference_text=text,
                )
            )
            flat_id += 1
    return refs


def _reference_wps(reference_text: str, *, source_duration_sec: float | None = None) -> float:
    words = len(reference_text.split())
    if source_duration_sec and source_duration_sec > 0:
        return round(words / source_duration_sec, 3)
    return round(words / max(0.1, words / 2.25), 3)


def build_utterance_entry(
    *,
    ref: ChallengeUtteranceRef,
    source_text: str,
    target_language: str = "en",
    source_duration_sec: float | None = None,
) -> dict[str, Any]:
    return {
        "utterance_id": ref.utterance_id,
        "utterance_index": int(ref.utterance_id),
        "dialogue_index": ref.dialogue_index,
        "dialogue_utterance_index": ref.utterance_index,
        "source_text": source_text,
        "utterance_translations": [
            {
                "language": target_language,
                "text": ref.reference_text,
                "reference_wps": _reference_wps(
                    ref.reference_text,
                    source_duration_sec=source_duration_sec,
                ),
            }
        ],
    }


def build_locale_challenge_doc(
    *,
    en_doc: dict[str, Any],
    locale: str,
    utterance_entries: list[dict[str, Any]],
    derived_from: str,
) -> dict[str, Any]:
    return {
        "challenge_uid": str(en_doc["challenge_uid"]),
        "language": locale,
        "source_language": locale,
        "target_language": "en",
        "derived_from": derived_from,
        "dialogues": en_doc.get("dialogues", []),
        "utterances": utterance_entries,
    }


def build_scoring_challenge_doc(utterance_entry: dict[str, Any], *, challenge_uid: str) -> dict[str, Any]:
    return {
        "challenge_uid": challenge_uid,
        "utterances": [utterance_entry],
    }


def list_challenge_utterance_ids(
    sample_path: Path,
    *,
    max_utterances: int | None = None,
    min_words: int = 4,
) -> list[str]:
    doc = json.loads(sample_path.read_text(encoding="utf-8"))
    utterances = doc.get("utterances")
    if isinstance(utterances, list) and utterances:
        ids = [str(entry.get("utterance_id")) for entry in utterances if isinstance(entry, dict)]
    else:
        ids = [ref.utterance_id for ref in iter_en_challenge_utterances(doc, min_words=min_words)]
    if max_utterances is not None:
        return ids[: max(0, max_utterances)]
    return ids


def _utterance_entry_from_doc(
    doc: dict[str, Any],
    *,
    utterance_id: str,
    min_words: int,
) -> dict[str, Any]:
    utterances = doc.get("utterances")
    if isinstance(utterances, list):
        for raw in utterances:
            if not isinstance(raw, dict):
                continue
            if str(raw.get("utterance_id")) == str(utterance_id):
                return raw

    for ref in iter_en_challenge_utterances(doc, min_words=min_words):
        if ref.utterance_id == str(utterance_id):
            return build_utterance_entry(
                ref=ref,
                source_text=ref.reference_text,
            )
    raise KeyError(f"utterance_id {utterance_id} not found in {doc.get('challenge_uid')}")


def load_miner_test_utterance(
    sample_path: Path,
    *,
    utterance_id: str = "0",
    require_source_audio: bool = True,
    min_words: int = 4,
) -> MinerTestUtterance:
    doc = json.loads(sample_path.read_text(encoding="utf-8"))
    challenge_uid = str(doc["challenge_uid"])
    entry = _utterance_entry_from_doc(doc, utterance_id=utterance_id, min_words=min_words)
    source_text = str(entry.get("source_text", "")).strip()
    if not source_text:
        raise ValueError(f"missing source_text for utterance {utterance_id} in {sample_path}")

    source_language = str(doc.get("source_language", doc.get("language", "en")))
    target_language = str(doc.get("target_language", "en"))
    challenge_doc = build_scoring_challenge_doc(entry, challenge_uid=challenge_uid)

    metadata = resolve_audio_reference_metadata(
        challenge_uid=challenge_uid,
        utterance_id=utterance_id,
        target_lang=target_language,
        challenge_doc=challenge_doc,
        metadata_source=str(sample_path),
    )
    source_audio_path = source_audio_sidecar_path(sample_path, utterance_id=utterance_id)
    if require_source_audio and not source_audio_path.is_file():
        raise FileNotFoundError(
            f"Missing cached source audio: {source_audio_path}. "
            "Run scripts/render_locale_source_audio.py first."
        )

    return MinerTestUtterance(
        challenge_uid=challenge_uid,
        utterance_id=str(utterance_id),
        source_language=source_language,
        target_language=target_language,
        source_text=source_text,
        reference_text=metadata.reference_text,
        challenge_doc=challenge_doc,
        sample_path=sample_path,
        source_audio_path=source_audio_path,
        dialogue_index=int(entry.get("dialogue_index", 0)),
        utterance_index=int(entry.get("dialogue_utterance_index", entry.get("utterance_index", 0))),
    )


def summarize_accuracy_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [item for item in results if isinstance(item.get("validator_score"), dict)]
    if not scored:
        return {
            "utterance_count": len(results),
            "scored_utterance_count": 0,
            "mean_accuracy": 0.0,
            "accuracy_pass_rate": 0.0,
            "mean_production_score": 0.0,
            "production_pass_rate": 0.0,
        }

    accuracies = [float(item["validator_score"].get("accuracy") or 0.0) for item in scored]
    production_scores = [float(item["validator_score"].get("score") or 0.0) for item in scored]
    accuracy_passes = [bool(item["validator_score"].get("accuracy_pass")) for item in scored]
    return {
        "utterance_count": len(results),
        "scored_utterance_count": len(scored),
        "mean_accuracy": round(sum(accuracies) / len(accuracies), 6),
        "min_accuracy": round(min(accuracies), 6),
        "max_accuracy": round(max(accuracies), 6),
        "accuracy_pass_rate": round(sum(1 for passed in accuracy_passes if passed) / len(scored), 6),
        "mean_production_score": round(sum(production_scores) / len(production_scores), 6),
        "production_pass_rate": round(sum(1 for score in production_scores if score > 0.0) / len(scored), 6),
    }
