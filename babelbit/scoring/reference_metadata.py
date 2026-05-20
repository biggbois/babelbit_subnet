from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from babelbit.scoring.scoring_common import (
    WordTS,
    canonical_utterance_ids,
    maybe_float,
    normalize_words,
    reference_wps,
)
from babelbit.utils.settings import get_settings

_METADATA_CACHE: Dict[Tuple[str, str], Tuple[Dict[str, Any], Path]] = {}
MetadataSource = Union[Path, str]


@dataclass(frozen=True)
class AudioReferenceMetadata:
    challenge_uid: str
    utterance_id: str
    reference_text: str
    reference_wps: float
    reference_words: List[WordTS]
    metadata_source: str


def _candidate_metadata_paths(root: Path, challenge_uid: str) -> List[Path]:
    return [root / challenge_uid / "challenge.json", root / f"{challenge_uid}.json"]


def _load_metadata_document(
    root: Path, challenge_uid: str
) -> Tuple[Dict[str, Any], Path]:
    cache_key = (str(root.resolve()), challenge_uid)
    if cache_key in _METADATA_CACHE:
        return _METADATA_CACHE[cache_key]

    for candidate in _candidate_metadata_paths(root, challenge_uid):
        if not candidate.exists():
            continue
        loaded = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid scoring metadata payload in {candidate}")
        result = (loaded, candidate)
        _METADATA_CACHE[cache_key] = result
        return result

    raise FileNotFoundError(
        f"No audio scoring metadata found for challenge {challenge_uid} under {root}"
    )


def _build_metadata(
    *,
    challenge_uid: str,
    utterance_id: str,
    source_path: MetadataSource,
    reference_text: str,
    reference_words: List[WordTS],
    provided_wps: Optional[float],
) -> AudioReferenceMetadata:
    clean_text = str(reference_text).strip()
    if not clean_text:
        raise ValueError("Missing reference text in scoring metadata")
    return AudioReferenceMetadata(
        challenge_uid=challenge_uid,
        utterance_id=utterance_id,
        reference_text=clean_text,
        reference_wps=reference_wps(reference_words, provided_wps),
        reference_words=reference_words,
        metadata_source=str(source_path),
    )


def _extract_reference_from_target_schema(
    raw_utterance: Dict[str, Any],
    *,
    challenge_uid: str,
    utterance_id: str,
    source_path: MetadataSource,
) -> AudioReferenceMetadata:
    target_meta = raw_utterance.get("target")
    if not isinstance(target_meta, dict):
        raise ValueError("Missing target scoring metadata")
    return _build_metadata(
        challenge_uid=challenge_uid,
        utterance_id=utterance_id,
        source_path=source_path,
        reference_text=str(target_meta.get("text", "")),
        reference_words=normalize_words(target_meta.get("words")),
        provided_wps=maybe_float(target_meta.get("wps")),
    )


def _extract_reference_from_flat_schema(
    raw_utterance: Dict[str, Any],
    *,
    challenge_uid: str,
    utterance_id: str,
    source_path: MetadataSource,
) -> AudioReferenceMetadata:
    return _build_metadata(
        challenge_uid=challenge_uid,
        utterance_id=utterance_id,
        source_path=source_path,
        reference_text=str(raw_utterance.get("reference_text", "")),
        reference_words=normalize_words(raw_utterance.get("reference_words")),
        provided_wps=maybe_float(raw_utterance.get("reference_wps")),
    )


def _extract_reference_from_translation_schema(
    raw_utterance: Dict[str, Any],
    *,
    challenge_uid: str,
    utterance_id: str,
    source_path: MetadataSource,
    target_lang: str,
) -> AudioReferenceMetadata:
    translations = raw_utterance.get("utterance_translations")
    if not isinstance(translations, list):
        raise ValueError("Missing utterance_translations in scoring metadata")

    selected: Optional[Dict[str, Any]] = None
    for translation in translations:
        if not isinstance(translation, dict):
            continue
        if str(translation.get("language", "")).strip().lower() == target_lang.lower():
            selected = translation
            break
        if selected is None:
            selected = translation

    if selected is None:
        raise ValueError("No usable utterance translation found in scoring metadata")

    return _build_metadata(
        challenge_uid=challenge_uid,
        utterance_id=utterance_id,
        source_path=source_path,
        reference_text=str(selected.get("text", "")),
        reference_words=normalize_words(selected.get("words")),
        provided_wps=maybe_float(selected.get("reference_wps")),
    )


def resolve_audio_reference_metadata(
    *,
    challenge_uid: str,
    utterance_id: str,
    target_lang: str = "en",
    metadata_root: Optional[Path] = None,
    challenge_doc: Optional[Dict[str, Any]] = None,
    metadata_source: Optional[str] = None,
) -> AudioReferenceMetadata:
    if challenge_doc is None:
        if metadata_root is None:
            metadata_root = get_settings().BB_AUDIO_SCORING_METADATA_ROOT
        if metadata_root is None:
            raise FileNotFoundError("BB_AUDIO_SCORING_METADATA_ROOT is not configured")
        challenge_doc, source_path = _load_metadata_document(
            metadata_root, challenge_uid
        )
    else:
        source_path = metadata_source or f"utterance_engine:{challenge_uid}"

    utterances = challenge_doc.get("utterances")
    if not isinstance(utterances, list):
        raise ValueError(f"Invalid scoring metadata format in {source_path}")

    requested_ids = canonical_utterance_ids(utterance_id)
    for index, raw_utterance in enumerate(utterances):
        if not isinstance(raw_utterance, dict):
            continue
        raw_id = raw_utterance.get(
            "utterance_id", raw_utterance.get("utterance_index", index)
        )
        if requested_ids.isdisjoint(canonical_utterance_ids(raw_id)):
            continue
        if isinstance(raw_utterance.get("target"), dict):
            return _extract_reference_from_target_schema(
                raw_utterance,
                challenge_uid=challenge_uid,
                utterance_id=utterance_id,
                source_path=source_path,
            )
        if "reference_text" in raw_utterance:
            return _extract_reference_from_flat_schema(
                raw_utterance,
                challenge_uid=challenge_uid,
                utterance_id=utterance_id,
                source_path=source_path,
            )
        if "utterance_translations" in raw_utterance:
            return _extract_reference_from_translation_schema(
                raw_utterance,
                challenge_uid=challenge_uid,
                utterance_id=utterance_id,
                source_path=source_path,
                target_lang=target_lang,
            )
        raise ValueError(
            f"Unsupported scoring metadata schema for utterance {utterance_id}"
        )

    raise KeyError(
        f"Utterance {utterance_id} was not found in scoring metadata for {challenge_uid}"
    )


__all__ = ["AudioReferenceMetadata", "resolve_audio_reference_metadata"]
