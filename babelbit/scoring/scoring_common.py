from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

WordTS = Dict[str, Union[str, float]]


def normalize_words(words: Any) -> List[WordTS]:
    out: List[WordTS] = []
    if not isinstance(words, list):
        return out
    for word in words:
        if isinstance(word, dict):
            raw_word = word.get("word")
            start = word.get("start")
            end = word.get("end")
        else:
            raw_word = getattr(word, "word", None)
            start = getattr(word, "start", None)
            end = getattr(word, "end", None)
        if not isinstance(raw_word, str) or start is None or end is None:
            continue
        try:
            out.append(
                {
                    "word": raw_word.strip(),
                    "start": float(start),
                    "end": float(end),
                }
            )
        except Exception:
            continue
    return out


def canonical_utterance_ids(utterance_id: Any) -> set[str]:
    values: set[str] = set()
    text = str(utterance_id).strip()
    if not text:
        return values
    values.add(text)
    tail = text.rsplit(":", 1)[-1].strip()
    if tail:
        values.add(tail)
    for candidate in list(values):
        try:
            values.add(str(int(candidate)))
        except Exception:
            continue
    return values


def maybe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def reference_wps(
    reference_words: List[WordTS], provided_wps: Optional[float]
) -> float:
    if provided_wps and provided_wps > 0:
        return float(provided_wps)
    if len(reference_words) < 2:
        return 4.0
    first_start = float(reference_words[0]["start"])
    last_end = float(reference_words[-1]["end"])
    duration = last_end - first_start
    if duration <= 0:
        return 4.0
    word_count = len(
        [word for word in reference_words if str(word.get("word", "")).strip()]
    )
    return word_count / duration if word_count > 0 else 4.0


__all__ = [
    "WordTS",
    "canonical_utterance_ids",
    "maybe_float",
    "normalize_words",
    "reference_wps",
]
