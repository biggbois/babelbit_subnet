from __future__ import annotations

import io
import wave
from hashlib import sha256
from logging import getLogger
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

import torch

from babelbit.scoring.reference_metadata import resolve_audio_reference_metadata
from babelbit.scoring.scoring_common import WordTS
from babelbit.scoring.stt import transcribe_wav_bytes, transcribe_wav_bytes_batch
from babelbit.scoring.text_embeddings import (
    cosine_similarity,
    embed_text,
    embed_texts_batch,
    get_reference_embedding,
)
from babelbit.utils.settings import get_settings

DEFAULT_EMBEDDER = "all-MiniLM-L6-v2"
SEMANTIC_AUDIO_SCORING_MODE = "semantic_audio_v1"
SEMANTIC_AUDIO_ERROR_MODE = "semantic_audio_v1_error"
_EMPTY_TRANSCRIPT_MIN_DURATION_SEC = 0.5
_DEFAULT_ACCURACY_THRESHOLD = 0.65
_DEFAULT_RATE_LOWER = 0.3
_DEFAULT_RATE_UPPER = 1.3
_DEFAULT_LATENCY_OVERSHOOT_FRACTION = 0.6
_DEFAULT_LATENCY_MIN_OVERSHOOT_SEC = 2.0
_DEFAULT_LATENCY_MAX_OVERSHOOT_SEC = 10.0
_DEFAULT_LATENCY_POWER = 2.0
logger = getLogger(__name__)


def _uses_legacy_weighted_scoring(settings: Any) -> bool:
    return hasattr(settings, "BB_AUDIO_SCORING_ACC_WEIGHT") and not hasattr(
        settings, "BB_AUDIO_SCORING_ACCURACY_THRESHOLD"
    )


def _compute_composite_score(
    *, accuracy: float, speech_rate_penalty: float, latency_score: float, settings: Any
) -> float:
    if _uses_legacy_weighted_scoring(settings):
        return (
            accuracy ** float(getattr(settings, "BB_AUDIO_SCORING_ACC_WEIGHT", 1.0))
            * speech_rate_penalty
            ** float(getattr(settings, "BB_AUDIO_SCORING_SR_PENALTY_WEIGHT", 1.0))
            * latency_score
            ** float(getattr(settings, "BB_AUDIO_SCORING_LATENCY_WEIGHT", 1.0))
        )

    accuracy_pass = accuracy >= getattr(
        settings,
        "BB_AUDIO_SCORING_ACCURACY_THRESHOLD",
        _DEFAULT_ACCURACY_THRESHOLD,
    )
    rate_pass = speech_rate_penalty > 0.0
    return latency_score if (accuracy_pass and rate_pass) else 0.0


def _empty_transcript_score_error(predicted_duration_sec: float) -> str | None:
    if predicted_duration_sec <= 0.0:
        return None
    if predicted_duration_sec >= _EMPTY_TRANSCRIPT_MIN_DURATION_SEC:
        return "RuntimeError:empty transcript from STT for non-trivial audio output"
    return (
        "RuntimeError:empty transcript from STT for short audio output "
        f"({predicted_duration_sec:.4f}s)"
    )


def compute_accuracy(stt_text: str, reference_text: str, embedder_model: str) -> float:
    if not stt_text.strip():
        return 0.0
    ref_vec = get_reference_embedding(reference_text, embedder_model)
    stt_vec = embed_text(stt_text, embedder_model)
    return max(0.0, min(1.0, cosine_similarity(stt_vec, ref_vec)))


def compute_accuracy_batch(
    stt_texts: List[str],
    reference_text: str,
    embedder_model: str,
) -> List[float]:
    if not stt_texts:
        return []

    ref_vec = get_reference_embedding(reference_text, embedder_model)

    valid_indices = [i for i, t in enumerate(stt_texts) if t.strip()]
    if not valid_indices:
        return [0.0] * len(stt_texts)

    valid_texts = [stt_texts[i] for i in valid_indices]
    stt_vecs = embed_texts_batch(valid_texts, embedder_model)

    results = [0.0] * len(stt_texts)
    for batch_idx, orig_idx in enumerate(valid_indices):
        sim = float(torch.dot(stt_vecs[batch_idx], ref_vec))
        results[orig_idx] = max(0.0, min(1.0, sim))

    return results


def compute_speech_rate_penalty(
    stt_words: List[WordTS],
    reference_wps: float,
    *,
    rate_lower: float = 0.3,
    rate_upper: float = 1.3,
) -> Dict[str, Any]:
    """Binary gate: reject speech that is anomalously fast or slow.

    Returns penalty=1.0 (pass) if miner WPS ratio is within [rate_lower, rate_upper]
    relative to reference.  Returns penalty=0.0 (fail) otherwise.
    """
    if not stt_words or reference_wps <= 0:
        return {
            "penalty": 1.0,
            "miner_wps": None,
            "reference_wps": round(reference_wps, 3),
            "ratio": None,
        }

    deduped: List[WordTS] = []
    for word in stt_words:
        if (
            deduped
            and float(word["start"]) == float(deduped[-1]["start"])
            and float(word["end"]) == float(deduped[-1]["end"])
        ):
            continue
        deduped.append(word)

    if not deduped:
        return {
            "penalty": 0.0,
            "miner_wps": None,
            "reference_wps": round(reference_wps, 3),
            "ratio": None,
        }

    speaking_duration = float(deduped[-1]["end"]) - float(deduped[0]["start"])
    if speaking_duration <= 0.05:
        return {
            "penalty": 0.0,
            "miner_wps": None,
            "reference_wps": round(reference_wps, 3),
            "ratio": None,
        }

    word_count = len([word for word in deduped if str(word.get("word", "")).strip()])
    miner_wps = word_count / speaking_duration
    ratio = miner_wps / reference_wps
    penalty = 1.0 if rate_lower <= ratio <= rate_upper else 0.0

    return {
        "penalty": round(penalty, 6),
        "miner_wps": round(miner_wps, 3),
        "reference_wps": round(reference_wps, 3),
        "ratio": round(ratio, 3),
    }


def compute_latency_score(
    completion_sec: float,
    source_duration_sec: float,
    *,
    overshoot_fraction: float = 0.3,
    min_overshoot_sec: float = 2.0,
    max_overshoot_sec: float = 10.0,
    latency_power: float = 2.0,
) -> Dict[str, Any]:
    """Score based on how quickly the miner finishes after the source ends.

    The allowed overshoot is relative to source duration (clamped to a
    [min, max] range).  Within the allowed window, the score falls off
    as a power curve: score = 1 - (overshoot / allowed)^power.
    Beyond the allowed window, score = 0.
    """
    overshoot = max(0.0, completion_sec - source_duration_sec)

    allowed = source_duration_sec * overshoot_fraction
    allowed = max(min_overshoot_sec, min(max_overshoot_sec, allowed))

    if overshoot <= 0:
        score = 1.0
    elif overshoot >= allowed:
        score = 0.0
    else:
        t = overshoot / allowed
        score = 1.0 - t**latency_power

    return {
        "score": round(score, 6),
        "completion_sec": round(completion_sec, 4),
        "source_duration_sec": round(source_duration_sec, 4),
        "overshoot_sec": round(overshoot, 4),
        "allowed_overshoot_sec": round(allowed, 4),
    }


def _wav_duration_sec(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        frame_count = wav_file.getnframes()
        sample_rate_hz = wav_file.getframerate()
        if sample_rate_hz <= 0:
            return 0.0
        return float(frame_count) / float(sample_rate_hz)


def _completion_sec_from_prediction(
    pred: Dict[str, Any],
    *,
    predicted_duration_sec: float,
) -> float:

    completion_sec: float = float(pred.get("completion_sec", 0.0) or 0.0)
    first_output_frame: int = int(pred.get("first_output_frame", 0) or 0)
    frame_rate_hz: float = float(pred.get("frame_rate_hz", 0.0) or 0.0)
    return (
        max(
            completion_sec,
            (float(first_output_frame) / frame_rate_hz) + predicted_duration_sec,
        )
        if frame_rate_hz > 0
        else predicted_duration_sec
    )


def score_audio_utterance_bytes(
    *,
    predicted_wav_bytes: bytes,
    challenge_uid: str,
    utterance_id: str,
    source_duration_sec: float,
    first_output_frame: int,
    frame_rate_hz: float,
    target_lang: str = "en",
    metadata_root: Optional[Path] = None,
    challenge_metadata: Optional[Dict[str, Any]] = None,
    metadata_source: Optional[str] = None,
    stt_model: Optional[str] = None,
    stt_device: Optional[str] = None,
    embedder_model: Optional[str] = None,
    stt_cache_path: Optional[Path] = None,
) -> Dict[str, Any]:
    settings = get_settings()
    stt_model = stt_model or settings.BB_AUDIO_SCORING_STT_MODEL
    stt_device = stt_device or settings.BB_AUDIO_SCORING_STT_DEVICE
    embedder_model = (
        embedder_model or settings.BB_AUDIO_SCORING_EMBEDDER or DEFAULT_EMBEDDER
    )
    stt_cache_path = stt_cache_path or settings.BB_AUDIO_SCORING_STT_CACHE_PATH

    metadata = resolve_audio_reference_metadata(
        challenge_uid=challenge_uid,
        utterance_id=utterance_id,
        target_lang=target_lang,
        metadata_root=metadata_root,
        challenge_doc=challenge_metadata,
        metadata_source=metadata_source,
    )

    predicted_duration_sec = _wav_duration_sec(predicted_wav_bytes)
    effective_completion_sec = _completion_sec_from_prediction(
        {
            "first_output_frame": first_output_frame,
            "frame_rate_hz": frame_rate_hz,
        },
        predicted_duration_sec=predicted_duration_sec,
    )
    wav_hash = sha256(predicted_wav_bytes).hexdigest()
    try:
        transcript_text, transcript_words, detected_language = transcribe_wav_bytes(
            predicted_wav_bytes,
            wav_hash=wav_hash,
            stt_model=stt_model,
            language=target_lang,
            device=stt_device,
            stt_cache_path=stt_cache_path,
        )
    except Exception as exc:
        return _build_error_score_result(
            score_error=f"{type(exc).__name__}:{exc}",
            source_duration_sec=source_duration_sec,
            predicted_duration_sec=predicted_duration_sec,
            effective_completion_sec=effective_completion_sec,
            reference_text=metadata.reference_text,
        )

    empty_transcript_error = _empty_transcript_score_error(predicted_duration_sec)
    if not transcript_text.strip() and not transcript_words and empty_transcript_error:
        return _build_error_score_result(
            score_error=empty_transcript_error,
            source_duration_sec=source_duration_sec,
            predicted_duration_sec=predicted_duration_sec,
            effective_completion_sec=effective_completion_sec,
            reference_text=metadata.reference_text,
        )

    accuracy = compute_accuracy(
        transcript_text, metadata.reference_text, embedder_model
    )

    # Language gate: if Whisper detected a language different from the target,
    # the miner produced wrong-language audio (e.g. echoing the source).
    # Force accuracy to 0 so the gate fails.
    target_lang_lower = target_lang.lower().strip()
    detected_lang_lower = detected_language.lower().strip()
    if detected_lang_lower and detected_lang_lower != target_lang_lower:
        logger.info(
            "Language gate failed: expected=%s detected=%s — forcing accuracy=0",
            target_lang_lower,
            detected_lang_lower,
        )
        accuracy = 0.0
    speech_rate = compute_speech_rate_penalty(
        transcript_words,
        metadata.reference_wps,
        rate_lower=getattr(
            settings, "BB_AUDIO_SCORING_RATE_LOWER", _DEFAULT_RATE_LOWER
        ),
        rate_upper=getattr(
            settings, "BB_AUDIO_SCORING_RATE_UPPER", _DEFAULT_RATE_UPPER
        ),
    )
    latency = compute_latency_score(
        effective_completion_sec,
        source_duration_sec,
        overshoot_fraction=getattr(
            settings,
            "BB_AUDIO_SCORING_LATENCY_OVERSHOOT_FRACTION",
            _DEFAULT_LATENCY_OVERSHOOT_FRACTION,
        ),
        min_overshoot_sec=getattr(
            settings,
            "BB_AUDIO_SCORING_LATENCY_MIN_OVERSHOOT_SEC",
            _DEFAULT_LATENCY_MIN_OVERSHOOT_SEC,
        ),
        max_overshoot_sec=getattr(
            settings,
            "BB_AUDIO_SCORING_LATENCY_MAX_OVERSHOOT_SEC",
            _DEFAULT_LATENCY_MAX_OVERSHOOT_SEC,
        ),
        latency_power=getattr(
            settings, "BB_AUDIO_SCORING_LATENCY_POWER", _DEFAULT_LATENCY_POWER
        ),
    )

    composite = _compute_composite_score(
        accuracy=accuracy,
        speech_rate_penalty=float(speech_rate["penalty"]),
        latency_score=float(latency["score"]),
        settings=settings,
    )
    accuracy_pass = accuracy >= getattr(
        settings,
        "BB_AUDIO_SCORING_ACCURACY_THRESHOLD",
        _DEFAULT_ACCURACY_THRESHOLD,
    )

    return {
        "score": round(composite, 6),
        "accuracy": round(accuracy, 6),
        "accuracy_pass": accuracy_pass,
        "speech_rate": speech_rate,
        "latency": latency,
        "stt_text": transcript_text,
        "gt_text": metadata.reference_text,
        "predicted_duration_sec": round(predicted_duration_sec, 4),
        "effective_completion_sec": round(effective_completion_sec, 4),
        "source_duration_sec": round(source_duration_sec, 4),
        "score_is_fallback": False,
        "score_method": SEMANTIC_AUDIO_SCORING_MODE,
        "scoring_metadata_source": metadata.metadata_source,
    }


def score_audio_utterance_batch(
    *,
    predictions: List[Dict[str, Any]],
    challenge_uid: str,
    utterance_id: str,
    source_duration_sec: float,
    target_lang: str = "en",
    metadata_root: Optional[Path] = None,
    challenge_metadata: Optional[Dict[str, Any]] = None,
    metadata_source: Optional[str] = None,
    stt_model: Optional[str] = None,
    stt_device: Optional[str] = None,
    embedder_model: Optional[str] = None,
    stt_cache_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    batch_started_at = perf_counter()
    settings = get_settings()
    stt_model = stt_model or settings.BB_AUDIO_SCORING_STT_MODEL
    stt_device = stt_device or settings.BB_AUDIO_SCORING_STT_DEVICE
    embedder_model = (
        embedder_model or settings.BB_AUDIO_SCORING_EMBEDDER or DEFAULT_EMBEDDER
    )
    stt_cache_path = stt_cache_path or settings.BB_AUDIO_SCORING_STT_CACHE_PATH

    metadata = resolve_audio_reference_metadata(
        challenge_uid=challenge_uid,
        utterance_id=utterance_id,
        target_lang=target_lang,
        metadata_root=metadata_root,
        challenge_doc=challenge_metadata,
        metadata_source=metadata_source,
    )

    reference_started_at = perf_counter()
    get_reference_embedding(metadata.reference_text, embedder_model)
    reference_sec = perf_counter() - reference_started_at

    prep_started_at = perf_counter()
    stt_items: List[Dict[str, Any]] = []
    pred_metadata: List[Dict[str, Any]] = []
    total_predicted_duration_sec = 0.0
    for pred in predictions:
        predicted_wav_bytes: bytes = pred["predicted_wav_bytes"]
        predicted_duration_sec = _wav_duration_sec(predicted_wav_bytes)
        total_predicted_duration_sec += predicted_duration_sec
        effective_completion_sec = _completion_sec_from_prediction(
            pred,
            predicted_duration_sec=predicted_duration_sec,
        )
        wav_hash = sha256(predicted_wav_bytes).hexdigest()
        stt_items.append({"wav_bytes": predicted_wav_bytes, "wav_hash": wav_hash})
        pred_metadata.append(
            {
                "predicted_duration_sec": predicted_duration_sec,
                "effective_completion_sec": effective_completion_sec,
            }
        )
    prep_sec = perf_counter() - prep_started_at

    stt_started_at = perf_counter()
    batch_results = transcribe_wav_bytes_batch(
        stt_items,
        stt_model=stt_model,
        language=target_lang,
        device=stt_device,
        stt_cache_path=stt_cache_path,
    )
    stt_sec = perf_counter() - stt_started_at

    transcription_started_at = perf_counter()
    transcriptions: List[Dict[str, Any]] = []
    for i, batch_result in enumerate(batch_results):
        meta = pred_metadata[i]
        transcript_text = str(batch_result.get("text", ""))
        transcript_words = batch_result.get("words", [])
        detected_language = str(batch_result.get("detected_language", ""))
        error = batch_result.get("error")
        transcriptions.append(
            {
                "text": transcript_text,
                "words": transcript_words,
                "detected_language": detected_language,
                "predicted_duration_sec": meta["predicted_duration_sec"],
                "effective_completion_sec": meta["effective_completion_sec"],
                "error": error,
            }
        )
    transcription_sec = perf_counter() - transcription_started_at

    stt_texts = [t["text"] for t in transcriptions]
    accuracy_started_at = perf_counter()
    accuracies = compute_accuracy_batch(
        stt_texts, metadata.reference_text, embedder_model
    )
    # Language gate: if Whisper detected a language different from the target,
    # the miner produced wrong-language audio (e.g. echoing the source).
    # Force accuracy to 0 so the gate fails.
    target_lang_lower = target_lang.lower().strip()
    for i, trans in enumerate(transcriptions):
        detected = trans["detected_language"].lower().strip()
        if detected and detected != target_lang_lower:
            logger.info(
                "Language gate failed: expected=%s detected=%s — forcing accuracy=0",
                target_lang_lower,
                detected,
            )
            accuracies[i] = 0.0
    accuracy_sec = perf_counter() - accuracy_started_at

    assembly_started_at = perf_counter()
    results: List[Dict[str, Any]] = []
    for i, trans in enumerate(transcriptions):
        pred = predictions[i]
        pred_source_duration = pred.get("source_duration_sec", source_duration_sec)
        accuracy = accuracies[i]

        if trans["error"] is not None:
            results.append(
                _build_error_score_result(
                    score_error=trans["error"],
                    source_duration_sec=pred_source_duration,
                    predicted_duration_sec=trans["predicted_duration_sec"],
                    effective_completion_sec=trans["effective_completion_sec"],
                    reference_text=metadata.reference_text,
                )
            )
            continue

        empty_transcript_error = _empty_transcript_score_error(
            trans["predicted_duration_sec"]
        )
        if not trans["text"].strip() and not trans["words"] and empty_transcript_error:
            results.append(
                _build_error_score_result(
                    score_error=empty_transcript_error,
                    source_duration_sec=pred_source_duration,
                    predicted_duration_sec=trans["predicted_duration_sec"],
                    effective_completion_sec=trans["effective_completion_sec"],
                    reference_text=metadata.reference_text,
                )
            )
            continue

        speech_rate = compute_speech_rate_penalty(
            trans["words"],
            metadata.reference_wps,
            rate_lower=getattr(
                settings, "BB_AUDIO_SCORING_RATE_LOWER", _DEFAULT_RATE_LOWER
            ),
            rate_upper=getattr(
                settings, "BB_AUDIO_SCORING_RATE_UPPER", _DEFAULT_RATE_UPPER
            ),
        )
        latency = compute_latency_score(
            trans["effective_completion_sec"],
            pred_source_duration,
            overshoot_fraction=getattr(
                settings,
                "BB_AUDIO_SCORING_LATENCY_OVERSHOOT_FRACTION",
                _DEFAULT_LATENCY_OVERSHOOT_FRACTION,
            ),
            min_overshoot_sec=getattr(
                settings,
                "BB_AUDIO_SCORING_LATENCY_MIN_OVERSHOOT_SEC",
                _DEFAULT_LATENCY_MIN_OVERSHOOT_SEC,
            ),
            max_overshoot_sec=getattr(
                settings,
                "BB_AUDIO_SCORING_LATENCY_MAX_OVERSHOOT_SEC",
                _DEFAULT_LATENCY_MAX_OVERSHOOT_SEC,
            ),
            latency_power=getattr(
                settings, "BB_AUDIO_SCORING_LATENCY_POWER", _DEFAULT_LATENCY_POWER
            ),
        )

        composite = _compute_composite_score(
            accuracy=accuracy,
            speech_rate_penalty=float(speech_rate["penalty"]),
            latency_score=float(latency["score"]),
            settings=settings,
        )
        accuracy_pass = accuracy >= getattr(
            settings,
            "BB_AUDIO_SCORING_ACCURACY_THRESHOLD",
            _DEFAULT_ACCURACY_THRESHOLD,
        )

        results.append(
            {
                "score": round(composite, 6),
                "accuracy": round(accuracy, 6),
                "accuracy_pass": accuracy_pass,
                "speech_rate": speech_rate,
                "latency": latency,
                "stt_text": trans["text"],
                "gt_text": metadata.reference_text,
                "predicted_duration_sec": round(trans["predicted_duration_sec"], 4),
                "effective_completion_sec": round(trans["effective_completion_sec"], 4),
                "source_duration_sec": round(pred_source_duration, 4),
                "score_is_fallback": False,
                "score_method": SEMANTIC_AUDIO_SCORING_MODE,
                "scoring_metadata_source": metadata.metadata_source,
            }
        )

    assembly_sec = perf_counter() - assembly_started_at
    total_sec = perf_counter() - batch_started_at
    logger.info(
        "Audio scoring batch profile: challenge=%s utterance=%s predictions=%d "
        "total_predicted_audio_sec=%.3f prep_sec=%.3f reference_embed_sec=%.3f "
        "stt_sec=%.3f transcription_sec=%.3f accuracy_embed_sec=%.3f "
        "assembly_sec=%.3f total_sec=%.3f stt_model=%s stt_device=%s embedder=%s",
        challenge_uid,
        utterance_id,
        len(predictions),
        total_predicted_duration_sec,
        prep_sec,
        reference_sec,
        stt_sec,
        transcription_sec,
        accuracy_sec,
        assembly_sec,
        total_sec,
        stt_model,
        stt_device,
        embedder_model,
    )
    return results


def _build_error_score_result(
    *,
    score_error: str,
    source_duration_sec: float,
    predicted_duration_sec: float,
    effective_completion_sec: float,
    reference_text: str,
) -> Dict[str, Any]:
    return {
        "score": 0.0,
        "accuracy": 0.0,
        "speech_rate": {"penalty": 0.0, "reason": "scoring_error"},
        "latency": {
            "score": 1.0,
            "completion_sec": round(effective_completion_sec, 4),
            "source_duration_sec": round(source_duration_sec, 4),
            "overshoot_sec": round(
                max(0.0, effective_completion_sec - source_duration_sec), 4
            ),
        },
        "stt_text": "",
        "gt_text": reference_text,
        "predicted_duration_sec": round(predicted_duration_sec, 4),
        "effective_completion_sec": round(effective_completion_sec, 4),
        "source_duration_sec": round(source_duration_sec, 4),
        "score_is_fallback": False,
        "score_method": SEMANTIC_AUDIO_ERROR_MODE,
        "scoring_metadata_source": None,
        "score_error": score_error,
    }


__all__ = [
    "DEFAULT_EMBEDDER",
    "SEMANTIC_AUDIO_SCORING_MODE",
    "compute_accuracy",
    "compute_accuracy_batch",
    "compute_latency_score",
    "compute_speech_rate_penalty",
    "score_audio_utterance_batch",
    "score_audio_utterance_bytes",
]
