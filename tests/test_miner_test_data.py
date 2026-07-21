from __future__ import annotations

import json
from pathlib import Path

import pytest

from babelbit.benchmarks.miner_test_data import (
    ChallengeUtteranceRef,
    build_utterance_entry,
    estimate_min_source_duration_sec,
    is_scorable_utterance,
    iter_en_challenge_utterances,
    list_challenge_utterance_ids,
    load_miner_test_utterance,
    normalize_locale_list,
    summarize_accuracy_results,
    transcript_word_recall,
    validate_source_audio_asr_roundtrip,
    validate_source_audio_duration,
    wav_duration_sec,
)


def test_is_scorable_utterance_filters_sound_effects() -> None:
    assert not is_scorable_utterance("(SOUNDBITE OF LAUGHTER)")
    assert is_scorable_utterance("Two of the nation's automakers are reportedly asking the government.")


def test_iter_en_challenge_utterances_assigns_flat_ids() -> None:
    en_doc = {
        "challenge_uid": "challenge-1",
        "dialogues": [
            {"utterances": ["(SOUNDBITE OF LAUGHTER)", "Hello world from NPR"]},
            {"utterances": ["Another valid sentence here today"]},
        ],
    }
    refs = iter_en_challenge_utterances(en_doc)
    assert [ref.utterance_id for ref in refs] == ["0", "1"]
    assert refs[0].reference_text.startswith("Hello world")
    assert refs[1].dialogue_index == 1


def test_load_miner_test_utterance_by_id_from_original_en(tmp_path: Path) -> None:
    sample = tmp_path / "en-sample.json"
    sample.write_text(
        json.dumps(
            {
                "challenge_uid": "challenge-1",
                "language": "en",
                "dialogues": [
                    {"utterances": ["(SOUNDBITE OF LAUGHTER)", "Hello world from NPR"]}
                ],
            }
        ),
        encoding="utf-8",
    )
    utterance = load_miner_test_utterance(
        sample,
        utterance_id="0",
        require_source_audio=False,
    )
    assert utterance.source_text == "Hello world from NPR"
    assert utterance.reference_text == "Hello world from NPR"


def test_list_challenge_utterance_ids_respects_max(tmp_path: Path) -> None:
    sample = tmp_path / "locale.json"
    sample.write_text(
        json.dumps(
            {
                "challenge_uid": "challenge-1",
                "language": "fr",
                "utterances": [
                    build_utterance_entry(
                        ref=ChallengeUtteranceRef("0", 0, 0, "One two three four"),
                        source_text="un deux trois quatre",
                    ),
                    build_utterance_entry(
                        ref=ChallengeUtteranceRef("1", 0, 1, "Five six seven eight"),
                        source_text="cinq six sept huit",
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    assert list_challenge_utterance_ids(sample, max_utterances=1) == ["0"]


def test_summarize_accuracy_results() -> None:
    summary = summarize_accuracy_results(
        [
            {"validator_score": {"accuracy": 0.9, "accuracy_pass": True, "score": 0.0}},
            {"validator_score": {"accuracy": 0.5, "accuracy_pass": False, "score": 0.0}},
        ]
    )
    assert summary["utterance_count"] == 2
    assert summary["mean_accuracy"] == 0.7
    assert summary["accuracy_pass_rate"] == 0.5
    assert summary["mean_production_score"] == 0.0


def test_normalize_locale_list_dedupes_append_default_bug() -> None:
    assert normalize_locale_list(["fr", "de"]) == ["fr", "de"]
    assert normalize_locale_list(None) == ["en", "fr", "de"]


def test_transcript_word_recall() -> None:
    recall = transcript_word_recall(
        "Two of the nation's automakers are asking the government",
        "Two of the nation's automakers are reportedly asking the government for billions",
    )
    assert recall > 0.7


def test_validate_source_audio_asr_roundtrip() -> None:
    ok, message = validate_source_audio_asr_roundtrip(
        "Bonjour le monde aujourd'hui",
        asr_text="Bonjour monde",
        min_recall=0.5,
    )
    assert ok
    assert message == ""


def test_estimate_min_source_duration_sec() -> None:
    assert estimate_min_source_duration_sec("one two three four five six seven eight") == pytest.approx(
        8 / 3.5
    )


def test_validate_source_audio_duration_flags_short_wav(tmp_path: Path) -> None:
    import wave

    wav_path = tmp_path / "short.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24_000)
        wav.writeframes(b"\x00\x00" * 24_000)
    assert wav_duration_sec(wav_path) == pytest.approx(1.0)
    ok, message = validate_source_audio_duration(
        "one two three four five six seven eight nine ten",
        wav_path,
    )
    assert not ok
    assert "too short" in message
