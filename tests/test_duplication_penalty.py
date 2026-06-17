from types import SimpleNamespace
from unittest.mock import Mock, patch
import io
import wave

import pytest
import torch

from babelbit.scoring.utterance_scoring import (
    apply_pairwise_duplication_penalty,
    score_audio_utterance_batch,
)


def _build_test_wav(frame_count: int = 10) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8)
        wav_file.writeframes(b"\0\0" * frame_count)
    return output.getvalue()


def _final_scores(adjustments):
    return [item["final_score"] for item in adjustments]


def test_duplication_penalty_singleton_response_keeps_raw_score():
    adjustments = apply_pairwise_duplication_penalty(
        raw_scores=[0.8],
        texts=["hello"],
        embedder_model="test-model",
        embeddings=torch.tensor([[1.0, 0.0]]),
    )

    assert adjustments[0]["final_score"] == 0.8
    assert adjustments[0]["duplicate_pressure"] == 1.0
    assert adjustments[0]["penalty"] == 1.0


def test_duplication_penalty_unrelated_responses_below_threshold_have_no_penalty():
    adjustments = apply_pairwise_duplication_penalty(
        raw_scores=[0.9, 0.7],
        texts=["alpha", "beta"],
        embedder_model="test-model",
        embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )

    assert _final_scores(adjustments) == [0.9, 0.7]
    assert [item["duplicate_pressure"] for item in adjustments] == [1.0, 1.0]


def test_duplication_penalty_two_identical_high_quality_responses_are_penalized():
    adjustments = apply_pairwise_duplication_penalty(
        raw_scores=[1.0, 1.0],
        texts=["same", "same"],
        embedder_model="test-model",
        embeddings=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    )

    assert adjustments[0]["duplicate_pressure"] == pytest.approx(2.0)
    assert adjustments[1]["duplicate_pressure"] == pytest.approx(2.0)
    assert adjustments[0]["final_score"] == pytest.approx(1.0 / (2.0**0.5), abs=1e-6)
    assert adjustments[1]["final_score"] == pytest.approx(1.0 / (2.0**0.5), abs=1e-6)


def test_duplication_penalty_ten_identical_high_quality_responses_follow_formula():
    adjustments = apply_pairwise_duplication_penalty(
        raw_scores=[1.0] * 10,
        texts=["same"] * 10,
        embedder_model="test-model",
        embeddings=torch.ones(10, 3),
    )

    for adjustment in adjustments:
        assert adjustment["duplicate_pressure"] == pytest.approx(10.0)
        assert adjustment["penalty"] == pytest.approx(1.0 / (10.0**0.5), abs=1e-6)
        assert adjustment["final_score"] == pytest.approx(1.0 / (10.0**0.5), abs=1e-6)


def test_duplication_penalty_low_quality_garbage_does_not_pressure_good_response():
    adjustments = apply_pairwise_duplication_penalty(
        raw_scores=[1.0, 0.1],
        texts=["good", "garbage"],
        embedder_model="test-model",
        embeddings=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    )

    assert adjustments[0]["duplicate_pressure"] == 1.0
    assert adjustments[0]["final_score"] == 1.0
    assert adjustments[1]["final_score"] < 0.1


def test_duplication_penalty_weaker_duplicate_does_not_pressure_stronger_peer():
    adjustments = apply_pairwise_duplication_penalty(
        raw_scores=[0.9, 0.3],
        texts=["strong", "weak"],
        embedder_model="test-model",
        embeddings=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        min_score_for_pressure=0.0,
        score_epsilon=0.02,
        gamma=1.0,
    )

    assert adjustments[0]["duplicate_pressure"] == 1.0
    assert adjustments[0]["final_score"] == 0.9
    assert adjustments[1]["duplicate_pressure"] == pytest.approx(1.9)
    assert adjustments[1]["final_score"] == pytest.approx(0.3 / 1.9, abs=1e-6)


def test_duplication_penalty_near_tie_within_epsilon_applies_partial_pressure():
    adjustments = apply_pairwise_duplication_penalty(
        raw_scores=[0.666, 0.656],
        texts=["almost same a", "almost same b"],
        embedder_model="test-model",
        embeddings=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        min_score_for_pressure=0.0,
        score_epsilon=0.02,
        gamma=1.0,
    )

    assert adjustments[0]["duplicate_pressure"] == pytest.approx(1.328, abs=1e-6)
    assert adjustments[1]["duplicate_pressure"] == pytest.approx(1.666, abs=1e-6)
    assert adjustments[0]["final_score"] == pytest.approx(0.666 / 1.328, abs=1e-6)
    assert adjustments[1]["final_score"] == pytest.approx(0.656 / 1.666, abs=1e-6)


def test_duplication_penalty_mixed_batch():
    adjustments = apply_pairwise_duplication_penalty(
        raw_scores=[1.0, 0.9, 0.9, 0.1],
        texts=["unique", "duplicate a", "duplicate b", "bad duplicate"],
        embedder_model="test-model",
        embeddings=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
    )

    assert adjustments[0]["final_score"] == pytest.approx(1.0)
    assert adjustments[1]["final_score"] < 0.9
    assert adjustments[2]["final_score"] < 0.9
    assert adjustments[3]["final_score"] < 0.1
    assert adjustments[3]["final_score"] < 0.07


def test_duplication_penalty_numerical_safety():
    assert apply_pairwise_duplication_penalty(
        raw_scores=[], texts=[], embedder_model="test-model", embeddings=torch.empty(0, 2)
    ) == []

    singleton = apply_pairwise_duplication_penalty(
        raw_scores=[float("nan")], texts=["one"], embedder_model="test-model"
    )
    assert singleton[0]["final_score"] == 0.0

    adjustments = apply_pairwise_duplication_penalty(
        raw_scores=[1.0, float("nan"), float("inf")],
        texts=["zero", "nan", "other"],
        embedder_model="test-model",
        embeddings=torch.tensor(
            [[0.0, 0.0], [float("nan"), 1.0], [1.0, 0.0]]
        ),
    )

    assert len(adjustments) == 3
    assert all(item["final_score"] >= 0.0 for item in adjustments)


def test_score_audio_utterance_batch_applies_duplication_adjustment(tmp_path):
    mock_settings = SimpleNamespace(
        BB_AUDIO_SCORING_STT_MODEL="faster-whisper-small",
        BB_AUDIO_SCORING_STT_DEVICE="cpu",
        BB_AUDIO_SCORING_EMBEDDER="test-model",
        BB_AUDIO_SCORING_STT_CACHE_PATH=tmp_path / "stt_cache.jsonl",
        BB_AUDIO_SCORING_ACC_WEIGHT=1.0,
        BB_AUDIO_SCORING_SR_PENALTY_WEIGHT=1.0,
        BB_AUDIO_SCORING_LATENCY_WEIGHT=1.0,
        BB_AUDIO_SCORING_DUPLICATION_SIMILARITY_THRESHOLD=0.88,
        BB_AUDIO_SCORING_DUPLICATION_GAMMA=0.5,
        BB_AUDIO_SCORING_DUPLICATION_MIN_SCORE_FOR_PRESSURE=0.2,
    )
    mock_metadata = SimpleNamespace(
        reference_text="hello world",
        reference_wps=2.0,
        metadata_source="test-metadata",
    )

    with (
        patch("babelbit.scoring.utterance_scoring.get_settings", return_value=mock_settings),
        patch(
            "babelbit.scoring.utterance_scoring.resolve_audio_reference_metadata",
            return_value=mock_metadata,
        ),
        patch("babelbit.scoring.utterance_scoring.get_reference_embedding", return_value=Mock()),
        patch(
            "babelbit.scoring.utterance_scoring.transcribe_wav_bytes_batch",
            return_value=[
                {
                    "text": "hello world",
                    "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
                    "detected_language": "en",
                    "error": None,
                },
                {
                    "text": "hello world",
                    "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
                    "detected_language": "en",
                    "error": None,
                },
            ],
        ),
        patch("babelbit.scoring.utterance_scoring.compute_accuracy_batch", return_value=[0.9, 0.9]),
        patch("babelbit.scoring.utterance_scoring._wav_duration_sec", return_value=1.0),
        patch(
            "babelbit.scoring.utterance_scoring.get_text_embeddings_cached",
            return_value=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        ),
    ):
        scores = score_audio_utterance_batch(
            predictions=[
                {"predicted_wav_bytes": _build_test_wav(), "first_output_frame": 0, "frame_rate_hz": 2.0},
                {"predicted_wav_bytes": _build_test_wav(), "first_output_frame": 0, "frame_rate_hz": 2.0},
            ],
            challenge_uid="challenge-s2s",
            utterance_id="challenge-s2s:0",
            source_duration_sec=1.0,
        )

    assert scores[0]["raw_score"] == 0.9
    assert scores[0]["score"] == pytest.approx(0.9 / (1.9**0.5), abs=1e-6)
    assert scores[0]["duplicate_penalty"]["duplicate_pressure"] == pytest.approx(1.9)
    assert scores[0]["duplicate_penalty"]["raw_score"] == 0.9
    assert scores[0]["duplicate_penalty"]["final_score"] == pytest.approx(
        0.9 / (1.9**0.5), abs=1e-6
    )
    assert scores[0]["duplicate_penalty"]["penalty_factor"] == pytest.approx(
        1.0 / (1.9**0.5), abs=1e-6
    )
    assert scores[0]["duplicate_penalty"]["max_peer_similarity"] == 1.0
    assert scores[0]["duplicate_penalty"]["score_epsilon"] == 0.1
    assert scores[1]["score"] == scores[0]["score"]
