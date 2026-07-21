from __future__ import annotations

import base64
import io
import json
import wave

from typing import Any

import pytest

from babelbit.benchmarks import api_challenge as module


def _pcm_wav(*, sample_rate: int, samples: list[int]) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(
            b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples)
        )
    return out.getvalue()


def test_flat_utterance_id_strips_challenge_prefix() -> None:
    assert module.flat_utterance_id("challenge-1:7", fallback_index=0) == "7"
    assert module.flat_utterance_id("3", fallback_index=0) == "3"


def test_decode_ue_audio_bytes_wraps_pcm_payload() -> None:
    wav_bytes = _pcm_wav(sample_rate=12_000, samples=[0, 1000, -1000, 500])
    payload = {
        "audio_b64": base64.b64encode(wav_bytes[44:]).decode("ascii"),
        "sample_rate_hz": 12_000,
        "channels": 1,
        "sample_width_bytes": 2,
    }
    decoded = module.decode_ue_audio_bytes(payload)
    with wave.open(io.BytesIO(decoded), "rb") as wav:
        assert wav.getframerate() == 12_000
        assert wav.getnframes() == 4


def test_resample_wav_to_rate_outputs_24k_mono() -> None:
    source = _pcm_wav(sample_rate=12_000, samples=list(range(120)))
    resampled = module.resample_wav_to_rate(source, target_rate_hz=24_000)
    with wave.open(io.BytesIO(resampled), "rb") as wav:
        assert wav.getframerate() == 24_000
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 240


def test_build_utterance_entry_from_ue_uses_transcription_metadata() -> None:
    wav_bytes = _pcm_wav(sample_rate=24_000, samples=[0, 500, 1000, 1500])
    payload = {
        "challenge_uid": "challenge-1",
        "utterance_id": "challenge-1:0",
        "utterance_index": 0,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
        "sample_rate_hz": 24_000,
        "channels": 1,
        "sample_width_bytes": 2,
    }
    transcription_metadata = {
        "challenge_uid": "challenge-1",
        "utterances": [
            {
                "utterance_id": 0,
                "utterance_translations": [
                    {
                        "language": "en",
                        "text": "hello world",
                        "reference_wps": 2.0,
                        "words": [
                            {"word": "hello", "start": 0.0, "end": 0.4},
                            {"word": "world", "start": 0.4, "end": 0.8},
                        ],
                    }
                ],
            }
        ],
    }
    entry = module.build_utterance_entry_from_ue(
        payload=payload,
        transcription_metadata=transcription_metadata,
        fallback_index=0,
        target_language="en",
        source_text="bonjour le monde",
    )
    assert entry["utterance_id"] == "0"
    assert entry["source_text"] == "bonjour le monde"
    assert entry["utterance_translations"][0]["text"] == "hello world"
    assert isinstance(entry["_source_wav_bytes"], (bytes, bytearray))


def test_fixture_tts_parallel_workers_respects_cap() -> None:
    assert module.fixture_tts_parallel_workers(30) == 6
    assert module.fixture_tts_parallel_workers(3) == 3


def test_prepare_fixtures_writes_each_wav_before_all_complete(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    seen: list[str] = []

    def _fake_build(
        *,
        item: dict[str, Any],
        challenge_uid: str,
        flat_index: int,
        source_language: str,
        target_language: str,
    ) -> dict[str, Any]:
        delay = 0.05 if flat_index % 2 else 0.01
        time.sleep(delay)
        return {
            "utterance_id": str(flat_index),
            "utterance_index": flat_index,
            "dialogue_index": flat_index,
            "dialogue_utterance_index": flat_index,
            "source_text": f"src-{flat_index}",
            "utterance_translations": [{"language": "en", "text": f"ref-{flat_index}"}],
            "_source_wav_bytes": f"wav-{flat_index}".encode("ascii"),
        }

    monkeypatch.setattr(module, "build_utterance_entry_from_dialogue_score", _fake_build)
    out_dir = tmp_path / "fixtures"
    module.prepare_fixtures_from_dialogue_scores(
        challenge_uid="challenge-abc",
        items=[
            {"dialogue_uid": "0", "utterance_number": 0, "ground_truth": "a"},
            {"dialogue_uid": "1", "utterance_number": 1, "ground_truth": "b"},
            {"dialogue_uid": "2", "utterance_number": 2, "ground_truth": "c"},
        ],
        out_dir=out_dir,
        source_language="fr",
        target_language="en",
        parallel_workers=3,
        on_progress=seen.append,
    )
    sample_path = out_dir / "challenge.json"
    assert sample_path.is_file()
    assert (out_dir / "challenge.u1.source.wav").read_bytes() == b"wav-1"
    doc = json.loads(sample_path.read_text(encoding="utf-8"))
    assert len(doc["utterances"]) == 3
    assert len(seen) == 3


def test_dialogue_scores_fetch_limit_uses_full_challenge_default() -> None:
    assert module.dialogue_scores_fetch_limit(None) == 200
    assert module.dialogue_scores_fetch_limit(8) == 50


def test_cache_dir_for_challenge_uses_miner_test_data(tmp_path: Any) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "miner-test-data").mkdir()
    cache_dir = module.cache_dir_for_challenge(workspace, challenge_uid="challenge-abc")
    assert cache_dir == workspace / "miner-test-data" / "api_challenges" / "challenge-abc"


def test_resolve_cached_challenge_path_reads_legacy_location(tmp_path: Any) -> None:
    workspace = tmp_path / "repo"
    legacy = (
        workspace
        / "babelbit_subnet"
        / "benchmark_outputs"
        / "api_challenges"
        / "challenge-abc"
    )
    legacy.mkdir(parents=True)
    (legacy / "challenge.json").write_text('{"challenge_uid":"challenge-abc"}\n', encoding="utf-8")
    cached = module.resolve_cached_challenge_path(workspace, challenge_uid="challenge-abc")
    assert cached == legacy / "challenge.json"


def test_migrate_legacy_api_challenge_fixtures_moves_qualifying_files(tmp_path: Any) -> None:
    workspace = tmp_path / "repo"
    legacy = (
        workspace
        / "babelbit_subnet"
        / "benchmark_outputs"
        / "api_challenges"
        / "challenge-abc"
    )
    legacy.mkdir(parents=True)
    (legacy / "challenge.json").write_text('{"challenge_uid":"challenge-abc"}\n', encoding="utf-8")
    (legacy / "challenge.u0.source.wav").write_bytes(b"RIFF")
    (legacy / "benchmark").mkdir()
    (legacy / "benchmark" / "u0").mkdir()

    reports = module.migrate_legacy_api_challenge_fixtures(workspace, challenge_uid="challenge-abc")
    assert len(reports) == 1
    assert reports[0]["moved_files"] == [
        "stages/qualifying/challenge.json",
        "stages/qualifying/challenge.u0.source.wav",
    ]

    target = workspace / "miner-test-data" / "api_challenges" / "challenge-abc"
    assert (target / "stages" / "qualifying" / "challenge.json").is_file()
    assert (target / "stages" / "qualifying" / "challenge.u0.source.wav").is_file()
    assert not (legacy / "challenge.json").exists()
    assert (legacy / "benchmark" / "u0").is_dir()


def test_fetch_submission_leader_reads_first_item(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "items": [
            {
                "miner_hotkey": "hk-leader",
                "score_overall": 0.77,
                "stage": "qualifying",
            }
        ]
    }
    monkeypatch.setattr(module, "fetch_json", lambda *args, **kwargs: payload)
    leader = module.fetch_submission_leader("challenge-abc", stage="qualifying")
    assert leader is not None
    assert leader["miner_hotkey"] == "hk-leader"


def test_fetch_top_miner_hotkey_reads_submission_leader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        module,
        "fetch_submission_leader",
        lambda *_args, **_kwargs: {"miner_hotkey": "hk-top"},
    )
    assert module.fetch_top_miner_hotkey("challenge-abc") == "hk-top"


def test_get_challenge_uid_reads_main_challenge_uid() -> None:
    assert module.get_challenge_uid({"main_challenge_uid": "challenge-abc"}) == "challenge-abc"


def test_get_latest_challenge_prefers_latest_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "items": [
            {"main_challenge_uid": "challenge-old", "latest_challenge": 0},
            {"main_challenge_uid": "challenge-new", "latest_challenge": 1},
        ]
    }
    monkeypatch.setattr(module, "fetch_json", lambda *args, **kwargs: payload)
    latest = module.get_latest_challenge(status="completed")
    assert module.get_challenge_uid(latest) == "challenge-new"


def test_dedupe_dialogue_score_items_keeps_unique_dialogue_pairs() -> None:
    items = [
        {"dialogue_uid": "0", "utterance_number": 0, "ground_truth": "a"},
        {"dialogue_uid": "0", "utterance_number": 0, "ground_truth": "a"},
        {"dialogue_uid": "1", "utterance_number": 1, "ground_truth": "b"},
    ]
    unique = module.dedupe_dialogue_score_items(items)
    assert len(unique) == 2
    assert unique[0]["ground_truth"] == "a"
    assert unique[1]["ground_truth"] == "b"


def test_extract_dialogue_score_transcript_reads_steps_json() -> None:
    item = {
        "transcript": None,
        "steps": json.dumps({"transcript": "hello from production"}),
    }
    assert module.extract_dialogue_score_transcript(item) == "hello from production"
