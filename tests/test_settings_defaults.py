from pathlib import Path

from babelbit.utils.settings import get_settings


def test_audio_scoring_metadata_root_defaults_to_cache_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("BB_AUDIO_SCORING_METADATA_ROOT", raising=False)
    monkeypatch.setenv("BABELBIT_CACHE_DIR", str(tmp_path / ".babelbit"))
    get_settings.cache_clear()

    try:
        settings = get_settings()
        assert settings.BB_AUDIO_SCORING_METADATA_ROOT == (
            Path(tmp_path / ".babelbit" / "audio_scoring" / "metadata")
            .expanduser()
            .resolve()
        )
        assert settings.BB_AUDIO_SCORING_DUPLICATION_SIMILARITY_THRESHOLD == 0.88
        assert settings.BB_AUDIO_SCORING_DUPLICATION_GAMMA == 0.5
        assert settings.BB_AUDIO_SCORING_DUPLICATION_MIN_SCORE_FOR_PRESSURE == 0.0
        assert settings.BB_AUDIO_SCORING_DUPLICATION_SCORE_EPSILON == 0.02
    finally:
        get_settings.cache_clear()


def test_s2s_drain_max_requests_defaults_to_eight(monkeypatch):
    monkeypatch.delenv("BB_S2S_DRAIN_MAX_REQUESTS", raising=False)
    monkeypatch.delenv("BB_S2S_INIT_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("BB_ARENA_INIT_BARRIER_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("BB_S2S_CHUNK_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("BB_S2S_DRAIN_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("BB_S2S_FINAL_DRAIN_MIN_TIMEOUT_SEC", raising=False)
    get_settings.cache_clear()

    try:
        settings = get_settings()
        assert settings.BB_S2S_DRAIN_MAX_REQUESTS == 8
        assert settings.BB_S2S_INIT_TIMEOUT_SEC == 600.0
        assert settings.BB_ARENA_INIT_BARRIER_TIMEOUT_SEC == 600.0
        assert settings.BB_S2S_CHUNK_TIMEOUT_SEC == 3.0
        assert settings.BB_S2S_DRAIN_TIMEOUT_SEC == 10.0
        assert settings.BB_S2S_FINAL_DRAIN_MIN_TIMEOUT_SEC == 5.0
    finally:
        get_settings.cache_clear()
