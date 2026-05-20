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
    finally:
        get_settings.cache_clear()


def test_s2s_drain_max_requests_defaults_to_eight(monkeypatch):
    monkeypatch.delenv("BB_S2S_DRAIN_MAX_REQUESTS", raising=False)
    monkeypatch.delenv("BB_S2S_INIT_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("BB_S2S_CHUNK_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("BB_S2S_DRAIN_TIMEOUT_SEC", raising=False)
    get_settings.cache_clear()

    try:
        settings = get_settings()
        assert settings.BB_S2S_DRAIN_MAX_REQUESTS == 8
        assert settings.BB_S2S_INIT_TIMEOUT_SEC == 60.0
        assert settings.BB_S2S_CHUNK_TIMEOUT_SEC == 3.0
        assert settings.BB_S2S_DRAIN_TIMEOUT_SEC == 10.0
    finally:
        get_settings.cache_clear()
