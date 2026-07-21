from __future__ import annotations

from babelbit.benchmarks.s2s_client import (
    S2sClientConfig,
    drain_deadline_sec,
    format_chunk_timeout_error,
    frame_response_timeout_sec,
    run_s2s_concurrent_validators,
    run_s2s_utterance,
    S2sUtteranceRequest,
)


def test_frame_response_timeout_uses_chunk_except_final_frame() -> None:
    cfg = S2sClientConfig(chunk_timeout_sec=3.0, drain_timeout_sec=10.0)
    assert frame_response_timeout_sec(frame_index=0, total_frames=10, config=cfg) == 3.0
    assert frame_response_timeout_sec(frame_index=9, total_frames=10, config=cfg) == 10.0


def test_drain_deadline_uses_max_of_global_and_min_timeout() -> None:
    cfg = S2sClientConfig(drain_timeout_sec=10.0, final_drain_min_timeout_sec=5.0)
    assert drain_deadline_sec(
        last_input_sent_at=100.0,
        drain_started_at=100.0,
        config=cfg,
    ) == 110.0
    assert drain_deadline_sec(
        last_input_sent_at=100.0,
        drain_started_at=109.0,
        config=cfg,
    ) == 114.0


def test_format_chunk_timeout_error_matches_validator_label() -> None:
    msg = format_chunk_timeout_error(frame_index=22, total_frames=174, timeout_sec=3.0)
    assert msg == (
        "AudioChallengeError:audio chunk response for frame 23/174 timed out after 3.00s"
    )


def test_run_s2s_utterance_times_out_on_slow_frame(monkeypatch) -> None:
    import httpx

    class _FakeResponse:
        def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code
            self.text = ""

        def json(self) -> dict[str, object]:
            return self._payload

    class _FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, url: str, json: dict[str, object], timeout: httpx.Timeout) -> _FakeResponse:
            self.calls += 1
            if json.get("kind") == "init":
                return _FakeResponse({"session_id": "sess-1"})
            if self.calls == 2:
                raise httpx.TimeoutException("slow")
            return _FakeResponse({"audio_b64": "", "out_eos": False, "n_bytes": 0})

        def close(self) -> None:
            return None

    fake = _FakeClient()
    result = run_s2s_utterance(
        "http://miner.test",
        S2sUtteranceRequest(
            frames=[b"x" * 4, b"y" * 4],
            challenge_uid="challenge-1",
            utterance_id="0",
        ),
        config=S2sClientConfig(chunk_timeout_sec=3.0),
        client=fake,  # type: ignore[arg-type]
    )
    assert result.prediction_error is not None
    assert "timed out after 3.00s" in result.prediction_error


def test_pace_realtime_sleeps_between_frames(monkeypatch) -> None:
    import httpx

    sleeps: list[float] = []

    class _FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload
            self.status_code = 200
            self.text = ""

        def json(self) -> dict[str, object]:
            return self._payload

    class _FakeClient:
        def post(self, url: str, json: dict[str, object], timeout: httpx.Timeout) -> _FakeResponse:
            if json.get("kind") == "init":
                return _FakeResponse({"session_id": "sess-1"})
            return _FakeResponse({"audio_b64": "", "out_eos": bool(json.get("in_eos")), "n_bytes": 0})

        def close(self) -> None:
            return None

    monkeypatch.setattr("babelbit.benchmarks.s2s_client.time.sleep", lambda s: sleeps.append(s))
    run_s2s_utterance(
        "http://miner.test",
        S2sUtteranceRequest(
            frames=[b"a" * 4, b"b" * 4, b"c" * 4],
            challenge_uid="challenge-1",
            utterance_id="0",
        ),
        config=S2sClientConfig(pace_realtime=True, chunk_timeout_sec=3.0),
        client=_FakeClient(),  # type: ignore[arg-type]
    )
    # Frame 0: no wait; frames 1..n-1 target cadence ≈ 0.08s
    assert len(sleeps) >= 1
    assert all(s > 0 for s in sleeps)
    assert result.timed_out_frames == [1]


def test_run_s2s_concurrent_validators_runs_all_sessions(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_run(
        base_url: str,
        request: S2sUtteranceRequest,
        *,
        config: S2sClientConfig | None = None,
        client: object | None = None,
    ):
        calls.append(request.validator_id)
        from babelbit.benchmarks.s2s_client import S2sUtteranceResult

        return S2sUtteranceResult(
            validator_id=request.validator_id,
            challenge_uid=request.challenge_uid,
            utterance_id=request.utterance_id,
            out_eos=True,
        )

    monkeypatch.setattr("babelbit.benchmarks.s2s_client.run_s2s_utterance", _fake_run)
    results = run_s2s_concurrent_validators(
        "http://miner.test",
        [
            S2sUtteranceRequest(frames=[b"a"], challenge_uid="c1", utterance_id="0", validator_id="v0"),
            S2sUtteranceRequest(frames=[b"b"], challenge_uid="c2", utterance_id="1", validator_id="v1"),
            S2sUtteranceRequest(frames=[b"c"], challenge_uid="c3", utterance_id="2", validator_id="v2"),
        ],
        max_workers=3,
    )
    assert len(results) == 3
    assert sorted(calls) == ["v0", "v1", "v2"]
