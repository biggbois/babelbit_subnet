from typing import Any

from pydantic import BaseModel, Field


class BBAudioUEUtterance(BaseModel):
    challenge_uid: str
    session_id: str
    utterance_index: int
    utterance_id: str
    language: str
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int
    utterance_frames: int
    end_of_utterance: bool = True
    done: bool = False
    audio_b64: str


class BBAudioMinerInitPayload(BaseModel):
    kind: str = "init"
    challenge_uid: str
    utterance_id: str
    language: str | None = None
    sample_rate_hz: int
    frame_rate_hz: float
    frame_samples: int
    dtype: str
    channels: int


class BBAudioMinerInitResponse(BaseModel):
    ready: bool
    miner_id: str | None = None
    session_id: str
    challenge_uid: str
    utterance_id: str
    language: str | None = None
    sample_rate_hz: int
    frame_rate_hz: float
    frame_samples: int
    dtype: str
    channels: int


class BBAudioMinerPredictPayload(BaseModel):
    kind: str = "predict"
    session_id: str
    audio_b64: str
    in_eos: bool = False


class BBAudioMinerPredictResponse(BaseModel):
    session_id: str
    audio_b64: str = ""
    out_eos: bool = False
    n_bytes: int = 0


class BBAudioUtteranceResult(BaseModel):
    challenge_uid: str
    utterance_index: int
    utterance_id: str
    language: str
    miner_uid: int
    miner_hotkey: str
    miner_session_id: str | None = None
    source_audio_path: str | None = None
    predicted_audio_path: str | None = None
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int
    dtype: str
    frame_rate_hz: float
    frame_samples: int
    frame_count_in: int = 0
    frame_count_out: int = 0
    source_num_bytes: int = 0
    predicted_num_bytes: int = 0
    completed: bool = False
    error: str | None = None
    latency_ms: float = 0.0
    score: float = 0.0
    score_is_fallback: bool = False
    score_method: str = "not_scored"
    accuracy: float = 0.0
    source_duration_sec: float = 0.0
    predicted_duration_sec: float = 0.0
    effective_completion_sec: float = 0.0
    reference_text: str = ""
    transcript_text: str = ""
    scoring_metadata_source: str | None = None
    score_breakdown: dict[str, Any] = Field(default_factory=dict)
    source_audio_bytes: bytes = Field(default=b"", exclude=True)
    predicted_audio_bytes: bytes = Field(default=b"", exclude=True)


class BBAudioChallengeResult(BaseModel):
    challenge_uid: str
    challenge_type: str
    miner_uid: int
    miner_hotkey: str
    utterances: list[BBAudioUtteranceResult] = Field(default_factory=list)
    completed: bool = False
    error: str | None = None
    score: float = 0.0
    protocol: str = "s2s_audio_v1"
    score_is_fallback: bool = False
    score_method: str = "not_scored"
