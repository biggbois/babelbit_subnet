from functools import lru_cache
from os import getenv
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr

__version__ = "0.1.0"


class Settings(BaseModel):
    # Bittensor
    BITTENSOR_WALLET_COLD: str
    BITTENSOR_WALLET_HOT: str
    BITTENSOR_WALLET_PATH: Path
    BITTENSOR_NETWORK: str
    BITTENSOR_SUBTENSOR_ENDPOINT: str
    BITTENSOR_SUBTENSOR_FALLBACK: str

    # Babelbit Core
    BABELBIT_NETUID: int
    BABELBIT_TEMPO: int
    BABELBIT_CACHE_DIR: Path
    BABELBIT_VERSION: str
    BABELBIT_API_TIMEOUT_S: int
    BABELBIT_MAX_CONCURRENT_API_CALLS: int
    BB_MINER_PREDICT_ENDPOINT: str
    BB_MINER_TIMEOUT_SEC: int
    BB_S2S_INIT_TIMEOUT_SEC: float
    BB_S2S_CHUNK_TIMEOUT_SEC: float
    BB_S2S_DRAIN_TIMEOUT_SEC: float
    BB_S2S_FINAL_DRAIN_MIN_TIMEOUT_SEC: float
    BB_S2S_DRAIN_MAX_REQUESTS: int
    BB_UTTERANCE_ENGINE_URL: str
    BB_AUDIO_SCORING_METADATA_ROOT: Optional[Path] = None
    BB_AUDIO_SCORING_STT_MODEL: str
    BB_AUDIO_SCORING_STT_DEVICE: str
    BB_AUDIO_SCORING_EMBEDDER: str
    BB_AUDIO_SCORING_STT_CACHE_PATH: Path
    BB_AUDIO_SCORING_ACCURACY_THRESHOLD: float
    BB_AUDIO_SCORING_RATE_LOWER: float
    BB_AUDIO_SCORING_RATE_UPPER: float
    BB_AUDIO_SCORING_LATENCY_OVERSHOOT_FRACTION: float
    BB_AUDIO_SCORING_LATENCY_MIN_OVERSHOOT_SEC: float
    BB_AUDIO_SCORING_LATENCY_MAX_OVERSHOOT_SEC: float
    BB_AUDIO_SCORING_LATENCY_POWER: float
    BB_RUNNER_ON_STARTUP: bool
    BB_SUBMIT_API_URL: str
    BB_ENABLE_ARENA_CHALLENGE: bool
    BB_ARENA_CADENCE_BLOCKS: int
    BB_ARENA_RUN_ON_STARTUP: bool
    BB_ARENA_GATEWAY_URL: str
    BB_ARENA_CONTAINERS_API_PATH: str
    BB_ARENA_CONTAINERS_STATUS: str
    BB_ARENA_CONTAINERS_WINDOW_SECONDS: int
    BB_ARENA_CONTAINERS_TIMEOUT_SEC: int
    BB_ARENA_ROUTE_READY_TIMEOUT_SEC: float
    BB_ARENA_ROUTE_READY_POLL_SEC: float
    BB_ARENA_RUNSYNC_API_PATH: str
    BB_ARENA_GATEWAY_AUTH_API_PATH: str
    BB_ARENA_GATEWAY_TIMEOUT_SEC: float
    BB_ARENA_INIT_BARRIER_TIMEOUT_SEC: float
    BB_ARENA_INIT_KEEPALIVE_ENABLED: bool
    BB_ARENA_INIT_KEEPALIVE_INTERVAL_SEC: float
    BB_ARENA_STARTUP_UTTERANCE_COUNT: int
    BB_ARENA_MAX_CONSECUTIVE_UTTERANCE_FAILURES: int
    BB_ARENA_MINER_TIMEOUT_SEC: int
    BB_ARENA_INCENTIVE_PERCENT: float = 90.0

    # HuggingFace
    HUGGINGFACE_USERNAME: str
    HUGGINGFACE_API_KEY: SecretStr
    HUGGINGFACE_CONCURRENCY: int

    # Signer
    SIGNER_URL: str
    SIGNER_SEED: SecretStr
    SIGNER_HOST: str
    SIGNER_PORT: int

    # Subtensor Gateway
    SUBTENSOR_GATEWAY_URL: str
    SUBTENSOR_GATEWAY_HOST: str
    SUBTENSOR_GATEWAY_PORT: int
    SUBTENSOR_GATEWAY_TIMEOUT_S: int

    # Database (PostgreSQL)
    PG_HOST: str
    PG_PORT: int
    PG_DB: str
    PG_USER: str
    PG_PASSWORD: SecretStr

    # S3 / Object Storage
    BB_ENABLE_S3_UPLOADS: bool = False
    S3_ENDPOINT_URL: str
    S3_REGION: str
    S3_ACCESS_KEY_ID: str
    S3_SECRET_ACCESS_KEY: SecretStr
    S3_BUCKET_NAME: str
    S3_SUBMISSIONS_DIR: str
    S3_LOG_DIR: str
    S3_ADDRESSING_STYLE: str
    S3_SIGNATURE_VERSION: str
    S3_USE_SSL: bool

    # Miner configuration
    MINER_MODEL_ID: str
    MINER_MODEL_REVISION: Optional[str]
    MINER_AXON_PORT: int
    MINER_DEVICE: str
    MINER_LOAD_IN_8BIT: bool
    MINER_LOAD_IN_4BIT: bool
    MINER_EXTERNAL_IP: Optional[str]

    # Development mode settings
    BB_DEV_MODE: bool = False
    BB_LOCAL_MINER_IP: Optional[str] = None


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        # Bittensor settings
        BITTENSOR_WALLET_COLD=getenv("BITTENSOR_WALLET_COLD", "default"),
        BITTENSOR_WALLET_HOT=getenv("BITTENSOR_WALLET_HOT", "default"),
        BITTENSOR_WALLET_PATH=Path(
            getenv("BITTENSOR_WALLET_PATH", "~/.bittensor/wallets")
        ).expanduser(),
        BITTENSOR_NETWORK=getenv("BITTENSOR_NETWORK", "finney"),
        BITTENSOR_SUBTENSOR_ENDPOINT=getenv("BITTENSOR_SUBTENSOR_ENDPOINT", "finney"),
        BITTENSOR_SUBTENSOR_FALLBACK=getenv(
            "BITTENSOR_SUBTENSOR_FALLBACK", "wss://lite.sub.latent.to:443"
        ),
        # Babelbit core
        BABELBIT_NETUID=int(getenv("BABELBIT_NETUID", "59")),
        # enforcing tempo to 100 blocks to avoid vtrust issues
        BABELBIT_TEMPO=100,
        BABELBIT_CACHE_DIR=Path(getenv("BABELBIT_CACHE_DIR", "~/.babelbit"))
        .expanduser()
        .resolve(),
        BABELBIT_VERSION=getenv("BABELBIT_VERSION", __version__),
        BABELBIT_API_TIMEOUT_S=int(getenv("BABELBIT_API_TIMEOUT_S", "10")),
        BABELBIT_MAX_CONCURRENT_API_CALLS=int(
            getenv("BABELBIT_MAX_CONCURRENT_API_CALLS", "1")
        ),
        BB_MINER_PREDICT_ENDPOINT=getenv("BB_MINER_PREDICT_ENDPOINT", "v1/predict"),
        BB_MINER_TIMEOUT_SEC=int(getenv("BB_MINER_TIMEOUT_SEC", "10")),
        BB_S2S_INIT_TIMEOUT_SEC=float(getenv("BB_S2S_INIT_TIMEOUT_SEC", "600")),
        BB_S2S_CHUNK_TIMEOUT_SEC=float(getenv("BB_S2S_CHUNK_TIMEOUT_SEC", "3")),
        BB_S2S_DRAIN_TIMEOUT_SEC=float(getenv("BB_S2S_DRAIN_TIMEOUT_SEC", "10")),
        BB_S2S_FINAL_DRAIN_MIN_TIMEOUT_SEC=float(
            getenv("BB_S2S_FINAL_DRAIN_MIN_TIMEOUT_SEC", "5")
        ),
        BB_S2S_DRAIN_MAX_REQUESTS=int(getenv("BB_S2S_DRAIN_MAX_REQUESTS", "8")),
        BB_UTTERANCE_ENGINE_URL=getenv(
            "BB_UTTERANCE_ENGINE_URL", "https://api.babelbit.ai"
        ),
        BB_AUDIO_SCORING_METADATA_ROOT=Path(
            getenv("BB_AUDIO_SCORING_METADATA_ROOT", "").strip()
            or str(
                Path(getenv("BABELBIT_CACHE_DIR", "~/.babelbit")).expanduser().resolve()
                / "audio_scoring"
                / "metadata"
            )
        )
        .expanduser()
        .resolve(),
        BB_AUDIO_SCORING_STT_MODEL=getenv(
            "BB_AUDIO_SCORING_STT_MODEL", "faster-whisper-small"
        ),
        BB_AUDIO_SCORING_STT_DEVICE=getenv("BB_AUDIO_SCORING_STT_DEVICE", "cpu"),
        BB_AUDIO_SCORING_EMBEDDER=getenv(
            "BB_AUDIO_SCORING_EMBEDDER", "all-MiniLM-L6-v2"
        ),
        BB_AUDIO_SCORING_STT_CACHE_PATH=Path(
            getenv(
                "BB_AUDIO_SCORING_STT_CACHE_PATH",
                "~/.babelbit/audio_scoring/stt_cache.jsonl",
            )
        )
        .expanduser()
        .resolve(),
        BB_AUDIO_SCORING_ACCURACY_THRESHOLD=float(
            getenv("BB_AUDIO_SCORING_ACCURACY_THRESHOLD", "0.65")
        ),
        BB_AUDIO_SCORING_RATE_LOWER=float(
            getenv("BB_AUDIO_SCORING_RATE_LOWER", "0.3")
        ),
        BB_AUDIO_SCORING_RATE_UPPER=float(
            getenv("BB_AUDIO_SCORING_RATE_UPPER", "1.3")
        ),
        BB_AUDIO_SCORING_LATENCY_OVERSHOOT_FRACTION=float(
            getenv("BB_AUDIO_SCORING_LATENCY_OVERSHOOT_FRACTION", "0.3")
        ),
        BB_AUDIO_SCORING_LATENCY_MIN_OVERSHOOT_SEC=float(
            getenv("BB_AUDIO_SCORING_LATENCY_MIN_OVERSHOOT_SEC", "2.0")
        ),
        BB_AUDIO_SCORING_LATENCY_MAX_OVERSHOOT_SEC=float(
            getenv("BB_AUDIO_SCORING_LATENCY_MAX_OVERSHOOT_SEC", "10.0")
        ),
        BB_AUDIO_SCORING_LATENCY_POWER=float(
            getenv("BB_AUDIO_SCORING_LATENCY_POWER", "2.0")
        ),
        BB_RUNNER_ON_STARTUP=getenv("BB_RUNNER_ON_STARTUP", "false").strip().lower()
        in ("1", "true", "yes"),
        BB_SUBMIT_API_URL=getenv("BB_SUBMIT_API_URL", "https://scoring.babelbit.ai"),
        BB_ENABLE_ARENA_CHALLENGE=getenv("BB_ENABLE_ARENA_CHALLENGE", "true")
        .strip()
        .lower()
        in ("1", "true", "yes"),
        BB_ARENA_CADENCE_BLOCKS=int(getenv("BB_ARENA_CADENCE_BLOCKS", "300")),
        BB_ARENA_RUN_ON_STARTUP=getenv("BB_ARENA_RUN_ON_STARTUP", "false")
        .strip()
        .lower()
        in ("1", "true", "yes"),
        BB_ARENA_GATEWAY_URL=getenv("BB_ARENA_GATEWAY_URL", "https://gw.babelbit.ai/"),
        BB_ARENA_CONTAINERS_API_PATH=getenv(
            "BB_ARENA_CONTAINERS_API_PATH", "/list_arena_miners"
        ),
        BB_ARENA_CONTAINERS_STATUS=getenv("BB_ARENA_CONTAINERS_STATUS", "running"),
        BB_ARENA_CONTAINERS_WINDOW_SECONDS=int(
            getenv("BB_ARENA_CONTAINERS_WINDOW_SECONDS", "86000")
        ),
        BB_ARENA_CONTAINERS_TIMEOUT_SEC=int(
            getenv("BB_ARENA_CONTAINERS_TIMEOUT_SEC", "10")
        ),
        BB_ARENA_ROUTE_READY_TIMEOUT_SEC=float(
            getenv("BB_ARENA_ROUTE_READY_TIMEOUT_SEC", "300")
        ),
        BB_ARENA_ROUTE_READY_POLL_SEC=float(
            getenv("BB_ARENA_ROUTE_READY_POLL_SEC", "10")
        ),
        BB_ARENA_RUNSYNC_API_PATH=getenv("BB_ARENA_RUNSYNC_API_PATH", "/runsync"),
        BB_ARENA_GATEWAY_AUTH_API_PATH=getenv(
            "BB_ARENA_GATEWAY_AUTH_API_PATH", "/auth/token"
        ),
        BB_ARENA_GATEWAY_TIMEOUT_SEC=float(
            getenv("BB_ARENA_GATEWAY_TIMEOUT_SEC", "300")
        ),
        BB_ARENA_INIT_BARRIER_TIMEOUT_SEC=float(
            getenv("BB_ARENA_INIT_BARRIER_TIMEOUT_SEC", "600")
        ),
        BB_ARENA_INIT_KEEPALIVE_ENABLED=getenv(
            "BB_ARENA_INIT_KEEPALIVE_ENABLED", "true"
        )
        .strip()
        .lower()
        not in {"0", "false", "no", "off"},
        BB_ARENA_INIT_KEEPALIVE_INTERVAL_SEC=float(
            getenv("BB_ARENA_INIT_KEEPALIVE_INTERVAL_SEC", "30")
        ),
        BB_ARENA_STARTUP_UTTERANCE_COUNT=int(
            getenv("BB_ARENA_STARTUP_UTTERANCE_COUNT", "3")
        ),
        BB_ARENA_MAX_CONSECUTIVE_UTTERANCE_FAILURES=int(
            getenv("BB_ARENA_MAX_CONSECUTIVE_UTTERANCE_FAILURES", "2")
        ),
        BB_ARENA_MINER_TIMEOUT_SEC=int(getenv("BB_ARENA_MINER_TIMEOUT_SEC", "10")),
        BB_ARENA_INCENTIVE_PERCENT=float(getenv("BB_ARENA_INCENTIVE_PERCENT", "90")),
        # Development / local testing flags
        BB_DEV_MODE=getenv("BB_DEV_MODE", "0").lower() in ("1", "true", "yes"),
        BB_LOCAL_MINER_IP=getenv("BB_LOCAL_MINER_IP", ""),
        # HuggingFace settings
        HUGGINGFACE_USERNAME=getenv("HUGGINGFACE_USERNAME", ""),
        HUGGINGFACE_API_KEY=SecretStr(getenv("HUGGINGFACE_API_KEY", "")),
        HUGGINGFACE_CONCURRENCY=int(getenv("HUGGINGFACE_CONCURRENCY", "2")),
        # Signer settings
        SIGNER_URL=getenv("SIGNER_URL", "http://signer:8080"),
        SIGNER_SEED=SecretStr(getenv("SIGNER_SEED", "")),
        SIGNER_HOST=getenv("SIGNER_HOST", "127.0.0.1"),
        SIGNER_PORT=int(getenv("SIGNER_PORT", "8080")),
        SUBTENSOR_GATEWAY_URL=getenv(
            "SUBTENSOR_GATEWAY_URL", "http://subtensor-gateway:8090"
        ),
        SUBTENSOR_GATEWAY_HOST=getenv("SUBTENSOR_GATEWAY_HOST", "0.0.0.0"),
        SUBTENSOR_GATEWAY_PORT=int(getenv("SUBTENSOR_GATEWAY_PORT", "8090")),
        SUBTENSOR_GATEWAY_TIMEOUT_S=int(getenv("SUBTENSOR_GATEWAY_TIMEOUT_S", "300")),
        # Database settings
        PG_HOST=getenv("PG_HOST", "db"),
        PG_PORT=int(getenv("PG_PORT", "5432")),
        PG_DB=getenv("PG_DB", "babelbit"),
        PG_USER=getenv("PG_USER", "babelbit"),
        PG_PASSWORD=SecretStr(getenv("PG_PASSWORD", "babelbit")),
        # S3 / Object Storage settings
        S3_ENDPOINT_URL=getenv("S3_ENDPOINT_URL", ""),
        S3_REGION=getenv("S3_REGION", "us-east-1"),
        S3_ACCESS_KEY_ID=getenv("S3_ACCESS_KEY_ID", ""),
        S3_SECRET_ACCESS_KEY=SecretStr(getenv("S3_SECRET_ACCESS_KEY", "")),
        S3_BUCKET_NAME=getenv("S3_BUCKET_NAME", ""),
        S3_SUBMISSIONS_DIR=getenv("S3_SUBMISSIONS_DIR", "challenges"),
        S3_LOG_DIR=getenv("S3_LOG_DIR", "logs"),
        S3_ADDRESSING_STYLE=getenv("S3_ADDRESSING_STYLE", "path"),
        S3_SIGNATURE_VERSION=getenv("S3_SIGNATURE_VERSION", "s3v4"),
        S3_USE_SSL=getenv("S3_USE_SSL", "true").lower() in ("true", "1", "yes"),
        # Miner configuration
        MINER_MODEL_ID=getenv("MINER_MODEL_ID", "babelbit-ai/base-miner"),
        MINER_MODEL_REVISION=getenv("MINER_MODEL_REVISION"),
        MINER_AXON_PORT=int(getenv("MINER_AXON_PORT", "8092")),
        MINER_DEVICE=getenv("MINER_DEVICE", "cpu"),
        MINER_LOAD_IN_8BIT=getenv("MINER_LOAD_IN_8BIT", "0").lower()
        in {"1", "true", "yes"},
        MINER_LOAD_IN_4BIT=getenv("MINER_LOAD_IN_4BIT", "0").lower()
        in {"1", "true", "yes"},
        MINER_EXTERNAL_IP=getenv("MINER_EXTERNAL_IP"),
    )
