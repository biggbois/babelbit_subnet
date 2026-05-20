# Validator Scoring Stress Benchmarks

These scripts are manual benchmarks, not CI tests. They are intended to challenge
validator CPU assumptions at larger miner counts.

## Audio Scoring

Run the synthetic scoring-loop benchmark:

```bash
uv run python tests/benchmarks/stress_validator_scoring.py --miners 50 100 200 --repeat 3 --warmup 1
```

Synthetic mode patches STT and embedding calls. It measures per-miner scoring
bookkeeping, WAV duration parsing, hashing, speech-rate scoring, latency scoring,
and result construction.

Run the real CPU scoring benchmark:

```bash
uv run python tests/benchmarks/stress_validator_scoring.py --mode real --miners 50 100 200 --repeat 1 --warmup 0 --stt-device cpu
```

Real mode loads the configured faster-whisper and sentence-transformer models and
therefore measures the expensive validator CPU path. Use it on the same host class
as production validators; first-run model download and cache warmup can dominate
the initial result.

Key columns:

- `wall_sec_med`: median elapsed seconds for one scoring batch.
- `cpu_sec_med`: median process CPU seconds for one scoring batch.
- `cpu/wall_med`: values near 1 mean single-core saturation; higher values mean
  native libraries used multiple CPU threads.
- `miners/wall_sec_med`: miner scoring throughput from the validator's point of view.
- `fallback`: number of miners that returned scoring errors.
- `max_rss_mb_med`: peak resident memory observed by the process.

## STT Only

Run a small CPU STT throughput benchmark:

```bash
uv run python tests/benchmarks/stress_stt.py --miners 10 25 50 --durations-sec 2 --audio-kind mixed --stt-device cpu
```

Run duration sweeps:

```bash
uv run python tests/benchmarks/stress_stt.py --miners 10 --durations-sec 2 5 10 --audio-kind mixed --stt-device cpu
```

Use real miner WAV files when available:

```bash
uv run python tests/benchmarks/stress_stt.py --wav-dir /path/to/wavs --miners 50 100 200 --stt-device cpu
```

The most useful STT column is `stt_x_realtime`. A value of `3.0` means the STT
model spent about 3 seconds for every 1 second of audio. For 200 miners each
submitting 10 seconds of audio, that would estimate to roughly 200 * 10 * 3 =
6,000 seconds of serial STT work before accounting for model threading.

Try explicit CPU thread counts:

```bash
uv run python tests/benchmarks/stress_stt.py --miners 10 --durations-sec 2 --audio-kind mixed --cpu-threads 1
uv run python tests/benchmarks/stress_stt.py --miners 10 --durations-sec 2 --audio-kind mixed --cpu-threads 2
uv run python tests/benchmarks/stress_stt.py --miners 10 --durations-sec 2 --audio-kind mixed --cpu-threads 4
```
