"""Validator-aligned benchmark using exact _playback_completion_sec logic.

This mirrors the real validator's predict_audio.py flow:
- _playback_completion_sec computes serial playback of output chunks
- completion_sec = max(playback_end, eos_sec)
- latency = 1 - (overshoot / allowed)^2
"""
from __future__ import annotations
import base64, io, json, struct, sys, time, wave, os
import httpx, numpy as np, torch
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModel

SR = 24000; FS = 1920; FR = 12.5
SAMPLE_WIDTH = 4  # float32le = 4 bytes
BYTES_PER_SEC = SR * 1 * SAMPLE_WIDTH

CHALLENGE_DIR = os.environ.get("CHALLENGE_DIR",
    "/root/miner-test-data/api_challenges/challenge-1783732211-def747a8")

def wav_to_frames(path):
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    samples = struct.unpack("<" + "h" * (len(raw)//2), raw)
    data = b"".join(struct.pack("<f", max(-1.0, min(1.0, s/32768.0))) for s in samples)
    bpf = FS * SAMPLE_WIDTH
    frames = [data[i:i+bpf] for i in range(0, len(data), bpf)]
    if frames and len(frames[-1]) < bpf:
        frames[-1] += b"\x00" * (bpf - len(frames[-1]))
    return frames, len(samples) / SR

def run_miner(base_url, frames, challenge_uid, utterance_id):
    """Collect ALL output chunks with their arrival frame indices.
    Returns: (total_pcm, chunk_info_list)
    chunk_info_list: [(arrival_frame, chunk_bytes), ...]
    """
    predict_url = base_url.rstrip("/") + "/v1/predict"
    all_raw = bytearray()
    chunk_info = []
    first_output_frame = None
    
    with httpx.Client(timeout=240) as cli:
        init = cli.post(predict_url, json={
            "kind": "init", "challenge_uid": challenge_uid, "utterance_id": utterance_id,
            "sample_rate_hz": SR, "frame_rate_hz": FR,
            "frame_samples": FS, "dtype": "float32le", "channels": 1
        })
        sid = init.json()["session_id"]
        
        for idx in range(len(frames)):
            is_eos = idx == len(frames) - 1
            resp = cli.post(predict_url, json={
                "kind": "predict", "session_id": sid,
                "audio_b64": base64.b64encode(frames[idx]).decode(),
                "in_eos": is_eos
            })
            r = resp.json()
            raw_chunk = base64.b64decode(r.get("audio_b64") or "")
            n_bytes = len(raw_chunk)
            if n_bytes > 0:
                if first_output_frame is None:
                    first_output_frame = idx + 1
                all_raw.extend(raw_chunk)
                chunk_info.append((idx + 1, raw_chunk))
            if r.get("out_eos"):
                break
        
        # Drain remaining chunks
        if not r.get("out_eos"):
            for _ in range(24):
                resp = cli.post(predict_url, json={
                    "kind": "predict", "session_id": sid, "audio_b64": "", "in_eos": True
                })
                r = resp.json()
                raw_chunk = base64.b64decode(r.get("audio_b64") or "")
                if len(raw_chunk) > 0:
                    all_raw.extend(raw_chunk)
                    chunk_info.append((len(frames), raw_chunk))
                if r.get("out_eos"):
                    break
    
    return bytes(all_raw), first_output_frame, chunk_info, len(frames)

def pcm_to_wav(raw_float32le):
    """Convert float32le raw PCM to 16-bit WAV bytes."""
    vals = struct.unpack("<" + "f" * (len(raw_float32le)//4), raw_float32le)
    pcm = bytearray()
    for v in vals:
        pcm.extend(int(max(-1.0, min(1.0, v)) * 32767).to_bytes(2, "little", signed=True))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(bytes(pcm))
    return buf.getvalue()

def playback_completion_sec(chunk_info, total_frames):
    """Validator's _playback_completion_sec logic.
    Each chunk starts playing no earlier than its arrival frame,
    and chunks play back serially.
    """
    if BYTES_PER_SEC <= 0:
        return float(total_frames) / FR
    
    eos_sec = float(total_frames) / FR
    playback_end = 0.0
    for arrival_frame, chunk_bytes in chunk_info:
        arrival_sec = float(arrival_frame) / FR
        chunk_dur = float(len(chunk_bytes)) / BYTES_PER_SEC
        playback_end = max(playback_end, arrival_sec) + chunk_dur
    
    return max(playback_end, eos_sec)

def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    
    challenge_json = CHALLENGE_DIR + "/stages/qualifying/challenge.json"
    with open(challenge_json) as f:
        challenge_doc = json.load(f)
    
    challenge_uid = challenge_doc.get("challenge_uid", "unknown")
    utterances = challenge_doc.get("utterances", [])[:8]
    
    print("Loading scoring models...")
    whisper = WhisperModel("small", device="cpu", compute_type="int8")
    embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
    embed_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    def embed(text):
        inp = embed_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            out = embed_model(**inp)
        attn = inp["attention_mask"]
        emb = out.last_hidden_state
        mask = attn.unsqueeze(-1).expand(emb.size()).float()
        pooled = (emb * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)[0]
    
    ref_data = []
    for u in utterances:
        uid = str(u.get("utterance_id", u.get("utterance_index", 0)))
        ref_text = u.get("utterance_translations", [{}])[0].get("text", "")
        ref_wps = u.get("utterance_translations", [{}])[0].get("reference_wps", 4.0)
        ref_vec = embed(ref_text).numpy()
        ref_data.append({"uid": uid, "ref_text": ref_text, "ref_wps": ref_wps, "ref_embedding": ref_vec})
    
    results = []; passed = 0
    
    for i, u in enumerate(utterances):
        rd = ref_data[i]
        uid = rd["uid"]
        wav_path = CHALLENGE_DIR + "/stages/qualifying/challenge.u" + uid + ".source.wav"
        
        if not os.path.exists(wav_path):
            print(f"u{uid}: no audio")
            continue
        
        frames, source_dur = wav_to_frames(wav_path)
        ref_count = len(rd["ref_text"].split())
        
        print(f"\n=== u{uid} ({ref_count}w ref, src={source_dur:.1f}s) ===")
        
        t0 = time.perf_counter()
        out_raw, first_out, chunk_info, total_frames = run_miner(base_url, frames, challenge_uid, uid)
        wall = time.perf_counter() - t0
        
        if not out_raw:
            print("  FAIL: no output")
            results.append({"uid": uid, "accuracy": 0, "score": 0, "passed": False})
            continue
        
        # Convert miner output to actual WAV for STT
        predicted_wav = pcm_to_wav(out_raw)
        
        # STT (validator-aligned)
        segs, info = whisper.transcribe(io.BytesIO(predicted_wav), language=None, word_timestamps=True)
        stt_text = " ".join(s.text.strip() for s in segs)
        detected_lang = getattr(info, "language", "")
        
        # Accuracy
        stt_vec = embed(stt_text)
        ref_vec = torch.as_tensor(rd["ref_embedding"], dtype=torch.float32)
        acc = max(0.0, min(1.0, torch.dot(stt_vec, ref_vec).item()))
        
        # Speech rate
        n_words = len(stt_text.split())
        wav_dur = len(predicted_wav) / (SR * 2)  # 16-bit PCM, 2 bytes/sample
        miner_wps = n_words / wav_dur if wav_dur > 0 else 0
        rate_ratio = miner_wps / rd["ref_wps"] if rd["ref_wps"] > 0 else 0
        rate_ok = 0.3 <= rate_ratio <= 1.3
        
        # Latency (validator-aligned _playback_completion_sec)
        completion = playback_completion_sec(chunk_info, total_frames)
        overshoot = max(0, completion - source_dur)
        allowed = min(10.0, max(2.0, source_dur * 0.6))
        lat_score = max(0, 1 - (overshoot / allowed) ** 2)
        
        # Final score
        score = lat_score if (acc >= 0.65 and rate_ok) else 0.0
        if score > 0:
            passed += 1
        status = "PASS" if score > 0 else "FAIL"
        
        print(f"  acc={acc:.3f} rate={rate_ratio:.3f} over={overshoot:.1f}s lat={lat_score:.3f} score={score:.3f} wall={wall:.1f}s lang={detected_lang} {status}")
        print(f"  chunks: {len(chunk_info)}, first_out_frame={first_out}, total_frames={total_frames}")
        print(f"  completion={completion:.1f}s, wav_dur={wav_dur:.1f}s")
        print(f"  ref: {rd['ref_text'][:100]}")
        print(f"  stt: {stt_text[:100]}")
        
        results.append({"uid": uid, "accuracy": acc, "score": score, "passed": score > 0})
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/8 passed")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  u{r['uid']}: acc={r['accuracy']:.3f} score={r['score']:.3f} {status}")

if __name__ == "__main__":
    main()
