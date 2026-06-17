from __future__ import annotations

import os
from hashlib import sha256
from typing import Any, Dict, List, Optional

import torch

from babelbit.utils.hf_runtime import ensure_hf_transfer_available

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_EMBEDDER_CACHE: Dict[str, Any] = {}
_REF_EMBED_CACHE: Dict[str, torch.Tensor] = {}
_TEXT_EMBED_CACHE: Dict[str, torch.Tensor] = {}


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_embedder(model_name: str, device: Optional[str] = None):
    effective_device = device or _detect_device()
    cache_key = f"{model_name}:{effective_device}"
    if cache_key not in _EMBEDDER_CACHE:
        ensure_hf_transfer_available()
        from sentence_transformers import SentenceTransformer

        _EMBEDDER_CACHE[cache_key] = SentenceTransformer(
            model_name, device=effective_device
        )
    return _EMBEDDER_CACHE[cache_key]


def embed_text(
    text: str, model_name: str, device: Optional[str] = None
) -> torch.Tensor:
    embedder = _get_embedder(model_name, device=device)
    return embedder.encode(
        [text],
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=False,
    )[0]


def embed_texts_batch(
    texts: List[str], model_name: str, device: Optional[str] = None
) -> torch.Tensor:
    if not texts:
        return torch.empty(0, dtype=torch.float32)
    embedder = _get_embedder(model_name, device=device)
    return embedder.encode(
        texts,
        normalize_embeddings=True,
        convert_to_tensor=True,
        batch_size=len(texts),
        show_progress_bar=False,
    )


def get_text_embeddings_cached(
    texts: List[str], model_name: str, device: Optional[str] = None
) -> torch.Tensor:
    if not texts:
        return torch.empty(0, dtype=torch.float32)

    keys = [f"{model_name}:{sha256(text.encode('utf-8')).hexdigest()}" for text in texts]
    missing_indices = [i for i, key in enumerate(keys) if key not in _TEXT_EMBED_CACHE]
    if missing_indices:
        missing_texts = [texts[i] for i in missing_indices]
        embeddings = embed_texts_batch(missing_texts, model_name, device=device)
        for batch_idx, original_idx in enumerate(missing_indices):
            _TEXT_EMBED_CACHE[keys[original_idx]] = embeddings[batch_idx].detach().cpu()

    return torch.stack([_TEXT_EMBED_CACHE[key] for key in keys])


def cosine_similarity(vector_a: torch.Tensor, vector_b: torch.Tensor) -> float:
    return float(torch.dot(vector_a, vector_b))


def get_reference_embedding(
    reference_text: str, model_name: str, device: Optional[str] = None
) -> torch.Tensor:
    key = f"{model_name}:{reference_text}"
    if key not in _REF_EMBED_CACHE:
        _REF_EMBED_CACHE[key] = embed_text(reference_text, model_name, device=device)
    return _REF_EMBED_CACHE[key]


def clear_reference_cache() -> None:
    _REF_EMBED_CACHE.clear()


def clear_text_embedding_cache() -> None:
    _TEXT_EMBED_CACHE.clear()


__all__ = [
    "clear_reference_cache",
    "clear_text_embedding_cache",
    "cosine_similarity",
    "embed_text",
    "embed_texts_batch",
    "get_reference_embedding",
    "get_text_embeddings_cached",
]
