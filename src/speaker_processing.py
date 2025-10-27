"""Utilities for optional speaker enrolment and identification inside the worker.

The RunPod handler only calls three public functions:
    * load_known_speakers_from_samples
    * identify_speakers_on_segments
    * relabel_speakers_by_avg_similarity

Everything else here exists to support those helpers while keeping heavy
dependencies (pyannote, librosa) lazily initialised and reused across calls.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np
import requests
import torch
from scipy.spatial.distance import cdist

try:
    from pyannote.audio import Inference
    from pyannote.core import SlidingWindowFeature
except Exception as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "pyannote.audio is required for speaker enrolment; ensure it is installed in the worker image."
    ) from exc

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logger = logging.getLogger("speaker_processing")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Globals / caches
# --------------------------------------------------------------------------- #

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_EMBEDDER: Optional[Inference] = None
_EMBEDDER_TOKEN: Optional[str] = None

# Cache embeddings by (name, url, file_path) tuple so repeated runs are cheap
_KNOWN_EMBEDDINGS: Dict[Tuple[str, Optional[str], Optional[str]], np.ndarray] = {}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _resolve_token(token: Optional[str]) -> Optional[str]:
    """Prefer explicit token, then environment fallbacks, else None."""
    if token:
        return token
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or None


def _ensure_embedder(token: Optional[str]) -> Optional[Inference]:
    """Return a cached pyannote embedder, re-initialising if token changes."""
    global _EMBEDDER, _EMBEDDER_TOKEN
    wanted = _resolve_token(token)
    if _EMBEDDER is not None and _EMBEDDER_TOKEN == wanted:
        return _EMBEDDER
    try:
        _EMBEDDER = Inference("pyannote/embedding", device=_DEVICE, use_auth_token=wanted)
        _EMBEDDER_TOKEN = wanted
        logger.debug("Initialised pyannote embedder (token provided=%s)", bool(wanted))
        return _EMBEDDER
    except Exception as exc:
        logger.error("Failed to initialise pyannote embedder: %s", exc, exc_info=True)
        _EMBEDDER = None
        return None


def _to_numpy(feature) -> np.ndarray:
    """Convert pyannote outputs into a 1-D numpy vector."""
    if isinstance(feature, np.ndarray):
        return feature.flatten()
    if torch.is_tensor(feature):
        return feature.detach().cpu().numpy().flatten()
    if isinstance(feature, SlidingWindowFeature):
        return feature.data.flatten()
    data = getattr(feature, "data", None)
    if isinstance(data, np.ndarray):
        return data.flatten()
    raise TypeError(f"Unsupported embedding type: {type(feature)}")


def _compute_embedding(waveform: np.ndarray, sample_rate: int, embedder: Inference) -> np.ndarray:
    """Compute a unit-normalised embedding for the provided waveform."""
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    result = embedder({"waveform": torch.tensor(waveform, dtype=torch.float32), "sample_rate": sample_rate})
    emb = _to_numpy(result)
    norm = np.linalg.norm(emb)
    if norm == 0.0:
        return emb
    return emb / norm


def _download_sample(url: str) -> str:
    """Download a speaker sample to a temp file and return its path."""
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    suffix = os.path.splitext(url.split("?")[0].split("#")[0])[-1]
    if not suffix or len(suffix) > 10 or "." not in suffix:
        suffix = ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(response.content)
    tmp.flush()
    tmp.close()
    return tmp.name


def _name_from_url(url: str) -> str:
    """Derive a friendly speaker name from the URL path if not provided."""
    base = url.split("?")[0].split("#")[0]
    stem = os.path.splitext(os.path.basename(base))[0]
    return stem or "speaker"


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #

def load_known_speakers_from_samples(
    speaker_samples: Iterable[dict],
    huggingface_access_token: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Load and cache embeddings for caller-provided speaker samples."""
    embedder = _ensure_embedder(huggingface_access_token)
    if embedder is None:
        return {}

    known: Dict[str, np.ndarray] = {}
    for sample in speaker_samples:
        name = sample.get("name") or _name_from_url(sample.get("url", ""))
        url = sample.get("url")
        file_path = sample.get("file_path")
        cache_key = (name, url, file_path)

        if cache_key in _KNOWN_EMBEDDINGS:
            known[name] = _KNOWN_EMBEDDINGS[cache_key]
            logger.debug("Using cached embedding for speaker '%s'.", name)
            continue

        tmp_path = None
        try:
            if file_path:
                path = file_path
            elif url:
                tmp_path = _download_sample(url)
                path = tmp_path
            else:
                logger.warning("Skipping speaker '%s': no url or file_path provided.", name)
                continue

            waveform, sr = librosa.load(path, sr=16000, mono=True)
            if waveform.size == 0:
                logger.warning("Skipping speaker '%s': empty audio.", name)
                continue

            embedding = _compute_embedding(waveform, sr, embedder)
            _KNOWN_EMBEDDINGS[cache_key] = embedding
            known[name] = embedding
            logger.debug("Enrolled speaker '%s' (dim=%d).", name, embedding.size)
        except Exception as exc:
            logger.error("Failed to enrol speaker '%s': %s", name, exc, exc_info=True)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    logger.debug("Could not remove temporary file %s", tmp_path)
    return known


def identify_speakers_on_segments(
    segments: List[dict],
    audio_path: str,
    enrolled: Dict[str, np.ndarray],
    threshold: float = 0.1,
    huggingface_access_token: Optional[str] = None,
) -> List[dict]:
    """Annotate diarized segments with the closest enrolled speaker."""
    if not segments or not enrolled:
        return segments

    embedder = _ensure_embedder(huggingface_access_token)
    if embedder is None:
        return segments

    names = list(enrolled.keys())
    matrix = np.stack([enrolled[name] for name in names])

    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        duration = max(0.0, end - start)
        if duration <= 0:
            continue

        try:
            waveform, sr = librosa.load(audio_path, sr=16000, mono=True, offset=start, duration=duration)
        except Exception as exc:
            logger.warning("Could not load segment audio [%s, %s]: %s", start, end, exc)
            continue
        if waveform.size == 0:
            continue

        embedding = _compute_embedding(waveform, sr, embedder)
        sims = 1.0 - cdist(embedding[np.newaxis, :], matrix, metric="cosine")[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= threshold:
            seg["speaker_id"] = names[best_idx]
        else:
            seg["speaker_id"] = "Unknown"
        seg["similarity"] = best_score
    return segments


def relabel_speakers_by_avg_similarity(segments: List[dict]) -> List[dict]:
    """Relabel diarized speaker tags using average similarity per cluster."""
    clusters: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for seg in segments:
        diar_label = seg.get("speaker")
        speaker_id = seg.get("speaker_id")
        similarity = seg.get("similarity")
        if diar_label and speaker_id and similarity is not None:
            clusters[diar_label].append((speaker_id, float(similarity)))

    relabel_map: Dict[str, str] = {}
    for diar_label, samples in clusters.items():
        if not samples:
            continue
        scores: Dict[str, List[float]] = defaultdict(list)
        for speaker_id, score in samples:
            scores[speaker_id].append(score)
        averaged = {speaker_id: sum(vals) / len(vals) for speaker_id, vals in scores.items()}
        relabel_map[diar_label] = max(averaged, key=averaged.get)

    for seg in segments:
        diar_label = seg.get("speaker")
        if diar_label in relabel_map:
            seg["speaker"] = relabel_map[diar_label]
    return segments
