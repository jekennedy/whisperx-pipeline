# src/rp_handler.py

try:
    from dotenv import load_dotenv, find_dotenv
except Exception:  # Graceful fallback when running outside the image
    def load_dotenv(*args, **kwargs):
        return False
    def find_dotenv(*args, **kwargs):
        return ""
import os
import sys
import json
import shutil
import logging
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import requests

import boto3
from botocore.client import Config
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import download_files_from_urls, rp_cleanup

from src.rp_schema import INPUT_VALIDATIONS
from src.predict import Predictor, Output

# Optional HF login for diarization or gated models
from huggingface_hub import login, whoami
import torch
import numpy as np

# Speaker helpers will be imported lazily inside functions to avoid heavy deps during module import

# -----------------------------------------------------------------------------
# Env and logging
# -----------------------------------------------------------------------------
try:
    load_dotenv(find_dotenv())
except Exception:
    pass

# If a bundled VAD model is present, advertise it to downstream code
VAD_MODEL_PATH = os.getenv("VAD_MODEL_PATH")
if VAD_MODEL_PATH and os.path.isfile(VAD_MODEL_PATH):
    os.environ["VAD_MODEL_PATH"] = VAD_MODEL_PATH

def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}

logger = logging.getLogger("rp_handler")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
# Console level: CONSOLE_LOG_LEVEL wins, else DEBUG env enables DEBUG, else INFO
_console_level_name = os.environ.get("CONSOLE_LOG_LEVEL", "").strip().upper()
if not _console_level_name:
    _console_level_name = "DEBUG" if _env_bool("DEBUG", False) else "INFO"
_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
console_handler.setLevel(_level_map.get(_console_level_name, logging.INFO))
console_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler("container_log.txt", mode="a")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if VAD_MODEL_PATH and os.path.isfile(VAD_MODEL_PATH):
    logger.info(f"Using bundled VAD model: {VAD_MODEL_PATH}")


# -----------------------------------------------------------------------------
# Hugging Face auth (optional)
# -----------------------------------------------------------------------------
raw_token = os.environ.get("HF_TOKEN", "").strip()
hf_token = raw_token
if hf_token and not hf_token.startswith("hf_"):
    logger.warning("HF_TOKEN does not start with 'hf_'; leaving as-is but authentication may fail.")

if hf_token:
    try:
        login(token=hf_token, add_to_git_credential=False)
        user = whoami(token=hf_token)
        logger.info(f"Hugging Face authenticated as: {user.get('name') or user.get('email')}")
    except Exception:
        logger.error("Failed to authenticate with Hugging Face", exc_info=True)
else:
    logger.info("No HF_TOKEN provided. Diarization with gated models may fail if required.")

# -----------------------------------------------------------------------------
# S3 / R2 client from environment for artifact uploads
# -----------------------------------------------------------------------------
STORAGE_ENDPOINT = os.environ.get("STORAGE_ENDPOINT")
STORAGE_BUCKET   = os.environ.get("STORAGE_BUCKET")
STORAGE_ACCESS   = os.environ.get("STORAGE_ACCESS_KEY")
STORAGE_SECRET   = os.environ.get("STORAGE_SECRET_KEY")
PREFIX_TRANSCRIPTS     = os.environ.get("PREFIX_TRANSCRIPTS", "transcripts/")
PRESIGN_TTL = int(os.environ.get("PRESIGN_TTL", "3600"))

_s3 = None
# Optional base dir for job downloads/scratch. Point this at your Network Volume mount
JOBS_DIR = os.environ.get("JOBS_DIR", "/jobs").rstrip("/") or "/jobs"
if all([STORAGE_ENDPOINT, STORAGE_BUCKET, STORAGE_ACCESS, STORAGE_SECRET]):
    try:
        _s3 = boto3.client(
            "s3",
            endpoint_url=STORAGE_ENDPOINT,
            aws_access_key_id=STORAGE_ACCESS,
            aws_secret_access_key=STORAGE_SECRET,
            config=Config(signature_version="s3v4"),
        )
        logger.info("Initialized S3/R2 client for artifact uploads.")
    except Exception:
        logger.error("Failed to initialize S3/R2 client. Will return inline results only.", exc_info=True)
        _s3 = None
else:
    logger.info("S3/R2 env not fully set. Will return inline results only.")

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
MODEL = Predictor()
MODEL.setup()

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _disk_log(path: str):
    try:
        usage = shutil.disk_usage(path)
        logger.info(
            f"Disk at {path}: total={usage.total/1e9:.1f}GB used={usage.used/1e9:.1f}GB free={usage.free/1e9:.1f}GB"
        )
    except Exception:
        logger.debug("disk usage probe failed", exc_info=True)

def cleanup_job_files(job_id, jobs_directory: str = None):
    base = jobs_directory or JOBS_DIR
    job_path = os.path.join(base, job_id)
    if os.path.exists(job_path):
        try:
            shutil.rmtree(job_path)
            logger.info(f"Removed job directory: {job_path}")
        except Exception:
            logger.error(f"Error removing job directory {job_path}", exc_info=True)
    else:
        logger.debug(f"Job directory not found: {job_path}")

def _stem_from(path_or_url: str) -> str:
    try:
        p = urlparse(path_or_url)
        base = os.path.basename(p.path)
    except Exception:
        base = os.path.basename(path_or_url)
    return os.path.splitext(base)[0]

def _put_text(key: str, text: str, content_type: str):
    _s3.put_object(
        Bucket=STORAGE_BUCKET,
        Key=key,
        Body=text.encode("utf-8"),
        ContentType=content_type,
    )

def _put_json(key: str, obj):
    _s3.put_object(
        Bucket=STORAGE_BUCKET,
        Key=key,
        Body=json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

def _ensure_job_dir(job_id: str) -> Path:
    base = Path(JOBS_DIR)
    path = base / job_id / "input_objects"
    path.mkdir(parents=True, exist_ok=True)
    return path

def _download_to_jobs(job_id: str, url: str) -> str:
    """Download URL to JOBS_DIR/<job_id>/input_objects using streaming.
    Returns file path as str.
    """
    dest_dir = _ensure_job_dir(job_id)
    # derive name
    try:
        parsed = urlparse(url)
        name = os.path.basename(parsed.path) or "input_audio"
    except Exception:
        name = "input_audio"
    dest = dest_dir / name

    _disk_log(str(dest_dir))
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB
                if chunk:
                    f.write(chunk)
    logger.debug(f"Downloaded to {dest}")
    return str(dest)

def _presign_get(key: str) -> str:
    return _s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": STORAGE_BUCKET, "Key": key},
        ExpiresIn=PRESIGN_TTL,
    )

def _sanitize_for_json(obj):
    """Recursively convert numpy types and other non-JSON-serializable values."""
    try:
        import numpy as _np
    except Exception:  # pragma: no cover
        _np = None

    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if _np is not None:
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.bool_,)):
            return bool(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    # Torch tensors occasionally sneak in; convert to list if small otherwise to float/int when scalar
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):
        try:
            arr = obj.detach().cpu().numpy()
            return _sanitize_for_json(arr)
        except Exception:
            return str(obj)
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return str(obj)
    return obj

# -----------------------------------------------------------------------------
# Main handler
# -----------------------------------------------------------------------------
def run(job):
    job_id    = job["id"]
    job_input = job["input"]

    validated = validate(job_input, INPUT_VALIDATIONS)
    if "errors" in validated:
        return {"error": validated["errors"]}

    # Optional tiny smoke response to validate control-plane return path
    if os.getenv("FORCE_TINY_OUTPUT", "").strip().lower() in {"1", "true", "yes", "on"}:
        logger.info("FORCE_TINY_OUTPUT enabled; returning minimal payload.")
        return {"output": {"ok": True, "note": "tiny"}}

    # 1) Download primary audio into JOBS_DIR (Network Volume if configured)
    try:
        audio_file_path = _download_to_jobs(job_id, job_input["audio_file"])
        logger.debug(f"Audio downloaded -> {audio_file_path}")
    except Exception as e:
        logger.error("Audio download failed", exc_info=True)
        return {"error": f"audio download: {e}"}

    # 2) Speaker profiles optional
    # Normalize speaker profiles: derive a friendly name from URL if not provided
    speaker_profiles = job_input.get("speaker_samples", [])
    embeddings = {}
    if speaker_profiles:
        # Lazy import heavy modules only if needed
        from src.speaker_processing import (
            load_known_speakers_from_samples,
            identify_speakers_on_segments,
            relabel_speakers_by_avg_similarity,
        )
        def _derive_name(u: str, idx: int) -> str:
            try:
                p = urlparse(u)
                qn = parse_qs(p.query).get("name")
                if qn and qn[0].strip():
                    return qn[0].strip()
                if p.fragment and p.fragment.strip():
                    return p.fragment.strip()
                base = os.path.basename(p.path)
                stem, _ = os.path.splitext(base)
                if stem:
                    return stem[:64]
            except Exception:
                pass
            return f"spk{idx+1}"

        # Build normalized list with explicit names
        norm_profiles = []
        for i, s in enumerate(speaker_profiles):
            if isinstance(s, str):
                norm_profiles.append({"name": _derive_name(s, i), "url": s})
            elif isinstance(s, dict):
                u = s.get("url", "")
                n = s.get("name") or _derive_name(u, i)
                # keep file_path if caller provided a local file
                ent = {"name": n, "url": u}
                if s.get("file_path"):
                    ent["file_path"] = s["file_path"]
                norm_profiles.append(ent)
            else:
                logger.warning(f"Unsupported speaker sample entry: {s!r}")
        try:
            embeddings = load_known_speakers_from_samples(
                norm_profiles,
                huggingface_access_token=job_input.get("huggingface_access_token") or hf_token
            )
            logger.info(f"Enrolled {len(embeddings)} speaker profiles successfully.")
        except Exception as e:
            logger.error("Enrollment failed", exc_info=True)

    # 3) WhisperX / diarization
    # Clip beam_size into a reasonable range if provided
    _beam = job_input.get("beam_size")
    if _beam is not None:
        try:
            _beam = max(1, min(10, int(_beam)))
        except Exception:
            _beam = None

    predict_input = {
        "audio_file": audio_file_path,
        "beam_size": _beam,
        "language": job_input.get("language"),
        "language_detection_min_prob": job_input.get("language_detection_min_prob", 0),
        "language_detection_max_tries": job_input.get("language_detection_max_tries", 5),
        "initial_prompt": job_input.get("initial_prompt"),
        "batch_size": job_input.get("batch_size", 64),
        "temperature": job_input.get("temperature", 0),
        "patience": job_input.get("patience"),
        "length_penalty": job_input.get("length_penalty"),
        "no_speech_threshold": job_input.get("no_speech_threshold"),
        "log_prob_threshold": job_input.get("log_prob_threshold"),
        "compression_ratio_threshold": job_input.get("compression_ratio_threshold"),
        "vad_onset": job_input.get("vad_onset", 0.50),
        "vad_offset": job_input.get("vad_offset", 0.363),
        "align_output": job_input.get("align_output", False),
        "diarization": job_input.get("diarization", False),
        "huggingface_access_token": job_input.get("huggingface_access_token") or hf_token,
        "min_speakers": job_input.get("min_speakers"),
        "max_speakers": job_input.get("max_speakers"),
        # Default job-level debug to env DEBUG when not provided
        "debug": job_input.get("debug", _env_bool("DEBUG", False)),
    }

    # Log effective ASR knobs for this job (info-level)
    try:
        logger.info(
            "ASR opts | beam=%s temp=%s batch=%s lang=%s vad_onset=%.3f vad_offset=%.3f prompt=%s align=%s diar=%s patience=%s len_pen=%s no_speech=%s log_prob=%s comp_ratio=%s",
            predict_input.get("beam_size"),
            predict_input.get("temperature"),
            predict_input.get("batch_size"),
            predict_input.get("language"),
            float(predict_input.get("vad_onset") or 0),
            float(predict_input.get("vad_offset") or 0),
            bool(predict_input.get("initial_prompt")),
            bool(predict_input.get("align_output")),
            bool(predict_input.get("diarization")),
            predict_input.get("patience"),
            predict_input.get("length_penalty"),
            predict_input.get("no_speech_threshold"),
            predict_input.get("log_prob_threshold"),
            predict_input.get("compression_ratio_threshold"),
        )
    except Exception:
        pass

    try:
        result: Output = MODEL.predict(**predict_input)
    except Exception as e:
        logger.error("WhisperX prediction failed", exc_info=True)
        return {"error": f"prediction: {e}"}

    output_dict = {
        "segments": result.segments,
        "detected_language": getattr(result, "detected_language", None),
    }

    # 4) Optional speaker identification on segments
    if embeddings:
        # Import here as well to keep top-level import light
        from src.speaker_processing import identify_speakers_on_segments, relabel_speakers_by_avg_similarity
        try:
            segments_with_speakers = identify_speakers_on_segments(
                segments=output_dict["segments"],
                audio_path=audio_file_path,
                enrolled=embeddings,
                threshold=0.1,
                huggingface_access_token=job_input.get("huggingface_access_token") or hf_token,
            )
            segments_with_final_labels = relabel_speakers_by_avg_similarity(segments_with_speakers)
            output_dict["segments"] = segments_with_final_labels
            logger.info("Speaker identification completed.")
        except Exception:
            logger.error("Speaker identification failed", exc_info=True)

    # 5) Upload artifacts to R2 when configured, else inline minimal results
    artifacts = {}
    stem = _stem_from(job_input.get("audio_file") or audio_file_path)

    if _s3:
        try:
            # Full transcript (primary JSON)
            transcript_obj = {
                "segments": _sanitize_for_json(output_dict["segments"]),
                "detected_language": output_dict.get("detected_language"),
            }
            transcript_key = f"{PREFIX_TRANSCRIPTS}{stem}.json"
            _put_json(transcript_key, transcript_obj)
            artifacts["transcript_key"] = transcript_key
            try:
                artifacts["transcript_url"] = _presign_get(transcript_key)
            except Exception:
                logger.debug("Presign transcript failed; continuing without URL", exc_info=True)

            # Also write segments-only JSON for compatibility
            seg_key = f"{PREFIX_TRANSCRIPTS}{stem}.segments.json"
            _put_json(seg_key, transcript_obj["segments"])
            artifacts["segments_key"] = seg_key
            try:
                artifacts["segments_url"] = _presign_get(seg_key)
            except Exception:
                logger.debug("Presign segments failed; continuing without URL", exc_info=True)
        except Exception:
            logger.error("Failed uploading segments JSON", exc_info=True)

        srt_text = getattr(result, "srt", None)
        if isinstance(srt_text, str) and srt_text.strip():
            try:
                srt_key = f"{PREFIX_TRANSCRIPTS}{stem}.srt"
                _put_text(srt_key, srt_text, "application/x-subrip")
                artifacts["srt_key"] = srt_key
                try:
                    artifacts["srt_url"] = _presign_get(srt_key)
                except Exception:
                    logger.debug("Presign SRT failed; continuing without URL", exc_info=True)
            except Exception:
                logger.error("Failed uploading SRT", exc_info=True)

        vtt_text = getattr(result, "vtt", None)
        if isinstance(vtt_text, str) and vtt_text.strip():
            try:
                vtt_key = f"{PREFIX_TRANSCRIPTS}{stem}.vtt"
                _put_text(vtt_key, vtt_text, "text/vtt")
                artifacts["vtt_key"] = vtt_key
                try:
                    artifacts["vtt_url"] = _presign_get(vtt_key)
                except Exception:
                    logger.debug("Presign VTT failed; continuing without URL", exc_info=True)
            except Exception:
                logger.error("Failed uploading VTT", exc_info=True)

    small_output = {
        "detected_language": output_dict.get("detected_language"),
    }
    if artifacts:
        small_output.update(artifacts)
    else:
        # Fallback when S3 is not configured
        segs = output_dict.get("segments") or []
        preview = []
        if isinstance(segs, list):
            for s in segs[:50]:
                try:
                    start = float(s.get("start", 0))
                except Exception:
                    start = 0.0
                try:
                    end = float(s.get("end", 0))
                except Exception:
                    end = 0.0
                text = s.get("text")
                if not isinstance(text, str):
                    text = str(text) if text is not None else ""
                preview.append({
                    "start": start,
                    "end": end,
                    "text": text[:500]
                })
        small_output["segments"] = preview

    # 6) Cleanup and return through RunPod helper
    try:
        # rp_cleanup expects default /jobs; safe to ignore failures when JOBS_DIR differs
        try:
            rp_cleanup.clean(["input_objects"])
        except Exception:
            logger.debug("rp_cleanup skipped or failed", exc_info=True)
        cleanup_job_files(job_id)
    except Exception:
        logger.warning("Cleanup issue", exc_info=True)

    # Log return payload size to help diagnose 400s from control plane
    try:
        body = {"output": small_output}
        approx_bytes = len(json.dumps(body, ensure_ascii=False))
        logger.info(f"[RETURN] payload bytes ~ {approx_bytes}")
    except Exception:
        logger.warning("Failed to size return payload", exc_info=True)

    return {"output": small_output}
