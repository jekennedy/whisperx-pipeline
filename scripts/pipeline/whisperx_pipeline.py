#!/usr/bin/env python3
import argparse
import os
import sys
import time
import json
import mimetypes
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import re

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
import requests
from urllib.parse import urlparse, parse_qs, urlunparse
from requests.exceptions import HTTPError
from dotenv import load_dotenv, find_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pyannote_helper import PyannoteClient  # noqa: E402

ENH_TAIL_RE = re.compile(r'^(?P<stem>.+?)(?:-esv2-(?P<enh>\d{1,3})p-bg-(?P<bg>\d{1,3})p)?$', re.IGNORECASE)
METADATA_KEYS = {"delayTime", "executionTime", "id", "status", "workerId"}
# Keys accepted by the worker input schema (rp_schema.py)
ALLOWED_INPUT_KEYS = {
    "audio_file",
    "language",
    "language_detection_min_prob",
    "language_detection_max_tries",
    "initial_prompt",
    "batch_size",
    "beam_size",
    "temperature",
    "vad_onset",
    "vad_offset",
    "num_speakers",
    "align_output",
    "diarization",
    "min_speakers",
    "max_speakers",
    "debug",
    "speaker_verification",
    "speaker_samples",
    "model",
    # advanced decode knobs
    "patience",
    "length_penalty",
    "no_speech_threshold",
    "log_prob_threshold",
    "compression_ratio_threshold",
}

def guess_content_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".mp3":
        return "audio/mpeg"
    if ext in (".wav", ".wave"):
        return "audio/wav"
    t, _ = mimetypes.guess_type(str(path))
    return t or "application/octet-stream"

def parse_cli():
    p = argparse.ArgumentParser(
        description="Upload enhanced audio to R2, run WhisperX transcription on RunPod, then fetch diarization via the pyannote API."
    )
    p.add_argument("file", help="Local path to enhanced audio (mp3/wav).")
    # R2 / S3
    p.add_argument("--s3-endpoint", default=os.getenv("STORAGE_ENDPOINT"), help="R2/S3 endpoint URL.")
    p.add_argument("--bucket", default=os.getenv("STORAGE_BUCKET"), help="R2 bucket name.")
    p.add_argument("--access-key", default=os.getenv("STORAGE_ACCESS_KEY"), help="R2 access key.")
    p.add_argument("--secret-key", default=os.getenv("STORAGE_SECRET_KEY"), help="R2 secret key.")
    p.add_argument("--url-expiry", type=int, default=3600, help="Presigned URL TTL seconds (default 3600).")
    p.add_argument("--prefix-audio", default="enhanced/", help="Key prefix in bucket for audio (default enhanced/).")
    p.add_argument("--prefix-transcripts", default="transcripts/", help="Key prefix in bucket for transcripts (default transcripts/).")
    p.add_argument("--force", action="store_true", help="Force re-upload even if object exists.")
    # RunPod
    p.add_argument("--endpoint", default=os.getenv("RUNPOD_ENDPOINT"), help="RunPod endpoint ID.")
    p.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"), help="RunPod API key.")
    p.add_argument("--sync", action="store_true", help="Use /runsync; else async /run + polling.")
    p.add_argument("--debug", action="store_true", help="Print full RunPod responses (secrets redacted).")
    # WhisperX options
    p.add_argument("--model", default="large-v3", help="Whisper model size (e.g., large-v3).")
    p.add_argument("--language", default=None, help="Language hint (e.g., en, it).")
    p.add_argument("--align-output", action="store_true", default=True, help="Enable alignment (default true).")
    p.add_argument("--no-align", dest="align_output", action="store_false", help="Disable alignment.")
    p.add_argument(
        "--diarization",
        action="store_true",
        default=True,
        help="Enable diarization using the pyannote API (requires --pyannote-api-key or PYANNOTE_API_KEY).",
    )
    p.add_argument("--no-diarization", dest="diarization", action="store_false", help="Skip the pyannote diarization call.")
    p.add_argument("--speaker-verification", action="store_true", help="Enable speaker verification if samples are provided.")
    p.add_argument("--speaker-sample", action="append", default=[], help="URL to a speaker sample WAV/MP3; can be used multiple times.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=None, help="Batch size for transcribe (worker default 64).")
    p.add_argument("--beam-size", type=int, default=None, help="Beam size for decoding (>=1; 1 ~ greedy).")
    p.add_argument("--initial-prompt", default=None, help="Initial prompt text (for long prompts consider --params-json).")
    p.add_argument("--vad-onset", type=float, default=None, help="VAD onset threshold (lower includes more speech).")
    p.add_argument("--vad-offset", type=float, default=None, help="VAD offset threshold (lower includes more speech).")
    p.add_argument("--num-speakers", type=int, default=None, help="Exact number of speakers (if known).")
    p.add_argument("--min-speakers", type=int, default=None, help="Minimum number of speakers (diarization).")
    p.add_argument("--max-speakers", type=int, default=None, help="Maximum number of speakers (diarization).")
    # pyannote API options
    p.add_argument("--pyannote-api-key", default=os.getenv("PYANNOTE_API_KEY"), help="pyannote API key; defaults to PYANNOTE_API_KEY env.")
    p.add_argument(
        "--pyannote-model",
        default="precision-2",
        choices=["precision-1", "precision-2", "community-1"],
        help="pyannote diarization model.",
    )
    p.add_argument("--pyannote-exclusive", action="store_true", help="Request exclusive diarization (no overlapping speech).")
    p.add_argument(
        "--pyannote-poll-interval",
        type=float,
        default=3.0,
        help="Polling interval (seconds) while waiting for pyannote job completion.",
    )
    p.add_argument(
        "--pyannote-timeout",
        type=float,
        default=900.0,
        help="Timeout (seconds) for pyannote job completion.",
    )
    p.add_argument(
        "--pyannote-json-in",
        type=Path,
        help="Path to a saved pyannote job JSON payload; reuse this instead of calling the API.",
    )
    p.add_argument(
        "--pyannote-job-json",
        dest="pyannote_json_in",
        type=Path,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    # Deepgram options (optional)
    p.add_argument("--deepgram-api-key", default=os.getenv("DEEPGRAM_API_KEY"), help="Deepgram API key; enable word confidence enrichment.")
    p.add_argument("--deepgram-model", default="nova-3", help="Deepgram model (default nova-3).")
    p.add_argument("--deepgram-tier", default=None, help="Optional Deepgram tier (e.g., base, pro).")
    p.add_argument("--deepgram-version", default=None, help="Optional Deepgram model version.")
    p.add_argument("--deepgram-json-in", type=Path, help="Path to cached Deepgram JSON; reuse instead of calling the API.")
    p.add_argument("--word-confidence-low-threshold", dest="word_confidence_low_threshold", type=float, default=0.6, help="Word confidence threshold to classify low-confidence speech.")
    p.add_argument("--word-confidence-high-threshold", type=float, default=0.85, help="Word confidence threshold to classify high-confidence segments.")
    p.add_argument("--low-confidence-min-dur", type=float, default=0.5, help="Minimum low-confidence duration (seconds) required to flag a segment.")
    # Optional: load extra input fields from a JSON file
    p.add_argument("--params-json", default=None, help="Path to JSON file with additional worker input fields. CLI flags override these values.")
    # Download options
    p.add_argument("--download-transcript", action="store_true", help="Download transcript files locally after processing.")
    p.add_argument("--download-dir", default="./transcripts", help="Local directory to save downloaded transcripts (default ./transcripts).")
    return p.parse_args()

def build_s3_client(args):
    missing = [k for k in ("s3_endpoint", "bucket", "access_key", "secret_key") if not getattr(args, k)]
    if missing:
        sys.exit(f"Missing R2/S3 config: {', '.join(missing)} (flags or env)")
    return boto3.client(
        "s3",
        endpoint_url=args.s3_endpoint,
        aws_access_key_id=args.access_key,
        aws_secret_access_key=args.secret_key,
        config=Config(signature_version="s3v4"),
    )

def object_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise

def presign_get(s3, bucket: str, key: str, ttl: int) -> str:
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=ttl,
    )

def upload_file(s3, bucket: str, key: str, local_path: Path, content_type: Optional[str] = None):
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    try:
        s3.upload_file(str(local_path), bucket, key, ExtraArgs=extra)
    except ClientError as e:
        sys.exit(f"S3 upload failed for {key}: {e.response.get('Error', {}).get('Message', str(e))}")

def put_bytes(s3, bucket: str, key: str, data: bytes, content_type: str):
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    except ClientError as e:
        sys.exit(f"S3 put_object failed for {key}: {e}")

def put_text(s3, bucket: str, key: str, text: str, content_type: str):
    put_bytes(s3, bucket, key, text.encode("utf-8"), content_type)

def download_file(s3, bucket: str, key: str, local_path: Path) -> bool:
    """Download a file from S3 to local path. Returns True if successful."""
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(local_path))
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            print(f"[warning] File not found in S3: s3://{bucket}/{key}")
            return False
        print(f"[error] Failed to download s3://{bucket}/{key}: {e}")
        return False
    except Exception as e:
        print(f"[error] Failed to download s3://{bucket}/{key}: {e}")
        return False

def redact(obj: Any) -> Any:
    try:
        s = json.dumps(obj)
    except Exception:
        return obj
    for k in ("api_key", "accessKeyId", "secretAccessKey", "Authorization", "authorization", "hf-token", "huggingface_access_token"):
        s = s.replace(k, f"{k} (redacted key name)")
    return json.loads(s)

def debug_print(enabled: bool, label: str, data: Any):
    if enabled:
        print(f"[DEBUG] {label}:")
        try:
            print(json.dumps(redact(data), indent=2, ensure_ascii=False))
        except Exception:
            print(str(data))

def looks_like_transcript(o: Any) -> bool:
    if not isinstance(o, dict):
        return False
    if isinstance(o.get("segments"), list) and o.get("segments"):
        return True
    if isinstance(o.get("text"), str) and o.get("text").strip():
        return True
    subs = o.get("subtitles") or {}
    if isinstance(subs, dict) and (isinstance(subs.get("srt"), str) or isinstance(subs.get("vtt"), str)):
        return True
    if isinstance(o.get("srt"), str) or isinstance(o.get("vtt"), str):
        return True
    # Accept artifacts referenced by URL or by storage keys
    if any(k in o for k in ("segments_url", "srt_url", "vtt_url", "segments_key", "srt_key", "vtt_key", "transcript_key", "transcript_url")):
        return True
    return False

def is_metadata_only(o: Any) -> bool:
    return isinstance(o, dict) and set(o.keys()).issubset(METADATA_KEYS)

def ensure_output_or_raise(resp: Dict[str, Any], context: str):
    actual_output = None

    # Case 1: RunPod response with nested 'output'
    if isinstance(resp.get("output"), dict) and isinstance(resp["output"].get("output"), dict):
        actual_output = resp["output"]["output"]
        if looks_like_transcript(actual_output):
            return actual_output

    # Case 2: RunPod response with direct 'output'
    if isinstance(resp.get("output"), dict):
        actual_output = resp["output"]
        if looks_like_transcript(actual_output):
            return actual_output
    
    # Case 3: 'resp' itself is the transcript (e.g. from polling if it returns only the output)
    if looks_like_transcript(resp):
        return resp
    raise RuntimeError(
        f"{context}: job returned no transcript payload. Full response below:\n"
        + json.dumps(resp, indent=2, ensure_ascii=False)
    )

def runpod_call(endpoint: str, api_key: str, payload: Dict[str, Any], sync: bool, debug: bool) -> Dict[str, Any]:
    def _normalize_endpoint(ep: str) -> str:
        if not ep:
            return ep
        ep = ep.strip().rstrip("/")
        # Accept full URLs; extract the ID after /v2/
        if ep.startswith("http://") or ep.startswith("https://"):
            try:
                from urllib.parse import urlparse
                p = urlparse(ep)
                parts = [s for s in p.path.split("/") if s]
                # Expect something like ["v2", "<id>", ...]
                if len(parts) >= 2 and parts[0] == "v2":
                    return parts[1]
                # Fallback to last non-empty segment
                return parts[-1] if parts else ep
            except Exception:
                return ep
        # If it contains /v2/<id> style path without scheme
        if "/v2/" in ep:
            try:
                parts = [s for s in ep.split("/") if s]
                v2_idx = parts.index("v2")
                if v2_idx >= 0 and len(parts) > v2_idx + 1:
                    return parts[v2_idx + 1]
            except Exception:
                pass
        # Otherwise assume it's already an ID
        return ep

    endpoint_id = _normalize_endpoint(endpoint)
    base = f"https://api.runpod.ai/v2/{endpoint_id}"
    route = "runsync" if sync else "run"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        r = requests.post(f"{base}/{route}", json={"input": payload}, headers=headers, timeout=1800)
        r.raise_for_status()
        data = r.json()
    except HTTPError as e:
        # Surface server message to help diagnose 400s, endpoint typos, etc.
        msg = getattr(e.response, "text", "") or str(e)
        raise RuntimeError(
            "RunPod API error: "
            + (msg.strip()[:2000])
            + f"\nNote: endpoint='{endpoint}' normalized='{endpoint_id}' route='{route}'."
        ) from e
    debug_print(debug, f"RunPod {route} response", data)

    if sync:
        status = data.get("status")
        job_id = data.get("id")
        if status == "COMPLETED":
            return ensure_output_or_raise(data, "runsync completed")
        if job_id and status in ("IN_QUEUE", "IN_PROGRESS"):
            while True:
                s = requests.get(f"{base}/status/{job_id}", headers=headers, timeout=60).json()
                debug_print(debug, "RunPod status poll", s)
                st = s.get("status")
                if st == "COMPLETED":
                    return ensure_output_or_raise(s, "runsync polled completion")
                if st == "FAILED":
                    raise RuntimeError(f"RunPod job failed: {json.dumps(s, indent=2)}")
                time.sleep(3)
        raise RuntimeError(
            "Unexpected runsync response; no output and not pollable. "
            + json.dumps(data, indent=2, ensure_ascii=False)
        )

    job_id = data.get("id")
    if not job_id:
        raise RuntimeError("Async run returned no job id: " + json.dumps(data, indent=2, ensure_ascii=False))
    while True:
        s = requests.get(f"{base}/status/{job_id}", headers=headers, timeout=60).json()
        debug_print(debug, "RunPod status poll", s)
        st = s.get("status")
        if st == "COMPLETED":
            return ensure_output_or_raise(s, "async polled completion")
        if st == "FAILED":
            raise RuntimeError(f"RunPod job failed: {json.dumps(s, indent=2)}")
        time.sleep(3)

def http_fetch(url: str, timeout: int = 120) -> Tuple[bytes, str]:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
    return resp.content, ctype or "application/octet-stream"

def _filter_allowed(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if k in ALLOWED_INPUT_KEYS}

def presign_s3_url_if_needed(s3, url: str, ttl: int) -> Tuple[str, Optional[str]]:
    """If `url` is s3://bucket/key, return a presigned HTTPS URL for it and a derived name.
    If it's http(s), return unchanged URL. Name can be provided as ?name= or #fragment.
    Returns (url_to_use, derived_name_or_None).
    """
    try:
        p = urlparse(url)
        # derive name from query or fragment regardless of scheme
        q = parse_qs(p.query).get("name")
        name = q[0].strip() if q and q[0].strip() else None
        if not name and p.fragment and p.fragment.strip():
            name = p.fragment.strip()

        if p.scheme.lower() != "s3":
            return url, name
        bucket = p.netloc
        key = p.path.lstrip("/")
        presigned = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=ttl,
        )
        return presigned, name
    except Exception:
        return url, None

def save_url_artifacts_if_any(s3, bucket: str, prefix_transcripts: str, stem: str, output: Dict[str, Any]) -> Dict[str, Optional[str]]:
    saved = {"segments_key": None, "srt_key": None, "vtt_key": None}

    seg_url = output.get("segments_url")
    if isinstance(seg_url, str):
        data, ctype = http_fetch(seg_url)
        try:
            parsed = json.loads(data.decode("utf-8"))
            data = json.dumps(parsed, indent=2, ensure_ascii=False).encode("utf-8")
            ctype = "application/json"
        except Exception:
            if "json" in ctype:
                ctype = "application/json"
        key = f"{prefix_transcripts}{stem}.segments.json"
        print(f"[upload] segments (from URL) -> s3://{bucket}/{key}")
        put_bytes(s3, bucket, key, data, ctype)
        saved["segments_key"] = key

    srt_url = output.get("srt_url")
    if isinstance(srt_url, str):
        data, _ctype = http_fetch(srt_url)
        key = f"{prefix_transcripts}{stem}.srt"
        print(f"[upload] SRT (from URL) -> s3://{bucket}/{key}")
        put_bytes(s3, bucket, key, data, "application/x-subrip")
        saved["srt_key"] = key

    vtt_url = output.get("vtt_url")
    if isinstance(vtt_url, str):
        data, _ctype = http_fetch(vtt_url)
        key = f"{prefix_transcripts}{stem}.vtt"
        print(f"[upload] VTT (from URL) -> s3://{bucket}/{key}")
        put_bytes(s3, bucket, key, data, "text/vtt")
        saved["vtt_key"] = key

    return saved
def _get_audio_duration_secs_via_ffprobe(local_path: Path) -> Optional[float]:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(local_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return None

def _extract_segments_from_dict(obj: Dict[str, Any]) -> Optional[list]:
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("segments"), list):
        return obj["segments"]
    out = obj.get("output")
    if isinstance(out, dict) and isinstance(out.get("segments"), list):
        return out["segments"]
    return None


def _load_segments_for_alignment(
    output: Dict[str, Any],
    s3,
    bucket: Optional[str],
    ttl: int,
) -> List[Dict[str, Any]]:
    """Hydrate transcript segments, fetching from URLs/keys if needed."""
    def _try_parse(data: bytes) -> Optional[List[Dict[str, Any]]]:
        try:
            decoded = json.loads(data.decode("utf-8"))
        except Exception:
            return None
        segs = _extract_segments_from_dict(decoded)
        if isinstance(segs, list) and segs:
            return segs
        if isinstance(decoded, list) and decoded:
            return decoded
        return None

    inline = _extract_segments_from_dict(output)
    if isinstance(inline, list) and inline:
        return inline

    seg_url = output.get("segments_url")
    if isinstance(seg_url, str):
        try:
            data, _ = http_fetch(seg_url)
            segs = _try_parse(data)
            if segs:
                return segs
        except Exception:
            pass

    transcript_url = output.get("transcript_url")
    if isinstance(transcript_url, str):
        try:
            data, _ = http_fetch(transcript_url)
            segs = _try_parse(data)
            if segs:
                return segs
        except Exception:
            pass

    if s3 and bucket:
        segments_key = output.get("segments_key")
        if isinstance(segments_key, str):
            try:
                url = presign_get(s3, bucket, segments_key, ttl)
                data, _ = http_fetch(url)
                segs = _try_parse(data)
                if segs:
                    return segs
            except Exception:
                pass
        transcript_key = output.get("transcript_key")
        if isinstance(transcript_key, str):
            try:
                url = presign_get(s3, bucket, transcript_key, ttl)
                data, _ = http_fetch(url)
                segs = _try_parse(data)
                if segs:
                    return segs
            except Exception:
                pass

    return []


def _extract_pyannote_segments(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    output = job.get("output") or {}
    segs = output.get("diarization") or []
    normalized: List[Dict[str, Any]] = []
    for seg in segs:
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
        except (TypeError, ValueError):
            continue
        speaker = seg.get("speaker")
        if speaker is None:
            speaker = seg.get("label")
        normalized.append(
            {
                "speaker": speaker,
                "start": start,
                "end": end,
                "confidence": seg.get("confidence"),
            }
        )
    return normalized


def _select_speaker(diar_segments: List[Dict[str, Any]], start: float, end: float) -> Optional[str]:
    best_label: Optional[str] = None
    best_overlap = 0.0
    for seg in diar_segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        overlap = max(0.0, min(end, seg_end) - max(start, seg_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = seg.get("speaker")
    return best_label


def _apply_diarization_to_segments(
    transcript_segments: List[Dict[str, Any]],
    diar_segments: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    assigned: List[Dict[str, Any]] = []
    speaker_totals: Dict[str, float] = {}
    if not transcript_segments:
        return assigned, speaker_totals

    for seg in transcript_segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        speaker = _select_speaker(diar_segments, seg_start, seg_end)
        new_seg = dict(seg)
        if speaker:
            new_seg["speaker"] = speaker
            speaker_totals[speaker] = speaker_totals.get(speaker, 0.0) + max(0.0, seg_end - seg_start)

        words = seg.get("words")
        if isinstance(words, list):
            new_words = []
            for w in words:
                w_start = float(w.get("start", seg_start))
                w_end = float(w.get("end", w_start))
                w_speaker = _select_speaker(diar_segments, w_start, w_end) or speaker
                new_word = dict(w)
                if w_speaker:
                    new_word["speaker"] = w_speaker
                new_words.append(new_word)
            new_seg["words"] = new_words
        assigned.append(new_seg)
    return assigned, speaker_totals


def _seconds_to_srt_ts(value: float) -> str:
    total_ms = int(round(max(value, 0.0) * 1000))
    hours, remainder = divmod(total_ms, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _seconds_to_vtt_ts(value: float) -> str:
    total_ms = int(round(max(value, 0.0) * 1000))
    hours, remainder = divmod(total_ms, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _segments_to_srt(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, seg in enumerate(segments, start=1):
        start = _seconds_to_srt_ts(float(seg.get("start", 0.0)))
        end = _seconds_to_srt_ts(float(seg.get("end", 0.0)))
        text = (seg.get("text") or "").strip()
        speaker = seg.get("speaker")
        if speaker:
            text = f"{speaker}: {text}" if text else f"{speaker}:"
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text or "<no speech>")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _segments_to_vtt(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = ["WEBVTT", ""]
    for seg in segments:
        start = _seconds_to_vtt_ts(float(seg.get("start", 0.0)))
        end = _seconds_to_vtt_ts(float(seg.get("end", 0.0)))
        text = (seg.get("text") or "").strip()
        speaker = seg.get("speaker")
        if speaker:
            text = f"{speaker}: {text}" if text else f"{speaker}:"
        lines.append(f"{start} --> {end}")
        lines.append(text or "<no speech>")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def run_pyannote_diarization(
    audio_url: str,
    api_key: str,
    *,
    model: str,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    exclusive: bool,
    poll_interval: float,
    timeout: float,
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    client = PyannoteClient(api_key)
    print(f"[pyannote] Submitting diarization job (model={model})")
    job_id = client.submit_diarization(
        audio_url,
        model=model,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        exclusive=exclusive,
    )
    print(f"[pyannote] Job queued: {job_id}")
    job = client.wait_for_job(job_id, interval=poll_interval, timeout=timeout)
    segments = _extract_pyannote_segments(job)
    print(f"[pyannote] Job {job_id} returned {len(segments)} diarization segment(s)")
    return job_id, job, segments


def _call_deepgram_api(
    audio_url: str,
    api_key: str,
    *,
    model: str,
    tier: Optional[str],
    version: Optional[str],
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "url": audio_url,
        "model": model,
        "smart_format": False,
        "punctuate": False,
        "utterances": False,
        "diarize": False,
        "paragraphs": False,
        "filler_words": False,
    }
    if tier:
        payload["tier"] = tier
    if version:
        payload["version"] = version

    resp = requests.post(
        "https://api.deepgram.com/v1/listen",
        headers=headers,
        json=payload,
        timeout=600,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        detail = None
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        raise RuntimeError(f"Deepgram API call failed: {detail}") from exc
    return resp.json()


def _extract_deepgram_words(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = payload.get("results") or {}
    channels = results.get("channels") or []
    words: List[Dict[str, Any]] = []
    for channel in channels:
        alternatives = channel.get("alternatives") or []
        for alt in alternatives:
            for w in alt.get("words") or []:
                try:
                    start = float(w.get("start"))
                    end = float(w.get("end"))
                    confidence = float(w.get("confidence"))
                except (TypeError, ValueError):
                    continue
                if end <= start:
                    continue
                words.append(
                    {
                        "start": start,
                        "end": end,
                        "confidence": confidence,
                        "word": w.get("word"),
                    }
                )
    words.sort(key=lambda w: (w["start"], w["end"]))
    return words


def _merge_low_confidence_intervals(
    low_words: List[Dict[str, Any]],
    min_duration: float,
    gap_tolerance: float = 0.3,
) -> List[Dict[str, Any]]:
    if not low_words:
        return []
    low_words.sort(key=lambda w: w["start"])
    merged: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for item in low_words:
        start = float(item["start"])
        end = float(item["end"])
        confidence = float(item["confidence"])
        speaker = item.get("speaker")
        if current is None:
            current = {
                "start": start,
                "end": end,
                "speaker": speaker,
                "confidence_sum": confidence,
                "confidence_min": confidence,
                "count": 1,
            }
            continue
        if (
            speaker == current.get("speaker")
            and start - current["end"] <= gap_tolerance
        ):
            current["end"] = max(current["end"], end)
            current["confidence_sum"] += confidence
            current["confidence_min"] = min(current["confidence_min"], confidence)
            current["count"] += 1
        else:
            duration = current["end"] - current["start"]
            if duration >= min_duration:
                merged.append(
                    {
                        "start": current["start"],
                        "end": current["end"],
                        "duration": duration,
                        "speaker": current.get("speaker"),
                        "avg_confidence": current["confidence_sum"] / current["count"],
                        "min_confidence": current["confidence_min"],
                    }
                )
            current = {
                "start": start,
                "end": end,
                "speaker": speaker,
                "confidence_sum": confidence,
                "confidence_min": confidence,
                "count": 1,
            }

    if current:
        duration = current["end"] - current["start"]
        if duration >= min_duration:
            merged.append(
                {
                    "start": current["start"],
                    "end": current["end"],
                    "duration": duration,
                    "speaker": current.get("speaker"),
                    "avg_confidence": current["confidence_sum"] / current["count"],
                    "min_confidence": current["confidence_min"],
                }
            )
    return merged


def _merge_word_confidence_into_segments(
    segments: List[Dict[str, Any]],
    deepgram_words: List[Dict[str, Any]],
    *,
    low_threshold: float,
    high_threshold: float,
    min_low_duration: float,
) -> Dict[str, Any]:
    whisper_words: List[Dict[str, Any]] = []
    for seg_idx, seg in enumerate(segments):
        words = seg.get("words")
        if not isinstance(words, list):
            continue
        for word_idx, word in enumerate(words):
            try:
                start = float(word.get("start"))
                end = float(word.get("end"))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            whisper_words.append(
                {
                    "segment": seg_idx,
                    "word_index": word_idx,
                    "start": start,
                    "end": end,
                    "mid": (start + end) / 2.0,
                }
            )

    whisper_words.sort(key=lambda w: (w["start"], w["end"]))
    dg_sorted = sorted(deepgram_words, key=lambda w: (w["start"], w["end"]))

    tolerance = 0.12
    used_dg = [False] * len(dg_sorted)
    dg_pointer = 0
    matched = 0
    unmatched_whisper = 0

    for w_word in whisper_words:
        start = w_word["start"]
        end = w_word["end"]
        mid = w_word["mid"]
        while (
            dg_pointer < len(dg_sorted)
            and dg_sorted[dg_pointer]["end"] < start - tolerance
        ):
            dg_pointer += 1
        best_idx = None
        best_overlap = -1.0
        best_mid_diff = float("inf")
        scan_idx = max(0, dg_pointer - 2)
        while scan_idx < len(dg_sorted):
            g_word = dg_sorted[scan_idx]
            if g_word["start"] > end + tolerance:
                break
            if used_dg[scan_idx]:
                scan_idx += 1
                continue
            overlap = min(end, g_word["end"]) - max(start, g_word["start"])
            g_mid = (g_word["start"] + g_word["end"]) / 2.0
            mid_diff = abs(mid - g_mid)
            if overlap > 0 or mid_diff <= tolerance:
                if overlap > best_overlap or (
                    abs(overlap - best_overlap) < 1e-6 and mid_diff < best_mid_diff
                ):
                    best_overlap = overlap
                    best_mid_diff = mid_diff
                    best_idx = scan_idx
            scan_idx += 1
        if best_idx is not None:
            used_dg[best_idx] = True
            matched += 1
            seg = segments[w_word["segment"]]
            seg_words = seg.get("words")
            if isinstance(seg_words, list):
                seg_words[w_word["word_index"]]["w_score"] = dg_sorted[best_idx]["confidence"]
        else:
            unmatched_whisper += 1

    unmatched_deepgram = used_dg.count(False)

    low_words: List[Dict[str, Any]] = []
    total_words_with_conf = 0
    total_low_words = 0

    for seg in segments:
        words = seg.get("words")
        if not isinstance(words, list):
            continue
        confidences: List[float] = []
        low_seconds = 0.0
        segment_duration = 0.0
        try:
            segment_duration = max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
        except Exception:
            segment_duration = 0.0

        for word in words:
            conf = word.get("w_score")
            try:
                w_start = float(word.get("start"))
                w_end = float(word.get("end"))
            except (TypeError, ValueError):
                continue
            duration = max(0.0, w_end - w_start)
            if conf is None:
                continue
            confidences.append(conf)
            total_words_with_conf += 1
            if conf < low_threshold:
                low_seconds += duration
                total_low_words += 1
                low_words.append(
                    {
                        "start": w_start,
                        "end": w_end,
                        "confidence": conf,
                        "speaker": seg.get("speaker"),
                    }
                )

        summary: Dict[str, Any] = {
            "word_count": len(confidences),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else None,
            "min_confidence": min(confidences) if confidences else None,
            "low_confidence_seconds": low_seconds,
            "low_confidence_ratio": (low_seconds / segment_duration) if (segment_duration and confidences) else None,
        }
        avg_conf_raw = summary["avg_confidence"]
        low_ratio_raw = summary["low_confidence_ratio"] or 0.0
        flag = None
        avg_conf = avg_conf_raw
        low_ratio = low_ratio_raw
        if avg_conf is not None:
            if avg_conf < low_threshold or (
                low_seconds >= max(0.0, min_low_duration) and low_ratio >= 0.5
            ):
                flag = "low"
            elif high_threshold and avg_conf >= high_threshold:
                flag = "high"
        if flag:
            seg["confidence_flag"] = flag

        def _round_val(value: Optional[float]) -> Optional[float]:
            if value is None:
                return None
            return round(value, 2)

        summary["avg_confidence"] = _round_val(summary["avg_confidence"])
        summary["min_confidence"] = _round_val(summary["min_confidence"])
        summary["low_confidence_seconds"] = _round_val(summary["low_confidence_seconds"])
        summary["low_confidence_ratio"] = _round_val(summary["low_confidence_ratio"])
        seg["confidence_summary"] = summary

    unclear_intervals = _merge_low_confidence_intervals(
        low_words,
        max(0.0, min_low_duration),
    )

    return {
        "word_count": total_words_with_conf,
        "matched_words": matched,
        "low_confidence_words": total_low_words,
        "unmatched_whisper_words": unmatched_whisper,
        "unmatched_deepgram_words": unmatched_deepgram,
        "unclear_intervals": unclear_intervals,
    }

def _fetch_segments_for_coverage(s3, bucket: str, stem: str, url_saves: Dict[str, Optional[str]], output: Dict[str, Any], ttl: int) -> Optional[list]:
    # 1) Inline
    segs = _extract_segments_from_dict(output)
    if isinstance(segs, list) and segs:
        return segs
    # 2) transcript_url
    t_url = output.get("transcript_url")
    if isinstance(t_url, str):
        try:
            data, _ = http_fetch(t_url)
            jd = json.loads(data.decode("utf-8"))
            segs = _extract_segments_from_dict(jd)
            if isinstance(segs, list) and segs:
                return segs
        except Exception:
            pass
    # 3) segments_url
    s_url = output.get("segments_url")
    if isinstance(s_url, str):
        try:
            data, _ = http_fetch(s_url)
            jd = json.loads(data.decode("utf-8"))
            if isinstance(jd, list) and jd:
                return jd
        except Exception:
            pass
    # 4) presign keys
    k = url_saves.get("transcript_key") or url_saves.get("segments_key")
    if k:
        try:
            url = presign_get(s3, bucket, k, ttl)
            data, _ = http_fetch(url)
            jd = json.loads(data.decode("utf-8"))
            segs = _extract_segments_from_dict(jd)
            if isinstance(segs, list) and segs:
                return segs
            if isinstance(jd, list) and jd:
                return jd
        except Exception:
            pass
    return None

def main():
    load_dotenv(find_dotenv(), override=True)
    args = parse_cli()
    audio_path = Path(args.file)
    if not audio_path.is_file():
        sys.exit(f"Input file not found: {audio_path}")

    if not (args.endpoint and args.api_key):
        sys.exit("Missing RunPod --endpoint/--api-key (or RUNPOD_* env)")

    pyannote_api_key = args.pyannote_api_key
    pyannote_json_in = getattr(args, "pyannote_json_in", None)
    if args.diarization and not pyannote_json_in and not pyannote_api_key:
        sys.exit("Diarization requested but no pyannote API key provided. Pass --pyannote-api-key or set PYANNOTE_API_KEY, or supply --pyannote-json-in.")

    s3 = build_s3_client(args)
    pending_uploads: List[Tuple[str, str, str, str]] = []

    def queue_upload(label: str, key: str, body: str, content_type: str):
        pending_uploads.append((label, key, body, content_type))

    audio_key = f"{args.prefix_audio}{audio_path.name}"
    if object_exists(s3, args.bucket, audio_key) and not args.force:
        print(f"[skip] Audio already exists: s3://{args.bucket}/{audio_key}")
    else:
        ctype = guess_content_type(audio_path)
        print(f"[upload] {audio_path} -> s3://{args.bucket}/{audio_key} ({ctype})")
        upload_file(s3, args.bucket, audio_key, audio_path, ctype)

    audio_url = presign_get(s3, args.bucket, audio_key, args.url_expiry)
    print(f"[presigned] {audio_url[:80]}...")

    payload: Dict[str, Any] = {
        "audio_file": audio_url,
        "model": args.model,
        "temperature": args.temperature,
        "align_output": args.align_output,
        "diarization": False,
        "debug": True,
    }
    # Merge extra params from JSON first (so CLI flags can override)
    if args.params_json:
        try:
            with open(args.params_json, "r", encoding="utf-8") as f:
                extra = json.load(f)
            if not isinstance(extra, dict):
                raise ValueError("params JSON must be an object")
            # Never allow external override of audio source
            for k in ("audio_file", "audio", "audio_url"):
                extra.pop(k, None)
            payload.update(_filter_allowed(extra))
        except Exception as e:
            sys.exit(f"Failed to load --params-json: {e}")

    # Apply explicit CLI overrides
    if args.language:
        payload["language"] = args.language
    if args.batch_size is not None:
        payload["batch_size"] = args.batch_size
    if args.beam_size is not None:
        payload["beam_size"] = args.beam_size
    if args.initial_prompt:
        payload["initial_prompt"] = args.initial_prompt
    if args.vad_onset is not None:
        payload["vad_onset"] = args.vad_onset
    if args.vad_offset is not None:
        payload["vad_offset"] = args.vad_offset
    if args.min_speakers is not None:
        payload["min_speakers"] = args.min_speakers
    if args.max_speakers is not None:
        payload["max_speakers"] = args.max_speakers
    if args.speaker_verification and args.speaker_sample:
        payload["speaker_verification"] = True
        samples = []
        for i, raw in enumerate(args.speaker_sample):
            # Presign s3:// URLs; also derive name from ?name= or #fragment if present
            u2, nm = presign_s3_url_if_needed(s3, raw, args.url_expiry)
            samples.append({
                "name": nm or f"spk{i+1}",
                "url": u2
            })
        payload["speaker_samples"] = samples

    print(f"[runpod] Submitting job to endpoint {args.endpoint} (sync={args.sync})")
    t0 = time.time()
    output = runpod_call(args.endpoint, args.api_key, payload, args.sync, args.debug)
    t1 = time.time()
    wall_seconds = t1 - t0

    if is_metadata_only(output) or not looks_like_transcript(output):
        raise RuntimeError(
            "RunPod returned no usable transcript payload; refusing to write transcript files.\n"
            + json.dumps(output, indent=2, ensure_ascii=False)
        )

    diarization_segments: List[Dict[str, Any]] = []
    speaker_totals: Dict[str, float] = {}
    merged_segments: List[Dict[str, Any]] = []
    pyannote_job: Optional[Dict[str, Any]] = None
    pyannote_job_id: Optional[str] = None
    pyannote_job_key: Optional[str] = None
    pyannote_job_json_text: Optional[str] = None
    deepgram_key: Optional[str] = None
    deepgram_json_text: Optional[str] = None
    deepgram_summary: Optional[Dict[str, Any]] = None

    transcript_segments = _load_segments_for_alignment(output, s3, args.bucket, args.url_expiry)
    if transcript_segments and not output.get("segments"):
        output["segments"] = transcript_segments

    if args.diarization:
        if pyannote_json_in:
            try:
                with open(pyannote_json_in, "r", encoding="utf-8") as f:
                    pyannote_job = json.load(f)
                print(f"[pyannote] Loaded diarization job payload from {pyannote_json_in}")
            except Exception as e:
                sys.exit(f"Failed to read --pyannote-json-in: {e}")
            diarization_segments = _extract_pyannote_segments(pyannote_job or {})
            pyannote_job_id = (
                str((pyannote_job or {}).get("jobId") or (pyannote_job or {}).get("id") or (pyannote_job or {}).get("job_id") or "")
            ) or None
        else:
            pyannote_job_id, pyannote_job, diarization_segments = run_pyannote_diarization(
                audio_url,
                pyannote_api_key,
                model=args.pyannote_model,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                exclusive=args.pyannote_exclusive,
                poll_interval=args.pyannote_poll_interval,
                timeout=args.pyannote_timeout,
            )

        if not transcript_segments:
            print("[warning] RunPod output provided no inline segments; attaching diarization without alignment.")

        merged_segments, speaker_totals = _apply_diarization_to_segments(transcript_segments, diarization_segments)
        if merged_segments:
            output["segments"] = merged_segments
        if diarization_segments:
            output["diarization"] = diarization_segments
        if speaker_totals:
            total_time = sum(speaker_totals.values())
            output["speaker_totals_seconds"] = speaker_totals
            speaker_breakdown = []
            for speaker, seconds in sorted(speaker_totals.items(), key=lambda item: item[1], reverse=True):
                entry = {"speaker": speaker, "seconds": seconds}
                if total_time > 0:
                    entry["percent"] = (seconds / total_time) * 100.0
                speaker_breakdown.append(entry)
            output["speakers"] = speaker_breakdown
            print("[pyannote] Speaker share (seconds): " + ", ".join(f"{s['speaker']}={s['seconds']:.1f}" for s in speaker_breakdown))

        seg_source = merged_segments or transcript_segments
        if seg_source:
            if not output.get("srt"):
                output["srt"] = _segments_to_srt(seg_source)
            if not output.get("vtt"):
                output["vtt"] = _segments_to_vtt(seg_source)
            subs = output.setdefault("subtitles", {})
            subs.setdefault("srt", output.get("srt"))
            subs.setdefault("vtt", output.get("vtt"))

        pyannote_meta = output.setdefault("pyannote", {})
        if pyannote_job_id:
            pyannote_meta["job_id"] = pyannote_job_id
        pyannote_meta["model"] = args.pyannote_model
        if pyannote_job is not None:
            pyannote_meta["job"] = pyannote_job
        pyannote_job_key = f"{args.prefix_transcripts}{audio_path.stem}.pyannote.job.json"
        if pyannote_job is not None:
            pyannote_job_json_text = json.dumps(pyannote_job, ensure_ascii=False, indent=2)

    # Deepgram integration (optional)
    use_deepgram = bool(args.deepgram_json_in or args.deepgram_api_key)
    if use_deepgram:
        if args.deepgram_json_in:
            try:
                deepgram_data = json.loads(args.deepgram_json_in.read_text(encoding="utf-8"))
                print(f"[deepgram] Loaded cached JSON from {args.deepgram_json_in}")
            except Exception as e:
                sys.exit(f"Failed to read --deepgram-json-in: {e}")
            deepgram_source = "cache"
        else:
            if not args.deepgram_api_key:
                sys.exit("Deepgram requested but no API key provided. Use --deepgram-api-key or --deepgram-json-in.")
            print(f"[deepgram] Requesting transcription confidence (model={args.deepgram_model})")
            deepgram_data = _call_deepgram_api(
                audio_url,
                args.deepgram_api_key,
                model=args.deepgram_model,
                tier=args.deepgram_tier,
                version=args.deepgram_version,
            )
            deepgram_source = "api"
        deepgram_words = _extract_deepgram_words(deepgram_data)
        merge_result = _merge_word_confidence_into_segments(
            output.get("segments") or [],
            deepgram_words,
            low_threshold=args.word_confidence_low_threshold,
            high_threshold=args.word_confidence_high_threshold,
            min_low_duration=args.low_confidence_min_dur,
        )
        deepgram_summary = {
            "source": deepgram_source,
            "model": args.deepgram_model,
            "tier": args.deepgram_tier,
            "version": args.deepgram_version,
            "word_confidence_low_threshold": args.word_confidence_low_threshold,
            "word_confidence_high_threshold": args.word_confidence_high_threshold,
            "low_confidence_min_duration": args.low_confidence_min_dur,
            "matched_words": merge_result["matched_words"],
            "unmatched_whisper_words": merge_result["unmatched_whisper_words"],
            "unmatched_deepgram_words": merge_result["unmatched_deepgram_words"],
            "low_confidence_word_count": merge_result["low_confidence_words"],
            "word_count_with_confidence": merge_result["word_count"],
            "low_confidence_intervals": len(merge_result["unclear_intervals"]),
        }
        deepgram_key = f"{args.prefix_transcripts}{audio_path.stem}.deepgram.json"
        deepgram_json_text = json.dumps(deepgram_data, ensure_ascii=False, indent=2)
        output["unclear_intervals"] = merge_result["unclear_intervals"]
        deepgram_meta = output.setdefault("deepgram", {})
        deepgram_meta.update(deepgram_summary)
        deepgram_meta["s3_key"] = deepgram_key
    else:
        output.setdefault("unclear_intervals", [])

    if pyannote_job_json_text and pyannote_job_key:
        output.setdefault("pyannote", {})["job_key"] = pyannote_job_key
        queue_upload("pyannote job JSON", pyannote_job_key, pyannote_job_json_text, "application/json")
    if deepgram_json_text and deepgram_key:
        queue_upload("Deepgram JSON", deepgram_key, deepgram_json_text, "application/json")

    transcript_key = f"{args.prefix_transcripts}{audio_path.stem}.json"
    vtt_key = f"{args.prefix_transcripts}{audio_path.stem}.vtt"
    srt_key = f"{args.prefix_transcripts}{audio_path.stem}.srt"

    # If worker provided keys/URL for artifacts, use them; otherwise, save any URLs.
    returned_keys = {
        "transcript_key": output.get("transcript_key"),
        "segments_key": output.get("segments_key"),
        "srt_key": output.get("srt_key"),
        "vtt_key": output.get("vtt_key"),
    }
    if any(returned_keys.values()):
        url_saves = returned_keys
    else:
        # Save artifacts referenced by URL (if any)
        url_saves = save_url_artifacts_if_any(s3, args.bucket, args.prefix_transcripts, audio_path.stem, output)

    # Write a small transcript JSON only if the worker didn't already create one
    transcript_store_key = url_saves.get("transcript_key") or transcript_key
    url_saves["transcript_key"] = transcript_store_key
    if args.diarization:
        output.setdefault("pyannote", {})["transcript_key"] = transcript_store_key
    raw_json = json.dumps(output, ensure_ascii=False, indent=2)
    queue_upload("transcript JSON", transcript_store_key, raw_json, "application/json")

    if output.get("segments"):
        segments_store_key = url_saves.get("segments_key") or f"{args.prefix_transcripts}{audio_path.stem}.segments.json"
        url_saves["segments_key"] = segments_store_key
        segments_json = json.dumps(output["segments"], ensure_ascii=False, indent=2)
        queue_upload("segments JSON", segments_store_key, segments_json, "application/json")

    srt_text = None
    vtt_text = None
    if isinstance(output, dict):
        if isinstance(output.get("srt"), str):
            srt_text = output["srt"]
        if isinstance(output.get("vtt"), str):
            vtt_text = output["vtt"]
        subs = output.get("subtitles")
        if isinstance(subs, dict):
            srt_text = srt_text or subs.get("srt")
            vtt_text = vtt_text or subs.get("vtt")

    if srt_text and not url_saves.get("srt_key"):
        queue_upload("SRT", srt_key, srt_text, "application/x-subrip")
    if vtt_text and not url_saves.get("vtt_key"):
        queue_upload("VTT", vtt_key, vtt_text, "text/vtt")

    if pending_uploads:
        for label, key, body, content_type in pending_uploads:
            print(f"[upload] {label} -> s3://{args.bucket}/{key}")
            put_text(s3, args.bucket, key, body, content_type)
        pending_uploads.clear()

    print("\n== Summary ==")
    print(f"Audio:      s3://{args.bucket}/{audio_key}")
    # Prefer worker-provided transcript key when available
    print("Transcript: ", end="")
    if url_saves.get("transcript_key"):
        print(f"s3://{args.bucket}/{url_saves['transcript_key']}")
    else:
        print(f"s3://{args.bucket}/{transcript_key}")
    # Also show presigned URLs if the worker provided them
    if isinstance(output, dict) and isinstance(output.get("transcript_url"), str):
        print(f"Transcript URL: {output['transcript_url']}")
    if url_saves.get("srt_key") or srt_text:
        print(f"SRT:        s3://{args.bucket}/{url_saves.get('srt_key') or srt_key}")
        if isinstance(output, dict) and isinstance(output.get("srt_url"), str):
            print(f"SRT URL:    {output['srt_url']}")
    if url_saves.get("vtt_key") or vtt_text:
        print(f"VTT:        s3://{args.bucket}/{url_saves.get('vtt_key') or vtt_key}")
        if isinstance(output, dict) and isinstance(output.get("vtt_url"), str):
            print(f"VTT URL:    {output['vtt_url']}")
    if url_saves.get("segments_key"):
        print(f"Segments:   s3://{args.bucket}/{url_saves.get('segments_key')}")
        if isinstance(output, dict) and isinstance(output.get("segments_url"), str):
            print(f"Segments URL: {output['segments_url']}")
    if args.diarization and pyannote_job_id:
        print(f"pyannote:   job {pyannote_job_id} (model {args.pyannote_model})")
    if args.diarization and pyannote_job_key:
        print(f"pyannote job JSON: s3://{args.bucket}/{pyannote_job_key}")
    if deepgram_key:
        detail = ""
        if deepgram_summary:
            detail = f" (model {deepgram_summary.get('model')}, low intervals {deepgram_summary.get('low_confidence_intervals')})"
        print(f"Deepgram:   s3://{args.bucket}/{deepgram_key}{detail}")

    # Coverage summary
    try:
        segs = _fetch_segments_for_coverage(s3, args.bucket, audio_path.stem, url_saves, output, args.url_expiry)
    except Exception:
        segs = None
    audio_secs = _get_audio_duration_secs_via_ffprobe(audio_path)
    if isinstance(segs, list) and segs:
        try:
            covered = sum((float(s.get("end", 0)) - float(s.get("start", 0))) for s in segs)
            max_end = max((float(s.get("end", 0)) for s in segs), default=0.0)
        except Exception:
            covered, max_end = 0.0, 0.0
        print("\n== Coverage Summary ==")
        print(f"Segments:   {len(segs)}")
        print(f"Covered:    {covered/60:.2f} min  (max_end ~ {max_end/60:.2f} min)")
        print(f"Wall time:  {wall_seconds:.2f} s")
        if audio_secs:
            rtf = audio_secs / max(1e-6, wall_seconds)
            print(f"RTF (audio/wall): {rtf:.1f}x for {audio_secs/60:.2f} min audio")
    
    # Download transcripts locally if requested
    if args.download_transcript:
        download_dir = Path(args.download_dir)
        print(f"\n== Downloading Transcripts ==")
        print(f"Download directory: {download_dir.absolute()}")
        
        downloaded_files = []
        
        # Download main transcript JSON
        transcript_s3_key = url_saves.get('transcript_key') or transcript_key
        local_transcript = download_dir / f"{audio_path.stem}.json"
        if download_file(s3, args.bucket, transcript_s3_key, local_transcript):
            print(f"[downloaded] {local_transcript}")
            downloaded_files.append(local_transcript)
        
        # Download SRT if available
        if url_saves.get('srt_key') or srt_text:
            srt_s3_key = url_saves.get('srt_key') or srt_key
            local_srt = download_dir / f"{audio_path.stem}.srt"
            if download_file(s3, args.bucket, srt_s3_key, local_srt):
                print(f"[downloaded] {local_srt}")
                downloaded_files.append(local_srt)
        
        # Download VTT if available
        if url_saves.get('vtt_key') or vtt_text:
            vtt_s3_key = url_saves.get('vtt_key') or vtt_key
            local_vtt = download_dir / f"{audio_path.stem}.vtt"
            if download_file(s3, args.bucket, vtt_s3_key, local_vtt):
                print(f"[downloaded] {local_vtt}")
                downloaded_files.append(local_vtt)
        
        # Download segments if available
        if url_saves.get('segments_key'):
            segments_s3_key = url_saves['segments_key']
            local_segments = download_dir / f"{audio_path.stem}.segments.json"
            if download_file(s3, args.bucket, segments_s3_key, local_segments):
                print(f"[downloaded] {local_segments}")
                downloaded_files.append(local_segments)
        
        if args.diarization and pyannote_job_key:
            local_pyannote_job = download_dir / f"{audio_path.stem}.pyannote.job.json"
            if download_file(s3, args.bucket, pyannote_job_key, local_pyannote_job):
                print(f"[downloaded] {local_pyannote_job}")
                downloaded_files.append(local_pyannote_job)
        if deepgram_key:
            local_deepgram = download_dir / f"{audio_path.stem}.deepgram.json"
            if download_file(s3, args.bucket, deepgram_key, local_deepgram):
                print(f"[downloaded] {local_deepgram}")
                downloaded_files.append(local_deepgram)

        if downloaded_files:
            print(f"\nSuccessfully downloaded {len(downloaded_files)} file(s) to {download_dir}")
        else:
            print("\nNo files were downloaded (none found or download failed)")

if __name__ == "__main__":
    main()
