# tests/test_worker_local.py
import os
import json
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # reads .env if present; falls back to shell envs

# Ensure imports can find src/ when running from tests/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Configure a small local audio file for testing
SAMPLE_AUDIO = Path(os.getenv("SAMPLE_AUDIO", "samples/minute.m4a"))
if not SAMPLE_AUDIO.is_file():
    raise SystemExit(f"Sample audio not found: {SAMPLE_AUDIO.resolve()}")

import src.rp_handler as rp_handler  # noqa: E402

def _fake_download_to_jobs(job_id, _url):
    job_dir = Path("/jobs") / job_id / "input_objects"
    job_dir.mkdir(parents=True, exist_ok=True)
    dst = job_dir / SAMPLE_AUDIO.name
    shutil.copy(SAMPLE_AUDIO, dst)
    return str(dst)

rp_handler._download_to_jobs = _fake_download_to_jobs

def main():
    job = {
        "id": "local-test-001",
        "input": {
            "audio_file": str(SAMPLE_AUDIO.resolve()),
            "model": os.getenv("WHISPER_MODEL", "large-v3"),
            "language": os.getenv("WHISPER_LANG") or None,
            "diarization": os.getenv("WHISPER_DIARIZATION", "false").lower() == "true",
            "align_output": os.getenv("WHISPER_ALIGN", "true").lower() == "true",
            "temperature": float(os.getenv("WHISPER_TEMPERATURE", "0.0")),
            "debug": True,
        }
    }

    out = rp_handler.run(job)
    print(json.dumps(out, indent=2, ensure_ascii=False))

    if isinstance(out, dict) and "output" in out:
        o = out["output"]
        ok = any(k in o for k in ("segments_key", "segments_url", "segments"))
        if not ok:
            raise SystemExit("Run completed but no segments info found in output.")
    else:
        ok = any(k in out for k in ("segments_key", "segments_url", "segments"))
        if not ok:
            raise SystemExit("Run completed but no segments info found in result.")

if __name__ == "__main__":
    main()
