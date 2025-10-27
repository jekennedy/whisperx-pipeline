# WhisperX Worker (RunPod) — PRD / Engineering Context

This document captures the current architecture, decisions, and operational guidance for the WhisperX RunPod worker, so another engineer or bot can jump in quickly.

## 1) Purpose & Scope

- High‑quality speech transcription using WhisperX with optional alignment and diarization.
- Runs as a RunPod Serverless worker, exposing a single handler that processes one audio job per request.
- Optimized for fast cold starts (optional preloading) and reliable runtime downloads with persistent caches.

Non‑goals: building a full API gateway, long‑running batch service, or adding unrelated model features.

## 2) High‑Level Architecture

- `src/init_and_run.py` — process entrypoint. Ensures cache dirs exist and starts the RunPod serverless loop.
- `src/rp_handler.py` — RunPod handler and orchestration. Validates input, downloads audio, calls prediction, and uploads artifacts to S3/R2 (if configured).
- `src/predict.py` — core transcription flow with WhisperX: model load, audio load, transcribe, optional alignment, optional diarization.
- `src/speaker_processing.py` — optional speaker verification/labeling helpers. Heavy models are lazy‑loaded.
- `tests/test_worker_local.py` — local CPU harness to run the worker without RunPod.

## 3) Build & Dependencies

- Multi‑stage Dockerfile (`Dockerfile`):
  - Base stage installs CUDA‑matching Torch (cu124) and system deps (`ffmpeg`, `git`).
  - General Python deps installed WITH dependencies (to avoid “whack‑a‑mole”).
  - Heavy ML stacks installed with `--no-deps` to avoid repinning Torch.
- Split requirements:
  - `builder/requirements.withdeps.txt` (install with deps):
    - Core: `runpod`, `boto3`, `requests`, `python-dotenv`, HF utils (`huggingface_hub`, `safetensors`, `packaging`, `filelock`, `tqdm`)
    - Audio/numerics: `ffmpeg-python`, `av`, `pandas`, `numpy>=1.24,<2.0`, `soundfile`, `librosa`, `nltk`
    - aiohttp stack: `aiohttp`, `aiosignal`, `frozenlist`, `multidict`, `yarl`, `attrs`
    - Pyannote ecosystem: `pyannote.audio==3.1.1`, `pyannote.core`, `pyannote.pipeline`, `omegaconf`, `hydra-core`, `einops`, `scikit-learn`, `scipy`, `torchmetrics`
  - `builder/requirements.nodeps.txt` (install with `--no-deps`):
    - `ctranslate2==4.6.0`, `faster-whisper==1.2.0`, `transformers==4.45.2`, `tokenizers==0.20.3`, `sentencepiece>=0.1.99`, `whisperx @ git+https://github.com/m-bain/whisperx.git`
- Optional model preloading at build:
  - `ARG PRELOAD_MODELS=0` (default off). When enabled, uses BuildKit cache mounts to avoid baking downloads into layers.
- VAD bundling: if `models/whisperx-vad-segmentation.bin` is present, it’s normalized to `/app/models/vad/…` and exposed via `VAD_MODEL_PATH`.

GPU runtime note (cuDNN):
- Set `LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH` to ensure cuDNN libraries bundled in Torch are discoverable on RunPod GPUs.

## 4) Model & Device Resolution

In `src/predict.py`:

- Device and precision:
  - `WHISPERX_DEVICE`: `auto` (default), `cpu`, or `cuda`.
  - `compute_type`: `float16` on GPU; `int8` on CPU.
  - One‑line log: `[Predict] device=… compute_type=…`.

- Model path/name resolver (robust):
  - If `WHISPERX_MODEL` or `WHISPERX_MODEL_NAME` set → use that (name or path).
  - Else if `WHISPERX_MODEL_DIR` contains `model.bin` → use that local dir (default `/app/models/faster-whisper-large-v3`).
  - Else → fallback to `'large-v3'` to auto‑download. One‑line log: `[Predict] whisper_arch=… (name|path)`.

- Caching:
  - Use a persistent volume and set: `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `TORCH_HOME`.
  - Local harness shortcut: mount host `~/.cache` to container `/root/.cache`.

## 5) Alignment, Diarization, Speaker Verification

Alignment:
- Do NOT use internal constants (removed upstream). Call public API only.
- `whisperx.load_align_model(…)` + `whisperx.align(…)` inside a try/except.
- If alignment fails or language unsupported → log and proceed without crashing.

Diarization:
- Use `from whisperx.diarize import DiarizationPipeline, assign_word_speakers`.
- Instantiate in a try/except and skip gracefully on failure:
  - `DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)`
  - `assign_word_speakers(diarize_segments, result)`
- Gated models require `HF_TOKEN` and accepting pyannote terms.

Speaker verification (SpeechBrain ECAPA):
- Lazy‑loaded via `get_ecapa()` in `speaker_processing.py` to avoid downloads at import time.
- Input shape (job `input`):
  - `speaker_verification: true`
  - `speaker_samples: [{ "url": "https://…" | "s3://…", "name": "optional" }, …]`
- URL handling:
  - Worker expects HTTP(S) URLs. If the client provides `s3://bucket/key`, the recommended client behavior is to presign to HTTPS before sending.
  - If caller passes `name` it is used as the display name. Otherwise, worker derives one from the sample URL in this precedence:
    1) query parameter `?name=swami`
    2) fragment `#swami`
    3) filename stem (truncated to 64 chars)
- Output: After diarization, segments are labeled with the most likely enrolled speaker name.

## 6) Inputs, Outputs, and Logging

- Payload shape expected by RunPod:
  ```json
  {
    "input": {
      "audio_file": "https://…",
      "language": null,
      "align_output": false,
      "diarization": false,
      "debug": true
    }
  }
  ```
- `debug: true` enables timing prints in `predict.py`.
- Console logging level can be controlled via env (see Deployment).
- If S3/R2 envs are set, artifacts are uploaded and the worker returns small JSON with keys and presigned URLs (when possible). Without S3, the worker returns a tiny inline preview (first 50 segments, no word arrays).

## 7) Local Testing

CPU harness (fast path):
- Build: `DOCKER_BUILDKIT=1 docker build -t whisperx-worker:local .`
- Run (persist caches):
  ```bash
  docker run --rm -it \
    -v "$PWD/tests:/app/tests" \
    -v "$PWD/samples:/app/samples" \
    -v "$HOME/.cache/whisperx-root:/root/.cache" \
    -e WHISPERX_MODEL=small \
    whisperx-worker:local \
    python3 /app/tests/test_worker_local.py
  ```

Apple Silicon note:
- Add `--platform=linux/amd64` to both `docker build` and `docker run` for local emulation.

## 8) Common Pitfalls & Fixes

- COPY failure for VAD file:
  - Symptom: `failed to calculate checksum … /models/vad/whisperx-vad-segmentation.bin not found`.
  - Fix: Dockerfile copies entire `models/` and normalizes VAD path.

- Slow builds due to model downloads:
  - Set `PRELOAD_MODELS=0` (default) and rely on runtime caches, or enable preloading with BuildKit cache mounts.

- Missing transitives (pandas/tz/dateutil, aiohttp, ffmpeg-python, av, safetensors):
  - Resolved by split requirements and explicit with‑deps list.

- Pyannote import errors / NumPy 2.0 breaks:
  - Install pyannote ecosystem with deps; pin `numpy>=1.24,<2.0`.

- Gated model downloads at import time:
  - Avoided by lazy‑loading (pyannote embedder, ECAPA) and removing unused imports.

- cuDNN load failure on GPU (RunPod):
  - Symptom: “Unable to load libcudnn_cnn.so” / `cudnnCreateConvolutionDescriptor`.
  - Fix: set `LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH`.
  - Temporary: force CPU with `WHISPERX_DEVICE=cpu`.

## 9) Deployment (RunPod)

Recommended endpoint envs:
- Caches (point to Network Storage Volume paths):
  - `HF_HOME=/runpod-volume/hf`
  - `HUGGINGFACE_HUB_CACHE=/runpod-volume/hf`
  - `TRANSFORMERS_CACHE=/runpod-volume/hf`
  - `TORCH_HOME=/runpod-volume/torch`
- Scratch and downloads (avoid filling root FS):
  - `JOBS_DIR=/runpod-volume/jobs`
  - `TMPDIR=/runpod-volume/tmp`
- Model selection:
  - `WHISPERX_MODEL=small` (for quick tests) or `large-v3` (quality)
  - Optionally leave unset to use preloaded local path.
- Device & cuDNN:
  - `WHISPERX_DEVICE=auto` (default) or `cuda`
  - `LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH`
- Diarization:
  - `HF_TOKEN=hf_…` (and accept pyannote terms).
- Optional S3/R2 for artifacts:
  - `STORAGE_ENDPOINT`, `STORAGE_BUCKET`, `STORAGE_ACCESS_KEY`, `STORAGE_SECRET_KEY`, `PREFIX_TRANSCRIPTS`.
  - `PRESIGN_TTL=3600` controls presigned URL expiry returned from the worker.
- Logging and diagnostics:
  - `DEBUG=1` sets console log level to DEBUG and defaults job‑level `debug` to true when not specified in input.
  - `CONSOLE_LOG_LEVEL=INFO|DEBUG|…` explicitly sets console verbosity.
  - `PRINT_MOUNTS=1` prints `df -h` at startup to confirm mounts.

## 10) Version Matrix (as of this build)

- Torch/Torchaudio: cu124 channel
- Faster‑Whisper: `1.2.0`
- WhisperX: latest `git` (HEAD)
- Transformers: `4.45.2`, Tokenizers: `0.20.3`, SentencePiece: `>=0.1.99`
- Pyannote Audio: `3.1.1` (+ related deps)
- NumPy: `>=1.24,<2.0`

## 11) TODO / Nice‑to‑Haves

- Add optional env‑driven console log level (e.g., `LOG_LEVEL=DEBUG`).
- Provide a smaller default Whisper model for tests in README examples.
- Pre‑bake a known‑good cuDNN path (`LD_LIBRARY_PATH`) in the Dockerfile.
- Small integration test that exercises ASR only (no HF/gated deps) in CI.

## 12) Quick Start (TL;DR)

1. Build locally (CPU):
   - `DOCKER_BUILDKIT=1 docker build -t whisperx-worker:local .`
2. Run harness with cached downloads:
   - `docker run --rm -it -v "$PWD/tests:/app/tests" -v "$PWD/samples:/app/samples" -v "$HOME/.cache/whisperx-root:/root/.cache" -e WHISPERX_MODEL=small whisperx-worker:local python3 /app/tests/test_worker_local.py`
3. Deploy to RunPod with a Network Volume and envs for caches, `WHISPERX_MODEL`, `LD_LIBRARY_PATH`, and `HF_TOKEN` if using diarization.

### Speaker Verification Quick Use

- Client should presign speaker sample URLs if stored on S3/R2 and pass them to the worker:

```json
{
  "input": {
    "audio_file": "https://…/minute.m4a",
    "diarization": true,
    "speaker_verification": true,
    "speaker_samples": [
      { "url": "https://…/swami-sample-1.m4a", "name": "swami" },
      { "url": "https://…/swami-sample-2.m4a#swami" }
    ]
  }
}
```

- If `name` is omitted, worker derives it from `?name=` or `#fragment`, else filename.
- Ensure `HF_TOKEN` is set and model terms accepted for pyannote diarization.
