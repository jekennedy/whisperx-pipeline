import os, pathlib, sys, subprocess

# ------------------------------------------------------------
# Ensure CUDA/cuDNN libraries are discoverable before imports
# ------------------------------------------------------------
def _ensure_ld_library_path():
    try:
        base = f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"
        candidates = [
            f"{base}/torch/lib",
            f"{base}/nvidia/cudnn/lib",
            f"{base}/nvidia/cublas/lib",
            f"{base}/nvidia/cusparse/lib",
            f"{base}/nvidia/cuda_runtime/lib",
            f"{base}/nvidia/nvjitlink/lib",
        ]
        current = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in current.split(":") if p]
        # Prepend unique candidates
        for d in candidates:
            if d and d not in parts:
                parts.insert(0, d)
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
    except Exception:
        pass

_ensure_ld_library_path()

import traceback

import runpod

# Import the RunPod handler with a safety net so that any
# import-time error is printed to stdout instead of crashing silently.
try:
    from . import rp_handler  # noqa: F401
    _IMPORT_OK = True
    print("[INIT] rp_handler import: OK", flush=True)
except Exception:
    _IMPORT_OK = False
    print("[INIT] rp_handler import: FAILED", flush=True)
    traceback.print_exc()

for p in [os.getenv("HF_HOME"),
          os.getenv("HUGGINGFACE_HUB_CACHE"),
          os.getenv("TRANSFORMERS_CACHE"),
          os.getenv("TORCH_HOME"),
          os.getenv("TMPDIR"),
          os.getenv("JOBS_DIR")]:
    if p:
        pathlib.Path(p).mkdir(parents=True, exist_ok=True)

# Optional: sanity log
print("[INIT] Caches:",
      "HF_HOME=", os.getenv("HF_HOME"),
      "HUGGINGFACE_HUB_CACHE=", os.getenv("HUGGINGFACE_HUB_CACHE"),
      "TRANSFORMERS_CACHE=", os.getenv("TRANSFORMERS_CACHE"),
      "TORCH_HOME=", os.getenv("TORCH_HOME"),
      "TMPDIR=", os.getenv("TMPDIR"),
      "JOBS_DIR=", os.getenv("JOBS_DIR", "/jobs"),
      flush=True)

# Lightly report presence (not values) of important RUNPOD env to help diagnostics
_rp_env_keys = [
    "RUNPOD_POD_ID",
    "RUNPOD_API_BASE_URL",
    "RUNPOD_HANDLER",
    "RUNPOD_ENDPOINT_ID",
]
_rp_env_present = {k: ("set" if os.getenv(k) else "missing") for k in _rp_env_keys}
print("[INIT] RunPod env:", _rp_env_present, flush=True)

# Optional: print mount points and disk usage when PRINT_MOUNTS=1 or DEBUG=true
_print_mounts = os.getenv("PRINT_MOUNTS", "").strip().lower() in {"1", "true", "yes", "on"} or \
                os.getenv("DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
if _print_mounts:
    try:
        out = subprocess.check_output(["df", "-h"], text=True)
        print("[INIT] df -h:\n" + out, flush=True)
    except Exception as e:
        print(f"[INIT] df -h failed: {e}", flush=True)

# Start serverless handler (clean separation from rp_handler import)
if _IMPORT_OK:
    runpod.serverless.start({"handler": rp_handler.run})
else:
    # Without the handler we can't start the worker; exit non-zero so the
    # control plane surfaces the error and pulls logs for visibility.
    sys.stderr.write("[INIT] Exiting due to rp_handler import failure.\n")
    sys.stderr.flush()
    sys.exit(1)
