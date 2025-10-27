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

import runpod
from . import rp_handler

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
runpod.serverless.start({"handler": rp_handler.run})
