#from cog import BasePredictor, Input, Path, BaseModel
try:
    # Prefer real cog if present (e.g. when running locally)
    from cog import BasePredictor, Input, Path, BaseModel
except ImportError:  # pragma: no cover
    # Fallback to local stub within the src package
    from src.cog_stub import BasePredictor, Input, Path, BaseModel
from typing import Any
from whisperx.audio import N_SAMPLES, log_mel_spectrogram
from scipy.spatial.distance import cosine
import gc
import math
import os
import shutil
import whisperx
import tempfile
import time
import torch


import logging
import sys
logger = logging.getLogger("predict")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler("container_log.txt", mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(console_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)






torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# Device selection: honor WHISPERX_DEVICE when set; otherwise auto-detect.
# Values: "cpu", "cuda", "auto" (default). "cuda" falls back to cpu if unavailable.
_DEVICE_ENV = os.getenv("WHISPERX_DEVICE", "auto").strip().lower()

def _resolve_device():
    if _DEVICE_ENV == "cpu":
        return "cpu"
    if _DEVICE_ENV == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"

device = _resolve_device()

# Choose compute_type appropriate to device: GPU -> float16, CPU -> int8
compute_type = "float16" if device == "cuda" else "int8"

try:
    # If TMPDIR is set, direct tempfile to use it (helps avoid /tmp space issues)
    _tmp = os.getenv("TMPDIR")
    if _tmp:
        os.makedirs(_tmp, exist_ok=True)
        tempfile.tempdir = _tmp
    forced = os.getenv("WHISPERX_DEVICE")
    if forced == "cuda" and device != "cuda":
        print("[Predict] Requested CUDA but not available; falling back to CPU", flush=True)
    print(f"[Predict] device={device} compute_type={compute_type}", flush=True)
except Exception:
    pass

# Allow runtime override of the model to speed validation or switch sizes.
# If WHISPERX_MODEL is set (e.g., "small", "medium", "large-v3"), use it directly.
# Otherwise, fall back to the local preloaded directory if present (WHISPERX_MODEL_DIR or default path).
_WHISPERX_MODEL_OVERRIDE = (
    os.getenv("WHISPERX_MODEL") or os.getenv("WHISPERX_MODEL_NAME") or ""
).strip()
_WHISPERX_MODEL_DIR = os.getenv(
    "WHISPERX_MODEL_DIR", 
    "/app/models/faster-whisper-large-v3"
).strip()

def _resolve_whisper_arch():
    # If override is set, use it as-is (model name or path)
    if _WHISPERX_MODEL_OVERRIDE:
        return _WHISPERX_MODEL_OVERRIDE
    # Prefer local dir only if it contains a model.bin
    if _WHISPERX_MODEL_DIR and os.path.isfile(os.path.join(_WHISPERX_MODEL_DIR, "model.bin")):
        return _WHISPERX_MODEL_DIR
    # Fallback to model name to trigger auto-download
    return os.getenv("WHISPERX_MODEL_NAME", "large-v3").strip() or "large-v3"

whisper_arch = _resolve_whisper_arch()
try:
    source_kind = "path" if os.path.isabs(whisper_arch) or "/" in whisper_arch else "name"
    print(f"[Predict] whisper_arch={whisper_arch} ({source_kind})", flush=True)
except Exception:
    pass


class Output(BaseModel):
    segments: Any
    detected_language: str


class Predictor(BasePredictor):
    def setup(self):
        source_folder = './models/vad'
        destination_folder = '../root/.cache/torch'
        file_name = 'whisperx-vad-segmentation.bin'

        os.makedirs(destination_folder, exist_ok=True)

        source_file_path = os.path.join(source_folder, file_name)
        if os.path.exists(source_file_path):
            destination_file_path = os.path.join(destination_folder, file_name)

            if not os.path.exists(destination_file_path):
                shutil.copy(source_file_path, destination_folder)

    def predict(
            self,
            audio_file: Path = Input(description="Audio file"),
            beam_size: int | None = Input(
                description="Beam size for decoding (None or 1 for greedy).",
                default=None
            ),
            patience: float | None = Input(
                description="Beam search patience (0-1); small >0 can marginally improve quality.",
                default=None
            ),
            length_penalty: float | None = Input(
                description="Beam search length penalty (e.g., 1.0).",
                default=None
            ),
            no_speech_threshold: float | None = Input(
                description="Threshold for no-speech probability (use defaults unless needed).",
                default=None
            ),
            log_prob_threshold: float | None = Input(
                description="Threshold for average log probability (use defaults unless needed).",
                default=None
            ),
            compression_ratio_threshold: float | None = Input(
                description="Threshold for gzip compression ratio (use defaults unless needed).",
                default=None
            ),
            language: str = Input(
                description="ISO code of the language spoken in the audio, specify None to perform language detection",
                default=None),
            language_detection_min_prob: float = Input(
                description="If language is not specified, then the language will be detected recursively on different "
                            "parts of the file until it reaches the given probability",
                default=0
            ),
            language_detection_max_tries: int = Input(
                description="If language is not specified, then the language will be detected following the logic of "
                            "language_detection_min_prob parameter, but will stop after the given max retries. If max "
                            "retries is reached, the most probable language is kept.",
                default=5
            ),
            initial_prompt: str = Input(
                description="Optional text to provide as a prompt for the first window",
                default=None),
            batch_size: int = Input(
                description="Parallelization of input audio transcription",
                default=64),
            temperature: float = Input(
                description="Temperature to use for sampling",
                default=0),
            vad_onset: float = Input(
                description="VAD onset",
                default=0.500),
            vad_offset: float = Input(
                description="VAD offset",
                default=0.363),
            align_output: bool = Input(
                description="Aligns whisper output to get accurate word-level timestamps",
                default=False),
            diarization: bool = Input(
                description="Assign speaker ID labels",
                default=False),
            huggingface_access_token: str = Input(
                description="To enable diarization, please enter your HuggingFace token (read). You need to accept "
                            "the user agreement for the models specified in the README.",
                default=None),
            min_speakers: int = Input(
                description="Minimum number of speakers if diarization is activated (leave blank if unknown)",
                default=None),
            max_speakers: int = Input(
                description="Maximum number of speakers if diarization is activated (leave blank if unknown)",
                default=None),
            debug: bool = Input(
                description="Print out compute/inference times and memory usage information",
                default=False),
            speaker_verification: bool = Input(
                description="Enable speaker verification",
                default=False),
            speaker_samples: list = Input(
                description="List of speaker samples for verification. Each sample should be a dict with 'url' and "
                            "optional 'name' and 'file_path'. If 'name' is not provided, the file name (without "
                            "extension) is used. If 'file_path' is provided, it will be used directly.",
                default=[]
            )
    ) -> Output:
        with torch.inference_mode():
            # Compatibility with WhisperX TranscriptionOptions: expects 'temperatures' (list)
            asr_options = {
                "initial_prompt": initial_prompt,
                "condition_on_previous_text": True,
            }
            # Beam vs greedy: in beam mode, do NOT pass temperatures
            _bs = None
            if beam_size is not None:
                try:
                    _bs = int(beam_size)
                except Exception:
                    _bs = None
            if _bs is not None and _bs >= 2:
                asr_options["beam_size"] = _bs
            else:
                # Greedy/sampling path uses 'temperatures' list as expected by WhisperX TranscriptionOptions
                asr_options["temperatures"] = [float(temperature)]
            # Optional advanced knobs
            if patience is not None:
                try:
                    p = float(patience)
                    if p >= 0:
                        asr_options["patience"] = p
                except Exception:
                    pass
            if length_penalty is not None:
                try:
                    lp = float(length_penalty)
                    asr_options["length_penalty"] = lp
                except Exception:
                    pass
            if no_speech_threshold is not None:
                try:
                    ns = float(no_speech_threshold)
                    asr_options["no_speech_threshold"] = ns
                except Exception:
                    pass
            if log_prob_threshold is not None:
                try:
                    lpt = float(log_prob_threshold)
                    asr_options["log_prob_threshold"] = lpt
                except Exception:
                    pass
            if compression_ratio_threshold is not None:
                try:
                    crt = float(compression_ratio_threshold)
                    asr_options["compression_ratio_threshold"] = crt
                except Exception:
                    pass

            vad_options = {
                "vad_onset": vad_onset,
                "vad_offset": vad_offset
            }

            audio_duration = get_audio_duration(audio_file)

            if language is None and language_detection_min_prob > 0 and audio_duration > 30000:
                segments_duration_ms = 30000

                language_detection_max_tries = min(
                    language_detection_max_tries,
                    math.floor(audio_duration / segments_duration_ms)
                )

                segments_starts = distribute_segments_equally(audio_duration, segments_duration_ms,
                                                              language_detection_max_tries)

                print("Detecting languages on segments starting at " + ', '.join(map(str, segments_starts)))

                detected_language_details = detect_language(audio_file, segments_starts, language_detection_min_prob,
                                                            language_detection_max_tries, asr_options, vad_options)

                detected_language_code = detected_language_details["language"]
                detected_language_prob = detected_language_details["probability"]
                detected_language_iterations = detected_language_details["iterations"]

                print(f"Detected language {detected_language_code} ({detected_language_prob:.2f}) after "
                      f"{detected_language_iterations} iterations.")

                language = detected_language_details["language"]

            start_time = time.time_ns() / 1e6

            model = whisperx.load_model(whisper_arch, device, compute_type=compute_type, language=language,
                                        asr_options=asr_options, vad_options=vad_options)

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to load model: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6

            audio = whisperx.load_audio(audio_file)

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to load audio: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6

            result = model.transcribe(audio, batch_size=batch_size)
            detected_language = result["language"]

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to transcribe: {elapsed_time:.2f} ms")

            gc.collect()
            torch.cuda.empty_cache()
            del model

            if align_output:
                try:
                    result = align(audio, result, debug)
                except Exception as e:
                    print(f"[Predict] alignment skipped: {e}", flush=True)

            if diarization:
                result = diarize(audio, result, debug, huggingface_access_token, min_speakers, max_speakers)

            if debug:
                print(f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")

        return Output(
            segments=result["segments"],
            detected_language=detected_language
        )


def get_audio_duration(file_path):
    import librosa
    # duration in seconds → convert to milliseconds
    d = librosa.get_duration(filename=str(file_path))
    return int(d * 1000)


def detect_language(full_audio_file_path, segments_starts, language_detection_min_prob,
                    language_detection_max_tries, asr_options, vad_options, iteration=1):
    model = whisperx.load_model(whisper_arch, device, compute_type=compute_type, asr_options=asr_options,
                                vad_options=vad_options)

    start_ms = segments_starts[iteration - 1]

    audio_segment_file_path = extract_audio_segment(full_audio_file_path, start_ms, 30000)

    audio = whisperx.load_audio(audio_segment_file_path)

    model_n_mels = model.model.feat_kwargs.get("feature_size")
    segment = log_mel_spectrogram(audio[: N_SAMPLES],
                                  n_mels=model_n_mels if model_n_mels is not None else 80,
                                  padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])
    encoder_output = model.model.encode(segment)
    results = model.model.model.detect_language(encoder_output)
    language_token, language_probability = results[0][0]
    language = language_token[2:-2]

    print(f"Iteration {iteration} - Detected language: {language} ({language_probability:.2f})")

    audio_segment_file_path.unlink()

    gc.collect()
    torch.cuda.empty_cache()
    del model

    detected_language = {
        "language": language,
        "probability": language_probability,
        "iterations": iteration
    }

    if language_probability >= language_detection_min_prob or iteration >= language_detection_max_tries:
        return detected_language

    next_iteration_detected_language = detect_language(full_audio_file_path, segments_starts,
                                                       language_detection_min_prob, language_detection_max_tries,
                                                       asr_options, vad_options, iteration + 1)

    if next_iteration_detected_language["probability"] > detected_language["probability"]:
        return next_iteration_detected_language

    return detected_language


def extract_audio_segment(input_file_path, start_time_ms, duration_ms):
    """
    Extract a segment using librosa/soundfile to avoid AudioSegment/pyaudioop.
    Returns a temporary file path with the same extension as the input.
    """
    import librosa
    import soundfile as sf
    from pathlib import Path

    input_file_path = Path(input_file_path) if not isinstance(input_file_path, Path) else input_file_path

    offset_s = start_time_ms / 1000.0
    duration_s = duration_ms / 1000.0

    # Preserve original sampling rate; mono=False preserves channels
    y, sr = librosa.load(str(input_file_path), sr=None, mono=False, offset=offset_s, duration=duration_s)

    # librosa returns (n,) or (channels, n); soundfile expects (n,) or (n, channels)
    if y.ndim == 2:
        y_to_write = y.T
    else:
        y_to_write = y

    suffix = input_file_path.suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        sf.write(str(tmp_path), y_to_write, sr)

    return tmp_path


def distribute_segments_equally(total_duration, segments_duration, iterations):
    available_duration = total_duration - segments_duration

    if iterations > 1:
        spacing = available_duration // (iterations - 1)
    else:
        spacing = 0

    start_times = [i * spacing for i in range(iterations)]

    if iterations > 1:
        start_times[-1] = total_duration - segments_duration

    return start_times


def align(audio, result, debug):
    start_time = time.time_ns() / 1e6

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device,
                            return_char_alignments=False)

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        print(f"Duration to align output: {elapsed_time:.2f} ms")

    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    return result


def diarize(audio, result, debug, huggingface_access_token, min_speakers, max_speakers):
    start_time = time.time_ns() / 1e6
    try:
        # Newer WhisperX exposes diarization via submodule
        from whisperx.diarize import DiarizationPipeline, assign_word_speakers

        # Try with explicit model_name first, fall back if signature changed
        try:
            diarize_model = DiarizationPipeline(
                model_name='pyannote/speaker-diarization@2.1',
                use_auth_token=huggingface_access_token,
                device=device,
            )
        except TypeError:
            diarize_model = DiarizationPipeline(
                use_auth_token=huggingface_access_token,
                device=device,
            )

        diarize_segments = diarize_model(
            audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        result = assign_word_speakers(diarize_segments, result)

        if debug:
            elapsed_time = time.time_ns() / 1e6 - start_time
            print(f"Duration to diarize segments: {elapsed_time:.2f} ms")

        gc.collect()
        torch.cuda.empty_cache()
        del diarize_model
        return result
    except Exception as e:
        print(f"[Predict] diarization skipped: {e}", flush=True)
        return result

def identify_speaker_for_segment(segment_embedding, known_embeddings, threshold=0.1):
    """
    Compare segment_embedding to known speaker embeddings using cosine similarity.
    Returns the speaker name with the highest similarity above the threshold,
    or "Unknown" if none match.
    """
    best_match = "Unknown"
    best_similarity = -1
    for speaker, known_emb in known_embeddings.items():
        similarity = 1 - cosine(segment_embedding, known_emb)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = speaker
    if best_similarity >= threshold:
        return best_match, best_similarity
    else:
        return "Unknown", best_similarity
