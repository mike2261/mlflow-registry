"""Register the shortlisted open-weight STT/TTS models into the MLflow registry.

Source of the list: reports/Open source STT TTS Vietnamese English.md
(revised shortlist, 26 Sep 2026). Re-runnable: each run creates a new version.

    uv run python scripts/register_shortlist.py            # all
    uv run python scripts/register_shortlist.py kokoro-82m # subset by name
"""

import logging
import sys
import time
from dataclasses import dataclass, field

from mlflow_registry import ModelRegistry

log = logging.getLogger("register")

# Skip duplicate weight formats and eval dumps; keep configs, tokenizers, READMEs.
_SAFETENSORS_ONLY = ["*.json", "*.txt", "*.md", "*.py", "model.safetensors", "*.tiktoken"]


@dataclass(frozen=True)
class Candidate:
    name: str
    source: str
    tags: dict[str, str]
    fetch_options: dict = field(default_factory=dict)


SHORTLIST = [
    # --- STT -------------------------------------------------------------------
    Candidate(
        "qwen3-asr-1.7b", "hf:Qwen/Qwen3-ASR-1.7B",
        {"task": "stt", "languages": "vi,en,multi", "license": "Apache-2.0",
         "format": "safetensors", "runtime": "transformers,vllm"},
    ),
    Candidate(
        "granite-speech-4.1-2b", "hf:ibm-granite/granite-speech-4.1-2b",
        {"task": "stt", "languages": "en", "license": "Apache-2.0",
         "format": "safetensors", "runtime": "transformers,vllm"},
        {"ignore_patterns": [".eval_results/*"]},
    ),
    Candidate(
        "gipformer1.5-68m-rnnt", "hf:g-group-ai-lab/gipformer1.5-68M-rnnt",
        {"task": "stt", "languages": "vi", "license": "MIT (asserted, no LICENSE file)",
         "format": "onnx", "runtime": "onnxruntime,sherpa-onnx"},
    ),
    Candidate(
        "parakeet-ctc-0.6b-vietnamese", "hf:nvidia/parakeet-ctc-0.6b-Vietnamese",
        {"task": "stt", "languages": "vi", "license": "NVIDIA Open Model License",
         "format": "nemo", "runtime": "nemo"},
    ),
    Candidate(
        "whisper-large-v3", "hf:openai/whisper-large-v3",
        {"task": "stt", "languages": "multi", "license": "Apache-2.0",
         "format": "safetensors", "runtime": "transformers,faster-whisper",
         "role": "baseline"},
        {"allow_patterns": _SAFETENSORS_ONLY},
    ),
    Candidate(
        "phowhisper-large", "hf:vinai/PhoWhisper-large",
        {"task": "stt", "languages": "vi", "license": "BSD-3-Clause",
         "format": "pytorch-bin", "runtime": "transformers",
         "note": "Whisper large fine-tuned on 844 h of Vietnamese (VinAI)"},
    ),
    # --- TTS -------------------------------------------------------------------
    Candidate(
        "voxcpm2", "hf:openbmb/VoxCPM2",
        {"task": "tts", "languages": "vi,en,multi", "license": "Apache-2.0",
         "format": "safetensors", "runtime": "voxcpm"},
    ),
    Candidate(
        "vieneu-tts-v3-turbo", "hf:pnnbao-ump/VieNeu-TTS-v3-Turbo",
        {"task": "tts", "languages": "vi,en", "license": "Apache-2.0",
         "format": "safetensors,onnx,gguf", "runtime": "vieneu"},
    ),
    Candidate(
        "kokoro-82m", "hf:hexgrad/Kokoro-82M",
        {"task": "tts", "languages": "en,multi", "license": "Apache-2.0",
         "format": "pth", "runtime": "kokoro"},
    ),
    Candidate(
        "qwen3-tts-1.7b-base", "hf:Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        {"task": "tts", "languages": "en,multi", "license": "Apache-2.0",
         "format": "safetensors", "runtime": "qwen-tts,vllm-omni"},
    ),
]


def main(only: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    registry = ModelRegistry()
    todo = [c for c in SHORTLIST if not only or c.name in only]
    failures = 0
    for c in todo:
        t0 = time.time()
        log.info("registering %s from %s", c.name, c.source)
        try:
            v = registry.register(c.name, c.source, tags=c.tags, **c.fetch_options)
        except Exception:
            failures += 1
            log.exception("FAILED %s", c.name)
            continue
        log.info("done %s -> version %s in %.0fs", c.name, v, time.time() - t0)
    log.info("finished: %d ok, %d failed", len(todo) - failures, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
