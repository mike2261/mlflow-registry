# MLflow model serving for the nine registered speech models

Date: 2026-09-28. Status: approved in conversation (scope: all nine models; deployment form:
one Docker image per model, robo-be style).

## Goal

Expose every registered STT and TTS model as a REST endpoint on the dev server (RTX 3090,
24 GB), served by `mlflow models serve`, so a client can call `POST /invocations` with audio
or text and get a transcript or a WAV back. The registry stays the single source of truth:
the serving containers pull both the serving wrapper and the weights from MLflow.

## Why this shape

* The registry stores raw Hugging Face snapshots (`log_artifacts`), not an MLflow flavor.
  `mlflow models serve` needs a `python_function` flavor, so each model gets a thin
  `mlflow.pyfunc.PythonModel` **adapter** that loads its own runtime.
* The nine models span seven runtimes with incompatible pins (qwen-asr and vieneu pin
  `transformers==4.57.6`, qwen-tts pins `4.57.3`, NeMo and voxcpm want newer torch).
  One image per model is the only isolation that works for all of them. robo-be already
  uses this pattern (shared CUDA base + thin per-worker images), so the layout is familiar.
* Weights are not copied into the pyfunc artifact. The pyfunc holds a pointer
  (`name`, `version`) and downloads the snapshot at load time into a shared volume, so the
  27 GB in MinIO stays single-copy and re-registering a wrapper costs kilobytes.

## Registry model

Each registered model gets a **version 2**: the pyfunc wrapper. Version 1 stays the raw
weights. A new alias `serving` points at the wrapper; `production` and `staging` remain
reserved for raw-weight versions so the consumer contract in the README does not change.

```
whisper-large-v3
  v1  raw weights          tags: source, hf_revision, task, runtime, ...
  v2  pyfunc wrapper       tags: flavor=pyfunc  wraps_version=1  task=stt  serving_image=whisper-large-v3
  aliases: serving -> 2
```

`mlflow models serve -m models:/whisper-large-v3@serving` is the only URI a container needs.

## Uniform REST contract

All nine models accept the MLflow `dataframe_records` / `inputs` payload and return
`{"predictions": [...]}`. Audio travels as base64 WAV.

STT input columns: `audio_b64` (required), `language` (optional, `vi` / `en` / omitted for
auto). STT output per row: `text`, `language` (detected or echoed, may be null).

TTS input columns: `text` (required), `voice` (optional preset name), `language` (optional),
`ref_audio_b64` and `ref_text` (optional, for cloning models). TTS output per row:
`audio_b64` (16-bit PCM WAV), `sample_rate`.

Per-model notes:

| Model | Runtime call | Notes |
|---|---|---|
| qwen3-asr-1.7b | `qwen_asr.Qwen3ASRModel.from_pretrained(dir)`; `.transcribe((np,16k), language=)` | language mapped `vi`→`Vietnamese`, `en`→`English` |
| granite-speech-4.1-2b | transformers `AutoModelForSpeechSeq2Seq` + `AutoProcessor`, chat template with `<|audio|>` | English only, needs `peft` |
| whisper-large-v3 | transformers ASR `pipeline` | `language` passed as generate kwarg |
| gipformer1.5-68m-rnnt | `sherpa_onnx.OfflineRecognizer.from_transducer(encoder.onnx, decoder.onnx, joiner.onnx, tokens.txt)` | CPU, no GPU reservation |
| parakeet-ctc-0.6b-vietnamese | `nemo.collections.asr.models.ASRModel.restore_from(*.nemo)`; `.transcribe([wav paths])` | writes temp WAVs |
| voxcpm2 | `voxcpm.VoxCPM.from_pretrained(dir, load_denoiser=False)`; `.generate(text, reference_wav_path=)` | 48 kHz; `voice` = description prefix |
| vieneu-tts-v3-turbo | `vieneu.Vieneu(mode="v3turbo", backbone_repo=dir)`; `.infer(text, voice=)` | 48 kHz; codec repo still fetched from HF at first start |
| kokoro-82m | `kokoro.KModel(repo_id, config=dir/config.json, model=dir/kokoro-v1_0.pth)` + `KPipeline(lang_code, model=)`; voice = `dir/voices/<v>.pt` | 24 kHz; needs `espeak-ng` |
| qwen3-tts-1.7b-base | `qwen_tts.Qwen3TTSModel.from_pretrained(dir)`; `.generate_voice_clone(text, language, ref_audio, ref_text)` | Base model **requires** `ref_audio_b64` + `ref_text` |

## Code layout

```
src/mlflow_registry/serving/
  __init__.py
  audio.py        base64 WAV <-> float32 mono 16 kHz; np -> base64 WAV
  weights.py      ensure_weights(name, version, cache_root) -> Path   (download once, marker file)
  base.py         SttModel / TtsModel (PythonModel): load_context, predict, signatures
  catalog.py      SERVING: name -> ServingSpec(task, wrapper import path, port, gpu)
  stt_qwen3.py  stt_granite.py  stt_whisper.py  stt_gipformer.py  stt_parakeet.py
  tts_voxcpm.py tts_vieneu.py   tts_kokoro.py   tts_qwen3.py
scripts/register_serving.py    log pyfunc v2 + alias `serving` for each catalog entry
scripts/serve.sh               build / up / down / logs / ps / invoke wrappers around compose
scripts/invoke.py              tiny client: wav or text in, transcript or wav out
serving/Dockerfile.base        CUDA 12.8 runtime + Python 3.12 + ffmpeg, libsndfile1, espeak-ng
                               + torch 2.8.0/torchaudio 2.8.0 (cu128) + mlflow
serving/Dockerfile.model       ARG MODEL; installs serving/requirements/<MODEL>.txt + this package
serving/requirements/<model>.txt  nine files, one per image
docker-compose.serving.yaml    nine services, profile per model, shared volume, GPU reservation
```

Wrappers import their runtime **inside** `load_context`, so registering a pyfunc from the
laptop needs only the base dependencies. cloudpickle stores the wrapper class by reference to
`mlflow_registry.serving.*`, which every image installs, so wrapper code changes ship with
the image and do not require re-registration.

## Runtime layout on the server

```
docker compose -f docker-compose.yaml -f docker-compose.serving.yaml --profile whisper-large-v3 up -d
                                          │
  mr-mlflow:5000 ◄── models:/whisper-large-v3@serving ── serve-whisper-large-v3 (port 5005 -> 8080)
       │                                                    │  load_context: ensure_weights()
  mr-minio:9000 ◄────────── weights download ───────────────┘  -> /models/whisper-large-v3/1
                                                               (volume serving_models, shared)
```

* Tracking URI inside the compose network is `http://mlflow:5000`; presigned MinIO URLs
  resolve there, so no proxy-multipart workaround is needed.
* Ports 5001..5009 on the host, one per model, in catalog order. Container port is 8080.
* `MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT=600` so long TTS requests are not cut at 60 s.
  Model loading happens before uvicorn starts listening, so it is not subject to the timeout.
* Healthcheck `GET /health` with `start_period` 900 s to cover the first weights download.
* `restart: unless-stopped`; GPU reservation on every service except gipformer.
* VRAM budget: the nine models resident together need about 28 GB, more than the card has.
  The launcher starts models one profile at a time; the README lists safe combinations.

## Testing

* Unit tests without runtimes: audio round-trip, `ensure_weights` marker logic, pyfunc
  predict through a `FakeStt` / `FakeTts` subclass loaded with `mlflow.pyfunc.load_model`,
  catalog integrity (unique ports, importable wrapper paths).
* Live-stack test: `register_serving` on a throwaway `test-*` model with the fake wrapper,
  then `models:/<name>@serving` loads and predicts.
* Docker smoke on the laptop (CPU): build the base and the `kokoro-82m` and
  `gipformer1.5-68m-rnnt` images, run against the local stack, curl `/invocations`.
* Dev server: build all images, register wrappers, bring up each model in turn, curl once.

## Out of scope

Streaming, batching across requests, authentication, autoscaling, and converting the
VieNeu codec into a registered model. robo-be's NATS contract is unchanged; this is a
parallel REST path for evaluation and quick trials.
