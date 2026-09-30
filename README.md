# mlflow-registry

A self-hosted **model registry** for the speech stack (STT and TTS, Vietnamese and English).
It answers one question for every other repo: *"give me the weights behind `<model>@production`"*,
so consumers never hard-code file paths or Hugging Face repo ids, and swapping or rolling back a
model is a one-line alias change instead of a redeploy.

Three pieces live here:

| Piece | What | Where |
|---|---|---|
| **Stack** | MLflow tracking server + Postgres (metadata) + MinIO (weights), one `docker compose up` | `docker-compose.yaml`, `.env` |
| **Client library** | `ModelRegistry` class and source adapters (`fetchers`) | `src/mlflow_registry/` |
| **CLI** | `mlflow-registry register / promote / resolve / download / list` | `src/mlflow_registry/cli.py` |

Plus `scripts/register_shortlist.py`, which registers the nine shortlisted models from the
research report in `reports/`, and a **serving layer** (section 5): a pyfunc wrapper per model,
one Docker image per model, `mlflow models serve` behind `POST /invocations`.

---

## 1. Architecture

```
                      ┌──────────────────────────────────────────────────────────┐
                      │  docker compose network                                  │
  laptop / devserver  │                                                          │
  ─────────────────   │   mr-mlflow  (ghcr.io/mlflow/mlflow:v3.16.0)             │
                      │   ├─ --backend-store-uri  postgresql://…@postgres:5432   │
  mlflow-registry ───►│   ├─ --artifacts-destination s3://mlflow-artifacts       │
  (CLI / library)     │   └─ --serve-artifacts   ← all weights flow through here │
        :5000         │          │                          │                    │
                      │     mr-postgres:5432            mr-minio:9000 (S3 API)   │
                      │     volume pgdata               mr-minio:9001 (console)  │
                      │     (names, versions,           volume miniodata         │
                      │      aliases, tags)             (weights, 27 GB)         │
                      └──────────────────────────────────────────────────────────┘
```

* **MLflow** is the only door. Clients talk to `:5000`; MLflow proxies weight uploads and
  downloads to MinIO (`--serve-artifacts`). Consumers never need MinIO credentials.
* **Postgres** holds the registry: registered model → versions → aliases and tags.
* **MinIO** holds the bytes under `s3://mlflow-artifacts/<experiment>/<run>/artifacts/model/`.
* **`minio-init`** creates the bucket once and exits.

### Registry model

```
Registered model  "qwen3-asr-1.7b"
  ├─ version 1   tags: source=hf:Qwen/Qwen3-ASR-1.7B  hf_revision=7278e1e7…  license=Apache-2.0
  │              task=stt  languages=vi,en,multi  format=safetensors  runtime=transformers,vllm
  ├─ version 2   …
  └─ aliases:  production → 1,  staging → 2
```

* **One registered model per model family**, one **version** per registration. Versions are immutable.
* **Aliases** (`production`, `staging`) are the seam consumers hang on to. `promote` moves an alias;
  moving it back is the rollback. Stages are not used (deprecated in MLflow).
* Every version carries **provenance tags**: `source` (the fetcher spec), `hf_revision` (exact Hub
  commit) and whatever the registrar adds (`license`, `task`, `languages`, `format`, `runtime`).

### Client library (`src/mlflow_registry/`)

```
config.py     Config + load_config()      .env / environment → tracking URI, MinIO creds
fetchers.py   source adapters             spec string → local directory of weights
registry.py   ModelRegistry               register / promote / get_current_version / resolve / download / list_versions
cli.py        argparse front door         same verbs, stdout = the answer
```

**Fetchers** turn "where the model comes from" into a local directory, which is all `register()`
needs. One spec syntax covers all three:

| Spec | Adapter | Use |
|---|---|---|
| `local:/path` or a bare path | `from_local` | weights already on disk |
| `hf:org/model[@revision]` | `from_huggingface` | download a Hub snapshot (supports `allow_patterns` / `ignore_patterns`) |
| `pretrained:pkg.module:callable` | `from_pretrained` | import and call a function returning a model or `(model, tokenizer)`; saved via `save_pretrained` — for post-finetune registration |

### Data flow of `register`

```
spec ──fetch_with_meta──► local dir (+ hf commit)
      ──MlflowClient────► experiment "<name>" → run "register-<name>" → log_artifacts(dir, "model")
      ──────────────────► create_model_version(source="runs:/<run>/model")
      ──────────────────► set_model_version_tag(source, hf_revision, …)
```

`resolve(name, alias)` returns that `runs:/<run>/model` URI; `download()` pulls it to a local dir
through the MLflow proxy.

---

## 2. Environments

| | Laptop (this repo) | Dev server (`ssh devserver`, RTX 3090) |
|---|---|---|
| Repo | `~/learn/mlflow-registry` | `~/mlflow-registry` (rsync copy) |
| Stack | `docker compose up -d` here | same compose, running |
| Tracking URI | `http://localhost:5000` | `http://localhost:5000` *on the server* |
| Models | 9 registered | 9 registered (identical) |
| GPU | none | yes → run inference / evaluation here |

Reach the dev server UI from the laptop with a tunnel:

```bash
ssh -N -L 5000:localhost:5000 -L 9001:localhost:9001 devserver   # then http://localhost:5000
```

Deploying to a fresh server = rsync the repo (without `.venv`), write `.env`, install `uv`,
`uv sync --group dev --extra hf`, `docker compose up -d`, `uv run pytest`, then
`uv run python scripts/register_shortlist.py`.

### Configuration (`.env`, see `.env.example`)

```
MLFLOW_TRACKING_URI=http://localhost:5000       # where the MLflow server is
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000    # MinIO (only needed for direct S3 access)
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY       # = MinIO user / password
MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=false      # see gotcha below
MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false
```

> **Gotcha: presigned URLs.** For files over 500 MB, MLflow 3.x hands clients presigned URLs that
> point at the *docker-internal* host `minio:9000`. From outside the compose network that name does
> not resolve, so big uploads fail and downloads hang. The two `MLFLOW_ENABLE_PROXY_MULTIPART_*`
> settings force everything through the MLflow proxy. `ModelRegistry` also sets them by default.
> A consumer running **inside** the same compose network (tracking URI `http://mlflow:5000`) can
> leave them on.

---

## 3. Registered models (as of 2026-09-26)

| Name | Task | Languages | License | Runtime |
|---|---|---|---|---|
| `qwen3-asr-1.7b` | STT | vi, en, multi | Apache-2.0 | transformers, vLLM |
| `granite-speech-4.1-2b` | STT | en | Apache-2.0 | transformers, vLLM |
| `gipformer1.5-68m-rnnt` | STT | vi | MIT (asserted, no LICENSE file) | onnxruntime / sherpa-onnx |
| `parakeet-ctc-0.6b-vietnamese` | STT | vi | NVIDIA Open Model License | NeMo |
| `whisper-large-v3` | STT | multi (baseline) | Apache-2.0 | transformers, faster-whisper |
| `phowhisper-large` | STT | vi | BSD-3-Clause | transformers (added 2026-09-29) |
| `voxcpm2` | TTS | vi, en, multi | Apache-2.0 | voxcpm |
| `vieneu-tts-v3-turbo` | TTS | vi, en | Apache-2.0 | vieneu |
| `kokoro-82m` | TTS | en, multi | Apache-2.0 | kokoro |
| `qwen3-tts-1.7b-base` | TTS | en, multi | Apache-2.0 | qwen-tts, vLLM-Omni |

Why these: `reports/Open source STT TTS Vietnamese English.md`. **No aliases are set yet** —
the evaluation phase decides what becomes `@production`.

```bash
uv run mlflow-registry list            # live inventory with tags and aliases
uv run mlflow-registry list --json     # machine-readable
```

---

## 4. How other repos use the registry

Two options. Pick **A** if the repo is Python and can take a dependency; pick **B** to stay on
plain `mlflow`.

### A. Depend on this package

```toml
# pyproject.toml of the consumer (evaluation, serving, …)
[project]
dependencies = ["mlflow-registry"]

[tool.uv.sources]
mlflow-registry = { path = "../mlflow-registry", editable = true }   # or a git URL once pushed
```

```python
from mlflow_registry import ModelRegistry

reg = ModelRegistry()                        # reads MLFLOW_TRACKING_URI from .env / environment
weights_dir = reg.download("qwen3-asr-1.7b", alias="production", dest_dir="/models/qwen3-asr")
```

`dest_dir` is worth fixing per model+version so repeated runs reuse the copy.

### B. Plain MLflow, no dependency on this repo

```python
import mlflow, os
os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")  # outside the compose net
mlflow.set_tracking_uri("http://localhost:5000")            # or http://mlflow:5000 inside compose

# by alias (what production code should do)
path = mlflow.artifacts.download_artifacts("models:/qwen3-asr-1.7b@production", dst_path="/models/qwen3-asr")

# by pinned version (what an evaluation run should do, for reproducibility)
path = mlflow.artifacts.download_artifacts("models:/qwen3-asr-1.7b/1", dst_path="/models/qwen3-asr-v1")

tags = mlflow.MlflowClient().get_model_version("qwen3-asr-1.7b", "1").tags   # runtime, license, hf_revision…
```

The result is the **same directory layout as the Hugging Face snapshot** (config, tokenizer,
safetensors / onnx / nemo…), so any loader that accepts a local path works unchanged.

### Running inference

The registry stores files, not a serving format. Load with the model's own runtime; the `runtime`
tag says which. Example, Whisper via transformers on the dev server:

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model=path, device="cuda")
print(asr("clip.wav")["text"])
```

Same shape for the others: `AutoModel…from_pretrained(path)` for Qwen3-ASR / Granite,
`sherpa_onnx` or `onnxruntime` pointed at `path/encoder.onnx` for Gipformer,
`nemo.collections.asr.ASRModel.restore_from(path/"parakeet-ctc-0.6b-vi.nemo")` for Parakeet,
and the `voxcpm`, `vieneu`, `kokoro`, `qwen-tts` packages for the TTS models. Those runtime
packages are **not** dependencies of this repo; the consumer installs what it needs.

### Evaluation repo: the intended loop

```bash
mlflow-registry list --json                              # discover candidates + versions
mlflow-registry download qwen3-asr-1.7b --alias staging --dest /models/qwen3-asr   # or by version in Python
# … run WER / MOS benchmarks, log results wherever the eval repo logs them …
mlflow-registry promote qwen3-asr-1.7b 1                 # winner → @production
mlflow-registry promote qwen3-asr-1.7b 1 --alias staging # or park a challenger on @staging
```

Record the `hf_revision` tag (or the registry version number) next to every metric so results
stay attributable to exact weights.

### Training / finetune repo: pushing a new version

```python
# in the training repo, e.g. train/export.py
def load_finetuned():
    model = AutoModelForSpeechSeq2Seq.from_pretrained("outputs/best")
    tok   = AutoProcessor.from_pretrained("outputs/best")
    return model, tok
```

```bash
mlflow-registry register whisper-large-v3 pretrained:train.export:load_finetuned \
    --tag license=Apache-2.0 --tag base=whisper-large-v3@v1 --tag dataset=bud500
# → prints the new version number; nothing moves to @production until promoted
```

Or, if the finetune is already saved to disk: `mlflow-registry register NAME local:outputs/best`.

### Serving repo (e.g. robo-be sidecars)

At container start: `download(name, alias="production", dest_dir=…)` → load → serve. A model swap is
`promote` on the registry followed by a restart; a rollback is another `promote`. Pin nothing but
the name and alias in the serving config.

---

## 5. Serving over REST (`mlflow models serve`)

The registry stores raw snapshots, and `mlflow models serve` needs a `python_function` flavor.
So every model also has a **version 2: a pyfunc wrapper** (a few KB) that points at the raw
weights version, downloads them at load time, and runs them with the model's own runtime.
The alias **`serving`** always points at the current wrapper; `production` / `staging` stay
reserved for raw weights, so nothing in section 4 changes.

```
whisper-large-v3
  v1  raw weights   (hf snapshot)            ← production / staging live here
  v2  pyfunc wrapper flavor=pyfunc wraps_version=1 task=stt serving_port=5005
  alias serving → 2

  container serve-whisper-large-v3:  mlflow models serve -m models:/whisper-large-v3@serving
        load_context → download models:/whisper-large-v3/1 into /models (shared volume)
        POST /invocations  {"dataframe_records":[{"audio_b64": "...", "language": "vi"}]}
```

### Layout

| Piece | Where |
|---|---|
| Wrappers (`SttModel` / `TtsModel` + one module per runtime) | `src/mlflow_registry/serving/` |
| Catalog: name → wrapper, host port, GPU | `src/mlflow_registry/serving/catalog.py` |
| Register the `@serving` versions | `scripts/register_serving.py` |
| Shared CUDA base image + thin per-model image | `serving/Dockerfile.base`, `serving/Dockerfile.model`, `serving/requirements/<name>.txt` |
| One compose service per model (generated) | `docker-compose.serving.yaml` ← `scripts/gen_compose.py` |
| Start / stop / logs | `scripts/serve.sh` |
| Client | `scripts/invoke.py` |

Same pattern as robo-be's sidecars: one base image with CUDA + torch, one thin image per model
because the runtimes pin incompatible `transformers` versions. Weights are **not** baked into
images; the first start downloads them from MLflow into the `serving_models` volume.

### Ports and REST contract

| Model | Task | Port | Request columns | Notes |
|---|---|---|---|---|
| `qwen3-asr-1.7b` | STT | 5001 | `audio_b64`, `language?` | vi / en / auto |
| `granite-speech-4.1-2b` | STT | 5002 | `audio_b64` | English |
| `gipformer1.5-68m-rnnt` | STT | 5003 | `audio_b64` | Vietnamese, CPU (sherpa-onnx) |
| `parakeet-ctc-0.6b-vietnamese` | STT | 5004 | `audio_b64` | Vietnamese, NeMo |
| `whisper-large-v3` | STT | 5005 | `audio_b64`, `language?` | multilingual |
| `voxcpm2` | TTS | 5006 | `text`, `voice?` (description), `ref_audio_b64?`, `ref_text?` | 48 kHz |
| `vieneu-tts-v3-turbo` | TTS | 5007 | `text`, `voice?` (preset, e.g. `Mai Anh`), `ref_audio_b64?` | 48 kHz |
| `kokoro-82m` | TTS | 5008 | `text`, `voice?` (`af_heart`, `bm_george`, …) | 24 kHz |
| `qwen3-tts-1.7b-base` | TTS | 5009 | `text`, `ref_audio_b64` **required**, `ref_text?`, `language?` | clone-only base model |
| `phowhisper-large` | STT | 5010 | `audio_b64`, `language?` | Vietnamese only; Whisper large fine-tuned by VinAI |

Audio goes in and out as **base64 WAV** (any libsndfile format in; 16-bit PCM WAV out).
STT returns `{"text", "language"}`, TTS returns `{"audio_b64", "sample_rate"}`:

```bash
# STT
curl -s localhost:5005/invocations -H 'Content-Type: application/json' \
  -d "{\"dataframe_records\":[{\"audio_b64\":\"$(base64 -w0 clip.wav)\",\"language\":\"vi\"}]}"
# → {"predictions":[{"text":"...","language":"vi"}]}

# TTS
curl -s localhost:5008/invocations -H 'Content-Type: application/json' \
  -d '{"dataframe_records":[{"text":"Hello from the registry","voice":"af_heart"}]}' \
  | python -c 'import sys,json,base64;p=json.load(sys.stdin)["predictions"][0];open("out.wav","wb").write(base64.b64decode(p["audio_b64"]))'

# or the bundled client (looks the port up in the catalog)
uv run python scripts/invoke.py stt whisper-large-v3 clip.wav --language vi
uv run python scripts/invoke.py tts kokoro-82m "Hello from the registry" --voice af_heart --out out.wav
```

`GET /health` answers once the model is loaded; `GET /version` gives the MLflow version.

### Bring a model up

```bash
uv run python scripts/register_serving.py            # once per registry: creates v2 + alias serving (all nine)
scripts/serve.sh build-base                          # once per machine: CUDA 12.8 + torch 2.8 + mlflow (~8 GB)
scripts/serve.sh up kokoro-82m                       # builds the thin image, starts it, prints the URL
scripts/serve.sh logs kokoro-82m                     # first start downloads the weights, then "Uvicorn running"
scripts/serve.sh health kokoro-82m
scripts/serve.sh down kokoro-82m
```

Under the hood: `docker compose -f docker-compose.yaml -f docker-compose.serving.yaml --profile <name> up -d serve-<name>`.
Every model is its own compose **profile**, so a plain `docker compose up -d` still starts only the
registry. Inside the compose network the tracking URI is `http://mlflow:5000`, where MinIO's presigned
URLs resolve, so the proxy-multipart workaround from section 2 is not needed there.

**VRAM.** The 3090 has 24 GB; the nine models resident together need roughly 28 GB, so do not start
them all. Rough per-model needs: qwen3-asr 5 GB, granite 6 GB, whisper 4 GB, parakeet 2 GB, voxcpm2 8 GB,
qwen3-tts 5 GB, vieneu 1 GB, kokoro <1 GB, gipformer 0 (CPU). Typical sets: *all STT* (~17 GB) or
*all TTS* (~15 GB).

**Runtime caveats.** `qwen3-tts-1.7b-base` is a clone-only checkpoint and refuses requests without
`ref_audio_b64`. `vieneu` downloads its MOSS audio tokenizer from Hugging Face on first start
(cached under `/models/.hf`). Set `GIPFORMER_QUANTIZE=int8` on the gipformer service for the smaller
int8 graphs, `VOXCPM_COMPILE=true` to enable torch.compile in VoxCPM.

**Changing a wrapper.** Wrapper classes are pickled *by reference*, so the code that runs is whatever
the image contains: rebuild the image (`serve.sh build NAME`) and restart. Re-run
`register_serving.py` only when the request/response signature or the wrapped weights version changes.
Edit the catalog, then `uv run python scripts/gen_compose.py`; a test fails if the compose file is stale.

---

## 5b. Evaluating STT against Google Chirp 3

`mlflow_registry.bench` runs the five STT models (through their serving containers) and
Google Speech-to-Text v2 `chirp_3` over a fixed dataset, computes **pooled** WER/CER, exact
match, English-word recall, latency/RTF, failures and `$/min`, writes a Markdown report and
logs one MLflow run per system and condition to the `eval-stt` experiment. Design:
`docs/superpowers/specs/2026-09-29-stt-eval-design.md`.

Dataset: `eval/stt-fixtures/` (robo-be's 14 fixture utterances, 37 words, adult/synthetic
speech). It proves the harness and catches gross failures; it cannot rank models. Swap in a
bigger set by pointing `--fixtures` at another folder with the same `manifest.jsonl` shape.

```bash
uv sync --group dev --extra hf --extra bench           # + jiwer, google-cloud-speech, opuslib (needs libopus0)

# laptop: Google leg (ADC via `gcloud auth application-default login`; project from gcloud config)
uv run python scripts/eval_stt.py collect --backend chirp --run eval/runs/2026-09-29-fixtures

# devserver: self-hosted leg (all five STT fit in VRAM together, ~17 GB)
scripts/serve.sh up qwen3-asr-1.7b granite-speech-4.1-2b gipformer1.5-68m-rnnt parakeet-ctc-0.6b-vietnamese whisper-large-v3
uv run python scripts/eval_stt.py collect --backend serving --run eval/runs/2026-09-29-fixtures
scripts/serve.sh down qwen3-asr-1.7b granite-speech-4.1-2b gipformer1.5-68m-rnnt parakeet-ctc-0.6b-vietnamese whisper-large-v3
rsync -av --exclude run.json devserver:~/mlflow-registry/eval/runs/2026-09-29-fixtures/ eval/runs/2026-09-29-fixtures/

# laptop: score + report + MLflow (tracking URI from .env)
uv run python scripts/eval_stt.py score --run eval/runs/2026-09-29-fixtures --mlflow
cp eval/runs/2026-09-29-fixtures/report.md reports/2026-09-29-stt-fixtures.md
```

Conditions: `hinted` (request carries the utterance language, as the LID router would) for
every system; `auto` (no hint) for Qwen3-ASR and Whisper (open-world detection) and Chirp 3
(detection restricted to `vi-VN`/`en-US`). One warm-up request, then
three passes; text from pass 1, latency = median. Chirp 3 is GA only in the `us` / `eu`
multi-regions (`--location`), so its latency includes that hop.

**Metrics shared with robo-be's benchmarks** (`robo-be/benchmarks/stt`, `tts`):

| robo-be | Here | Note |
|---|---|---|
| `bench_wer.py` aggregate WER / CER (Σ edits / Σ words) | In-domain WER, CER | same pooling; ours adds Unicode NFC |
| `bench_stt.py` `wer_mean` | Mean WER | mean of per-utterance WER |
| `bench_stt.py` code-switch pass rate | CS pass | every expected English word present; vacuously true without English |
| `bench_stt.py` category winner | Category winners | code-switch dominates, then mean WER; only systems supporting the language compete |
| `bench_tts.py` mean / median / p95 / min / max | Latency table | p95 by the same linear interpolation |
| `bench_wer.py` Opus 24 kbps VOIP, 20 ms frames | `eval/stt-fixtures-opus24/` | encoded and decoded offline with `opuslib`, as robo-be's clients and server do |

Not comparable: robo-be times end of speech to final transcript over its streaming WebSocket;
here latency is one HTTP request for the whole clip (inference plus transfer, no VAD or NATS).
Split-vs-full mode waits for the t2xx recordings; the code-switch pipeline metrics (language
tag accuracy, repairs) measure robo-be's router, not a model.

**Public dataset (`public-v1`).** FLEURS Vietnamese and English test splits (CC-BY-4.0, 857 + 647
read sentences, ~5 h) and the VIVOS test split (CC-BY-NC-SA-4.0, non-commercial, 760 Vietnamese
read utterances, ~45 min), in one manifest with categories `fleurs_vi`, `fleurs_en`, `vivos_vi`.
About 1 GB of audio, so it is built locally and gitignored; `source.json` records origin and
licence. Use one pass and parallel Google requests; keep self-hosted models sequential.

```bash
uv run python scripts/eval_stt.py datasets                                  # -> eval/datasets/public-v1
uv run python scripts/eval_stt.py collect --backend chirp --fixtures eval/datasets/public-v1 \
    --run eval/runs/<id>-public --passes 1 --concurrency 8
# devserver (rsync eval/datasets/public-v1 first):
uv run python scripts/eval_stt.py collect --backend serving --fixtures eval/datasets/public-v1 \
    --run eval/runs/<id>-public --passes 1
uv run python scripts/eval_stt.py score --run eval/runs/<id>-public --fixtures eval/datasets/public-v1
```

Large reports list only the 40 worst utterances per system; every hypothesis stays in the JSONL.

```bash
# Opus condition: build the degraded set once (committed), then collect/score it like any dataset
uv run python scripts/eval_stt.py opus                  # eval/stt-fixtures -> eval/stt-fixtures-opus24
uv run python scripts/eval_stt.py collect --backend chirp --fixtures eval/stt-fixtures-opus24 --run eval/runs/<id>-opus24
# ... serving leg with the same --fixtures on the devserver, then score as above ...
uv run python scripts/eval_stt.py compare --run clean=eval/runs/<id> --run opus24=eval/runs/<id>-opus24 --out reports/<id>-clean-vs-opus24.md
```

---

## 6. Development

```bash
uv sync --group dev --extra hf --extra bench   # deps (+ huggingface_hub for hf: specs, + bench)
docker compose up -d                 # local stack
uv run pytest                        # 158 tests; many hit the live stack and clean up after themselves
uv run mlflow-registry --help
```

Tests create throwaway `test-<hex>` models and delete them on teardown. `tests/_fake_train.py` is
the import target used to exercise `pretrained:` specs.

### Known operational notes

* **Dev-server DNS**: the office resolver (10.241.93.251) has been failing; the server cannot
  resolve any hostname until it is fixed (`sudo resolvectl dns wlp57s0 1.1.1.1 8.8.8.8 10.241.93.251`).
  Models already registered are unaffected. Workaround used: copy the Hugging Face cache blobs over
  SSH and register with `HF_HUB_OFFLINE=1`.
* **Hugging Face cache layout (hub ≥ 2.0)**: large files live in a shared store
  `~/.cache/huggingface/hub/blobs/<xx>/<sha256>`; per-model folders only hold symlinks. Copy the
  resolved blobs, not just the model folder.
* **MLflow search pagination**: 100 items per page; `list_versions()` drains all pages.
* **`.env` is secret** and git-ignored; `.env.example` is the template. The dev server has its own
  passwords.
