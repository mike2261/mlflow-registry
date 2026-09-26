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
research report in `reports/`.

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

## 5. Development

```bash
uv sync --group dev --extra hf       # deps (+ huggingface_hub for hf: specs)
docker compose up -d                 # local stack
uv run pytest                        # 45 tests; most hit the live stack and clean up after themselves
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
