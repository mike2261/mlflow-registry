# STT evaluation: five self-hosted models against Google Chirp 3

Date: 2026-09-29. Status: approved in conversation (scope: STT only, robo-be's 14 fixture
utterances, pooled WER as the headline, Chirp 3 as the only Google baseline).

## Goal

Answer, with numbers produced by one reproducible harness, how the five self-hosted STT models
in this registry compare with the Google Speech-to-Text v2 `chirp_3` model that production
calls today. This first round uses robo-be's 14 fixture utterances. Its job is to prove the
harness end to end on all six systems, log the results to MLflow, and catch gross failures.
It does **not** select a model. The dataset is a manifest, so the larger sets that will
justify a decision (t2xx adult recordings, child speech, public corpora) are a manifest
change, not a harness change.

This is the "Tier 1" evaluation the v1 ARCHITECTURE.md describes and records as never run.

## Systems under test

| System | Source | Languages | Auto-detect |
|---|---|---|---|
| `qwen3-asr-1.7b` | serving container, port 5001 | vi, en | yes |
| `granite-speech-4.1-2b` | serving container, port 5002 | en | no |
| `gipformer1.5-68m-rnnt` | serving container, port 5003 | vi | no |
| `parakeet-ctc-0.6b-vietnamese` | serving container, port 5004 | vi | no |
| `whisper-large-v3` | serving container, port 5005 | vi, en | yes |
| `chirp_3` | Google Speech-to-Text v2, called directly | vi, en | restricted to vi/en (`language_codes=["vi-VN", "en-US"]`) |

The five models are called over the REST contract from the serving spec
(`POST /invocations`, `audio_b64` + optional `language`). That gives each model its own
dependency set for free and measures the code that will actually be deployed.

Chirp 3 is called from the harness with Application Default Credentials, not through
robo-bridge. Chirp 3 is GA in the `us` and `eu` multi-regions only; the harness defaults to
`us` and takes the location as a flag. Vietnamese is `vi-VN`, English is `en-US`. Synchronous
`Recognize` is used, which covers audio under one minute. No other Google model is run.

## Dataset

`eval/stt-fixtures/`: the 14 WAVs from `robo-be/benchmarks/stt/fixtures/` (16 kHz mono
PCM16, 776 KB in total, about 24 s of audio) plus `manifest.jsonl`, one line per utterance:

```json
{"id": "00", "file": "00.vi_short.wav", "text": "anh em", "lang": "vi", "category": "vi_short", "en_words": [], "duration_s": 1.41}
```

Categories: 10 × `vi_short` (two-word phrases), 1 × `vi_medium` (eight words), 2 × `en_short`,
1 × `en_medium` (six words). 37 reference words. No code-switched audio despite the
category names in robo-be; `en_words` is kept so the metric is ready when mixed audio arrives.

The dataset hash is the SHA-256 over the manifest text and every WAV's bytes, and is stamped
on every result so a fixture edit can never be confused with a model change.

## Metrics

All text metrics are computed after one normalizer, applied identically to reference and
hypothesis, and stamped with `normalizer_version`:

1. Unicode NFC (typed references and API output often differ in composition form; without
   this a visually identical Vietnamese word scores several character errors).
2. Lowercase.
3. Replace every character that is not a word character, whitespace or a Vietnamese letter
   with a space (robo-be's pattern).
4. Collapse whitespace, strip.

Edit distances come from `jiwer`.

| Metric | Definition | Why |
|---|---|---|
| **Pooled WER** (headline) | Σ word edits / Σ reference words over the utterances in the slice | every word counts once; short phrases are not inflated; comparable to model cards |
| Pooled CER | same at character level, spaces removed | a wrong tone mark is one character, not a whole word |
| Exact match rate | share of utterances whose normalized hypothesis equals the reference | the unit that matters for two-word commands |
| English word recall | share of `en_words` present in the hypothesis tokens, pooled | robo-be's code-switch preservation metric, kept for continuity |
| Latency | wall time of one request, median over passes per utterance; system-level median and p90 | includes the network hop, which is the cost the robot pays |
| RTF | Σ latency / Σ audio duration | throughput-independent speed |
| Failure rate | requests that raised, timed out or returned empty text | kept out of WER so a crash does not hide as 100 % |
| Cost per minute | `chirp_3`: USD 0.016 per minute, list price checked 2026-09-29 at cloud.google.com/speech-to-text/pricing; self-hosted: not priced in this round | the architecture doc asks for a `$/min` column |

Every percentage in the report carries its raw counts (`3/30`), and the report opens with the
caveat that 37 words cannot rank models.

**Slices.** Each metric is reported for `all`, per `lang`, per `category`, and for
`in_domain`: the utterances whose language the system claims to support (see the table
above). `in_domain` is the headline column; out-of-language rows are reported, not dropped,
because they show what happens when routing gets it wrong.

## Protocol

**Conditions.**

* `hinted`: the request carries the utterance's language (`vi` / `en`), as the LID router
  would in production. Every system runs this condition; single-language systems receive
  the hint too and ignore it.
* `auto`: no language hint. Only systems with auto-detect run it: `qwen3-asr-1.7b`,
  `whisper-large-v3`, `chirp_3`. Chirp 3 gets restricted detection over
  `["vi-VN", "en-US"]` (decided 2026-09-29 after open-world `["auto"]` read one-second
  Vietnamese clips as Korean and Chinese). Qwen3-ASR and Whisper keep open-world detection,
  because the serving contract cannot restrict their language set; the report reads their
  `auto` rows with that asymmetry in mind.

**Passes.** One warm-up request per system (first utterance, discarded), then three full
passes over the manifest. Text metrics use pass 1. Latency uses the median of the three.
If a system's hypotheses differ between passes for any utterance, the report lists it under
"non-deterministic outputs" with the variants.

**Collect and score are separate.** `collect` calls one backend and appends raw records to
a run directory. `score` reads a run directory and produces metrics, the report and the
MLflow runs. This lets the self-hosted leg run on the devserver (where the models are) and
the Google leg on the laptop (where the credentials are), with neither an SSH tunnel nor
copied credentials contaminating the other. Re-scoring after a normalizer fix calls no API.

Run directory `eval/runs/<run-id>/`:

```
run.json                 run_id, dataset, dataset_hash, created, harness git sha
<system>.jsonl           one record per request
report.md                written by score
```

Record:

```json
{"system": "whisper-large-v3", "condition": "hinted", "pass": 1, "utt": "00",
 "lang_hint": "vi", "text": "anh em", "language": "vi", "latency_s": 0.412,
 "error": null, "ts": "2026-09-29T10:12:03Z",
 "backend": {"kind": "serving", "host": "localhost", "port": 5005, "mlflow_version": "3.16.0"}}
```

For `chirp_3` the `backend` object holds `kind`, `project`, `location`, `model`.

**Run on the devserver.** All five STT models together need about 17 GB of the 24 GB, so one
`serve.sh up` of the five, one `collect --backend serving`, one `serve.sh down`.

**MLflow.** Experiment `eval-stt` on the same tracking server as the registry. One run per
(system, condition). Params: `system`, `condition`, `dataset`, `dataset_hash`,
`normalizer_version`, `passes`, and the backend fields above. Metrics: every metric in the
table for the `in_domain` and `all` slices plus per-language WER (`wer_vi`, `wer_en`).
Artifacts: the system's JSONL and `report.md`. Tag `run_id`. This is what makes "best on
slice X" a query later.

## Success criteria for this run (fixed before it executes)

1. Every system returns a transcript for every utterance in the `hinted` condition.
2. The `chirp_3` row is present in the report.
3. `report.md` exists and the MLflow runs exist with the metrics above.
4. Any system with in-domain pooled WER above 50 % is investigated as a harness bug before it
   is believed.
5. No model is selected or rejected on this dataset.

## Code layout

```
src/mlflow_registry/bench/
  __init__.py
  manifest.py    Utterance dataclass; load(dir) -> list[Utterance]; dataset_hash(dir) -> str
  normalize.py   NORMALIZER_VERSION; normalize(text) -> str
  metrics.py     pooled WER/CER, exact match, en recall, latency stats; slice helpers
  backends.py    Backend protocol: name, languages, auto_detect, transcribe(wav_bytes, lang|None)
                 -> Hypothesis(text, language, latency_s)
                 ServingBackend(name, host, port, post=urllib)   post injectable for tests
                 ChirpBackend(project, location, model, client=None)  client injectable
  collect.py     run(backend, manifest, conditions, passes, out_dir) -> appends JSONL
  score.py       load run dir -> ScoreTable; render_report(...) -> str; log_to_mlflow(...)
scripts/eval_stt.py    CLI:
  collect --backend serving [--models a,b] [--host] --run eval/runs/<id>
  collect --backend chirp [--project] [--location us] --run eval/runs/<id>
  score --run eval/runs/<id> [--mlflow]
eval/stt-fixtures/     manifest.jsonl + 14 WAVs (committed)
eval/runs/             results, gitignored; the report of a run worth keeping is copied to reports/
```

Dependencies go in a new optional extra `bench = ["jiwer", "google-cloud-speech"]` so the
serving images and the base install do not pick them up. `pyyaml` stays in dev.

System language capabilities live in `backends.py` as a small table keyed by registered
model name, next to the code that needs them; the serving catalog is not changed.

## Workflow for one run

```bash
# laptop: Google leg
uv sync --extra bench
uv run python scripts/eval_stt.py collect --backend chirp --run eval/runs/2026-09-29-fixtures

# devserver: self-hosted leg
scripts/serve.sh up qwen3-asr-1.7b granite-speech-4.1-2b gipformer1.5-68m-rnnt parakeet-ctc-0.6b-vietnamese whisper-large-v3
uv run python scripts/eval_stt.py collect --backend serving --run eval/runs/2026-09-29-fixtures
scripts/serve.sh down <same five>
# copy eval/runs/2026-09-29-fixtures/*.jsonl back to the laptop (rsync)

# laptop: score, report, MLflow (tracking URI via .env, SSH tunnel to the devserver stack)
uv run python scripts/eval_stt.py score --run eval/runs/2026-09-29-fixtures --mlflow
cp eval/runs/2026-09-29-fixtures/report.md reports/2026-09-29-stt-fixtures.md
```

## Report shape

1. Header: run id, date, dataset and hash, normalizer version, passes, and the 30-word caveat.
2. Summary table per condition: system · in-domain WER (edits/words) · CER · exact match ·
   EN recall · median latency · RTF · failures · $/min. Chirp 3 row always present.
3. Per-language and per-category WER tables.
4. Per-system detail: one row per utterance with reference, hypothesis, WER, latency.
5. Non-deterministic outputs, if any.

## Error handling

* A backend exception or timeout becomes a record with `error` set and `text` null; collect
  continues with the next request. Score counts it as a failure and excludes it from WER.
* An empty hypothesis is a failure, not a 100 % WER.
* `collect` refuses to run if the run directory's `dataset_hash` differs from the manifest on
  disk, so two legs of one run cannot use different fixtures.
* `score` refuses a run directory with no `chirp_3.jsonl` unless `--allow-missing-baseline`
  is passed, so the Google row cannot silently drop out.

## Testing

* Unit, no network: `normalize` (NFC, punctuation, whitespace, Vietnamese letters kept);
  `metrics` against hand-computed values, including the pooled-vs-averaged example (two
  errors in six words plus 13 perfect utterances = 6.7 % pooled); manifest loading and hash
  stability; `ServingBackend` with an injected fake `post`; `ChirpBackend` with a fake
  client; `score` on a canned run directory produces the expected numbers and report
  sections; determinism flagging; failure exclusion.
* Live, optional: `ServingBackend` against a running `gipformer1.5-68m-rnnt` container,
  skipped when the port is closed.
* No test calls Google.

## Out of scope

TTS evaluation, larger or child datasets, `chirp_2` or other Google models, streaming
latency, the robo-bridge path, pricing self-hosted GPU time, statistical intervals (no
meaning on 37 words), and any change to the serving containers or the registry contract.
