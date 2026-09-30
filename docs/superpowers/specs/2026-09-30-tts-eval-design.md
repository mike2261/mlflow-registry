# TTS evaluation: four self-hosted models against Google Chirp 3 HD

Date: 2026-09-30. Status: decisions delegated to the implementer in conversation ("choose and
decide by yourself"); the choices below are recorded with their reasons so they can be revisited.

## Goal

Answer, with numbers from one reproducible harness, how the four self-hosted TTS models in this
registry compare with Google Cloud Text-to-Speech on robo-be's domain: a Vietnamese tutor voice
talking to children, with English words mixed in. The metrics are automatic. They rank systems
and catch failures (skipped words, runaway audio, wrong language, a voice that does not match);
they do not replace a listening test, which is deferred to the two finalists.

The harness reuses the STT evaluation's shape (manifest, hash-pinned run directory, collect then
score, one MLflow run per system) and its code where it fits: normalizer, pooled WER/CER, English
word recall, latency statistics, the serving REST client, and the STT systems as judges.

## Systems under test

| System | Source | Languages | Voice |
|---|---|---|---|
| `voxcpm2` | serving container, port 5006 | vi, en | clone of the reference clip (`ref_audio_b64` + `ref_text`) |
| `vieneu-tts-v3-turbo` | serving container, port 5007 | vi (en words via its G2P) | clone of the reference clip (`ref_audio_b64`) |
| `qwen3-tts-1.7b-base` | serving container, port 5009 | en only (Vietnamese is not in its supported languages; the server rejects it) | clone of the reference clip (`ref_audio_b64` + `ref_text`) |
| `kokoro-82m` | serving container, port 5008 | en only | preset `af_heart` |
| `chirp3-hd` | Google Cloud TTS, called directly | vi, en | preset female voice `Aoede`: `vi-VN-Chirp3-HD-Aoede` / `en-US-Chirp3-HD-Aoede` |

Decisions:

- **Cloning, one reference voice for every cloning system.** Production wants one consistent
  tutor voice, and three of the four models are clone-first (qwen3-tts Base can only clone).
  Cloning all three from the same clip makes them comparable and lets speaker similarity be
  measured. robo-be never committed its `reference.wav`, so the reference is a 5-8 s clean
  female utterance from FLEURS vi test (CC-BY 4.0), chosen once and committed with its
  transcript under `eval/tts-reference/`.
- **Google Chirp 3 HD as the paid baseline**, the analogue of Chirp 3 in the STT run. The laptop
  already has ADC for Speech v2. It cannot clone, so it uses one preset female voice; its
  speaker-similarity cell is "n/a". Mixed sentences go to `vi-VN`.
- **kokoro-82m competes on English only.** It has no Vietnamese voice; running it on Vietnamese
  text would only measure a known gap.
- robo-be's OmniVoice is not in the registry and is out of scope.

## Dataset

`eval/tts-sentences/manifest.jsonl`, one line per sentence (no audio: the audio is the output):

```json
{"id": "vi-03", "text": "Hôm nay chúng ta sẽ học về các con vật sống trong rừng nhé.", "lang": "vi", "category": "vi_medium", "en_words": []}
```

About 60 sentences:


| Category | Count | Source |
|---|---|---|
| `vi_short`, `vi_medium`, `vi_long`, `mix`, `vi_expressive` | 15 (3 / 4 / 3 / 3 / 2) | robo-be's `benchmarks/tts/sentences.json`, all of it, ids `rb-00`..`rb-14`: the text production's TTS bench uses |
| `vi_short`, `vi_medium`, `vi_long` | ~15 | written for this set in the same tutor register, so each category has enough sentences |
| `en_short`, `en_medium` | ~12 | written for this set: tutor phrases in English |
| `mix` (code-switch) | ~12 more | Vietnamese sentences carrying English words a tutor teaches ("Con **apple** này màu đỏ"), `en_words` filled |
| `vi_numbers` | ~6 | dates, times, counts, prices, to see which models read digits |

`lang` of a `mix` sentence is `vi` (the request language for every system). The dataset hash is
SHA-256 over the manifest and the reference clip, stamped on every record as in the STT harness.

## Collect

`scripts/eval_tts.py collect --backend serving|google --run eval/runs/<id>-tts [--models ...]`

- One request per sentence per system per pass, `--passes 3` by default. Pass 1 is the judged
  draw (sampling models are judged on one sample); passes 2..3 are timed only and their audio is
  discarded, so the median latency is not a single sample. Concurrency 1.
- Output per system: `<system>.jsonl` (id, pass, latency_s, sample_rate, duration_s, error) and
  `audio/<system>/<id>.wav` (16-bit PCM, the model's native rate, pass 1 only).
- A request that errors, returns under 0.2 s of audio, or times out (60 s) is a failure; the
  sentence is scored with an empty transcript so failures cost WER instead of disappearing.

## Judge

`scripts/eval_tts.py judge --run ... --asr serving:qwen3-asr-1.7b --asr chirp_3` then
`--quality` on the devserver.

1. **Round-trip intelligibility.** Each pass-1 WAV is resampled to 16 kHz and transcribed by two
   ASR judges: `qwen3-asr-1.7b` (best self-hosted bilingual STT in the STT run) and `chirp_3`
   (best overall). Language hint = the sentence's `lang`. Transcripts are stored per judge in
   `asr/<judge>/<system>.jsonl` with the same record shape as the STT harness, so `score_text`
   and `en_recall_counts` apply unchanged. Two judges because every ASR has its own floor and
   biases (a model can sound like the judge's training data); the report shows both and ranks on
   their mean.
2. **Naturalness proxy: UTMOS** (`utmos22_strong` from SpeechMOS, CPU). Trained on English; for
   Vietnamese it is a relative signal between systems only, and the report says so.
3. **Speaker similarity** for the cloning systems: cosine similarity of WavLM x-vectors
   (`microsoft/wavlm-base-plus-sv`, via transformers, which the repo already understands; chosen
   over speechbrain's ECAPA to avoid its torchaudio version pinning) between each output and the
   reference clip. Preset-voice systems get "n/a".
4. **Duration sanity.** Seconds of audio per normalized character, per system and language. An
   utterance more than 2× or under 0.5× the median for that sentence is
   flagged (runaway generation, truncation, skipped clauses). The comparison is against the median
   of the *other* systems, so it works with two systems as well as five. Flags are counted and
   listed.

Quality judges run on the devserver next to the containers (they need torch); ASR judges reuse
the STT backends (containers on the devserver, Chirp from the laptop). Each judge writes its own
file, so they can run on different machines and be rsynced together, like the STT legs.

## Metrics and report

Per system, overall and per category:

| Metric | Definition |
|---|---|
| Round-trip WER / CER | pooled edits over reference words, per judge and mean of the two judges; STT normalizer |
| EN recall (`mix`) | fraction of expected English words recognised by the judge |
| UTMOS | mean predicted MOS (1-5) |
| Speaker sim | mean cosine to the reference (cloning systems) |
| Duration flags | count of out-of-range utterances |
| Failures | failed requests / sentences |
| Latency | mean, median, p95, min, max of request time (robo-be `bench_tts.py` statistics) |
| RTF | latency / audio duration, median |
| Cost | Google: list price per 1M characters converted to $/1k sentences; self-hosted "—" |

Headline table ranks by mean round-trip WER, then EN recall on `mix`. Category winners follow the
STT report's rule (code-switch dominates, then WER), only among systems supporting the language.
The report lists the worst 10 sentences per system with both judges' transcripts, and links the
WAVs by path so a listener can check them.

`score --mlflow` logs one run per system to the `eval-tts` experiment: metrics, params (voice,
reference hash, dataset hash, judge versions), the JSONL files and the report; audio stays in the
run directory (gitignored) to keep MLflow small.

## Out of scope

Human MOS/AB listening (next step, on the two finalists, using the committed WAV paths);
streaming time-to-first-audio (the REST contract returns whole clips; robo-be's WebSocket path
measures that); child voices; emotion or prosody control; the Opus condition (TTS audio goes to
the child's device over Opus, but degradation is the codec's, not the model's).

## Testing

Unit tests with fake transports and tiny synthetic WAVs, as in `tests/bench/`: manifest and hash,
collect record shape and failure rules, duration flags, the TTS→ASR judge wiring, scoring and the
report on a hand-made run directory. UTMOS and ECAPA are called through an injectable scorer so
tests never download models.
