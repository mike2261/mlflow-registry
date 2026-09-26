# Verification of six open-weight speech models (state as of 2026-09-26)

Scope: OmniVoice, Gipformer-65M/68M-RNNT, VoxCPM2, Fun-ASR-MLT-Nano-2512, IndexTTS-2 (+2.5, + Vietnamese fine-tune), Cohere Transcribe 03-2026. Facts were pulled directly from the Hugging Face model API/raw files, GitHub API/raw files, PyPI, and the projects' own docs/issues on 2026-09-26. "Download size" = sum of repo file sizes reported by the HF API (`?blobs=true`).

---

## 1. OmniVoice (k2-fsa)

### Takeaway
OmniVoice's HF repo `k2-fsa/OmniVoice` still exists (last updated 2026-07-03; code at 0.2.1, 2026-07-16) and does list Vietnamese with 8,481.98 training hours, but the "Apache-2.0" claim for the weights is **no longer true**: since 2026-07-03 the model card says the pre-trained weights are CC-BY-NC (code stays Apache-2.0), the maintainer confirmed on 2026-09-07 that "the CC-BY-NC license does not allow any commercial usage" and no commercial checkpoint exists, and the bundled audio tokenizer carries the Boson Higgs Audio 2 Community License. The model is non-autoregressive and has no true streaming.

### Cited Findings

**Repos / existence / versions**
- HF repo `k2-fsa/OmniVoice` exists; created 2026-03-30, last modified 2026-07-03, 1,357,647 downloads, 1,429 likes, pipeline `text-to-speech`, library `omnivoice`, base model `Qwen/Qwen3-0.6B` — [HF API](https://huggingface.co/api/models/k2-fsa/OmniVoice); [HF model card](https://huggingface.co/k2-fsa/OmniVoice)
- A second HF repo `k2-fsa/OmniVoice-Emilia` exists (created 2026-03-30); its card carries no license metadata — [HF org listing](https://huggingface.co/api/models?author=k2-fsa&search=OmniVoice); [OmniVoice-Emilia card](https://huggingface.co/k2-fsa/OmniVoice-Emilia)
- GitHub `k2-fsa/OmniVoice`: repo license detected as Apache-2.0, 13,879 stars, 66 open issues, last push 2026-09-21 — [GitHub API](https://api.github.com/repos/k2-fsa/OmniVoice); [LICENSE file (Apache 2.0)](https://github.com/k2-fsa/OmniVoice/blob/master/LICENSE)
- GitHub releases: 0.1.2 (2026-04-04), 0.1.3 (2026-04-07), 0.1.4 (2026-04-13), 0.1.5 (2026-04-28), 0.2.0 (2026-07-06), 0.2.1 (2026-07-16, latest) — [Releases](https://github.com/k2-fsa/OmniVoice/releases)
- PyPI `omnivoice` latest 0.2.1 uploaded 2026-07-16 — [PyPI](https://pypi.org/project/omnivoice/)
- 0.2.1 added `VoiceClonePrompt.save()/load()`, `asr_device`, opt-in `normalize_text=True`; 0.2.0 added Intel XPU support — [0.2.1 release](https://github.com/k2-fsa/OmniVoice/releases/tag/0.2.1); [0.2.0 release](https://github.com/k2-fsa/OmniVoice/releases/tag/0.2.0)
- No newer model checkpoint than the original 2026-03-30 upload was found; HF commits after 2026-04-13 are README-only ("Update README.md" 2026-04-22, 2026-05-07, 2026-07-03) — [HF commit history](https://huggingface.co/k2-fsa/OmniVoice/commits/main)

**License (code vs weights)**
- Current HF card text: "Our code is released under the Apache 2.0 License. The pre-trained model is licensed under the CC-BY-NC due to constraints from its training data (e.g., Emilia)." — [HF README](https://huggingface.co/k2-fsa/OmniVoice/blob/main/README.md)
- That sentence was introduced in HF commit `c5fdb5cc` on 2026-07-03; every earlier README revision (2026-04-01 through 2026-05-07) had `license: apache-2.0` in the YAML frontmatter. The current frontmatter has no `license:` key at all (HF API `cardData.license` = null) — [HF commits](https://huggingface.co/k2-fsa/OmniVoice/commits/main); [HF API](https://huggingface.co/api/models/k2-fsa/OmniVoice)
- Maintainer zhu-han, 2026-07-09 (issue #60): "we were recently informed that our pre-trained model is restricted by portions of its training data and thus may only be distributed under a non-commercial license CC-BY-NC. The Higgs Audio tokenizer imposes an additional constraint on licensing as well. Our code is released under Apache 2.0, so users may utilize our training scripts to train their own models" — [Issue #60 "Licensing"](https://github.com/k2-fsa/OmniVoice/issues/60)
- Maintainer zhu-han, 2026-09-07 (issue #258, open): "the CC-BY-NC license does not allow any commercial usage, and we currently do not have a commercially usable checkpoint. We have plans for this, but I cannot promise an exact release timeline." — [Issue #258](https://github.com/k2-fsa/OmniVoice/issues/258)
- `audio_tokenizer/LICENSE` in the HF repo is the "BOSON HIGGS AUDIO 2 COMMUNITY LICENSE AGREEMENT", which incorporates the Meta Llama 3 Community License — [audio_tokenizer/LICENSE](https://huggingface.co/k2-fsa/OmniVoice/blob/main/audio_tokenizer/LICENSE)
- There is no top-level LICENSE file in the HF repo (only `audio_tokenizer/LICENSE`) — [HF file list](https://huggingface.co/k2-fsa/OmniVoice/tree/main)
- A second open commercial-license request exists: issue #235 (2026-07-24) — [Issue #235](https://github.com/k2-fsa/OmniVoice/issues/235)

**Size / parameters**
- `model.safetensors` 2,450.3 MB with 612,577,288 parameters (F32); `audio_tokenizer/model.safetensors` 805.7 MB; `tokenizer.json` 11.4 MB; total repo ≈ 3.27 GB — [HF API with blobs](https://huggingface.co/api/models/k2-fsa/OmniVoice?blobs=true)

**Languages / Vietnamese**
- `docs/languages.md`: "OmniVoice supports 646 languages with a total of 581k hours of training data"; row 607: Vietnamese, ID `vi`, ISO `vie`, 8,481.98 h — [docs/languages.md](https://github.com/k2-fsa/OmniVoice/blob/master/docs/languages.md)
- `vi` and `en` are both in the HF card language list — [HF API](https://huggingface.co/api/models/k2-fsa/OmniVoice)
- Voice design (`instruct`) "is trained on Chinese and English data only. It can generalize to other languages, but may produce unstable results for some low-resource languages" — [GitHub README](https://github.com/k2-fsa/OmniVoice#voice-design)
- Cross-lingual cloning caveat: "the generated speech will carry an accent from the reference audio's language" — [GitHub README](https://github.com/k2-fsa/OmniVoice#voice-cloning)

**Features (instruct, non-verbal tags)**
- `instruct` attributes: gender, age, pitch, style (whisper), English accent, Chinese dialect — [GitHub README](https://github.com/k2-fsa/OmniVoice#voice-design)
- Supported non-verbal tags: `[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-en]`, `[question-ah]`, `[question-oh]`, `[question-ei]`, `[question-yi]`, `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[surprise-yo]`, `[dissatisfaction-hnn]`; pinyin and CMU-phoneme pronunciation control — [GitHub README](https://github.com/k2-fsa/OmniVoice#non-verbal--pronunciation-control)

**Inference stack**
- Official path: `pip install omnivoice` (PyTorch; CUDA, Apple MPS, Intel XPU); optional FlashInfer acceleration ~2–2.6x; CLIs `omnivoice-demo`, `omnivoice-infer`, `omnivoice-infer-batch`. Not sherpa-onnx, not transformers, not vLLM — [GitHub README](https://github.com/k2-fsa/OmniVoice#installation)
- Community ports listed by the project: omnivoice-rs, omnivoice-trtllm, OmniVoice-MLX, audio.cpp — [0.1.5 release](https://github.com/k2-fsa/OmniVoice/releases/tag/0.1.5); [0.2.1 release](https://github.com/k2-fsa/OmniVoice/releases/tag/0.2.1); community GGUF: [Serveurperso/OmniVoice-GGUF](https://huggingface.co/Serveurperso/OmniVoice-GGUF)

**Streaming**
- Maintainer zhu-han, 2026-04-03: "OmniVoice is a non-autoregressive TTS model... its architecture does not support true 'streaming' inference. But... you can achieve pseudo-streaming inference by splitting the text into smaller chunks" — [Issue #6](https://github.com/k2-fsa/OmniVoice/issues/6)
- Maintainer, 2026-04-11: synthesize sentence-by-sentence, `num_step=16`, use `cross_fade_chunks` to reduce boundary artifacts; "This is not true streaming" — [Issue #77](https://github.com/k2-fsa/OmniVoice/issues/77)

**Published quality numbers**
- Paper arXiv 2604.00688 (v1 2026-04-01, v3 2026-04-21) reports evaluations on "Chinese, English, and diverse multilingual benchmarks"; RTF "as low as 0.025" — [arXiv](https://arxiv.org/abs/2604.00688); [HF README](https://huggingface.co/k2-fsa/OmniVoice)
- Vietnamese fine-tuning question (#137): maintainer suggested `prompt_ratio_range=[0.0, 0.0]` for single-speaker adaptation and "I have not yet conducted relevant experiments" on minimum data — [Issue #137](https://github.com/k2-fsa/OmniVoice/issues/137)
- Open bug 2026-08-12 on cloning/pronunciation accuracy: [Issue #246](https://github.com/k2-fsa/OmniVoice/issues/246)

### Inferences
- The "~8.5k hours Vietnamese" claim in the original evaluation is accurate (8,481.98 h). The "Apache-2.0" claim was accurate only for the code and for the model card between 2026-04-01 and 2026-07-03; it is now wrong for the weights.
- For a commercial Vietnamese/English product, OmniVoice pretrained weights are not usable; only training your own model with the Apache-2.0 code is.

### Gaps
- No Vietnamese-specific WER/CER/MOS/SIM numbers were found in the paper abstract or README; the paper body was not parsed.
- Whether `k2-fsa/OmniVoice-Emilia` differs materially (data subset? license?) could not be confirmed; its card has no license field.
- `g-group-ai-lab/g-omnivoice` (created 2026-06-30, possibly a Vietnamese fine-tune) is gated; its card could not be read.

---

## 2. Gipformer-65M-RNNT → renamed gipformer-68M-RNNT (g-group-ai-lab)

### Takeaway
`g-group-ai-lab/gipformer-65M-rnnt` no longer exists under that name: it 307-redirects to `g-group-ai-lab/gipformer-68M-rnnt` after a 2026-09-17 rename to the exact parameter count (68,625,511). The MIT license claim is stated in the card/README but no LICENSE file exists in either the HF or GitHub repo. The 9/12 SOTA benchmark table is confirmed; a newer `gipformer1.5-68M-rnnt` (2026-08-21) exists; int8 ONNX is shipped; there is no streaming variant and the "streaming" tag was explicitly dropped on 2026-09-14.

### Cited Findings

**Repo / rename / versions**
- `https://huggingface.co/g-group-ai-lab/gipformer-65M-rnnt` returns HTTP 307 → `/g-group-ai-lab/gipformer-68M-rnnt`; HF API resolves the old id to `g-group-ai-lab/gipformer-68M-rnnt` — [old URL](https://huggingface.co/g-group-ai-lab/gipformer-65M-rnnt); [HF API](https://huggingface.co/api/models/g-group-ai-lab/gipformer-65M-rnnt)
- Commits 2026-09-17: "rename model to 68M (accurate parameter count)", "config: exact parameter count (68,625,511)", "docs: the zipformer baseline is 68M too (VietASR paper, Table 3)" — [HF commits](https://huggingface.co/g-group-ai-lab/gipformer-68M-rnnt/commits/main)
- Repo created 2026-03-12, last modified 2026-09-17, 628 downloads, 26 likes, library `onnxruntime`, language `vi` only — [HF API](https://huggingface.co/api/models/g-group-ai-lab/gipformer-68M-rnnt)
- Newer release: `g-group-ai-lab/gipformer1.5-68M-rnnt` created 2026-08-21 — [HF org listing](https://huggingface.co/api/models?author=g-group-ai-lab); [gipformer1.5 card](https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt)
- Other org models: `gwen-tts-0.6B` (2026-04-02), `g-omnivoice` (2026-06-30, gated) — [HF org listing](https://huggingface.co/api/models?author=g-group-ai-lab)
- No larger Gipformer (e.g., >68M) was found on HF as of 2026-09-26 — [HF org listing](https://huggingface.co/api/models?author=g-group-ai-lab)

**License**
- Card frontmatter `license: mit`; README: "This model is released under the [MIT License](LICENSE)" — but the `LICENSE` link target does not exist in the HF repo (file list: `.gitattributes, README.md, bpe.model, config.json, decoder.int8.onnx, decoder.onnx, encoder.int8.onnx, encoder.onnx, epoch-999.pt, joiner.int8.onnx, joiner.onnx, model.pt, tokens.txt`) — [HF README](https://huggingface.co/g-group-ai-lab/gipformer-68M-rnnt/blob/main/README.md); [HF API file list](https://huggingface.co/api/models/g-group-ai-lab/gipformer-68M-rnnt)
- GitHub `ggroup-ai-lab/gipformer`: README badge "License: MIT" and "This project is licensed under the MIT License", but the repo root contains only `.gitignore, README.md, data, infer_onnx.py, infer_pytorch.py, pyproject.toml` (no LICENSE file); GitHub API `license: null` — [GitHub README](https://github.com/ggroup-ai-lab/gipformer); [GitHub contents API](https://api.github.com/repos/ggroup-ai-lab/gipformer/contents/)

**Size / variants**
- `encoder.onnx` 261.1 MB, `encoder.int8.onnx` 70.9 MB, `decoder.onnx` 5.2 MB, `joiner.onnx`/`joiner.int8.onnx` (<5 MB), `model.pt` 279.0 MB, `epoch-999.pt` 279.0 MB; total repo ≈ 902 MB — [HF API with blobs](https://huggingface.co/api/models/g-group-ai-lab/gipformer-68M-rnnt?blobs=true)
- int8 variant confirmed: `--quantize int8` flag in `infer_onnx.py` — [GitHub README Quick Start](https://github.com/ggroup-ai-lab/gipformer#quick-start)

**Benchmarks (WER %, normalized: lowercase, punctuation removed, numbers spoken)**
- gipformer-68M-rnnt vs 10 baselines on 12 sets: tele-medium 15.53, tele-diff-north 25.10, tele-diff-middle 32.27, tele-diff-south 32.62, MultiMED 19.35, VietMed 19.41, vlsp-t1 13.39, vlsp-t2 20.40, LSVSC 8.96, Fleurs 12.92, ViMD 7.17, vivos 4.12; "#1 9/12", "#2 1/12 (LSVSC)", "#3 2/12 (vlsp-2020-task-1, Fleurs)" — [HF README](https://huggingface.co/g-group-ai-lab/gipformer-68M-rnnt/blob/main/README.md)
- Four of the 12 sets (tele-*) are private call-center sets — [HF README](https://huggingface.co/g-group-ai-lab/gipformer-68M-rnnt/blob/main/README.md)
- gipformer1.5-68M-rnnt on 16 sets: tele-medium 15.44, tele-hard-south 32.48, vi-asr-tech 27.49, vi-asr-edu 23.32, vi-asr-finance 22.34, vi-asr-pubadmin 14.82, VietMed 19.23, MultiMED 19.17, ViMD 7.00 (bold/best); vivos 4.25, Common Voice 6.45, vlsp-t1 13.37, LSVSC 8.97, Fleurs 12.65 — [gipformer1.5 card](https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt/blob/main/README.md)
- The four new public domain test sets are published under `g-group-ai-lab/vi-asr-{tech,edu,finance,pubadmin}-test` — [gipformer1.5 card](https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt/blob/main/README.md)

**Inference stack / streaming**
- Official inference is the project's own `infer_onnx.py` (onnxruntime; "CPU/GPU/mobile") and `infer_pytorch.py` (icefall stack, Linux+CUDA); `--version 1|1.5`. sherpa-onnx appears only in Acknowledgments — [GitHub README](https://github.com/ggroup-ai-lab/gipformer#quick-start)
- The ONNX files follow the encoder/decoder/joiner + `tokens.txt` layout used by sherpa-onnx zipformer transducers, but no sherpa-onnx recipe or config is provided in either repo — [HF file list](https://huggingface.co/g-group-ai-lab/gipformer-68M-rnnt/tree/main)
- HF commit 2026-09-14: "docs: clarify WER normalization wording, drop streaming tag"; no "streaming" mention remains in the README of either model — [HF commits](https://huggingface.co/g-group-ai-lab/gipformer-68M-rnnt/commits/main)
- Demo Space: [g-group-ai-lab/gipformer-demo](https://huggingface.co/spaces/g-group-ai-lab/gipformer-demo)

### Inferences
- Any pinned reference to `gipformer-65M-rnnt` still resolves today via redirect, but code should be updated to `gipformer-68M-rnnt` (or `gipformer1.5-68M-rnnt`).
- The model is Vietnamese-only (no English), so it cannot handle code-switched English segments by itself.
- Legal risk: MIT is asserted only in prose/metadata; there is no license text file to rely on.

### Gaps
- No sherpa-onnx compatibility test or official config found; whether the exported ONNX is a non-streaming (offline) zipformer export compatible with sherpa-onnx offline recognizer is not documented.
- No int8 vs fp32 WER comparison published.
- No English WER (model is Vietnamese-only).

---

## 3. VoxCPM2 (OpenBMB)

### Takeaway
`openbmb/VoxCPM2` exists (2.29B BF16 params, ≈4.96 GB), is Apache-2.0 for both weights and code, and explicitly lists Vietnamese among 30 languages. The latest release is 2.0.3 (2026-05-11); no VoxCPM 2.x beyond 2.0.3 and no VoxCPM 3 exists as of 2026-09-26. Streaming (`generate_streaming`), voice cloning (basic/controllable/ultimate) and ~8 GB VRAM are confirmed. Pinning 2.0.2 hits a known macOS MPS/bf16 noise bug and a vLLM-Omni cloning stop-token bug, both fixed later.

### Cited Findings

**Repo / versions**
- HF `openbmb/VoxCPM2`: created 2026-04-03, last modified 2026-08-18, 325,361 downloads, 1,643 likes, `license: apache-2.0`, library `voxcpm` — [HF API](https://huggingface.co/api/models/openbmb/VoxCPM2)
- GitHub `OpenBMB/VoxCPM`: Apache-2.0, 37,986 stars, 124 open issues, last push 2026-09-02 — [GitHub API](https://api.github.com/repos/OpenBMB/VoxCPM)
- Releases: 2.0.0 (2026-04-06), 2.0.1 (2026-04-08, "removed auto-trim feature for reference audio"), 2.0.2 (2026-04-08, "fixed some bugs"), 2.0.3 (2026-05-11, latest) — [Releases](https://github.com/OpenBMB/VoxCPM/releases)
- PyPI `voxcpm` latest 2.0.3 (2026-05-11); all versions: 1.0.1–1.0.5, 1.5.0, 2.0.0–2.0.3 — [PyPI](https://pypi.org/project/voxcpm/)
- HF org has only `VoxCPM2`, `VoxCPM1.5`, `VoxCPM-0.5B` — no 2.5/3.0 checkpoint — [HF org listing](https://huggingface.co/api/models?author=openbmb&search=VoxCPM)
- VoxCPM2 Technical Report arXiv 2606.06928 submitted 2026-06-05 — [arXiv](https://arxiv.org/abs/2606.06928)
- HF weight commit 2026-04-16 "feat: add custom tokenizer with multi-char Chinese token splitting (#8)" — a repo change after 2.0.2 shipped — [HF commits](https://huggingface.co/openbmb/VoxCPM2/commits/main)

**License**
- Card: "Released under the Apache-2.0 license, free for commercial use"; GitHub README: "Weights and code released under the Apache-2.0 license, free for commercial use" — [HF README](https://huggingface.co/openbmb/VoxCPM2/blob/main/README.md); [GitHub README](https://github.com/OpenBMB/VoxCPM)

**Size / params / VRAM**
- `model.safetensors` 4,580.1 MB, 2,290,004,544 params (BF16); `audiovae.pth` 377.0 MB; total ≈ 4.96 GB — [HF API with blobs](https://huggingface.co/api/models/openbmb/VoxCPM2?blobs=true)
- "VRAM ~8 GB"; RTF ~0.30 (RTX 4090 PyTorch), ~0.13 with Nano-vLLM; dtype bfloat16; 48 kHz output; max seq 8192 tokens — [HF README Model Details](https://huggingface.co/openbmb/VoxCPM2/blob/main/README.md)
- Requirements: Python ≥3.10 (<3.13 per 2.0.3 notes), PyTorch ≥2.5.0, CUDA ≥12.0 — [HF README](https://huggingface.co/openbmb/VoxCPM2/blob/main/README.md); [2.0.3 release](https://github.com/OpenBMB/VoxCPM/releases/tag/2.0.3)

**Languages**
- 30 languages listed explicitly including Vietnamese and English; frontmatter `language:` includes `vi` and `en`; "No language tag needed" — [HF README](https://huggingface.co/openbmb/VoxCPM2/blob/main/README.md)

**Streaming / cloning / serving**
- Streaming API `model.generate_streaming(...)`; 2.0.3 "Improve VoxCPM2 streaming VAE decode with a stateful StreamingVAEDecoder" — [HF README](https://huggingface.co/openbmb/VoxCPM2/blob/main/README.md); [2.0.3 release](https://github.com/OpenBMB/VoxCPM/releases/tag/2.0.3)
- Cloning modes: basic (`reference_wav_path`), controllable (style text in parentheses), "Ultimate" (`prompt_wav_path` + `prompt_text`); Voice Design from text description — [HF README](https://huggingface.co/openbmb/VoxCPM2/blob/main/README.md)
- Production serving: Nano-vLLM-VoxCPM (`pip install nano-vllm-voxcpm`) and vLLM-Omni (`vllm serve openbmb/VoxCPM2 --omni`, OpenAI-compatible `/v1/audio/speech`); llama.cpp-omni GGUF port — [GitHub README](https://github.com/OpenBMB/VoxCPM#-production-deployment-nano-vllm)
- Fine-tuning: full SFT and LoRA "with as little as 5–10 minutes of audio" — [HF README](https://huggingface.co/openbmb/VoxCPM2/blob/main/README.md)

**Published Vietnamese/English numbers**
- MiniMax-Multilingual-Test, Vietnamese WER: MiniMax 0.88, ElevenLabs 73.415, FishAudio S2 7.410, VoxCPM2 3.307; Vietnamese SIM: VoxCPM2 80.6 (best; MiniMax 74.3, FishAudio 74.0, ElevenLabs 36.9) — [GitHub README Performance](https://github.com/OpenBMB/VoxCPM#-performance)
- A 30-language per-language table in the GitHub README lists `vi (Vietnamese) WER 1.56% | 5.56%` and "Average (30 languages) 1.68%" for the first column — [GitHub README Performance](https://github.com/OpenBMB/VoxCPM#-performance)

**Issues relevant to a 2.0.2 pin**
- #301 (2026-05-10, closed 2026-05-19) "No voice, just noise output with version 2.0.2 on macOS": maintainer a710128: "`voxcpm==2.0.2` was released before the macOS float32-related change was merged (#263), so this may be related to MPS + bfloat16"; fixed by upgrading (2.0.3 promotes MPS dtypes to float32) — [Issue #301](https://github.com/OpenBMB/VoxCPM/issues/301); [2.0.3 release](https://github.com/OpenBMB/VoxCPM/releases/tag/2.0.3)
- vLLM-Omni #2896 (2026-04-18): "VoxCPM2 voice-cloning decoder never emits stop token, output always ~5 min" on VoxCPM 2.0.2 + vLLM 0.19.0; addressed via PR #2894 — [vllm-omni #2896](https://github.com/vllm-project/vllm-omni/issues/2896)
- #272 (open, 2026-04-19): chirp/click artifact at start of one-shot cloned audio — [Issue #272](https://github.com/OpenBMB/VoxCPM/issues/272)
- #269 (not planned, 2026-04-18): CUDA allocator/cudagraph race with ≥2 concurrent subprocesses — [Issue #269](https://github.com/OpenBMB/VoxCPM/issues/269)
- #299 (open, 2026-05-10): tokenizer compatibility issue — [Issue #299](https://github.com/OpenBMB/VoxCPM/issues/299)
- #357 (open, 2026-07-13): hallucination on very short single-word utterances (Polish) — [Issue #357](https://github.com/OpenBMB/VoxCPM/issues/357)
- Card limitation: "Voice Design and Style Control results may vary between runs; generating 1–3 times is recommended"; "Performance varies across languages depending on training data availability" — [HF README](https://huggingface.co/openbmb/VoxCPM2/blob/main/README.md)

### Inferences
- A 2.0.2 pin is safe on Linux/CUDA but not on Apple Silicon; the 2026-04-16 HF tokenizer change also means code and weights pinned at different dates may diverge (see #299). 2.0.3 is the sensible pin.
- Among the six models, VoxCPM2 is the only TTS here with an unambiguous permissive license on the weights **and** explicit Vietnamese support with published Vietnamese WER/SIM.

### Gaps
- The second column of the 30-language WER table (5.56% for vi) has no header captured in my extraction; which baseline it is could not be confirmed.
- No MOS numbers for Vietnamese found; no code-switching evaluation found.

---

## 4. Fun-ASR-MLT-Nano-2512 (FunAudioLLM / Tongyi)

### Takeaway
`FunAudioLLM/Fun-ASR-MLT-Nano-2512` exists (Apache-2.0, ~1.99 GB, 800M, 31 languages incl. Vietnamese and English), but the GitHub project moved to `QwenAudio/Fun-ASR` and the only newer 2026 artifacts are Nano (zh/en/ja) repackagings (transformers-native, GGUF, vLLM) — no new MLT checkpoint. The `batch_size>1 NotImplementedError` comes from `inference_prepare()` in the FunASR `fun_asr_nano` model code; recent FunASR adds a `_inference_llm_batch` path and vLLM batching. The card explicitly says **no** checkpoint-specific WER exists for the MLT model.

### Cited Findings

**Repo / versions**
- HF repo exists: created 2025-12-15, last modified 2026-08-05, 483 downloads, 67 likes, `license: apache-2.0`, library `funasr`, tags include `streaming`, `vllm`, `31-languages` — [HF API](https://huggingface.co/api/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512)
- Model weights unchanged since 2025-12-23 (`Update config.yaml`); all 2026 commits are docs: "Add model card YAML metadata" (2026-05-25), "Fix MLT model card quickstarts and benchmark scope" (2026-07-13), "Update FunASR install floor to 1.4.1" (2026-08-05) — [HF commits](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512/commits/main)
- GitHub: `https://github.com/FunAudioLLM/Fun-ASR` now resolves to `QwenAudio/Fun-ASR` (Apache-2.0, 1,553 stars, 7 open issues, pushed 2026-09-10) — [GitHub API](https://api.github.com/repos/FunAudioLLM/Fun-ASR); [QwenAudio/Fun-ASR](https://github.com/QwenAudio/Fun-ASR)
- GitHub release v1.0.0 (2026-05-25) "Fun-ASR-Nano and Fun-ASR-MLT-Nano" lists both checkpoints at 800M; later releases are llama.cpp runtime builds (latest runtime-llamacpp-v0.2.1 2026-08-27, v0.1.10 2026-09-05) — [Releases](https://github.com/QwenAudio/Fun-ASR/releases)
- Newer HF repos in 2026 are Nano-only: `Fun-ASR-Nano-2512-hf` (2026-05-24), `Fun-ASR-Nano-GGUF` (2026-06-20), `Fun-ASR-Nano-2512-GGUF` (2026-07-29), `Fun-ASR-Nano-2512-vllm` (2026-08-29). No new MLT repo — [HF org listing](https://huggingface.co/api/models?author=FunAudioLLM&search=Fun-ASR)
- GitHub README: "the 31-language MLT checkpoint is separate" from the transformers-native Nano path — [GitHub README](https://github.com/QwenAudio/Fun-ASR)

**License**
- Card `license: apache-2.0`; GitHub: "Source code in this repository is licensed under the Apache License 2.0... The official Fun-ASR-Nano and Fun-ASR-MLT-Nano cards currently list Apache-2.0; review the card for the specific artifact you download." — [GitHub README License](https://github.com/QwenAudio/Fun-ASR#license); [HF README](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512/blob/main/README.md)

**Size / architecture**
- `model.pt` 1,971.1 MB; `Qwen3-0.6B/` tokenizer files (11.4 MB tokenizer.json); total ≈ 1.99 GB; card: "800M-parameter multilingual checkpoint"; Qwen3-0.6B folder present as the LLM decoder tokenizer — [HF API with blobs](https://huggingface.co/api/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512?blobs=true); [HF README](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512/blob/main/README.md)

**Languages**
- 31 languages incl. Chinese, English, Cantonese, Japanese, Korean, Vietnamese, Indonesian, Thai, Malay, Filipino, Arabic, Hindi + 19 European; frontmatter includes `vi`, `en`; "language control: accepts the language names shown in the inference example" (Chinese names, e.g., `越南语`) — [HF README](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512/blob/main/README.md)

**Inference stack / streaming / batching**
- Official path: `funasr>=1.4.1`, `AutoModel(model=..., hub="hf", trust_remote_code=True, remote_code="./model.py")`, `batch_size=1` in all examples; optional `fsmn-vad` — [HF README](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512/blob/main/README.md)
- `model.py` is **not** in the HF repo file list (files: config.yaml, configuration.json, model.pt, Qwen3-0.6B/*, multilingual.tiktoken, examples) — it must come from the GitHub clone or the FunASR package — [HF API file list](https://huggingface.co/api/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512)
- Source of the limitation: `funasr/models/fun_asr_nano/model.py` `inference_prepare()`: `if len(data_in) > 1: raise NotImplementedError("batch decoding is not implemented")` — [FunASR model.py](https://github.com/modelscope/FunASR/blob/main/funasr/models/fun_asr_nano/model.py)
- Current FunASR `inference_llm()` routes `len(data_in) > 1` to `_inference_llm_batch(...)` when no CTC decoder is loaded ("CTC timestamps are not produced in batched mode"), i.e., batching now exists on the LLM path — [FunASR model.py](https://github.com/modelscope/FunASR/blob/main/funasr/models/fun_asr_nano/model.py)
- `AutoModel` forces `kwargs["batch_size"] = 1` whenever the device falls back to CPU — [FunASR auto_model.py](https://github.com/modelscope/FunASR/blob/main/funasr/auto/auto_model.py)
- GitHub README documents `batch_size_s=120` to batch VAD segments through the LLM decoder, plus vLLM `AutoModelVLLM` (offline batch, "340x" RTF claim) and `FunASRNanoStreamingVLLM` streaming SDK (`chunk_ms=720`); vLLM path requires `funasr>=1.3.26`, `vllm>=0.12.0` — [GitHub README](https://github.com/QwenAudio/Fun-ASR#vllm-high-throughput-inference-)
- Card TODO: "Support returning timestamps", "Support speaker diarization", "Support model training" — [HF README](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512/blob/main/README.md)
- Known Nano checkpoint issue (not MLT-specific): HF `Fun-ASR-Nano-2512` artifact lacks CTC tensors so timestamps are omitted; use `hub="ms"` — [Issue #70](https://github.com/QwenAudio/Fun-ASR/issues/70); [GitHub README](https://github.com/QwenAudio/Fun-ASR)

**Published WER**
- Card disclaimer: "The tables below are Fun-ASR family results reproduced from the project report. They do not contain a column identified as `Fun-ASR-MLT-Nano-2512`, so they must not be interpreted as checkpoint-specific results for this MLT model. MLT per-language results will be added when a reproducible evaluation is published." — [HF README](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512/blob/main/README.md)
- Fun-ASR Technical Report (arXiv 2509.12508) Table 7 for the larger 5-language "Fun-ASR-ML" model (not the Nano checkpoint): Vietnamese WER — commonvoice 7.41, gigaspeech2-test 8.98, in-house 7.06 (Whisper-large-v3: 13.51 / 13.82 / 11.46); English — librispeech-test-clean 1.62, test-other 3.39, fleurs 3.18, commonvoice 7.67 — [arXiv 2509.12508 §6.2.6](https://arxiv.org/html/2509.12508v3)
- Fun-ASR-nano (zh/en/ja) family numbers on the card: Librispeech clean 1.76 / other 4.33, Fleurs-en 5.96 — [HF README](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512/blob/main/README.md)

### Inferences
- The batch limitation as originally hit (April 2026) is real and is in `inference_prepare`; with FunASR ≥1.4.x the LLM path batches when no CTC decoder is loaded, and vLLM is the recommended high-throughput route. Whether the vLLM/streaming SDK examples (all shown with `Fun-ASR-Nano-2512`) work unchanged with the MLT checkpoint is not documented.
- No Vietnamese WER exists for this exact checkpoint; the 7–9% Vietnamese WERs belong to the larger Fun-ASR-ML model.

### Gaps
- No checkpoint-specific Vietnamese or English WER for Fun-ASR-MLT-Nano-2512 anywhere official.
- Streaming SDK / vLLM support for the MLT checkpoint specifically: not confirmed.
- Parameter split "800M encoder + Qwen3-0.6B decoder" as stated in the original evaluation could not be verified; official docs say 800M total with a Qwen3-0.6B tokenizer folder.

---

## 5. IndexTTS-2 (bilibili), IndexTTS-2.5, and `dinhthuan/index-tts-2-vietnamese`

### Takeaway
IndexTTS-2 weights are under the "bilibili Model Use License Agreement" (HF `LICENSE.txt`, GitHub `LICENSE` since 2025-09-09), whose §2.2 requires a separate license only if the licensee/affiliates exceeded 100M MAU in the prior month or RMB 1 billion revenue in the prior year; it also bans using the model to improve other commercial AI models and requires PRC-law arbitration. IndexTTS-2.5 (2026-08-10) added Japanese, Spanish and Arabic — **not Vietnamese**. The community fine-tune `dinhthuan/index-tts-2-vietnamese` exists (Nov 2025) but its `apache-2.0` tag conflicts with the upstream license and its own "commercial use requires permission" note.

### Cited Findings

**IndexTTS-2 repo / license**
- HF `IndexTeam/IndexTTS-2`: created 2025-06-18, last modified 2026-01-20, languages `en, zh`, no `license` field in card metadata; repo contains `LICENSE.txt` and `LICENSE_ZH.txt` — [HF API](https://huggingface.co/api/models/IndexTeam/IndexTTS-2); [HF file tree](https://huggingface.co/IndexTeam/IndexTTS-2/tree/main)
- `LICENSE.txt` = "bilibili Model Use License Agreement"; §1.4 defines the Model as "bilibili indextts2 ... model weights and final code ... published by us at https://github.com/index-tts/index-tts" — [LICENSE.txt](https://huggingface.co/IndexTeam/IndexTTS-2/blob/main/LICENSE.txt)
- §2.2: "If ... either (i) your or any of your Affiliates' products or services had more than 100 million monthly active users in the immediately preceding calendar month, or (ii) your or any of your Affiliates' annual revenue in the immediately preceding calendar year exceeded RMB 1 billion, You must request a separated license from us, which We may grant to You in our sole discretion." — [LICENSE.txt](https://huggingface.co/IndexTeam/IndexTTS-2/blob/main/LICENSE.txt)
- §1.5 defines Derivative Work to include fine-tunes/LoRA/merged checkpoints; §3.4(c): "You may not Use the bilibili indextts2 or any Derivative Work to improve any AI model, except for the bilibili indextts2 itself, its Derivative Works, or non-commercial AI models"; §4.1(a) requires a specific disclaimer on distributed derivatives; §4.2 prohibits high-risk uses; §6 PRC law, Shanghai Arbitration Commission; §9 Chinese text prevails — [LICENSE.txt](https://huggingface.co/IndexTeam/IndexTTS-2/blob/main/LICENSE.txt)
- GitHub `index-tts/index-tts` `LICENSE` is the identical bilibili agreement; LICENSE history: "init infer code" 2025-03-25, "Update license for IndexTTS-2" and "Update License (#300)" 2025-09-09; a former `INDEX_MODEL_LICENSE` file was removed on 2025-09-09 (now 404); GitHub API license = NOASSERTION — [GitHub LICENSE](https://github.com/index-tts/index-tts/blob/main/LICENSE); [LICENSE commits](https://github.com/index-tts/index-tts/commits/main/LICENSE); [GitHub API](https://api.github.com/repos/index-tts/index-tts)
- README: "This project is released under the bilibili Model Use License Agreement" and "For commercial usage and cooperation, please contact indexspeech@bilibili.com" — [GitHub README](https://github.com/index-tts/index-tts#-license)
- Repo stats: 24,188 stars, 415 open issues, last push 2026-08-18 — [GitHub API](https://api.github.com/repos/index-tts/index-tts)

**IndexTTS-2 size**
- `gpt.pth` 3,484.7 MB, `s2mel.pth` 1,202.2 MB, `qwen0.6bemo4-merge/model.safetensors` 1,192.1 MB; total ≈ 5.90 GB (plus auxiliary w2v-bert-2.0 / MaskGCT / CAMPPlus / BigVGAN downloaded at first run) — [HF API with blobs](https://huggingface.co/api/models/IndexTeam/IndexTTS-2?blobs=true); [IndexTTS-2.5 card](https://huggingface.co/IndexTeam/IndexTTS-2.5/blob/main/README.md)

**IndexTTS-2.5**
- HF `IndexTeam/IndexTTS-2.5` created 2026-08-10, last modified 2026-08-12; `license: other`, `license_name: bilibili-model-license`; languages `zh, en, ja, es, ar`; "~0.8B (GPT backbone)"; "roughly 6 GB of VRAM"; 22.05 kHz output; paper arXiv 2601.03888 — [HF API](https://huggingface.co/api/models/IndexTeam/IndexTTS-2.5); [HF README](https://huggingface.co/IndexTeam/IndexTTS-2.5/blob/main/README.md)
- "Compared with IndexTTS-2, it adds Japanese, Spanish and Arabic" — Vietnamese is not listed — [HF README](https://huggingface.co/IndexTeam/IndexTTS-2.5/blob/main/README.md)
- Files: gpt.pth 3,259.6 MB, s2mel.pth 414.9 MB, codec.pth 607.3 MB, qwen0.6bemo4-merge 1,192.1 MB; total ≈ 5.49 GB — [HF API with blobs](https://huggingface.co/api/models/IndexTeam/IndexTTS-2.5?blobs=true)
- GitHub releases v2.5.0 and v2.0.0 both tagged 2026-08-13; vLLM production recipe at recipes.vllm.ai — [Releases](https://github.com/index-tts/index-tts/releases); [GitHub README News](https://github.com/index-tts/index-tts#-news)
- IndexTTS-2.5 LICENSE on HF is byte-identical (except trailing newline) to the GitHub LICENSE — [HF LICENSE](https://huggingface.co/IndexTeam/IndexTTS-2.5/blob/main/LICENSE)
- No PyPI package `indextts` exists; install is `git clone` + `uv sync --all-extras` — [PyPI 404](https://pypi.org/pypi/indextts/json); [HF README](https://huggingface.co/IndexTeam/IndexTTS-2.5/blob/main/README.md)

**`dinhthuan/index-tts-2-vietnamese`**
- Exists: created 2025-11-17, last modified 2025-11-18, 63 downloads, 20 likes, `license: apache-2.0`, language `vi`, `base_model: IndexTeam/IndexTTS-2` — [HF API](https://huggingface.co/api/models/dinhthuan/index-tts-2-vietnamese)
- Card: "Languages: Vietnamese (primary), retains English/Chinese from base checkpoint"; limitation column: "Commercial use requires permission from IndexTTS authors"; "For commercial inquiries: indexspeech@bilibili.com" — [HF README](https://huggingface.co/dinhthuan/index-tts-2-vietnamese/blob/main/README.md)
- Self-reported evaluation: "Subjective MOS on internal Vietnamese validation set (~4.3 MOS @ 22 kHz)"; "Word Error Rate similar to base IndexTTS2 for English/Chinese prompts" (no numbers, no methodology) — [HF README](https://huggingface.co/dinhthuan/index-tts-2-vietnamese/blob/main/README.md)
- Files: gpt.pth 1,742.4 MB, s2mel.pth 1,202.2 MB, qwen0.6bemo4-merge 1,192.1 MB; total ≈ 4.15 GB; inference via `indextts.infer_v2.IndexTTS2` with `use_fp16=True`; training: AdamW LR 1e-5, base `gpt_old.pth` — [HF API with blobs](https://huggingface.co/api/models/dinhthuan/index-tts-2-vietnamese?blobs=true); [HF README](https://huggingface.co/dinhthuan/index-tts-2-vietnamese/blob/main/README.md)

### Inferences
- The "free commercial under 100M MAU / RMB 1B" summary is correct as a threshold reading of §2.2, but the license carries additional material restrictions (no use to improve other commercial AI models, mandatory derivative disclaimer, high-risk-use ban, PRC law/arbitration, Chinese text prevails) that a plain "free commercial" label understates.
- The Vietnamese fine-tune's `apache-2.0` tag is not a valid relicensing: under §1.5 it is a Derivative Work bound by the bilibili agreement, and its own card acknowledges the need for permission.
- IndexTTS-2.5 offers no Vietnamese path; a Vietnamese deployment would still rely on the 2025 community fine-tune of IndexTTS-2.

### Gaps
- No independently measured Vietnamese MOS/WER for the fine-tune; the ~4.3 MOS is unverifiable.
- Whether the fine-tune's `gpt.pth` (1.74 GB vs 3.48 GB upstream) is fp16 or a pruned model is not stated.
- IndexTTS-2 streaming support was not assessed (out of scope of this verification; not mentioned on either HF card).

---

## 6. CohereLabs/cohere-transcribe-03-2026

### Takeaway
The repo exists, is gated with HF "auto" approval (accept conditions + share contact info), and is licensed Apache 2.0 per the card ("This model is governed by an Apache 2.0 license"), so self-hosted commercial use is permitted by the stated license; the gate is an access step, not a different license. It lists Vietnamese and English among 14 languages, is 2.07B params (~4.13 GB), runs natively in transformers ≥5.4.0 and vLLM, and reports 5.42 avg English WER on the Open ASR Leaderboard; per-language (incl. Vietnamese) WER is only shown as a figure. A newer sibling `cohere-transcribe-arabic-07-2026` exists.

### Cited Findings
- Repo `CohereLabs/cohere-transcribe-03-2026`: created 2026-03-24, last modified 2026-06-10, `gated: "auto"`, `license: apache-2.0`, library `transformers`, tags `custom_code`, `hf-asr-leaderboard`, DOI 10.57967/hf/8653; 212,043 downloads, 1,149 likes — [HF API](https://huggingface.co/api/models/CohereLabs/cohere-transcribe-03-2026)
- Gate banner: "You need to agree to share your contact information to access this model. This repository is publicly accessible, but you have to accept the conditions to access its files and content." (raw file access returns "Access ... is restricted") — [HF model page](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- Card "Terms of Use": "We hope that the release of this model will make community-based research efforts more accessible, by releasing the weights of a highly performant 2 billion parameter model to researchers all over the world. This model is governed by an Apache 2.0 license." — [HF model page](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- Release blog: "open-sourced on huggingface under an Apache 2.0 licence"; release date 2026-03-26 — [HF blog](https://huggingface.co/blog/CohereLabs/cohere-transcribe-03-2026-release); [Cohere blog](https://cohere.com/blog/transcribe)
- Languages: "Trained on 14 languages: European: English, French, German, Italian, Spanish, Portuguese, Greek, Dutch, Polish; APAC: Chinese (Mandarin), Japanese, Korean, Vietnamese; MENA: Arabic"; frontmatter includes `vi`, `en` — [HF model page](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026); [HF API](https://huggingface.co/api/models/CohereLabs/cohere-transcribe-03-2026)
- Size: `model.safetensors` 4,131.9 MB, 2,065,804,048 params (BF16) — [HF API with blobs](https://huggingface.co/api/models/CohereLabs/cohere-transcribe-03-2026?blobs=true)
- Architecture: "conformer-based encoder-decoder", audio → log-Mel, auto-resampled to 16 kHz — [HF model page](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- Inference: "supported natively in transformers" (`pip install transformers>=5.4.0`, `CohereAsrForConditionalGeneration`), plus a vLLM integration example (`/v1/audio/transcriptions`); community ONNX port `onnx-community/cohere-transcribe-03-2026-ONNX` — [HF model page](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026); [ONNX port](https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX)
- English: Open ASR Leaderboard (as of 2026-03-26) average WER 5.42 (AMI 8.15, Earnings22 10.84, Gigaspeech 9.33, LS clean 1.25, LS other 2.37, SPGISpeech 3.08, Tedlium 2.49, Voxpopuli 5.87), RTFx 524.88 — [HF model page](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- Per-language WER "averaged over FLEURS, Common Voice 17.0, MLS and Wenet" is shown only as a figure ("CER for zh, ja, ko — WER otherwise") — [HF model page](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- Newer sibling: `CohereLabs/cohere-transcribe-arabic-07-2026` (created 2026-06-18) — [HF org listing](https://huggingface.co/api/models?author=CohereLabs&search=transcribe)
- Third-party OpenASR runtime page: 2B, Apache-2.0, "3.2× real-time", peak memory 3.2 GB on CPU/Apple Silicon — [openasr.org](https://openasr.org/models/cohere-transcribe-03-2026/)
- Cohere docs: hosted API free tier with rate limits; "Model Vault" for production — [Cohere docs](https://docs.cohere.com/docs/transcribe)

### Inferences
- "Gated" here means HF contact-info acceptance with automatic approval; nothing found indicates a non-commercial or additional-terms license, so self-hosting commercially is allowed under Apache 2.0 as stated. Redistribution of the weights is also allowed by Apache 2.0, though the gate means end users cannot `hf download` anonymously.

### Gaps
- The exact text of the gate conditions requires login and could not be read; whether it adds any usage clause beyond Apache 2.0 is unconfirmed.
- No numeric Vietnamese WER is published on the card (figure only); the "technical blog post" with WERs was not retrieved.
- Streaming: no streaming inference mode is documented on the card or docs; not confirmed either way.
