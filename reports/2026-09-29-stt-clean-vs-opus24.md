# STT evaluation: clean vs opus24

- `clean`: dataset `stt-fixtures` `sha256:92e4316a0274ecb1188e6c6713166e518944b27a0b12ace5a6e8bf96c37608f6`
- `opus24`: dataset `stt-fixtures-opus24` `sha256:8c654cfc65da185fc6ad570a1e07d290ee922d40b62d9567e24d491d12976c9b`

Δ WER is each dataset's pooled in-domain WER minus `clean`'s.

## hinted

| System | clean WER | clean Mean WER | clean CS pass | clean median latency | opus24 WER | opus24 Mean WER | opus24 CS pass | opus24 median latency | Δ WER (opus24) |
|---|---|---|---|---|---|---|---|---|---|
| chirp_3 | 2.7% (1/37) | 0.9% | 100.0% (14/14) | 1.67s | 2.7% (1/37) | 0.9% | 100.0% (14/14) | 1.71s | +0.0 pp |
| gipformer1.5-68m-rnnt | 3.6% (1/28) | 1.1% | 78.6% (11/14) | 0.03s | 3.6% (1/28) | 1.1% | 78.6% (11/14) | 0.03s | +0.0 pp |
| granite-speech-4.1-2b | 0.0% (0/9) | 0.0% | 100.0% (14/14) | 0.13s | 0.0% (0/9) | 0.0% | 100.0% (14/14) | 0.14s | +0.0 pp |
| parakeet-ctc-0.6b-vietnamese | 14.3% (4/28) | 8.0% | 78.6% (11/14) | 0.06s | 14.3% (4/28) | 8.0% | 78.6% (11/14) | 0.06s | +0.0 pp |
| qwen3-asr-1.7b | 2.7% (1/37) | 0.9% | 100.0% (14/14) | 0.14s | 2.7% (1/37) | 0.9% | 100.0% (14/14) | 0.14s | +0.0 pp |
| whisper-large-v3 | 5.4% (2/37) | 1.8% | 100.0% (14/14) | 0.24s | 605.4% (224/37) | 794.6% | 100.0% (14/14) | 0.28s | +600.0 pp |

## auto

| System | clean WER | clean Mean WER | clean CS pass | clean median latency | opus24 WER | opus24 Mean WER | opus24 CS pass | opus24 median latency | Δ WER (opus24) |
|---|---|---|---|---|---|---|---|---|---|
| chirp_3 | 2.7% (1/37) | 0.9% | 100.0% (14/14) | 1.04s | 2.7% (1/37) | 0.9% | 100.0% (14/14) | 1.68s | +0.0 pp |
| qwen3-asr-1.7b | 8.1% (3/37) | 8.0% | 100.0% (14/14) | 0.22s | 8.1% (3/37) | 8.0% | 100.0% (14/14) | 0.22s | +0.0 pp |
| whisper-large-v3 | 21.6% (8/37) | 23.2% | 100.0% (14/14) | 0.31s | 21.6% (8/37) | 23.2% | 100.0% (14/14) | 0.31s | +0.0 pp |
