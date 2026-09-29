# STT evaluation

- Run: `2026-09-29-fixtures` created 2026-09-29T07:12:34Z (harness 09e59fd)
- Dataset: `stt-fixtures` `sha256:92e4316a0274ecb1188e6c6713166e518944b27a0b12ace5a6e8bf96c37608f6`: 14 utterances, 37 words, 23.8s of audio
- Normalizer: `nfc-lower-vi-1`; text metrics from pass 1, latency = median over passes
- Chirp 3 price: $0.016/min, list price checked 2026-09-29 at https://cloud.google.com/speech-to-text/pricing

> This dataset has 37 reference words. It **cannot rank models**; it proves the harness and catches gross failures. Every percentage carries its raw counts.

## Summary: hinted

In-domain = utterances in a language the system claims to support. In-domain WER is pooled (as robo-be bench_wer.py); Mean WER is the mean of per-utterance WER (as bench_stt.py). CS pass = every expected English word present, vacuously true for utterances without English (as bench_stt.py). EN recall, CS pass, latency, RTF and failures are over all utterances.

0 of 14 utterances contain a number (a digit in the reference or in any system's output). Numbers have several correct written forms ("1537" vs "một nghìn năm trăm ba mươi bảy") and WER charges a correctly heard number as several errors when the forms differ, so "WER, no numbers" leaves those utterances out for every system.

| System | In-domain WER | WER, no numbers | Mean WER | CER | Exact | EN recall | CS pass | Median latency | p95 | RTF | Failed utts (pass 1) | $/min | Failed requests (all passes) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chirp_3 | 2.7% (1/37) | 2.7% (1/37) | 0.9% | 1.5% (2/134) | 92.9% (13/14) | 100.0% (9/9) | 100.0% (14/14) | 1.67s | 2.03s | 0.92 | 0/14 | $0.016 | 0/42 |
| gipformer1.5-68m-rnnt | 3.6% (1/28) | 3.6% (1/28) | 1.1% | 2.2% (2/89) | 90.9% (10/11) | 22.2% (2/9) | 78.6% (11/14) | 0.03s | 0.07s | 0.02 | 0/14 | — | 0/42 |
| granite-speech-4.1-2b | 0.0% (0/9) | 0.0% (0/9) | 0.0% | 0.0% (0/45) | 100.0% (3/3) | 100.0% (9/9) | 100.0% (14/14) | 0.13s | 0.27s | 0.09 | 0/14 | — | 0/42 |
| parakeet-ctc-0.6b-vietnamese | 14.3% (4/28) | 14.3% (4/28) | 8.0% | 6.7% (6/89) | 81.8% (9/11) | 0.0% (0/9) | 78.6% (11/14) | 0.06s | 0.06s | 0.04 | 0/14 | — | 0/42 |
| qwen3-asr-1.7b | 2.7% (1/37) | 2.7% (1/37) | 0.9% | 1.5% (2/134) | 92.9% (13/14) | 100.0% (9/9) | 100.0% (14/14) | 0.14s | 0.25s | 0.09 | 0/14 | — | 0/42 |
| whisper-large-v3 | 5.4% (2/37) | 5.4% (2/37) | 1.8% | 2.2% (3/134) | 92.9% (13/14) | 100.0% (9/9) | 100.0% (14/14) | 0.24s | 0.41s | 0.16 | 0/14 | — | 0/42 |

### Latency: hinted

Per utterance, median over passes; statistics as in robo-be's bench_tts.py.

| System | mean | median | p95 | min | max |
|---|---|---|---|---|---|
| chirp_3 | 1.57s | 1.67s | 2.03s | 0.87s | 2.40s |
| gipformer1.5-68m-rnnt | 0.04s | 0.03s | 0.07s | 0.02s | 0.11s |
| granite-speech-4.1-2b | 0.16s | 0.13s | 0.27s | 0.10s | 0.30s |
| parakeet-ctc-0.6b-vietnamese | 0.06s | 0.06s | 0.06s | 0.06s | 0.07s |
| qwen3-asr-1.7b | 0.16s | 0.14s | 0.25s | 0.09s | 0.30s |
| whisper-large-v3 | 0.26s | 0.24s | 0.41s | 0.16s | 0.55s |

### Category winners: hinted

robo-be rule: among systems that support the category's language, code-switch pass dominates, then lower mean per-utterance WER; equal values tie.

| Category | Winner |
|---|---|
| en_medium | chirp_3 = granite-speech-4.1-2b = qwen3-asr-1.7b = whisper-large-v3 (tie) |
| en_short | chirp_3 = granite-speech-4.1-2b = qwen3-asr-1.7b = whisper-large-v3 (tie) |
| vi_medium | chirp_3 = gipformer1.5-68m-rnnt = qwen3-asr-1.7b (tie) |
| vi_short | chirp_3 = gipformer1.5-68m-rnnt = qwen3-asr-1.7b = whisper-large-v3 (tie) |

### WER by language

| System | en | vi |
|---|---|---|
| chirp_3 | 0.0% (0/9) | 3.6% (1/28) |
| gipformer1.5-68m-rnnt | 88.9% (8/9) | 3.6% (1/28) |
| granite-speech-4.1-2b | 0.0% (0/9) | 96.4% (27/28) |
| parakeet-ctc-0.6b-vietnamese | 111.1% (10/9) | 14.3% (4/28) |
| qwen3-asr-1.7b | 0.0% (0/9) | 3.6% (1/28) |
| whisper-large-v3 | 0.0% (0/9) | 7.1% (2/28) |

### WER by category

| System | en_medium | en_short | vi_medium | vi_short |
|---|---|---|---|---|
| chirp_3 | 0.0% (0/6) | 0.0% (0/3) | 12.5% (1/8) | 0.0% (0/20) |
| gipformer1.5-68m-rnnt | 66.7% (4/6) | 133.3% (4/3) | 12.5% (1/8) | 0.0% (0/20) |
| granite-speech-4.1-2b | 0.0% (0/6) | 0.0% (0/3) | 87.5% (7/8) | 100.0% (20/20) |
| parakeet-ctc-0.6b-vietnamese | 100.0% (6/6) | 133.3% (4/3) | 37.5% (3/8) | 5.0% (1/20) |
| qwen3-asr-1.7b | 0.0% (0/6) | 0.0% (0/3) | 12.5% (1/8) | 0.0% (0/20) |
| whisper-large-v3 | 0.0% (0/6) | 0.0% (0/3) | 25.0% (2/8) | 0.0% (0/20) |

## Summary: auto

In-domain = utterances in a language the system claims to support. In-domain WER is pooled (as robo-be bench_wer.py); Mean WER is the mean of per-utterance WER (as bench_stt.py). CS pass = every expected English word present, vacuously true for utterances without English (as bench_stt.py). EN recall, CS pass, latency, RTF and failures are over all utterances.

0 of 14 utterances contain a number (a digit in the reference or in any system's output). Numbers have several correct written forms ("1537" vs "một nghìn năm trăm ba mươi bảy") and WER charges a correctly heard number as several errors when the forms differ, so "WER, no numbers" leaves those utterances out for every system.

| System | In-domain WER | WER, no numbers | Mean WER | CER | Exact | EN recall | CS pass | Median latency | p95 | RTF | Failed utts (pass 1) | $/min | Failed requests (all passes) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chirp_3 | 2.7% (1/37) | 2.7% (1/37) | 0.9% | 1.5% (2/134) | 92.9% (13/14) | 100.0% (9/9) | 100.0% (14/14) | 1.04s | 1.80s | 0.71 | 0/14 | $0.016 | 0/42 |
| qwen3-asr-1.7b | 8.1% (3/37) | 8.1% (3/37) | 8.0% | 3.0% (4/134) | 85.7% (12/14) | 100.0% (9/9) | 100.0% (14/14) | 0.22s | 0.32s | 0.14 | 0/14 | — | 0/42 |
| whisper-large-v3 | 21.6% (8/37) | 21.6% (8/37) | 23.2% | 11.2% (15/134) | 71.4% (10/14) | 100.0% (9/9) | 100.0% (14/14) | 0.31s | 0.48s | 0.20 | 0/14 | — | 0/42 |

### Latency: auto

Per utterance, median over passes; statistics as in robo-be's bench_tts.py.

| System | mean | median | p95 | min | max |
|---|---|---|---|---|---|
| chirp_3 | 1.20s | 1.04s | 1.80s | 0.87s | 1.88s |
| qwen3-asr-1.7b | 0.23s | 0.22s | 0.32s | 0.19s | 0.37s |
| whisper-large-v3 | 0.34s | 0.31s | 0.48s | 0.25s | 0.62s |

### Category winners: auto

robo-be rule: among systems that support the category's language, code-switch pass dominates, then lower mean per-utterance WER; equal values tie.

| Category | Winner |
|---|---|
| en_medium | chirp_3 = qwen3-asr-1.7b = whisper-large-v3 (tie) |
| en_short | chirp_3 = qwen3-asr-1.7b = whisper-large-v3 (tie) |
| vi_medium | chirp_3 = qwen3-asr-1.7b (tie) |
| vi_short | chirp_3 |

### WER by language

| System | en | vi |
|---|---|---|
| chirp_3 | 0.0% (0/9) | 3.6% (1/28) |
| qwen3-asr-1.7b | 0.0% (0/9) | 10.7% (3/28) |
| whisper-large-v3 | 0.0% (0/9) | 28.6% (8/28) |

### WER by category

| System | en_medium | en_short | vi_medium | vi_short |
|---|---|---|---|---|
| chirp_3 | 0.0% (0/6) | 0.0% (0/3) | 12.5% (1/8) | 0.0% (0/20) |
| qwen3-asr-1.7b | 0.0% (0/6) | 0.0% (0/3) | 12.5% (1/8) | 10.0% (2/20) |
| whisper-large-v3 | 0.0% (0/6) | 0.0% (0/3) | 25.0% (2/8) | 30.0% (6/20) |

## Detail: chirp_3 (auto)

| Utt | Ref | Hyp | Lang | WER | Latency |
|---|---|---|---|---|---|
| 00 | anh em | anh em | vi | 0.0% (0/2) | 1.25s |
| 01 | buổi sáng | buổi sáng | vi | 0.0% (0/2) | 1.54s |
| 02 | bánh mì | bánh mì | vi | 0.0% (0/2) | 0.96s |
| 03 | bác sĩ | bác sĩ | vi | 0.0% (0/2) | 1.07s |
| 04 | cảm ơn | Cảm ơn. | vi | 0.0% (0/2) | 1.08s |
| 05 | cuộc sống | cuộc sống | vi | 0.0% (0/2) | 1.02s |
| 06 | cái gì | Cái gì? | vi | 0.0% (0/2) | 0.89s |
| 07 | cơ hội | cơ hội | vi | 0.0% (0/2) | 0.90s |
| 08 | chiến đấu | chiến đấu | vi | 0.0% (0/2) | 1.00s |
| 09 | anh dũng | Anh Dũng | vi | 0.0% (0/2) | 1.76s |
| 11 | cộng hoà xã hội chủ nghĩa việt nam | Cộng hòa xã hội chủ nghĩa Việt Nam | vi | 12.5% (1/8) | 1.72s |
| 13 | good morning | Good morning. | en | 0.0% (0/2) | 0.88s |
| 15 | elephant | elephant | en | 0.0% (0/1) | 0.87s |
| 17 | a rolling stone gathers no moss | A rolling stone gathers no moss. | en | 0.0% (0/6) | 1.88s |

## Detail: chirp_3 (hinted)

| Utt | Ref | Hyp | Lang | WER | Latency |
|---|---|---|---|---|---|
| 00 | anh em | anh em | vi | 0.0% (0/2) | 1.63s |
| 01 | buổi sáng | buổi sáng | vi | 0.0% (0/2) | 1.67s |
| 02 | bánh mì | bánh mì | vi | 0.0% (0/2) | 1.64s |
| 03 | bác sĩ | bác sĩ | vi | 0.0% (0/2) | 1.70s |
| 04 | cảm ơn | Cảm ơn. | vi | 0.0% (0/2) | 1.78s |
| 05 | cuộc sống | cuộc sống | vi | 0.0% (0/2) | 1.66s |
| 06 | cái gì | Cái gì? | vi | 0.0% (0/2) | 0.87s |
| 07 | cơ hội | cơ hội | vi | 0.0% (0/2) | 1.72s |
| 08 | chiến đấu | chiến đấu | vi | 0.0% (0/2) | 1.19s |
| 09 | anh dũng | anh dũng | vi | 0.0% (0/2) | 1.09s |
| 11 | cộng hoà xã hội chủ nghĩa việt nam | Cộng hòa xã hội chủ nghĩa Việt Nam | vi | 12.5% (1/8) | 2.40s |
| 13 | good morning | Good morning. | en | 0.0% (0/2) | 1.05s |
| 15 | elephant | elephant | en | 0.0% (0/1) | 1.70s |
| 17 | a rolling stone gathers no moss | A rolling stone gathers no moss. | en | 0.0% (0/6) | 1.84s |

## Detail: gipformer1.5-68m-rnnt (hinted)

| Utt | Ref | Hyp | Lang | WER | Latency |
|---|---|---|---|---|---|
| 00 | anh em | ANH EM | vi | 0.0% (0/2) | 0.03s |
| 01 | buổi sáng | BUỔI SÁNG | vi | 0.0% (0/2) | 0.03s |
| 02 | bánh mì | BÁNH MÌ | vi | 0.0% (0/2) | 0.04s |
| 03 | bác sĩ | BÁC SĨ | vi | 0.0% (0/2) | 0.03s |
| 04 | cảm ơn | CẢM ƠN | vi | 0.0% (0/2) | 0.03s |
| 05 | cuộc sống | CUỘC SỐNG | vi | 0.0% (0/2) | 0.03s |
| 06 | cái gì | CÁI GÌ | vi | 0.0% (0/2) | 0.03s |
| 07 | cơ hội | CƠ HỘI | vi | 0.0% (0/2) | 0.03s |
| 08 | chiến đấu | CHIẾN ĐẤU | vi | 0.0% (0/2) | 0.03s |
| 09 | anh dũng | ANH DŨNG | vi | 0.0% (0/2) | 0.04s |
| 11 | cộng hoà xã hội chủ nghĩa việt nam | CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM | vi | 12.5% (1/8) | 0.11s |
| 13 | good morning | KẾT NỐI NÀY | vi | 150.0% (3/2) | 0.02s |
| 15 | elephant | ELOFENT | vi | 100.0% (1/1) | 0.03s |
| 17 | a rolling stone gathers no moss | A WALLNG GATORS NO | vi | 66.7% (4/6) | 0.05s |

## Detail: granite-speech-4.1-2b (hinted)

| Utt | Ref | Hyp | Lang | WER | Latency |
|---|---|---|---|---|---|
| 00 | anh em | and um | vi | 100.0% (2/2) | 0.11s |
| 01 | buổi sáng | Bui sang. | vi | 100.0% (2/2) | 0.15s |
| 02 | bánh mì | Ban Me | vi | 100.0% (2/2) | 0.10s |
| 03 | bác sĩ | Back seat. | vi | 100.0% (2/2) | 0.13s |
| 04 | cảm ơn | Gam-um | vi | 100.0% (2/2) | 0.13s |
| 05 | cuộc sống | Oxum. | vi | 100.0% (2/2) | 0.15s |
| 06 | cái gì | Gaiye | vi | 100.0% (2/2) | 0.13s |
| 07 | cơ hội | Go ahead. | vi | 100.0% (2/2) | 0.13s |
| 08 | chiến đấu | 吉莉莉 | vi | 100.0% (2/2) | 0.25s |
| 09 | anh dũng | And young? | vi | 100.0% (2/2) | 0.13s |
| 11 | cộng hoà xã hội chủ nghĩa việt nam | Gua sa hoa chu mi Viet Nam. | vi | 87.5% (7/8) | 0.30s |
| 13 | good morning | Good morning. | en | 0.0% (0/2) | 0.13s |
| 15 | elephant | Elephant. | en | 0.0% (0/1) | 0.13s |
| 17 | a rolling stone gathers no moss | A rolling stone gathers no moss. | en | 0.0% (0/6) | 0.23s |

## Detail: parakeet-ctc-0.6b-vietnamese (hinted)

| Utt | Ref | Hyp | Lang | WER | Latency |
|---|---|---|---|---|---|
| 00 | anh em | Anh em. | vi | 0.0% (0/2) | 0.06s |
| 01 | buổi sáng | Buổi sáng. | vi | 0.0% (0/2) | 0.06s |
| 02 | bánh mì | Bánh mì. | vi | 0.0% (0/2) | 0.06s |
| 03 | bác sĩ | Bác sĩ. | vi | 0.0% (0/2) | 0.06s |
| 04 | cảm ơn | Cảm ơn. | vi | 0.0% (0/2) | 0.06s |
| 05 | cuộc sống | Cuộc sống. | vi | 0.0% (0/2) | 0.06s |
| 06 | cái gì | Cái gì? | vi | 0.0% (0/2) | 0.06s |
| 07 | cơ hội | Cơ hỏi. | vi | 50.0% (1/2) | 0.06s |
| 08 | chiến đấu | Chiến đấu. | vi | 0.0% (0/2) | 0.06s |
| 09 | anh dũng | Anh Dũng | vi | 0.0% (0/2) | 0.06s |
| 11 | cộng hoà xã hội chủ nghĩa việt nam | Công xã H hội chủ nghĩa Việt Nam | vi | 37.5% (3/8) | 0.07s |
| 13 | good morning | Kết nối nàyặng. | vi | 150.0% (3/2) | 0.06s |
| 15 | elephant | El | vi | 100.0% (1/1) | 0.06s |
| 17 | a rolling stone gathers no moss | Vánhton Garer. | vi | 100.0% (6/6) | 0.06s |

## Detail: qwen3-asr-1.7b (auto)

| Utt | Ref | Hyp | Lang | WER | Latency |
|---|---|---|---|---|---|
| 00 | anh em | An M. | en | 100.0% (2/2) | 0.19s |
| 01 | buổi sáng | buổi sáng. | vi | 0.0% (0/2) | 0.22s |
| 02 | bánh mì | bánh mì. | vi | 0.0% (0/2) | 0.24s |
| 03 | bác sĩ | bác sĩ. | vi | 0.0% (0/2) | 0.22s |
| 04 | cảm ơn | cảm ơn. | vi | 0.0% (0/2) | 0.25s |
| 05 | cuộc sống | cuộc sống. | vi | 0.0% (0/2) | 0.22s |
| 06 | cái gì | cái gì? | vi | 0.0% (0/2) | 0.22s |
| 07 | cơ hội | cơ hội. | vi | 0.0% (0/2) | 0.22s |
| 08 | chiến đấu | chiến đấu. | vi | 0.0% (0/2) | 0.22s |
| 09 | anh dũng | anh Dũng. | vi | 0.0% (0/2) | 0.20s |
| 11 | cộng hoà xã hội chủ nghĩa việt nam | Cộng hòa xã hội chủ nghĩa Việt Nam. | vi | 12.5% (1/8) | 0.37s |
| 13 | good morning | Good morning. | en | 0.0% (0/2) | 0.19s |
| 15 | elephant | Elephant. | en | 0.0% (0/1) | 0.19s |
| 17 | a rolling stone gathers no moss | A rolling stone gathers no moss. | en | 0.0% (0/6) | 0.29s |

## Detail: qwen3-asr-1.7b (hinted)

| Utt | Ref | Hyp | Lang | WER | Latency |
|---|---|---|---|---|---|
| 00 | anh em | anh em | vi | 0.0% (0/2) | 0.09s |
| 01 | buổi sáng | buổi sáng. | vi | 0.0% (0/2) | 0.14s |
| 02 | bánh mì | bánh mì. | vi | 0.0% (0/2) | 0.17s |
| 03 | bác sĩ | bác sĩ. | vi | 0.0% (0/2) | 0.14s |
| 04 | cảm ơn | cảm ơn. | vi | 0.0% (0/2) | 0.17s |
| 05 | cuộc sống | cuộc sống. | vi | 0.0% (0/2) | 0.14s |
| 06 | cái gì | cái gì? | vi | 0.0% (0/2) | 0.15s |
| 07 | cơ hội | cơ hội. | vi | 0.0% (0/2) | 0.14s |
| 08 | chiến đấu | chiến đấu. | vi | 0.0% (0/2) | 0.14s |
| 09 | anh dũng | anh Dũng. | vi | 0.0% (0/2) | 0.12s |
| 11 | cộng hoà xã hội chủ nghĩa việt nam | Cộng hòa xã hội chủ nghĩa Việt Nam. | vi | 12.5% (1/8) | 0.30s |
| 13 | good morning | Good morning. | en | 0.0% (0/2) | 0.12s |
| 15 | elephant | Elephant. | en | 0.0% (0/1) | 0.12s |
| 17 | a rolling stone gathers no moss | A rolling stone gathers no moss. | en | 0.0% (0/6) | 0.22s |

## Detail: whisper-large-v3 (auto)

| Utt | Ref | Hyp | Lang | WER | Latency |
|---|---|---|---|---|---|
| 00 | anh em | And, um... | — | 100.0% (2/2) | 0.31s |
| 01 | buổi sáng | buổi sáng | — | 0.0% (0/2) | 0.33s |
| 02 | bánh mì | Bonne mai ! | — | 100.0% (2/2) | 0.31s |
| 03 | bác sĩ | Backseat. | — | 100.0% (2/2) | 0.25s |
| 04 | cảm ơn | Cảm ơn. | — | 0.0% (0/2) | 0.41s |
| 05 | cuộc sống | Cuộc sống | — | 0.0% (0/2) | 0.31s |
| 06 | cái gì | Cái gì? | — | 0.0% (0/2) | 0.28s |
| 07 | cơ hội | Cơ hội | — | 0.0% (0/2) | 0.31s |
| 08 | chiến đấu | Chiến đấu | — | 0.0% (0/2) | 0.31s |
| 09 | anh dũng | Anh Dũng | — | 0.0% (0/2) | 0.31s |
| 11 | cộng hoà xã hội chủ nghĩa việt nam | Công Hòa Xã Hội Chủ Nghĩa Việt Nam | — | 25.0% (2/8) | 0.62s |
| 13 | good morning | Good morning. | — | 0.0% (0/2) | 0.25s |
| 15 | elephant | Elephant. | — | 0.0% (0/1) | 0.31s |
| 17 | a rolling stone gathers no moss | A rolling stone gathers no moss. | — | 0.0% (0/6) | 0.41s |

## Detail: whisper-large-v3 (hinted)

| Utt | Ref | Hyp | Lang | WER | Latency |
|---|---|---|---|---|---|
| 00 | anh em | anh em | vi | 0.0% (0/2) | 0.16s |
| 01 | buổi sáng | buổi sáng | vi | 0.0% (0/2) | 0.26s |
| 02 | bánh mì | Bánh mì | vi | 0.0% (0/2) | 0.24s |
| 03 | bác sĩ | Bác sĩ | vi | 0.0% (0/2) | 0.24s |
| 04 | cảm ơn | Cảm ơn. | vi | 0.0% (0/2) | 0.34s |
| 05 | cuộc sống | Cuộc sống | vi | 0.0% (0/2) | 0.24s |
| 06 | cái gì | Cái gì? | vi | 0.0% (0/2) | 0.21s |
| 07 | cơ hội | Cơ hội | vi | 0.0% (0/2) | 0.24s |
| 08 | chiến đấu | Chiến đấu | vi | 0.0% (0/2) | 0.24s |
| 09 | anh dũng | Anh Dũng | vi | 0.0% (0/2) | 0.24s |
| 11 | cộng hoà xã hội chủ nghĩa việt nam | Công Hòa Xã Hội Chủ Nghĩa Việt Nam | vi | 25.0% (2/8) | 0.55s |
| 13 | good morning | Good morning. | en | 0.0% (0/2) | 0.18s |
| 15 | elephant | Elephant. | en | 0.0% (0/1) | 0.24s |
| 17 | a rolling stone gathers no moss | A rolling stone gathers no moss. | en | 0.0% (0/6) | 0.34s |

## Non-deterministic outputs

None: every system produced identical text across passes.
