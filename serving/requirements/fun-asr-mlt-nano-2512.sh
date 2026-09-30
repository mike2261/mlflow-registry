#!/bin/sh
# The FunASRNano model class is not in the weights repo: unpack a pinned commit of
# FunAudioLLM/Fun-ASR (Apache-2.0) into /opt/fun-asr, where stt_funasr.py imports it from.
set -eu
COMMIT=0339018ba74a7defa3b6b6a96718d17b816be77b
python - "$COMMIT" <<'PY'
import io, shutil, sys, tarfile, urllib.request
commit = sys.argv[1]
data = urllib.request.urlopen(f"https://codeload.github.com/FunAudioLLM/Fun-ASR/tar.gz/{commit}").read()
with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
    top = tf.getnames()[0].split("/", 1)[0]
    tf.extractall("/tmp/fun-asr")
shutil.copytree(f"/tmp/fun-asr/{top}", "/opt/fun-asr", dirs_exist_ok=True)
shutil.rmtree("/tmp/fun-asr")
print("Fun-ASR code", commit, "-> /opt/fun-asr")
PY
test -f /opt/fun-asr/model.py
