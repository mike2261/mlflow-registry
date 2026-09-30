"""Call a served model over REST.

    uv run python scripts/invoke.py stt whisper-large-v3 clip.wav [--language vi]
    uv run python scripts/invoke.py tts kokoro-82m "Hello there" --voice af_heart --out hello.wav
    uv run python scripts/invoke.py tts qwen3-tts-1.7b-base "Xin chao" --ref ref.wav --ref-text "..." --out out.wav

The port comes from the serving catalog; ``--host`` defaults to localhost
(use an SSH tunnel to reach the dev server). Only the standard library is
needed, so this also works outside the project venv with plain python.
"""

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:  # port lookup is a convenience; --port always works
    from mlflow_registry.serving.catalog import SERVING
except ImportError:  # pragma: no cover - running outside the venv
    SERVING = {}


def _post(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        sys.exit(f"HTTP {e.code} from {url}: {detail[:2000]}")


def _endpoint(args) -> str:
    port = args.port or (SERVING[args.model].port if args.model in SERVING else None)
    if port is None:
        sys.exit(f"unknown model {args.model!r}; pass --port")
    return f"http://{args.host}:{port}/invocations"


def cmd_stt(args) -> int:
    record = {"audio_b64": base64.b64encode(Path(args.audio).read_bytes()).decode()}
    if args.language:
        record["language"] = args.language
    out = _post(_endpoint(args), {"dataframe_records": [record]}, args.timeout)
    pred = out["predictions"][0]
    print(json.dumps(pred, ensure_ascii=False) if args.json else pred["text"])
    return 0


def cmd_tts(args) -> int:
    record = {"text": args.text}
    for key in ("voice", "language", "ref_text"):
        if getattr(args, key):
            record[key] = getattr(args, key)
    if args.ref:
        record["ref_audio_b64"] = base64.b64encode(Path(args.ref).read_bytes()).decode()
    out = _post(_endpoint(args), {"dataframe_records": [record]}, args.timeout)
    pred = out["predictions"][0]
    Path(args.out).write_bytes(base64.b64decode(pred["audio_b64"]))
    print(f"wrote {args.out} ({pred['sample_rate']} Hz)")
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("stt", "tts"):
        sp = sub.add_parser(name)
        sp.add_argument("model", help="registered model name, e.g. whisper-large-v3")
        sp.add_argument("--host", default="localhost")
        sp.add_argument("--port", type=int, help="override the catalog port")
        sp.add_argument("--timeout", type=float, default=600)
    sub.choices["stt"].add_argument("audio", help="wav/flac/ogg file")
    sub.choices["stt"].add_argument("--language")
    sub.choices["stt"].add_argument("--json", action="store_true", help="print the full prediction record")
    sub.choices["tts"].add_argument("text")
    sub.choices["tts"].add_argument("--voice")
    sub.choices["tts"].add_argument("--language")
    sub.choices["tts"].add_argument("--ref", help="reference clip for voice cloning")
    sub.choices["tts"].add_argument("--ref-text", dest="ref_text", help="transcript of the reference clip")
    sub.choices["tts"].add_argument("--out", default="out.wav")
    args = p.parse_args(argv)
    return cmd_stt(args) if args.cmd == "stt" else cmd_tts(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
