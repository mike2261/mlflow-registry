import base64
import io
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd
import soundfile as sf

from mlflow_registry.serving import catalog
from mlflow_registry.serving.audio import temp_wav
from mlflow_registry.serving.base import SttModel, TtsModel

SHORTLIST_NAMES = [
    "qwen3-asr-1.7b", "granite-speech-4.1-2b", "gipformer1.5-68m-rnnt",
    "parakeet-ctc-0.6b-vietnamese", "whisper-large-v3",
    "voxcpm2", "vieneu-tts-v3-turbo", "kokoro-82m", "qwen3-tts-1.7b-base",
    "phowhisper-large",
]


def test_catalog_covers_exactly_the_shortlisted_models():
    assert list(catalog.SERVING) == SHORTLIST_NAMES


def test_ports_are_unique_and_in_the_5001_5010_range():
    ports = [s.port for s in catalog.SERVING.values()]
    assert len(set(ports)) == len(ports)
    assert min(ports) == 5001 and max(ports) == 5010


def test_phowhisper_reuses_the_whisper_wrapper_on_port_5010():
    spec = catalog.spec("phowhisper-large")
    assert spec.task == "stt" and spec.port == 5010 and spec.gpu
    assert spec.wrapper == catalog.SERVING["whisper-large-v3"].wrapper


def test_every_wrapper_imports_without_its_runtime_and_subclasses_the_right_base():
    for spec in catalog.SERVING.values():
        cls = spec.wrapper_cls()
        expected = SttModel if spec.task == "stt" else TtsModel
        assert issubclass(cls, expected), spec.name
        assert cls.task == spec.task


def test_make_wrapper_pickles_by_reference_and_reloads(tmp_path):
    """The pyfunc artifact must stay tiny: class by reference, no runtime state."""
    spec = catalog.spec("whisper-large-v3")
    model = spec.make_wrapper("1")
    path = tmp_path / "m"
    mlflow.pyfunc.save_model(str(path), python_model=model, signature=spec.signature, pip_requirements=["mlflow"])
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    assert size < 200_000, f"pyfunc artifact unexpectedly large: {size} bytes"
    import cloudpickle
    obj = cloudpickle.loads((path / "python_model.pkl").read_bytes())
    assert type(obj).__module__ == "mlflow_registry.serving.stt_whisper"
    assert (obj.name, obj.version, obj.weights_dir) == ("whisper-large-v3", "1", None)


def test_signature_matches_task():
    assert catalog.spec("kokoro-82m").signature.inputs.input_names()[0] == "text"
    assert catalog.spec("whisper-large-v3").signature.inputs.input_names()[0] == "audio_b64"


def test_unknown_name_gives_helpful_error():
    try:
        catalog.spec("nope")
    except KeyError as e:
        assert "kokoro-82m" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_temp_wav_roundtrip_and_cleanup():
    samples = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    with temp_wav(samples, 16_000) as path:
        data, sr = sf.read(path, dtype="float32")
        assert sr == 16_000 and data.shape[0] == 1000
        kept = Path(path)
    assert not kept.exists()
