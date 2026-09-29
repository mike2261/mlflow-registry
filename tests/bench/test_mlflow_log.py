import uuid

import pytest
from mlflow.tracking import MlflowClient

from mlflow_registry.bench import mlflow_log, score
from mlflow_registry.bench.manifest import Utterance
from mlflow_registry.bench.metrics import scored
from mlflow_registry.config import load_config

UTTS = [
    Utterance("00", "00.wav", "anh em", "vi", "vi_short", (), 1.0),
    Utterance("13", "13.wav", "good morning", "en", "en_short", ("good", "morning"), 0.5),
]


def _cell() -> score.Cell:
    return score.Cell("chirp_3", "hinted", [
        scored(UTTS[0], "anh em", None, 0.9),
        scored(UTTS[1], None, "BackendError: 500", None),
    ], backend={"kind": "chirp", "project": "p", "location": "us", "model": "chirp_3",
                "auto_language_codes": ["vi-VN", "en-US"]},
        passes=3, requests=6, failed_requests=3, languages={"00": "vi", "13": None})


def test_metric_dict_is_flat_and_skips_none():
    m = mlflow_log.metric_dict(_cell())
    assert m["in_domain.wer"] == 0.0 and m["all.failures"] == 1 and m["all.n"] == 2
    assert m["wer_vi"] == 0.0
    assert "wer_en" not in m            # the only en row failed -> no reference words -> None -> dropped
    assert m["cost_usd_per_min"] == pytest.approx(0.016)
    assert m["failed_requests"] == 3 and m["requests"] == 6
    assert m["in_domain_no_numbers.wer"] == 0.0 and m["in_domain_no_numbers.ref_words"] == 2
    assert all(isinstance(v, (int, float)) for v in m.values())


@pytest.fixture
def experiment():
    cfg = load_config()
    name = f"test-eval-{uuid.uuid4().hex[:8]}"
    yield cfg.tracking_uri, name
    client = MlflowClient(tracking_uri=cfg.tracking_uri)
    exp = client.get_experiment_by_name(name)
    if exp:
        client.delete_experiment(exp.experiment_id)


def test_log_run_writes_params_metrics_and_artifacts(tmp_path, experiment):
    tracking_uri, exp_name = experiment
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "chirp_3.jsonl").write_text('{"system": "chirp_3"}\n')
    (run_dir / "report.md").write_text("# report\n")
    meta = {"run_id": "r1", "dataset": "stt-fixtures", "dataset_hash": "sha256:00",
            "created": "t", "harness_git_sha": "abc"}

    run_id = mlflow_log.log_run(meta, _cell(), run_dir, tracking_uri=tracking_uri,
                                experiment=exp_name)

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    assert run.data.params["system"] == "chirp_3"
    assert run.data.params["condition"] == "hinted"
    assert run.data.params["dataset_hash"] == "sha256:00"
    assert run.data.params["normalizer_version"] == mlflow_log.NORMALIZER_VERSION
    assert run.data.params["passes"] == "3"
    assert run.data.params["backend.kind"] == "chirp"
    assert run.data.params["backend.location"] == "us"
    assert run.data.params["backend.auto_language_codes"] == "vi-VN,en-US"
    assert run.data.tags["run_id"] == "r1"
    assert run.data.metrics["in_domain.wer"] == 0.0
    assert run.data.metrics["all.failures"] == 1.0
    names = {a.path for a in client.list_artifacts(run_id)}
    assert {"chirp_3.jsonl", "report.md"} <= names
