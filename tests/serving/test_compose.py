import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import gen_compose  # noqa: E402

from mlflow_registry.serving.catalog import SERVING  # noqa: E402


def test_checked_in_compose_file_matches_catalog():
    assert gen_compose.main(["--check"]) == 0, "run: uv run python scripts/gen_compose.py"


def test_compose_has_one_service_per_model_with_matching_port_and_gpu():
    doc = yaml.safe_load(gen_compose.render())
    services = doc["services"]
    assert "serving-base" in services
    for spec in SERVING.values():
        svc = services[f"serve-{spec.name}"]
        assert svc["profiles"] == [spec.name]
        assert svc["ports"] == [f"{spec.port}:8080"]
        assert svc["environment"]["SERVING_MODEL_URI"] == f"models:/{spec.name}@serving"
        assert svc["build"]["args"]["MODEL"] == spec.name
        assert ("deploy" in svc) == spec.gpu, spec.name
    assert "serving_models" in doc["volumes"]


def test_requirements_file_exists_for_every_model():
    for name in SERVING:
        assert (ROOT / "serving" / "requirements" / f"{name}.txt").exists(), name
