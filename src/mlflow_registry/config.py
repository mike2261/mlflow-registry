import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv


@dataclass(frozen=True)
class Config:
    tracking_uri: str                  # MLflow server (required)
    s3_endpoint_url: str = ""          # MinIO — only for direct S3 access
    aws_access_key_id: str = ""        # MinIO user — only for direct S3 access
    aws_secret_access_key: str = ""    # MinIO password — only for direct S3 access


def load_config(dotenv_path: str | None = None, override: bool = False) -> Config:
    """Load settings from a .env file (if present) plus the real environment.

    Values already exported in the environment win over the .env file unless
    ``override`` is set. Loading into ``os.environ`` also lets mlflow and boto3
    pick the credentials up on their own.

    Only ``MLFLOW_TRACKING_URI`` is required: consumers behind the MLflow
    artifact proxy (``--serve-artifacts``) never talk to MinIO directly.
    """
    load_dotenv(dotenv_path or find_dotenv(usecwd=True), override=override)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("Missing environment variable: MLFLOW_TRACKING_URI")

    return Config(
        tracking_uri=tracking_uri,
        s3_endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL", ""),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    )
