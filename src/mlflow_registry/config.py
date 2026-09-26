import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv


@dataclass(frozen=True)
class Config:
    tracking_uri: str            # MLflow server
    s3_endpoint_url: str         # MinIO
    aws_access_key_id: str       # MinIO user
    aws_secret_access_key: str   # MinIO password


def load_config(dotenv_path: str | None = None, override: bool = False) -> Config:
    """Load settings from a .env file (if present) plus the real environment.

    Values already exported in the environment win over the .env file unless
    ``override`` is set. Loading into ``os.environ`` also lets mlflow and boto3
    pick the credentials up on their own.
    """
    load_dotenv(dotenv_path or find_dotenv(usecwd=True), override=override)

    def require(key: str) -> str:
        val = os.environ.get(key)
        if not val:
            raise RuntimeError(f"Missing environment variable: {key}")
        return val

    return Config(
        tracking_uri=require("MLFLOW_TRACKING_URI"),
        s3_endpoint_url=require("MLFLOW_S3_ENDPOINT_URL"),
        aws_access_key_id=require("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=require("AWS_SECRET_ACCESS_KEY"),
    )
