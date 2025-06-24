import enum
from pathlib import Path
from tempfile import gettempdir
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

TEMP_DIR = Path(gettempdir())


class LogLevel(str, enum.Enum):
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    # quantity of workers for uvicorn
    workers_count: int = 1
    # Enable uvicorn reloading
    reload: bool = False

    # Current environment
    environment: str = "dev"

    log_level: LogLevel = LogLevel.INFO

    # Sentry's configuration.
    sentry_dsn: Optional[str] = None
    sentry_sample_rate: float = 1.0

    # Grpc endpoint for opentelemetry.
    # E.G. http://localhost:4317
    opentelemetry_endpoint: Optional[str] = None

    # S3 configuration for cloudcasting data - dont worry about the empty string here, it is initialized from .env
    s3_bucket_name: str = ""
    s3_region_name: str = ""
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_download_interval: int = 30

    # Path to store downloaded zarr files
    zarr_storage_path: str = "cloudcasting_backend/static/zarr_files"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CLOUDCASTING_BACKEND_",
        env_file_encoding="utf-8",
    )


settings = Settings()
