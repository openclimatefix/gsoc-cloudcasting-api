"""API endpoints for cloudcasting data."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from cloudcasting_backend.services.s3_downloader import (
    download_s3_folder,
    get_current_forecast_folder,
)
from cloudcasting_backend.settings import settings

router = APIRouter(prefix="/cloudcasting", tags=["cloudcasting"])


class CloudcastingResponse(BaseModel):
    """Response model for cloudcasting endpoint."""

    message: str
    data_path: str


@router.get("/status", response_model=CloudcastingResponse)
async def get_cloudcasting_status() -> CloudcastingResponse:
    """
    Get status of cloudcasting data.

    Returns information about the latest zarr file.
    """
    zarr_storage_path = Path(settings.zarr_storage_path)
    latest_path = zarr_storage_path / "cloudcasting_forecast" / "latest.zarr"

    # Check if the latest.zarr directory exists
    if not latest_path.exists():
        return CloudcastingResponse(message="No data available yet", data_path="")

    # Check if the directory has content
    file_count = sum(1 for _ in latest_path.rglob("*") if _.is_file())
    if file_count == 0:
        return CloudcastingResponse(message="Zarr directory is empty", data_path="")

    # Return path to latest data
    relative_path = latest_path.relative_to(Path(settings.zarr_storage_path).parent)

    return CloudcastingResponse(
        message="Data available",
        data_path=f"/static/{relative_path}",
    )


@router.post("/trigger-download", response_model=CloudcastingResponse)
async def trigger_download() -> CloudcastingResponse:
    """
    Manually trigger a download of the latest cloudcasting data.
    """
    try:
        bucket_name = settings.s3_bucket_name
        s3_folder = get_current_forecast_folder()
        local_dir = settings.zarr_storage_path

        # Ensure directory exists
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        # Download the data
        download_s3_folder(bucket_name, s3_folder, local_dir)

        latest_path = Path(local_dir) / "cloudcasting_forecast" / "latest.zarr"
        if latest_path.exists():
            relative_path = latest_path.relative_to(
                Path(settings.zarr_storage_path).parent,
            )
            data_path = f"/static/{relative_path}"
        else:
            data_path = ""

        return CloudcastingResponse(
            message="Download triggered successfully",
            data_path=data_path,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download data: {e!s}",
        )
