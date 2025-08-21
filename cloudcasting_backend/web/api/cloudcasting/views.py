"""API endpoints for cloudcasting data."""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import sentry_sdk
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from cloudcasting_backend.services.s3_downloader import (
    get_download_status,
    run_update_job,
)
from cloudcasting_backend.settings import settings

router = APIRouter(prefix="/cloudcasting", tags=["cloudcasting"])


class CloudcastingResponse(BaseModel):
    """Response model for cloudcasting endpoint."""

    message: str
    data_path: str


class AvailableLayersResponse(BaseModel):
    """Response model for available layers."""

    channels: List[str]
    steps: List[int]


class BackgroundTaskResponse(BaseModel):
    """Response model for background tasks."""

    task_id: str
    message: str


class DownloadStatusResponse(BaseModel):
    """Response model for download status."""

    is_running: bool
    current_task: Optional[str] = None
    last_completed: Optional[str] = None
    error: Optional[str] = None


class ProcessedDataInfoResponse(BaseModel):
    """Response model for processed data information."""

    file_exists: bool
    init_time: Optional[str] = None
    forecast_steps: Optional[List[int]] = None
    variables: Optional[List[str]] = None
    file_size_mb: Optional[float] = None
    last_modified: Optional[str] = None
    time_range: Optional[dict] = None
    error: Optional[str] = None


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


@router.post("/trigger-download", response_model=BackgroundTaskResponse)
async def trigger_download() -> BackgroundTaskResponse:
    """
    Trigger a background download of the latest cloudcasting data.
    Downloads the data and converts it to GeoTIFF format in the background.
    Returns immediately with a task ID that can be used to check status.
    """
    try:
        # Directly call run_update_job to start the background process
        run_update_job()

        # Get the current status to return task info
        status_info = get_download_status()
        task_id = status_info.get("current_task", "unknown")

        return BackgroundTaskResponse(
            task_id=task_id,
            message=f"Background download started: {task_id}",
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start download: {e!s}",
        )


@router.get("/download-status", response_model=DownloadStatusResponse)
async def get_download_status_endpoint() -> DownloadStatusResponse:
    """
    Get the current status of background download processes.
    """
    status_info = get_download_status()
    return DownloadStatusResponse(**status_info)


@router.get("/layers", response_model=AvailableLayersResponse)
async def get_available_layers() -> AvailableLayersResponse:
    """
    Get list of available channels and steps for TIF layers.

    Returns the available variable names (channels) and step indices.
    """
    layers_path = Path(settings.geotiff_storage_path)

    if not layers_path.exists():
        return AvailableLayersResponse(channels=[], steps=[])

    channels = []
    all_steps = set()

    # Scan for available channels (variable directories)
    for channel_dir in layers_path.iterdir():
        if channel_dir.is_dir():
            channels.append(channel_dir.name)

            # Scan for available steps in this channel
            for tif_file in channel_dir.glob("*.tif"):
                # Extract step number from filename (e.g., "step_0.tif" -> 0)
                if tif_file.stem.isdigit():
                    all_steps.add(int(tif_file.stem))

    return AvailableLayersResponse(
        channels=sorted(channels),
        steps=sorted(list(all_steps)),
    )


@router.get("/data-info", response_model=ProcessedDataInfoResponse)
async def get_processed_data_info() -> ProcessedDataInfoResponse:
    """
    Get detailed information about the processed data including time data.

    Returns metadata about the latest processed data including:
    - Last processed timestamp from the update logic
    - Available variables and steps (from processed GeoTIFF layers)
    - Processing status information
    """
    # Path to the timestamp file used by the update logic
    timestamp_file = (
        Path(settings.geotiff_storage_path) / "_last_processed_timestamp.txt"
    )
    layers_path = Path(settings.geotiff_storage_path)

    # Check if we have any processed data
    if not layers_path.exists() or not any(layers_path.iterdir()):
        return ProcessedDataInfoResponse(
            file_exists=False, error="No processed data available yet"
        )

    try:
        # Get the last processed timestamp from the update logic
        last_processed_time = None
        if timestamp_file.exists():
            try:
                timestamp_content = timestamp_file.read_text().strip()
                last_processed_time = timestamp_content
            except Exception as e:
                logging.warning(f"Could not read timestamp file: {e}")

        # Get available variables and steps from the processed GeoTIFF layers
        variables = []
        all_steps = set()

        for channel_dir in layers_path.iterdir():
            if channel_dir.is_dir() and not channel_dir.name.startswith("_"):
                variables.append(channel_dir.name)

                # Get available steps from this channel
                for tif_file in channel_dir.glob("*.tif"):
                    if tif_file.stem.isdigit():
                        all_steps.add(int(tif_file.stem))

        steps = sorted(list(all_steps))

        # Calculate total size of processed layers
        file_size_mb = None
        try:
            total_size = sum(
                f.stat().st_size for f in layers_path.rglob("*.tif") if f.is_file()
            )
            file_size_mb = round(total_size / (1024 * 1024), 2)
        except Exception as e:
            logging.warning(f"Could not calculate file size: {e}")

        # Create time range info using the timestamp from the update logic
        time_range = None
        if last_processed_time:
            time_range = {
                "last_processed": last_processed_time,
                "data_source": "S3 download timestamp",
                "total_forecast_steps": len(steps) if steps else 0,
                "available_variables": len(variables),
            }

        return ProcessedDataInfoResponse(
            file_exists=True,
            init_time=None,  # Not available from processed layers (zarr is temporary)
            forecast_steps=steps,
            variables=variables,
            file_size_mb=file_size_mb,
            last_modified=last_processed_time,
            time_range=time_range,
        )

    except Exception as e:
        return ProcessedDataInfoResponse(
            file_exists=False, error=f"Error reading processed data: {str(e)}"
        )


@router.get("/layers/{channel}/{step}.tif")
async def get_tif_layer(channel: str, step: int):
    """
    Serve a specific TIF file for the given channel and step.

    Args:
        channel: The variable name (channel)
        step: The step number

    Returns:
        FileResponse: The TIF file for display in Mapbox
    """
    # Construct the file path
    tif_path = Path(settings.geotiff_storage_path) / channel / f"{step}.tif"

    # Check if file exists
    if not tif_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"TIF file not found for channel '{channel}' and step '{step}'",
        )

    # Return the file
    return FileResponse(
        path=str(tif_path),
        media_type="image/tiff",
        filename=f"{channel}_step_{step}.tif",
    )
