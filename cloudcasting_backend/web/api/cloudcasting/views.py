"""API endpoints for cloudcasting data."""

import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from cloudcasting_backend.services.s3_downloader import (
    download_s3_folder,
    get_current_forecast_folder,
    convert_zarr_to_geotiffs,
    GEOTIFF_STORAGE_PATH,
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
    Downloads the data and converts it to GeoTIFF format, then completes.
    """
    try:
        bucket_name = settings.s3_bucket_name
        s3_folder = get_current_forecast_folder()
        local_dir = settings.zarr_storage_path

        # Ensure directory exists
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        # Download the data
        downloaded_zarr_path = download_s3_folder(bucket_name, s3_folder, local_dir)

        # Convert to GeoTIFF if download was successful
        if downloaded_zarr_path:
            convert_zarr_to_geotiffs(downloaded_zarr_path, GEOTIFF_STORAGE_PATH)

        # Check if data is available and return path
        latest_path = Path(local_dir) / "cloudcasting_forecast" / "latest.zarr"
        if latest_path.exists():
            relative_path = latest_path.relative_to(
                Path(settings.zarr_storage_path).parent,
            )
            data_path = f"/static/{relative_path}"
        else:
            data_path = ""

        return CloudcastingResponse(
            message="Download and conversion completed successfully",
            data_path=data_path,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download data: {e!s}",
        )


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
        steps=sorted(list(all_steps))
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
            detail=f"TIF file not found for channel '{channel}' and step '{step}'"
        )
    
    # Return the file
    return FileResponse(
        path=str(tif_path),
        media_type="image/tiff",
        filename=f"{channel}_step_{step}.tif"
    )
