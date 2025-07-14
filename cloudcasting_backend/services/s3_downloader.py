"""
Module for downloading data from an S3 bucket on a 30-minute interval,
followed by conversion of the downloaded Zarr data to GeoTIFF format.
"""

import datetime
import os
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

import boto3
import numpy as np
import pyproj
import rasterio
import xarray as xr
from loguru import logger as log
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from scipy.interpolate import griddata

from cloudcasting_backend.settings import settings

# --- Configuration for GeoTIFF Conversion ---
GEOTIFF_STORAGE_PATH = "layers"  # Base directory for all GeoTIFF output
# Bounding box for cropping the GeoTIFF output [lon_min, lat_min, lon_max, lat_max]
GEOTIFF_BBOX = [-19.0, 42.0, 15.0, 65.0]
GEOTIFF_RESOLUTION = 0.075  # Output grid resolution in degrees

# ==============================================================================
# == GEO-TIFF CONVERSION FUNCTIONS
# ==============================================================================

def save_to_geotiff(filename: str, data: np.ndarray, lat_grid: np.ndarray, lon_grid: np.ndarray, nodata: float = np.nan) -> None:
    """
    Save a 2D data array with lat/lon grids to a GeoTIFF file in WGS84.

    Args:
        filename: Output GeoTIFF path.
        data: 2D numpy array of values.
        lat_grid: 2D latitudes of the interpolated grid.
        lon_grid: 2D longitudes of the interpolated grid.
        nodata: Value to represent no data.
    """
    height, width = data.shape
    lon_min, lon_max = np.min(lon_grid), np.max(lon_grid)
    lat_min, lat_max = np.min(lat_grid), np.max(lat_grid)

    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)

    try:
        with rasterio.open(
            filename,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=str(data.dtype),
            crs=CRS.from_epsg(4326),  # WGS84
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(data, 1)
        log.success(f"Successfully saved GeoTIFF to {filename}")
    except Exception as e:
        log.opt(exception=e).error(f"Failed to save GeoTIFF file: {filename}")


def convert_zarr_to_geotiffs(zarr_path: str, output_dir: str) -> None:
    """
    Load a cloud dataset from Zarr, iterate through all variables and steps,
    and save each slice as a cropped GeoTIFF file.

    Args:
        zarr_path: Path to the root of the downloaded Zarr dataset.
        output_dir: The directory where GeoTIFF files will be saved.
    """
    log.info(f"Starting Zarr to GeoTIFF conversion for: {zarr_path}")
    if not os.path.exists(zarr_path):
        log.error(f"Zarr path does not exist: {zarr_path}. Aborting conversion.")
        return

    try:
        ds = xr.open_zarr(zarr_path)
    except Exception as e:
        log.opt(exception=e).error(f"Failed to open Zarr dataset at {zarr_path}")
        return

    # --- 1. Pre-calculate projections and grids (these are constant for all layers) ---
    log.info("Preparing coordinate transformation...")
    x_coords = ds.x_geostationary.values
    y_coords = ds.y_geostationary.values
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)

    geos_proj = pyproj.Proj(proj='geos', h=35785831, lon_0=9.6, sweep='y')
    wgs84_proj = pyproj.Proj(proj='latlong', datum='WGS84')
    transformer = pyproj.Transformer.from_proj(geos_proj, wgs84_proj, always_xy=True)
    lon_grid, lat_grid = transformer.transform(x_mesh, y_mesh)

    mask = ~np.isnan(ds.sat_pred.isel(init_time=0, step=0, variable=0).values) & np.isfinite(lon_grid) & np.isfinite(lat_grid)
    points = np.column_stack((lon_grid[mask], lat_grid[mask]))

    lon_out = np.arange(GEOTIFF_BBOX[0], GEOTIFF_BBOX[2], GEOTIFF_RESOLUTION)
    lat_out = np.arange(GEOTIFF_BBOX[3], GEOTIFF_BBOX[1], -GEOTIFF_RESOLUTION)
    lon_target, lat_target = np.meshgrid(lon_out, lat_out)
    log.info(f"Target grid for interpolation created with shape {lon_target.shape}.")

    # --- 2. Iterate through all variables and steps ---
    variables = ds['variable'].values
    steps = ds['step'].values
    total_files = len(variables) * len(steps)
    log.info(f"Found {len(variables)} variables and {len(steps)} steps. Processing {total_files} files.")

    for var_idx, var_name in enumerate(variables):
        var_output_dir = os.path.join(output_dir, str(var_name))
        Path(var_output_dir).mkdir(parents=True, exist_ok=True)

        for step_idx in range(len(steps)):
            log.info(f"Processing: Variable '{var_name}' (index {var_idx}), Step {step_idx}")

            try:
                # Select the data slice
                data_slice = ds.sat_pred.isel(init_time=0, variable=var_idx, step=step_idx).values
                values = data_slice[mask]

                # Interpolate data onto the target WGS84 grid
                interp_data = griddata(
                    points, values, (lon_target, lat_target), method='cubic', fill_value=np.nan
                ).astype(np.float32)

                # Define final output path and save the file
                output_filename = os.path.join(var_output_dir, f"step_{step_idx}.tif")
                save_to_geotiff(output_filename, interp_data, lat_target, lon_target)

            except Exception as e:
                log.opt(exception=e).error(f"Error processing slice: var='{var_name}', step={step_idx}")
                continue # Continue to the next file

    ds.close()
    log.success(f"Finished GeoTIFF conversion for {zarr_path}. Processed {total_files} files.")


# ==============================================================================
# == S3 DATA DOWNLOADER FUNCTIONS
# ==============================================================================

def ensure_directory_exists(func: Callable[..., None]) -> Callable[..., None]:
    """Decorator to ensure directory exists before running the function."""
    @wraps(func)
    def wrapper(*args: object, **kwargs: object) -> None:
        local_dir = kwargs.get("local_dir") or args[2] if len(args) > 2 else None
        if local_dir:
            Path(str(local_dir)).mkdir(parents=True, exist_ok=True)
        return func(*args, **kwargs)
    return wrapper


@ensure_directory_exists
def download_s3_folder(
    bucket_name: str,
    s3_folder: str,
    local_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Download the contents of a folder directory from S3.

    Args:
        bucket_name: The name of the S3 bucket.
        s3_folder: The folder path in the S3 bucket.
        local_dir: A relative or absolute directory path in the local file system.

    Returns:
        The local path to the downloaded folder, or None on failure.
    """
    s3 = boto3.resource(
        "s3",
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        region_name=settings.s3_region_name,
    )

    log.info(f"Starting download from bucket '{bucket_name}', folder '{s3_folder}'")
    bucket = s3.Bucket(bucket_name)
    downloaded_files = 0
    
    # Define a specific destination directory for this download run
    zarr_folder_name = os.path.basename(s3_folder.rstrip('/'))
    destination_path = os.path.join(local_dir, zarr_folder_name)
    Path(destination_path).mkdir(parents=True, exist_ok=True)

    objects = list(bucket.objects.filter(Prefix=s3_folder).limit(1))
    if not objects:
        log.warning(f"No objects found with prefix '{s3_folder}'. Trying 'latest.zarr' as a fallback.")
        s3_folder = "cloudcasting_forecast/latest.zarr/"
        # Update destination path for fallback
        destination_path = os.path.join(local_dir, "latest.zarr")
        Path(destination_path).mkdir(parents=True, exist_ok=True)

    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = os.path.join(destination_path, os.path.relpath(obj.key, s3_folder))
        
        if obj.key.endswith("/"):
            os.makedirs(target, exist_ok=True)
            continue
        
        # Ensure parent directory of the file exists
        os.makedirs(os.path.dirname(target), exist_ok=True)
        
        bucket.download_file(obj.key, target)
        downloaded_files += 1

    if downloaded_files > 0:
        log.info(f"Finished downloading {downloaded_files} files from '{s3_folder}' to '{destination_path}'")
        return destination_path
    else:
        log.error(f"Download failed: No files were found in '{s3_folder}' to download.")
        return None


def get_current_forecast_folder(
    target_time: Optional[datetime.datetime] = None,
) -> str:
    """
    Get the S3 forecast folder name for a specific timestamp.
    """
    if not target_time:
        target_time = datetime.datetime.now(datetime.timezone.utc)
        minutes = target_time.minute
        target_time = target_time.replace(minute=(0 if minutes < 30 else 30), second=0, microsecond=0)

    forecast_folder = f"cloudcasting_forecast/{target_time.strftime('%Y-%m-%dT%H:%M')}.zarr/"
    log.info(f"Determined S3 forecast folder: {forecast_folder}")
    return forecast_folder


def scheduled_download(
    bucket_name: str = settings.s3_bucket_name,
    zarr_local_dir: str = settings.zarr_storage_path,
    geotiff_local_dir: str = GEOTIFF_STORAGE_PATH,
) -> None:
    """
    Schedule the download of an S3 folder and subsequent conversion to GeoTIFF.
    """
    log.info(f"Starting scheduled task. Zarrs will be saved to '{zarr_local_dir}', GeoTIFFs to '{geotiff_local_dir}'.")
    generation_delay = 15  # Wait 15 mins past the hour/half-hour for data to be ready

    while True:
        try:
            now = datetime.datetime.now(datetime.timezone.utc)
            
            # Determine the most recent, ready-to-download forecast time
            if now.minute >= 30 + generation_delay:
                target_time = now.replace(minute=30, second=0, microsecond=0)
            elif now.minute >= generation_delay:
                target_time = now.replace(minute=0, second=0, microsecond=0)
            else: # Not yet 15 mins past the hour, get previous half-hour slot
                target_time = now.replace(second=0, microsecond=0) - datetime.timedelta(minutes=now.minute % 30 + 30)
            
            log.info(f"Targeting forecast for time: {target_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            
            # --- 1. Download the data ---
            s3_folder = get_current_forecast_folder(target_time=target_time)
            downloaded_zarr_path = download_s3_folder(bucket_name, s3_folder, zarr_local_dir)
            
            # --- 2. Convert to GeoTIFF ---
            if downloaded_zarr_path:
                geotiff_run_dir = os.path.join(geotiff_local_dir, os.path.basename(downloaded_zarr_path))
                convert_zarr_to_geotiffs(downloaded_zarr_path, geotiff_run_dir)
            else:
                log.warning("Skipping GeoTIFF conversion because download failed or returned no files.")

            # --- 3. Calculate sleep time until next run ---
            now = datetime.datetime.now(datetime.timezone.utc)
            if now.minute < 30:
                next_run_base = now.replace(minute=30, second=0, microsecond=0)
            else:
                next_run_base = (now + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            
            next_run_with_delay = next_run_base + datetime.timedelta(minutes=generation_delay)
            sleep_seconds = (next_run_with_delay - now).total_seconds()
            
            if sleep_seconds <= 0:
                sleep_seconds = 60 # Default to 1 minute if calculation is off

            log.info(f"Next download check scheduled for: {next_run_with_delay.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            log.info(f"Sleeping for {sleep_seconds/60:.2f} minutes.")
            time.sleep(sleep_seconds)

        except Exception as e:
            log.opt(exception=e).critical("An unhandled error occurred in the scheduling loop.")
            log.info("Retrying in 5 minutes...")
            time.sleep(300)


def start_download_thread() -> None:
    """Start the download and conversion process in a separate thread."""
    download_thread = threading.Thread(
        target=scheduled_download,
        daemon=True,
    )
    log.info("Starting S3 downloader and GeoTIFF converter thread.")
    download_thread.start()


if __name__ == "__main__":
    # This block allows running the script directly for testing.
    # Ensure you have a 'cloudcasting_backend/settings.py' file or mock the settings object.
    log.info("Running S3 downloader and converter script directly...")
    scheduled_download(
        bucket_name=settings.s3_bucket_name,
        zarr_local_dir=settings.zarr_storage_path,
        geotiff_local_dir=GEOTIFF_STORAGE_PATH,
    )