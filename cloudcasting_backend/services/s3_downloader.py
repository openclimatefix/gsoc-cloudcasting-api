"""
Module for on-demand downloading and processing of S3 Zarr data,
triggered by an external scheduler (e.g., cron).
"""

import datetime
import multiprocessing
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
import numpy as np
import pyproj
import rasterio
import xarray as xr
from botocore.exceptions import ClientError
from loguru import logger as log
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from scipy.interpolate import griddata

from cloudcasting_backend.settings import settings

# --- Configuration for GeoTIFF Conversion ---
GEOTIFF_STORAGE_PATH = (
    "cloudcasting_backend/static/layers"  # Base directory for all GeoTIFF output
)
# Bounding box for cropping the GeoTIFF output [lon_min, lat_min, lon_max, lat_max]
GEOTIFF_BBOX = [-19.0, 42.0, 15.0, 65.0]
GEOTIFF_RESOLUTION = 0.075  # Output grid resolution in degrees
S3_ZARR_PREFIX = (
    "cloudcasting_forecast/latest.zarr/"  # The S3 prefix for the latest forecast
)

# --- State Management for Cron Job ---
_process_lock = multiprocessing.Lock()
_current_process: Optional[multiprocessing.Process] = None
_TIMESTAMP_FILE = Path(GEOTIFF_STORAGE_PATH) / "_last_processed_timestamp.txt"


# ==============================================================================
# == GEO-TIFF CONVERSION FUNCTIONS (Unchanged)
# ==============================================================================


def save_to_geotiff(
    filename: str,
    data: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    nodata: float = np.nan,
) -> None:
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
    # Get bounds from the target grid, which is already cropped
    lon_min, lon_max = np.min(lon_grid), np.max(lon_grid)
    lat_min, lat_max = np.min(lat_grid), np.max(lat_grid)

    # from_bounds expects: west, south, east, north
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)

    try:
        with rasterio.open(
            filename,
            "w",
            driver="GTiff",
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


def parse_orbital_parameters(orbital_params_str: str) -> Dict[str, Any]:
    """
    Parse the orbital parameters string from Zarr attributes.

    Args:
        orbital_params_str: The multi-line string from metadata.

    Returns:
        A dictionary of parameter values.
    """
    params = {}
    for line in orbital_params_str.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            try:
                params[key.strip()] = float(value.strip())
            except ValueError:
                params[key.strip()] = value.strip()
    return params


def extract_satellite_info(ds: xr.Dataset) -> Dict[str, float]:
    """
    Extract satellite position information from the Zarr dataset attributes.
    This function checks for Meteosat-10/11 specifically and has fallbacks.

    Args:
        ds: The opened xarray Dataset.

    Returns:
        Dictionary with satellite parameters {'longitude': float, 'height': float}.
    """
    # Default values set to Meteosat-11 parameters
    satellite_info = {"longitude": 9.6, "height": 35785831.0}
    platform_name = "Unknown"

    # Calibration values for Meteosat-10 and Meteosat-11
    METEOSAT_10_LONGITUDE = 0.0
    METEOSAT_11_LONGITUDE = 9.5
    SATELLITE_HEIGHT = 35785831.0

    if hasattr(ds, "sat_pred") and hasattr(ds.sat_pred, "attrs"):
        attrs = ds.sat_pred.attrs
        if "platform_name" in attrs:
            platform_name = attrs["platform_name"]
            log.info(f"📡 Detected satellite platform: {platform_name}")

            if "Meteosat-10" in platform_name:
                satellite_info["longitude"] = METEOSAT_10_LONGITUDE
                satellite_info["height"] = SATELLITE_HEIGHT
                log.info("Using hard-coded parameters for Meteosat-10.")
            elif "Meteosat-11" in platform_name:
                satellite_info["longitude"] = METEOSAT_11_LONGITUDE
                satellite_info["height"] = SATELLITE_HEIGHT
                log.info("Using hard-coded parameters for Meteosat-11.")
            else:
                log.warning(
                    f"Platform '{platform_name}' not specifically handled. Attempting fallback attribute search.",
                )
                # Fallback for other satellites by parsing attributes
                if "orbital_parameters" in attrs:
                    orbital_params = parse_orbital_parameters(
                        attrs["orbital_parameters"],
                    )
                    if "satellite_actual_longitude" in orbital_params:
                        satellite_info["longitude"] = orbital_params[
                            "satellite_actual_longitude"
                        ]
                    if "satellite_actual_altitude" in orbital_params:
                        satellite_info["height"] = orbital_params[
                            "satellite_actual_altitude"
                        ]
                elif "area" in attrs:
                    area_str = attrs["area"]
                    lon_match = re.search(r"lon_0:\s*([\d.-]+)", area_str)
                    if lon_match:
                        satellite_info["longitude"] = float(lon_match.group(1))
                    h_match = re.search(r"h:\s*([\d.-]+)", area_str)
                    if h_match:
                        satellite_info["height"] = float(h_match.group(1))

    log.info(
        f"Using projection parameters for {platform_name}: "
        f"Longitude={satellite_info['longitude']}°, Height={satellite_info['height']}m",
    )
    return satellite_info


def convert_zarr_to_geotiffs(zarr_path: str, output_dir: str) -> None:
    """
    Load a cloud dataset from Zarr, iterate through all variables and steps,
    and save each slice as a cropped and reprojected GeoTIFF file.

    This function dynamically determines the satellite projection based on Zarr
    metadata, supporting platforms like Meteosat-10 and Meteosat-11.

    Args:
        zarr_path: Path to the root of the downloaded Zarr dataset.
        output_dir: The base directory where variable-specific GeoTIFF folders will be created.
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

    # --- 1. Extract Satellite Info & Pre-calculate transformations ---
    satellite_info = extract_satellite_info(ds)
    x_coords, y_coords = ds.x_geostationary.values, ds.y_geostationary.values
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)

    geos_proj = pyproj.Proj(
        proj="geos",
        h=satellite_info["height"],
        lon_0=satellite_info["longitude"],
        sweep="y",
    )
    wgs84_proj = pyproj.Proj(proj="latlong", datum="WGS84")
    transformer = pyproj.Transformer.from_proj(geos_proj, wgs84_proj, always_xy=True)
    lon_grid, lat_grid = transformer.transform(x_mesh, y_mesh)
    valid_coord_mask = np.isfinite(lon_grid) & np.isfinite(lat_grid)

    lon_out = np.arange(GEOTIFF_BBOX[0], GEOTIFF_BBOX[2], GEOTIFF_RESOLUTION)
    lat_out = np.arange(GEOTIFF_BBOX[3], GEOTIFF_BBOX[1], -GEOTIFF_RESOLUTION)
    lon_target, lat_target = np.meshgrid(lon_out, lat_out)
    log.info(f"Target grid for interpolation created with shape {lon_target.shape}.")

    # --- 2. Iterate through all variables and steps ---
    variables = ds["variable"].values
    steps = ds["step"].values
    total_files = len(variables) * len(steps)
    log.info(
        f"Found {len(variables)} variables and {len(steps)} steps. Processing {total_files} files.",
    )

    for var_idx, var_name in enumerate(variables):
        # Save layers to /layers/VAR_NAME/0.tif, /layers/VAR_NAME/1.tif etc.
        var_output_dir = os.path.join(output_dir, str(var_name))
        Path(var_output_dir).mkdir(parents=True, exist_ok=True)

        for step_idx in range(len(steps)):
            log.info(f"Processing: Variable '{var_name}', Step {step_idx}")
            try:
                data_slice = ds.sat_pred.isel(
                    init_time=0,
                    variable=var_idx,
                    step=step_idx,
                ).values
                valid_data_mask = ~np.isnan(data_slice)
                final_mask = valid_coord_mask & valid_data_mask

                points = np.column_stack((lon_grid[final_mask], lat_grid[final_mask]))
                values = data_slice[final_mask]

                if points.shape[0] < 4:
                    log.warning(
                        f"Not enough valid data points ({points.shape[0]}) for interpolation. Skipping.",
                    )
                    continue

                interp_data = griddata(
                    points,
                    values,
                    (lon_target, lat_target),
                    method="cubic",
                    fill_value=np.nan,
                ).astype(np.float32)

                output_filename = os.path.join(var_output_dir, f"{step_idx}.tif")
                save_to_geotiff(output_filename, interp_data, lat_target, lon_target)

            except Exception as e:
                log.opt(exception=e).error(
                    f"Error processing slice: var='{var_name}', step={step_idx}",
                )
                continue

    ds.close()
    log.success(
        f"Finished GeoTIFF conversion for {zarr_path}. Processed {total_files} files.",
    )


# ==============================================================================
# == S3 DATA DOWNLOADER FUNCTIONS
# ==============================================================================


def download_s3_folder(
    bucket_name: str,
    s3_folder: str,
    local_dir: str,
) -> Optional[str]:
    """
    Download the contents of a folder directory from S3.

    Args:
        bucket_name: The name of the S3 bucket.
        s3_folder: The prefix (folder) to download from S3.
        local_dir: A local directory path to download to.

    Returns:
        The local path to the downloaded folder, or None on failure.
    """
    s3 = boto3.resource(
        "s3",
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        region_name=settings.s3_region_name,
    )
    bucket = s3.Bucket(bucket_name)
    objects = list(bucket.objects.filter(Prefix=s3_folder).limit(1))
    if not objects:
        log.error(
            f"S3 prefix '{s3_folder}' does not exist or is empty in bucket '{bucket_name}'.",
        )
        return None

    log.info(f"Starting download from s3://{bucket_name}/{s3_folder} to '{local_dir}'")
    downloaded_files = 0
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not obj.key.endswith("/"):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            bucket.download_file(obj.key, target)
            downloaded_files += 1

    if downloaded_files > 0:
        log.success(f"Finished downloading {downloaded_files} files to '{local_dir}'")
        return local_dir
    log.error(f"Download failed: No files were found under prefix '{s3_folder}'.")
    return None


# ================================================================================
# == CRON-TRIGGERED JOB LOGIC
# ================================================================================


def get_s3_timestamp(bucket_name: str, s3_prefix: str) -> Optional[datetime.datetime]:
    """Get the LastModified timestamp of the .zattrs file in the S3 prefix."""
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            region_name=settings.s3_region_name,
        )
        # A key that should always exist and represent the dataset's age
        key_to_check = f"{s3_prefix.rstrip('/')}/.zattrs"
        response = s3_client.head_object(Bucket=bucket_name, Key=key_to_check)
        s3_time = response["LastModified"]
        log.info(f"S3 timestamp for '{key_to_check}' is {s3_time}")
        return s3_time
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            log.error(f"S3 object not found: {key_to_check}")
        else:
            log.opt(exception=e).error("Failed to get S3 object metadata.")
        return None


def get_local_timestamp() -> Optional[datetime.datetime]:
    """Reads the last processed timestamp from the local file."""
    try:
        if _TIMESTAMP_FILE.exists():
            content = _TIMESTAMP_FILE.read_text().strip()
            ts = datetime.datetime.fromisoformat(content)
            log.info(f"Found local timestamp: {ts}")
            return ts
        log.info("Local timestamp file not found. Will assume this is a first run.")
        return None
    except Exception as e:
        log.opt(exception=e).error("Could not read or parse local timestamp file.")
        return None


def _job_worker(bucket_name: str, s3_prefix: str, geotiff_dir: str) -> None:
    """
    The core logic for the update job, designed to be run in a separate process.
    """
    try:
        # 1. Check if an update is needed
        s3_time = get_s3_timestamp(bucket_name, s3_prefix)
        if not s3_time:
            log.error("Could not retrieve S3 timestamp. Aborting job.")
            return

        local_time = get_local_timestamp()
        if local_time and s3_time <= local_time:
            log.info("S3 data is not newer than last processed data. No update needed.")
            return

        log.info("Newer data found on S3. Starting download and conversion process.")

        # 2. Download to a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_local_path = os.path.join(tmpdir, "latest.zarr")
            downloaded_path = download_s3_folder(
                bucket_name=bucket_name,
                s3_folder=s3_prefix,
                local_dir=zarr_local_path,
            )

            if not downloaded_path:
                log.error("Download failed. Aborting job.")
                return

            # 3. Convert to GeoTIFFs in the final destination
            # First, ensure the destination exists
            Path(geotiff_dir).mkdir(parents=True, exist_ok=True)
            convert_zarr_to_geotiffs(downloaded_path, geotiff_dir)

            # 4. On success, update the local timestamp file
            try:
                _TIMESTAMP_FILE.write_text(s3_time.isoformat())
                log.success(
                    f"Successfully updated local timestamp to {s3_time.isoformat()}",
                )
            except Exception as e:
                log.opt(exception=e).error(
                    "CRITICAL: Failed to write new timestamp after successful processing.",
                )

    except Exception as e:
        log.opt(exception=e).critical(
            "An unhandled exception occurred in the job worker process.",
        )
    finally:
        log.info("Job worker process finished.")


def run_update_job() -> None:
    """
    The main entry point to be called by an external scheduler (cron).

    It manages a single background process, ensuring that only one instance
    of the download/conversion job runs at a time. If a new job is triggered
    while an old one is running, the new request is ignored.
    """
    global _current_process
    with _process_lock:
        if _current_process and _current_process.is_alive():
            log.info(
                f"Previous job (PID: {_current_process.pid}) is still running. Ignoring new request.",
            )
            return

        log.info("Starting new background job for data download and conversion.")
        _current_process = multiprocessing.Process(
            target=_job_worker,
            args=(
                settings.s3_bucket_name,
                S3_ZARR_PREFIX,
                GEOTIFF_STORAGE_PATH,
            ),
            daemon=True,
        )
        _current_process.start()
        log.info(f"Job started with PID: {_current_process.pid}")


# ==============================================================================
# == API INTEGRATION FUNCTIONS
# ==============================================================================


def trigger_background_download() -> str:
    """
    Trigger a background download and return a task ID.

    Returns:
        A task ID that can be used to track the job.

    Raises:
        RuntimeError: If a job is already running.
    """
    global _current_process
    with _process_lock:
        if _current_process and _current_process.is_alive():
            raise RuntimeError(
                f"Download job already running with PID: {_current_process.pid}",
            )

        # Start the background job
        run_update_job()

        # Return the process ID as task ID
        if _current_process:
            return str(_current_process.pid)
        raise RuntimeError("Failed to start background process")


def get_download_status() -> Dict[str, Any]:
    """
    Get the current status of the download process.

    Returns:
        Dictionary with status information.
    """
    global _current_process

    with _process_lock:
        is_running = bool(_current_process and _current_process.is_alive())

        status_info = {
            "is_running": is_running,
            "current_task": None,
            "last_completed": None,
            "error": None,
        }

        if is_running and _current_process:
            status_info["current_task"] = (
                f"Process {_current_process.pid} downloading and converting data"
            )

        # Check for last processed timestamp
        if _TIMESTAMP_FILE.exists():
            try:
                last_processed = _TIMESTAMP_FILE.read_text().strip()
                status_info["last_completed"] = last_processed
            except Exception as e:
                log.opt(exception=e).warning("Could not read last processed timestamp")

        return status_info


def get_current_forecast_folder() -> Optional[str]:
    """
    Get the path to the current forecast folder if it exists.

    Returns:
        Path to the current forecast folder or None if not available.
    """
    layers_path = Path(GEOTIFF_STORAGE_PATH)
    if layers_path.exists() and any(layers_path.iterdir()):
        return str(layers_path)
    return None
