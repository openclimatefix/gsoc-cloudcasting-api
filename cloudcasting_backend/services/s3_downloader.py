"""Module for downloading data from S3 bucket on a 30 min interval."""

import datetime
import os
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

import boto3
from loguru import logger as log

from cloudcasting_backend.settings import settings


def ensure_directory_exists(func: Callable[..., None]) -> Callable[..., None]:
    """Decorator to ensure directory exists before downloading."""

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
) -> None:
    """
    Download the contents of a folder directory from S3.

    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system

        If None, files will be placed in the current directory
    """
    # Configure S3 client
    s3 = boto3.resource(
        "s3",
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        region_name=settings.s3_region_name,
    )

    log.info(f"Starting download from bucket {bucket_name}, folder {s3_folder}")
    bucket = s3.Bucket(bucket_name)
    downloaded_files = 0

    # If by an chance the folder utc timestamp based folder is not found, try to download latest.zarr
    objects = list(bucket.objects.filter(Prefix=s3_folder).limit(1))
    if not objects:
        log.warning(
            f"No objects found in bucket {bucket_name} with prefix {s3_folder} thus trying latest.zarr",
        )
        s3_folder = "cloudcasting_forecast/latest.zarr/"

    # Download the files
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = (
            obj.key
            if local_dir is None
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        )

        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))

        if obj.key[-1] == "/":
            continue

        bucket.download_file(obj.key, target)
        downloaded_files += 1

    log.info(
        f"Finished downloading {downloaded_files} files from {s3_folder} to {local_dir}",
    )


def get_current_forecast_folder(
    bucket_name: Optional[str] = None,
    target_time: Optional[datetime.datetime] = None,
) -> str:
    """
    Get the forecast folder name for a specific timestamp.

    Args:
        bucket_name: Unused parameter kept for backward compatibility
        target_time: Specific datetime to use for the folder name (defaults to current time if None)
                     Should be one of the half-hour marks (XX:00 or XX:30)

    Returns the path to the time-specific zarr folder.
    """
    if not target_time:
        # Default to current UTC time
        target_time = datetime.datetime.now(datetime.timezone.utc)
        # Round to the nearest half hour
        minutes = target_time.minute
        if minutes < 30:
            target_time = target_time.replace(minute=0, second=0, microsecond=0)
        else:
            target_time = target_time.replace(minute=30, second=0, microsecond=0)

    # Format the folder name as YYYY-MM-DDThh:mm.zarr/
    forecast_folder = (
        f"cloudcasting_forecast/{target_time.strftime('%Y-%m-%dT%H:%M')}.zarr/"
    )
    log.info(f"Using forecast folder: {forecast_folder}")
    return forecast_folder


def scheduled_download(
    bucket_name: str = settings.s3_bucket_name,
    local_dir: str = settings.zarr_storage_path,
    interval_minutes: int = 30,  # 30 min interval as the model updates every half hour
) -> None:
    """
    Schedule the download of S3 folder at regular intervals aligned to half-hour marks.

    Note: Files are generated with a ~15 minute delay after the half-hour mark.
    For example, the 18:30 file is ready around 18:45.

    Args:
        bucket_name: the name of the s3 bucket
        local_dir: local directory to store downloaded files
        interval_minutes: interval in minutes between downloads (default is 30 for half-hour marks)
    """
    log.info(
        f"Starting scheduled downloads every {interval_minutes} minutes, aligned to half-hour marks",
    )

    # Generation delay in minutes (time to wait after a half-hour mark for files to be generated)
    generation_delay = 15

    while True:
        try:
            # Get current UTC time
            now = datetime.datetime.now(datetime.timezone.utc)
            current_time = now.strftime("%Y-%m-%d %H:%M:%S UTC")
            log.info(f"Current time: {current_time}")

            # Determine which files are ready to download based on the 15-minute generation delay
            minutes_past_half_hour = now.minute % 30

            if now.minute < 30:
                if minutes_past_half_hour < generation_delay:
                    target_time = now.replace(
                        minute=30,
                        second=0,
                        microsecond=0,
                    ) - datetime.timedelta(hours=1)
                else:
                    target_time = now.replace(minute=0, second=0, microsecond=0)
            elif minutes_past_half_hour < generation_delay:
                target_time = now.replace(minute=0, second=0, microsecond=0)
            else:
                target_time = now.replace(minute=30, second=0, microsecond=0)

            log.info(
                f"Based on current time {current_time} with {generation_delay}min generation delay, downloading files for: {target_time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            )

            s3_folder = get_current_forecast_folder(target_time=target_time)
            current_half_hour = target_time.strftime("%Y-%m-%d %H:%M:%S UTC")
            log.info(f"Downloading data for half-hour mark: {current_half_hour}")
            download_s3_folder(bucket_name, s3_folder, local_dir)
            log.info(
                f"Download completed at {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            )

            # Calculate time until the next half-hour mark plus generation delay
            now = datetime.datetime.now(datetime.timezone.utc)
            if now.minute < 30:
                next_target = now.replace(minute=30, second=0, microsecond=0)
            else:
                next_target = now.replace(
                    minute=0,
                    second=0,
                    microsecond=0,
                ) + datetime.timedelta(hours=1)
            next_target_with_delay = next_target + datetime.timedelta(
                minutes=generation_delay,
            )

            # Calculate sleep time until next download (factoring in generation delay)
            sleep_seconds = (next_target_with_delay - now).total_seconds()

            # Ensure we don't have a negative sleep time due to execution delays
            if sleep_seconds <= 0:
                sleep_seconds = 120  # Default to 2 minute if calculation went wrong

            log.info(
                f"Next download scheduled for: {next_target_with_delay.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            )
            log.info(f"Sleeping for {sleep_seconds/60:.2f} minutes until next download")
            time.sleep(sleep_seconds)

        except Exception as e:
            log.opt(exception=e).error("Error during scheduled download")
            log.info("Retrying in 5 minutes")
            time.sleep(300)


def start_download_thread() -> None:
    """Start the download process in a separate thread."""
    download_thread = threading.Thread(
        target=scheduled_download,
        daemon=True,
    )
    log.info("Starting S3 downloader thread")
    download_thread.start()


if __name__ == "__main__":
    # just in case we want to run this script to test this file directly
    bucket_name = settings.s3_bucket_name
    local_dir = settings.zarr_storage_path
    log.info(
        "Starting S3 downloader with time-specific folder names (format: YYYY-MM-DDThh:mm.zarr/)",
    )
    scheduled_download(bucket_name, local_dir)
