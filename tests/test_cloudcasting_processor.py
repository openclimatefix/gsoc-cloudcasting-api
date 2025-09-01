"""
Tests for cloudcasting_processor module functions including conversion functions,
satellite detection functions, timestamp handling, and local time functions.
"""

import datetime
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import xarray as xr
from botocore.exceptions import ClientError

from cloudcasting_backend.services.cloudcasting_processor import (
    GEOTIFF_STORAGE_PATH,
    S3_ZARR_PREFIX,
    _TIMESTAMP_FILE,
    convert_zarr_to_geotiffs,
    extract_satellite_info,
    get_current_forecast_folder,
    get_download_status,
    get_local_timestamp,
    get_s3_timestamp,
    parse_orbital_parameters,
    save_to_geotiff,
    trigger_background_download,
)


class TestConversionFunctions:
    """Test suite for data conversion functions."""

    def test_save_to_geotiff_success(self):
        """Test successful GeoTIFF file creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            data = np.random.rand(10, 10).astype(np.float32)
            lat_grid = np.linspace(50, 60, 10)
            lon_grid = np.linspace(-10, 0, 10)
            lat_grid, lon_grid = np.meshgrid(lat_grid, lon_grid)

            filename = os.path.join(temp_dir, "test.tif")

            # Test the function
            save_to_geotiff(filename, data, lat_grid, lon_grid)

            # Verify file was created
            assert os.path.exists(filename)

            # Verify file can be opened with rasterio
            import rasterio

            with rasterio.open(filename) as src:
                assert src.count == 1
                assert src.crs.to_epsg() == 4326
                assert src.shape == data.shape

    def test_save_to_geotiff_with_nodata(self):
        """Test GeoTIFF creation with custom nodata value."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data = np.full((5, 5), -999.0, dtype=np.float32)
            lat_grid = np.linspace(45, 55, 5)
            lon_grid = np.linspace(-5, 5, 5)
            lat_grid, lon_grid = np.meshgrid(lat_grid, lon_grid)

            filename = os.path.join(temp_dir, "test_nodata.tif")

            save_to_geotiff(filename, data, lat_grid, lon_grid, nodata=-999.0)

            assert os.path.exists(filename)

            import rasterio

            with rasterio.open(filename) as src:
                assert src.nodata == -999.0

    @patch("cloudcasting_backend.services.cloudcasting_processor.log")
    def test_save_to_geotiff_error_handling(self, mock_log):
        """Test error handling in save_to_geotiff."""
        # Invalid path should trigger error
        data = np.random.rand(5, 5)
        lat_grid = np.linspace(45, 55, 5)
        lon_grid = np.linspace(-5, 5, 5)
        lat_grid, lon_grid = np.meshgrid(lat_grid, lon_grid)

        invalid_filename = "/invalid/path/test.tif"

        save_to_geotiff(invalid_filename, data, lat_grid, lon_grid)

        # Should log an error
        mock_log.opt.assert_called()

    @patch("cloudcasting_backend.services.cloudcasting_processor.xr.open_zarr")
    @patch(
        "cloudcasting_backend.services.cloudcasting_processor.extract_satellite_info"
    )
    @patch("cloudcasting_backend.services.cloudcasting_processor.save_to_geotiff")
    def test_convert_zarr_to_geotiffs_success(
        self, mock_save, mock_extract, mock_open_zarr
    ):
        """Test successful conversion from Zarr to GeoTIFFs."""
        # Mock dataset
        mock_ds = Mock()
        mock_ds.x_geostationary.values = np.linspace(-1000000, 1000000, 10)
        mock_ds.y_geostationary.values = np.linspace(-1000000, 1000000, 10)

        # Mock the variable accessor properly
        mock_variable = Mock()
        mock_variable.values = ["variable1", "variable2"]
        mock_ds.__getitem__ = Mock(return_value=mock_variable)

        mock_ds.step.values = [0, 1, 2]
        mock_ds.sat_pred.isel.return_value.values = np.random.rand(10, 10)
        mock_ds.close = Mock()

        mock_open_zarr.return_value = mock_ds
        mock_extract.return_value = {
            "longitude": 9.5,
            "height": 35785831.0,
            "platform_name": "Meteosat-11",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            zarr_path = os.path.join(temp_dir, "test.zarr")
            os.makedirs(zarr_path)
            output_dir = os.path.join(temp_dir, "output")

            convert_zarr_to_geotiffs(zarr_path, output_dir)

            # Verify mocks were called
            mock_open_zarr.assert_called_once_with(zarr_path)
            mock_extract.assert_called_once_with(mock_ds)
            assert mock_save.call_count > 0

    @patch("cloudcasting_backend.services.cloudcasting_processor.log")
    def test_convert_zarr_to_geotiffs_nonexistent_path(self, mock_log):
        """Test error handling for non-existent Zarr path."""
        convert_zarr_to_geotiffs("/nonexistent/path", "/output")

        mock_log.error.assert_called()

    @patch("cloudcasting_backend.services.cloudcasting_processor.xr.open_zarr")
    @patch("cloudcasting_backend.services.cloudcasting_processor.log")
    def test_convert_zarr_to_geotiffs_open_error(self, mock_log, mock_open_zarr):
        """Test error handling when Zarr file cannot be opened."""
        mock_open_zarr.side_effect = Exception("Cannot open Zarr")

        with tempfile.TemporaryDirectory() as temp_dir:
            zarr_path = os.path.join(temp_dir, "test.zarr")
            os.makedirs(zarr_path)

            convert_zarr_to_geotiffs(zarr_path, "/output")

            mock_log.opt.assert_called()


class TestSatelliteDetectionFunctions:
    """Test suite for satellite detection and parameter parsing functions."""

    def test_parse_orbital_parameters_valid_string(self):
        """Test parsing valid orbital parameters string."""
        params_str = """
        satellite_actual_longitude: 9.5
        satellite_actual_altitude: 35785831.0
        satellite_nominal_longitude: 9.5
        projection_origin_longitude: 9.5
        """

        result = parse_orbital_parameters(params_str)

        assert result["satellite_actual_longitude"] == 9.5
        assert result["satellite_actual_altitude"] == 35785831.0
        assert result["satellite_nominal_longitude"] == 9.5
        assert result["projection_origin_longitude"] == 9.5

    def test_parse_orbital_parameters_mixed_types(self):
        """Test parsing orbital parameters with mixed data types."""
        params_str = """
        satellite_name: Meteosat-11
        longitude: 9.5
        altitude: 35785831
        status: operational
        """

        result = parse_orbital_parameters(params_str)

        assert result["satellite_name"] == "Meteosat-11"
        assert result["longitude"] == 9.5
        assert result["altitude"] == 35785831.0
        assert result["status"] == "operational"

    def test_parse_orbital_parameters_invalid_values(self):
        """Test parsing orbital parameters with invalid numeric values."""
        params_str = """
        satellite_name: Meteosat-11
        invalid_number: not_a_number
        longitude: 9.5
        """

        result = parse_orbital_parameters(params_str)

        assert result["satellite_name"] == "Meteosat-11"
        assert result["invalid_number"] == "not_a_number"  # Should remain as string
        assert result["longitude"] == 9.5

    def test_extract_satellite_info_meteosat_11(self):
        """Test satellite info extraction for Meteosat-11."""
        # Mock dataset with Meteosat-11 attributes
        mock_ds = Mock()
        mock_ds.sat_pred.attrs = {"platform_name": "Meteosat-11"}

        result = extract_satellite_info(mock_ds)

        assert result["longitude"] == 9.5
        assert result["height"] == 35785831.0

    def test_extract_satellite_info_meteosat_10(self):
        """Test satellite info extraction for Meteosat-10."""
        mock_ds = Mock()
        mock_ds.sat_pred.attrs = {"platform_name": "Meteosat-10"}

        result = extract_satellite_info(mock_ds)

        assert result["longitude"] == 0.0
        assert result["height"] == 35785831.0

    def test_extract_satellite_info_unknown_platform_with_orbital_params(self):
        """Test satellite info extraction for unknown platform with orbital parameters."""
        mock_ds = Mock()
        mock_ds.sat_pred.attrs = {
            "platform_name": "Unknown-Satellite",
            "orbital_parameters": """
            satellite_actual_longitude: 15.0
            satellite_actual_altitude: 36000000.0
            """,
        }

        result = extract_satellite_info(mock_ds)

        assert result["longitude"] == 15.0
        assert result["height"] == 36000000.0

    def test_extract_satellite_info_area_attribute(self):
        """Test satellite info extraction using area attribute."""
        mock_ds = Mock()
        mock_ds.sat_pred.attrs = {
            "platform_name": "Custom-Satellite",
            "area": "proj=geos lon_0: 12.0 h: 35786023.0 a=6378169.0",
        }

        result = extract_satellite_info(mock_ds)

        assert result["longitude"] == 12.0
        assert result["height"] == 35786023.0

    def test_extract_satellite_info_no_attributes(self):
        """Test satellite info extraction with no relevant attributes."""
        mock_ds = Mock()
        mock_ds.sat_pred.attrs = {}

        result = extract_satellite_info(mock_ds)

        # Should return default Meteosat-11 values
        assert result["longitude"] == 9.6
        assert result["height"] == 35785831.0

    def test_extract_satellite_info_no_sat_pred(self):
        """Test satellite info extraction when sat_pred is missing."""
        mock_ds = Mock()
        del mock_ds.sat_pred  # Remove sat_pred attribute

        result = extract_satellite_info(mock_ds)

        # Should return default values
        assert result["longitude"] == 9.6
        assert result["height"] == 35785831.0


class TestTimestampFunctions:
    """Test suite for timestamp-related functions."""

    @patch("cloudcasting_backend.services.cloudcasting_processor.boto3.client")
    def test_get_s3_timestamp_success(self, mock_boto_client):
        """Test successful S3 timestamp retrieval."""
        # Mock S3 client response
        mock_client = Mock()
        mock_boto_client.return_value = mock_client

        expected_time = datetime.datetime(2024, 1, 15, 12, 0, 0)
        mock_client.head_object.return_value = {"LastModified": expected_time}

        result = get_s3_timestamp("test-bucket", "test-prefix/")

        assert result == expected_time
        mock_client.head_object.assert_called_once_with(
            Bucket="test-bucket", Key="test-prefix/zarr.json"
        )

    @patch("cloudcasting_backend.services.cloudcasting_processor.boto3.client")
    @patch("cloudcasting_backend.services.cloudcasting_processor.log")
    def test_get_s3_timestamp_not_found(self, mock_log, mock_boto_client):
        """Test S3 timestamp retrieval when object not found."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client

        # Mock 404 error
        error_response = {"Error": {"Code": "404"}}
        mock_client.head_object.side_effect = ClientError(error_response, "HeadObject")

        result = get_s3_timestamp("test-bucket", "test-prefix/")

        assert result is None
        mock_log.error.assert_called()

    @patch("cloudcasting_backend.services.cloudcasting_processor.boto3.client")
    @patch("cloudcasting_backend.services.cloudcasting_processor.log")
    def test_get_s3_timestamp_other_error(self, mock_log, mock_boto_client):
        """Test S3 timestamp retrieval with other AWS errors."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client

        # Mock other error
        error_response = {"Error": {"Code": "403"}}
        mock_client.head_object.side_effect = ClientError(error_response, "HeadObject")

        result = get_s3_timestamp("test-bucket", "test-prefix/")

        assert result is None
        mock_log.opt.assert_called()

    def test_get_local_timestamp_success(self):
        """Test successful local timestamp retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a temporary timestamp file
            timestamp_file = Path(temp_dir) / "timestamp.txt"
            test_time = datetime.datetime(2024, 1, 15, 12, 0, 0)
            timestamp_file.write_text(test_time.isoformat())

            # Patch the global timestamp file path
            with patch(
                "cloudcasting_backend.services.cloudcasting_processor._TIMESTAMP_FILE",
                timestamp_file,
            ):
                result = get_local_timestamp()

                assert result == test_time

    def test_get_local_timestamp_file_not_found(self):
        """Test local timestamp retrieval when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_file = Path(temp_dir) / "nonexistent.txt"

            with patch(
                "cloudcasting_backend.services.cloudcasting_processor._TIMESTAMP_FILE",
                nonexistent_file,
            ):
                result = get_local_timestamp()

                assert result is None

    @patch("cloudcasting_backend.services.cloudcasting_processor.log")
    def test_get_local_timestamp_invalid_format(self, mock_log):
        """Test local timestamp retrieval with invalid timestamp format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            timestamp_file = Path(temp_dir) / "timestamp.txt"
            timestamp_file.write_text("invalid-timestamp-format")

            with patch(
                "cloudcasting_backend.services.cloudcasting_processor._TIMESTAMP_FILE",
                timestamp_file,
            ):
                result = get_local_timestamp()

                assert result is None
                mock_log.opt.assert_called()


class TestAPIIntegrationFunctions:
    """Test suite for API integration functions."""

    @patch("cloudcasting_backend.services.cloudcasting_processor.run_update_job")
    @patch("cloudcasting_backend.services.cloudcasting_processor._current_process")
    def test_trigger_background_download_success(
        self, mock_current_process, mock_run_job
    ):
        """Test successful background download trigger."""
        mock_current_process.is_alive.return_value = False
        mock_current_process.pid = 12345

        with patch(
            "cloudcasting_backend.services.cloudcasting_processor._process_lock"
        ):
            result = trigger_background_download()

            assert result == "12345"
            mock_run_job.assert_called_once()

    @patch("cloudcasting_backend.services.cloudcasting_processor._current_process")
    def test_trigger_background_download_already_running(self, mock_current_process):
        """Test background download trigger when job already running."""
        mock_current_process.is_alive.return_value = True
        mock_current_process.pid = 12345

        with patch(
            "cloudcasting_backend.services.cloudcasting_processor._process_lock"
        ):
            with pytest.raises(RuntimeError, match="Download job already running"):
                trigger_background_download()

    @patch("cloudcasting_backend.services.cloudcasting_processor._current_process")
    def test_get_download_status_running(self, mock_current_process):
        """Test download status when job is running."""
        mock_current_process.is_alive.return_value = True
        mock_current_process.pid = 12345

        with patch(
            "cloudcasting_backend.services.cloudcasting_processor._process_lock"
        ):
            with patch(
                "cloudcasting_backend.services.cloudcasting_processor._TIMESTAMP_FILE"
            ) as mock_file:
                mock_file.exists.return_value = True
                mock_file.read_text.return_value = "2024-01-15T12:00:00"

                result = get_download_status()

                assert result["is_running"] is True
                assert "Process 12345" in result["current_task"]
                assert result["last_completed"] == "2024-01-15T12:00:00"

    @patch(
        "cloudcasting_backend.services.cloudcasting_processor._current_process", None
    )
    def test_get_download_status_not_running(self):
        """Test download status when no job is running."""
        with patch(
            "cloudcasting_backend.services.cloudcasting_processor._process_lock"
        ):
            with patch(
                "cloudcasting_backend.services.cloudcasting_processor._TIMESTAMP_FILE"
            ) as mock_file:
                mock_file.exists.return_value = False

                result = get_download_status()

                assert result["is_running"] is False
                assert result["current_task"] is None
                assert result["last_completed"] is None

    def test_get_current_forecast_folder_exists(self):
        """Test getting current forecast folder when it exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some files in the directory
            test_file = Path(temp_dir) / "test_file.txt"
            test_file.write_text("test")

            with patch(
                "cloudcasting_backend.services.cloudcasting_processor.GEOTIFF_STORAGE_PATH",
                temp_dir,
            ):
                result = get_current_forecast_folder()

                assert result == temp_dir

    def test_get_current_forecast_folder_empty(self):
        """Test getting current forecast folder when it's empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "cloudcasting_backend.services.cloudcasting_processor.GEOTIFF_STORAGE_PATH",
                temp_dir,
            ):
                result = get_current_forecast_folder()

                assert result is None

    def test_get_current_forecast_folder_not_exists(self):
        """Test getting current forecast folder when it doesn't exist."""
        nonexistent_path = "/nonexistent/path"

        with patch(
            "cloudcasting_backend.services.cloudcasting_processor.GEOTIFF_STORAGE_PATH",
            nonexistent_path,
        ):
            result = get_current_forecast_folder()

            assert result is None


class TestConstants:
    """Test suite for module constants and configuration."""

    def test_constants_defined(self):
        """Test that required constants are properly defined."""
        assert GEOTIFF_STORAGE_PATH == "cloudcasting_backend/static/layers"
        assert S3_ZARR_PREFIX == "cloudcasting_forecast/latest.zarr/"

        # Test that _TIMESTAMP_FILE is properly constructed
        expected_timestamp_file = (
            Path(GEOTIFF_STORAGE_PATH) / "_last_processed_timestamp.txt"
        )
        assert _TIMESTAMP_FILE == expected_timestamp_file
