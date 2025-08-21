import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from starlette import status


@pytest.mark.anyio
async def test_health(client: AsyncClient, fastapi_app: FastAPI) -> None:
    """
    Checks the health endpoint.

    :param client: client for the app.
    :param fastapi_app: current FastAPI application.
    """
    url = fastapi_app.url_path_for("health_check")
    response = await client.get(url)
    assert response.status_code == status.HTTP_200_OK


@pytest.mark.anyio
async def test_data_info_endpoint(client: AsyncClient, fastapi_app: FastAPI) -> None:
    """
    Test the data-info endpoint.

    :param client: client for the app.
    :param fastapi_app: current FastAPI application.
    """
    response = await client.get("/api/cloudcasting/data-info")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert "file_exists" in data
    assert "forecast_steps" in data
    assert "variables" in data
    assert "file_size_mb" in data
    assert "last_modified" in data
    assert "time_range" in data
