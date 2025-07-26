import logging
from importlib import metadata
from pathlib import Path
import os
import tomli

import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import UJSONResponse
from fastapi.staticfiles import StaticFiles
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

from cloudcasting_backend.log import configure_logging
from cloudcasting_backend.settings import settings
from cloudcasting_backend.web.api.router import api_router
from cloudcasting_backend.web.lifespan import lifespan_setup


def get_app() -> FastAPI:
    """
    Get FastAPI application.

    This is the main constructor of an application.

    :return: application.
    """
    configure_logging()
    if settings.sentry_dsn:
        # Enables sentry integration.
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            traces_sample_rate=settings.sentry_sample_rate,
            environment=settings.environment,
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                LoggingIntegration(
                    level=logging.getLevelName(
                        settings.log_level.value,
                    ),
                    event_level=logging.ERROR,
                ),
            ],
        )
    try:
        version = metadata.version("cloudcasting_backend")
    except metadata.PackageNotFoundError:
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                pyproject_data = tomli.load(f)
                version = pyproject_data["tool"]["poetry"]["version"]
        else:
            version = "0.1.0"

    app = FastAPI(
        title="cloudcasting_backend",
        version=version,
        lifespan=lifespan_setup,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        default_response_class=UJSONResponse,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )

    # Main router for the API.
    app.include_router(router=api_router, prefix="/api")

    # Mount static files directory
    static_dir = Path("cloudcasting_backend/static")
    static_dir.mkdir(exist_ok=True, parents=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app
