from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import (
    DEPLOYMENT_ENVIRONMENT,
    SERVICE_NAME,
    TELEMETRY_SDK_LANGUAGE,
    Resource,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider

from cloudcasting_backend.services import start_download_thread
from cloudcasting_backend.settings import settings


def setup_opentelemetry(app: FastAPI) -> None:  # pragma: no cover
    """
    Enables opentelemetry instrumentation.

    :param app: current application.
    """
    if not settings.opentelemetry_endpoint:
        return

    tracer_provider = TracerProvider(
        resource=Resource(
            attributes={
                SERVICE_NAME: "cloudcasting_backend",
                TELEMETRY_SDK_LANGUAGE: "python",
                DEPLOYMENT_ENVIRONMENT: settings.environment,
            },
        ),
    )

    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=settings.opentelemetry_endpoint,
                insecure=True,
            ),
        ),
    )

    excluded_endpoints = [
        app.url_path_for("health_check"),
        app.url_path_for("openapi"),
        app.url_path_for("swagger_ui_html"),
        app.url_path_for("swagger_ui_redirect"),
        app.url_path_for("redoc_html"),
    ]

    FastAPIInstrumentor().instrument_app(
        app,
        tracer_provider=tracer_provider,
        excluded_urls=",".join(excluded_endpoints),
    )

    set_tracer_provider(tracer_provider=tracer_provider)


def stop_opentelemetry(app: FastAPI) -> None:  # pragma: no cover
    """
    Disables opentelemetry instrumentation.

    :param app: current application.
    """
    if not settings.opentelemetry_endpoint:
        return

    FastAPIInstrumentor().uninstrument_app(app)


@asynccontextmanager
async def lifespan_setup(
    app: FastAPI,
) -> AsyncGenerator[None, None]:  # pragma: no cover
    """
    Actions to run on application startup.

    This function uses fastAPI app to store data
    in the state, such as db_engine.

    :param app: the fastAPI application.
    :return: function that actually performs actions.
    """

    app.middleware_stack = None
    setup_opentelemetry(app)
    app.middleware_stack = app.build_middleware_stack()

    # Start the scheduled S3 download thread
    # Disabled: Auto-running thread removed, use /trigger-download endpoint instead
    # if settings.environment != "pytest":  # Don't start in test environment
    #     start_download_thread()

    yield
    stop_opentelemetry(app)
