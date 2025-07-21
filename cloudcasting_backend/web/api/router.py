from fastapi.routing import APIRouter

from cloudcasting_backend.web.api import cloudcasting, echo, monitoring

api_router = APIRouter()
api_router.include_router(monitoring.router)
api_router.include_router(echo.router, prefix="/echo", tags=["echo"])
api_router.include_router(cloudcasting.router, tags=["cloudcasting"])
