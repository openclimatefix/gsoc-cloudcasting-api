from fastapi.routing import APIRouter

from cloudcasting_backend.web.api import cloudcasting

api_router = APIRouter()
api_router.include_router(cloudcasting.router, tags=["cloudcasting"])
