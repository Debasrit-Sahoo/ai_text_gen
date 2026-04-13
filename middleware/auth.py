from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from config import settings

class GatewayAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        token = request.headers.get("X-Internal-Gateway-Auth")

        if token != settings.gateway_secret:
            return JSONResponse(
                status_code=401,
                content={"detail": "unauthorized"}
            )

        return await call_next(request)