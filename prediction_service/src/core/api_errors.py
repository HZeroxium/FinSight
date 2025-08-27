# app/api_errors.py
import uuid
from datetime import datetime, timezone

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.status import (HTTP_400_BAD_REQUEST,
                              HTTP_422_UNPROCESSABLE_ENTITY)

USE_400_FOR_VALIDATION = True


def _envelope(code: str, message: str, details=None):
    return {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "details": details or [],
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": str(uuid.uuid4()),
    }


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = []
    for e in exc.errors():
        loc = ".".join(str(x) for x in e.get("loc", []) if x != "body")
        details.append(
            {
                "field": loc or "body",
                "type": e.get("type"),
                "message": e.get("msg"),
                "input": e.get("input", None),
                "ctx": e.get("ctx", None),
            }
        )
    status = (
        HTTP_400_BAD_REQUEST
        if USE_400_FOR_VALIDATION
        else HTTP_422_UNPROCESSABLE_ENTITY
    )
    return JSONResponse(
        status_code=status,
        content=_envelope("VALIDATION_ERROR", "Request validation failed", details),
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    # Chuẩn hóa mọi HTTPException (404, 400, 401, 500 do bạn raise trong code)
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope("HTTP_ERROR", str(exc.detail), []),
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    # Fallback cho lỗi chưa bắt
    return JSONResponse(
        status_code=500,
        content=_envelope("INTERNAL_ERROR", "Internal server error"),
    )
