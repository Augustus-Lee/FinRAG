"""统一异常体系与 FastAPI 异常处理器。"""

from enum import StrEnum

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ErrorCode(StrEnum):
    """业务错误码（对外契约）。"""

    NOT_FOUND = "NOT_FOUND"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMITED = "RATE_LIMITED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    SQL_REJECTED = "SQL_REJECTED"
    INTERNAL = "INTERNAL"


class FinRAGError(Exception):
    """业务异常基类。"""

    def __init__(self, code: ErrorCode, message: str, detail: dict | None = None) -> None:
        self.code = code
        self.message = message
        self.detail = detail or {}
        super().__init__(message)


class NotFoundError(FinRAGError):
    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__(ErrorCode.NOT_FOUND, message, detail)


class ValidationError(FinRAGError):
    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__(ErrorCode.VALIDATION_ERROR, message, detail)


class UnauthorizedError(FinRAGError):
    def __init__(self, message: str = "未认证或登录已过期", detail: dict | None = None) -> None:
        super().__init__(ErrorCode.UNAUTHORIZED, message, detail)


class ForbiddenError(FinRAGError):
    def __init__(self, message: str = "无权访问", detail: dict | None = None) -> None:
        super().__init__(ErrorCode.FORBIDDEN, message, detail)


class ServiceUnavailableError(FinRAGError):
    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__(ErrorCode.SERVICE_UNAVAILABLE, message, detail)


class SQLRejectedError(FinRAGError):
    """SQL 安全校验未通过。"""

    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__(ErrorCode.SQL_REJECTED, message, detail)


def register_exception_handlers(app: FastAPI) -> None:
    """把 FinRAGError 与未捕获异常统一映射为 JSON 错误响应。"""

    @app.exception_handler(FinRAGError)
    async def handle_finrag_error(request: Request, exc: FinRAGError) -> JSONResponse:
        return JSONResponse(
            status_code=_status_of(exc.code),
            content={"error": {"code": exc.code.value, "message": exc.message, "detail": exc.detail}},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": {"code": ErrorCode.INTERNAL.value, "message": "服务内部错误", "detail": {}}},
        )


def _status_of(code: ErrorCode) -> int:
    mapping = {
        ErrorCode.NOT_FOUND: 404,
        ErrorCode.VALIDATION_ERROR: 422,
        ErrorCode.UNAUTHORIZED: 401,
        ErrorCode.FORBIDDEN: 403,
        ErrorCode.RATE_LIMITED: 429,
        ErrorCode.SERVICE_UNAVAILABLE: 503,
        ErrorCode.EXTERNAL_SERVICE_ERROR: 502,
        ErrorCode.SQL_REJECTED: 422,
        ErrorCode.INTERNAL: 500,
    }
    return mapping.get(code, 500)
