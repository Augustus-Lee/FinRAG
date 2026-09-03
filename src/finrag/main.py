"""FastAPI 应用入口（应用工厂）。

请求链（外→内）：RequestId（429 也带请求 ID）→ RateLimit（差异化三档，进路由前拒绝）→ 路由。
"""

from contextlib import asynccontextmanager

import jwt as pyjwt
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from finrag.api.router import api_router
from finrag.config import get_settings
from finrag.core.rate_limiter import RateLimiter
from finrag.logging import bind_request_id, get_logger, setup_logging
from finrag.utils.errors import ErrorCode, register_exception_handlers

settings = get_settings()
logger = get_logger("finrag.main")

# 进程级单例：内存计数与 Redis 熔断状态跨请求共享
_rate_limiter = RateLimiter(settings)

# 探活/文档路径豁免限流（探活不能被限流打挂）
_EXEMPT_PATHS = {"/", "/docs", "/openapi.json", "/redoc", f"{settings.api_prefix}/health"}


def _resolve_rule(path: str) -> tuple[str, int] | None:
    """路径 → (scope, 限额)；豁免返回 None。差异化三档见 config 注释。"""
    if path in _EXEMPT_PATHS:
        return None
    prefix = settings.api_prefix
    if path == f"{prefix}/auth/login":
        return "login", settings.rate_limit_login_per_min
    if path.startswith(f"{prefix}/chat"):
        return "chat", settings.rate_limit_chat_per_min
    if path.startswith(prefix):
        return "default", settings.rate_limit_default_per_min
    return None  # 非业务路径（静态资源等）不限


def _identity(request: Request, scope: str) -> str:
    """计数主体：认证档取 JWT sub（仅验签，不查 DB——鉴权仍由依赖层负责），
    无 token/验签失败回退 IP。login 档固定 IP（登录前无身份）。

    IP 取 request.client.host；不信任 X-Forwarded-For（可伪造绕过限流），
    反向代理场景请在代理层传递真实 IP。
    """
    client_ip = request.client.host if request.client else "unknown"
    if scope == "login":
        return f"ip:{client_ip}"
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        try:
            payload = pyjwt.decode(auth[7:], settings.secret_key, algorithms=["HS256"])
            return f"user:{payload['sub']}"
        except Exception:
            pass  # 无效/过期 token → 回退 IP（401 由依赖层裁决，限流只管计数）
    return f"ip:{client_ip}"


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="金融企业知识问答平台：数据字典 / 智能问数(NL2SQL) / 文档知识库（RAG）",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        """差异化限流：进入路由前拒绝，省掉后续全部开销（DB/LLM/检索）。"""
        if settings.rate_limit_enabled:
            rule = _resolve_rule(request.url.path)
            if rule is not None:
                scope, limit = rule
                identity = _identity(request, scope)
                allowed, retry_after, count = _rate_limiter.check(scope, identity, limit)
                if not allowed:
                    logger.warning(
                        "rate_limit_rejected",
                        scope=scope,
                        identity=identity,
                        limit_per_min=limit,
                        retry_after=retry_after,
                    )
                    # 中间件在异常处理器之外，raise 会变 500 → 直接构造 429 响应
                    return JSONResponse(
                        status_code=429,
                        headers={
                            "Retry-After": str(retry_after),
                            "X-RateLimit-Limit": str(limit),
                        },
                        content={
                            "error": {
                                "code": ErrorCode.RATE_LIMITED.value,
                                "message": "请求过于频繁，请稍后再试",
                                "detail": {"scope": scope, "limit_per_min": limit},
                            }
                        },
                    )
                request.state.rate_limit_count = count
        return await call_next(request)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        import uuid

        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
        bind_request_id(request_id)
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            # 原始异常向上传播时不覆盖 response 头
            if response is not None:
                response.headers["X-Request-Id"] = request_id
        return response

    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_prefix)

    @app.get("/", include_in_schema=False)
    def root() -> JSONResponse:
        return JSONResponse({"service": settings.app_name, "docs": "/docs", "health": f"{settings.api_prefix}/health"})

    return app


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.debug)
    logger.info(
        "app_startup",
        app=settings.app_name,
        version="0.1.0",
        rate_limit_enabled=settings.rate_limit_enabled,
    )
    yield
    logger.info("app_shutdown")


app = create_app()
