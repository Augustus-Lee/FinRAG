"""结构化日志：structlog JSON 输出 + 请求级上下文绑定（request_id）。"""

import logging
from contextvars import ContextVar

import structlog

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def bind_request_id(request_id: str) -> None:
    """绑定当前请求的 correlation id（中间件调用）。"""
    request_id_var.set(request_id)


def setup_logging(debug: bool = False) -> None:
    """初始化 structlog 与标准库 logging 的对接。"""
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(level=level, format="%(message)s")

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(ensure_ascii=False),
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "finrag") -> structlog.stdlib.BoundLogger:
    """获取绑定日志器；自动附带当前 request_id。"""
    return structlog.get_logger(name).bind(request_id=request_id_var.get())
