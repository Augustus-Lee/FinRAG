"""限流器（M4）：滑动窗口日志，Redis ZSET 优先 + 进程内内存降级。

设计要点：
- 算法：滑动窗口日志（窗口内请求时间戳精确计数，公平直观、无突发透支）。
- 后端：Redis ZSET + Lua 脚本原子执行（清窗/计数/写入一个往返，多实例全局生效）；
  Redis 不可用 → 降级进程内内存窗口（dict + deque + Lock，单机语义）+ 60s 熔断：
  降级发生后熔断期内不再尝试 Redis，避免每请求付连接超时。
- 契约：check() 永不抛异常——限流组件故障不能阻断主链路，降级也要继续限。
"""

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable

from finrag.config import Settings
from finrag.logging import get_logger

logger = get_logger("finrag.rate_limiter")

# Redis 故障后的熔断时长：期间直接走内存，不再重试连接
_CIRCUIT_SECONDS = 60.0
# connect 快失败定位网络问题；socket 稍宽——容器 CPU 高负载偶发抖动不应误判为 Redis 故障
# （实测空闲 p50=0.2ms，全量测试期间可达数百 ms）
_REDIS_CONNECT_TIMEOUT = 0.5
_REDIS_SOCKET_TIMEOUT = 1.0

# 原子滑动窗口：返回值 ≤0 表示允许（取反后为当前计数），>0 为需等待秒数
_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if count < limit then
    redis.call('ZADD', key, now, member)
    redis.call('EXPIRE', key, window)
    return -(count + 1)
end
local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
if #oldest < 2 then
    return 1
end
return math.ceil(tonumber(oldest[2]) + window - now)
"""


class RateLimiter:
    """滑动窗口限流：Redis 原子计数，故障降级内存（带熔断）。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._redis = None  # 懒建：首次 check 时连接
        self._circuit_open_until = 0.0
        self._lock = threading.Lock()
        self._memory: dict[str, deque[float]] = {}
        self._now: Callable[[], float] = time.monotonic  # 可测试性：时间可注入

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def check(self, scope: str, identity: str, limit_per_min: int) -> tuple[bool, int, int]:
        """窗口（60s）内是否放行。

        返回 (allowed, retry_after_seconds, current_count)。永不抛异常：
        Redis 故障降级内存继续限流；两者都异常时放行（限流不能杀死主链路）。
        """
        if limit_per_min <= 0:
            return True, 0, 0
        window = 60.0
        key = f"ratelimit:{scope}:{identity}"
        try:
            return self._redis_check(key, window, limit_per_min)
        except Exception as exc:
            self._open_circuit(exc)
            try:
                return self._memory_check(key, window, limit_per_min)
            except Exception:  # pragma: no cover 内存实现本身无外部依赖
                logger.error("rate_limit_check_failed_allow", error=str(exc)[:200])
                return True, 0, 0

    # ------------------------------------------------------------------
    # Redis 后端
    # ------------------------------------------------------------------
    def _redis_check(self, key: str, window: float, limit: int) -> tuple[bool, int, int]:
        if time.monotonic() < self._circuit_open_until:
            raise ConnectionError("circuit open")  # 熔断期内视为不可用 → 走内存
        client = self._get_redis()
        result = int(
            client.eval(_SLIDING_WINDOW_LUA, 1, key, self._now(), window, limit, uuid.uuid4().hex)
        )
        if result <= 0:
            return True, 0, -result
        return False, max(1, result), limit

    def _get_redis(self):
        if self._redis is None:
            import redis

            self._redis = redis.Redis.from_url(
                self._settings.redis_url,
                socket_connect_timeout=_REDIS_CONNECT_TIMEOUT,
                socket_timeout=_REDIS_SOCKET_TIMEOUT,
                decode_responses=True,
            )
        return self._redis

    def _open_circuit(self, exc: Exception) -> None:
        was_open = time.monotonic() < self._circuit_open_until
        self._circuit_open_until = time.monotonic() + _CIRCUIT_SECONDS
        if not was_open:
            # 只在「关闭→打开」边沿打一条；熔断期内每次请求都降级，重复日志无信息量
            logger.warning(
                "rate_limit_redis_unavailable_fallback_memory",
                error=str(exc)[:200],
                circuit_seconds=_CIRCUIT_SECONDS,
            )

    # ------------------------------------------------------------------
    # 内存后端（降级）：与 Redis 语义一致
    # ------------------------------------------------------------------
    def _memory_check(self, key: str, window: float, limit: int) -> tuple[bool, int, int]:
        now = self._now()
        with self._lock:
            q = self._memory.setdefault(key, deque())
            while q and q[0] <= now - window:
                q.popleft()
            if len(q) < limit:
                q.append(now)
                return True, 0, len(q)
            retry_after = max(1, int(q[0] + window - now) + 1)
            return False, retry_after, len(q)

    # 测试辅助：清空进程内计数（Redis key 需调用方自行清理或换 identity）
    def reset_memory(self) -> None:
        with self._lock:
            self._memory.clear()
