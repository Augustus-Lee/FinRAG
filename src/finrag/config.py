"""运行时配置（pydantic-settings，支持环境变量 / .env）。"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置。所有项均提供生产级合理默认值，环境变量以 FINRAG_ 前缀覆盖。"""

    model_config = SettingsConfigDict(env_prefix="FINRAG_", env_file=".env", extra="ignore")

    # ---- 应用 ----
    app_name: str = "FinRAG"
    debug: bool = False
    api_prefix: str = "/api/v1"
    secret_key: str = "change-me-in-production"
    cors_origins: list[str] = ["*"]

    # ---- 认证（RBAC / JWT） ----
    auth_token_expire_hours: float = 12.0  # access token 有效期；权限每请求从 DB 重载，撤销即时生效

    # ---- 限流（M4，滑动窗口 + Redis/内存双后端） ----
    rate_limit_enabled: bool = True
    rate_limit_login_per_min: int = 10     # /auth/login 按 IP（防暴力破解 + pbkdf2 CPU 保护）
    rate_limit_chat_per_min: int = 20      # /chat/* 按用户（LLM 资源保护）
    rate_limit_default_per_min: int = 120  # 其余端点按用户

    # ---- 数据库（业务元数据） ----
    db_url: str = "sqlite:///./finrag.db"

    # ---- 业务数据源（智能问数查询的真实业务库，生产环境与元数据库拆分） ----
    business_db_url: str = ""  # 为空则回退到 db_url

    # ---- Redis ----
    redis_url: str = "redis://localhost:6379/0"

    # ---- Qdrant 向量库 ----
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "finrag_docs"
    qdrant_dict_collection: str = "finrag_dict"  # 数据字典专用向量集合（与文档检索隔离）
    vector_size: int = 1024

    # ---- LLM 网关（cloud / local / auto） ----
    llm_mode: str = "auto"  # cloud | local | auto
    llm_timeout: float = 120.0
    llm_max_retries: int = 2
    llm_stream_enabled: bool = True  # knowledge 场景启用内部流式（stream=true 保活，规避长生成 read 超时）
    llm_cloud_base_url: str = "https://api.deepseek.com/v1"
    llm_cloud_api_key: str = ""
    llm_cloud_model: str = "deepseek-chat"
    llm_local_base_url: str = "http://localhost:11434/v1"
    llm_local_model: str = "qwen3.8:27b"  # Qwen3.8-27B dense, Apache-2.0, 262K 上下文，Ollama Q4_K_M ~18GB

    # ---- Embedding（双模式：local / api） ----
    embedding_provider: str = "local"  # local(bge-m3) | api(OpenAI 兼容)
    embedding_model: str = "BAAI/bge-m3"
    embedding_api_base_url: str = ""
    embedding_api_key: str = ""
    embedding_api_model: str = ""  # 云端模型名；为空则回退 embedding_model

    # ---- Rerank（双模式：local / api） ----
    rerank_enabled: bool = True
    rerank_provider: str = "local"  # local(bge cross-encoder) | api(Cohere 兼容 /rerank)
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_api_base_url: str = ""
    rerank_api_key: str = ""
    rerank_api_model: str = ""  # 云端模型名；为空则回退 rerank_model
    rerank_api_format: str = "cohere"  # cohere(扁平 /rerank) | dashscope(嵌套 input/parameters)

    # ---- 混合检索 ----
    rrf_k: int = 60
    rrf_vector_weight: float = 1.0  # RRF 向量路权重（A/B 调参用）
    rrf_bm25_weight: float = 1.0    # RRF BM25 路权重
    retrieve_top_k: int = 20
    rerank_top_k: int = 5

    # ---- 意图路由（mode=auto） ----
    intent_router_enabled: bool = True  # 关闭后 auto 一律按 knowledge 处理
    intent_confidence_threshold: float = 0.6

    # ---- 查询改写（多轮指代消解；意图识别之后执行） ----
    query_rewrite_enabled: bool = True  # 关闭后多轮残缺问题原样透传给 pipeline

    # ---- 切分 ----
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ---- 文件上传 ----
    upload_dir: str = "data/uploads"  # multipart 上传文件落盘目录

    # ---- 智能问数（SQL 安全管控由 MCP Server 端负责） ----
    sql_max_rows: int = Field(default=100, ge=1)  # LLM 生成提示上限（强制管控由 MCP Server 端负责）

    # ---- MCP 执行器 ----
    mcp_enabled: bool = False
    mcp_server_command: str = ""
    mcp_server_url: str = ""  # 外部 DB MCP Server（streamable-http），如 http://mcp-db:8080/mcp
    mcp_api_key: str = ""  # MCP Server 鉴权（可选，Bearer）
    mcp_timeout: float = 30.0

    # ---- Celery ----
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"


@lru_cache
def get_settings() -> Settings:
    """获取全局配置单例（依赖注入用）。"""
    return Settings()
