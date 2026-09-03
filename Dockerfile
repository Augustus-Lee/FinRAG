# FinRAG 生产镜像：多阶段构建 + 依赖分层缓存
#
# 默认不安装 embedding/rerank（torch 系），镜像体积小、启动快；
# 需要本地向量模型时构建时加 --build-arg INSTALL_AI_EXTRAS=1
ARG PYTHON_VERSION=3.11-slim
ARG INSTALL_AI_EXTRAS=0

# ---------- 阶段一：构建依赖 ----------
FROM python:${PYTHON_VERSION} AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# PyMuPDF / jieba / sqlglot 等需要编译的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml ./
COPY src ./src

RUN pip install --upgrade pip && pip install -e .

# ---------- 阶段二：运行镜像 ----------
FROM python:${PYTHON_VERSION} AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    FINRAG_DEBUG=false \
    GIT_PYTHON_REFRESH=quiet

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY pyproject.toml ./
COPY src ./src
# 评估资产：评估集数据（M3 seed）+ 知识库语料（README/docs 供评估上传）
COPY data ./data
COPY README.md ./
COPY docs ./docs

# 可选：本地 embedding/rerank 模型依赖（torch ~2GB+，按需开启）
ARG INSTALL_AI_EXTRAS
RUN if [ "$INSTALL_AI_EXTRAS" = "1" ]; then \
        pip install "sentence-transformers>=3.0" "torch>=2.2"; \
    fi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

# 默认启动 API；worker 由 docker-compose 用 command 覆盖
CMD ["uvicorn", "finrag.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
