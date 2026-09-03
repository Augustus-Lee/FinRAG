# FinRAG — 面向金融企业的生产级 RAG 平台

> 数据字典 · 智能问数(NL2SQL) · 企业内部知识库 —— 三大金融场景，一套统一引擎。

FinRAG 是一个面向金融企业真实业务场景的 RAG 平台，覆盖「数据字典检索、自然语言转 SQL（智能问数）、企业文档知识库问答」三个场景，自研混合检索、金融规则切分、Schema Linking、两级意图路由、多轮查询改写等核心组件，SQL 安全管控委托 MCP Server 执行边界，并配套标准五表 RBAC 权限体系与 RAGAS 评估闭环。代码结构生产化：分层清晰、191 项测试、Docker Compose 一键部署、延迟导入降重量。

---

## 目录

- [1. 核心特性](#1-核心特性)
- [2. 系统架构](#2-系统架构)
- [3. 目录结构（模块地图）](#3-目录结构模块地图)
- [4. 快速开始](#4-快速开始)
- [5. 配置说明](#5-配置说明)
- [6. API 一览](#6-api-一览)
- [7. 测试与质量](#7-测试与质量)
- [8. 评估闭环](#8-评估闭环)
- [9. 权限体系（RBAC）与限流](#9-权限体系rbac与限流)
- [10. 路线图（里程碑）](#10-路线图里程碑)
- [11. 选型与技术决策](#11-选型与技术决策)

---

## 1. 核心特性

| # | 亮点 | 说明 |
|---|------|------|
| 1 | **自研混合检索 RRF 融合** | 向量召回 + BM25 关键词召回，用 Reciprocal Rank Fusion（k=60）按排序位置融合，不依赖异构分数尺度；字典与文档双索引隔离 |
| 2 | **金融规则切分器** | 按标题层级记录 `section_path`（引用溯源）、**表格整体保留为一块**（`table_meta` 记录表头/行数）、**句级中文切分**（句读切句 + token 回退 overlap，M3 实测忠实度 0.9559 → 0.9877） |
| 3 | **多格式文档解析** | PDF（PyMuPDF，含表格 bbox 排序还原）/ Word（python-docx）/ Excel（openpyxl，多 sheet 表格化）/ MD / TXT，按扩展名白名单分发 |
| 4 | **两级混合意图路由** | `mode=auto`：规则层（正则信号词，零成本零延迟）→ LLM 兜底（few-shot 单 token 分类）→ 默认 knowledge；45 题实测规则层 71% → 两级完整 **95.6%**；会话级 mode 继承 |
| 5 | **多轮查询改写** | 意图识别之后、pipeline 分发之前：短路层（指代/省略信号检测，无历史零成本透传）+ 按意图定制改写目标（nl2sql 补全约束 / dictionary 术语归一 / knowledge 指代消解）+ 防御性输出解析 |
| 6 | **SQL 安全委托执行边界** | SELECT-only / 注入防护 / 条数限制由 MCP Server 端统一管控，RAG 层零校验原样透传（避免双处维护安全规则与重复解析开销）；生成期提示词仍引导 SELECT + LIMIT（≤ max_rows）降低拒绝率，执行异常（含 Server 拒绝）错误回灌自修正 |
| 7 | **MCP 外部取数执行层** | `HttpMcpExecutor`（streamable-http）：tools/list 动态发现工具 → 关键词优先级映射 SQL 槽位；连接类异常（含 `asyncio.CancelledError`）**运行时自动降级** DbDirect 直连，SQL 业务错误透传给 NL2SQL 自修正重试 |
| 8 | **标准五表 RBAC** | sys_user/sys_role/sys_permission + 两组多对多关联；PyJWT HS256 + pbkdf2（600k 迭代随机盐）；**权限每请求从 DB 重载**（停用/撤权即时生效）；场景码与权限码一一对应，auto 路由结果同样受控 |
| 9 | **双模流式 LLM 网关** | `cloud / local / auto` 三模式 + `stream_chat` SSE 流式读取（read 超时按相邻 chunk 间隔计时，规避长生成 ReadTimeout），持久连接池 + 分离超时 |
| 10 | **RAGAS 评估闭环** | 45 题四场景评估集（knowledge/nl2sql/dictionary/intent），含混淆矩阵与 per-scene 准确率；`run_eval` / `compare_eval` CLI 产出 A/B 对比报告（配置快照 diff 可解释） |
| 11 | **VectorStore 抽象层** | 向量库不锁死品牌：`VectorStore` ABC + Qdrant 实现，规模增长可平滑切换 Milvus / ES |

> 核心组件均自研（`src/finrag/core/`），不依赖 LangChain 的 RetrievalQA 黑盒 —— 每一层都可控、可测、可替换。

---

## 2. 系统架构

```
┌────────────────────────────── 应用层 ──────────────────────────────┐
│  FastAPI (REST API) + JWT 鉴权（RBAC 五表，场景码=权限码）           │
│   /knowledge  文档知识库    /chat  统一会话(auto 路由)               │
│   /dictionary 字典检索      /admin  用户角色管理    /evaluate 评估   │
├──────────────────────── 请求处理链 ────────────────────────────────┤
│  ChatService.ask：                                                  │
│   意图路由(mode=auto，两级混合) → 场景权限校验(RBAC)                 │
│   → 查询改写(多轮指代消解) → 会话mode继承 → pipeline 分发            │
├────────────────────────────── 核心引擎层 ──────────────────────────┤
│  RAG：多格式解析 → 句级切分 → 混合检索(RRF) → Rerank → 生成+引用溯源 │
│  NL2SQL：Schema Linking → LLM 生成 → 执行(Server端管控:SELECT/注入/条数)│
│         → 执行异常错误回灌自修正重试；MCP 不可用降级直连              │
│  字典：字典专用双路索引检索 → LLM 口径汇总                           │
├────────────────────────────── 基础设施层 ──────────────────────────┤
│  Qdrant(向量) │ BM25(内存索引) │ MySQL(元数据/业务库) │ Redis+Celery │
│  LLM 网关(cloud/local/auto + SSE流式) │ MCP 执行器                   │
└────────────────────────────────────────────────────────────────────┘
```

三大场景分别对应三种 RAG 形态：

- **数据字典**：元数据即知识 —— 字典检索 + LLM 口径解释；
- **智能问数**：结构化数据问答 —— NL2SQL + 校验 + MCP/直连只读执行 + 结果翻译；
- **知识库问答**：非结构化文档 —— 混合检索 RAG + 引用溯源；
- **auto 模式**：不选场景直接问 —— 意图路由自动分发 + 查询改写补全多轮上下文。

---

## 3. 目录结构（模块地图）

```
src/finrag/
├── config.py               # pydantic-settings 配置（FINRAG_ 前缀）
├── logging.py              # structlog JSON 日志 + request_id
├── container.py            # 依赖容器（组件装配与全局单例）
├── main.py                 # FastAPI 应用工厂
├── core/                   # ★ 自研核心组件
│   ├── hybrid_retriever.py # 混合检索 + RRF 融合
│   ├── bm25.py             # 内存 BM25（Robertson 平滑 IDF + jieba 分词）
│   ├── chunker.py          # 金融规则切分器（标题分层/表格保护/句级切分）
│   ├── document_parser.py  # 多格式解析（PDF表格/Word/Excel/MD/TXT）
│   ├── embedding.py        # Embedding 抽象（bge-m3 本地 / API）
│   ├── vectorstore.py      # VectorStore 抽象 + Qdrant 实现
│   ├── reranker.py         # Rerank 抽象（bge-reranker / API / Noop 降级）
│   ├── llm_gateway.py      # 双模 LLM 网关（chat + stream_chat SSE）
│   ├── schema_linker.py    # Schema Linking（字典 → 候选 schema）
│   ├── mcp_executor.py     # 查询执行抽象（Http MCP / 直连，自动降级；安全由 Server 端管控）
│   ├── intent_router.py    # 两级混合意图路由（规则 + LLM 兜底）
│   ├── query_rewriter.py   # 多轮查询改写（指代消解/省略补全）
│   ├── dictionary_indexer.py # 字段级索引构建（同义词扩展）
│   └── ragas_evaluator.py  # RAGAS 评估器
├── pipelines/              # 三大场景管线
│   ├── rag.py / nl2sql.py / dictionary.py / ingest.py
├── models/                 # SQLAlchemy ORM（知识库/字典/会话/评估/RBAC五表）
├── schemas/                # Pydantic 请求/响应模型（含 admin）
├── services/               # 业务服务层
│   ├── chat_service.py     # 会话入口：路由→权限→改写→分发
│   ├── auth_service.py     # 密码哈希/JWT/认证上下文装配
│   ├── admin_service.py    # 用户/角色/权限 CRUD
│   └── knowledge / dictionary / eval_service
├── api/v1/                 # REST 路由（health/auth/chat/knowledge/dictionary/evaluate/admin）
├── tasks/                  # Celery 异步任务（文档解析入库，同步降级）
├── scripts/                # CLI 工具
│   ├── init_db.py          # 建表（幂等）
│   ├── import_dictionary.py# 字典元数据导入
│   ├── seed_rbac.py        # RBAC 种子（权限/角色/admin，含旧表幂等迁移）
│   ├── seed_eval_cases.py  # 45 题评估集 + 知识库语料
│   ├── run_eval.py         # 评估运行（scene: knowledge/nl2sql/dictionary/intent）
│   └── compare_eval.py     # A/B 报告对比（指标 + 配置快照 diff）
└── utils/errors.py         # 统一异常体系 + HTTP 映射
tests/                      # 191 项测试（核心组件/管线/API/RBAC/意图/改写/限流）
docs/
├── FinRAG-需求分析.md       # 需求规格（PRD v0.2）
├── RAGAS评估体系拆解.md      # 评估方法论
├── eval-reports/            # M3 A/B 对比报告
└── specs/001-project-framework.md  # SDD 开发规格
```

---

## 4. 快速开始

### 4.1 本地开发（无 GPU 可运行）

> 框架默认不加载 torch / sentence-transformers（延迟导入 + 自动降级），冒烟测试零重依赖。

```bash
# 1) 创建虚拟环境并安装（core + test）
python -m venv .venv
.venv/bin/pip install -e ".[test]"          # macOS/Linux
# .venv/Scripts/python -m pip install -e ".[test]"   # Windows

# 2) 配置环境变量（可复制 .env.example 为 .env 后修改）
cp .env.example .env

# 3) 初始化元数据库 + RBAC 种子 + 导入示例数据字典
python -m finrag.scripts.init_db
python -m finrag.scripts.seed_rbac          # 6 权限/3 角色/admin 账号
python -m finrag.scripts.import_dictionary

# 4) 启动 API（可选 --reload）
uvicorn finrag.main:app --host 0.0.0.0 --port 8000

# 5) 验证：登录拿 token（默认 admin/admin123，请尽快修改）
curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'
open http://localhost:8000/docs     # Swagger UI
```

> 提示：`finrag.scripts.*` 需在项目根目录执行；脚本内已处理 `sys.path`。
> 除 `/health` 与 `/auth/login` 外全部端点需 `Authorization: Bearer <token>`。

### 4.2 Docker Compose 全栈（推荐演示）

```bash
docker compose up -d --build
# 初始化（首次）
docker compose exec api python -m finrag.scripts.init_db
docker compose exec api python -m finrag.scripts.seed_rbac
docker compose exec api python -m finrag.scripts.import_dictionary
# 可选：评估集 + 知识库语料
docker compose exec api python -m finrag.scripts.seed_eval_cases --upload-docs
# 查看日志 / 停止
docker compose logs -f api
docker compose down
```

服务：`api:8000`（REST）、`worker`（Celery）、`qdrant:6333`（向量库）、`redis:6379`、`mysql:3306`（元数据 + 业务库）。

### 4.3 典型调用（auto 模式体验意图路由）

```bash
TOKEN=<上一步的 access_token>
# auto：意图路由自动判定 dictionary（问字段口径）
curl -s -X POST http://localhost:8000/api/v1/chat/messages \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"question":"trade_amount 字段的口径是什么","mode":"auto"}'
# auto：判定 nl2sql（问数据值）；多轮带 history 触发查询改写
curl -s -X POST http://localhost:8000/api/v1/chat/messages \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"question":"那6月呢","mode":"nl2sql","history":[{"role":"user","content":"2024年7月股票交易的总金额是多少"},{"role":"assistant","content":"7月总金额为1200万元"}]}'
```

### 4.4 可选：本地 Embedding / Rerank（需 GPU 或耐心）

```bash
pip install -e ".[embedding,rerank]"        # 安装 torch 系依赖
# .env 中 FINRAG_EMBEDDING_PROVIDER=local, FINRAG_RERANK_PROVIDER=local
```

无 GPU / 不想装 torch？两者均支持云端模式（OpenAI 兼容 embedding、多格式 rerank，硅基流动 / Jina / Cohere / 阿里 DashScope 均可）：

```bash
# .env —— 方式一：硅基流动（Cohere 兼容格式，推荐，简单）
FINRAG_EMBEDDING_PROVIDER=api
FINRAG_EMBEDDING_API_BASE_URL=https://api.siliconflow.cn/v1
FINRAG_EMBEDDING_API_KEY=sk-xxx
FINRAG_EMBEDDING_API_MODEL=BAAI/bge-m3

FINRAG_RERANK_PROVIDER=api
FINRAG_RERANK_API_BASE_URL=https://api.siliconflow.cn/v1
FINRAG_RERANK_API_KEY=sk-xxx
FINRAG_RERANK_API_MODEL=BAAI/bge-reranker-v2-m3
FINRAG_RERANK_API_FORMAT=cohere          # 默认值，可省略

# .env —— 方式二：阿里 DashScope（gte-rerank-v2 原生接口）
FINRAG_RERANK_PROVIDER=api
FINRAG_RERANK_API_BASE_URL=https://dashscope.aliyuncs.com/api/v1/services/rerank
FINRAG_RERANK_API_KEY=sk-dashscope-xxx
FINRAG_RERANK_API_MODEL=gte-rerank-v2
FINRAG_RERANK_API_FORMAT=dashscope        # 阿里原生嵌套格式
```

---

## 5. 配置说明

所有配置项以 `FINRAG_` 前缀的环境变量覆盖，默认值见 `.env.example`。核心几组：

| 分组 | 关键变量 | 默认 |
|------|----------|------|
| 元数据库 | `FINRAG_DB_URL` | `sqlite:///./finrag.db`（生产用 MySQL） |
| 业务数据源 | `FINRAG_BUSINESS_DB_URL` | 空（回退 `db_url`） |
| 向量库 | `FINRAG_QDRANT_HOST/PORT/COLLECTION` | `localhost:6333 / finrag_docs` |
| 认证 | `FINRAG_AUTH_TOKEN_EXPIRE_HOURS`、`FINRAG_SECRET_KEY` | `12.0` |
| LLM | `FINRAG_LLM_MODE=cloud\|local\|auto`、`FINRAG_LLM_CLOUD_API_KEY`、`FINRAG_LLM_STREAM_ENABLED` | `auto / true` |
| Embedding | `FINRAG_EMBEDDING_PROVIDER=local\|api`、`FINRAG_EMBEDDING_API_*` | `local`（bge-m3） |
| Rerank | `FINRAG_RERANK_PROVIDER=local\|api`、`FINRAG_RERANK_API_*/FORMAT` | `local` |
| 混合检索 | `FINRAG_RRF_K`、`FINRAG_RETRIEVE_TOP_K`、`FINRAG_RERANK_TOP_K` | `60 / 20 / 5` |
| 意图路由 | `FINRAG_INTENT_ROUTER_ENABLED`、`FINRAG_INTENT_CONFIDENCE_THRESHOLD` | `true / 0.6` |
| 查询改写 | `FINRAG_QUERY_REWRITE_ENABLED` | `true` |
| 限流 | `FINRAG_RATE_LIMIT_ENABLED`、`FINRAG_RATE_LIMIT_LOGIN_PER_MIN`、`FINRAG_RATE_LIMIT_CHAT_PER_MIN`、`FINRAG_RATE_LIMIT_DEFAULT_PER_MIN` | `true / 10 / 20 / 120` |
| 切分 | `FINRAG_CHUNK_SIZE`、`FINRAG_CHUNK_OVERLAP` | `512 / 64` |
| 问数生成提示 | `FINRAG_SQL_MAX_ROWS`（强制管控由 MCP Server 端负责） | `100` |
| MCP | `FINRAG_MCP_ENABLED`、`FINRAG_MCP_SERVER_URL`、`FINRAG_MCP_API_KEY/TIMEOUT` | 关 |
| Celery | `FINRAG_CELERY_BROKER_URL/BACKEND` | Redis |

---

## 6. API 一览

除标注「公开」外全部需 Bearer token；权限列标注所需权限码（any-of）。

| 方法 | 路径 | 权限 | 说明 |
|------|------|------|------|
| GET | `/api/v1/health` | 公开 | 探活 + Qdrant 状态 |
| POST | `/api/v1/auth/login` | 公开 | 登录换 JWT（返回 roles/perms） |
| POST | `/api/v1/chat/sessions` | 登录 | 创建会话（mode 含 auto） |
| POST | `/api/v1/chat/sessions/{id}/messages` | 登录 | 会话内提问（归属校验+场景权限） |
| POST | `/api/v1/chat/messages` | 登录 | 免建会话直接提问 |
| GET | `/api/v1/chat/sessions/{id}/messages` | 登录 | 会话历史（仅本人） |
| POST | `/api/v1/knowledge/categories` | kb_manage | 创建知识库分类 |
| POST | `/api/v1/knowledge/documents` | kb_manage | 创建文档记录 |
| POST | `/api/v1/knowledge/documents/upload` | kb_manage | multipart 上传（异步入库） |
| GET | `/api/v1/knowledge/documents/{id}` | knowledge | 查询文档状态 |
| DELETE | `/api/v1/knowledge/documents/{id}` | kb_manage | 删除文档 |
| POST | `/api/v1/dictionary/search` | dictionary | 字段语义检索 |
| GET | `/api/v1/dictionary/tables` | dictionary | 表清单 |
| GET | `/api/v1/dictionary/tables/{id}/fields` | dictionary | 表内字段清单 |
| POST | `/api/v1/evaluate/run` | eval_manage | 运行评估（含 intent 混淆矩阵） |
| GET | `/api/v1/evaluate/reports` | eval_manage | 评估报告列表 |
| GET | `/api/v1/admin/permissions` | system_manage | 权限码清单 |
| GET/POST | `/api/v1/admin/roles` | system_manage | 角色列表 / 创建（绑权限） |
| PATCH | `/api/v1/admin/roles/{id}` | system_manage | 更新角色（权限全量替换） |
| GET/POST | `/api/v1/admin/users` | system_manage | 用户列表 / 创建（绑角色） |
| PATCH | `/api/v1/admin/users/{id}` | system_manage | 重置密码 / 换角色 / 停用启用 |

---

## 7. 测试与质量

```bash
pytest -v                    # 全部测试（当前 191 个）
ruff check src tests         # 代码规范
```

覆盖范围：金融切分器（标题分层/表格保护/句级切分）、RRF 融合排序、Schema Linking、NL2SQL 管线（错误回灌重试/重试耗尽/SQL 原样透传/工作流图重试）、多格式解析（PDF 表格/Word/Excel）、LLM 网关（流式/重试）、MCP 执行器（发现/映射/降级）、意图路由（规则层/LLM 兜底/噪音容忍）、查询改写（短路/触发/防御解析/接线）、RBAC（哈希/JWT/端点保护矩阵/auto 路由 403/会话隔离/管理 API 全链路）、限流（窗口滑出/身份隔离/Redis 降级/熔断恢复/login 429/豁免/档位隔离，含真实 Redis 主路径回归）、API 冒烟。测试运行在内存 SQLite（conftest 自动灌 RBAC 种子），**不依赖外部服务**。

---

## 8. 评估闭环

```bash
# 容器内：运行四场景评估（45 题种子集）
docker compose exec api python -m finrag.scripts.run_eval --scene knowledge
docker compose exec api python -m finrag.scripts.run_eval --scene dictionary
docker compose exec api python -m finrag.scripts.run_eval --scene nl2sql
docker compose exec api python -m finrag.scripts.run_eval --scene intent   # 路由准确率+混淆矩阵
# A/B 对比（自动 diff 配置快照，回答"参数变了还是系统变了"）
docker compose exec api python -m finrag.scripts.compare_eval --baseline <run_a> --compare <run_b>
```

| 场景 | 指标 | 实测 |
|---|---|---|
| knowledge | 忠实度 / 答案相关性 | 0.9877 / 0.96+（句级切分上线后） |
| dictionary | hit_rate@5 / recall@5 / MRR | 1.0 / 1.0 / 0.95 |
| nl2sql | 执行成功率 / 金标准行集合对照 | 评估集可复跑 |
| intent | 路由准确率（两级混合） | **0.9556**（规则层单独 0.7111） |

报告落库 `eval_report`（detail 含逐题明细、聚合指标、配置快照），M3 A/B 报告见 `docs/eval-reports/`。

---

## 9. 权限体系（RBAC）与限流

### RBAC：标准五表

`sys_user`（is_active 软删）· `sys_role` · `sys_permission` · `sys_user_role` · `sys_role_permission`（权限取角色并集）。

- **权限码**（6 个）：场景码 `knowledge` / `nl2sql` / `dictionary` 与 chat mode 一一对应，另有 `kb_manage` / `system_manage` / `eval_manage`
- **预设角色**：admin（全部）/ analyst（问数+字典）/ kb_operator（知识库问答+管理）
- **认证**：PyJWT HS256（TTL 12h，`FINRAG_AUTH_TOKEN_EXPIRE_HOURS`）；密码 pbkdf2-sha256 600k 迭代随机盐恒定时间比较；登录失败不区分原因（防用户枚举）
- **即时生效**：token 只携带身份，权限每请求从 DB 重载——停用账号/撤销角色立即 401/403
- **chat 场景校验**：effective mode（显式 / auto 路由结果 / 会话继承）统一校验；路由到无权限场景 403 不静默降级；会话归属隔离（他人会话 403）

```bash
# 种子（幂等可重跑，含线上旧表迁移：补 is_active 列、放宽废弃 role 列）
docker compose exec api python -m finrag.scripts.seed_rbac
```

### 限流：差异化三档滑动窗口（`core/rate_limiter.py`）

中间件层实现（进路由前拒绝，省掉 DB/LLM/检索全部开销），Redis ZSET + Lua 原子滑动窗口，**Redis 故障自动降级进程内内存 + 60s 熔断**（降级不失效限流、不拖慢主链路）：

| 档位 | 路径 | 计数主体 | 默认限额 | 保护目标 |
|---|---|---|---|---|
| 豁免 | `/health`、`/docs` 等 | — | 不限 | 探活不被限流打挂 |
| login | `/auth/login` | IP | 10/min | 防暴力破解 + pbkdf2 CPU 保护 |
| chat | `/chat/*` | 用户（JWT sub，回退 IP） | 20/min | LLM 资源保护 |
| default | 其余 API | 用户（回退 IP） | 120/min | 轻量端点防滥用 |

超限返回 `429 + Retry-After + X-RateLimit-Limit`；不信任 X-Forwarded-For（可伪造），反代场景在代理层传真实 IP。

---

## 10. 路线图（里程碑）

| 里程碑 | 内容 | 状态 |
|--------|------|------|
| **M1** 框架骨架 | 分层架构、配置/日志/异常、ORM、核心组件、API、容器化 | ✅ |
| **M2** 三场景闭环 | 知识库入库+问答、NL2SQL 端到端、字典问答跑通真实数据 | ✅ |
| **M3** 评估体系 | RAGAS 指标、45 题评估集、A/B 回归报告、句级切分优化 | ✅ |
| **M4** 生产加固 | **五表 RBAC 权限、差异化三档限流、监控 |  ✅ |
| **M4+** 智能化 | 意图路由 + 查询改写、MCP 外部取数、前端 |  ✅ |

---

## 11. 选型与技术决策

**Q1：为什么同时使用 LangChain 和 LangGraph？**
按**状态复杂度**分工：固定线性管线（解析→切分→检索→生成）用 LangChain 组件更轻；需要**循环/分支状态**的场景（场景路由、NL2SQL 失败重试）用 LangGraph StateGraph 表达。把图用在刀刃上，避免"为了用而用"。

**Q2：为什么选择 Qdrant 作为向量库？**
生产级与否取决于**策略与抽象**，不是品牌：① 自研 RRF 融合 + Rerank 双层精排，不裸用向量检索；② `VectorStore` 抽象层，规模增长可切 Milvus/ES；③ 字典与文档双集合隔离。选 Qdrant 是因为单机性能与运维成本最优，且通过抽象层保留了演进空间。

**Q3：意图路由为什么放规则层在前、LLM 兜底在后？**
成本结构决定顺序：规则层正则零成本零延迟，处理明确信号（聚合词→nl2sql、标识符+元信息词→dictionary）；仅混淆带触发一次 LLM 单 token 分类（temperature=0）。实测规则层单独 71%，两级混合 95.6%——**大多数请求不付 LLM 成本**。查询改写在路由之后，因为改写目标依赖意图（问数补约束 / 字典归一术语 / 知识库消指代）。

**Q4：多轮对话怎么处理"那6月呢"这类残缺问题？**
查询改写器（意图识别后、pipeline 分发前）：短路层用指代/省略正则检测（偏召回设计：漏检=链路损坏，误检=多一次 LLM 调用），命中才调 LLM 改写为 self-contained question，三个 pipeline 无需各自感知历史；输出做防御性清洗（前缀噪音/拒答词/膨胀检查），失败回退原文不阻断问答。E2E 实测：改写前 SQL 瞎猜表字段，改写后完整继承上文约束。

**Q5：智能问数如何保障 SQL 安全？**
SQL 安全校验（SELECT-only / 注入防护 / 条数限制）**委托给 MCP Server 端统一管控**，RAG 层不再做本地校验：安全规则收敛到单一执行边界，避免双处维护与每次问数多付一次 AST 解析；LLM 生成 SQL 后原样透传，提示词仍引导 SELECT + LIMIT（≤ max_rows）以降低 Server 拒绝率，执行异常（含 Server 拒绝）错误回灌自修正重试。**RAG 后端 = MCP Client**（streamable-http，tools/list 动态发现 + 关键词映射 SQL 槽位）；连接类异常（含 SDK 任务组内表现为 `CancelledError` 的超时）触发**运行时自动降级**直连业务库（开发/受信环境），SQL 业务错误则透传给自修正——**降级不吞业务错误**。生产环境应启用 MCP，由 Server 端管控兜底。

**Q6：RBAC 为什么"每请求从 DB 重载权限"而不是信任 token claims？**
权限撤销的即时性 vs 一条 join 查询的成本。token 只带身份（sub/iat/exp），12h 有效期内角色/权限变更、账号停用都能即时生效（E2E 验证：停用后原 token 立即 401）。若把权限塞进 token，撤销需等 token 过期或引入黑名单，复杂度更高。

**Q7：密码哈希为什么用 pbkdf2 而不是 bcrypt/passlib？**
passlib 已停止维护（2024 起无更新，与新版 bcrypt 兼容性问题频发）；pbkdf2_hmac 是 stdlib、NIST 认可、OWASP 推荐 600k 迭代（SHA-256）。每用户随机盐 + `hmac.compare_digest` 恒定时间比较防时序侧信道。token 用 PyJWT 是行业标准选择，两者不冲突。

**Q8：限流为什么分三档？Redis 挂了限流怎么办？**
保护对象成本不同决定差异化：login 是暴力破解入口且 pbkdf2 600k 迭代 CPU 密集（10/min/IP），chat 每次调用真实花 LLM token 成本（20/min/用户），其余端点轻量（120/min）。算法用滑动窗口日志（公平无突发透支），Redis ZSET + Lua 脚本原子执行（清窗/计数/写入一个往返，多实例全局生效）。Redis 故障时降级进程内内存窗口 + 60s 熔断——**降级不失效限流**（安全属性不能因基础设施故障而丢失），熔断避免每请求付连接超时。

---

## License

MIT
