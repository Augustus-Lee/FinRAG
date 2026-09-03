#!/bin/bash
# FinRAG 全栈启动脚本（WSL2 / Linux 环境）
# 用法: bash scripts/startup.sh
set -e

echo "============================================================"
echo "FinRAG 全栈启动"
echo "============================================================"

# 进入项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 1. 检查 .env 是否存在
if [ ! -f .env ]; then
    echo "[!] .env 文件不存在，从 .env.example 创建"
    cp .env.example .env
    echo "[!] 请编辑 .env 填入 API key 后重新运行"
    exit 1
fi
echo "[1] .env 已存在"

# 2. 检查 Docker
if ! docker --version >/dev/null 2>&1; then
    echo "[!] Docker 未安装或未启动"
    exit 1
fi
echo "[2] Docker: $(docker --version)"

# 3. 构建并启动全栈
echo "[3] 构建镜像并启动服务（首次约 3-5 分钟）..."
docker compose up -d --build

# 4. 等待 MySQL 健康
echo "[4] 等待 MySQL 就绪..."
for i in $(seq 1 30); do
    status=$(docker compose ps mysql --format json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['Health'])" 2>/dev/null || echo "starting")
    if [ "$status" = "healthy" ]; then
        echo "    MySQL healthy"
        break
    fi
    echo "    waiting... ($i/30)"
    sleep 3
done

# 5. 初始化元数据库
echo "[5] 初始化元数据库（建表）..."
docker compose exec -T api python -m finrag.scripts.init_db

# 6. 导入数据字典
echo "[6] 导入数据字典（3 表 100 字段）..."
docker compose exec -T api python -m finrag.scripts.import_dictionary

# 7. 验证服务健康
echo "[7] 验证服务健康..."
sleep 3
API_HEALTH=$(curl -s http://localhost:8000/api/v1/health 2>/dev/null || echo "failed")
echo "    /health: $API_HEALTH"

# 8. 验证数据字典检索
echo "[8] 验证数据字典检索..."
curl -s -X POST http://localhost:8000/api/v1/dictionary/search \
    -H "Content-Type: application/json" \
    -d '{"question": "客户手机号是多少", "top_k": 5}' 2>/dev/null | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    hits = r.get('hits', [])
    print(f'    hits: {len(hits)} 条')
    for h in hits[:3]:
        print(f'      {h[\"table_name\"]}.{h[\"field_name\"]} - {h[\"comment\"]}')
except:
    print('    (解析失败)')
"

echo ""
echo "============================================================"
echo "全栈启动完成！"
echo "============================================================"
echo "API:     http://localhost:8000"
echo "API 文档: http://localhost:8000/docs"
echo "Qdrant:  http://localhost:6333/dashboard"
echo "MySQL:   localhost:3306 (finrag/finrag)"
echo "Redis:   localhost:6379"
echo ""
echo "下一步三场景联调："
echo "  数据字典: curl -X POST http://localhost:8000/api/v1/dictionary/search -H 'Content-Type: application/json' -d '{\"question\":\"客户手机号\",\"top_k\":5}'"
echo "  NL2SQL:   curl -X POST http://localhost:8000/api/v1/chat/sessions -H 'Content-Type: application/json' -d '{\"mode\":\"nl2sql\"}'"
echo "  知识库:   curl -X POST http://localhost:8000/api/v1/knowledge/documents -F 'file=@README.md'"
echo ""
echo "查看日志: docker compose logs -f api"
echo "停止服务: docker compose down"
echo "============================================================"
