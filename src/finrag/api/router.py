"""路由聚合。"""

from fastapi import APIRouter

from finrag.api.v1 import admin, auth, chat, dictionary, evaluate, health, knowledge

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(admin.router)
api_router.include_router(knowledge.router)
api_router.include_router(chat.router)
api_router.include_router(dictionary.router)
api_router.include_router(evaluate.router)
