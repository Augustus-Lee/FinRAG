"""RBAC 种子脚本（M4，幂等可重跑）：权限码 / 预设角色 / 管理员账号。

用法（容器内）:
    python -m finrag.scripts.seed_rbac

幂等语义：
- 4 张新表（sys_role/sys_permission/sys_user_role/sys_role_permission）经 create_all 自动建；
- 线上已有 sys_user 表缺 is_active 列时幂等 ALTER（init_db 不 ALTER 旧表，此处补迁移通路）；
- 权限码/角色按 code upsert（重跑更新名称与权限绑定，不重复插入）；
- admin 用户仅在不存在时创建（不覆盖已改过的密码）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy import inspect, text  # noqa: E402

from finrag.db.session import SessionLocal, engine  # noqa: E402
from finrag.logging import get_logger  # noqa: E402
from finrag.models import (  # noqa: E402
    Base,
    SysPermission,
    SysRole,
    SysRolePermission,
    SysUser,
    SysUserRole,
)
from finrag.services.auth_service import hash_password  # noqa: E402

logger = get_logger("finrag.seed_rbac")

# 权限码全集（场景码与 chat mode 一一对应 + 管理类权限）
PERMISSIONS = [
    ("knowledge", "知识库问答", "/chat 的 knowledge 模式"),
    ("nl2sql", "智能问数", "/chat 的 nl2sql 模式"),
    ("dictionary", "数据字典", "/dictionary/* 与 /chat 的 dictionary 模式"),
    ("kb_manage", "知识库管理", "文档上传/删除、分类与文档创建"),
    ("system_manage", "系统管理", "用户/角色管理 API"),
    ("eval_manage", "评估管理", "评估运行与报告查询"),
]

# 预设角色：role_code -> (role_name, description, [perm_codes])
ROLES = {
    "admin": ("管理员", "全部权限", ["knowledge", "nl2sql", "dictionary", "kb_manage", "system_manage", "eval_manage"]),
    "analyst": ("数据分析师", "智能问数 + 数据字典", ["nl2sql", "dictionary"]),
    "kb_operator": ("知识库管理员", "知识库问答与管理", ["knowledge", "kb_manage"]),
}

DEFAULT_ADMIN = ("admin", "admin123")


def _ensure_user_is_active_column() -> None:
    """线上已有 sys_user 表缺 is_active 列时幂等补列（create_all 不会 ALTER 旧表）。"""
    inspector = inspect(engine)
    if "sys_user" not in inspector.get_table_names():
        return  # 表尚不存在，create_all 会带列建表
    columns = {c["name"]: c for c in inspector.get_columns("sys_user")}
    if "is_active" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE sys_user ADD COLUMN is_active TINYINT(1) NOT NULL DEFAULT 1"))
        logger.info("sys_user_altered", added="is_active")
    # 旧 role 列 NOT NULL 无默认值会阻塞新模型 INSERT（模型已不映射该列）→ 放宽为可空
    legacy_role = columns.get("role")
    if legacy_role is not None and not legacy_role.get("nullable", True):
        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE sys_user MODIFY COLUMN role VARCHAR(16) NULL DEFAULT 'user'")
            )
        logger.info("sys_user_altered", relaxed="role(nullable)")


def seed() -> dict[str, int]:
    stats = {"permissions": 0, "roles": 0, "bindings": 0, "users": 0}

    # 1) 建表（含 4 张新表；已有表不动）+ 幂等补列
    Base.metadata.create_all(bind=engine)
    _ensure_user_is_active_column()

    db = SessionLocal()
    try:
        # 2) 权限码 upsert
        perm_ids: dict[str, int] = {}
        for code, name, desc in PERMISSIONS:
            row = db.query(SysPermission).filter(SysPermission.perm_code == code).first()
            if row is None:
                row = SysPermission(perm_code=code, perm_name=name, description=desc)
                db.add(row)
                db.flush()
                stats["permissions"] += 1
            else:
                row.perm_name = name
                row.description = desc
            perm_ids[code] = row.id

        # 3) 角色 upsert + 权限绑定全量替换
        for code, (name, desc, perms) in ROLES.items():
            role = db.query(SysRole).filter(SysRole.role_code == code).first()
            if role is None:
                role = SysRole(role_code=code, role_name=name, description=desc)
                db.add(role)
                db.flush()
                stats["roles"] += 1
            else:
                role.role_name = name
                role.description = desc
            db.query(SysRolePermission).filter(SysRolePermission.role_id == role.id).delete()
            for p in perms:
                db.add(SysRolePermission(role_id=role.id, perm_id=perm_ids[p]))
                stats["bindings"] += 1

        # 4) admin 用户仅缺失时创建（不覆盖已有密码）
        username, password = DEFAULT_ADMIN
        if db.query(SysUser).filter(SysUser.username == username).first() is None:
            admin = db.query(SysRole).filter(SysRole.role_code == "admin").one()
            user = SysUser(username=username, password_hash=hash_password(password))
            db.add(user)
            db.flush()
            db.add(SysUserRole(user_id=user.id, role_id=admin.id))
            stats["users"] += 1
            logger.info("admin_seeded", username=username, password="admin123（请尽快修改）")

        db.commit()
        logger.info("rbac_seed_done", **stats)
        return stats
    finally:
        db.close()


def main() -> None:
    seed()
    print("RBAC 种子完成：6 权限 / 3 角色 / admin(默认 admin/admin123)。")


if __name__ == "__main__":
    main()
