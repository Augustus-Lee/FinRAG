"""管理服务：用户/角色/权限 CRUD（RBAC 管理面，system_manage 权限内调用）。"""

from sqlalchemy.orm import Session

from finrag.logging import get_logger
from finrag.models import SysPermission, SysRole, SysRolePermission, SysUser, SysUserRole
from finrag.services.auth_service import hash_password
from finrag.utils.errors import NotFoundError, ValidationError

logger = get_logger("finrag.admin_service")


class AdminService:
    # ------------------------------------------------------------------
    # 权限
    # ------------------------------------------------------------------
    def list_permissions(self, db: Session) -> list[SysPermission]:
        return db.query(SysPermission).order_by(SysPermission.perm_code).all()

    def _perm_ids_by_codes(self, db: Session, codes: list[str]) -> list[int]:
        """权限码 → id；任一非法码直接 422（防绑定拼写错误导致静默无权限）。"""
        if not codes:
            return []
        rows = db.query(SysPermission).filter(SysPermission.perm_code.in_(codes)).all()
        found = {p.perm_code: p.id for p in rows}
        unknown = [c for c in codes if c not in found]
        if unknown:
            raise ValidationError(f"未知权限码: {unknown}")
        return [found[c] for c in codes]

    # ------------------------------------------------------------------
    # 角色
    # ------------------------------------------------------------------
    def list_roles(self, db: Session) -> list[SysRole]:
        return db.query(SysRole).order_by(SysRole.id).all()

    def create_role(self, db: Session, role_code: str, role_name: str, description: str, perm_codes: list[str]) -> SysRole:
        if db.query(SysRole).filter(SysRole.role_code == role_code).first():
            raise ValidationError(f"角色编码已存在: {role_code}")
        role = SysRole(role_code=role_code, role_name=role_name, description=description)
        db.add(role)
        db.flush()
        self._bind_role_perms(db, role.id, perm_codes)
        db.commit()
        db.refresh(role)
        logger.info("role_created", role_code=role_code, perms=perm_codes)
        return role

    def update_role(self, db: Session, role_id: int, role_name: str | None, description: str | None, perm_codes: list[str] | None) -> SysRole:
        role = db.get(SysRole, role_id)
        if not role:
            raise NotFoundError(f"角色不存在: {role_id}")
        if role_name is not None:
            role.role_name = role_name
        if description is not None:
            role.description = description
        if perm_codes is not None:
            db.query(SysRolePermission).filter(SysRolePermission.role_id == role_id).delete()
            self._bind_role_perms(db, role_id, perm_codes)
        db.commit()
        db.refresh(role)
        logger.info("role_updated", role_id=role_id, perms=perm_codes)
        return role

    def _bind_role_perms(self, db: Session, role_id: int, perm_codes: list[str]) -> None:
        for perm_id in self._perm_ids_by_codes(db, perm_codes):
            db.add(SysRolePermission(role_id=role_id, perm_id=perm_id))

    # ------------------------------------------------------------------
    # 用户
    # ------------------------------------------------------------------
    def list_users(self, db: Session) -> list[SysUser]:
        return db.query(SysUser).order_by(SysUser.id).all()

    def create_user(self, db: Session, username: str, password: str, role_ids: list[int], is_active: bool) -> SysUser:
        if db.query(SysUser).filter(SysUser.username == username).first():
            raise ValidationError(f"用户名已存在: {username}")
        user = SysUser(username=username, password_hash=hash_password(password), is_active=is_active)
        db.add(user)
        db.flush()
        self._bind_user_roles(db, user.id, role_ids)
        db.commit()
        db.refresh(user)
        logger.info("user_created", username=username, roles=len(role_ids))
        return user

    def update_user(self, db: Session, user_id: int, password: str | None, role_ids: list[int] | None, is_active: bool | None) -> SysUser:
        user = db.get(SysUser, user_id)
        if not user:
            raise NotFoundError(f"用户不存在: {user_id}")
        if password is not None:
            user.password_hash = hash_password(password)
        if is_active is not None:
            user.is_active = is_active
        if role_ids is not None:
            db.query(SysUserRole).filter(SysUserRole.user_id == user_id).delete()
            self._bind_user_roles(db, user_id, role_ids)
        db.commit()
        db.refresh(user)
        logger.info("user_updated", user_id=user_id, active=is_active)
        return user

    def _bind_user_roles(self, db: Session, user_id: int, role_ids: list[int]) -> None:
        roles = db.query(SysRole).filter(SysRole.id.in_(role_ids)).all() if role_ids else []
        found = {r.id for r in roles}
        unknown = [i for i in role_ids if i not in found]
        if unknown:
            raise ValidationError(f"未知角色 id: {unknown}")
        for role_id in role_ids:
            db.add(SysUserRole(user_id=user_id, role_id=role_id))
