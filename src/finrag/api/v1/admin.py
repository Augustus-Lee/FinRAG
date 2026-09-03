"""管理接口：用户/角色/权限 CRUD（router 级 system_manage 权限保护）。"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from finrag.api.deps import db_session, require_perms
from finrag.models import SysRole, SysUser
from finrag.schemas.admin import (
    PermissionOut,
    RoleCreate,
    RoleOut,
    RoleUpdate,
    UserCreate,
    UserOut,
    UserUpdate,
)
from finrag.services.admin_service import AdminService

router = APIRouter(
    prefix="/admin", tags=["admin"], dependencies=[Depends(require_perms("system_manage"))]
)

_service = AdminService()


# ---- 权限 ----

@router.get("/permissions", response_model=list[PermissionOut])
def list_permissions(db: Session = Depends(db_session)) -> list[PermissionOut]:
    """权限码清单（前端角色编辑器的选项来源）。"""
    return [PermissionOut.model_validate(p) for p in _service.list_permissions(db)]


# ---- 角色 ----

@router.get("/roles", response_model=list[RoleOut])
def list_roles(db: Session = Depends(db_session)) -> list[RoleOut]:
    return [
        RoleOut(
            id=r.id,
            role_code=r.role_code,
            role_name=r.role_name,
            description=r.description,
            perm_codes=[p.perm_code for p in r.permissions],
        )
        for r in _service.list_roles(db)
    ]


@router.post("/roles", response_model=RoleOut, status_code=201)
def create_role(req: RoleCreate, db: Session = Depends(db_session)) -> RoleOut:
    role = _service.create_role(db, req.role_code, req.role_name, req.description, req.perm_codes)
    return RoleOut(
        id=role.id,
        role_code=role.role_code,
        role_name=role.role_name,
        description=role.description,
        perm_codes=[p.perm_code for p in role.permissions],
    )


@router.patch("/roles/{role_id}", response_model=RoleOut)
def update_role(role_id: int, req: RoleUpdate, db: Session = Depends(db_session)) -> RoleOut:
    role: SysRole = _service.update_role(db, role_id, req.role_name, req.description, req.perm_codes)
    return RoleOut(
        id=role.id,
        role_code=role.role_code,
        role_name=role.role_name,
        description=role.description,
        perm_codes=[p.perm_code for p in role.permissions],
    )


# ---- 用户 ----

@router.get("/users", response_model=list[UserOut])
def list_users(db: Session = Depends(db_session)) -> list[UserOut]:
    return [
        UserOut(
            id=u.id,
            username=u.username,
            is_active=u.is_active,
            roles=[r.role_code for r in u.roles],
        )
        for u in _service.list_users(db)
    ]


@router.post("/users", response_model=UserOut, status_code=201)
def create_user(req: UserCreate, db: Session = Depends(db_session)) -> UserOut:
    user: SysUser = _service.create_user(db, req.username, req.password, req.role_ids, req.is_active)
    return UserOut(
        id=user.id,
        username=user.username,
        is_active=user.is_active,
        roles=[r.role_code for r in user.roles],
    )


@router.patch("/users/{user_id}", response_model=UserOut)
def update_user(user_id: int, req: UserUpdate, db: Session = Depends(db_session)) -> UserOut:
    user: SysUser = _service.update_user(db, user_id, req.password, req.role_ids, req.is_active)
    return UserOut(
        id=user.id,
        username=user.username,
        is_active=user.is_active,
        roles=[r.role_code for r in user.roles],
    )
