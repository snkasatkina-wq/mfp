from datetime import datetime
from pathlib import Path
import secrets
import time
import sqlite3
import re  # пока не используется, но можно пригодиться
import asyncio
from typing import Optional
from typing import Any
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .database import SessionLocal
from . import crud
from typing import Any
from fastapi.responses import HTMLResponse,  JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Expense Tracker Backend", version="0.1.0")


def get_db() -> Session:
    """Выдаёт сессию БД для одного запроса."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ===================== Схемы =====================

class LinkCodeResponse(BaseModel):
    device_id: str
    code: str
    expires_in_seconds: int


class UserCreate(BaseModel):
    telegram_user_id: int
    telegram_chat_id: int


class UserOut(BaseModel):
    id: int
    telegram_user_id: int
    telegram_chat_id: int
    email: str | None
    threshold_percent: int

    class Config:
        from_attributes = True  # можно возвращать ORM-объект User


class LinkConfirmRequest(BaseModel):
    telegram_chat_id: str
    code: str


class LinkConfirmResponse(BaseModel):
    ok: bool
    device_id: str


class EnqueueExpenseRequest(BaseModel):
    telegram_chat_id: str
    amount_cents: int
    description: str
    category: str
    subcategory: str | None = None
    occurred_at: str | None = None  # ISO


class EnqueueExpenseResponse(BaseModel):
    ok: bool
    device_id: str
    queued_count: int


# ===================== Health =====================

@app.get("/health")
def health():
    return {"status": "ok"}


# ===================== /users =====================

@app.post("/users", response_model=UserOut)
def create_or_get_user(
    payload: UserCreate,
    db: Session = Depends(get_db),
) -> UserOut:
    """Создаёт пользователя по Telegram ID или возвращает существующего."""
    user = crud.get_or_create_user(
        db=db,
        telegram_user_id=payload.telegram_user_id,
        telegram_chat_id=payload.telegram_chat_id,
    )
    return user

@app.get("/miniapp/expenses", response_class=JSONResponse)
def miniapp_expenses(limit: int = 10, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = (
        db.execute(
            """
            SELECT e.id,
                   e.amount_cents,
                   e.description,
                   e.occurred_at,
                   c.name AS category_name
            FROM expenses e
            LEFT JOIN categories c ON c.id = e.category_id
            ORDER BY e.id DESC
            LIMIT :limit
            """,
            {"limit": limit},
        )
        .mappings()
        .all()
    )
    return [dict(r) for r in rows]


# ===================== Линковка устройства =====================

# Временное хранилище кодов (MVP, потом заменим)
# device_id -> {"code": str, "expires_at": int}
LINK_CODES: dict[str, dict[str, int | str]] = {}

# Временное хранилище привязок (MVP, потом заменим на БД)
# telegram_chat_id -> device_id
LINKS: dict[str, str] = {}

# SQLite-файл для хранения привязок
DB_PATH = Path(__file__).with_name("backend_state.sqlite")


def init_db_sqlite() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS links (
                telegram_chat_id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL
            )
            """
        )
        conn.commit()


def load_links_from_db() -> None:
    LINKS.clear()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT telegram_chat_id, device_id FROM links"
        ).fetchall()
    for chat_id, device_id in rows:
        LINKS[str(chat_id)] = str(device_id)


def save_link_to_db(telegram_chat_id: str, device_id: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO links (telegram_chat_id, device_id)
            VALUES (?, ?)
            ON CONFLICT(telegram_chat_id) DO UPDATE SET device_id=excluded.device_id
            """,
            (telegram_chat_id, device_id),
        )
        conn.commit()


# Инициализация при старте процесса
init_db_sqlite()
load_links_from_db()


@app.post("/link/code", response_model=LinkCodeResponse)
def create_link_code(device_id: str):
    """Создаёт одноразовый код привязки для device_id."""
    code = secrets.token_urlsafe(4)[:6].upper()
    ttl = 10 * 60  # 10 минут
    expires_at = int(time.time()) + ttl

    LINK_CODES[device_id] = {"code": code, "expires_at": expires_at}

    return LinkCodeResponse(
        device_id=device_id,
        code=code,
        expires_in_seconds=ttl,
    )


@app.post("/link/confirm", response_model=LinkConfirmResponse)
def confirm_link(req: LinkConfirmRequest):
    """Подтверждает привязку чата по коду."""
    now = int(time.time())

    matched_device_id: str | None = None
    for device_id, payload in LINK_CODES.items():
        if payload["code"] == req.code and payload["expires_at"] >= now:
            matched_device_id = device_id
            break

    if matched_device_id is None:
        # для простоты MVP — просто ok=false
        return LinkConfirmResponse(ok=False, device_id="")

    # сохраняем привязку
    LINKS[req.telegram_chat_id] = matched_device_id
    save_link_to_db(req.telegram_chat_id, matched_device_id)

    # делаем код одноразовым
    del LINK_CODES[matched_device_id]

    return LinkConfirmResponse(ok=True, device_id=matched_device_id)


# ===================== Очередь расходов =====================

# device_id -> [items]
QUEUE: dict[str, list[dict]] = {}


@app.post("/queue/expense", response_model=EnqueueExpenseResponse)
def enqueue_expense(req: EnqueueExpenseRequest):
    """Кладёт расход от бота во временную очередь по device_id."""
    device_id = LINKS.get(req.telegram_chat_id)
    if not device_id:
        return EnqueueExpenseResponse(ok=False, device_id="", queued_count=0)

    item = {
        "amount_cents": req.amount_cents,
        "description": req.description,
        "category": req.category,
        "subcategory": req.subcategory,
        "occurred_at": req.occurred_at or datetime.utcnow().isoformat(),
        "source": "telegram",
    }

    QUEUE.setdefault(device_id, []).append(item)

    return EnqueueExpenseResponse(
        ok=True,
        device_id=device_id,
        queued_count=len(QUEUE[device_id]),
    )


# ===================== Ручной flush (debug) =====================

@app.post("/debug/flush-queue")
def flush_queue(
    telegram_chat_id: str,
    db: Session = Depends(get_db),
):
    """
    Берёт все расходы из временной очереди для этого чата
    и записывает их в таблицу expenses в PostgreSQL.
    """
    device_id = LINKS.get(telegram_chat_id)
    if not device_id:
        raise HTTPException(
            status_code=400,
            detail="Нет привязки для этого telegram_chat_id",
        )

    items = QUEUE.get(device_id, [])
    if not items:
        return {"ok": True, "written": 0}

    # находим / создаём пользователя по chat_id
    user = crud.get_or_create_user(
        db=db,
        telegram_user_id=int(telegram_chat_id),  # если user_id = chat_id
        telegram_chat_id=int(telegram_chat_id),
    )

    written = 0
    for item in items:
        crud.create_expense(
            db=db,
            user_id=user.id,
            amount_cents=item["amount_cents"],
            description=item["description"],
            category_name=item["category"],
            subcategory_name=item["subcategory"],
            occurred_at=item["occurred_at"],
            source=item["source"],
        )
        written += 1

    # очищаем корзинку для этого device_id
    QUEUE[device_id] = []

    return {"ok": True, "written": written}


# ===================== Авто-flush в фоне =====================

async def auto_flush_worker(interval_seconds: int = 10):
    """
    Периодически перекладывает все расходы из QUEUE в БД.
    """
    from .database import SessionLocal  # локальный импорт против циклов

    while True:
        if not QUEUE:
            await asyncio.sleep(interval_seconds)
            continue

        # копия на момент начала цикла
        snapshot = list(QUEUE.items())  # [(device_id, [items])]

        for device_id, items in snapshot:
            if not items:
                continue

            # находим telegram_chat_id по device_id
            telegram_chat_id: Optional[str] = None
            for chat_id, d_id in LINKS.items():
                if d_id == device_id:
                    telegram_chat_id = chat_id
                    break

            if telegram_chat_id is None:
                continue

            db = SessionLocal()
            try:
                user = crud.get_or_create_user(
                    db=db,
                    telegram_user_id=int(telegram_chat_id),
                    telegram_chat_id=int(telegram_chat_id),
                )

                for item in items:
                    crud.create_expense(
                        db=db,
                        user_id=user.id,
                        amount_cents=item["amount_cents"],
                        description=item["description"],
                        category_name=item["category"],
                        subcategory_name=item["subcategory"],
                        occurred_at=item["occurred_at"],
                        source=item["source"],
                    )

                QUEUE[device_id] = []
            finally:
                db.close()

        await asyncio.sleep(interval_seconds)


@app.on_event("startup")
async def start_auto_flush_worker():
    asyncio.create_task(auto_flush_worker(interval_seconds=10))


#JSON‑эндпоинт+++++++++
@app.get("/miniapp/expenses", response_class=JSONResponse)
def miniapp_expenses(limit: int = 10, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = (
        db.execute(
            """
            SELECT e.id,
                   e.amount_cents,
                   e.description,
                   e.occurred_at,
                   c.name AS category_name
            FROM expenses e
            LEFT JOIN categories c ON c.id = e.category_id
            ORDER BY e.id DESC
            LIMIT :limit
            """,
            {"limit": limit},
        )
        .mappings()
        .all()
    )
    return [dict(r) for r in rows]

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/miniapp", response_class=HTMLResponse)
def miniapp_page():
    index_path = FRONTEND_DIR / "index.html"
    with index_path.open("r", encoding="utf-8") as f:
        return f.read()
