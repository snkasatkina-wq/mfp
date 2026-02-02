from fastapi import FastAPI
from pydantic import BaseModel
import secrets
import time
import sqlite3
from pathlib import Path
import re



app = FastAPI(title="Expense Tracker Backend", version="0.1.0")


# Временное хранилище (позже заменим на нормальную БД на сервере)
# device_id -> {"code": str, "expires_at": int}
LINK_CODES: dict[str, dict[str, int | str]] = {}

class LinkCodeResponse(BaseModel):
    device_id: str
    code: str
    expires_in_seconds: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/link/code", response_model=LinkCodeResponse)
def create_link_code(device_id: str):
    # код на 6 символов (цифры+буквы), достаточно для MVP
    code = secrets.token_urlsafe(4)[:6].upper()
    ttl = 10 * 60  # 10 минут
    expires_at = int(time.time()) + ttl

    LINK_CODES[device_id] = {"code": code, "expires_at": expires_at}

    return LinkCodeResponse(
        device_id=device_id,
        code=code,
        expires_in_seconds=ttl,
    )
# Временное хранилище привязок (позже заменим на БД)
# telegram_chat_id -> device_id
LINKS: dict[str, str] = {}

#Привязка CoDE чтобы не гонять заново
DB_PATH = Path(__file__).with_name("backend_state.sqlite")


def init_db() -> None:
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
        rows = conn.execute("SELECT telegram_chat_id, device_id FROM links").fetchall()
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


# Инициализация при старте
init_db()
load_links_from_db()

QUEUE: dict[str, list[dict]] = {}



class LinkConfirmRequest(BaseModel):
    telegram_chat_id: str
    code: str


class LinkConfirmResponse(BaseModel):
    ok: bool
    device_id: str


@app.post("/link/confirm", response_model=LinkConfirmResponse)
def confirm_link(req: LinkConfirmRequest):
    now = int(time.time())

    # ищем device_id по коду
    matched_device_id = None
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


    # можно удалить код, чтобы он был одноразовым
    del LINK_CODES[matched_device_id]

    return LinkConfirmResponse(ok=True, device_id=matched_device_id)

from datetime import datetime

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

@app.post("/queue/expense", response_model=EnqueueExpenseResponse)
def enqueue_expense(req: EnqueueExpenseRequest):
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

    return EnqueueExpenseResponse(ok=True, device_id=device_id, queued_count=len(QUEUE[device_id]))
