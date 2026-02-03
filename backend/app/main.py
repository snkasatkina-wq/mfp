"""Основной модуль FastAPI‑бекенда.

Здесь описаны:
- создание приложения FastAPI;
- Pydantic‑схемы запросов/ответов;
- HTTP‑эндпоинты для пользователей, линковки устройств, очереди расходов;
- эндпоинты для мини‑приложения и статики;
- фоновый воркер, который периодически сбрасывает очередь расходов в БД.
"""

from datetime import datetime
from pathlib import Path
import secrets
import time
import asyncio
from typing import Optional, Any

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .database import SessionLocal, init_db
from . import crud
from .models import User

app = FastAPI(title="Expense Tracker Backend", version="0.1.0")


def get_db() -> Session:
    """Выдаёт сессию БД для одного HTTP‑запроса (FastAPI‑dependency)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ===================== Схемы =====================

class LinkCodeResponse(BaseModel):
    """Ответ при создании одноразового кода привязки устройства."""

    device_id: str
    code: str
    expires_in_seconds: int


class UserCreate(BaseModel):
    """Тело запроса для создания/поиска пользователя по Telegram."""

    telegram_user_id: int
    telegram_chat_id: int


class UserOut(BaseModel):
    id: int
    telegram_user_id: int
    telegram_chat_id: int
    email: str | None
    threshold_percent: int

    class Config:
        # Позволяет возвращать напрямую ORM‑объект User из SQLAlchemy
        from_attributes = True


class LinkConfirmRequest(BaseModel):
    """Тело запроса для подтверждения привязки чата по коду."""
    telegram_chat_id: str
    code: str


class LinkConfirmResponse(BaseModel):
    """Ответ на подтверждение привязки чата."""
    ok: bool
    device_id: str


class EnqueueExpenseRequest(BaseModel):
    """Тело запроса от Telegram‑бота для помещения расхода в очередь.

    Дата/время передаётся строкой в формате \"DD.MM.YYYY HH:MM\" или None,
    если нужно использовать текущее время.
    """
    telegram_chat_id: str
    amount_cents: int
    description: str
    category: str
    subcategory: str | None = None
    # строка в формате "DD.MM.YYYY HH:MM" или None (тогда используем текущее время)
    occurred_at: str | None = None


class EnqueueExpenseResponse(BaseModel):
    """Ответ при постановке расхода в очередь на стороне бекенда."""
    ok: bool
    device_id: str
    queued_count: int


class ExpenseCreateRequest(BaseModel):
    """Запрос на создание расхода от бота."""
    telegram_user_id: int
    telegram_chat_id: int
    amount_cents: int
    description: str
    category: str
    subcategory: str | None = None
    occurred_at: str | None = None  # формат "DD.MM.YYYY HH:MM" или None


# ===================== Health =====================

@app.get("/health")
def health():
    """Простой health‑чек, чтобы убедиться, что сервис жив."""
    return {"status": "ok"}


# ===================== /users =====================

@app.post("/users", response_model=UserOut)
def create_or_get_user(
    payload: UserCreate,
    db: Session = Depends(get_db),
) -> UserOut:
    """Создаёт пользователя по Telegram ID или возвращает уже существующего."""
    user = crud.get_or_create_user(
        db=db,
        telegram_user_id=payload.telegram_user_id,
        telegram_chat_id=payload.telegram_chat_id,
    )
    return user

@app.post("/expenses")
def create_expense(
    payload: ExpenseCreateRequest,
    db: Session = Depends(get_db),
):
    """Создаёт расход сразу в БД (без очереди).
    
    Используется ботом для прямого сохранения расходов.
    """
    # Находим или создаём пользователя
    user = crud.get_or_create_user(
        db=db,
        telegram_user_id=payload.telegram_user_id,
        telegram_chat_id=payload.telegram_chat_id,
    )
    
    # Конвертируем дату/время
    if payload.occurred_at:
        occurred_dt = datetime.strptime(payload.occurred_at, "%d.%m.%Y %H:%M")
    else:
        occurred_dt = datetime.utcnow()
    
    # Создаём расход
    expense = crud.create_expense(
        db=db,
        user_id=user.id,
        amount_cents=payload.amount_cents,
        description=payload.description,
        category_name=payload.category,
        subcategory_name=payload.subcategory,
        occurred_at=occurred_dt,
        source="telegram",
    )
    
    return {"ok": True, "expense_id": expense.id}


@app.post("/miniapp/get-user-by-context", response_class=JSONResponse)
def get_user_by_context(payload: dict[str, Any], db: Session = Depends(get_db)) -> dict[str, Any]:
    """Пытается определить пользователя по контексту WebApp, если initData недоступен.
    
    Это fallback метод, когда Telegram не передаёт initData.
    """
    init_data_unsafe = payload.get("initDataUnsafe", {})
    
    # Пробуем получить user из initDataUnsafe
    user_data = init_data_unsafe.get("user")
    if user_data and user_data.get("id"):
        return {
            "telegram_user_id": user_data.get("id"),
            "first_name": user_data.get("first_name", ""),
            "last_name": user_data.get("last_name", ""),
        }
    
    # Если не получилось, возвращаем ошибку
    raise HTTPException(status_code=400, detail="Не удалось определить пользователя из контекста")


@app.post("/miniapp/user-info", response_class=JSONResponse)
def get_user_info_from_init_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Парсит initData от Telegram WebApp и возвращает информацию о пользователе.
    
    Используется как fallback, если фронтенд не может распарсить initData.
    """
    import urllib.parse
    import json
    
    init_data = payload.get("initData", "")
    if not init_data:
        raise HTTPException(status_code=400, detail="initData не предоставлен")
    
    try:
        # Парсим query string
        # initData может быть строкой вида "user=...&auth_date=..."
        params = urllib.parse.parse_qs(init_data)
        user_str = params.get("user", [None])[0]
        
        if not user_str:
            # Пробуем альтернативный формат - может быть уже декодировано
            # или в другом формате
            raise ValueError("Параметр 'user' не найден в initData")
        
        # Декодируем и парсим JSON
        user_data = json.loads(urllib.parse.unquote(user_str))
        
        telegram_user_id = user_data.get("id")
        if not telegram_user_id:
            raise ValueError("telegram_user_id не найден в данных пользователя")
        
        return {
            "telegram_user_id": telegram_user_id,
            "first_name": user_data.get("first_name", ""),
            "last_name": user_data.get("last_name", ""),
        }
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Ошибка парсинга JSON из initData: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка обработки initData: {str(e)}")


@app.get("/miniapp/expenses", response_class=JSONResponse)
def miniapp_expenses(
    telegram_user_id: int,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """Возвращает последние расходы конкретного пользователя для мини‑приложения.

    Дата/время `occurred_at` конвертируется в строку формата \"DD.MM.YYYY HH:MM\"
    для отображения пользователю.
    """
    # Находим пользователя по telegram_user_id
    user = (
        db.query(User)
        .filter(User.telegram_user_id == telegram_user_id)
        .first()
    )
    
    if not user:
        return []
    
    rows = (
        db.execute(
            text("""
            SELECT e.id,
                   e.amount_cents,
                   e.description,
                   e.occurred_at,
                   c.name AS category_name
            FROM expenses e
            LEFT JOIN categories c ON c.id = e.category_id
            WHERE e.user_id = :user_id
            ORDER BY e.occurred_at DESC, e.id DESC
            LIMIT :limit
            """),
            {"user_id": user.id, "limit": limit},
        )
        .mappings()
        .all()
    )
    result: list[dict[str, Any]] = []
    for r in rows:
        item = dict(r)
        occurred = item.get("occurred_at")
        if isinstance(occurred, datetime):
            item["occurred_at"] = occurred.strftime("%d.%m.%Y %H:%M")
        result.append(item)
    return result


# ===================== Линковка устройства =====================

# Временное in‑memory хранилище одноразовых кодов линковки (MVP‑реализация).
# Структура: device_id -> {"code": str, "expires_at": int}
LINK_CODES: dict[str, dict[str, int | str]] = {}


@app.post("/link/code", response_model=LinkCodeResponse)
def create_link_code(device_id: str):
    """Создаёт одноразовый код привязки для указанного device_id.

    Код живёт ограниченное время (TTL) и хранится в памяти процесса.
    """
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
    """Подтверждает привязку чата по одноразовому коду.

    Если код найден и ещё не истёк, создаётся/обновляется запись
    связи `telegram_chat_id -> device_id` в PostgreSQL.
    """
    now = int(time.time())

    matched_device_id: str | None = None
    for device_id, payload in LINK_CODES.items():
        if payload["code"] == req.code and payload["expires_at"] >= now:
            matched_device_id = device_id
            break

    if matched_device_id is None:
        # для простоты MVP — просто ok=false
        return LinkConfirmResponse(ok=False, device_id="")

    # сохраняем привязку в PostgreSQL
    with SessionLocal() as db:
        crud.upsert_chat_device_link(
            db=db,
            telegram_chat_id=int(req.telegram_chat_id),
            device_id=matched_device_id,
        )

    # делаем код одноразовым
    del LINK_CODES[matched_device_id]

    return LinkConfirmResponse(ok=True, device_id=matched_device_id)


# ===================== Очередь расходов =====================

# Очередь во временной памяти процесса: device_id -> [список расходов].
# Время от времени фоновый воркер сбрасывает эти данные в основную БД.
QUEUE: dict[str, list[dict]] = {}


@app.post("/queue/expense", response_model=EnqueueExpenseResponse)
def enqueue_expense(req: EnqueueExpenseRequest):
    """Кладёт расход от бота во временную очередь по связанному device_id.

    Здесь не происходит записи в основную БД, только подготовка данных
    для последующего flush‑а (ручного или автоматического).
    """
    # ищем device_id, привязанный к этому чату
    with SessionLocal() as db:
        device_id = crud.get_device_id_for_chat(
            db=db,
            telegram_chat_id=int(req.telegram_chat_id),
        )
    if not device_id:
        return EnqueueExpenseResponse(ok=False, device_id="", queued_count=0)

    # конвертируем дату/время из пользовательского формата в datetime
    if req.occurred_at:
        # ожидаем формат "DD.MM.YYYY HH:MM"
        occurred_dt = datetime.strptime(req.occurred_at, "%d.%m.%Y %H:%M")
    else:
        occurred_dt = datetime.utcnow()

    item = {
        "amount_cents": req.amount_cents,
        "description": req.description,
        "category": req.category,
        "subcategory": req.subcategory,
        "occurred_at": occurred_dt,
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
    """Ручной сброс очереди расходов для конкретного Telegram‑чата.

    Используется для отладки: берёт все накопленные в QUEUE расходы,
    привязанные к этому чату, и записывает их в таблицу `expenses`.
    """
    with SessionLocal() as tmp_db:
        device_id = crud.get_device_id_for_chat(
            db=tmp_db,
            telegram_chat_id=int(telegram_chat_id),
        )
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
    """Фоновый воркер, который периодически сбрасывает очередь расходов в БД.

    Раз в `interval_seconds` секунд обходится очередь QUEUE, и все накопленные
    расходы для каждого device_id переносятся в таблицу `expenses`.
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

            # находим telegram_chat_id по device_id в БД
            db = SessionLocal()
            try:
                telegram_chat_id = crud.get_chat_id_for_device(db=db, device_id=device_id)
                if telegram_chat_id is None:
                    continue

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
    """Хук FastAPI, вызываемый при старте приложения.

    Здесь мы:
    - создаём таблицы в БД, если их ещё нет (`init_db`);
    - запускаем фоновую задачу `auto_flush_worker`.
    """
    init_db()
    asyncio.create_task(auto_flush_worker(interval_seconds=10))


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/miniapp", response_class=HTMLResponse)
def miniapp_page():
    index_path = FRONTEND_DIR / "index.html"
    with index_path.open("r", encoding="utf-8") as f:
        return f.read()
