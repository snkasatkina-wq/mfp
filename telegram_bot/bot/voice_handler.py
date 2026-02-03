# ======================================================================
# FILE: backend/run_dev.py
# ======================================================================
"""
Dev-runner для бэкенда.

Запуск:
    python run_dev.py

Что делает:
- поднимает Uvicorn
- включает autoreload для разработки
"""

import uvicorn

if __name__ == "__main__":
    # LAST CHANGES:
    # - файл был в одну строку → привели к нормальному виду
    # - логика не менялась
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)


# ======================================================================
# FILE: backend/app/database.py
# ======================================================================
"""
Настройка подключения к БД.

Почему так:
- секреты (пароли) не храним в репозитории
- URL БД берём из переменной окружения DATABASE_URL
- можно быстро переключаться между dev/prod

ENV:
- DATABASE_URL (например: postgresql://user:pass@localhost:5432/dbname)
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base


def _make_engine():
    """
    Создаёт SQLAlchemy engine.

    Поддерживаем:
    - PostgreSQL
    - SQLite (для быстрого локального MVP)

    LAST CHANGES:
    - убрали хардкод DATABASE_URL с паролем из репозитория,
      потому что это небезопасно и ломает деплой/переключение окружений.
    """
    database_url = os.getenv("DATABASE_URL")

    # Если DATABASE_URL не задан — используем sqlite локально (простое MVP).
    if not database_url:
        database_url = "sqlite:///./mvp_costs.db"

    # Для SQLite нужно добавить connect_args
    if database_url.startswith("sqlite"):
        return create_engine(
            database_url,
            echo=False,
            connect_args={"check_same_thread": False},
        )

    return create_engine(database_url, echo=False)


engine = _make_engine()

# Сессии для запросов (одна сессия на запрос)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def init_db() -> None:
    """
    Создаёт таблицы по моделям, если их ещё нет.

    Важно:
    - это “простая” инициализация для MVP
    - если пойдёшь в прод и начнутся изменения схемы — лучше миграции (Alembic)
    """
    Base.metadata.create_all(bind=engine)


# ======================================================================
# FILE: backend/app/models.py
# ======================================================================
"""
SQLAlchemy модели.

LAST CHANGES:
1) User.telegram_chat_id:
   - было: unique=True, nullable=False
   - стало: nullable=True, НЕ unique
   Почему:
   - главный идентификатор пользователя — telegram_user_id
   - chat_id нужен как “куда слать уведомления”, он может меняться и не должен быть уникальным ключом личности

2) CategoryLimit:
   - было: category_id unique=True
   - стало: уникальность по (category_id, period_type)
   Почему:
   - по ТЗ лимиты бывают на разные периоды (day/week/month), значит на одну категорию может быть несколько лимитов.
"""

from typing import Optional
from datetime import datetime

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import (
    Integer,
    String,
    TIMESTAMP,
    ForeignKey,
    BigInteger,
    Boolean,
    func,
    UniqueConstraint,
)


class Base(DeclarativeBase):
    """Базовый класс для всех моделей."""
    pass


class User(Base):
    """
    Пользователь Telegram.

    telegram_user_id — главный идентификатор.
    telegram_chat_id — куда отправлять уведомления (может обновляться).
    """
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    telegram_user_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        unique=True,
        index=True,
    )

    telegram_chat_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        nullable=True,
        index=True,
    )

    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    threshold_percent: Mapped[int] = mapped_column(Integer, nullable=False, default=80)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False),
        server_default=func.now(),
        nullable=False,
    )


class Category(Base):
    """Категория расходов (пользовательская)."""
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False),
        server_default=func.now(),
        nullable=False,
    )

    # Одна категория с таким именем на пользователя.
    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_categories_user_name"),
    )


class Subcategory(Base):
    """Подкатегория (в рамках категории)."""
    __tablename__ = "subcategories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    category_id: Mapped[int] = mapped_column(
        ForeignKey("categories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("category_id", "name", name="uq_subcategories_category_name"),
    )


class Expense(Base):
    """Расход (операция)."""
    __tablename__ = "expenses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    amount_cents: Mapped[int] = mapped_column(Integer, nullable=False)

    occurred_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False),
        nullable=False,
        index=True,
    )

    description: Mapped[str] = mapped_column(String, nullable=False)

    category_id: Mapped[int] = mapped_column(
        ForeignKey("categories.id"),
        nullable=False,
        index=True,
    )

    subcategory_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("subcategories.id"),
        nullable=True,
        index=True,
    )

    source: Mapped[str] = mapped_column(String(50), nullable=False)

    receipt_group_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False),
        server_default=func.now(),
        nullable=False,
    )


class CategoryLimit(Base):
    """
    Лимит по категории.

    period_type: day/week/month (или любые ваши значения)
    """
    __tablename__ = "category_limits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    category_id: Mapped[int] = mapped_column(
        ForeignKey("categories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    period_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    limit_amount_cents: Mapped[int] = mapped_column(Integer, nullable=False)

    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        # LAST CHANGES: составная уникальность вместо unique=True на category_id
        UniqueConstraint("category_id", "period_type", name="uq_limits_category_period"),
    )


# ======================================================================
# FILE: backend/app/crud.py
# ======================================================================
"""
CRUD-слой (работа с БД).

LAST CHANGES:
- get_or_create_user теперь ищет по telegram_user_id, а telegram_chat_id обновляет,
  потому что chat_id — это “адрес доставки уведомлений”, а не паспорт пользователя.
- create_expense теперь:
  - нормально обрабатывает occurred_at (str|datetime|None)
  - создаёт категорию при необходимости
  - создаёт подкатегорию при необходимости
  - пишет subcategory_id в Expense
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence

from sqlalchemy.orm import Session

from .models import User, Category, Subcategory, Expense


# -------------------------
# Пользователи
# -------------------------

def get_or_create_user(db: Session, telegram_user_id: int, telegram_chat_id: Optional[int]) -> User:
    """
    Возвращает пользователя по telegram_user_id или создаёт нового.

    Если chat_id изменился — обновляем (чтобы уведомления уходили в актуальный чат).

    LAST CHANGES:
    - раньше искали по (telegram_user_id AND telegram_chat_id) => это ломало обновления chat_id
    - теперь ищем только по telegram_user_id => правильная идентификация
    """
    user = db.query(User).filter(User.telegram_user_id == telegram_user_id).first()
    if user is not None:
        # Обновляем chat_id при необходимости
        if telegram_chat_id is not None and user.telegram_chat_id != telegram_chat_id:
            user.telegram_chat_id = telegram_chat_id
            db.add(user)
            db.commit()
            db.refresh(user)
        return user

    user = User(
        telegram_user_id=telegram_user_id,
        telegram_chat_id=telegram_chat_id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# -------------------------
# Категории / Подкатегории
# -------------------------

def get_or_create_category(db: Session, user_id: int, name: str) -> Category:
    """Находит или создаёт категорию по имени для пользователя."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Category name is empty")

    category = (
        db.query(Category)
        .filter(Category.user_id == user_id, Category.name == name)
        .first()
    )
    if category is not None:
        return category

    category = Category(user_id=user_id, name=name)
    db.add(category)
    db.commit()
    db.refresh(category)
    return category


def get_or_create_subcategory(db: Session, category_id: int, name: str) -> Subcategory:
    """Находит или создаёт подкатегорию по имени внутри категории."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Subcategory name is empty")

    sub = (
        db.query(Subcategory)
        .filter(Subcategory.category_id == category_id, Subcategory.name == name)
        .first()
    )
    if sub is not None:
        return sub

    sub = Subcategory(category_id=category_id, name=name)
    db.add(sub)
    db.commit()
    db.refresh(sub)
    return sub


def get_categories(db: Session, user_id: int) -> Sequence[Category]:
    """Возвращает все категории пользователя."""
    return (
        db.query(Category)
        .filter(Category.user_id == user_id)
        .order_by(Category.name.asc())
        .all()
    )


# -------------------------
# Расходы
# -------------------------

def _parse_occurred_at(occurred_at: Optional[str | datetime]) -> datetime:
    """
    Приводит occurred_at к datetime.

    Принимаем:
    - None -> utcnow()
    - datetime -> как есть
    - ISO string -> datetime.fromisoformat

    Важно:
    - timezone-aware здесь не поддерживаем (модель TIMESTAMP без timezone).
    """
    if occurred_at is None:
        return datetime.utcnow()

    if isinstance(occurred_at, datetime):
        return occurred_at

    s = str(occurred_at).strip()
    if not s:
        return datetime.utcnow()

    # ISO: "2026-02-03T12:34:56" или "2026-02-03 12:34:56"
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except ValueError:
        # Если прилетело что-то странное — не падаем, берём текущее
        return datetime.utcnow()


def create_expense(
    db: Session,
    user_id: int,
    amount_cents: int,
    description: str,
    category_name: str,
    subcategory_name: Optional[str],
    occurred_at: Optional[str | datetime],
    source: str,
) -> Expense:
    """
    Создаёт расход для пользователя.

    Что делает:
    - гарантирует наличие категории
    - при subcategory_name создаёт/находит подкатегорию
    - приводит occurred_at к datetime
    """
    category = get_or_create_category(db=db, user_id=user_id, name=category_name)

    subcategory_id: Optional[int] = None
    if subcategory_name:
        sub = get_or_create_subcategory(db=db, category_id=category.id, name=subcategory_name)
        subcategory_id = sub.id

    expense = Expense(
        user_id=user_id,
        category_id=category.id,
        subcategory_id=subcategory_id,
        amount_cents=int(amount_cents),
        description=(description or "").strip(),
        occurred_at=_parse_occurred_at(occurred_at),
        source=(source or "unknown").strip(),
    )
    db.add(expense)
    db.commit()
    db.refresh(expense)
    return expense


# ======================================================================
# FILE: backend/app/main.py
# ======================================================================
"""
FastAPI приложение бэкенда.

LAST CHANGES (главное):
- Убрали “линковку устройства” (/link/code, /link/confirm) и очередь (/queue/expense),
  потому что в Telegram Mini App нет отдельного “телефона/устройства”. Есть пользователь Telegram.
- Добавили:
  - POST /users/upsert (и оставили POST /users как алиас)
  - POST /expenses — бот пишет расход сразу в БД
- Починили:
  - дублированный эндпоинт /miniapp/expenses (теперь один)
  - запись occurred_at как datetime (через crud.create_expense)
- Добавили CORS (управляется переменной окружения ALLOWED_ORIGINS)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .database import SessionLocal, init_db
from . import crud

app = FastAPI(title="Expense Tracker Backend", version="0.2.0")


# -------------------------
# DB dependency
# -------------------------

def get_db() -> Session:
    """
    Dependency: выдаёт SQLAlchemy-сессию на один запрос.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------------
# Pydantic схемы
# -------------------------

class UserUpsert(BaseModel):
    """
    Вход для создания/обновления пользователя.

    telegram_user_id — паспорт пользователя Telegram
    telegram_chat_id — куда слать уведомления (может меняться)
    """
    telegram_user_id: int
    telegram_chat_id: Optional[int] = None


class UserOut(BaseModel):
    """
    Выход пользователя наружу.
    """
    id: int
    telegram_user_id: int
    telegram_chat_id: Optional[int] = None
    email: Optional[str] = None
    threshold_percent: int = 80

    class Config:
        from_attributes = True


class ExpenseCreate(BaseModel):
    """
    Вход для создания расхода (для бота и mini app).

    occurred_at — ISO строка или None (тогда берём текущее время).
    """
    telegram_user_id: int
    telegram_chat_id: Optional[int] = None

    amount_cents: int = Field(ge=1)
    description: str
    category: str
    subcategory: Optional[str] = None
    occurred_at: Optional[str] = None


class ExpenseOut(BaseModel):
    """
    Минимальный ответ после создания расхода.
    """
    ok: bool = True
    expense_id: int


# -------------------------
# Startup
# -------------------------

@app.on_event("startup")
async def on_startup() -> None:
    """
    Инициализация приложения.

    Что делаем:
    - создаём таблицы, если их нет (MVP)
    """
    init_db()


# -------------------------
# CORS
# -------------------------

def _setup_cors() -> None:
    """
    CORS для mini app (если фронт будет не на том же домене, что бэкенд).

    ENV:
      ALLOWED_ORIGINS="*" или "https://example.com,https://another.com"

    LAST CHANGES:
    - добавили, чтобы mini app нормально ходила в API из WebView/браузера.
    """
    origins_raw = os.getenv("ALLOWED_ORIGINS", "*").strip()
    if origins_raw == "*":
        allow_origins = ["*"]
    else:
        allow_origins = [o.strip() for o in origins_raw.split(",") if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


_setup_cors()


# -------------------------
# Health
# -------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Простой healthcheck."""
    return {"status": "ok"}


# -------------------------
# Users
# -------------------------

@app.post("/users/upsert", response_model=UserOut)
def users_upsert(payload: UserUpsert, db: Session = Depends(get_db)) -> UserOut:
    """
    Создаёт пользователя по telegram_user_id или обновляет chat_id.

    Это дергает бот на /start, чтобы:
    - бэкенд точно знал, куда слать уведомления
    """
    user = crud.get_or_create_user(
        db=db,
        telegram_user_id=payload.telegram_user_id,
        telegram_chat_id=payload.telegram_chat_id,
    )
    return user


@app.post("/users", response_model=UserOut)
def users_alias(payload: UserUpsert, db: Session = Depends(get_db)) -> UserOut:
    """
    Алиас на /users/upsert (оставили для совместимости).

    LAST CHANGES:
    - раньше /users мог создавать дубли при изменении chat_id.
    - теперь это просто алиас на корректную логику upsert.
    """
    return users_upsert(payload, db)


# -------------------------
# Expenses (API)
# -------------------------

@app.post("/expenses", response_model=ExpenseOut)
def create_expense(payload: ExpenseCreate, db: Session = Depends(get_db)) -> ExpenseOut:
    """
    Создаёт расход сразу в БД.

    Важно:
    - Никакой очереди и “привязки телефона” больше нет.
    - Пользователь определяется по telegram_user_id.
    """
    user = crud.get_or_create_user(
        db=db,
        telegram_user_id=payload.telegram_user_id,
        telegram_chat_id=payload.telegram_chat_id,
    )

    expense = crud.create_expense(
        db=db,
        user_id=user.id,
        amount_cents=payload.amount_cents,
        description=payload.description,
        category_name=payload.category,
        subcategory_name=payload.subcategory,
        occurred_at=payload.occurred_at,
        source="telegram",
    )

    return ExpenseOut(ok=True, expense_id=expense.id)


# -------------------------
# Mini App endpoints
# -------------------------

@app.get("/miniapp/expenses", response_class=JSONResponse)
def miniapp_expenses(limit: int = 10, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    """
    Возвращает последние расходы (для мини-приложения).

    LAST CHANGES:
    - раньше эндпоинт был задублирован дважды => оставили один.
    """
    rows = (
        db.execute(
            """
            SELECT
                e.id,
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


# -------------------------
# Static mini app (frontend)
# -------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

# Монтируем статику, если папка есть
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/miniapp", response_class=HTMLResponse)
def miniapp_page() -> str:
    """
    Отдаёт HTML мини-приложения (если оно лежит в /frontend/index.html).

    Если index.html нет — даём понятную ошибку.
    """
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"frontend/index.html not found at: {index_path}",
        )

    return index_path.read_text(encoding="utf-8")


# ======================================================================
# FILE: telegram_bot/bot/expense_text_handler.py
# ======================================================================
"""
Обработка текстовых расходов вида:
  сумма // описание // категория // (подкатегория опционально)

LAST CHANGES:
- /queue/expense -> /expenses
- добавили telegram_user_id в payload
- убрали сообщение “сделай /link”, потому что больше нет привязки телефона
"""

from aiogram.types import Message


def parse_expense_text(text: str) -> dict:
    """
    Парсит формат: сумма // описание // категория // (подкатегория)

    Возвращает dict с amount (float), amount_cents (int), description, category, subcategory.
    """
    parts = [p.strip() for p in (text or "").split("//")]
    parts = [p for p in parts if p != ""]

    if len(parts) < 3 or len(parts) > 4:
        raise ValueError("Формат: сумма // описание // категория // (подкатегория опционально)")

    amount_raw = parts[0].replace(" ", "").replace(",", ".")
    amount = float(amount_raw)
    amount_cents = int(round(amount * 100))

    description = parts[1]
    category = parts[2]
    subcategory = parts[3] if len(parts) == 4 else ""

    return {
        "amount": amount,
        "amount_cents": amount_cents,
        "description": description,
        "category": category,
        "subcategory": subcategory,
    }


async def handle_text_expense(*, m: Message, client, backend_url: str) -> None:
    """
    Главный хэндлер текста.

    Что делает:
    - игнорирует команды (/start и т.п.)
    - парсит текст
    - отправляет на backend /expenses

    backend_url должен быть без trailing slash или мы его подчистим.
    """
    text = (m.text or "").strip()
    if not text or text.startswith("/"):
        return

    try:
        data = parse_expense_text(text)
    except ValueError as e:
        await m.answer(str(e))
        return

    payload = {
        # LAST CHANGES: добавили user_id и chat_id
        "telegram_user_id": m.from_user.id,
        "telegram_chat_id": m.chat.id,
        "amount_cents": data["amount_cents"],
        "description": data["description"],
        "category": data["category"],
        "subcategory": data["subcategory"] if data["subcategory"] else None,
        "occurred_at": None,
    }

    try:
        r = await client.post(f"{backend_url.rstrip('/')}/expenses", json=payload)
        r.raise_for_status()
        resp = r.json()
    except Exception as e:
        await m.answer(f"Ошибка связи с сервером: {e}")
        return

    if not resp.get("ok", True):
        await m.answer(resp.get("error", "Не получилось сохранить расход."))
        return

    await m.answer(
        f"внесен расход за ({data['description']}), на сумму ({data['amount']}), "
        f"категория ({data['category']}), подкатегория ({data['subcategory']})"
    )


# ======================================================================
# FILE: telegram_bot/bot/voice_handler.py
# ======================================================================
"""
Обработка голосовых расходов.

Пайплайн:
1) скачать голосовое
2) Whisper (OpenAI) -> текст
3) парсинг "сумма + описание + категория(последнее слово)"
4) POST на backend /expenses

LAST CHANGES:
- /queue/expense -> /expenses
- добавили telegram_user_id
- убрали “сделай /link” (привязки устройства больше нет)
"""

import asyncio
import os
import re
import tempfile

import httpx
from aiogram.types import Message


PROMPT_RU = (
    "Распознавай короткие голосовые заметки о расходах денег. "
    "Выводи БЕЗ запятых, одной строкой, строго в виде: "
    "<СУММА_ЦИФРАМИ> <ОПИСАНИЕ> <КАТЕГОРИЯ>. "
    "Примеры: '300 яйца продукты', '2000 бензин транспорт', '1500 кофе развлечения'. "
    "Сумму всегда пиши цифрами (например, '2000', а не 'две тысячи'). "
    "Слова 'рублей/руб/р' не добавляй. "
    "Описание может быть из нескольких слов. "
    "Категория — последнее слово."
)

# --- числа словами (минимально, чтобы прожить) ---
NUM_WORDS_RU = {
    "ноль": 0,
    "один": 1, "одна": 1,
    "два": 2, "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
    "шестьдесят": 60,
    "семьдесят": 70,
    "восемьдесят": 80,
    "девяносто": 90,
    "сто": 100,
    "двести": 200,
    "триста": 300,
    "четыреста": 400,
    "пятьсот": 500,
    "шестьсот": 600,
    "семьсот": 700,
    "восемьсот": 800,
    "девятьсот": 900,
    "тысяча": 1000,
    "тысячи": 1000,
    "тысяч": 1000,
    "тыща": 1000,
    "тыщи": 1000,
}


def _ru_words_to_number(text: str) -> int | None:
    """
    Упрощённый перевод слов в число:
    понимает 'триста', 'двести пятьдесят', 'две тыщи', 'тысяча двести'.
    """
    tokens = re.findall(r"[а-яё]+", (text or "").lower())
    if not tokens:
        return None

    total = 0
    current = 0
    got_any = False

    for t in tokens:
        if t not in NUM_WORDS_RU:
            continue

        got_any = True
        v = NUM_WORDS_RU[t]

        if v == 1000:
            if current == 0:
                current = 1
            total += current * 1000
            current = 0
        else:
            current += v

    if not got_any:
        return None

    return total + current


def _extract_amount_any(text: str) -> tuple[float, str]:
    """
    Вытаскивает сумму:
    1) ищет цифры: 300, 1500, 1299.90
    2) если нет — пытается слова: 'триста', 'две тыщи'

    Возвращает:
      (amount_rub, rest_text_after_amount)
    """
    t = (text or "").strip().lower()
    t = t.replace("₽", " ")
    t = t.replace(",", ".")
    t = re.sub(r"\b(руб(лей)?\.?|руб\.|р)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # 1) цифры
    m = re.search(r"(\d+(?:\.\d+)?)", t)
    if m:
        amount = float(m.group(1))
        rest = (t[m.end():]).strip(" ,.-")
        return amount, rest

    # 2) словами
    n = _ru_words_to_number(t)
    if n is not None:
        tokens = re.findall(r"[а-яё]+|\d+(?:[.,]\d+)?", t)
        keep = []
        for tok in tokens:
            if tok in NUM_WORDS_RU:
                continue
            if tok in ("рубль", "рубля", "рублей", "руб", "р"):
                continue
            keep.append(tok)
        rest = " ".join(keep).strip(" ,.-")
        return float(n), rest

    raise ValueError("Не нашла сумму ни цифрами, ни словами.")


def parse_expense_voice(text: str) -> tuple[float, str, str]:
    """
    Ожидаем формат:
      '300 яйца продукты'
      '2000 бензин транспорт'
      '1500 кофе на вынос продукты'

    Категория — последнее слово.
    """
    if not text or not text.strip():
        raise ValueError("Пустой текст")

    amount_rub, rest = _extract_amount_any(text)
    rest = re.sub(r"\s+", " ", rest).strip()

    if not rest:
        raise ValueError("После суммы не осталось описания и категории.")

    words = rest.split(" ")
    if len(words) < 2:
        raise ValueError("Нужно минимум: сумма описание категория. Пример: '300 яйца продукты'.")

    category = words[-1].strip(" .,!?:;")
    description = " ".join(words[:-1]).strip(" .,!?:;")

    if not description:
        raise ValueError("Не нашла описание (между суммой и категорией).")

    return amount_rub, description, category


async def _whisper_transcribe_file(openai_client, file_path: str) -> str:
    """
    Whisper-вызов в openai-python синхронный, поэтому уводим в отдельный поток.
    """
    def _call() -> str:
        with open(file_path, "rb") as f:
            res = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="ru",
                response_format="text",
                prompt=PROMPT_RU,
            )
        if hasattr(res, "text"):
            return (res.text or "").strip()
        return str(res).strip()

    return await asyncio.to_thread(_call)


def register_voice_whisper_handler(
    *,
    dp,
    bot,
    http_client: httpx.AsyncClient,
    openai_client,
    telegram_bot_token: str,
    backend_url: str,
) -> None:
    """
    Регистрирует обработчик голосовых сообщений в aiogram Dispatcher.

    backend_url: базовый URL бэкенда
    """
    @dp.message(lambda m: m.voice is not None)
    async def voice_expense(m: Message):
        """
        Хэндлер голосового сообщения.

        LAST CHANGES:
        - теперь отправляем в /expenses
        - добавляем telegram_user_id
        - нет “/link”
        """
        tg_file = await bot.get_file(m.voice.file_id)
        tg_url = f"https://api.telegram.org/file/bot{telegram_bot_token}/{tg_file.file_path}"

        tmp_path = None
        try:
            # 1) скачиваем .ogg
            r = await http_client.get(tg_url)
            r.raise_for_status()

            # 2) сохраняем во временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
                tmp.write(r.content)
                tmp_path = tmp.name

            # 3) распознаём Whisper
            recognized = await _whisper_transcribe_file(openai_client, tmp_path)
            if not recognized:
                await m.answer("Не смогла получить текст из голосового (пусто).")
                return

            await m.answer(f"Распознано: {recognized}")

            # 4) парсим распознанное
            try:
                amount_rub, description, category = parse_expense_voice(recognized)
            except ValueError as e:
                await m.answer(
                    f"{e}\n"
                    "Скажи так: '300 яйца продукты' или '2000 бензин транспорт'.\n"
                    "Категория — последнее слово."
                )
                return

            # 5) пишем на бэкенд
            amount_cents = int(round(amount_rub * 100))
            payload = {
                "telegram_user_id": m.from_user.id,
                "telegram_chat_id": m.chat.id,
                "amount_cents": amount_cents,
                "description": description,
                "category": category,
                "subcategory": None,
                "occurred_at": None,
            }

            try:
                rr = await http_client.post(f"{backend_url.rstrip('/')}/expenses", json=payload)
                rr.raise_for_status()
                resp = rr.json()
            except Exception as e:
                await m.answer(f"Ошибка связи с сервером: {e}")
                return

            if not resp.get("ok", True):
                await m.answer(resp.get("error", "Не получилось сохранить расход."))
                return

            await m.answer(
                f"внесен расход за ({description}), на сумму ({amount_rub}), "
                f"категория ({category}), подкатегория ()"
            )
        finally:
            # Убираем временный файл
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass


# ======================================================================
# FILE: telegram_bot/bot/main.py
# ======================================================================
"""
Точка входа Telegram-бота.

LAST CHANGES:
- убрали /link и всю “привязку телефона”
- /start теперь делает users/upsert на бэкенде (регистрация/обновление chat_id)
- /app теперь использует MINIAPP_URL (или fallback на BACKEND_URL/miniapp)
"""

import os
import asyncio
import logging

import httpx
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, WebAppInfo

from bot.voice_handler import register_voice_whisper_handler
from bot.expense_text_handler import handle_text_expense


logging.basicConfig(level=logging.INFO)
load_dotenv()


def env(name: str, default: str | None = None) -> str:
    """
    Чтение env-переменных.

    Если default не задан — переменная обязательная.
    """
    v = os.getenv(name, default)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


TELEGRAM_BOT_TOKEN = env("TELEGRAM_BOT_TOKEN")
BACKEND_URL = env("BACKEND_URL").rstrip("/")
OPENAI_API_KEY = env("OPENAI_API_KEY")

# URL мини-приложения можно вынести отдельно (на фронтовый домен),
# либо оставить как бэкенд раздаёт /miniapp
MINIAPP_URL = os.getenv("MINIAPP_URL", f"{BACKEND_URL}/miniapp").strip()


async def main() -> None:
    """
    Запуск бота:
    - создаём Bot/Dispatcher
    - создаём http client
    - регистрируем хэндлеры
    - стартуем polling
    """
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()

    client = httpx.AsyncClient(timeout=httpx.Timeout(20.0))

    # Инициализация OpenAI клиента
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # Регистрируем голосовой хэндлер (Whisper)
    register_voice_whisper_handler(
        dp=dp,
        bot=bot,
        http_client=client,
        openai_client=openai_client,
        telegram_bot_token=TELEGRAM_BOT_TOKEN,
        backend_url=BACKEND_URL,
    )

    @dp.message(Command("start"))
    async def start(m: Message):
        """
        /start:
        - upsert user на бэкенде
        - даём инструкцию и путь в mini app
        """
        # LAST CHANGES:
        # - раньше был /link для “привязки телефона”
        # - теперь регистрируем пользователя по telegram_user_id и сохраняем chat_id
        payload = {
            "telegram_user_id": m.from_user.id,
            "telegram_chat_id": m.chat.id,
        }
        try:
            await client.post(f"{BACKEND_URL}/users/upsert", json=payload)
        except Exception as e:
            await m.answer(f"Не смог зарегистрировать пользователя на сервере: {e}")
            return

        await m.answer("Я жив. Пиши расход текстом или голосом. Мини-приложение: /app")

    @dp.message(Command("app"))
    async def open_app(m: Message):
        """
        /app:
        - отдаём кнопку открытия Mini App
        """
        kb = ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(
                        text="Открыть расходы",
                        web_app=WebAppInfo(url=MINIAPP_URL),
                    )
                ]
            ],
            resize_keyboard=True,
        )
        await m.answer("Нажми кнопку, чтобы открыть мини-приложение расходов:", reply_markup=kb)

    @dp.message()
    async def text_expense(m: Message):
        """
        Любой текст:
        - пытаемся распарсить как расход (с форматированием //)
        """
        await handle_text_expense(m=m, client=client, backend_url=BACKEND_URL)

    try:
        await dp.start_polling(bot)
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())

