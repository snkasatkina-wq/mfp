"""Слой работы с БД (CRUD‑функции).

Содержит логику:
- создания/поиска пользователей;
- работы с категориями и подкатегориями;
- создания расходов;
- хранения связки telegram‑чата и device_id.

Все функции принимают уже открытую сессию SQLAlchemy `Session` и сами её не создают,
кроме вспомогательного генератора `get_db`, который используется в FastAPI‑эндпоинтах.
"""

from collections.abc import Generator
from typing import Sequence

from sqlalchemy.orm import Session

from .database import SessionLocal
from .models import User, Category, Subcategory, Expense, ChatDeviceLink


# ---------- Общая функция для сессий ----------

def get_db() -> Generator[Session, None, None]:
    """Генератор сессий БД (одна сессия на запрос для FastAPI‑эндпоинта)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------- Пользователи ----------

def get_or_create_user(
    db: Session,
    telegram_user_id: int,
    telegram_chat_id: int,
) -> User:
    """Находит пользователя по Telegram ID/чат ID или создаёт нового.

    Пользователь считается тем же самым, если совпадает пара
    (`telegram_user_id`, `telegram_chat_id`).
    """
    user = (
        db.query(User)
        .filter(
            User.telegram_user_id == telegram_user_id,
            User.telegram_chat_id == telegram_chat_id,
        )
        .first()
    )
    if user is not None:
        return user

    user = User(
        telegram_user_id=telegram_user_id,
        telegram_chat_id=telegram_chat_id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ---------- Категории / Подкатегории ----------

def create_category(
    db: Session,
    user_id: int,
    name: str,
) -> Category:
    """Создаёт новую категорию для пользователя без проверок на дубликаты."""
    category = Category(
        user_id=user_id,
        name=name,
    )
    db.add(category)
    db.commit()
    db.refresh(category)
    return category


def get_categories(
    db: Session,
    user_id: int,
) -> Sequence[Category]:
    """Возвращает все категории пользователя, отсортированные по имени."""
    categories = (
        db.query(Category)
        .filter(Category.user_id == user_id)
        .order_by(Category.name.asc())
        .all()
    )
    return categories


def get_or_create_category(
    db: Session,
    user_id: int,
    name: str,
) -> Category:
    """Находит категорию по имени для пользователя или создаёт новую."""
    category = (
        db.query(Category)
        .filter(
            Category.user_id == user_id,
            Category.name == name,
        )
        .first()
    )
    if category is not None:
        return category

    category = Category(
        user_id=user_id,
        name=name,
    )
    db.add(category)
    db.commit()
    db.refresh(category)
    return category


def get_or_create_subcategory(
    db: Session,
    category_id: int,
    name: str,
) -> Subcategory:
    """Находит подкатегорию по имени для категории или создаёт новую."""
    subcategory = (
        db.query(Subcategory)
        .filter(
            Subcategory.category_id == category_id,
            Subcategory.name == name,
        )
        .first()
    )
    if subcategory is not None:
        return subcategory

    subcategory = Subcategory(
        category_id=category_id,
        name=name,
    )
    db.add(subcategory)
    db.commit()
    db.refresh(subcategory)
    return subcategory


# ---------- Расходы ----------

def create_expense(
    db: Session,
    user_id: int,
    amount_cents: int,
    description: str,
    category_name: str,
    subcategory_name: str | None,
    occurred_at,
    source: str,
) -> Expense:
    """Создаёт расход для пользователя.

    Параметр `occurred_at` ожидается как объект `datetime` (уже распарсенный из строки,
    если она пришла от пользователя).
    """
    category = get_or_create_category(
        db=db,
        user_id=user_id,
        name=category_name,
    )

    subcategory_id: int | None = None
    if subcategory_name is not None:
        subcategory = get_or_create_subcategory(
            db=db,
            category_id=category.id,
            name=subcategory_name,
        )
        subcategory_id = subcategory.id

    expense = Expense(
        user_id=user_id,
        category_id=category.id,
        subcategory_id=subcategory_id,
        amount_cents=amount_cents,
        description=description,
        occurred_at=occurred_at,
        source=source,
    )
    db.add(expense)
    db.commit()
    db.refresh(expense)
    return expense


# ---------- Привязка чата к устройству ----------

def upsert_chat_device_link(
    db: Session,
    telegram_chat_id: int,
    device_id: str,
) -> None:
    """Создаёт или обновляет привязку telegram_chat_id -> device_id.

    На уровне БД обеспечивается уникальность по `telegram_chat_id`,
    поэтому для каждого чата хранится не более одного связанного device_id.
    """
    link = (
        db.query(ChatDeviceLink)
        .filter(ChatDeviceLink.telegram_chat_id == telegram_chat_id)
        .first()
    )
    if link is None:
        link = ChatDeviceLink(
            telegram_chat_id=telegram_chat_id,
            device_id=device_id,
        )
        db.add(link)
    else:
        link.device_id = device_id

    db.commit()


def get_device_id_for_chat(
    db: Session,
    telegram_chat_id: int,
) -> str | None:
    """Возвращает device_id, привязанный к telegram_chat_id (или None)."""
    link = (
        db.query(ChatDeviceLink)
        .filter(ChatDeviceLink.telegram_chat_id == telegram_chat_id)
        .first()
    )
    return link.device_id if link is not None else None


def get_chat_id_for_device(
    db: Session,
    device_id: str,
) -> int | None:
    """Возвращает telegram_chat_id по device_id (или None)."""
    link = (
        db.query(ChatDeviceLink)
        .filter(ChatDeviceLink.device_id == device_id)
        .first()
    )
    return link.telegram_chat_id if link is not None else None


def update_user_custom_image_path(
    db: Session,
    user_id: int,
    custom_image_path: str | None,
) -> User:
    """Обновляет путь к сгенерированной картинке пользователя.
    
    `custom_image_path` должен быть относительным путём от директории static
    (например, "user_images/123.png").
    """
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise ValueError(f"Пользователь с id={user_id} не найден")
    
    user.custom_image_path = custom_image_path
    db.commit()
    db.refresh(user)
    return user
