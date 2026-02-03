from collections.abc import Generator
from typing import Sequence

from sqlalchemy.orm import Session

from .database import SessionLocal
from .models import User, Category, Expense


# ---------- Общая функция для сессий ----------

def get_db() -> Generator[Session, None, None]:
    """Генератор сессий БД (одна сессия на запрос)."""
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
    """Находит пользователя по Telegram ID или создаёт, если его нет."""
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


# ---------- Категории ----------

def create_category(
    db: Session,
    user_id: int,
    name: str,
) -> Category:
    """Создаёт новую категорию для пользователя."""
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
    """Возвращает все категории пользователя."""
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
    """Находит или создаёт категорию по имени для пользователя."""
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


# ---------- Расходы ----------

def create_expense(
    db: Session,
    user_id: int,
    amount_cents: int,
    description: str,
    category_name: str,
    subcategory_name: str | None,
    occurred_at: str,
    source: str,
) -> Expense:
    """Создаёт расход для пользователя."""
    category = get_or_create_category(
        db=db,
        user_id=user_id,
        name=category_name,
    )

    expense = Expense(
        user_id=user_id,
        category_id=category.id,
        amount_cents=amount_cents,
        description=description,
        occurred_at=occurred_at,
        source=source,
        # если в модели есть поле subcategory_name — можно добавить:
        # subcategory_name=subcategory_name,
    )
    db.add(expense)
    db.commit()
    db.refresh(expense)
    return expense
