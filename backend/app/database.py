"""Инициализация подключения к базе данных и фабрика сессий.

Содержит:
- создание SQLAlchemy engine для подключения к БД;
- фабрику сессий `SessionLocal`, которая используется во всём приложении;
- функцию `init_db`, создающую таблицы по объявленным ORM‑моделям.

По умолчанию URL БД берётся из переменной окружения `DATABASE_URL`.
Если она не задана, используется локальный SQLite‑файл (MVP‑режим).
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base


def _make_engine():
    """Создаёт SQLAlchemy engine с поддержкой PostgreSQL и SQLite.

    Предпочитаем PostgreSQL, если указан `DATABASE_URL`.
    Если нет — падаем обратно на SQLite‑файл в корне проекта.
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

    # Остальные драйверы (например, PostgreSQL) — без специальных настроек
    return create_engine(database_url, echo=False)


engine = _make_engine()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def init_db() -> None:
    """Создаёт таблицы по моделям, если их ещё нет (простая инициализация для MVP)."""
    Base.metadata.create_all(bind=engine)
