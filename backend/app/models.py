"""Описание ORM‑моделей SQLAlchemy для бекенда.

Модели соответствуют таблицам в PostgreSQL и используются для работы через ORM:
- `User` — данные пользователя, связанного с Telegram;
- `Category` / `Subcategory` — категории и подкатегории расходов;
- `Expense` — отдельный расход пользователя;
- `CategoryLimit` — лимиты по категориям (на будущее развитие сервиса);
- `ChatDeviceLink` — связь telegram‑чата и device_id устройства.
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
)


class Base(DeclarativeBase):
    """Базовый класс для всех моделей (декларативная база SQLAlchemy)."""

    pass


class User(Base):
    __tablename__ = "users"

    # Внутренний идентификатор пользователя в БД
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Идентификатор пользователя в Telegram (user_id)
    telegram_user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, unique=True)
    # Идентификатор чата в Telegram (chat_id) — для отправки уведомлений и связи с ботом
    telegram_chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False, unique=True)
    # E‑mail пользователя (опционально, пригодится позже)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # Пороговый процент для уведомлений о перерасходе (MVP‑настройка)
    threshold_percent: Mapped[int] = mapped_column(Integer, nullable=False, default=80)
    # Путь к сгенерированной пользователем картинке (относительно директории static, например "user_images/123.png")
    custom_image_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # Дата/время создания записи пользователя
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False), server_default=func.now(), nullable=False
    )


class Category(Base):
    __tablename__ = "categories"

    # Уникальный идентификатор категории
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Владелец категории (пользователь)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Название категории (например, "Еда", "Транспорт")
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    # Время создания категории
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False), server_default=func.now(), nullable=False
    )


class Subcategory(Base):
    __tablename__ = "subcategories"

    # Идентификатор подкатегории
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Родительская категория
    category_id: Mapped[int] = mapped_column(
        ForeignKey("categories.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Название подкатегории (например, "Кофейни" внутри категории "Еда")
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    # Время создания подкатегории
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False), server_default=func.now(), nullable=False
    )


class Expense(Base):
    __tablename__ = "expenses"

    # Идентификатор расхода
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Пользователь, которому принадлежит расход
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Сумма расхода в центах (то есть amount * 100)
    amount_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    # Когда расход произошёл (системное поле datetime в БД)
    occurred_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False), nullable=False
    )
    # Текстовое описание (например, "кофе", "такси до аэропорта")
    description: Mapped[str] = mapped_column(String, nullable=False)
    # Категория расхода
    category_id: Mapped[int] = mapped_column(
        ForeignKey("categories.id"),
        nullable=False,
    )
    # Подкатегория (опционально)
    subcategory_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("subcategories.id"),
        nullable=True,
    )
    # Источник, откуда пришёл расход (например, "telegram")
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    # Группа чека/пакета расходов (опционально, для сканов чеков и т.п.)
    receipt_group_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    # Время создания записи расхода
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False), server_default=func.now(), nullable=False
    )


class CategoryLimit(Base):
    __tablename__ = "category_limits"

    # Идентификатор лимита
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Категория, к которой привязан лимит
    category_id: Mapped[int] = mapped_column(
        ForeignKey("categories.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    # Тип периода, для которого действует лимит (например, "month", "week")
    period_type: Mapped[str] = mapped_column(String(20), nullable=False)
    # Сумма лимита в центах
    limit_amount_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    # Флаг, включён ли лимит
    is_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )
    # Время создания записи лимита
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False), server_default=func.now(), nullable=False
    )


class ChatDeviceLink(Base):
    __tablename__ = "chat_device_links"

    # Идентификатор записи
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Telegram chat_id, с которым связана конкретная запись/устройство
    telegram_chat_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        unique=True,
    )
    # Технический идентификатор устройства (device_id), к которому привязан чат
    device_id: Mapped[str] = mapped_column(String(100), nullable=False)


class Receipt(Base):
    __tablename__ = "receipts"

    # Идентификатор записи чека
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Пользователь, которому принадлежит чек
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Уникальный идентификатор группы чека (для связи с несколькими расходами)
    receipt_group_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    # Путь к сохранённому файлу чека на сервере
    file_path: Mapped[str] = mapped_column(String(255), nullable=False)
    # Дата/время загрузки чека
    uploaded_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False), server_default=func.now(), nullable=False
    )
