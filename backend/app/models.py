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
    """Базовый класс для всех моделей."""
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    telegram_user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, unique=True)
    telegram_chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False, unique=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    threshold_percent: Mapped[int] = mapped_column(Integer, nullable=False, default=80)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False), server_default=func.now(), nullable=False
    )


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False), server_default=func.now(), nullable=False
    )


class Subcategory(Base):
    __tablename__ = "subcategories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category_id: Mapped[int] = mapped_column(
        ForeignKey("categories.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False), server_default=func.now(), nullable=False
    )


class Expense(Base):
    __tablename__ = "expenses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    amount_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False), nullable=False
    )
    description: Mapped[str] = mapped_column(String, nullable=False)
    category_id: Mapped[int] = mapped_column(
        ForeignKey("categories.id"),
        nullable=False,
    )
    subcategory_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("subcategories.id"),
        nullable=True,
    )
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    receipt_group_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False), server_default=func.now(), nullable=False
    )


class CategoryLimit(Base):
    __tablename__ = "category_limits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category_id: Mapped[int] = mapped_column(
        ForeignKey("categories.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    period_type: Mapped[str] = mapped_column(String(20), nullable=False)
    limit_amount_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    is_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False), server_default=func.now(), nullable=False
    )

