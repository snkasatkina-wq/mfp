from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base

DATABASE_URL = "postgresql://mvp_user:Test12345!@localhost:5432/mvp_costs"


engine = create_engine(DATABASE_URL, echo=False)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def init_db() -> None:
    """Создаёт таблицы по моделям, если их ещё нет."""
    Base.metadata.create_all(bind=engine)
