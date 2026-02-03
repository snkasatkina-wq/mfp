from .database import SessionLocal
from . import crud


def main() -> None:
    db = SessionLocal()

    try:
        # 1. создаём или находим пользователя
        user = crud.get_or_create_user(
            db=db,
            telegram_user_id=123456789,
            telegram_chat_id=123456789,
        )
        print("User:", user.id, user.telegram_user_id)

        # 2. создаём пару категорий
        crud.create_category(db, user_id=user.id, name="Еда")
        crud.create_category(db, user_id=user.id, name="Транспорт")

        # 3. получаем категории пользователя
        categories = crud.get_categories(db, user_id=user.id)
        print("Categories:")
        for c in categories:
            print("-", c.id, c.name)

    finally:
        db.close()


if __name__ == "__main__":
    main()
