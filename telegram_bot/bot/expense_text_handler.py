from aiogram.types import Message


def parse_expense_text(text: str) -> dict:
    """Парсит строку расхода вида:

    сумма // описание // категория // (подкатегория опционально)

    Примеры:
      300 // яйца и хлеб // продукты
      1500,50 // такси до аэропорта // транспорт // работа
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Пустое сообщение. Формат: сумма // описание // категория // (подкатегория опционально)")

    # Не выкидываем пустые части, чтобы корректно ловить отсутствующие поля
    parts = [p.strip() for p in raw.split("//")]

    if len(parts) < 3 or len(parts) > 4:
        raise ValueError("Формат: сумма // описание // категория // (подкатегория опционально)")

    amount_part, description_part, category_part = parts[0], parts[1], parts[2]
    subcategory_part = parts[3] if len(parts) == 4 else ""

    if not amount_part:
        raise ValueError("Не найдена сумма. Формат: сумма // описание // категория // (подкатегория опционально)")
    if not description_part:
        raise ValueError("Не найдено описание. Формат: сумма // описание // категория // (подкатегория опционально)")
    if not category_part:
        raise ValueError("Не найдена категория. Формат: сумма // описание // категория // (подкатегория опционально)")

    # Поддерживаем пробелы и запятую как разделитель дробной части
    amount_raw = amount_part.replace(" ", "").replace(",", ".")
    try:
        amount = float(amount_raw)
    except ValueError:
        raise ValueError("Не получилось распознать сумму. Пример: 300 или 1500.50")

    if amount <= 0:
        raise ValueError("Сумма должна быть больше нуля.")

    amount_cents = int(round(amount * 100))

    description = description_part.strip()
    category = category_part.strip()
    subcategory = subcategory_part.strip()

    return {
        "amount": amount,
        "amount_cents": amount_cents,
        "description": description,
        "category": category,
        "subcategory": subcategory,
    }


async def handle_text_expense(*, m: Message, client, backend_url: str) -> None:
    text = (m.text or "").strip()
    if not text or text.startswith("/"):
        return

    try:
        data = parse_expense_text(text)
    except ValueError as e:
        # Показываем пользователю понятное сообщение об ошибке формата
        await m.answer(str(e))
        return

    # 1) гарантируем, что пользователь существует на бэкенде
    try:
        await client.post(
            f"{backend_url.rstrip('/')}/users/upsert",
            json={
                "telegram_user_id": m.from_user.id,
                "telegram_chat_id": m.chat.id,
            },
        )
    except Exception:
        # даже если не получилось — попробуем сохранить расход, пусть бэк сам скажет что не так
        pass

    payload = {
        "telegram_user_id": m.from_user.id,
        "telegram_chat_id": m.chat.id,  # полезно для уведомлений
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
        # универсальная ошибка от бэка
        await m.answer(resp.get("error", "Не получилось сохранить расход."))
        return

    await m.answer(
        f"внесен расход за ({data['description']}), на сумму ({data['amount']}), "
        f"категория ({data['category']}), подкатегория ({data['subcategory']})"
    )
