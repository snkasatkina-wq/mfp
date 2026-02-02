import re
from aiogram.types import Message

def parse_expense_text(text: str) -> dict:
    parts = [p.strip() for p in text.split("//")]
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


async def handle_text_expense(
    *,
    m: Message,
    client,
    backend_url: str,
) -> None:
    text = (m.text or "").strip()
    if not text:
        return
    if text.startswith("/"):
        return  # команды не трогаем

    try:
        data = parse_expense_text(text)
    except ValueError as e:
        await m.answer(str(e))
        return

    payload = {
        "telegram_chat_id": str(m.chat.id),
        "amount_cents": data["amount_cents"],
        "description": data["description"],
        "category": data["category"],
        "subcategory": data["subcategory"] if data["subcategory"] else None,
        "occurred_at": None,
    }

    try:
        r = await client.post(f"{backend_url}/queue/expense", json=payload)
        r.raise_for_status()
        resp = r.json()
    except Exception as e:
        await m.answer(f"Ошибка связи с сервером: {e}")
        return

    if not resp.get("ok"):
        await m.answer("Бот не привязан к телефону. Сделай привязку командой /link КОД")
        return

    await m.answer(
        f"внесен расход за ({data['description']}), на сумму ({data['amount']}), категория ({data['category']}), подкатегория ({data['subcategory']})"
    )