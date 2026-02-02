import asyncio
import os
import re
import tempfile

import httpx
from aiogram.types import Message


PROMPT_RU = (
    "Распознавай короткие голосовые заметки о расходах денег. "
    "Пример: триста яйцо продукты или 2 тыщи рублей бензин транспорт"
    "Первое - всегда сумма, с уточнением 'рублей' или без этого уточнения, просто сумма"
    "Потом идет описание слово или слова - описание расхода, затем - категория расхода с названием категории"
    "Формат вывода: '1500 рублей, кофейня, еда вне дома'. "
    "Всегда пиши сумму цифрами (например, '2000', а не 'две тысячи'). "
    "Не добавляй лишних слов, только: сумма, место, категория — через запятые."
)


def _extract_amount_rub(text: str) -> float:
    """
    Достаём сумму из начала строки: '1500 рублей' / '1500 руб' / '1500' / '1500.50'
    """
    t = text.strip().lower()
    t = t.replace("₽", " ")
    t = t.replace(",", ".")
    t = re.sub(r"\b(руб(лей)?\.?|р)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    m = re.search(r"(\d+(?:\.\d+)?)", t)
    if not m:
        raise ValueError("Не нашла сумму в первой части фразы.")
    return float(m.group(1))


def parse_expense_comma(text: str) -> tuple[float, str, str]:
    """
    Ожидаем: '1500 рублей, кофейня, еда вне дома'
    Возвращаем: amount_rub, description, category
    """
    parts = [p.strip() for p in text.split(",")]
    parts = [p for p in parts if p]

    if len(parts) < 3:
        raise ValueError("Формат: '1500 рублей, кофейня, категория еда вне дома' (3 части через запятую).")

    amount_rub = _extract_amount_rub(parts[0])
    description = parts[1]
    category = ", ".join(parts[2:])  # если в категории случайно есть запятая
    return amount_rub, description, category


async def _whisper_transcribe_file(openai_client, file_path: str) -> str:
    """
    Вызов openai-python синхронный, поэтому уводим в отдельный поток.
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
        # response_format="text" иногда возвращает строку, иногда объект с .text
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
    Регистрирует aiogram-хэндлер для voice.
    Требует:
      - bot (aiogram Bot)
      - http_client (httpx AsyncClient)
      - openai_client (OpenAI() из openai-python)
      - telegram_bot_token (для скачивания файла из Telegram)
      - backend_url (например http://127.0.0.1:8000)
    """

    @dp.message(lambda m: m.voice is not None)
    async def voice_expense(m: Message):
        # 1) скачать voice во временный файл
        tg_file = await bot.get_file(m.voice.file_id)
        tg_url = f"https://api.telegram.org/file/bot{telegram_bot_token}/{tg_file.file_path}"

        tmp_path = None
        try:
            r = await http_client.get(tg_url)
            r.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
                tmp.write(r.content)
                tmp_path = tmp.name

            # 2) распознать через whisper-1
            recognized = await _whisper_transcribe_file(openai_client, tmp_path)
            if not recognized:
                await m.answer("Не смогла получить текст из голосового (пусто).")
                return

            # обязательная валидация распознанного
            await m.answer(f"Распознано: {recognized}")

            # 3) распарсить "сумма, описание, категория"
            try:
                amount_rub, description, category = parse_expense_comma(recognized)
            except ValueError as e:
                await m.answer(str(e))
                return

            amount_cents = int(round(amount_rub * 100))

            # 4) отправить в backend очередь
            payload = {
                "telegram_chat_id": str(m.chat.id),
                "amount_cents": amount_cents,
                "description": description,
                "category": category,
                "subcategory": None,
                "occurred_at": None,
            }

            try:
                rr = await http_client.post(f"{backend_url.rstrip('/')}/queue/expense", json=payload)
                rr.raise_for_status()
                resp = rr.json()
            except Exception as e:
                await m.answer(f"Ошибка связи с сервером: {e}")
                return

            if not resp.get("ok"):
                await m.answer("Бот не привязан к телефону. Сделай привязку командой /link КОД")
                return

            # 5) подтверждение
            await m.answer(
                f"внесен расход за ({description}), на сумму ({amount_rub}), категория ({category}), подкатегория ()"
            )

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
