import asyncio
import os
import re
import tempfile

import httpx
from aiogram.types import Message


PROMPT_RU = (
    "Распознавай короткие голосовые заметки о расходах денег. "
    "Выводи БЕЗ запятых, одной строкой, строго в виде: "
    "<СУММА_ЦИФРАМИ> <ОПИСАНИЕ> <КАТЕГОРИЯ>. "
    "Примеры: '300 яйца продукты', '2000 бензин транспорт', '1500 кофе развлечения'. "
    "Сумму всегда пиши цифрами (например, '2000', а не 'две тысячи'). "
    "Слова 'рублей/руб/р' не добавляй. "
    "Описание может быть из нескольких слов. Категория — последнее слово."
)


# --- числа словами (минимально, чтобы прожить) ---
NUM_WORDS_RU = {
    "ноль": 0,
    "один": 1, "одна": 1,
    "два": 2, "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
    "шестьдесят": 60,
    "семьдесят": 70,
    "восемьдесят": 80,
    "девяносто": 90,
    "сто": 100,
    "двести": 200,
    "триста": 300,
    "четыреста": 400,
    "пятьсот": 500,
    "шестьсот": 600,
    "семьсот": 700,
    "восемьсот": 800,
    "девятьсот": 900,
    "тысяча": 1000, "тысячи": 1000, "тысяч": 1000,
    "тыща": 1000, "тыщи": 1000,  # разговорное
}


def _ru_words_to_number(text: str) -> int | None:
    """
    Очень упрощённый перевод: понимает 'триста', 'двести пятьдесят', 'две тыщи', 'тысяча двести'.
    """
    tokens = re.findall(r"[а-яё]+", (text or "").lower())
    if not tokens:
        return None

    total = 0
    current = 0
    got_any = False

    for t in tokens:
        if t not in NUM_WORDS_RU:
            continue
        got_any = True
        v = NUM_WORDS_RU[t]
        if v == 1000:
            if current == 0:
                current = 1
            total += current * 1000
            current = 0
        else:
            current += v

    if not got_any:
        return None
    return total + current


def _extract_amount_any(text: str) -> tuple[float, str]:
    """
    Пытаемся вытащить сумму:
    - сначала ищем цифры (300, 1500, 1299.90)
    - если нет — пробуем слова ('триста', 'две тыщи')
    Возвращаем (amount_rub, rest_text_after_amount)
    """
    t = (text or "").strip().lower()
    t = t.replace("₽", " ")
    t = t.replace(",", ".")
    t = re.sub(r"\b(руб(лей)?\.?|руб\.|р)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # 1) цифры
    m = re.search(r"(\d+(?:\.\d+)?)", t)
    if m:
        amount = float(m.group(1))
        rest = (t[m.end():]).strip(" ,.-")
        return amount, rest

    # 2) словами
    n = _ru_words_to_number(t)
    if n is not None:
        # вырежем слова-числа и "рублей", оставим остальное как rest
        tokens = re.findall(r"[а-яё]+|\d+(?:[.,]\d+)?", t)
        keep = []
        for tok in tokens:
            if tok in NUM_WORDS_RU:
                continue
            if tok in ("рубль", "рубля", "рублей", "руб", "р"):
                continue
            keep.append(tok)
        rest = " ".join(keep).strip(" ,.-")
        return float(n), rest

    raise ValueError("Не нашла сумму ни цифрами, ни словами.")


def parse_expense_voice(text: str) -> tuple[float, str, str]:
    """
    Ожидаем голосовой формат БЕЗ запятых:
    '300 яйца продукты'
    '2000 бензин транспорт'
    '1500 кофе на вынос продукты'
    Сумма может быть без слова 'рублей'.
    Категория — последнее слово.
    """
    if not text or not text.strip():
        raise ValueError("Пустой текст")

    amount_rub, rest = _extract_amount_any(text)

    rest = re.sub(r"\s+", " ", rest).strip()
    if not rest:
        raise ValueError("После суммы не осталось описания и категории.")

    words = rest.split(" ")
    if len(words) < 2:
        raise ValueError("Нужно минимум: сумма описание категория. Пример: '300 яйца продукты'.")

    category = words[-1].strip(" .,!?:;")
    description = " ".join(words[:-1]).strip(" .,!?:;")

    if not description:
        raise ValueError("Не нашла описание (между суммой и категорией).")

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

    @dp.message(lambda m: m.voice is not None)
    async def voice_expense(m: Message):
        tg_file = await bot.get_file(m.voice.file_id)
        tg_url = f"https://api.telegram.org/file/bot{telegram_bot_token}/{tg_file.file_path}"

        tmp_path = None
        try:
            r = await http_client.get(tg_url)
            r.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
                tmp.write(r.content)
                tmp_path = tmp.name

            recognized = await _whisper_transcribe_file(openai_client, tmp_path)
            if not recognized:
                await m.answer("Не смогла получить текст из голосового (пусто).")
                return

            await m.answer(f"Распознано: {recognized}")

            try:
                amount_rub, description, category = parse_expense_voice(recognized)
            except ValueError as e:
                await m.answer(
                    f"{e}\n"
                    "Скажи так: '300 яйца продукты' или '2000 бензин транспорт'. "
                    "Категория — последнее слово."
                )
                return

            amount_cents = int(round(amount_rub * 100))

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

            await m.answer(
                f"внесен расход за ({description}), на сумму ({amount_rub}), категория ({category}), подкатегория ()"
            )

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
