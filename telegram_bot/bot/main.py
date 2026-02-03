import os
import asyncio
from dotenv import load_dotenv

import httpx
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, WebAppInfo

from bot.voice_handler import register_voice_whisper_handler
from bot.expense_text_handler import handle_text_expense

import logging
logging.basicConfig(level=logging.INFO)

load_dotenv()


def env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


TELEGRAM_BOT_TOKEN = env("TELEGRAM_BOT_TOKEN")
BACKEND_URL = env("BACKEND_URL").rstrip("/")
OPENAI_API_KEY = env("OPENAI_API_KEY")


async def main() -> None:
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()
    client = httpx.AsyncClient(timeout=httpx.Timeout(20.0))

    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    register_voice_whisper_handler(
        dp=dp,
        bot=bot,
        http_client=client,
        openai_client=openai_client,
        telegram_bot_token=TELEGRAM_BOT_TOKEN,
        backend_url=BACKEND_URL,
    )

    @dp.message(Command("me"))
    async def me(m: Message):
        await m.answer(f"chat_id = {m.chat.id}")

    @dp.message(Command("start"))
    async def start(m: Message):
        await m.answer("Я жив. Для привязки: /link КОД")

    @dp.message(Command("link"))
    async def link(m: Message):
        parts = (m.text or "").split()
        if len(parts) != 2:
            await m.answer("Формат: /link КОД")
            return

        code = parts[1].strip().upper()

        payload = {
            "telegram_chat_id": str(m.chat.id),
            "code": code,
        }

        try:
            r = await client.post(f"{BACKEND_URL}/link/confirm", json=payload)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            await m.answer(f"Ошибка связи с сервером: {e}")
            return

        if data.get("ok"):
            await m.answer(f"Привязка успешна. device_id={data.get('device_id')}")
        else:
            await m.answer("Код неверный или истёк. Сгенерируй новый в приложении и попробуй снова.")

    @dp.message(Command("app"))
    async def open_app(m: Message):
        kb = ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(
                        text="Открыть расходы",
                        web_app=WebAppInfo(url="http://127.0.0.1:8000/miniapp"),
                    )
                ]
            ],
            resize_keyboard=True,
        )
        await m.answer(
            "Нажми кнопку, чтобы открыть мини‑приложение расходов:",
            reply_markup=kb,
        )

    @dp.message()  # общий хэндлер для текста
    async def text_expense(m: Message):
        await handle_text_expense(
            m=m,
            client=client,
            backend_url=BACKEND_URL,
        )

    try:
        await dp.start_polling(bot)
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
