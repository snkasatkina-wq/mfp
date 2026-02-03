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
        # Регистрируем/обновляем пользователя на бэкенде
        try:
            await client.post(
                f"{BACKEND_URL}/users",
                json={
                    "telegram_user_id": m.from_user.id,
                    "telegram_chat_id": m.chat.id,
                },
            )
        except Exception as e:
            await m.answer(f"Ошибка регистрации на сервере: {e}")
            return
        
        await m.answer(
            "Привет! Я бот для учёта расходов.\n\n"
            "Отправь расход в формате:\n"
            "сумма // описание // категория\n\n"
            "Пример: 300 // яйца и хлеб // продукты\n\n"
            "Мини-приложение: /app"
        )

    @dp.message(Command("app"))
    async def open_app(m: Message):
        miniapp_url = os.getenv("MINIAPP_URL", f"{BACKEND_URL}/miniapp")
        # Добавляем telegram_user_id в URL для идентификации пользователя
        user_id = m.from_user.id
        miniapp_url_with_user = f"{miniapp_url}?user_id={user_id}"
        
        # Логируем для отладки
        print(f"[DEBUG] Открытие мини-приложения для user_id={user_id}")
        print(f"[DEBUG] URL: {miniapp_url_with_user}")
        
        kb = ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(
                        text="Открыть расходы",
                        web_app=WebAppInfo(url=miniapp_url_with_user),
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
