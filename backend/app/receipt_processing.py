"""Модуль обработки чеков через OpenAI Vision и категоризации расходов.

Содержит функции:
- `parse_receipt_image`: парсинг чека через OpenAI Vision (извлечение позиций и сумм);
- `classify_item`: определение категории для одной позиции через OpenAI Chat;
- `process_receipt`: полная обработка чека (парсинг + категоризация всех позиций).
"""

import base64
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI


def load_receipt_parsing_prompt() -> str:
    """Читает промпт для парсинга чеков из файла чеки/prompt_receipt_parsing.md."""
    # Ищем промпт относительно корня проекта
    project_root = Path(__file__).resolve().parents[2]
    prompt_path = project_root / "чеки" / "prompt_receipt_parsing.md"
    return prompt_path.read_text(encoding="utf-8")


def load_category_classification_prompt() -> str:
    """Читает промпт для категоризации расходов из файла чеки/prompt_category_classification.md."""
    project_root = Path(__file__).resolve().parents[2]
    prompt_path = project_root / "чеки" / "prompt_category_classification.md"
    return prompt_path.read_text(encoding="utf-8")


def encode_image_to_data_url(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Кодирует изображение в data URL для передачи в OpenAI Vision."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def parse_receipt_image(
    client: OpenAI,
    prompt_text: str,
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
) -> dict[str, Any]:
    """Парсит чек через OpenAI Vision и возвращает список позиций с ценами.

    Args:
        client: клиент OpenAI.
        prompt_text: текст промпта для парсинга.
        image_bytes: байты изображения чека.
        mime_type: MIME-тип изображения (например, "image/jpeg", "image/png").

    Returns:
        Словарь с ключом "items", содержащий список словарей:
        [{"name": str, "raw_name": str, "total_price": float}, ...]

    Raises:
        ValueError: если ответ модели пустой или невалидный JSON.
        json.JSONDecodeError: если не удалось распарсить JSON из ответа модели.
    """
    image_url = encode_image_to_data_url(image_bytes, mime_type)

    messages = [
        {
            "role": "system",
            "content": "Ты помощник, который структурирует кассовые чеки в России.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=1500,
        temperature=0.1,
    )

    content = resp.choices[0].message.content
    if not content:
        raise ValueError("Пустой ответ модели")

    # Убираем пробелы в начале и конце
    content = content.strip()

    # Часто модель оборачивает JSON в ```json ... ```
    if content.startswith("```"):
        lines = content.splitlines()
        # Убираем первую строку с ``` или ```json
        if lines:
            lines = lines[1:]
        # Если последняя строка тоже ``` — убираем
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        # Выводим сырой ответ для отладки
        print("----- RAW MODEL OUTPUT -----")
        print(content)
        print("----- END RAW OUTPUT -----")
        raise e


def classify_item(
    client: OpenAI,
    prompt_text: str,
    item_name: str,
    categories: list[str],
) -> str:
    """Определяет категорию для одной позиции расхода через OpenAI Chat.

    Args:
        client: клиент OpenAI.
        prompt_text: текст промпта для категоризации.
        item_name: наименование позиции расхода (например, "Морковь", "Такси").
        categories: список допустимых категорий (включая "не определено").

    Returns:
        Название категории из списка categories (строка).

    Raises:
        ValueError: если ответ модели пустой.
    """
    # Дополнительный контекст в виде JSON, чтобы модель однозначно поняла формат входа
    data_ctx = {
        "item_name": item_name,
        "categories": categories,
    }

    messages = [
        {
            "role": "system",
            "content": "Ты помощник, который выбирает категорию расхода строго из заданного списка.",
        },
        {
            "role": "user",
            "content": (
                f"{prompt_text}\n\n"
                "Вот данные для категоризации в формате JSON:\n"
                f"{json.dumps(data_ctx, ensure_ascii=False)}\n\n"
                "Ответь ТОЛЬКО названием одной категории из списка, без кавычек и без лишнего текста."
            ),
        },
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=50,
        temperature=0.1,
    )

    content = resp.choices[0].message.content or ""
    # Берём первую строку и чистим пробелы
    line = content.strip().splitlines()[0].strip()
    return line
