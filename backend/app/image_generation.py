"""Модуль для генерации картинок через OpenAI DALL-E.

Содержит функции:
- улучшения промпта через ChatGPT;
- генерации изображения через DALL-E;
- сохранения сгенерированной картинки на диск.
"""

import os
from pathlib import Path
from typing import Optional

import httpx
from openai import OpenAI


def improve_prompt_with_chatgpt(
    openai_client: OpenAI,
    user_prompt: str,
) -> str:
    """Улучшает промпт пользователя через ChatGPT для лучшей генерации изображения.
    
    Args:
        openai_client: Клиент OpenAI для работы с API.
        user_prompt: Исходный промпт от пользователя (на русском или английском).
    
    Returns:
        Улучшенный промпт на английском языке, готовый для передачи в DALL-E.
    """
    improvement_prompt = (
        "Пользователь хочет сгенерировать картинку для приложения учета расходов. "
        "Его описание: \"{user_prompt}\"\n\n"
        "Улучши это описание для генерации картинки через DALL-E. "
        "Сделай описание более детальным и подходящим для генерации изображения, "
        "но сохрани основную идею пользователя. "
        "Ответь ТОЛЬКО улучшенным промптом на английском языке, без дополнительных объяснений."
    ).format(user_prompt=user_prompt)
    
    try:
        chat_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Используем более дешевую модель для улучшения промпта
            messages=[
                {
                    "role": "system",
                    "content": "Ты помощник, который улучшает описания для генерации изображений. Отвечай только улучшенным промптом на английском языке."
                },
                {
                    "role": "user",
                    "content": improvement_prompt
                }
            ],
            max_tokens=200,
            temperature=0.7,
        )
        
        improved_prompt = chat_response.choices[0].message.content.strip()
        
        # Если ChatGPT вернул что-то странное, используем оригинальный промпт с базовым улучшением
        if not improved_prompt or len(improved_prompt) < 10:
            improved_prompt = (
                f"A cute, modern illustration: {user_prompt}, "
                "minimalist style, suitable for a personal finance app, "
                "warm colors, friendly design, digital art style."
            )
        
        return improved_prompt
    except Exception as e:
        # В случае ошибки возвращаем базовое улучшение промпта
        return (
            f"A cute, modern illustration: {user_prompt}, "
            "minimalist style, suitable for a personal finance app, "
            "warm colors, friendly design, digital art style."
        )


def generate_image_with_dalle(
    openai_client: OpenAI,
    prompt: str,
    size: str = "1024x1024",
) -> str:
    """Генерирует изображение через DALL-E и возвращает URL.
    
    Args:
        openai_client: Клиент OpenAI для работы с API.
        prompt: Промпт для генерации (уже улучшенный, на английском).
        size: Размер изображения. DALL-E 3 поддерживает "1024x1024" или "1792x1024".
    
    Returns:
        URL сгенерированного изображения.
    
    Raises:
        Exception: Если генерация не удалась.
    """
    response = openai_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,  # DALL-E 3 не поддерживает 512x512, минимальный - 1024x1024
        quality="standard",
        n=1,
    )
    
    return response.data[0].url


def save_image_from_url(
    image_url: str,
    user_id: int,
    frontend_dir: Path,
) -> str:
    """Скачивает изображение по URL и сохраняет его на диск.
    
    Args:
        image_url: URL изображения для скачивания.
        user_id: ID пользователя (используется в имени файла).
        frontend_dir: Путь к директории frontend (где находится папка user_images).
    
    Returns:
        Относительный путь к сохранённому файлу (относительно директории static),
        например "user_images/123.png".
    
    Raises:
        Exception: Если скачивание или сохранение не удалось.
    """
    # Скачиваем картинку
    with httpx.Client(timeout=30.0) as http_client:
        img_response = http_client.get(image_url)
        img_response.raise_for_status()
        
        # Создаём директорию для картинок пользователей, если её нет
        user_images_dir = frontend_dir / "user_images"
        user_images_dir.mkdir(exist_ok=True)
        
        # Сохраняем картинку
        image_filename = f"{user_id}.png"
        image_path = user_images_dir / image_filename
        image_path.write_bytes(img_response.content)
        
        # Возвращаем относительный путь (относительно static)
        return f"user_images/{image_filename}"


def generate_and_save_user_image(
    user_prompt: str,
    user_id: int,
    frontend_dir: Path,
    openai_api_key: Optional[str] = None,
) -> str:
    """Полный цикл генерации картинки: улучшение промпта, генерация, сохранение.
    
    Args:
        user_prompt: Промпт от пользователя (на русском или английском).
        user_id: ID пользователя в БД (для имени файла).
        frontend_dir: Путь к директории frontend.
        openai_api_key: API ключ OpenAI. Если не указан, берётся из переменной окружения.
    
    Returns:
        Относительный путь к сохранённому файлу (относительно директории static).
    
    Raises:
        ValueError: Если API ключ не найден.
        Exception: Если генерация или сохранение не удались.
    """
    # Получаем API ключ
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY не настроен")
    
    # Создаём клиент OpenAI
    openai_client = OpenAI(api_key=openai_api_key)
    
    # Улучшаем промпт
    improved_prompt = improve_prompt_with_chatgpt(
        openai_client=openai_client,
        user_prompt=user_prompt,
    )
    
    # Генерируем картинку
    image_url = generate_image_with_dalle(
        openai_client=openai_client,
        prompt=improved_prompt,
    )
    
    # Сохраняем картинку
    relative_path = save_image_from_url(
        image_url=image_url,
        user_id=user_id,
        frontend_dir=frontend_dir,
    )
    
    return relative_path
