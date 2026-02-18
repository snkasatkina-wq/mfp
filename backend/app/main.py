"""Основной модуль FastAPI‑бекенда.

Здесь описаны:
- создание приложения FastAPI;
- Pydantic‑схемы запросов/ответов;
- HTTP‑эндпоинты для пользователей, линковки устройств, очереди расходов;
- эндпоинты для мини‑приложения и статики;
- фоновый воркер, который периодически сбрасывает очередь расходов в БД.
"""

from datetime import datetime
from pathlib import Path
import secrets
import time
import asyncio
import os
from typing import Optional, Any

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text

from .database import SessionLocal, init_db
from . import crud
from .models import User
from . import image_generation
from . import receipt_processing

app = FastAPI(title="Expense Tracker Backend", version="0.1.0")


def get_db() -> Session:
    """Выдаёт сессию БД для одного HTTP‑запроса (FastAPI‑dependency)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ===================== Схемы =====================

class LinkCodeResponse(BaseModel):
    """Ответ при создании одноразового кода привязки устройства."""

    device_id: str
    code: str
    expires_in_seconds: int


class UserCreate(BaseModel):
    """Тело запроса для создания/поиска пользователя по Telegram."""

    telegram_user_id: int
    telegram_chat_id: int


class UserOut(BaseModel):
    id: int
    telegram_user_id: int
    telegram_chat_id: int
    email: str | None
    threshold_percent: int

    class Config:
        # Позволяет возвращать напрямую ORM‑объект User из SQLAlchemy
        from_attributes = True


class LinkConfirmRequest(BaseModel):
    """Тело запроса для подтверждения привязки чата по коду."""
    telegram_chat_id: str
    code: str


class LinkConfirmResponse(BaseModel):
    """Ответ на подтверждение привязки чата."""
    ok: bool
    device_id: str


class EnqueueExpenseRequest(BaseModel):
    """Тело запроса от Telegram‑бота для помещения расхода в очередь.

    Дата/время передаётся строкой в формате \"DD.MM.YYYY HH:MM\" или None,
    если нужно использовать текущее время.
    """
    telegram_chat_id: str
    amount_cents: int
    description: str
    category: str
    subcategory: str | None = None
    # строка в формате "DD.MM.YYYY HH:MM" или None (тогда используем текущее время)
    occurred_at: str | None = None


class EnqueueExpenseResponse(BaseModel):
    """Ответ при постановке расхода в очередь на стороне бекенда."""
    ok: bool
    device_id: str
    queued_count: int


class ExpenseCreateRequest(BaseModel):
    """Запрос на создание расхода от бота."""
    telegram_user_id: int
    telegram_chat_id: int
    amount_cents: int
    description: str
    category: str
    subcategory: str | None = None
    occurred_at: str | None = None  # формат "DD.MM.YYYY HH:MM" или None


class GenerateImageRequest(BaseModel):
    """Запрос на генерацию картинки с промптом от пользователя."""

    prompt: str


class MiniappCategoriesRequest(BaseModel):
    """Запрос на создание категорий из мини‑приложения.

    Поле `categories` — одна строка с перечислением категорий через запятую.
    """

    categories: str


# ===================== Health =====================

@app.get("/health")
def health():
    """Простой health‑чек, чтобы убедиться, что сервис жив."""
    return {"status": "ok"}


# ===================== /users =====================

@app.post("/users", response_model=UserOut)
def create_or_get_user(
    payload: UserCreate,
    db: Session = Depends(get_db),
) -> UserOut:
    """Создаёт пользователя по Telegram ID или возвращает уже существующего."""
    user = crud.get_or_create_user(
        db=db,
        telegram_user_id=payload.telegram_user_id,
        telegram_chat_id=payload.telegram_chat_id,
    )
    return user

@app.post("/expenses")
def create_expense(
    payload: ExpenseCreateRequest,
    db: Session = Depends(get_db),
):
    """Создаёт расход сразу в БД (без очереди).
    
    Используется ботом для прямого сохранения расходов.
    """
    # Находим или создаём пользователя
    user = crud.get_or_create_user(
        db=db,
        telegram_user_id=payload.telegram_user_id,
        telegram_chat_id=payload.telegram_chat_id,
    )
    
    # Конвертируем дату/время
    if payload.occurred_at:
        occurred_dt = datetime.strptime(payload.occurred_at, "%d.%m.%Y %H:%M")
    else:
        occurred_dt = datetime.utcnow()
    
    # Создаём расход
    expense = crud.create_expense(
        db=db,
        user_id=user.id,
        amount_cents=payload.amount_cents,
        description=payload.description,
        category_name=payload.category,
        subcategory_name=payload.subcategory,
        occurred_at=occurred_dt,
        source="telegram",
    )
    
    return {"ok": True, "expense_id": expense.id}


@app.post("/miniapp/get-user-by-context", response_class=JSONResponse)
def get_user_by_context(payload: dict[str, Any], db: Session = Depends(get_db)) -> dict[str, Any]:
    """Пытается определить пользователя по контексту WebApp, если initData недоступен.
    
    Это fallback метод, когда Telegram не передаёт initData.
    """
    init_data_unsafe = payload.get("initDataUnsafe", {})
    
    # Пробуем получить user из initDataUnsafe
    user_data = init_data_unsafe.get("user")
    if user_data and user_data.get("id"):
        return {
            "telegram_user_id": user_data.get("id"),
            "first_name": user_data.get("first_name", ""),
            "last_name": user_data.get("last_name", ""),
        }
    
    # Если не получилось, возвращаем ошибку
    raise HTTPException(status_code=400, detail="Не удалось определить пользователя из контекста")


@app.post("/miniapp/user-info", response_class=JSONResponse)
def get_user_info_from_init_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Парсит initData от Telegram WebApp и возвращает информацию о пользователе.
    
    Используется как fallback, если фронтенд не может распарсить initData.
    """
    import urllib.parse
    import json
    
    init_data = payload.get("initData", "")
    if not init_data:
        raise HTTPException(status_code=400, detail="initData не предоставлен")
    
    try:
        # Парсим query string
        # initData может быть строкой вида "user=...&auth_date=..."
        params = urllib.parse.parse_qs(init_data)
        user_str = params.get("user", [None])[0]
        
        if not user_str:
            # Пробуем альтернативный формат - может быть уже декодировано
            # или в другом формате
            raise ValueError("Параметр 'user' не найден в initData")
        
        # Декодируем и парсим JSON
        user_data = json.loads(urllib.parse.unquote(user_str))
        
        telegram_user_id = user_data.get("id")
        if not telegram_user_id:
            raise ValueError("telegram_user_id не найден в данных пользователя")
        
        return {
            "telegram_user_id": telegram_user_id,
            "first_name": user_data.get("first_name", ""),
            "last_name": user_data.get("last_name", ""),
        }
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Ошибка парсинга JSON из initData: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка обработки initData: {str(e)}")


@app.get("/miniapp/user-image", response_class=JSONResponse)
def get_user_image(
    telegram_user_id: int,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Возвращает информацию о картинке пользователя для мини‑приложения.
    
    Если у пользователя есть сгенерированная картинка, возвращает её путь,
    иначе возвращает путь к дефолтной картинке.
    """
    user = (
        db.query(User)
        .filter(User.telegram_user_id == telegram_user_id)
        .first()
    )
    
    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    
    # Если есть сгенерированная картинка, возвращаем её
    if user.custom_image_path:
        return {
            "image_url": f"/static/{user.custom_image_path}",
            "is_custom": True,
        }
    
    # Иначе возвращаем дефолтную
    return {
        "image_url": "/static/piggy.png",
        "is_custom": False,
    }


@app.post("/miniapp/categories", response_class=JSONResponse)
def miniapp_create_categories(
    telegram_user_id: int,
    payload: MiniappCategoriesRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Создаёт категории для пользователя из мини‑приложения.

    Пользователь передаёт строку с категориями, разделёнными запятыми.
    По умолчанию также создаётся категория «не определено».
    """
    # Находим пользователя по telegram_user_id
    user = (
        db.query(User)
        .filter(User.telegram_user_id == telegram_user_id)
        .first()
    )

    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    # Гарантируем наличие базовой категории «не определено»
    base_category_name = "не определено"
    crud.get_or_create_category(
        db=db,
        user_id=user.id,
        name=base_category_name,
    )

    raw = (payload.categories or "").strip()
    if not raw:
        raise HTTPException(
            status_code=400,
            detail="Строка категорий пуста",
        )

    parts = [p.strip() for p in raw.split(",")]
    created_or_existing: list[str] = []

    for name in parts:
        if not name:
            continue
        # Не дублируем базовую категорию, она уже создана выше
        if name.lower() == base_category_name.lower():
            continue

        cat = crud.get_or_create_category(
            db=db,
            user_id=user.id,
            name=name,
        )
        created_or_existing.append(cat.name)

    return {
        "ok": True,
        "categories": created_or_existing or [base_category_name],
    }


@app.get("/miniapp/categories/list", response_class=JSONResponse)
def miniapp_list_categories(
    telegram_user_id: int,
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """Возвращает все категории пользователя для мини‑приложения."""
    user = (
        db.query(User)
        .filter(User.telegram_user_id == telegram_user_id)
        .first()
    )

    if not user:
        return []

    categories = crud.get_categories(db=db, user_id=user.id)
    return [
        {"id": c.id, "name": c.name}
        for c in categories
    ]


@app.get("/miniapp/expenses/by-category", response_class=JSONResponse)
def miniapp_expenses_by_category(
    telegram_user_id: int,
    category_name: str,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """Возвращает расходы пользователя по конкретной категории для мини‑приложения."""
    # Находим пользователя по telegram_user_id
    user = (
        db.query(User)
        .filter(User.telegram_user_id == telegram_user_id)
        .first()
    )

    if not user:
        return []

    rows = (
        db.execute(
            text(
                """
            SELECT e.id,
                   e.amount_cents,
                   e.description,
                   e.occurred_at,
                   c.name AS category_name
            FROM expenses e
            LEFT JOIN categories c ON c.id = e.category_id
            WHERE e.user_id = :user_id
              AND c.name = :category_name
            ORDER BY e.occurred_at DESC, e.id DESC
            LIMIT :limit
            """
            ),
            {"user_id": user.id, "category_name": category_name, "limit": limit},
        )
        .mappings()
        .all()
    )

    result: list[dict[str, Any]] = []
    for r in rows:
        item = dict(r)
        occurred = item.get("occurred_at")
        if isinstance(occurred, datetime):
            item["occurred_at"] = occurred.strftime("%d.%m.%Y %H:%M")
        result.append(item)
    return result


@app.post("/miniapp/receipts")
async def upload_receipt(
    telegram_user_id: int,
    file: UploadFile = File(...),
):
    """Обрабатывает загруженный чек: парсит через OpenAI Vision, категоризирует позиции и создаёт расходы.
    
    Возвращает статусы обработки через Server-Sent Events (SSE):
    - "чек загружен"
    - "чек распознан"
    - "категории определены"
    - "расходы внесены в базу"
    
    Логика:
    1. Находит пользователя по telegram_user_id.
    2. Сохраняет файл чека на сервер.
    3. Парсит чек через OpenAI Vision (извлекает позиции и суммы).
    4. Получает список категорий пользователя (включая "не определено").
    5. Для каждой позиции определяет категорию через OpenAI Chat.
    6. Создаёт запись в таблице receipts.
    7. Создаёт записи расходов в таблице expenses для каждой позиции.
    """
    import uuid
    import json as json_lib
    
    # Читаем файл один раз до создания генератора (UploadFile можно прочитать только один раз)
    file_bytes = await file.read()
    file_content_type = file.content_type
    file_filename = file.filename
    
    def process_receipt_with_status():
        """Генератор, который отправляет статусы обработки чека."""
        # Создаём отдельную сессию БД для генератора, чтобы она оставалась открытой
        db = SessionLocal()
        try:
            # Находим пользователя
            user = crud.get_user_by_telegram_id(db=db, telegram_user_id=telegram_user_id)
            if not user:
                yield f"data: {json_lib.dumps({'status': 'error', 'message': 'Пользователь не найден'})}\n\n"
                return
            
            # Проверяем формат файла
            if not file_content_type or not file_content_type.startswith("image/"):
                yield f"data: {json_lib.dumps({'status': 'error', 'message': 'Файл должен быть изображением (jpg, png, webp и т.д.)'})}\n\n"
                return
            
            # Определяем MIME-тип для OpenAI
            mime_type = file_content_type
            if mime_type == "image/jpg":
                mime_type = "image/jpeg"
            
            # Генерируем уникальный receipt_group_id
            receipt_group_id = str(uuid.uuid4())
            
            # Определяем путь для сохранения файла
            BASE_DIR = Path(__file__).resolve().parent.parent
            RECEIPTS_DIR = BASE_DIR / "backend" / "receipts" / str(user.id)
            RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Определяем расширение файла
            file_ext = Path(file_filename or "receipt.jpg").suffix
            if not file_ext:
                file_ext = ".jpg"
            
            file_path = RECEIPTS_DIR / f"{receipt_group_id}{file_ext}"
            
            # Сохраняем файл
            try:
                file_path.write_bytes(file_bytes)
            except Exception as e:
                yield f"data: {json_lib.dumps({'status': 'error', 'message': f'Ошибка сохранения файла: {str(e)}'})}\n\n"
                return
            
            # Статус: чек загружен
            yield f"data: {json_lib.dumps({'status': 'uploaded', 'message': 'Чек загружен'})}\n\n"
            
            # Относительный путь для БД (от корня проекта)
            relative_file_path = f"backend/receipts/{user.id}/{receipt_group_id}{file_ext}"
            
            # Получаем API ключ OpenAI
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                yield f"data: {json_lib.dumps({'status': 'error', 'message': 'OPENAI_API_KEY не настроен'})}\n\n"
                return
            
            client = OpenAI(api_key=openai_api_key)
            
            # Загружаем промпты
            parsing_prompt = receipt_processing.load_receipt_parsing_prompt()
            classification_prompt = receipt_processing.load_category_classification_prompt()
            
            # Парсим чек
            try:
                parsed_result = receipt_processing.parse_receipt_image(
                    client=client,
                    prompt_text=parsing_prompt,
                    image_bytes=file_bytes,
                    mime_type=mime_type,
                )
            except Exception as e:
                # Удаляем сохранённый файл при ошибке парсинга
                try:
                    file_path.unlink()
                except:
                    pass
                yield f"data: {json_lib.dumps({'status': 'error', 'message': f'Ошибка парсинга чека: {str(e)}'})}\n\n"
                return
            
            # Статус: чек распознан
            yield f"data: {json_lib.dumps({'status': 'parsed', 'message': 'Чек распознан'})}\n\n"
            
            items = parsed_result.get("items", [])
            if not items:
                # Удаляем сохранённый файл, если позиций не найдено
                try:
                    file_path.unlink()
                except:
                    pass
                yield f"data: {json_lib.dumps({'status': 'error', 'message': 'На чеке не найдено позиций для обработки'})}\n\n"
                return
            
            # Получаем категории пользователя (включая "не определено")
            categories = crud.get_categories(db=db, user_id=user.id)
            category_names = [c.name for c in categories]
            
            # Убеждаемся, что категория "не определено" существует
            if "не определено" not in category_names:
                crud.get_or_create_category(db=db, user_id=user.id, name="не определено")
                category_names.append("не определено")
            
            # Статус: категории определены
            yield f"data: {json_lib.dumps({'status': 'categories_ready', 'message': 'Категории определены'})}\n\n"
            
            # Классифицируем каждую позицию и создаём расходы
            created_count = 0
            current_time = datetime.now()
            
            try:
                for item in items:
                    item_name = item.get("name", item.get("raw_name", "Неизвестная позиция"))
                    total_price = float(item.get("total_price", 0.0))
                    
                    if total_price <= 0:
                        continue  # Пропускаем позиции с нулевой или отрицательной ценой
                    
                    # Определяем категорию
                    try:
                        category_name = receipt_processing.classify_item(
                            client=client,
                            prompt_text=classification_prompt,
                            item_name=item_name,
                            categories=category_names,
                        )
                        # Проверяем, что категория действительно в списке
                        if category_name not in category_names:
                            category_name = "не определено"
                    except Exception as e:
                        # В случае ошибки категоризации используем "не определено"
                        category_name = "не определено"
                    
                    # Создаём расход
                    amount_cents = int(round(total_price * 100))
                    crud.create_expense_from_receipt_item(
                        db=db,
                        user_id=user.id,
                        amount_cents=amount_cents,
                        description=item_name,
                        category_name=category_name,
                        receipt_group_id=receipt_group_id,
                        occurred_at=current_time,
                    )
                    created_count += 1
                
                # Создаём запись о чеке в таблице receipts
                crud.create_receipt(
                    db=db,
                    user_id=user.id,
                    receipt_group_id=receipt_group_id,
                    file_path=relative_file_path,
                )
                
                # Статус: расходы внесены в базу
                yield f"data: {json_lib.dumps({'status': 'completed', 'message': 'Расходы внесены в базу', 'created': created_count})}\n\n"
                
            except Exception as e:
                # В случае ошибки удаляем сохранённый файл
                try:
                    file_path.unlink()
                except:
                    pass
                yield f"data: {json_lib.dumps({'status': 'error', 'message': f'Ошибка обработки чека: {str(e)}'})}\n\n"
                
        except Exception as e:
            yield f"data: {json_lib.dumps({'status': 'error', 'message': f'Неожиданная ошибка: {str(e)}'})}\n\n"
        finally:
            # Закрываем сессию БД
            db.close()
    
    return StreamingResponse(
        process_receipt_with_status(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


class GenerateImageRequest(BaseModel):
    """Запрос на генерацию картинки с промптом от пользователя."""
    prompt: str


@app.post("/miniapp/generate-image", response_class=JSONResponse)
def generate_user_image(
    telegram_user_id: int,
    payload: GenerateImageRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Генерирует картинку для пользователя через OpenAI DALL-E и сохраняет её.
    
    Принимает промпт от пользователя, улучшает его через ChatGPT,
    затем генерирует картинку через DALL-E.
    
    Возвращает URL сгенерированной картинки.
    """
    # Валидация длины промпта
    user_prompt = payload.prompt.strip()
    if len(user_prompt) > 200:
        raise HTTPException(
            status_code=400,
            detail="Промпт слишком длинный. Максимум 200 символов."
        )
    
    if not user_prompt:
        raise HTTPException(
            status_code=400,
            detail="Промпт не может быть пустым"
        )
    
    # Находим пользователя
    user = (
        db.query(User)
        .filter(User.telegram_user_id == telegram_user_id)
        .first()
    )
    
    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    
    # Определяем путь к директории frontend
    BASE_DIR = Path(__file__).resolve().parent.parent
    FRONTEND_DIR = BASE_DIR.parent / "frontend"
    
    try:
        # Генерируем и сохраняем картинку
        relative_path = image_generation.generate_and_save_user_image(
            user_prompt=user_prompt,
            user_id=user.id,
            frontend_dir=FRONTEND_DIR,
        )
        
        # Обновляем путь в БД
        crud.update_user_custom_image_path(
            db=db,
            user_id=user.id,
            custom_image_path=relative_path,
        )
        
        return {
            "ok": True,
            "image_url": f"/static/{relative_path}",
        }
            
    except ValueError as e:
        # Ошибка конфигурации (например, отсутствует API ключ)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка генерации картинки: {str(e)}"
        )


@app.get("/miniapp/expenses", response_class=JSONResponse)
def miniapp_expenses(
    telegram_user_id: int,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """Возвращает последние расходы конкретного пользователя для мини‑приложения.

    Дата/время `occurred_at` конвертируется в строку формата \"DD.MM.YYYY HH:MM\"
    для отображения пользователю.
    """
    # Находим пользователя по telegram_user_id
    user = (
        db.query(User)
        .filter(User.telegram_user_id == telegram_user_id)
        .first()
    )
    
    if not user:
        return []
    
    rows = (
        db.execute(
            text("""
            SELECT e.id,
                   e.amount_cents,
                   e.description,
                   e.occurred_at,
                   c.name AS category_name
            FROM expenses e
            LEFT JOIN categories c ON c.id = e.category_id
            WHERE e.user_id = :user_id
            ORDER BY e.occurred_at DESC, e.id DESC
            LIMIT :limit
            """),
            {"user_id": user.id, "limit": limit},
        )
        .mappings()
        .all()
    )
    result: list[dict[str, Any]] = []
    for r in rows:
        item = dict(r)
        occurred = item.get("occurred_at")
        if isinstance(occurred, datetime):
            item["occurred_at"] = occurred.strftime("%d.%m.%Y %H:%M")
        result.append(item)
    return result


# ===================== Линковка устройства =====================

# Временное in‑memory хранилище одноразовых кодов линковки (MVP‑реализация).
# Структура: device_id -> {"code": str, "expires_at": int}
LINK_CODES: dict[str, dict[str, int | str]] = {}


@app.post("/link/code", response_model=LinkCodeResponse)
def create_link_code(device_id: str):
    """Создаёт одноразовый код привязки для указанного device_id.

    Код живёт ограниченное время (TTL) и хранится в памяти процесса.
    """
    code = secrets.token_urlsafe(4)[:6].upper()
    ttl = 10 * 60  # 10 минут
    expires_at = int(time.time()) + ttl

    LINK_CODES[device_id] = {"code": code, "expires_at": expires_at}

    return LinkCodeResponse(
        device_id=device_id,
        code=code,
        expires_in_seconds=ttl,
    )


@app.post("/link/confirm", response_model=LinkConfirmResponse)
def confirm_link(req: LinkConfirmRequest):
    """Подтверждает привязку чата по одноразовому коду.

    Если код найден и ещё не истёк, создаётся/обновляется запись
    связи `telegram_chat_id -> device_id` в PostgreSQL.
    """
    now = int(time.time())

    matched_device_id: str | None = None
    for device_id, payload in LINK_CODES.items():
        if payload["code"] == req.code and payload["expires_at"] >= now:
            matched_device_id = device_id
            break

    if matched_device_id is None:
        # для простоты MVP — просто ok=false
        return LinkConfirmResponse(ok=False, device_id="")

    # сохраняем привязку в PostgreSQL
    with SessionLocal() as db:
        crud.upsert_chat_device_link(
            db=db,
            telegram_chat_id=int(req.telegram_chat_id),
            device_id=matched_device_id,
        )

    # делаем код одноразовым
    del LINK_CODES[matched_device_id]

    return LinkConfirmResponse(ok=True, device_id=matched_device_id)


# ===================== Очередь расходов =====================

# Очередь во временной памяти процесса: device_id -> [список расходов].
# Время от времени фоновый воркер сбрасывает эти данные в основную БД.
QUEUE: dict[str, list[dict]] = {}


@app.post("/queue/expense", response_model=EnqueueExpenseResponse)
def enqueue_expense(req: EnqueueExpenseRequest):
    """Кладёт расход от бота во временную очередь по связанному device_id.

    Здесь не происходит записи в основную БД, только подготовка данных
    для последующего flush‑а (ручного или автоматического).
    """
    # ищем device_id, привязанный к этому чату
    with SessionLocal() as db:
        device_id = crud.get_device_id_for_chat(
            db=db,
            telegram_chat_id=int(req.telegram_chat_id),
        )
    if not device_id:
        return EnqueueExpenseResponse(ok=False, device_id="", queued_count=0)

    # конвертируем дату/время из пользовательского формата в datetime
    if req.occurred_at:
        # ожидаем формат "DD.MM.YYYY HH:MM"
        occurred_dt = datetime.strptime(req.occurred_at, "%d.%m.%Y %H:%M")
    else:
        occurred_dt = datetime.utcnow()

    item = {
        "amount_cents": req.amount_cents,
        "description": req.description,
        "category": req.category,
        "subcategory": req.subcategory,
        "occurred_at": occurred_dt,
        "source": "telegram",
    }

    QUEUE.setdefault(device_id, []).append(item)

    return EnqueueExpenseResponse(
        ok=True,
        device_id=device_id,
        queued_count=len(QUEUE[device_id]),
    )


# ===================== Ручной flush (debug) =====================

@app.post("/debug/flush-queue")
def flush_queue(
    telegram_chat_id: str,
    db: Session = Depends(get_db),
):
    """Ручной сброс очереди расходов для конкретного Telegram‑чата.

    Используется для отладки: берёт все накопленные в QUEUE расходы,
    привязанные к этому чату, и записывает их в таблицу `expenses`.
    """
    with SessionLocal() as tmp_db:
        device_id = crud.get_device_id_for_chat(
            db=tmp_db,
            telegram_chat_id=int(telegram_chat_id),
        )
    if not device_id:
        raise HTTPException(
            status_code=400,
            detail="Нет привязки для этого telegram_chat_id",
        )

    items = QUEUE.get(device_id, [])
    if not items:
        return {"ok": True, "written": 0}

    # находим / создаём пользователя по chat_id
    user = crud.get_or_create_user(
        db=db,
        telegram_user_id=int(telegram_chat_id),  # если user_id = chat_id
        telegram_chat_id=int(telegram_chat_id),
    )

    written = 0
    for item in items:
        crud.create_expense(
            db=db,
            user_id=user.id,
            amount_cents=item["amount_cents"],
            description=item["description"],
            category_name=item["category"],
            subcategory_name=item["subcategory"],
            occurred_at=item["occurred_at"],
            source=item["source"],
        )
        written += 1

    # очищаем корзинку для этого device_id
    QUEUE[device_id] = []

    return {"ok": True, "written": written}


# ===================== Авто-flush в фоне =====================

async def auto_flush_worker(interval_seconds: int = 10):
    """Фоновый воркер, который периодически сбрасывает очередь расходов в БД.

    Раз в `interval_seconds` секунд обходится очередь QUEUE, и все накопленные
    расходы для каждого device_id переносятся в таблицу `expenses`.
    """
    from .database import SessionLocal  # локальный импорт против циклов

    while True:
        if not QUEUE:
            await asyncio.sleep(interval_seconds)
            continue

        # копия на момент начала цикла
        snapshot = list(QUEUE.items())  # [(device_id, [items])]

        for device_id, items in snapshot:
            if not items:
                continue

            # находим telegram_chat_id по device_id в БД
            db = SessionLocal()
            try:
                telegram_chat_id = crud.get_chat_id_for_device(db=db, device_id=device_id)
                if telegram_chat_id is None:
                    continue

                user = crud.get_or_create_user(
                    db=db,
                    telegram_user_id=int(telegram_chat_id),
                    telegram_chat_id=int(telegram_chat_id),
                )

                for item in items:
                    crud.create_expense(
                        db=db,
                        user_id=user.id,
                        amount_cents=item["amount_cents"],
                        description=item["description"],
                        category_name=item["category"],
                        subcategory_name=item["subcategory"],
                        occurred_at=item["occurred_at"],
                        source=item["source"],
                    )

                QUEUE[device_id] = []
            finally:
                db.close()

        await asyncio.sleep(interval_seconds)


@app.on_event("startup")
async def start_auto_flush_worker():
    """Хук FastAPI, вызываемый при старте приложения.

    Здесь мы:
    - создаём таблицы в БД, если их ещё нет (`init_db`);
    - запускаем фоновую задачу `auto_flush_worker`.
    """
    init_db()
    asyncio.create_task(auto_flush_worker(interval_seconds=10))


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/miniapp", response_class=HTMLResponse)
def miniapp_page():
    index_path = FRONTEND_DIR / "index.html"
    with index_path.open("r", encoding="utf-8") as f:
        return f.read()
