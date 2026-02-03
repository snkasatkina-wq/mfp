"""Точка входа для запуска бекенда в режиме разработки.

Запускает Uvicorn с включённым `reload=True`, чтобы при изменении кода
приложение автоматически перезапускалось.
"""

import uvicorn

if __name__ == "__main__":
    # Запускаем FastAPI‑приложение `app.main:app` на localhost:8000
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
