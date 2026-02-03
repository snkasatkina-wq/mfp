"""Точка входа для запуска бекенда в продакшене.

Запускает Uvicorn без reload, с несколькими воркерами для обработки запросов.
"""

import uvicorn

if __name__ == "__main__":
    # Продакшен-режим: без reload, несколько воркеров
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",  # слушаем только localhost, Nginx проксирует
        port=8000,
        workers=2,  # можно увеличить, если нужно
        log_level="info",
    )
