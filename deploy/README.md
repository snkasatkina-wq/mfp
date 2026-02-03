# Инструкция по деплою на сервер

## Подготовка

1. **Подключись к серверу по SSH:**
   ```bash
   ssh user@5.35.92.150
   ```

2. **Установи зависимости (если ещё не установлены):**
   ```bash
   # Python 3 и pip должны быть установлены
   python3 --version
   pip3 --version
   ```

## Деплой кода

1. **Склонируй репозиторий (или загрузи код через git/scp):**
   ```bash
   cd ~
   git clone <твой-репозиторий> mvp-cost
   # или загрузи код вручную в ~/mvp-cost
   ```

2. **Создай виртуальное окружение:**
   ```bash
   cd ~/mvp-cost
   python3 -m venv venv
   source venv/bin/activate  # для bash
   # или
   . venv/bin/activate  # для sh
   ```

3. **Установи зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Создай файл `.env` с переменными окружения:**
   ```bash
   cp .env.example .env
   nano .env  # или vi .env
   ```
   
   Заполни все переменные:
   - `DATABASE_URL` — строка подключения к PostgreSQL на бегете
   - `BACKEND_URL=https://mvp.kasven.ru`
   - `TELEGRAM_BOT_TOKEN` — токен бота из BotFather
   - `OPENAI_API_KEY` — ключ OpenAI (если используешь голосовые)

## Настройка systemd (автозапуск)

1. **Создай systemd-юнит для бэкенда:**
   ```bash
   sudo nano /etc/systemd/system/mvp-backend.service
   ```
   
   Вставь содержимое из `deploy/mvp-backend.service` (см. ниже)

2. **Создай systemd-юнит для бота:**
   ```bash
   sudo nano /etc/systemd/system/mvp-bot.service
   ```
   
   Вставь содержимое из `deploy/mvp-bot.service` (см. ниже)

3. **Перезагрузи systemd и запусти сервисы:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable mvp-backend
   sudo systemctl enable mvp-bot
   sudo systemctl start mvp-backend
   sudo systemctl start mvp-bot
   ```

4. **Проверь статус:**
   ```bash
   sudo systemctl status mvp-backend
   sudo systemctl status mvp-bot
   ```

## Настройка Nginx

1. **Создай конфиг для домена:**
   ```bash
   sudo nano /etc/nginx/sites-available/mvp.kasven.ru
   ```
   
   Вставь содержимое из `deploy/nginx.conf` (см. ниже)

2. **Активируй конфиг:**
   ```bash
   sudo ln -s /etc/nginx/sites-available/mvp.kasven.ru /etc/nginx/sites-enabled/
   sudo nginx -t  # проверка конфига
   sudo systemctl reload nginx
   ```

## Полезные команды

**Просмотр логов:**
```bash
sudo journalctl -u mvp-backend -f
sudo journalctl -u mvp-bot -f
```

**Перезапуск сервисов:**
```bash
sudo systemctl restart mvp-backend
sudo systemctl restart mvp-bot
```

**Остановка:**
```bash
sudo systemctl stop mvp-backend
sudo systemctl stop mvp-bot
```
