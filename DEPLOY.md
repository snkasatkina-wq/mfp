# Быстрая инструкция по деплою

## 1. Подготовка на сервере

```bash
# Подключись к серверу
ssh user@5.35.92.150

# Создай директорию для проекта
mkdir -p ~/mvp-cost
cd ~/mvp-cost
```

## 2. Загрузи код на сервер

**Вариант А: через git (если репозиторий есть)**
```bash
git clone <твой-репозиторий> .
```

**Вариант Б: через scp (с локального компьютера)**
```bash
# На локальной машине:
scp -r backend telegram_bot frontend requirements.txt user@5.35.92.150:~/mvp-cost/
```

## 3. Установка зависимостей

```bash
cd ~/mvp-cost
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 4. Настройка переменных окружения

```bash
nano ~/mvp-cost/.env
```

Вставь (замени значения на свои):
```
DATABASE_URL=postgresql://mvp_user:ПАРОЛЬ@localhost:5432/mvp_costs
BACKEND_URL=https://mvp.kasven.ru
TELEGRAM_BOT_TOKEN=1234567890:ABC...
OPENAI_API_KEY=sk-...
```

## 5. Настройка systemd

**Бэкенд:**
```bash
sudo nano /etc/systemd/system/mvp-backend.service
```

Вставь (замени `username` на своего пользователя):
```ini
[Unit]
Description=MVP Cost Backend
After=network.target

[Service]
Type=simple
User=username
WorkingDirectory=/home/username/mvp-cost/backend
Environment="PATH=/home/username/mvp-cost/venv/bin"
ExecStart=/home/username/mvp-cost/venv/bin/python run_prod.py
Restart=always
EnvironmentFile=/home/username/mvp-cost/.env

[Install]
WantedBy=multi-user.target
```

**Бот:**
```bash
sudo nano /etc/systemd/system/mvp-bot.service
```

Вставь:
```ini
[Unit]
Description=MVP Cost Telegram Bot
After=network.target

[Service]
Type=simple
User=username
WorkingDirectory=/home/username/mvp-cost/telegram_bot
Environment="PATH=/home/username/mvp-cost/venv/bin"
ExecStart=/home/username/mvp-cost/venv/bin/python -m bot.main
Restart=always
EnvironmentFile=/home/username/mvp-cost/.env

[Install]
WantedBy=multi-user.target
```

**Запуск:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable mvp-backend mvp-bot
sudo systemctl start mvp-backend mvp-bot
```

## 6. Настройка Nginx

```bash
sudo nano /etc/nginx/sites-available/mvp.kasven.ru
```

Вставь (замени пути к SSL-сертификатам):
```nginx
server {
    listen 443 ssl http2;
    server_name mvp.kasven.ru;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Активируй:
```bash
sudo ln -s /etc/nginx/sites-available/mvp.kasven.ru /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## 7. Проверка

```bash
# Проверь статус сервисов
sudo systemctl status mvp-backend
sudo systemctl status mvp-bot

# Проверь логи
sudo journalctl -u mvp-backend -f
sudo journalctl -u mvp-bot -f

# Проверь, что бэкенд отвечает
curl https://mvp.kasven.ru/health
```

## Полезные команды

```bash
# Перезапуск
sudo systemctl restart mvp-backend
sudo systemctl restart mvp-bot

# Остановка
sudo systemctl stop mvp-backend
sudo systemctl stop mvp-bot

# Просмотр логов
sudo journalctl -u mvp-backend -n 50
sudo journalctl -u mvp-bot -n 50
```
