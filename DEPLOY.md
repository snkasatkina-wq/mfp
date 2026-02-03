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

Вставь (если работаешь от root, используй пути ниже; если от другого пользователя — замени `root` на своего):
```ini
[Unit]
Description=MVP Cost Backend
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/mvp-cost/backend
Environment="PATH=/root/mvp-cost/venv/bin"
ExecStart=/root/mvp-cost/venv/bin/python run_prod.py
Restart=always
EnvironmentFile=/root/mvp-cost/.env

[Install]
WantedBy=multi-user.target
```

**Бот:**
```bash
sudo nano /etc/systemd/system/mvp-bot.service
```

Вставь (если работаешь от root, используй пути ниже):
```ini
[Unit]
Description=MVP Cost Telegram Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/mvp-cost/telegram_bot
Environment="PATH=/root/mvp-cost/venv/bin"
ExecStart=/root/mvp-cost/venv/bin/python -m bot.main
Restart=always
EnvironmentFile=/root/mvp-cost/.env

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

**Если у тебя уже есть конфиг с SSL (Certbot), просто дополни его:**

```bash
sudo nano /etc/nginx/sites-available/mvp.kasven.ru
```

В блоке `server { listen 443 ssl; ... }` внутри `location / { ... }` добавь строки (если их ещё нет):

```nginx
server {
    server_name mvp.kasven.ru;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Дополнительные настройки (добавь эти строки):
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    listen 443 ssl; # managed by Certbot
    ssl_certificate /etc/letsencrypt/live/mvp.kasven.ru/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mvp.kasven.ru/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
}
```

**Проверь и перезагрузи:**
```bash
sudo nginx -t  # проверка конфига
sudo systemctl reload nginx
```

**Если конфига ещё нет, создай новый** (но судя по твоему вопросу, он уже есть).

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
