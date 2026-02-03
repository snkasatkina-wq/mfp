import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).with_name("backend_state.sqlite")

def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS links (
                telegram_chat_id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO links (telegram_chat_id, device_id)
            VALUES (?, ?)
            ON CONFLICT(telegram_chat_id) DO UPDATE SET device_id=excluded.device_id
            """,
            ("343835302", "browser-123"),
        )
        conn.commit()
        print("OK, записали ссылку для 343835302 -> browser-123")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
