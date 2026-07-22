import sqlite3
import os
import json
import logging

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "routes.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    logging.info(f"Initializing unified database at: {DB_PATH}")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                scheduled_time TEXT NOT NULL,
                last_sent TEXT,
                payload TEXT NOT NULL
            )
        """)
        conn.commit()

# --- Common Alert Operations ---

def add_alert(chat_id: str, alert_type: str, scheduled_time: str, payload: dict) -> int:
    payload_str = json.dumps(payload)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO alerts (chat_id, alert_type, scheduled_time, last_sent, payload)
            VALUES (?, ?, ?, NULL, ?)
        """, (chat_id, alert_type, scheduled_time, payload_str))
        conn.commit()
        return cursor.lastrowid

def get_user_alerts(chat_id: str, alert_type: str):
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, chat_id, alert_type, scheduled_time, payload FROM alerts
            WHERE chat_id = ? AND alert_type = ?
        """, (chat_id, alert_type))
        results = []
        for row in cursor.fetchall():
            item = dict(row)
            try:
                payload_dict = json.loads(item.pop("payload"))
                item.update(payload_dict)
            except Exception as e:
                logging.error(f"Error parsing payload JSON: {e}")
            results.append(item)
        return results

def delete_user_alert(chat_id: str, alert_type: str, alert_id: int) -> bool:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM alerts
            WHERE chat_id = ? AND alert_type = ? AND id = ?
        """, (chat_id, alert_type, alert_id))
        conn.commit()
        return cursor.rowcount > 0

def get_alerts_to_trigger(alert_type: str, current_time: str, current_date: str):
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, chat_id, alert_type, scheduled_time, payload FROM alerts
            WHERE alert_type = ? AND scheduled_time = ? AND (last_sent IS NULL OR last_sent != ?)
        """, (alert_type, current_time, current_date))
        results = []
        for row in cursor.fetchall():
            item = dict(row)
            try:
                payload_dict = json.loads(item.pop("payload"))
                item.update(payload_dict)
            except Exception as e:
                logging.error(f"Error parsing payload JSON: {e}")
            results.append(item)
        return results

def update_last_sent(alert_id: int, current_date: str):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE alerts
            SET last_sent = ?
            WHERE id = ?
        """, (current_date, alert_id))
        conn.commit()
