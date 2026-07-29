import os
import json
import logging

import psycopg
import psycopg.rows

DATABASE_URL = os.getenv("DATABASE_URL")

def get_connection():
    return psycopg.connect(DATABASE_URL)

def init_db():
    logging.info("Initializing unified database (Postgres)...")
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id SERIAL PRIMARY KEY,
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
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO alerts (chat_id, alert_type, scheduled_time, last_sent, payload)
                VALUES (%s, %s, %s, NULL, %s)
                RETURNING id
            """, (chat_id, alert_type, scheduled_time, payload_str))
            new_id = cursor.fetchone()[0]
        conn.commit()
        return new_id

def get_user_alerts(chat_id: str, alert_type: str):
    with get_connection() as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cursor:
            cursor.execute("""
                SELECT id, chat_id, alert_type, scheduled_time, payload FROM alerts
                WHERE chat_id = %s AND alert_type = %s
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
        with conn.cursor() as cursor:
            cursor.execute("""
                DELETE FROM alerts
                WHERE chat_id = %s AND alert_type = %s AND id = %s
            """, (chat_id, alert_type, alert_id))
            deleted = cursor.rowcount > 0
        conn.commit()
        return deleted

def get_alerts_to_trigger(alert_type: str, current_time: str, current_date: str):
    with get_connection() as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cursor:
            cursor.execute("""
                SELECT id, chat_id, alert_type, scheduled_time, payload FROM alerts
                WHERE alert_type = %s AND scheduled_time = %s AND (last_sent IS NULL OR last_sent != %s)
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
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE alerts
                SET last_sent = %s
                WHERE id = %s
            """, (current_date, alert_id))
        conn.commit()

def get_all_user_alerts(chat_id: str):
    with get_connection() as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cursor:
            cursor.execute("""
                SELECT id, chat_id, alert_type, scheduled_time, payload FROM alerts
                WHERE chat_id = %s
            """, (chat_id,))
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
