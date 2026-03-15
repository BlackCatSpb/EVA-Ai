# database_utils.py
import sqlite3
import logging

logger = logging.getLogger("cogniflex.database")

def get_connection(db_path: str) -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        logger.debug(f"Соединение с базой данных: {db_path}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Ошибка подключения к базе данных {db_path}: {e}")
        raise

def execute_query(conn: sqlite3.Connection, query: str, params: tuple = ()) -> list:
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        if query.strip().upper().startswith("SELECT"):
            result = cursor.fetchall()
            logger.debug(f"Выполнен SELECT-запрос: {query}")
            return result
        conn.commit()
        logger.debug(f"Выполнен запрос: {query}")
        return []
    except sqlite3.Error as e:
        logger.error(f"Ошибка выполнения запроса {query}: {e}")
        raise