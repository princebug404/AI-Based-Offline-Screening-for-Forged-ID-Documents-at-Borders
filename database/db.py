# SIH26188 - Database connection and initialization

import os
import sqlite3

from database.models import CREATE_DOCUMENTS_TABLE, CREATE_PROCESSING_RESULTS_TABLE


def get_connection(db_path):
    """
    Open a SQLite connection with Row factory enabled.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        sqlite3.Connection with Row factory.
    """
    # Ensure the directory for the database file exists
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path):
    """
    Initialize the database by creating tables if they do not exist.

    This is safe to call multiple times — it uses CREATE TABLE IF NOT EXISTS.

    Args:
        db_path: Path to the SQLite database file.
    """
    conn = get_connection(db_path)
    try:
        conn.execute(CREATE_DOCUMENTS_TABLE)
        conn.execute(CREATE_PROCESSING_RESULTS_TABLE)
        conn.commit()
    finally:
        conn.close()
