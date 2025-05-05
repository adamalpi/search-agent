import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional

# Define the path for the SQLite database file relative to this file
DB_PATH = os.path.join(os.path.dirname(__file__), "analysis_history.db")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def init_db():
    """Initializes the SQLite database and creates the analysis_tasks table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_tasks (
                task_id TEXT PRIMARY KEY,
                industry TEXT NOT NULL,
                status TEXT NOT NULL,
                result_summary TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Add an index for faster querying by status and timestamp
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status_timestamp ON analysis_tasks (status, timestamp);
        """)
        conn.commit()
        logging.info(f"Database initialized successfully at {DB_PATH}")
    except sqlite3.Error as e:
        logging.error(f"Database error during initialization: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()


def log_task_status(task_id: str, industry: str, status: str, result_summary: Optional[str] = None):
    """Logs or updates the status of an analysis task in the database."""
    conn = None  # Ensure conn is defined before try block
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Use INSERT OR REPLACE to handle both new tasks and updates
        # Update timestamp on replace to reflect the latest status change
        cursor.execute(
            """
            INSERT OR REPLACE INTO analysis_tasks
            (task_id, industry, status, result_summary, timestamp)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            (task_id, industry, status, result_summary),
        )
        conn.commit()
        logging.debug(f"Logged status '{status}' for task {task_id} (Industry: {industry})")
    except sqlite3.Error as e:
        logging.error(f"Database error logging task {task_id}: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()


def query_tasks(limit: int = 5, industry_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Queries completed analysis tasks from the database."""
    conn = None
    results = []
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Return results as dict-like rows
        cursor = conn.cursor()

        base_query = """
            SELECT task_id, industry, status, result_summary, timestamp
            FROM analysis_tasks
            WHERE status = 'COMPLETED'
        """
        params = []

        if industry_filter:
            base_query += " AND LOWER(industry) = LOWER(?)"
            params.append(industry_filter)

        base_query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(base_query, params)
        results = [dict(row) for row in cursor.fetchall()]
        logging.debug(f"Queried {len(results)} tasks (limit={limit}, industry='{industry_filter}')")

    except sqlite3.Error as e:
        logging.error(f"Database error querying tasks: {e}", exc_info=True)
        # Return empty list on error, or could raise an exception
    finally:
        if conn:
            conn.close()
    return results


# Example usage (optional, for testing)
if __name__ == "__main__":
    print("Initializing DB...")
    init_db()
    print("Logging sample tasks...")
    log_task_status("task-1", "Automotive", "COMPLETED", "Summary for Automotive task 1...")
    log_task_status("task-2", "Tech", "COMPLETED", "Summary for Tech task 1...")
    log_task_status("task-3", "Automotive", "FAILED", "Error message...")
    log_task_status("task-4", "Automotive", "COMPLETED", "Summary for Automotive task 2...")
    print("Querying last 3 completed tasks:")
    tasks = query_tasks(limit=3)
    for task in tasks:
        print(task)
    print("\nQuerying last 3 completed Automotive tasks:")
    auto_tasks = query_tasks(limit=3, industry_filter="Automotive")
    for task in auto_tasks:
        print(task)
