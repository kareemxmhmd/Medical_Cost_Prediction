import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

DEFAULT_DB_PATH = "audit_predictions.db"


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path))


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Initializes the SQLite schema for inference audit logging and enables WAL mode."""
    with get_connection(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                age INTEGER NOT NULL,
                bmi REAL NOT NULL,
                children INTEGER NOT NULL,
                smoker TEXT NOT NULL,
                region TEXT NOT NULL,
                predicted_cost REAL NOT NULL,
                interval_lower REAL NOT NULL,
                interval_upper REAL NOT NULL,
                premium_tier TEXT NOT NULL,
                model_version TEXT NOT NULL
            )
            """
        )
        conn.commit()


def log_prediction(
    applicant: dict,
    predicted_cost: float,
    interval_80: tuple[float, float],
    premium_tier: str,
    model_version: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Persists a single scored applicant profile and model decision."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO prediction_logs (
                timestamp, age, bmi, children, smoker, region,
                predicted_cost, interval_lower, interval_upper,
                premium_tier, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_iso,
                int(applicant["age"]),
                float(applicant["bmi"]),
                int(applicant["children"]),
                str(applicant["smoker"]),
                str(applicant["region"]),
                float(predicted_cost),
                float(interval_80[0]),
                float(interval_80[1]),
                str(premium_tier),
                str(model_version),
            ),
        )
        conn.commit()
        return cursor.lastrowid


def log_predictions_batch(
    records: list[dict],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Persists a batch of scored applicant profiles atomically in a single transaction."""
    if not records:
        return 0
    now_iso = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            rec.get("decision_timestamp", now_iso),
            int(rec["applicant"]["age"]),
            float(rec["applicant"]["bmi"]),
            int(rec["applicant"]["children"]),
            str(rec["applicant"]["smoker"]),
            str(rec["applicant"]["region"]),
            float(rec["predicted_cost"]),
            float(rec["interval_80"][0]),
            float(rec["interval_80"][1]),
            str(rec["premium_tier"]),
            str(rec["model_version"]),
        )
        for rec in records
    ]
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO prediction_logs (
                timestamp, age, bmi, children, smoker, region,
                predicted_cost, interval_lower, interval_upper,
                premium_tier, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return len(rows)


def get_recent_predictions(
    limit: int = 500, 
    db_path: str | Path = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """Retrieves recent prediction records for drift detection and auditing."""
    with get_connection(db_path) as conn:
        query = "SELECT * FROM prediction_logs ORDER BY id DESC LIMIT ?"
        return pd.read_sql_query(query, conn, params=(limit,))


def get_prediction_count(db_path: str | Path = DEFAULT_DB_PATH) -> int:
    """Returns total count of logged predictions."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM prediction_logs")
        return cursor.fetchone()[0]
