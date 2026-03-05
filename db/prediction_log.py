"""SQLite-backed prediction audit logger.

Every call to /predict is written here with full input features,
output scores, and metadata. Used for:
  - Regulatory audit trail (FCRA/ECOA compliance)
  - Drift detection reference window
  - Future retraining dataset

Usage:
    from db.prediction_log import PredictionLogger
    logger = PredictionLogger("db/predictions.db")
    logger.log(application_id, features, result)
    rows = logger.recent(hours=24)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id   TEXT    NOT NULL,
    timestamp        TEXT    NOT NULL,
    default_prob     REAL    NOT NULL,
    risk_score       INTEGER NOT NULL,
    decision         TEXT    NOT NULL,
    confidence       REAL    NOT NULL,
    model_version    TEXT    NOT NULL,
    latency_ms       REAL    NOT NULL,
    features_json    TEXT    NOT NULL,
    shap_json        TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_ts
    ON predictions (timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_appid
    ON predictions (application_id);
"""


class PredictionLogger:
    """Thread-safe SQLite prediction logger.

    Args:
        db_path: Path to SQLite file. Created if absent.
    """

    def __init__(self, db_path: str | Path = "db/predictions.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_CREATE_TABLE)
        logger.info("Prediction DB ready: %s", self.db_path)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def log(
        self,
        application_id: str,
        features: dict[str, Any],
        default_prob: float,
        risk_score: int,
        decision: str,
        confidence: float,
        model_version: str,
        latency_ms: float,
        shap_values: dict[str, float] | None = None,
    ) -> None:
        """Write a prediction record to the database.

        Args:
            application_id: Unique request identifier.
            features: Raw input feature dict.
            default_prob: Predicted default probability.
            risk_score: 300–850 credit score.
            decision: APPROVE / REVIEW / DECLINE.
            confidence: Distance from decision boundary.
            model_version: Model artifact version string.
            latency_ms: Request processing time.
            shap_values: Optional SHAP value dict.
        """
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO predictions
                    (application_id, timestamp, default_prob, risk_score,
                     decision, confidence, model_version, latency_ms,
                     features_json, shap_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    application_id,
                    ts,
                    default_prob,
                    risk_score,
                    decision,
                    confidence,
                    model_version,
                    latency_ms,
                    json.dumps(features, default=str),
                    json.dumps(shap_values) if shap_values else None,
                ),
            )
        logger.debug("Logged prediction %s → %s (%.3f)", application_id, decision, default_prob)

    def recent(self, hours: int = 24) -> list[dict[str, Any]]:
        """Fetch predictions from the last N hours.

        Args:
            hours: Lookback window.

        Returns:
            List of prediction dicts.
        """
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE timestamp >= ? ORDER BY timestamp DESC",
                (since,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_feature_vectors(self, hours: int = 168) -> list[dict[str, Any]]:
        """Return raw feature dicts for drift detection.

        Args:
            hours: Lookback window (default 7 days).

        Returns:
            List of feature dicts parsed from features_json.
        """
        rows = self.recent(hours=hours)
        result = []
        for r in rows:
            try:
                result.append(json.loads(r["features_json"]))
            except (json.JSONDecodeError, KeyError):
                pass
        return result

    def decision_counts(self, hours: int = 24) -> dict[str, int]:
        """Count decisions in the last N hours.

        Args:
            hours: Lookback window.

        Returns:
            Dict of decision → count.
        """
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT decision, COUNT(*) as cnt
                FROM predictions
                WHERE timestamp >= ?
                GROUP BY decision
                """,
                (since,),
            ).fetchall()
        return {r["decision"]: r["cnt"] for r in rows}

    def latency_percentiles(self, hours: int = 1) -> dict[str, float]:
        """Compute p50/p95/p99 latency from recent predictions.

        Args:
            hours: Lookback window.

        Returns:
            Dict with p50, p95, p99 in milliseconds.
        """
        rows = self.recent(hours=hours)
        if not rows:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        latencies = sorted(r["latency_ms"] for r in rows)
        n = len(latencies)

        def pct(p: float) -> float:
            idx = int(p / 100 * n)
            return latencies[min(idx, n - 1)]

        return {"p50": pct(50), "p95": pct(95), "p99": pct(99)}

    def total_count(self) -> int:
        """Return total number of predictions logged."""
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM predictions").fetchone()
        return int(row["cnt"])
