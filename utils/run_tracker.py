"""
Run Tracker — collects per-run metrics and writes them to agent_run_log.

Usage:
    tracker = RunTracker(agent_name="concert", run_type="web_search")
    tracker.start()
    ...
    tracker.inc("nimble_search_calls")
    tracker.inc("nimble_search_successes")
    ...
    tracker.finish(status="success")
"""
from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
from typing import Any, Optional

from db.supabase_client import get_supabase_client
from utils.logger import get_logger

logger = get_logger(__name__)

METRIC_FIELDS = {
    "nimble_search_calls",
    "nimble_search_successes",
    "nimble_search_failures",
    "nimble_extract_calls",
    "nimble_extract_successes",
    "nimble_extract_failures",
    "nimble_instagram_calls",
    "nimble_instagram_successes",
    "nimble_instagram_failures",
    "nimble_tiktok_calls",
    "nimble_tiktok_successes",
    "nimble_tiktok_failures",
    "queries_executed",
    "pages_fetched_round1",
    "pages_fetched_round2",
    "raw_entries_parsed",
    "entries_inserted",
    "intra_batch_dupes_removed",
    "cross_db_dupes_removed",
    "entries_missing_address",
    "entries_missing_date",
    "entries_missing_media",
    "entries_archived",
    "venues_geocoded",
    "venues_from_cache",
    "media_enricher_lookups",
    "media_enricher_found",
    "ticketing_sources_queried",
    "ticketing_entries_found",
}


class RunTracker:
    def __init__(self, agent_name: str, run_type: str):
        self.agent_name = agent_name
        self.run_type = run_type
        self.entry_batch_id: Optional[str] = None
        self._metrics: dict[str, int] = {k: 0 for k in METRIC_FIELDS}
        self._start_time: Optional[float] = None
        self._started_at: Optional[str] = None

    def start(self) -> "RunTracker":
        self._start_time = time.time()
        self._started_at = datetime.now(timezone.utc).isoformat()
        return self

    def inc(self, metric: str, amount: int = 1) -> None:
        if metric in self._metrics:
            self._metrics[metric] += amount
        else:
            logger.warning(f"RunTracker: unknown metric '{metric}'")

    def set(self, metric: str, value: int) -> None:
        if metric in self._metrics:
            self._metrics[metric] = value
        else:
            logger.warning(f"RunTracker: unknown metric '{metric}'")

    def get(self, metric: str) -> int:
        return self._metrics.get(metric, 0)

    def count_missing_fields(self, entries: list) -> None:
        """Count entries missing key data points."""
        for entry in entries:
            if not getattr(entry, "address", None):
                self._metrics["entries_missing_address"] += 1
            if not getattr(entry, "date", None):
                self._metrics["entries_missing_date"] += 1
            if not getattr(entry, "media_url", None):
                self._metrics["entries_missing_media"] += 1

    def finish(self, status: str = "success", error_message: Optional[str] = None) -> None:
        duration = time.time() - (self._start_time or time.time())
        completed_at = datetime.now(timezone.utc).isoformat()

        row: dict[str, Any] = {
            "agent_name": self.agent_name,
            "run_type": self.run_type,
            "entry_batch_id": self.entry_batch_id,
            "started_at": self._started_at,
            "completed_at": completed_at,
            "duration_seconds": round(duration, 1),
            "status": status,
            "error_message": error_message,
        }
        row.update(self._metrics)

        try:
            client = get_supabase_client()
            client.table("agent_run_log").insert(row).execute()
            logger.info(f"RunTracker: logged run to agent_run_log ({status}, {duration:.1f}s)")
        except Exception as e:
            logger.error(f"RunTracker: failed to write run log: {e}")
            logger.debug(traceback.format_exc())
