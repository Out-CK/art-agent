from __future__ import annotations

from db.operations import get_past_entries, insert_past_event_entry, delete_event_entry
from utils.logger import get_logger

logger = get_logger(__name__)


class PastEventArchiver:
    """
    Pure data logic — no LLM required.
    Moves all Event Entry Database rows with dates in the past to the
    Past Event Entry Database.
    """

    def run(self) -> int:
        """Archive past events. Returns the count of archived entries."""
        logger.info("PastEventArchiver: querying for past events…")
        past_entries = get_past_entries()

        if not past_entries:
            logger.info("PastEventArchiver: no past events to archive")
            return 0

        logger.info(f"PastEventArchiver: archiving {len(past_entries)} entries…")
        archived_count = 0

        for entry in past_entries:
            event_entry_id = entry.get("event_entry_id", "?")
            try:
                insert_past_event_entry(entry)
            except Exception as e:
                logger.error(
                    f"PastEventArchiver: failed to insert {event_entry_id} into past DB — "
                    f"skipping delete to avoid data loss. Error: {e}"
                )
                continue

            try:
                delete_event_entry(event_entry_id)
                archived_count += 1
                logger.debug(f"PastEventArchiver: archived {event_entry_id}")
            except Exception:
                # Error already logged as CRITICAL inside delete_event_entry
                pass

        logger.info(f"PastEventArchiver: archived {archived_count} entries")
        return archived_count
