from utils.logger import get_logger

logger = get_logger(__name__)


def get_next_event_entry_id(supabase_client) -> str:
    """
    Query both event_entry_database and past_event_entry_database for the
    current global maximum event_entry_id, then return the next ID as a
    12-character zero-padded string.
    """
    max_id = 0

    try:
        result = (
            supabase_client.table("event_entry_database")
            .select("event_entry_id")
            .order("event_entry_id", desc=True)
            .limit(1)
            .execute()
        )
        if result.data:
            max_id = max(max_id, int(result.data[0]["event_entry_id"]))
    except Exception as e:
        logger.warning(f"Could not query event_entry_database for max ID: {e}")

    try:
        result = (
            supabase_client.table("past_event_entry_database")
            .select("event_entry_id")
            .order("event_entry_id", desc=True)
            .limit(1)
            .execute()
        )
        if result.data:
            max_id = max(max_id, int(result.data[0]["event_entry_id"]))
    except Exception as e:
        logger.warning(f"Could not query past_event_entry_database for max ID: {e}")

    next_id = max_id + 1
    logger.debug(f"Next event_entry_id: {next_id}")
    return next_id


class IDGenerator:
    """
    Generates sequential, globally-unique event_entry_id values for a single
    Concert Run, starting from the global maximum found in both DB tables.
    """

    def __init__(self, supabase_client):
        self._current = get_next_event_entry_id(supabase_client)
        logger.info(f"IDGenerator initialized — starting from {self._format(self._current)}")

    def _format(self, n: int) -> str:
        return str(n).zfill(12)

    def next(self) -> str:
        value = self._format(self._current)
        self._current += 1
        return value
