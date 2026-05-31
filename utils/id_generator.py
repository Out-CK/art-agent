from utils.logger import get_logger

logger = get_logger(__name__)


class IDGenerator:
    """
    Generates sequential, globally-unique event_entry_id values using a
    Postgres SEQUENCE (event_entry_id_seq) via the next_event_entry_id() RPC.

    Each call to next() atomically increments the sequence, so concurrent
    agents will never collide.
    """

    def __init__(self, supabase_client):
        self._client = supabase_client
        logger.info("IDGenerator initialized — using Postgres sequence")

    def next(self) -> str:
        result = self._client.rpc("next_event_entry_id").execute()
        event_id = result.data
        logger.debug(f"Generated event_entry_id: {event_id}")
        return event_id
