import os
from supabase import create_client, Client
from utils.logger import get_logger

logger = get_logger(__name__)

_client: Client | None = None


def get_supabase_client() -> Client:
    """Return the singleton Supabase client, initializing it if needed."""
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        _client = create_client(url, key)
        logger.info("Supabase client initialized")
    return _client
