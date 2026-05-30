"""
All Supabase read/write operations. No raw SQL should appear outside this file.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any

from db.supabase_client import get_supabase_client
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Event Web Database
# ---------------------------------------------------------------------------

def insert_web_batch(records: list[dict[str, Any]]) -> None:
    """Bulk-insert raw web content records into event_web_database."""
    if not records:
        return
    client = get_supabase_client()
    try:
        client.table("event_web_database").insert(records).execute()
        logger.info(f"Inserted {len(records)} records into event_web_database")
    except Exception as e:
        logger.error(f"Failed to insert web batch records: {e}\nData sample: {records[:2]}")
        raise


# ---------------------------------------------------------------------------
# Event Entry Database
# ---------------------------------------------------------------------------

def get_existing_venue_coords() -> dict[str, tuple[float, float, str]]:
    """Return a mapping of venue → (lat, lng, address) for all entries that already have coords."""
    client = get_supabase_client()
    try:
        result = (
            client.table("event_entry_database")
            .select("venue, address, lat, lng")
            .not_.is_("lat", "null")
            .execute()
        )
        cache: dict[str, tuple[float, float, str]] = {}
        for r in result.data:
            if r.get("lat") is not None and r.get("lng") is not None and r.get("venue"):
                cache[r["venue"]] = (float(r["lat"]), float(r["lng"]), r.get("address") or "")
        return cache
    except Exception as e:
        logger.warning(f"Could not fetch existing venue coords: {e}")
        return {}


def get_existing_venue_addresses() -> dict[str, str]:
    """Return a mapping of venue → address for all entries that already have an address."""
    coords = get_existing_venue_coords()
    return {venue: addr for venue, (_, _, addr) in coords.items() if addr}


def get_existing_future_entries() -> list[dict[str, Any]]:
    """Fetch all entries in event_entry_database where date >= today."""
    client = get_supabase_client()
    today_str = date.today().strftime("%m-%d-%Y")
    try:
        result = (
            client.table("event_entry_database")
            .select("event_entry_id, artist, venue, date, start_time")
            .gte("date", today_str)
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.error(f"Failed to fetch existing future entries: {e}")
        return []


def insert_event_entries(entries: list[dict[str, Any]]) -> int:
    """Bulk-insert Event Entries into event_entry_database. Returns insert count."""
    if not entries:
        return 0
    client = get_supabase_client()
    try:
        client.table("event_entry_database").insert(entries).execute()
        logger.info(f"Inserted {len(entries)} entries into event_entry_database")
        return len(entries)
    except Exception as e:
        logger.error(
            f"Failed to insert event entries: {e}\n"
            f"Data: {json.dumps(entries, default=str)[:2000]}"
        )
        raise


def get_past_entries() -> list[dict[str, Any]]:
    """Fetch all entries in event_entry_database where date < today."""
    client = get_supabase_client()
    today_str = date.today().strftime("%m-%d-%Y")
    try:
        result = (
            client.table("event_entry_database")
            .select("*")
            .lt("date", today_str)
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.error(f"Failed to fetch past entries: {e}")
        return []


def delete_event_entry(event_entry_id: str) -> None:
    """Delete a single entry from event_entry_database by event_entry_id."""
    client = get_supabase_client()
    try:
        client.table("event_entry_database").delete().eq(
            "event_entry_id", event_entry_id
        ).execute()
    except Exception as e:
        logger.critical(
            f"CRITICAL: Failed to delete event_entry_id={event_entry_id} "
            f"after archiving. Manual cleanup required. Error: {e}"
        )
        raise


# ---------------------------------------------------------------------------
# Past Event Entry Database
# ---------------------------------------------------------------------------

def get_unmapped_venues(event_type: str) -> dict[str, list[str]]:
    """Return {venue_name: [event_entry_id, ...]} for future events of a given type with no lat/lng."""
    client = get_supabase_client()
    today_str = date.today().strftime("%m-%d-%Y")
    try:
        result = (
            client.table("event_entry_database")
            .select("event_entry_id, venue")
            .is_("lat", "null")
            .eq("event_type", event_type)
            .gte("date", today_str)
            .execute()
        )
        venues: dict[str, list[str]] = {}
        for r in result.data or []:
            venue = r.get("venue") or ""
            if venue:
                venues.setdefault(venue, []).append(r["event_entry_id"])
        return venues
    except Exception as e:
        logger.error(f"Failed to fetch unmapped venues: {e}")
        return {}


def update_venue_coords(venue: str, lat: float, lng: float, address: str) -> None:
    """Update lat, lng, and address for all events with the given venue name and no coords."""
    client = get_supabase_client()
    try:
        client.table("event_entry_database").update(
            {"lat": lat, "lng": lng, "address": address}
        ).eq("venue", venue).is_("lat", "null").execute()
        logger.info(f"Updated coords for venue '{venue}': ({lat}, {lng})")
    except Exception as e:
        logger.error(f"Failed to update coords for venue '{venue}': {e}")
        raise


def insert_past_event_entry(entry: dict[str, Any]) -> None:
    """Insert a single entry into past_event_entry_database."""
    client = get_supabase_client()
    # Remove primary key to let DB auto-assign a new one, keep event_entry_id
    entry_copy = {k: v for k, v in entry.items() if k != "id"}
    try:
        client.table("past_event_entry_database").insert(entry_copy).execute()
    except Exception as e:
        logger.error(
            f"Failed to insert into past_event_entry_database: {e}\n"
            f"Entry: {json.dumps(entry_copy, default=str)[:500]}"
        )
        raise
