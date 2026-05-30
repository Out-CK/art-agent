"""
VenueEnricher — finds addresses for events that couldn't be geocoded.

Strategy per venue (in order):
  1. Re-try Nominatim (may succeed for venues recently added to OSM,
     or that the geocoder missed due to query ordering).
  2. Nimble web search → top result snippets → LLM extracts a clean address.
  3. Geocode the LLM-extracted address via Nominatim.
  4. Update all events sharing that venue name in Supabase.
"""
from __future__ import annotations

from typing import Optional

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from db.operations import get_unmapped_venues, update_venue_coords
from tools.nimble_search_tool import NimbleSearchTool
from utils.geocoder import lookup_coords
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL = "claude-sonnet-4-6"


class VenueAddress(BaseModel):
    address: Optional[str] = None  # Full street address, or None if not found


class VenueEnricher:
    def __init__(self, event_type: str) -> None:
        self._event_type = event_type
        self._llm = ChatAnthropic(model=MODEL).with_structured_output(VenueAddress)
        self._search = NimbleSearchTool()

    def run(self) -> None:
        venues = get_unmapped_venues(self._event_type)
        if not venues:
            logger.info("VenueEnricher: no unmapped venues found")
            return

        logger.info(f"VenueEnricher [{self._event_type}]: {len(venues)} unmapped venue(s) to process")
        resolved = 0

        for venue_name, event_ids in venues.items():
            result = self._enrich_venue(venue_name)
            if result:
                lat, lng, address = result
                update_venue_coords(venue_name, lat, lng, address)
                logger.info(
                    f"  ✓ '{venue_name}' → {address} ({len(event_ids)} event(s) updated)"
                )
                resolved += 1
            else:
                logger.info(f"  ✗ '{venue_name}' — could not locate")

        logger.info(f"VenueEnricher [{self._event_type}] complete: {resolved}/{len(venues)} venues resolved")

    def _enrich_venue(self, venue: str) -> Optional[tuple[float, float, str]]:
        # 1. Re-try Nominatim directly
        result = lookup_coords(venue)
        if result:
            return result

        # 2. Nimble search → LLM address extraction → Nominatim geocode
        address = self._search_for_address(venue)
        if not address:
            return None

        result = lookup_coords(address)
        if result:
            return result

        # If Nominatim can't geocode the extracted address string, try it as-is
        # with a more permissive bare search
        from utils.geocoder import _nominatim_search_bare, _build_address
        try:
            results = _nominatim_search_bare(address)
            if results:
                r = results[0]
                return float(r["lat"]), float(r["lon"]), _build_address(r)
        except Exception as e:
            logger.warning(f"Bare geocode of extracted address failed for '{venue}': {e}")

        return None

    def _search_for_address(self, venue: str) -> Optional[str]:
        """Use Nimble to find the venue, then have the LLM extract a clean street address."""
        queries = [
            f"{venue} NYC address",
            f"{venue} New York location",
        ]

        snippets: list[str] = []
        for query in queries:
            try:
                results = self._search.run({"query": query, "query_type": "niche"})
                for r in results[:3]:
                    title = r.get("title", "")
                    content = (r.get("content") or "")[:1000]
                    snippets.append(f"Title: {title}\nContent: {content}")
            except Exception as e:
                logger.warning(f"Nimble search failed for '{query}': {e}")

        if not snippets:
            return None

        combined = "\n\n---\n".join(snippets[:6])
        prompt = (
            f"I'm trying to find the street address of this NYC venue: \"{venue}\"\n\n"
            f"Here are search results that may contain the address:\n\n{combined}\n\n"
            f"Extract the full street address of this venue (e.g. '35 W 35th St, New York, NY 10001'). "
            f"If you can't find a specific street address, return null."
        )

        try:
            result: VenueAddress = self._llm.invoke([{"role": "user", "content": prompt}])
            if result.address:
                logger.debug(f"LLM extracted address for '{venue}': {result.address}")
            return result.address
        except Exception as e:
            logger.warning(f"LLM address extraction failed for '{venue}': {e}")
            return None
