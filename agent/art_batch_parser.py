from __future__ import annotations

from typing import List, Optional

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from utils.logger import get_logger

logger = get_logger(__name__)

MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 3

SYSTEM_PROMPT = """You are an art exhibition data extraction specialist. For each web page provided,
extract all individual NYC art gallery openings and exhibitions mentioned.

Rules:
- Create a SEPARATE entry for each distinct exhibition or opening event found on a page.
- Set event_type = "art" always.
- Include: gallery openings, solo shows, group shows, museum exhibitions, art fairs, pop-up shows,
  and artist talks/receptions. Skip concerts, theater, comedy, film, fitness, and conferences.
- event_title format: "[Exhibition Name] at [Gallery]" — e.g. "Soft Machines at Gavin Brown's Enterprise"
  For a solo show: "[Artist Name]: [Show Title] at [Gallery]"
  For a group show with no distinct title: "Group Exhibition at [Gallery]"
- artist: the featured artist(s) name(s), comma-separated. For group shows use "Various Artists".
- venue: the gallery or museum name only — no city/state suffix.
- date: The OPENING date (or start date of the run) in "MM-DD-YYYY" format.
  If only a date range is shown (e.g. "June 5 – July 20"), use the start date.
- start_time: opening reception time if given, in "00:00am/pm" format (e.g. "06:00pm").
  Leave null if no specific event time is listed.
- genre: assign exactly ONE from this list based on the type of work shown:
  Contemporary, Modern, Photography, Sculpture, Painting, Drawing, Mixed Media,
  Installation, Video Art, Performance, Printmaking, Ceramics, Textile, Street Art, Other.
- description: a 1-2 sentence description of the exhibition from the page. Include the medium,
  themes, or notable context if available.
- If the venue's full street address is visible anywhere on the page, populate the `address` field
  (e.g. "541 West 25th Street, New York, NY 10001"). Include street number, street name, city, state, zip.
  If no street address is visible, leave `address` empty.
- If exhibition name, gallery/venue, OR date cannot be confidently extracted, SKIP that entry.
- DO NOT set event_entry_id or entry_batch_id — leave them as empty strings "".
- Return a JSON object with key "entries" containing an array of ArtEntry objects.
"""


class ArtEntry(BaseModel):
    event_entry_id: str = ""
    entry_batch_id: str = ""
    event_title: str
    description: str
    artist: str
    venue: str
    event_type: str = "art"
    multi_day_event: bool = True  # exhibitions almost always span multiple days
    date: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    tickets_source_1: Optional[str] = None
    tickets_webpage_contents_1: Optional[str] = None
    tickets_source_2: Optional[str] = None
    tickets_webpage_contents_2: Optional[str] = None
    tickets_source_3: Optional[str] = None
    tickets_webpage_contents_3: Optional[str] = None
    tickets_source_4: Optional[str] = None
    tickets_webpage_contents_4: Optional[str] = None
    no_tickets_source_1: Optional[str] = None
    no_tickets_webpage_contents_1: Optional[str] = None
    no_tickets_source_2: Optional[str] = None
    no_tickets_webpage_contents_2: Optional[str] = None
    no_tickets_source_3: Optional[str] = None
    no_tickets_webpage_contents_3: Optional[str] = None
    no_tickets_source_4: Optional[str] = None
    no_tickets_webpage_contents_4: Optional[str] = None
    genre: Optional[str] = None
    webpage_contents: Optional[str] = None
    address: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None


class ArtEntryList(BaseModel):
    entries: List[ArtEntry]


class ArtBatchParser:
    def __init__(self):
        self._llm = ChatAnthropic(model=MODEL).with_structured_output(ArtEntryList)

    def parse(self, web_batch: list[dict]) -> list[ArtEntry]:
        """Parse a full Web Batch into Art Entries, processing BATCH_SIZE pages per LLM call."""
        logger.info(f"ArtBatchParser processing {len(web_batch)} pages in batches of {BATCH_SIZE}…")
        all_entries: list[ArtEntry] = []

        for batch_start in range(0, len(web_batch), BATCH_SIZE):
            batch = web_batch[batch_start: batch_start + BATCH_SIZE]
            try:
                entries = self._parse_batch(batch)
                logger.info(f"Batch {batch_start}–{batch_start + len(batch)}: parsed {len(entries)} entries")
                all_entries.extend(entries)
            except Exception as e:
                logger.error(f"ArtBatchParser batch {batch_start}–{batch_start + len(batch)} failed: {e}")

        logger.info(f"ArtBatchParser total entries parsed: {len(all_entries)}")
        return all_entries

    def _parse_batch(self, batch: list[dict]) -> list[ArtEntry]:
        pages_text = ""
        for record in batch:
            content_snippet = (record.get("content") or "")[:5000]
            pages_text += (
                f"\n\n---\n"
                f"PAGE URL: {record.get('url', '')}\n"
                f"QUERY USED: {record.get('query_used', '')}\n"
                f"CONTENT:\n{content_snippet}"
            )

        result: ArtEntryList = self._llm.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract art exhibition entries from these pages:{pages_text}"},
            ]
        )
        entries = result.entries or []
        for entry in entries:
            if not entry.webpage_contents:
                for record in batch:
                    if record.get("url"):
                        entry.webpage_contents = (record.get("content") or "")[:10000]
                        break
        return entries
