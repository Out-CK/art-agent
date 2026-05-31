"""
ArtInstagramParser — converts raw gallery/museum Instagram profile content into ArtEntry objects.

Each record is a dict with:
    url     : str   (Instagram profile URL)
    handle  : str   (Instagram handle)
    content : str   (combined bio + post captions)
"""
from __future__ import annotations

from datetime import date
from typing import List

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from agent.art_batch_parser import ArtEntry
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 5

SYSTEM_PROMPT = """You are an art exhibition data extraction specialist. You will receive content
from NYC gallery and museum Instagram profiles — biographies and post captions.
Extract all upcoming NYC art gallery openings and exhibitions mentioned.

Rules:
- Create a SEPARATE entry for each distinct exhibition or opening event.
- Only extract events that are UPCOMING (future dates relative to today). Skip past events.
- Only extract events in NYC (Manhattan, Brooklyn, Queens, Bronx, Staten Island).
- Set event_type = "art" always. Skip concerts, theater, comedy, film, and conferences.
- event_title format: "[Exhibition Name] at [Gallery]" — e.g. "Soft Machines at Gagosian"
  For a solo show: "[Artist Name]: [Show Title] at [Gallery]"
  For a group show with no distinct title: "Group Exhibition at [Gallery]"
- artist: the featured artist(s) name(s), comma-separated. For group shows use "Various Artists".
- venue: the gallery or museum name only — no city/state suffix.
- date: The OPENING date (or start date of the run) in "MM-DD-YYYY" format.
  If only a date range is shown (e.g. "June 5 – July 20"), use the start date.
- start_time: opening reception time if given, in "00:00am/pm" format (e.g. "06:00pm").
  Leave null if no specific reception time is listed.
- genre: assign exactly ONE from this list:
  Contemporary, Modern, Photography, Sculpture, Painting, Drawing, Mixed Media,
  Installation, Video Art, Performance, Printmaking, Ceramics, Textile, Street Art, Other.
- description: a 1-2 sentence description from the caption. Include medium, themes, or context.
- Populate no_tickets_source_1 with the SPECIFIC POST URL if one is provided (e.g.
  "https://www.instagram.com/p/ABC123/"). If no specific post URL is available, use
  the Instagram profile URL instead.
- Populate no_tickets_webpage_contents_1 with the relevant post caption text.
- If exhibition name, gallery/venue, OR date cannot be confidently extracted, SKIP that entry.
- DO NOT set event_entry_id or entry_batch_id — leave them as empty strings "".
- Return a JSON object with key "entries" containing an array of ArtEntry objects.
- Be conservative: only extract events you are highly confident about.
"""


class ArtEntryList(BaseModel):
    entries: List[ArtEntry]


class ArtInstagramParser:
    def __init__(self):
        self._llm = ChatAnthropic(model=MODEL).with_structured_output(ArtEntryList)

    def parse(self, post_pages: list[dict]) -> list[ArtEntry]:
        """Parse a list of gallery/museum Instagram profile records into ArtEntry objects."""
        logger.info(
            f"ArtInstagramParser processing {len(post_pages)} profiles "
            f"in batches of {BATCH_SIZE}…"
        )
        all_entries: list[ArtEntry] = []

        for batch_start in range(0, len(post_pages), BATCH_SIZE):
            batch = post_pages[batch_start: batch_start + BATCH_SIZE]
            try:
                entries = self._parse_batch(batch)
                logger.info(
                    f"Batch {batch_start}–{batch_start + len(batch)}: "
                    f"parsed {len(entries)} entries"
                )
                all_entries.extend(entries)
            except Exception as e:
                logger.error(
                    f"ArtInstagramParser batch "
                    f"{batch_start}–{batch_start + len(batch)} failed: {e}"
                )

        logger.info(f"ArtInstagramParser total entries parsed: {len(all_entries)}")
        return all_entries

    def _parse_batch(self, batch: list[dict]) -> list[ArtEntry]:
        pages_text = ""
        for record in batch:
            pages_text += (
                f"\n\n---\n"
                f"INSTAGRAM PROFILE: @{record.get('handle', '')}\n"
                f"URL: {record.get('url', '')}\n"
                f"CONTENT:\n{record.get('content', '')[:5000]}"
            )

        result: ArtEntryList = self._llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Today's date is {date.today().strftime('%m-%d-%Y')}. "
                    f"Extract NYC art exhibition entries from these Instagram profiles:{pages_text}"
                ),
            },
        ])
        return result.entries or []
