from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Optional

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from db.operations import get_existing_future_entries
from agent.art_batch_parser import ArtEntry as EventEntry
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL = "claude-sonnet-4-6"

MERGE_SYSTEM_PROMPT = """You are comparing two or more art exhibition entries that refer to the same
show (same artist/exhibition, venue, and date). Select the BEST version of event_title and description
from the candidates. Return a JSON object with keys "event_title" and "description"."""

_VENUE_SUFFIX_RE = re.compile(
    r",?\s*(new york(?: city)?|nyc|brooklyn|queens|bronx|staten island|manhattan)"
    r"(,?\s*(ny|new york))?\s*$",
    re.IGNORECASE,
)

_TIME_RE = re.compile(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)$", re.IGNORECASE)


def _normalize_venue(venue: str) -> str:
    v = venue.strip()
    v = _VENUE_SUFFIX_RE.sub("", v).strip().rstrip(",").strip()
    return v.lower()


def _normalize_time(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    m = _TIME_RE.match(t.strip())
    if not m:
        return t.strip().lower()
    hour, minute, meridiem = m.group(1), m.group(2) or "00", m.group(3).lower()
    return f"{int(hour):02d}:{minute}{meridiem}"


def _normalize_artist(artist: str) -> str:
    a = artist.strip().lower()
    a = re.sub(r"\s*[\(\[].*?[\)\]]", "", a)
    a = re.sub(r"\s+(feat(uring)?|ft)\.?\s+.+$", "", a)
    return a.strip()


class MergeChoice(BaseModel):
    event_title: str
    description: str


class DuplicateFinder:
    def __init__(self, id_generator):
        self._llm = ChatAnthropic(model=MODEL).with_structured_output(MergeChoice)
        self._id_gen = id_generator

    def deduplicate_batch(self, entries: list[EventEntry]) -> list[EventEntry]:
        logger.info(f"Intra-batch dedup: starting with {len(entries)} entries")

        parent = list(range(len(entries)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        avd_index: dict[tuple, int] = {}
        vdt_index: dict[tuple, int] = {}

        for i, e in enumerate(entries):
            v = _normalize_venue(e.venue)
            a = _normalize_artist(e.artist)
            d = e.date.strip()
            t = _normalize_time(e.start_time)

            key_avd = (a, v, d)
            if key_avd in avd_index:
                union(i, avd_index[key_avd])
            else:
                avd_index[key_avd] = i

            if t:
                key_vdt = (v, d, t)
                if key_vdt in vdt_index:
                    union(i, vdt_index[key_vdt])
                else:
                    vdt_index[key_vdt] = i

        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(len(entries)):
            groups[find(i)].append(i)

        deduplicated: list[EventEntry] = []
        removed = 0
        for root, members in groups.items():
            group = [entries[i] for i in members]
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                merged = self._merge_group(group)
                deduplicated.append(merged)
                removed += len(group) - 1
                logger.info(
                    f"Merged {len(group)} duplicates: "
                    + " | ".join(f"[{e.event_entry_id}] {e.artist} @ {e.venue}" for e in group)
                )

        logger.info(
            f"Intra-batch dedup complete: {len(deduplicated)} entries remain, {removed} removed"
        )
        return deduplicated

    def _merge_group(self, group: list[EventEntry]) -> EventEntry:
        candidates_text = "\n\n".join(
            f"Candidate {i + 1}:\n  event_title: {e.event_title}\n  description: {e.description}"
            for i, e in enumerate(group)
        )
        try:
            choice: MergeChoice = self._llm.invoke(
                [
                    {"role": "system", "content": MERGE_SYSTEM_PROMPT},
                    {"role": "user", "content": candidates_text},
                ]
            )
            best_title = choice.event_title
            best_desc = choice.description
        except Exception as e:
            logger.warning(f"LLM merge failed, using first entry values: {e}")
            best_title = group[0].event_title
            best_desc = group[0].description

        base = min(
            group,
            key=lambda e: (e.start_time is None, len(e.venue), len(e.artist)),
        )

        merged_dict: dict[str, Any] = {
            "event_entry_id": self._id_gen.next(),
            "entry_batch_id": base.entry_batch_id,
            "event_title": best_title,
            "description": best_desc,
            "artist": base.artist,
            "venue": base.venue,
            "event_type": "art",
            "multi_day_event": base.multi_day_event,
            "date": base.date,
            "start_time": base.start_time,
            "end_time": base.end_time,
            "genre": base.genre,
            "webpage_contents": base.webpage_contents,
            "address": base.address,
            "lat": base.lat,
            "lng": base.lng,
        }

        merged_dict.update(self._merge_source_slots(group, "tickets"))
        merged_dict.update(self._merge_source_slots(group, "no_tickets"))

        return EventEntry(**merged_dict)

    def _merge_source_slots(self, group: list[EventEntry], prefix: str) -> dict[str, Optional[str]]:
        urls: list[str] = []
        contents: list[str] = []

        for entry in group:
            for slot in range(1, 5):
                url = getattr(entry, f"{prefix}_source_{slot}", None)
                content = getattr(entry, f"{prefix}_webpage_contents_{slot}", None)
                if url and url not in urls:
                    urls.append(url)
                    contents.append(content or "")

        if len(urls) > 4:
            urls = urls[:4]
            contents = contents[:4]

        result: dict[str, Optional[str]] = {}
        for slot in range(1, 5):
            if slot <= len(urls):
                result[f"{prefix}_source_{slot}"] = urls[slot - 1]
                result[f"{prefix}_webpage_contents_{slot}"] = contents[slot - 1]
            else:
                result[f"{prefix}_source_{slot}"] = None
                result[f"{prefix}_webpage_contents_{slot}"] = None
        return result

    def cross_reference_db(self, entries: list[EventEntry]) -> list[EventEntry]:
        logger.info(f"Cross-DB dedup: checking {len(entries)} entries against DB…")
        existing = get_existing_future_entries()

        db_avd: set[tuple] = set()
        db_vdt: set[tuple] = set()

        for r in existing:
            v = _normalize_venue(r["venue"])
            a = _normalize_artist(r["artist"])
            d = r["date"].strip()
            t = _normalize_time(r.get("start_time"))

            db_avd.add((a, v, d))
            if t:
                db_vdt.add((v, d, t))

        net_new: list[EventEntry] = []
        removed = 0
        for entry in entries:
            v = _normalize_venue(entry.venue)
            a = _normalize_artist(entry.artist)
            d = entry.date.strip()
            t = _normalize_time(entry.start_time)

            is_dupe = (a, v, d) in db_avd or (t and (v, d, t) in db_vdt)

            if is_dupe:
                logger.info(
                    f"Cross-DB duplicate skipped: [{entry.event_entry_id}] "
                    f"{entry.artist} @ {entry.venue} on {entry.date}"
                )
                removed += 1
            else:
                net_new.append(entry)

        logger.info(f"Cross-DB dedup: {len(net_new)} net-new entries, {removed} removed")
        return net_new
