"""
ArtTikTokParser — converts raw TikTok video records from gallery/museum accounts
and art hashtags into ArtEntry objects.

Each record is a dict with at minimum:
    caption        : str   (video caption text)
    creator_handle : str   (TikTok @handle, without the @)
    post_id        : str   (numeric video ID)
    description    : str   (full description from video page, if fetched; else same as caption)
    posted_at      : str   (ISO date string when video was posted)
"""
from __future__ import annotations

import asyncio
import re
from datetime import date
from typing import Any, List

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from agent.art_batch_parser import ArtEntry
from tools.nimble_tiktok_tool import NimbleTikTokAccountTool, NimbleTikTokHashtagTool
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 10
CONCURRENCY_LIMIT = 4

# Curated NYC gallery and museum TikTok accounts
ART_TIKTOK_ACCOUNTS = [
    # Major museums
    "themuseumofmodernart",
    "whitneymuseum",
    "guggenheim",
    "metmuseum",
    "brooklynmuseum",
    "newmuseum",
    "thejewishmuseum",
    "cooperhewitt",
    "thestudiomuseum",
    "miacnew",            # Museum of Arts and Design
    # Galleries
    "gagosian",
    "pacegallery",
    "davidzwirner",
    "hauserwirth",
    "perrotin",
    "luhringaugustine",
    "gladstonegallery",
    "matthewmarks",
    "tanyabonakdar",
    "petzelgallery",
    # Art media / aggregators
    "artsy",
    "hyperallergic",
    "friezearts",
    "artnews",
    "theartnewspaper",
    # NYC art scene
    "pioneerworks",
    "printedmatter",
    "nycgo",
]

# Art-specific TikTok hashtags
ART_HASHTAGS = [
    "nycart",
    "nycgallery",
    "artnyc",
    "galleryopening",
    "artopening",
    "nycartscene",
    "artexhibition",
    "contemporaryartnyc",
    "nycmuseum",
]

# Caption must mention art/gallery context to be worth parsing
_ART_KEYWORDS_RE = re.compile(
    r"\b(exhibition|gallery|opening|museum|artwork|installation|sculpture|"
    r"painting|photography|artist|curator|vernissage|art fair|solo show|"
    r"group show|moma|whitney|guggenheim|met museum|brooklyn museum|"
    r"gagosian|pace gallery|david zwirner|hauser|perrotin|"
    r"new museum|pioneer works|artsy|frieze)\b",
    re.IGNORECASE,
)


class ArtEntryList(BaseModel):
    entries: List[ArtEntry]


SYSTEM_PROMPT = """You are an art exhibition data extraction specialist. You will receive TikTok
videos (captions, descriptions, and source URLs) from NYC galleries, museums, and art accounts.
Extract all upcoming NYC art gallery openings and exhibitions mentioned.

Rules:
- Create a SEPARATE entry for each distinct exhibition or opening event.
- Only extract events that are UPCOMING (future dates relative to today). Skip past events.
- Only extract events in NYC (Manhattan, Brooklyn, Queens, Bronx, Staten Island).
- Set event_type = "art" always. Skip concerts, theater, comedy, film, and conferences.
- event_title format: "[Exhibition Name] at [Gallery]" — e.g. "New Works at Gagosian"
  For a solo show: "[Artist Name]: [Show Title] at [Gallery]"
  For a group show with no distinct title: "Group Exhibition at [Gallery]"
- artist: the featured artist(s) name(s), comma-separated. For group shows use "Various Artists".
- venue: the gallery or museum name only — no city/state suffix.
- date: The OPENING date (or start date of the run) in "MM-DD-YYYY" format.
- start_time: opening reception time if given, in "00:00am/pm" format (e.g. "06:00pm").
  Leave null if no specific reception time is listed.
- genre: assign exactly ONE from:
  Contemporary, Modern, Photography, Sculpture, Painting, Drawing, Mixed Media,
  Installation, Video Art, Performance, Printmaking, Ceramics, Textile, Street Art, Other.
- description: a 1-2 sentence description from the caption.
- Populate no_tickets_source_1 with the TikTok video URL.
- Populate no_tickets_webpage_contents_1 with the caption/description text.
- If exhibition name, gallery/venue, OR date cannot be confidently extracted, SKIP that entry.
- DO NOT set event_entry_id or entry_batch_id — leave them as empty strings "".
- Return a JSON object with key "entries" containing an array of ArtEntry objects.
- Be conservative: only extract events you are highly confident about.
"""


class ArtTikTokParser:
    def __init__(self):
        self._llm = ChatAnthropic(model=MODEL).with_structured_output(ArtEntryList)
        self._hashtag_tool = NimbleTikTokHashtagTool()
        self._account_tool = NimbleTikTokAccountTool()

    # ------------------------------------------------------------------
    # Scraping
    # ------------------------------------------------------------------

    def scrape(self) -> list[dict[str, Any]]:
        """Scrape curated art accounts and hashtags. Returns raw video records."""
        logger.info(
            f"ArtTikTokParser: scraping {len(ART_TIKTOK_ACCOUNTS)} accounts "
            f"+ {len(ART_HASHTAGS)} hashtags"
        )
        records = asyncio.run(self._scrape_concurrent())
        # Deduplicate by post_id
        seen: set[str] = set()
        unique: list[dict] = []
        for r in records:
            pid = r.get("post_id") or ""
            if pid and pid not in seen:
                seen.add(pid)
                unique.append(r)
            elif not pid:
                unique.append(r)
        logger.info(f"ArtTikTokParser: {len(unique)} unique posts collected")
        return unique

    async def _scrape_concurrent(self) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        loop = asyncio.get_event_loop()

        async def scrape_account(handle: str) -> list[dict]:
            async with semaphore:
                try:
                    data = await loop.run_in_executor(
                        None, lambda: self._account_tool._run(handle)
                    )
                    return self._normalize_account_posts(handle, data)
                except Exception as e:
                    logger.error(f"TikTok account scrape failed for @{handle}: {e}")
                    return []

        async def scrape_hashtag(tag: str) -> list[dict]:
            async with semaphore:
                try:
                    items = await loop.run_in_executor(
                        None, lambda: self._hashtag_tool._run(tag)
                    )
                    for item in items:
                        item.setdefault("source_tag", tag)
                    return items
                except Exception as e:
                    logger.error(f"TikTok hashtag scrape failed for #{tag}: {e}")
                    return []

        tasks = (
            [scrape_account(h) for h in ART_TIKTOK_ACCOUNTS]
            + [scrape_hashtag(t) for t in ART_HASHTAGS]
        )
        results_nested = await asyncio.gather(*tasks)
        return [item for sublist in results_nested for item in sublist]

    @staticmethod
    def _normalize_account_posts(handle: str, data: dict[str, Any]) -> list[dict]:
        posts = data.get("top_posts_data") or []
        records = []
        for post in posts:
            post_id = post.get("post_id") or ""
            description = post.get("description") or ""
            records.append({
                "post_id": post_id,
                "caption": description,
                "description": description,
                "creator_handle": handle,
                "posted_at": post.get("create_date") or "",
                "source_tag": f"art_account:{handle}",
                "tiktok_url": post.get("post_url") or (
                    f"https://www.tiktok.com/@{handle}/video/{post_id}" if post_id else ""
                ),
            })
        return records

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    @staticmethod
    def filter_art_videos(videos: list[dict]) -> list[dict]:
        """Keep only videos whose caption mentions art/gallery context."""
        return [
            v for v in videos
            if _ART_KEYWORDS_RE.search((v.get("caption") or "").strip())
        ]

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse(self, records: list[dict]) -> list[ArtEntry]:
        """Parse TikTok video records into ArtEntry objects."""
        logger.info(
            f"ArtTikTokParser parsing {len(records)} videos "
            f"in batches of {BATCH_SIZE}…"
        )
        all_entries: list[ArtEntry] = []

        for batch_start in range(0, len(records), BATCH_SIZE):
            batch = records[batch_start: batch_start + BATCH_SIZE]
            try:
                entries = self._parse_batch(batch)
                logger.info(
                    f"Batch {batch_start}–{batch_start + len(batch)}: "
                    f"parsed {len(entries)} entries"
                )
                all_entries.extend(entries)
            except Exception as e:
                logger.error(
                    f"ArtTikTokParser batch "
                    f"{batch_start}–{batch_start + len(batch)} failed: {e}"
                )

        logger.info(f"ArtTikTokParser total entries parsed: {len(all_entries)}")
        return all_entries

    def _parse_batch(self, batch: list[dict]) -> list[ArtEntry]:
        videos_text = ""
        for rec in batch:
            handle = rec.get("creator_handle", "unknown")
            post_id = rec.get("post_id", "")
            caption = rec.get("description") or rec.get("caption") or ""
            tiktok_url = rec.get("tiktok_url") or (
                f"https://www.tiktok.com/@{handle}/video/{post_id}" if post_id else ""
            )
            videos_text += (
                f"\n\n---\n"
                f"TIKTOK URL: {tiktok_url}\n"
                f"CREATOR: @{handle}\n"
                f"POSTED AT: {rec.get('posted_at', '')}\n"
                f"CAPTION/DESCRIPTION:\n{caption[:3000]}"
            )

        result: ArtEntryList = self._llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Today's date is {date.today().strftime('%m-%d-%Y')}. "
                    f"Extract NYC art exhibition entries from these TikTok videos:{videos_text}"
                ),
            },
        ])
        return result.entries or []
