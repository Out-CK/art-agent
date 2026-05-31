"""
Media Enricher — fallback image search for entries missing a media_url.

After the parser extracts events, some entries may lack a media_url (the page
had no suitable images). This module searches the web for a relevant image
and uses Claude to pick the best one.
"""
from __future__ import annotations

import re
from typing import Optional

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from tools.nimble_search_tool import NimbleSearchTool
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_LOOKUPS = 15

SYSTEM_PROMPT = """You are an image URL extractor. You will receive search result snippets about an event or artist.
Your job is to find the single best image URL from the content.

Rules:
- Look for URLs ending in .jpg, .jpeg, .png, .webp, or from known image CDNs
  (e.g., images.squarespace-cdn.com, img.evbuc.com, cdn.eventbrite.com, i.scdn.co, s3.amazonaws.com).
- Prefer: artist/performer photos, event posters, show artwork, venue hero images.
- Skip: tiny icons, logos, tracking pixels, social media buttons, ad banners, placeholder images.
- Return the single best image URL, or null if none found.
"""


class MediaResult(BaseModel):
    media_url: Optional[str] = None


class MediaEnricher:
    def __init__(self):
        self._search = NimbleSearchTool()
        self._llm = ChatAnthropic(model=MODEL).with_structured_output(MediaResult)

    def enrich(self, entries: list) -> list:
        """For entries missing media_url, search for an image. Caps at MAX_LOOKUPS."""
        missing = [e for e in entries if not e.media_url and getattr(e, "artist", None)]
        to_process = missing[:MAX_LOOKUPS]

        if not to_process:
            logger.info("MediaEnricher: all entries already have media_url or no artist")
            return entries

        logger.info(f"MediaEnricher: searching for images for {len(to_process)} entries")
        found = 0
        for entry in to_process:
            try:
                url = self._find_image(entry)
                if url:
                    entry.media_url = url
                    found += 1
            except Exception as e:
                logger.debug(f"MediaEnricher failed for '{entry.artist}': {e}")

        logger.info(f"MediaEnricher: found images for {found}/{len(to_process)} entries")
        return entries

    def _find_image(self, entry) -> Optional[str]:
        query = f"{entry.artist} {entry.venue} photo"
        results = self._search._run(query, "niche")
        if not results:
            return None

        # First try: extract image URLs directly from search snippets using regex
        for r in results[:3]:
            content = r.get("content", "")
            img_urls = re.findall(
                r'https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|webp)(?:\?[^\s"\'<>]*)?',
                content,
                re.IGNORECASE,
            )
            if img_urls:
                return img_urls[0]

        # Fallback: ask Claude to extract from combined snippets
        snippets = "\n---\n".join(
            f"URL: {r.get('url', '')}\nContent: {r.get('content', '')[:500]}"
            for r in results[:3]
        )
        result: MediaResult = self._llm.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Find the best image URL for: {entry.artist} at {entry.venue}\n\nSearch results:\n{snippets}",
                },
            ]
        )
        return result.media_url
