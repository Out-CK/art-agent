"""
Direct scraper for art aggregator listing pages and gallery exhibition pages.
Fetches curated URLs and returns page content for downstream parsing.
"""
from __future__ import annotations

import asyncio
from typing import Any

from tools.nimble_extract_tool import NimbleExtractTool
from utils.logger import get_logger
from utils.run_tracker import RunTracker

logger = get_logger(__name__)

CONCURRENCY_LIMIT = 5

# ---------------------------------------------------------------------------
# Art aggregator listing pages — each contains dozens of current exhibitions
# ---------------------------------------------------------------------------
AGGREGATOR_URLS = [
    # NY Art Beat — the most comprehensive NYC art calendar
    "https://www.nyartbeat.com/events",
    "https://www.nyartbeat.com/events/?type=opening",
    # Artsy — NYC shows
    "https://www.artsy.net/shows/new-york-ny",
    # Time Out — NYC art exhibitions
    "https://www.timeout.com/newyork/art/best-art-exhibitions-in-nyc",
    "https://www.timeout.com/newyork/art/best-gallery-shows-in-nyc",
    # The Art Newspaper — NYC events
    "https://www.theartnewspaper.com/new-york-art-events",
    # ArtCards — NYC gallery guide
    "https://artcards.cc/new-york",
    # New York Times — gallery listings
    "https://www.nytimes.com/spotlight/new-york-art-galleries",
    # Artforum — NYC listings
    "https://www.artforum.com/events/new-york",
    # Hyperallergic — exhibitions roundup
    "https://hyperallergic.com/tag/nyc-gallery-scene/",
    # Gallery Guide
    "https://www.galleryguide.org/new-york",
    # See Saw — NYC art calendar
    "https://www.seesaw.nyc/calendar",
]

# ---------------------------------------------------------------------------
# Gallery & museum exhibition pages — structured, reliable data sources
# ---------------------------------------------------------------------------
GALLERY_URLS = [
    # Major museums
    "https://www.moma.org/calendar/exhibitions",
    "https://whitney.org/exhibitions",
    "https://www.guggenheim.org/exhibitions",
    "https://www.metmuseum.org/exhibitions",
    "https://www.brooklynmuseum.org/exhibitions",
    "https://www.newmuseum.org/exhibitions",
    "https://thejewishmuseum.org/exhibitions",
    "https://www.cooperhewitt.org/exhibitions/",
    "https://www.studiomuseum.org/exhibitions",
    "https://madmuseum.org/exhibitions",
    "https://www.elmuseo.org/exhibitions/",
    "https://www.noguchi.org/exhibitions/",
    "https://www.icp.org/exhibitions",
    "https://www.rubin.art/exhibitions/",
    # Blue-chip galleries
    "https://gagosian.com/exhibitions/",
    "https://www.pacegallery.com/exhibitions/",
    "https://www.davidzwirner.com/exhibitions",
    "https://www.hauserwirth.com/exhibitions/",
    "https://www.perrotin.com/exhibitions/current",
    "https://luhringaugustine.com/exhibitions",
    "https://gladstonegallery.com/exhibitions/",
    "https://www.matthewmarks.com/exhibitions",
    "https://www.petzel.com/exhibitions",
    "https://www.blumandpoe.com/exhibitions",
    "https://www.skarstedt.com/exhibitions",
    "https://www.mariangoodman.com/exhibitions",
    "https://spruethmagers.com/exhibitions/",
    "https://www.lehmannmaupin.com/exhibitions",
    "https://www.kurimanzutto.com/exhibitions",
    "https://www.kasminegallery.com/exhibitions",
    "https://www.jamescohan.com/exhibitions",
    "https://www.paulkasmingallery.com/exhibitions",
    # Mid-tier / downtown galleries
    "https://www.55walker.com/exhibitions",
    "https://pioneerworks.org/exhibitions",
    "https://www.printedmatter.org/programs",
    "https://www.nfrfrancoisghebaly.com/exhibitions",
    "https://www.pfrfrancoisghebaly.com/exhibitions",
    "https://www.ppowgallery.com/exhibitions",
    "https://milesmc.gallery/exhibitions",
    "https://www.bitforms.art/exhibitions",
    "https://www.artistsspace.org/exhibitions",
    "https://www.sculpture-center.org/exhibitions",
    "https://www.drawingcenter.org/exhibitions",
    "https://www.swissinstitute.net/exhibitions/",
    "https://www.canadanewyork.com/exhibitions",
    "https://www.47canal.us/exhibitions",
    "https://www.jttnyc.com/exhibitions",
    "https://www.bfriendsandcompanion.com/exhibitions",
]


class ArtDirectScraper:
    """Scrapes aggregator listing pages and gallery exhibition pages."""

    def __init__(self):
        self._extract_tool = NimbleExtractTool()

    def scrape_aggregators(self, tracker: RunTracker | None = None) -> list[dict]:
        """Scrape all aggregator listing pages. Returns list of {url, content} dicts."""
        logger.info(f"Scraping {len(AGGREGATOR_URLS)} aggregator listing pages…")
        results = asyncio.run(self._extract_concurrent(AGGREGATOR_URLS, tracker))
        pages = [r for r in results if r.get("content")]
        logger.info(f"Aggregator scrape: {len(pages)}/{len(AGGREGATOR_URLS)} pages returned content")
        return pages

    def scrape_galleries(self, tracker: RunTracker | None = None) -> list[dict]:
        """Scrape all gallery/museum exhibition pages. Returns list of {url, content} dicts."""
        logger.info(f"Scraping {len(GALLERY_URLS)} gallery exhibition pages…")
        results = asyncio.run(self._extract_concurrent(GALLERY_URLS, tracker))
        pages = [r for r in results if r.get("content")]
        logger.info(f"Gallery scrape: {len(pages)}/{len(GALLERY_URLS)} pages returned content")
        return pages

    async def _extract_concurrent(
        self, urls: list[str], tracker: RunTracker | None
    ) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

        async def extract_one(url: str) -> dict[str, Any]:
            async with semaphore:
                loop = asyncio.get_event_loop()
                if tracker:
                    tracker.inc("nimble_extract_calls")
                try:
                    result = await loop.run_in_executor(
                        None, lambda: self._extract_tool._run(url)
                    )
                    if result.get("content"):
                        if tracker:
                            tracker.inc("nimble_extract_successes")
                    else:
                        if tracker:
                            tracker.inc("nimble_extract_failures")
                    return result
                except Exception as e:
                    logger.error(f"Direct scrape failed for {url}: {e}")
                    if tracker:
                        tracker.inc("nimble_extract_failures")
                    return {"url": url, "content": None}

        return list(await asyncio.gather(*[extract_one(u) for u in urls]))
