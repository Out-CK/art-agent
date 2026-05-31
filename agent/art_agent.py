"""
Art Agent — orchestrator for the Art Gallery & Exhibition discovery pipeline.
Finds NYC gallery openings, solo shows, group exhibitions, and museum shows.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Literal

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from agent.art_batch_parser import ArtBatchParser, ArtEntry
from agent.art_instagram_parser import ArtInstagramParser
from agent.art_tiktok_parser import ArtTikTokParser
from agent.duplicate_finder import DuplicateFinder
from agent.art_link_finder import ArtLinkFinderAgent as LinkFinderAgent
from agent.past_event_archiver import PastEventArchiver
from db.operations import insert_event_entries, insert_web_batch, get_existing_venue_coords
from db.supabase_client import get_supabase_client
from tools.nimble_extract_tool import NimbleExtractTool
from tools.nimble_instagram_tool import NimbleInstagramProfileTool
from tools.nimble_search_tool import NimbleSearchTool
from utils.geocoder import enrich_entries_with_coords
from utils.id_generator import IDGenerator
from utils.logger import get_logger

logger = get_logger(__name__)

CONCURRENCY_LIMIT = 5
MODEL = "claude-sonnet-4-6"

# Curated NYC gallery and museum Instagram accounts
ART_INSTAGRAM_ACCOUNTS = [
    # Major museums
    "themuseumofmodernart",
    "whitneymuseum",
    "guggenheim",
    "metmuseum",
    "brooklynmuseum",
    "newmuseum",
    "thejewishmuseum",
    "cooperhewitt",
    "studiomuseum",
    "madnyc",              # Museum of Arts and Design
    # Blue-chip galleries
    "gagosian",
    "pacegallery",
    "davidzwirner",
    "hauserwirth",
    "perrotin",
    "luhringaugustine",
    "gladstonegallery",
    "matthewmarks",
    "tanyabonakdargallery",
    "petzelgallery",
    "blumandpoe",
    "skarstedtgallery",
    "mariangoodman",
    "spruethmagers",
    # Mid-tier / downtown galleries
    "55walker",
    "pioneerworks",
    "printedmatternyc",
    "bridgewater_art",
    "culturehub_nyc",
    # Art media / aggregators
    "artsy",
    "hyperallergic",
    "artforum",
    "friezearts",
    "artnews",
    "theartnewspaper",
    "nyartbeat",
]

# ---------------------------------------------------------------------------
# Search Plan
# ---------------------------------------------------------------------------

ART_SEARCH_PLAN_PROMPT = """You are an expert at generating search queries to find upcoming NYC
art gallery openings and exhibitions. Generate exactly 40 search queries.

Rules:
- Generate EXACTLY 40 queries — no more, no fewer.
- Each query must have query_type "broad" or "niche".
- Broad queries (~15): General searches covering the NYC art scene, e.g.:
    "NYC art gallery openings this week", "new exhibitions New York City",
    "art shows opening NYC 2026", "gallery openings Manhattan this weekend",
    "NYC museum exhibitions summer 2026", "Chelsea gallery openings",
    "Lower East Side gallery shows NYC", "Bushwick art openings",
    "upcoming art exhibitions New York", "NYC art events this month".
- Niche queries (~25): Gallery/museum-specific or neighborhood-specific, e.g.:
    "Gagosian Gallery New York exhibitions 2026",
    "Pace Gallery NYC upcoming shows",
    "David Zwirner Gallery NYC exhibitions",
    "Hauser & Wirth New York gallery shows",
    "MoMA upcoming exhibitions 2026",
    "Whitney Museum exhibitions New York",
    "New Museum Lower East Side shows",
    "Guggenheim Museum NYC exhibitions",
    "Metropolitan Museum of Art exhibitions 2026",
    "Brooklyn Museum exhibitions 2026",
    "Artsy NYC gallery openings",
    "Frieze New York 2026",
    "Hyperallergic NYC gallery openings",
    "Time Out New York art exhibitions",
    "Perrotin Gallery NYC",
    "Luhring Augustine Gallery Chelsea",
    "Gladstone Gallery NYC exhibitions",
    "Matthew Marks Gallery NYC",
    "55 Walker Street gallery",
    "Bushwick Collective art openings",
    "Pioneer Works Brooklyn exhibitions",
    "Printed Matter NYC art",
    "Art in America NYC gallery shows",
    "NY Art Beat gallery openings",
    "Artforum NYC shows".
- Output a JSON array of objects, each with "query" (string) and "query_type" ("broad" or "niche").
"""


class ArtSearchQuery(BaseModel):
    query: str
    query_type: Literal["broad", "niche"]


class ArtSearchPlan(BaseModel):
    queries: list[ArtSearchQuery]


class ArtSearchPlanAgent:
    def __init__(self):
        self._llm = ChatAnthropic(model=MODEL).with_structured_output(ArtSearchPlan)

    def generate(self) -> ArtSearchPlan:
        logger.info("Generating 40-query Art Search Plan…")
        for attempt in range(2):
            try:
                plan: ArtSearchPlan = self._llm.invoke(
                    [{"role": "user", "content": ART_SEARCH_PLAN_PROMPT}]
                )
                if len(plan.queries) != 40:
                    logger.warning(f"Art search plan returned {len(plan.queries)} queries (expected 40). Attempt {attempt + 1}/2.")
                    if attempt == 1:
                        raise ValueError(f"LLM produced {len(plan.queries)} queries after 2 attempts; expected 40.")
                    continue
                logger.info(f"Art Search Plan generated with {len(plan.queries)} queries")
                return plan
            except ValueError:
                raise
            except Exception as e:
                logger.error(f"Art Search Plan LLM call failed on attempt {attempt + 1}: {e}")
                if attempt == 1:
                    raise
        raise RuntimeError("Art Search Plan generation failed unexpectedly")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ArtAgent:
    def __init__(self):
        self._search_tool = NimbleSearchTool()
        self._extract_tool = NimbleExtractTool()
        self._instagram_profile_tool = NimbleInstagramProfileTool()
        self._supabase = get_supabase_client()

    def run(self) -> None:
        run_start = time.time()
        entry_batch_id = datetime.now().strftime("%m%d%Y_%H%M%S")
        web_batch_id = datetime.now().strftime("%m%d%Y")
        logger.info(f"=== Art Run START | entry_batch_id={entry_batch_id} ===")

        stats = {
            "instagram_profiles_scraped": 0,
            "instagram_entries_parsed": 0,
            "tiktok_videos_collected": 0,
            "tiktok_entries_parsed": 0,
            "queries_executed": 0,
            "pages_round1": 0,
            "pages_round2": 0,
            "entries_parsed": 0,
            "dupes_intrabatch": 0,
            "dupes_crossdb": 0,
            "entries_inserted": 0,
            "entries_archived": 0,
        }

        # Step 0a — Instagram: Scrape gallery/museum profiles
        self._step_log("Step 0a: Instagram Gallery & Museum Scraping")
        social_entries: list[ArtEntry] = []
        id_generator = IDGenerator(self._supabase)
        try:
            raw_profiles = asyncio.run(
                self._scrape_instagram_profiles_concurrent(ART_INSTAGRAM_ACCOUNTS)
            )
            post_pages: list[dict] = []
            for handle, profile_data in raw_profiles:
                if not profile_data:
                    continue
                posts = profile_data.get("posts") or []
                bio = profile_data.get("biography") or ""
                profile_url = (
                    profile_data.get("profile_url")
                    or f"https://www.instagram.com/{handle}/"
                )
                if not posts:
                    continue
                stats["instagram_profiles_scraped"] += 1
                combined_text = f"BIOGRAPHY: {bio}\n\n"
                for post in posts:
                    caption = self._extract_post_caption(post)
                    if caption:
                        combined_text += f"---\nPOST: {caption}\n"
                post_pages.append({
                    "url": profile_url,
                    "handle": handle,
                    "content": combined_text[:30000],
                })
            logger.info(
                f"Instagram: scraped {stats['instagram_profiles_scraped']} profiles "
                f"from {len(ART_INSTAGRAM_ACCOUNTS)} accounts"
            )
            if post_pages:
                ig_entries = ArtInstagramParser().parse(post_pages)
                stats["instagram_entries_parsed"] = len(ig_entries)
                for entry in ig_entries:
                    entry.entry_batch_id = entry_batch_id
                    entry.event_entry_id = id_generator.next()
                social_entries.extend(ig_entries)
                logger.info(f"Instagram: parsed {len(ig_entries)} art entries")
        except Exception as e:
            logger.error(f"Step 0a failed: {e}")

        # Step 0b — TikTok: Scrape gallery/museum accounts and art hashtags
        self._step_log("Step 0b: TikTok Gallery & Museum Scraping")
        try:
            tiktok_parser = ArtTikTokParser()
            raw_videos = tiktok_parser.scrape()
            filtered_videos = tiktok_parser.filter_art_videos(raw_videos)
            stats["tiktok_videos_collected"] = len(filtered_videos)
            logger.info(
                f"TikTok: {len(raw_videos)} raw videos → {len(filtered_videos)} after keyword filter"
            )
            if filtered_videos:
                tk_entries = tiktok_parser.parse(filtered_videos)
                stats["tiktok_entries_parsed"] = len(tk_entries)
                for entry in tk_entries:
                    entry.entry_batch_id = entry_batch_id
                    entry.event_entry_id = id_generator.next()
                social_entries.extend(tk_entries)
                logger.info(f"TikTok: parsed {len(tk_entries)} art entries")
        except Exception as e:
            logger.error(f"Step 0b failed: {e}")

        # Step 1 — Generate Search Plan
        self._step_log("Step 1: Generate Art Search Plan")
        try:
            search_plan = ArtSearchPlanAgent().generate()
        except Exception as e:
            logger.error(f"Step 1 failed: {e}")
            return

        # Step 2 — Web Search Round 1
        self._step_log("Step 2: Web Search Round 1")
        try:
            round1_results = asyncio.run(self._run_searches_concurrent(search_plan.queries))
            stats["queries_executed"] = len(search_plan.queries)
            seen_urls: set[str] = set()
            web_batch: list[dict] = []
            for result in round1_results:
                if result["url"] not in seen_urls:
                    seen_urls.add(result["url"])
                    web_batch.append(result)
            stats["pages_round1"] = len(web_batch)
            logger.info(f"Round 1: {len(web_batch)} unique pages collected")
        except Exception as e:
            logger.error(f"Step 2 failed: {e}")
            web_batch = []

        # Step 3 — Store Round 1 Web Batch
        self._step_log("Step 3: Store Round 1 Web Batch")
        if web_batch:
            try:
                insert_web_batch([
                    {"web_batch_id": web_batch_id, "source_url": r["url"],
                     "query_used": r.get("query_used", ""), "round": 1, "content": r.get("content", "")}
                    for r in web_batch
                ])
            except Exception as e:
                logger.error(f"Step 3 failed: {e}")

        # Step 4 — Find Additional Gallery Links
        self._step_log("Step 4: Link Finder")
        additional_urls: list[str] = []
        try:
            additional_urls = LinkFinderAgent().find_links(web_batch, seen_urls)
            logger.info(f"Link Finder found {len(additional_urls)} additional URLs")
        except Exception as e:
            logger.error(f"Step 4 failed: {e}")

        # Step 5 — Web Extract Round 2
        self._step_log("Step 5: Web Extract Round 2")
        round2_batch: list[dict] = []
        try:
            if additional_urls:
                round2_results = asyncio.run(self._run_extracts_concurrent(additional_urls))
                round2_batch = [r for r in round2_results if r.get("content")]
                stats["pages_round2"] = len(round2_batch)
                logger.info(f"Round 2: {len(round2_batch)} pages extracted")
        except Exception as e:
            logger.error(f"Step 5 failed: {e}")

        # Step 6 — Store Round 2 Web Content
        self._step_log("Step 6: Store Round 2 Web Content")
        if round2_batch:
            try:
                insert_web_batch([
                    {"web_batch_id": web_batch_id, "source_url": r["url"],
                     "query_used": "link_finder", "round": 2, "content": r.get("content", "")}
                    for r in round2_batch
                ])
            except Exception as e:
                logger.error(f"Step 6 failed: {e}")

        # Step 7 — Parse Web Batch into Art Entries
        self._step_log("Step 7: Parse Web Batch")
        full_batch = web_batch + round2_batch
        entry_batch: list[ArtEntry] = []
        try:
            raw_entries = ArtBatchParser().parse(full_batch)
            stats["entries_parsed"] = len(raw_entries)
            for entry in raw_entries:
                entry.entry_batch_id = entry_batch_id
                entry.event_entry_id = id_generator.next()
            # Merge social entries + web entries into one batch
            entry_batch = social_entries + raw_entries
            logger.info(
                f"Parsed {len(raw_entries)} web entries + "
                f"{len(social_entries)} social entries = {len(entry_batch)} total"
            )
        except Exception as e:
            logger.error(f"Step 7 failed: {e}")
            entry_batch = social_entries  # fall back to social-only if web parsing fails

        # Step 7b — Geocoding Enrichment
        self._step_log("Step 7b: Geocoding Enrichment")
        try:
            known_coords = get_existing_venue_coords()
            entry_dicts = [e.model_dump() for e in entry_batch]
            entry_dicts = enrich_entries_with_coords(entry_dicts, known_coords)
            for entry, d in zip(entry_batch, entry_dicts):
                entry.address = d.get("address")
                entry.lat = d.get("lat")
                entry.lng = d.get("lng")
        except Exception as e:
            logger.error(f"Step 7b failed: {e}")

        # Step 7c — Media Enrichment
        self._step_log("Step 7c: Media Enrichment")
        try:
            from agent.media_enricher import MediaEnricher
            entry_batch = MediaEnricher().enrich(entry_batch)
        except Exception as e:
            logger.error(f"Step 7c failed: {e}")

        # Step 8 — Intra-Batch Deduplication
        self._step_log("Step 8: Intra-Batch Deduplication")
        dup_finder = DuplicateFinder(id_generator)
        try:
            pre_count = len(entry_batch)
            entry_batch = dup_finder.deduplicate_batch(entry_batch)
            stats["dupes_intrabatch"] = pre_count - len(entry_batch)
        except Exception as e:
            logger.error(f"Step 8 failed: {e}")

        # Step 9 — Cross-DB Deduplication
        self._step_log("Step 9: Cross-DB Deduplication")
        try:
            pre_count = len(entry_batch)
            entry_batch = dup_finder.cross_reference_db(entry_batch)
            stats["dupes_crossdb"] = pre_count - len(entry_batch)
        except Exception as e:
            logger.error(f"Step 9 failed: {e}")

        # Step 10 — Insert
        self._step_log("Step 10: Insert Art Entries")
        try:
            rows = [e.model_dump() for e in entry_batch]
            stats["entries_inserted"] = insert_event_entries(rows)
        except Exception as e:
            logger.error(f"Step 10 failed: {e}")

        # Step 11 — Archive Past Events
        self._step_log("Step 11: Archive Past Events")
        try:
            stats["entries_archived"] = PastEventArchiver().run()
        except Exception as e:
            logger.error(f"Step 11 failed: {e}")

        duration = time.time() - run_start
        logger.info(
            f"=== Art Run COMPLETE | entry_batch_id={entry_batch_id} | duration={duration:.1f}s ===\n"
            f"  Instagram profiles scraped:  {stats['instagram_profiles_scraped']}\n"
            f"  Instagram entries parsed:    {stats['instagram_entries_parsed']}\n"
            f"  TikTok videos collected:     {stats['tiktok_videos_collected']}\n"
            f"  TikTok entries parsed:       {stats['tiktok_entries_parsed']}\n"
            f"  Web queries executed:        {stats['queries_executed']}\n"
            f"  Pages fetched (Round 1):     {stats['pages_round1']}\n"
            f"  Pages fetched (Round 2):     {stats['pages_round2']}\n"
            f"  Web entries parsed:          {stats['entries_parsed']}\n"
            f"  Intra-batch dupes removed:   {stats['dupes_intrabatch']}\n"
            f"  Cross-DB dupes removed:      {stats['dupes_crossdb']}\n"
            f"  New entries inserted:        {stats['entries_inserted']}\n"
            f"  Entries archived:            {stats['entries_archived']}"
        )

    async def _scrape_instagram_profiles_concurrent(
        self, handles: list[str]
    ) -> list[tuple[str, dict]]:
        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

        async def scrape_one(handle: str) -> tuple[str, dict]:
            async with semaphore:
                loop = asyncio.get_event_loop()
                try:
                    data = await loop.run_in_executor(
                        None, lambda: self._instagram_profile_tool._run(handle)
                    )
                    return (handle, data)
                except Exception as e:
                    logger.error(f"Instagram profile failed for @{handle}: {e}")
                    return (handle, {})

        return list(await asyncio.gather(*[scrape_one(h) for h in handles]))

    @staticmethod
    def _extract_post_caption(post: dict) -> str:
        """Pull the caption text out of a post object regardless of schema shape."""
        for key in ("caption", "description", "text", "edge_media_to_caption"):
            val = post.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            if isinstance(val, dict):
                edges = val.get("edges") or []
                if edges:
                    node_text = (edges[0].get("node") or {}).get("text") or ""
                    if node_text:
                        return node_text.strip()
        return ""

    async def _run_searches_concurrent(self, queries) -> list[dict]:
        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

        async def search_one(sq):
            async with semaphore:
                loop = asyncio.get_event_loop()
                try:
                    results = await loop.run_in_executor(
                        None, lambda: self._search_tool._run(sq.query, sq.query_type)
                    )
                    return [{**r, "query_used": sq.query} for r in results]
                except Exception as e:
                    logger.error(f"Search failed for '{sq.query}': {e}")
                    return []

        results_nested = await asyncio.gather(*[search_one(sq) for sq in queries])
        return [item for sublist in results_nested for item in sublist]

    async def _run_extracts_concurrent(self, urls: list[str]) -> list[dict]:
        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

        async def extract_one(url: str):
            async with semaphore:
                loop = asyncio.get_event_loop()
                try:
                    return await loop.run_in_executor(None, lambda: self._extract_tool._run(url))
                except Exception as e:
                    logger.error(f"Extract failed for '{url}': {e}")
                    return {"url": url, "content": None}

        return list(await asyncio.gather(*[extract_one(u) for u in urls]))

    @staticmethod
    def _step_log(step_name: str) -> None:
        logger.info(f"--- {step_name} ---")
