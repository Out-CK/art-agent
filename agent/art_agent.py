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
from agent.art_direct_scraper import ArtDirectScraper
from agent.art_instagram_parser import ArtInstagramParser
from agent.duplicate_finder import DuplicateFinder
from agent.art_link_finder import ArtLinkFinderAgent as LinkFinderAgent
from agent.past_event_archiver import PastEventArchiver
from db.operations import (
    insert_event_entries, insert_web_batch, get_existing_venue_coords,
    get_recent_instagram_scrapes, upsert_instagram_scrape,
)
from db.supabase_client import get_supabase_client
from tools.nimble_extract_tool import NimbleExtractTool
from tools.nimble_instagram_tool import NimbleInstagramProfileTool
from tools.nimble_search_tool import NimbleSearchTool
from utils.geocoder import enrich_entries_with_coords
from utils.id_generator import IDGenerator
from utils.logger import get_logger
from utils.run_tracker import RunTracker

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
art gallery openings and exhibitions. Generate exactly 25 search queries.

Rules:
- Generate EXACTLY 25 queries — no more, no fewer.
- Each query must have query_type "broad" or "niche".
- Broad queries (~10): General searches covering the NYC art scene, e.g.:
    "NYC art gallery openings this week", "new exhibitions New York City",
    "art shows opening NYC 2026", "gallery openings Manhattan this weekend",
    "NYC museum exhibitions summer 2026", "Chelsea gallery openings",
    "Lower East Side gallery shows NYC", "Bushwick art openings",
    "upcoming art exhibitions New York", "NYC art events this month".
- Niche queries (~15): Neighborhood-specific or aggregator-specific, e.g.:
    "Artsy NYC gallery openings",
    "Frieze New York 2026",
    "Hyperallergic NYC gallery openings",
    "Time Out New York art exhibitions",
    "NY Art Beat gallery openings",
    "Chelsea gallery openings this week",
    "Bushwick Collective art openings",
    "Pioneer Works Brooklyn exhibitions",
    "Lower East Side gallery openings NYC",
    "Tribeca art exhibitions 2026",
    "SoHo gallery shows NYC",
    "Williamsburg art openings Brooklyn",
    "Art in America NYC gallery shows",
    "Artforum NYC shows",
    "Printed Matter NYC art".
  Avoid queries for specific galleries whose Instagram accounts are already scraped
  (Gagosian, Pace, Zwirner, Hauser & Wirth, Perrotin, etc.) — those are covered by Instagram.
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
        logger.info("Generating 25-query Art Search Plan…")
        for attempt in range(2):
            try:
                plan: ArtSearchPlan = self._llm.invoke(
                    [{"role": "user", "content": ART_SEARCH_PLAN_PROMPT}]
                )
                if len(plan.queries) != 25:
                    logger.warning(f"Art search plan returned {len(plan.queries)} queries (expected 25). Attempt {attempt + 1}/2.")
                    if attempt == 1:
                        raise ValueError(f"LLM produced {len(plan.queries)} queries after 2 attempts; expected 25.")
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
        tracker = RunTracker(agent_name="art", run_type="web_search").start()
        run_start = time.time()
        entry_batch_id = datetime.now().strftime("%m%d%Y_%H%M%S")
        web_batch_id = datetime.now().strftime("%m%d%Y")
        tracker.entry_batch_id = entry_batch_id
        logger.info(f"=== Art Run START | entry_batch_id={entry_batch_id} ===")

        stats = {
            "instagram_profiles_scraped": 0,
            "instagram_entries_parsed": 0,
            "aggregator_pages_scraped": 0,
            "aggregator_entries_parsed": 0,
            "gallery_pages_scraped": 0,
            "gallery_entries_parsed": 0,
            "queries_executed": 0,
            "pages_round1": 0,
            "pages_round2": 0,
            "entries_parsed": 0,
            "dupes_intrabatch": 0,
            "dupes_crossdb": 0,
            "entries_inserted": 0,
            "entries_archived": 0,
        }

        run_status = "success"
        error_msg = None

        # Step 0a — Instagram: Scrape gallery/museum profiles
        self._step_log("Step 0a: Instagram Gallery & Museum Scraping")
        social_entries: list[ArtEntry] = []
        id_generator = IDGenerator(self._supabase)
        try:
            recently_scraped = get_recent_instagram_scrapes(days=5)
            accounts_to_scrape = [h for h in ART_INSTAGRAM_ACCOUNTS if h not in recently_scraped]
            logger.info(
                f"Instagram: {len(accounts_to_scrape)} accounts to scrape "
                f"({len(recently_scraped)} skipped — scraped within 5 days)"
            )
            raw_profiles = asyncio.run(
                self._scrape_instagram_profiles_concurrent(accounts_to_scrape, tracker)
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
                upsert_instagram_scrape(handle)
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
            tracker.set("instagram_profiles_scraped", stats["instagram_profiles_scraped"])
            logger.info(
                f"Instagram: scraped {stats['instagram_profiles_scraped']} profiles "
                f"from {len(accounts_to_scrape)} accounts"
            )
            if post_pages:
                ig_entries = ArtInstagramParser().parse(post_pages)
                stats["instagram_entries_parsed"] = len(ig_entries)
                tracker.set("instagram_entries_parsed", len(ig_entries))
                for entry in ig_entries:
                    entry.entry_batch_id = entry_batch_id
                    entry.event_entry_id = id_generator.next()
                social_entries.extend(ig_entries)
                logger.info(f"Instagram: parsed {len(ig_entries)} art entries")
        except Exception as e:
            logger.error(f"Step 0a failed: {e}")

        # Step 0b — Direct Aggregator Scraping
        self._step_log("Step 0b: Direct Aggregator Scraping")
        aggregator_pages: list[dict] = []
        aggregator_entries: list[ArtEntry] = []
        try:
            scraper = ArtDirectScraper()
            aggregator_pages = scraper.scrape_aggregators(tracker)
            stats["aggregator_pages_scraped"] = len(aggregator_pages)
            tracker.set("aggregator_pages_scraped", len(aggregator_pages))
            if aggregator_pages:
                insert_web_batch([
                    {"web_batch_id": web_batch_id, "source_url": r["url"],
                     "query_used": "aggregator_direct", "round": 0, "content": (r.get("content") or "")[:10000]}
                    for r in aggregator_pages
                ])
                agg_raw = ArtBatchParser().parse(aggregator_pages)
                stats["aggregator_entries_parsed"] = len(agg_raw)
                tracker.set("aggregator_entries_parsed", len(agg_raw))
                for entry in agg_raw:
                    entry.entry_batch_id = entry_batch_id
                    entry.event_entry_id = id_generator.next()
                aggregator_entries = agg_raw
                logger.info(f"Aggregator scrape: parsed {len(agg_raw)} entries from {len(aggregator_pages)} pages")
        except Exception as e:
            logger.error(f"Step 0b failed: {e}")

        # Step 0c — Direct Gallery Website Scraping
        self._step_log("Step 0c: Direct Gallery Website Scraping")
        gallery_pages: list[dict] = []
        gallery_entries: list[ArtEntry] = []
        try:
            if not aggregator_pages:
                scraper = ArtDirectScraper()
            gallery_pages = scraper.scrape_galleries(tracker)
            stats["gallery_pages_scraped"] = len(gallery_pages)
            tracker.set("gallery_pages_scraped", len(gallery_pages))
            if gallery_pages:
                insert_web_batch([
                    {"web_batch_id": web_batch_id, "source_url": r["url"],
                     "query_used": "gallery_direct", "round": 0, "content": (r.get("content") or "")[:10000]}
                    for r in gallery_pages
                ])
                gal_raw = ArtBatchParser().parse(gallery_pages)
                stats["gallery_entries_parsed"] = len(gal_raw)
                tracker.set("gallery_entries_parsed", len(gal_raw))
                for entry in gal_raw:
                    entry.entry_batch_id = entry_batch_id
                    entry.event_entry_id = id_generator.next()
                gallery_entries = gal_raw
                logger.info(f"Gallery scrape: parsed {len(gal_raw)} entries from {len(gallery_pages)} pages")
        except Exception as e:
            logger.error(f"Step 0c failed: {e}")

        # Step 1 — Generate Search Plan
        self._step_log("Step 1: Generate Art Search Plan")
        try:
            search_plan = ArtSearchPlanAgent().generate()
        except Exception as e:
            logger.error(f"Step 1 failed: {e}")
            tracker.finish(status="failed", error_message=str(e))
            return

        # Step 2 — Web Search Round 1
        self._step_log("Step 2: Web Search Round 1")
        try:
            round1_results = asyncio.run(
                self._run_searches_concurrent(search_plan.queries, tracker)
            )
            stats["queries_executed"] = len(search_plan.queries)
            tracker.set("queries_executed", len(search_plan.queries))
            seen_urls: set[str] = set()
            web_batch: list[dict] = []
            for result in round1_results:
                if result["url"] not in seen_urls:
                    seen_urls.add(result["url"])
                    web_batch.append(result)
            stats["pages_round1"] = len(web_batch)
            tracker.set("pages_fetched_round1", len(web_batch))
            logger.info(f"Round 1: {len(web_batch)} unique pages collected")
        except Exception as e:
            logger.error(f"Step 2 failed: {e}")
            web_batch = []
            run_status = "partial"
            error_msg = f"Step 2: {e}"

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
                round2_results = asyncio.run(
                    self._run_extracts_concurrent(additional_urls, tracker)
                )
                round2_batch = [r for r in round2_results if r.get("content")]
                stats["pages_round2"] = len(round2_batch)
                tracker.set("pages_fetched_round2", len(round2_batch))
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
            tracker.set("raw_entries_parsed", len(raw_entries))
            for entry in raw_entries:
                entry.entry_batch_id = entry_batch_id
                entry.event_entry_id = id_generator.next()
            # Merge all entry sources into one batch
            entry_batch = social_entries + aggregator_entries + gallery_entries + raw_entries
            logger.info(
                f"Parsed {len(raw_entries)} web entries + "
                f"{len(social_entries)} social entries + "
                f"{len(aggregator_entries)} aggregator entries + "
                f"{len(gallery_entries)} gallery entries = {len(entry_batch)} total"
            )
        except Exception as e:
            logger.error(f"Step 7 failed: {e}")
            entry_batch = social_entries + aggregator_entries + gallery_entries

        # Step 7b — Geocoding Enrichment
        self._step_log("Step 7b: Geocoding Enrichment")
        try:
            known_coords = get_existing_venue_coords()
            entry_dicts = [e.model_dump() for e in entry_batch]
            cached_before = sum(1 for d in entry_dicts if d.get("lat"))
            entry_dicts = enrich_entries_with_coords(entry_dicts, known_coords)
            cached_after = sum(1 for d in entry_dicts if d.get("lat"))
            tracker.set("venues_geocoded", cached_after - cached_before)
            tracker.set("venues_from_cache", cached_before)
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
            enricher = MediaEnricher()
            pre_media = sum(1 for e in entry_batch if getattr(e, "media_url", None))
            entry_batch = enricher.enrich(entry_batch)
            post_media = sum(1 for e in entry_batch if getattr(e, "media_url", None))
            tracker.set("media_enricher_lookups", len([e for e in entry_batch if not getattr(e, "media_url", None)]) + (post_media - pre_media))
            tracker.set("media_enricher_found", post_media - pre_media)
        except Exception as e:
            logger.error(f"Step 7c failed: {e}")

        # Step 8 — Intra-Batch Deduplication
        self._step_log("Step 8: Intra-Batch Deduplication")
        dup_finder = DuplicateFinder(id_generator)
        try:
            pre_count = len(entry_batch)
            entry_batch = dup_finder.deduplicate_batch(entry_batch)
            stats["dupes_intrabatch"] = pre_count - len(entry_batch)
            tracker.set("intra_batch_dupes_removed", pre_count - len(entry_batch))
        except Exception as e:
            logger.error(f"Step 8 failed: {e}")

        # Step 9 — Cross-DB Deduplication
        self._step_log("Step 9: Cross-DB Deduplication")
        try:
            pre_count = len(entry_batch)
            entry_batch = dup_finder.cross_reference_db(entry_batch)
            stats["dupes_crossdb"] = pre_count - len(entry_batch)
            tracker.set("cross_db_dupes_removed", pre_count - len(entry_batch))
        except Exception as e:
            logger.error(f"Step 9 failed: {e}")

        # Count missing fields before insert
        tracker.count_missing_fields(entry_batch)

        # Step 10 — Insert
        self._step_log("Step 10: Insert Art Entries")
        try:
            fresh_id_gen = IDGenerator(self._supabase)
            for entry in entry_batch:
                entry.event_entry_id = fresh_id_gen.next()
            rows = [e.model_dump() for e in entry_batch]
            stats["entries_inserted"] = insert_event_entries(rows)
            tracker.set("entries_inserted", stats["entries_inserted"])
        except Exception as e:
            logger.error(f"Step 10 failed: {e}")
            run_status = "failed"
            error_msg = f"Step 10: {e}"

        # Step 11 — Archive Past Events
        self._step_log("Step 11: Archive Past Events")
        try:
            stats["entries_archived"] = PastEventArchiver().run()
            tracker.set("entries_archived", stats["entries_archived"])
        except Exception as e:
            logger.error(f"Step 11 failed: {e}")

        duration = time.time() - run_start
        logger.info(
            f"=== Art Run COMPLETE | entry_batch_id={entry_batch_id} | duration={duration:.1f}s ===\n"
            f"  Instagram profiles scraped:  {stats['instagram_profiles_scraped']}\n"
            f"  Instagram entries parsed:    {stats['instagram_entries_parsed']}\n"
            f"  Aggregator pages scraped:    {stats['aggregator_pages_scraped']}\n"
            f"  Aggregator entries parsed:   {stats['aggregator_entries_parsed']}\n"
            f"  Gallery pages scraped:       {stats['gallery_pages_scraped']}\n"
            f"  Gallery entries parsed:      {stats['gallery_entries_parsed']}\n"
            f"  Web queries executed:        {stats['queries_executed']}\n"
            f"  Pages fetched (Round 1):     {stats['pages_round1']}\n"
            f"  Pages fetched (Round 2):     {stats['pages_round2']}\n"
            f"  Web entries parsed:          {stats['entries_parsed']}\n"
            f"  Intra-batch dupes removed:   {stats['dupes_intrabatch']}\n"
            f"  Cross-DB dupes removed:      {stats['dupes_crossdb']}\n"
            f"  New entries inserted:        {stats['entries_inserted']}\n"
            f"  Entries archived:            {stats['entries_archived']}"
        )

        tracker.finish(status=run_status, error_message=error_msg)

    async def _scrape_instagram_profiles_concurrent(
        self, handles: list[str], tracker: RunTracker
    ) -> list[tuple[str, dict]]:
        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

        async def scrape_one(handle: str) -> tuple[str, dict]:
            async with semaphore:
                loop = asyncio.get_event_loop()
                tracker.inc("nimble_instagram_calls")
                try:
                    data = await loop.run_in_executor(
                        None, lambda: self._instagram_profile_tool._run(handle)
                    )
                    tracker.inc("nimble_instagram_successes")
                    return (handle, data)
                except Exception as e:
                    logger.error(f"Instagram profile failed for @{handle}: {e}")
                    tracker.inc("nimble_instagram_failures")
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

    async def _run_searches_concurrent(self, queries, tracker: RunTracker) -> list[dict]:
        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

        async def search_one(sq):
            async with semaphore:
                loop = asyncio.get_event_loop()
                tracker.inc("nimble_search_calls")
                try:
                    results = await loop.run_in_executor(
                        None, lambda: self._search_tool._run(sq.query, sq.query_type)
                    )
                    tracker.inc("nimble_search_successes")
                    return [{**r, "query_used": sq.query} for r in results]
                except Exception as e:
                    logger.error(f"Search failed for '{sq.query}': {e}")
                    tracker.inc("nimble_search_failures")
                    return []

        results_nested = await asyncio.gather(*[search_one(sq) for sq in queries])
        return [item for sublist in results_nested for item in sublist]

    async def _run_extracts_concurrent(self, urls: list[str], tracker: RunTracker) -> list[dict]:
        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

        async def extract_one(url: str):
            async with semaphore:
                loop = asyncio.get_event_loop()
                tracker.inc("nimble_extract_calls")
                try:
                    result = await loop.run_in_executor(
                        None, lambda: self._extract_tool._run(url)
                    )
                    if result.get("content"):
                        tracker.inc("nimble_extract_successes")
                    else:
                        tracker.inc("nimble_extract_failures")
                    return result
                except Exception as e:
                    logger.error(f"Extract failed for '{url}': {e}")
                    tracker.inc("nimble_extract_failures")
                    return {"url": url, "content": None}

        return list(await asyncio.gather(*[extract_one(u) for u in urls]))

    @staticmethod
    def _step_log(step_name: str) -> None:
        logger.info(f"--- {step_name} ---")
