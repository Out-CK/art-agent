import time

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from utils.logger import get_logger

logger = get_logger(__name__)

eastern = pytz.timezone("America/New_York")


def run_art_run() -> None:
    """Entry point for the daily Art Gallery Run."""
    from agent.art_agent import ArtAgent
    logger.info("Scheduled Art Run triggered")
    try:
        ArtAgent().run()
    except Exception as e:
        logger.error(f"Scheduled Art Run failed: {e}", exc_info=True)


def start_scheduler() -> None:
    """Start the APScheduler and block until Ctrl+C."""
    scheduler = BackgroundScheduler(timezone=eastern)
    scheduler.add_job(
        run_art_run,
        trigger=CronTrigger(hour=9, minute=0, timezone=eastern),
        id="daily_art_run",
        name="Daily Art Gallery Run",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler started — Art Run fires daily at 09:00 America/New_York")
    logger.info("Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down scheduler…")
        scheduler.shutdown()
        logger.info("Scheduler stopped")
