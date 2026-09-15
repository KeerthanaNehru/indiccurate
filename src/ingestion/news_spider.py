"""Scrapy spider for Hindu Tamil Thisai (hindutamil.in).

Site choice rationale: ``hindutamil.in/robots.txt`` is fully permissive
(``User-agent: *`` / ``Allow: /``, no disallowed paths), and every article
page embeds a clean ``schema.org/NewsArticle`` JSON-LD block with the
headline, full body text, and publish date — far more robust to scrape than
hand-picked CSS selectors on a React-rendered page.

Crawl strategy: seed from the Google-News-style ``news_sitemap.xml`` (recent
article URLs, refreshed continuously), then parse each article's JSON-LD.

Run standalone:
    python -m src.ingestion.news_spider --output data/raw/news_raw.jsonl --max-articles 50
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from src.common.paths import RAW_DIR
from src.common.schema import make_text_record
from src.common.stats import log_stage_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SITEMAP_URL = "https://www.hindutamil.in/news_sitemap.xml"
SOURCE_NAME = "scrapy:hindutamil.in"
LICENSE_STR = "All rights reserved (publisher) — research/educational use; verify terms before redistribution."
USER_AGENT = "IndicCurate-research-bot/0.1 (+https://github.com/; contact: dataset curation research project)"


class HinduTamilNewsSpider(scrapy.Spider):
    """Crawls hindutamil.in articles seeded from its news sitemap."""

    name = "hindutamil_news"
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "USER_AGENT": USER_AGENT,
        "DOWNLOAD_DELAY": 1.0,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "AUTOTHROTTLE_ENABLED": True,
        "RETRY_TIMES": 2,
        "LOG_LEVEL": "INFO",
    }

    def __init__(self, sitemap_url: str = SITEMAP_URL, max_articles: Optional[int] = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sitemap_url = sitemap_url
        self.max_articles = max_articles
        self._yielded = 0
        self._seen_sitemap = False

    async def start(self) -> Any:
        """Kick off the crawl by requesting the news sitemap XML (Scrapy >= 2.13 API)."""
        yield scrapy.Request(self.sitemap_url, callback=self.parse_sitemap)

    def start_requests(self) -> Iterator[scrapy.Request]:
        """Kick off the crawl (legacy sync API, kept for Scrapy < 2.13 compatibility)."""
        yield scrapy.Request(self.sitemap_url, callback=self.parse_sitemap)

    def parse_sitemap(self, response: scrapy.http.Response) -> Iterator[scrapy.Request]:
        """Extract article URLs from the news sitemap and schedule article requests.

        Only ``max_articles`` requests are scheduled up front (rather than
        relying on a counter checked inside ``parse_article``), since all
        sitemap URLs would otherwise be scheduled instantly, well before any
        article response — and thus any yield-count check — comes back.
        """
        response.selector.remove_namespaces()
        urls = response.xpath("//url/loc/text()").getall()
        logger.info("Sitemap listed %d article URLs.", len(urls))
        if self.max_articles is not None:
            urls = urls[: self.max_articles]
        for url in urls:
            yield scrapy.Request(url, callback=self.parse_article)

    def parse_article(self, response: scrapy.http.Response) -> Iterator[Dict[str, Any]]:
        """Extract the NewsArticle JSON-LD block from an article page and yield a record."""
        if self.max_articles is not None and self._yielded >= self.max_articles:
            return

        for script in response.xpath('//script[@type="application/ld+json"]/text()').getall():
            try:
                data = json.loads(script)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            if data.get("@type") != "NewsArticle":
                continue

            body = (data.get("articleBody") or "").strip()
            if not body:
                continue

            authors = data.get("author") or []
            author_name = None
            if isinstance(authors, list) and authors:
                author_name = authors[0].get("name")
            elif isinstance(authors, dict):
                author_name = authors.get("name")

            self._yielded += 1
            yield make_text_record(
                text=body,
                source=SOURCE_NAME,
                license=LICENSE_STR,
                pipeline_stage="raw",
                url=response.url,
                title=data.get("headline"),
                publish_date=data.get("datePublished"),
                author=author_name,
                section=data.get("articleSection"),
            )
            return


def run(output_path: Path, max_articles: Optional[int], sitemap_url: str = SITEMAP_URL) -> None:
    """Run the Scrapy crawl and write results as JSONL, then log funnel stats.

    Args:
        output_path: Destination ``.jsonl`` file for scraped article records.
        max_articles: Cap on number of articles to scrape (``None`` = all sitemap entries).
        sitemap_url: URL of the news sitemap to seed the crawl from.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    settings = get_project_settings()
    settings.set("FEEDS", {str(output_path): {"format": "jsonlines", "encoding": "utf8", "overwrite": True}})

    process = CrawlerProcess(settings)
    process.crawl(HinduTamilNewsSpider, sitemap_url=sitemap_url, max_articles=max_articles)
    process.start()

    output_count = 0
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            output_count = sum(1 for line in f if line.strip())

    logger.info("Scraped %d articles into %s", output_count, output_path)
    log_stage_stats(
        stage="ingestion.news_spider",
        input_count=output_count,
        output_count=output_count,
        removed_count=0,
        extra={"sitemap_url": sitemap_url, "max_articles": max_articles, "source": SOURCE_NAME},
    )


def main() -> None:
    """CLI entrypoint: crawl hindutamil.in and write article records to JSONL."""
    parser = argparse.ArgumentParser(description="Scrape Tamil news articles from hindutamil.in.")
    parser.add_argument("--output", type=Path, default=RAW_DIR / "news_raw.jsonl", help="Output JSONL path.")
    parser.add_argument("--max-articles", type=int, default=50, help="Max number of articles to scrape (demo/local-sized default; use -1 for unlimited).")
    parser.add_argument("--sitemap-url", type=str, default=SITEMAP_URL)
    args = parser.parse_args()

    max_articles = None if args.max_articles is not None and args.max_articles < 0 else args.max_articles
    run(output_path=args.output, max_articles=max_articles, sitemap_url=args.sitemap_url)

    # Twisted's reactor can leave background threads alive after
    # CrawlerProcess.start() returns, hanging process exit even though the
    # crawl (and our stats logging) already completed successfully.
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
