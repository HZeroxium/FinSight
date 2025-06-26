# news_crawler/src/adapters/default_crawler.py

"""
news_crawler/src/adapters/default_crawler.py: Default NewsCrawler implementation using HTTPX + BeautifulSoup.
"""

from typing import List, Optional
from datetime import datetime
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from common.interfaces.crawler import NewsCrawler
from common.models.article import RawArticle


class DefaultNewsCrawler(NewsCrawler):
    """
    Default crawler that:
      1. Fetches a listing page and extracts article URLs via CSS selector.
      2. Fetches each article page and parses fields (title, published_at, author, content).
    """

    def __init__(
        self,
        listing_url: str,
        listing_selector: str,
        title_selector: str,
        date_selector: str,
        author_selector: Optional[str],
        content_selector: str,
        date_format: str = "%Y-%m-%dT%H:%M:%S",
    ) -> None:
        """
        Args:
            listing_url (str): URL of the page containing article links.
            listing_selector (str): CSS selector to locate <a> tags for articles.
            title_selector (str): CSS selector to locate the title element.
            date_selector (str): CSS selector to locate the datetime element.
            author_selector (Optional[str]): CSS selector for author; set to None if not available.
            content_selector (str): CSS selector to locate the article body.
            date_format (str): strptime format to parse the published_at string.
        """
        self.listing_url = listing_url
        self.listing_selector = listing_selector
        self.title_selector = title_selector
        self.date_selector = date_selector
        self.author_selector = author_selector
        self.content_selector = content_selector
        self.date_format = date_format

    def fetch_listings(self) -> List[str]:
        """
        Retrieve a list of absolute article URLs from the listing page.

        Returns:
            List[str]: Fully qualified URLs to crawl.
        """
        resp = httpx.get(self.listing_url, timeout=10.0)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.select(self.listing_selector)

        urls: List[str] = []
        for tag in links:
            href = tag.get("href")
            if not href:
                continue
            full_url = urljoin(self.listing_url, href)
            urls.append(full_url)
        return urls

    def fetch_article(self, url: str) -> RawArticle:
        """
        Fetch and parse a single article page.

        Args:
            url (str): The article's URL.

        Returns:
            RawArticle: Parsed article with metadata and content.
        """
        resp = httpx.get(url, timeout=10.0)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Title
        title_tag = soup.select_one(self.title_selector)
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Published datetime
        date_tag = soup.select_one(self.date_selector)
        date_str = (
            date_tag.get("datetime") or date_tag.get_text(strip=True)
            if date_tag
            else ""
        )
        published_at = datetime.strptime(date_str, self.date_format)

        # Author
        author = ""
        if self.author_selector:
            author_tag = soup.select_one(self.author_selector)
            author = author_tag.get_text(strip=True) if author_tag else ""

        # Content
        content_tags = soup.select(self.content_selector)
        paragraphs = [p.get_text(strip=True) for p in content_tags]
        content = "\n\n".join(paragraphs)

        return RawArticle(
            id=url,
            url=url,
            title=title,
            published_at=published_at,
            author=author or None,
            content=content,
            source=self.listing_url,
        )
