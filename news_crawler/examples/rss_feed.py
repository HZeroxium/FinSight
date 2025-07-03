import feedparser
import json
from datetime import datetime

# Định nghĩa các feed muốn crawl
FEEDS = {
    # "cointelegraph": "https://cointelegraph.com/rss",
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss",
}


def parse_feed(name, url):
    d = feedparser.parse(url)
    result = []
    for entry in d.entries:
        item = {
            "source": name,
            "title": entry.get("title", "").strip(),
            "url": entry.get("link", ""),
            "description": entry.get("description", "").strip(),
            "published_at": (
                datetime(*entry.published_parsed[:6]).isoformat()
                if entry.get("published_parsed")
                else None
            ),
            "tags": [tag.term for tag in entry.get("tags", [])],
            "author": entry.get("author", ""),
            "guid": entry.get("id") or entry.get("guid") or entry.get("link"),
            "fetched_at": datetime.now().isoformat(),
            "_raw": entry,  # lưu nguyên FeedParserDict; khi dump sẽ dùng default=str
        }
        result.append(item)
    return result


def main():
    all_items = []
    for name, url in FEEDS.items():
        all_items.extend(parse_feed(name, url))

    # Lưu thành file JSON; default=str để convert những object không serialize được
    with open("rss_simplified.json", "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
