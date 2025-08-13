import requests
import feedparser
import json
from datetime import datetime

FEEDS = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/115.0.0.0 Safari/537.36"
}


def parse_feed(name, url):
    # fetch thủ công với header browser
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    # parse nội dung nhận được
    d = feedparser.parse(resp.content)

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
            "fetched_at": datetime.utcnow().isoformat(),
            "_raw": entry,
        }
        result.append(item)
    return result


def main():
    all_items = []
    for name, url in FEEDS.items():
        all_items.extend(parse_feed(name, url))

    with open("rss_simplified.json", "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
