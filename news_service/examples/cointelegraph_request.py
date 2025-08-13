# examples/cointelegraph_request.py

import requests
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 1. Define endpoint & GraphQL query
URL = "https://conpletus.cointelegraph.com/v1/"
QUERY = """
query CategoryPagePostsQuery(
  $short: String,
  $slug: String!,
  $offset: Int = 0,
  $length: Int = 10,
  $hideFromMainPage: Boolean = null
) {
  locale(short: $short) {
    category(slug: $slug) {
      id
      posts(
        order: "postPublishedTime"
        offset: $offset
        length: $length
        hideFromMainPage: $hideFromMainPage
      ) {
        data {
          id
          slug
          views
          postTranslate {
            id
            title
            avatar
            published
            publishedHumanFormat
            leadText
            author {
              id
              slug
              innovationCircleUrl
              authorTranslates {
                id
                name
              }
            }
          }
          category {
            id
            slug
            categoryTranslates {
              id
              title
            }
          }
          author {
            id
            slug
            authorTranslates {
              id
              name
            }
          }
          postBadge {
            id
            label
            postBadgeTranslates {
              id
              title
            }
          }
          showShares
          showStats
        }
        postsCount
      }
    }
  }
}
"""

# 2. Define variables for the query
VARIABLES = {
    "cacheTimeInMS": 300000,
    "hideFromMainPage": False,
    "length": 10,
    "offset": 0,
    "short": "en",
    "slug": "latest-news",
}


# Tạo session để giữ kết nối, thêm retry và header chung
session = requests.Session()
session.headers.update(
    {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36",
    }
)

# Cấu hình retry
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://", HTTPAdapter(max_retries=retries))


def run_query(url, query, variables):
    payload = {"query": query, "variables": variables}
    resp = session.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def main():
    result = run_query(URL, QUERY, VARIABLES)
    with open("data/cointelegraph.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("✅ Data saved to data/cointelegraph.json")


if __name__ == "__main__":
    main()
