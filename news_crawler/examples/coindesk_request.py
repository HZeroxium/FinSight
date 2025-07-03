import requests

response = requests.get(
    "https://data-api.coindesk.com/news/v1/article/list",
    params={"lang": "EN", "limit": 10},
    headers={"Content-type": "application/json; charset=UTF-8"},
)

json_response = response.json()

# Save to a file
with open("data/coindesk_news_v1_article_list.json", "w", encoding="utf-8") as f:
    import json

    json.dump(json_response, f, ensure_ascii=False, indent=2)
