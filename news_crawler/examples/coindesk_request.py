import requests

response = requests.get(
    "https://data-api.coindesk.com/news/v1/article/list",
    params={"lang": "EN", "limit": 10},
    headers={
        "Content-type": "application/json; charset=UTF-8",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    },
)

json_response = response.json()

# Print status code and response length
print(f"Status Code: {response.status_code}")
print(f"Response Length: {len(json_response.get('Data', []))}")

# Save to a file
with open("data/coindesk_news_v1_article_list.json", "w", encoding="utf-8") as f:
    import json

    json.dump(json_response, f, ensure_ascii=False, indent=2)
    print(
        f"Saved {len(json_response['Data'])} articles to data/coindesk_news_v1_article_list.json"
    )
