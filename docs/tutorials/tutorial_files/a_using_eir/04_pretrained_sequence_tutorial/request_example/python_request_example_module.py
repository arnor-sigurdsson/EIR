import requests


def send_request(url: str, payload: list[dict]) -> dict:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


payload = [
    {
        "imdb_reviews_windowed": "This movie was great! I loved it!",
        "imdb_reviews_longformer": "This movie was great! I loved it!",
        "imdb_reviews_tiny_bert": "This movie was great! I loved it!",
    },
]

response = send_request(url="http://localhost:8000/predict", payload=payload)
print(response)
