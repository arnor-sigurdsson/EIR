import requests


def send_request(url: str, payload: list[dict]) -> list[dict]:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


payload = [
    {"english": "", "spanish": "Tengo mucho hambre"},
]

response = send_request(url="http://localhost:8000/predict", payload=payload)
print(response)
