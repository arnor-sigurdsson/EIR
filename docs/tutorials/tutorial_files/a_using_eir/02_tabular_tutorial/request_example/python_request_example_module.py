import requests


def send_request(url: str, payload: list[dict]):
    response = requests.post(url, json=payload)
    return response.json()


payload = [
    {
        "poker_hands": {
            "S1": "3",
            "C1": "12",
            "S2": "3",
            "C2": "2",
            "S3": "3",
            "C3": "11",
            "S4": "4",
            "C4": "5",
            "S5": "2",
            "C5": "5",
        }
    }
]

response = send_request(url="http://localhost:8000/predict", payload=payload)
print(response)
