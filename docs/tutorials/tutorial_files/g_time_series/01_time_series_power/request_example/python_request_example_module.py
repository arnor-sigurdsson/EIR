import requests


def send_request(url: str, payload: list[dict]) -> list[dict]:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


payload = [
    {
        "power_input": "19 3 10 11 14 3 3 4 4 9 39 27 12 5 20 20 38 39 41 61 "
        "52 31 43 31",
        "power_output": "",
    },
]

response = send_request(url="http://localhost:8000/predict", payload=payload)
print(response)
