import requests


def send_request(url: str, payload: list[dict]):
    response = requests.post(url, json=payload)
    return response.json()


payload = [
    {
        "flchain": {
            "age": 65,
            "sex": "M",
            "flcgrp": "1",
            "kappa": 1.5,
            "lambdaport": 1.2,
            "creatinine": 1.1,
            "mgus": "yes",
        }
    }
]

response = send_request(url="http://localhost:8000/predict", payload=payload)
print(response)
