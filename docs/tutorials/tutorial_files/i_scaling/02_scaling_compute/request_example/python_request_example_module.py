import requests


def send_request(url: str, payload: list[dict]) -> list[dict]:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


payload = [
    {
        "text_output": "### Instruction: Write a short story about a robot "
        "learning to feel emotions. ### Response:"
    },
]

response = send_request(url="http://localhost:8000/predict", payload=payload)
print(response)
