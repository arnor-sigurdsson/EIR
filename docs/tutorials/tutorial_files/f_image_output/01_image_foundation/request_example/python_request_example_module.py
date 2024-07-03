import base64

import requests


def encode_image_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as image_file:
        image_bytes = image_file.read()
        return base64.b64encode(image_bytes).decode("utf-8")


def send_request(url: str, payload: list[dict]) -> list[dict]:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


image_base = "eir_tutorials/f_image_output/01_image_foundation/data/images"
payload = [
    {"image": encode_image_to_base64(f"{image_base}/000000000009.jpg")},
]

response = send_request(url="http://localhost:8000/predict", payload=payload)
print(response)
