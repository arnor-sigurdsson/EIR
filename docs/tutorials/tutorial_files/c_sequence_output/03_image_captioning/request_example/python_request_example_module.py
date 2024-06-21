import base64
from io import BytesIO

import requests
from PIL import Image


def encode_image_to_base64(file_path: str) -> str:
    with Image.open(file_path) as image:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


def send_request(url: str, payload: list[dict]) -> list[dict]:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


image_base = "eir_tutorials/c_sequence_output/03_image_captioning/data/images"
payload = [
    {
        "image_captioning": encode_image_to_base64(f"{image_base}/000000000009.jpg"),
        "captions": "",
    },
    {
        "image_captioning": encode_image_to_base64(f"{image_base}/000000000034.jpg"),
        "captions": "",
    },
    {
        "image_captioning": encode_image_to_base64(f"{image_base}/000000581929.jpg"),
        "captions": "A horse",
    },
]

response = send_request(url="http://localhost:8000/predict", payload=payload)
print(response)
