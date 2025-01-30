import base64
from io import BytesIO

import requests
from PIL import Image


def encode_image_to_base64(file_path: str) -> str:
    with Image.open(file_path) as image:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


def send_request(url: str, payload: list[dict]) -> dict:
    response_ = requests.post(url, json=payload)
    response_.raise_for_status()
    return response_.json()


base = (
    "eir_tutorials/a_using_eir/05_image_tutorial/data/hot_dog_not_hot_dog/food_images"
)
payload = [
    {
        "hot_dog_efficientnet": encode_image_to_base64(f"{base}/1040579.jpg"),
        "hot_dog_resnet18": encode_image_to_base64(f"{base}/1040579.jpg"),
    },
    {
        "hot_dog_efficientnet": encode_image_to_base64(f"{base}/108743.jpg"),
        "hot_dog_resnet18": encode_image_to_base64(f"{base}/108743.jpg"),
    },
]

response = send_request(url="http://localhost:8000/predict", payload=payload)
print(response)
