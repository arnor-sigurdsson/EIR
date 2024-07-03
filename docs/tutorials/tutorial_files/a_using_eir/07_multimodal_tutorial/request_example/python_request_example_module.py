import base64
from copy import deepcopy
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


base = {
    "pets_tabular": {
        "Type": "Cat",
        "Name": "Nibble",
        "Age": 1.0,
        "Breed1": "Tabby",
        "Breed2": "0",
        "Gender": "Male",
        "Color1": "Black",
        "Color2": "White",
        "Color3": "0",
        "MaturitySize": "Small",
        "FurLength": "Short",
        "Vaccinated": "No",
        "Dewormed": "No",
        "Sterilized": "No",
        "Health": "Healthy",
        "Quantity": 1.0,
        "Fee": "Free",
        "State": "Selangor",
        "VideoAmt": 0.0,
        "PhotoAmt": 1.0,
    },
    "pet_descriptions": "A super cute tabby cat!!!",
    "cute_pet_images": encode_image_to_base64(
        "eir_tutorials/a_using_eir/07_multimodal_tutorial/data/images/86e1089a3.jpg"
    ),
}

payload = [deepcopy(base)]

response = send_request(url="http://localhost:8000/predict", payload=payload)
print(response)
