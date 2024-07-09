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


image_base = "eir_tutorials/f_image_output/03_mnist_diffusion/data/data/images"
payload = [
    {
        "image": encode_image_to_base64(f"{image_base}/00000.png"),
        "mnist_tabular": {"CLASS": "0"},
    },
]

response = send_request(url="http://localhost:8000/predict", payload=payload)
print(response)
