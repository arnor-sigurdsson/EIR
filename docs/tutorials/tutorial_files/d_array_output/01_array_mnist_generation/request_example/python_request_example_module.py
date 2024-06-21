import base64

import numpy as np
import requests


def encode_array_to_base64(file_path: str) -> str:
    array_np = np.load(file_path)
    array_bytes = array_np.tobytes()
    return base64.b64encode(array_bytes).decode("utf-8")


def send_request(url: str, payload: list[dict]) -> list[dict]:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


image_base = "eir_tutorials/d_array_output/01_array_mnist_generation/data/mnist_npy"
payload = [
    {"mnist": encode_array_to_base64(f"{image_base}/10001.npy")},
]

response = send_request(url="http://localhost:8000/predict", payload=payload)
print(response)
