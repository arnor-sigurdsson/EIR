import base64

import numpy as np
import requests


def encode_numpy_array(file_path: str) -> str:
    array = np.load(file_path)
    encoded = base64.b64encode(array.tobytes()).decode("utf-8")
    return encoded


def send_request(url: str, payload: list[dict]):
    response = requests.post(url, json=payload)
    return response.json()


encoded_data = encode_numpy_array(
    file_path="eir_tutorials/a_using_eir/01_basic_tutorial/data/"
    "processed_sample_data/arrays/A_French-4.DG.npy"
)
response = send_request(
    url="http://localhost:8000/predict", payload=[{"genotype": encoded_data}]
)
print(response)
