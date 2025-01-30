import base64

import numpy as np
import requests


def encode_array_to_base64(file_path: str) -> str:
    array_np = np.load(file_path)
    array_bytes = array_np.tobytes()
    return base64.b64encode(array_bytes).decode("utf-8")


def send_request(url: str, payload: list[dict]) -> dict:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


base = (
    "eir_tutorials/a_using_eir/08_array_tutorial/data/processed_sample_data/arrays_3d"
)
payload = [
    {"genotype_as_array": encode_array_to_base64(f"{base}/A374.npy")},
    {"genotype_as_array": encode_array_to_base64(f"{base}/Ayodo_468C.npy")},
    {"genotype_as_array": encode_array_to_base64(f"{base}/NOR146.npy")},
]

response = send_request(url="http://localhost:8000/predict", payload=payload)
print(response)
