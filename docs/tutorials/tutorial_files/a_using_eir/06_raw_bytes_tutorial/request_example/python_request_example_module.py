import base64

import numpy as np
import requests


def load_and_encode_data(data_pointer: str) -> str:
    arr = np.fromfile(data_pointer, dtype="uint8")
    arr_bytes = arr.tobytes()
    return base64.b64encode(arr_bytes).decode("utf-8")


def send_request(url: str, payload: list[dict]) -> dict:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


base = "eir_tutorials/a_using_eir/03_sequence_tutorial/data/IMDB/IMDB_Reviews"
payload = [
    {
        "imdb_reviews_bytes_base_transformer": load_and_encode_data(
            f"{base}/10021_2.txt"
        )
    },
    {
        "imdb_reviews_bytes_base_transformer": load_and_encode_data(
            f"{base}/10132_9.txt"
        )
    },
]

response = send_request(url="http://localhost:8000/predict", payload=payload)
print(response)
