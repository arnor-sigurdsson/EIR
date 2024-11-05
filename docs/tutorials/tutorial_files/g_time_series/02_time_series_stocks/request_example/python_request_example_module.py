import base64

import numpy as np
import requests

input_array = np.array(
    [
        31,
        32,
        31,
        30,
        31,
        31,
        31,
        31,
        31,
        30,
        31,
        31,
        32,
        33,
        32,
        32,
        33,
        33,
        33,
        34,
        35,
        34,
        34,
        33,
        32,
        32,
        32,
        33,
        33,
        34,
        34,
        34,
        33,
        34,
        34,
        34,
        34,
        36,
        36,
        36,
        36,
        37,
        35,
        35,
        34,
        35,
        35,
        34,
        35,
        35,
        36,
        35,
        36,
        35,
        34,
        34,
        34,
        33,
        33,
        31,
        31,
        31,
        32,
        32,
    ],
    dtype=np.float32,
)

output_base = np.zeros(shape=input_array.shape, dtype=np.float32)


def encode_array_to_base64(array_np: np.ndarray) -> str:
    array_bytes = array_np.tobytes()
    return base64.b64encode(array_bytes).decode("utf-8")


def send_request(url: str, payload: list[dict]) -> list[dict]:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


payload = [
    {
        "stock_input": encode_array_to_base64(array_np=input_array),
        "stock_output": encode_array_to_base64(array_np=output_base),
    },
]

response = send_request(url="http://localhost:8000/predict", payload=payload)
print(response)
