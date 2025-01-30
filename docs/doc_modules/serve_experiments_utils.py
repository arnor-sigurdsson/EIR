import base64
import json
import os
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests

from eir.setup.input_setup_modules.setup_image import default_image_loader
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


def load_data_for_serve(data: dict[str, Any]) -> dict[str, Any]:
    loaded_data = {}
    for key, data_pointer in data.items():
        if isinstance(data_pointer, str):
            if data_pointer.endswith(".npy"):
                loaded_data[key] = _serialize_array_to_base64(file_path=data_pointer)
            elif data_pointer.endswith(".txt"):
                with open(data_pointer) as f:
                    loaded_data[key] = f.read()
            elif data_pointer.endswith((".png", ".jpg", ".jpeg")):
                loaded_data[key] = _serialize_image_to_base64(file_path=data_pointer)
            else:
                loaded_data[key] = data_pointer
        else:
            loaded_data[key] = data_pointer
    return loaded_data


def _serialize_image_to_base64(file_path: str) -> str:
    image = default_image_loader(path=file_path)
    buffered = BytesIO()
    image_format = "JPEG" if file_path.lower().endswith((".jpg", ".jpeg")) else "PNG"
    image.save(buffered, format=image_format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def send_request(url: str, payload: list[dict]) -> dict:
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred when sending {payload} to {url}: {e}")
        return {}


def copy_inputs(example_requests: list[dict[str, Any]], output_folder: str) -> None:
    os.makedirs(output_folder, exist_ok=True)

    for idx, request in enumerate(example_requests):
        for key, file_path in request.items():
            if isinstance(file_path, str) and Path(file_path).is_file():
                out_name = f"{Path(file_path).stem}_{idx}{Path(file_path).suffix}"
                shutil.copy(file_path, Path(output_folder) / out_name)
            else:
                with open(Path(output_folder) / f"{key}_{idx}.json", "w") as f:
                    json.dump(file_path, f, ensure_ascii=False, indent=4)


def _serialize_array_to_base64(file_path: str) -> str:
    array_np = np.load(file=file_path)
    array_bytes = array_np.tobytes()
    base64_encoded = base64.b64encode(array_bytes).decode("utf-8")
    return base64_encoded
