import base64
import io
import string
import uuid
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image
from pydantic import BaseModel

from eir.setup.streaming_data_setup.protocol import PROTOCOL_VERSION

app = FastAPI()


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.validation_ids: set[str] = set()
        self.sample_index: int = 0

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        handshake_message = await websocket.receive_json()
        is_not_handshake = handshake_message["type"] != "handshake"
        is_incompatible_version = handshake_message["version"] != PROTOCOL_VERSION
        if is_not_handshake or is_incompatible_version:
            await websocket.send_json(
                {
                    "type": "error",
                    "payload": {"message": "Incompatible protocol version"},
                }
            )
            await websocket.close()
            return

        await websocket.send_json({"type": "handshake", "version": PROTOCOL_VERSION})
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

    def set_validation_ids(self, ids: list[str]):
        self.validation_ids = set(ids)

    def reset(self):
        self.sample_index = 0


manager = ConnectionManager()


def generate_deterministic_sample(index: int) -> dict[str, Any]:
    sequence = "".join(string.ascii_lowercase[i % 26] for i in range(index, index + 10))
    column1 = ["Positive", "Negative", "Neutral"][index % 3]
    column2 = (index % 100) / 100.0

    omics = np.random.rand(4, 100).astype(np.bool_)
    omics_b64 = base64.b64encode(omics.tobytes()).decode("utf-8")

    array_input = np.random.rand(10, 5).astype(np.float32)
    array_input_b64 = base64.b64encode(array_input.tobytes()).decode("utf-8")

    image_input = (np.random.rand(16, 16, 3) * 128).astype(np.uint8)
    pil_image = Image.fromarray(image_input)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    image_input_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    test_target = index % 2 * 1000

    array_output = np.random.rand(5, 3).astype(np.float32)
    array_output_b64 = base64.b64encode(array_output.tobytes()).decode("utf-8")

    image_output = (np.random.rand(16, 16, 3) * 128).astype(np.uint8)
    pil_image = Image.fromarray(image_output)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    image_output_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    random_float = np.random.rand(1).astype(np.float32).item()

    sequence_output = "".join(
        string.ascii_lowercase[i % 26] for i in range(index, index + 10)
    )

    return {
        "inputs": {
            "sequence_data": sequence,
            "omics_data": omics_b64,
            "array_data": array_input_b64,
            "image_data": image_input_b64,
            "tabular_data": {
                "column1": column1,
                "column2": column2,
            },
        },
        "target_labels": {
            "test_output": {"test_target": test_target},
            "output_array": {"output_array": array_output_b64},
            "output_image": {"output_image": image_output_b64},
            "output_sequence": {"output_sequence": sequence_output},
            "output_survival": {
                "Origin": str(int(test_target)),
                "Height": float(test_target) + random_float,
            },
        },
        "sample_id": str(uuid.uuid4()),
    }


def generate_batch(batch_size: int) -> list[dict[str, Any]]:
    batch = []
    while len(batch) < batch_size:
        sample = generate_deterministic_sample(index=manager.sample_index)
        manager.sample_index += 1
        if sample["sample_id"] not in manager.validation_ids:
            batch.append(sample)
    return batch


class InputInfo(BaseModel):
    type: str
    shape: list[int] = None


class OutputInfo(BaseModel):
    type: str
    shape: list[int] = None


class DatasetInfo(BaseModel):
    inputs: dict[str, InputInfo]
    outputs: dict[str, OutputInfo]


def get_dataset_info():
    return DatasetInfo(
        inputs={
            "sequence_data": InputInfo(type="sequence"),
            "tabular_data": InputInfo(type="tabular"),
            "omics_data": InputInfo(type="omics", shape=[4, 100]),
            "image_data": InputInfo(type="image", shape=[16, 16, 3]),
            "array_data": InputInfo(type="array", shape=[10, 5]),
        },
        outputs={
            "test_output": OutputInfo(type="tabular"),
            "output_array": OutputInfo(type="array", shape=[5, 3]),
            "output_image": OutputInfo(type="image", shape=[16, 16, 3]),
            "output_sequence": OutputInfo(type="sequence"),
            "output_survival": OutputInfo(type="survival"),
        },
    )


MAX_ITERATIONS = 300
current_iteration = 0


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global current_iteration
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            payload = data.get("payload", {})

            if message_type == "getInfo":
                dataset_info = get_dataset_info()
                await manager.send_personal_message(
                    message={
                        "type": "info",
                        "payload": dataset_info.model_dump(),
                    },
                    websocket=websocket,
                )

            elif message_type == "getData":
                if current_iteration >= MAX_ITERATIONS:
                    await manager.send_personal_message(
                        message={
                            "type": "data",
                            "payload": ["terminate"],
                        },
                        websocket=websocket,
                    )

                current_iteration += 1
                batch_size = payload.get("batch_size", 64)
                batch_data = generate_batch(batch_size=batch_size)
                await manager.send_personal_message(
                    message={
                        "type": "data",
                        "payload": batch_data,
                    },
                    websocket=websocket,
                )

            elif message_type == "setValidationIds":
                validation_ids = payload.get("validation_ids", [])
                manager.set_validation_ids(ids=validation_ids)
                await manager.send_personal_message(
                    message={
                        "type": "validationIdsConfirmation",
                        "payload": {
                            "message": f"Received {len(validation_ids)} validation IDs",
                        },
                    },
                    websocket=websocket,
                )

            elif message_type == "reset":
                current_iteration = 0
                manager.reset()
                await manager.broadcast(
                    message={
                        "type": "reset",
                        "payload": {
                            "message": "Reset command received",
                        },
                    }
                )
                await manager.send_personal_message(
                    message={
                        "type": "resetConfirmation",
                        "payload": {
                            "message": "Reset command broadcasted",
                        },
                    },
                    websocket=websocket,
                )

            elif message_type == "status":
                status_data = {
                    "active_connections": len(manager.active_connections),
                    "validation_ids_count": len(manager.validation_ids),
                    "current_sample_index": manager.sample_index,
                }

                await manager.send_personal_message(
                    message={
                        "type": "status",
                        "payload": status_data,
                    },
                    websocket=websocket,
                )

            elif message_type == "heartbeat":
                await manager.send_personal_message(
                    message={
                        "type": "heartbeat",
                        "payload": {
                            "message": "Received heartbeat",
                        },
                    },
                    websocket=websocket,
                )

            else:
                await manager.send_personal_message(
                    message={
                        "type": "error",
                        "payload": {
                            "message": f"Unknown message type: {message_type}",
                        },
                    },
                    websocket=websocket,
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ws_ping_timeout=60,
    )
