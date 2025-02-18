from typing import Any

from datasets import load_dataset
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from eir.setup.streaming_data_setup.protocol import PROTOCOL_VERSION

app = FastAPI()


class InputInfo(BaseModel):
    type: str
    shape: list[int] | None = None


class OutputInfo(BaseModel):
    type: str
    shape: list[int] | None = None


class DatasetInfo(BaseModel):
    inputs: dict[str, InputInfo]
    outputs: dict[str, OutputInfo]


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.dataset = None
        self.current_position = 0
        self.max_sequences = 200_000
        self.sequence_length = 256
        self.validation_ids: set[str] = set()

        self.load_dataset()

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
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

    def reset(self):
        self.current_position = 0

    def load_dataset(self):
        if self.dataset is None:
            self.dataset = load_dataset(
                "Skylion007/openwebtext",
                split="train",
                trust_remote_code=True,
            )

    def get_sequence_batch(self, batch_size: int) -> list[dict[str, Any]]:
        batch = []

        while len(batch) < batch_size and self.current_position < self.max_sequences:
            try:
                text = self.dataset[self.current_position]["text"].strip()

                words = text.split()
                chunks = [
                    " ".join(words[i : i + self.sequence_length])
                    for i in range(0, len(words), self.sequence_length)
                ]

                for chunk_idx, chunk in enumerate(chunks):
                    if len(batch) >= batch_size:
                        break

                    if len(chunk.split()) >= 20:
                        sample_id = f"sample_{self.current_position}_{chunk_idx}"

                        if sample_id not in self.validation_ids:
                            batch.append(
                                {
                                    "inputs": {"text_output": chunk},
                                    "target_labels": {
                                        "text_output": {"text_output": chunk}
                                    },
                                    "sample_id": sample_id,
                                }
                            )

                self.current_position += 1

            except IndexError:
                self.current_position = 0
                break

        return batch


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "getInfo":
                dataset_info = DatasetInfo(
                    inputs={},
                    outputs={
                        "text_output": OutputInfo(type="sequence"),
                    },
                )

                await manager.send_personal_message(
                    message={"type": "info", "payload": dataset_info.model_dump()},
                    websocket=websocket,
                )

            elif message_type == "getData":
                batch_size = data.get("payload", {}).get("batch_size", 32)
                batch = manager.get_sequence_batch(batch_size=batch_size)

                if not batch:
                    await manager.send_personal_message(
                        message={"type": "data", "payload": ["terminate"]},
                        websocket=websocket,
                    )
                    break

                await manager.send_personal_message(
                    message={"type": "data", "payload": batch}, websocket=websocket
                )

            elif message_type == "setValidationIds":
                validation_ids = data.get("payload", {}).get("validation_ids", [])
                manager.validation_ids = set(validation_ids)

                await manager.send_personal_message(
                    message={
                        "type": "validationIdsConfirmation",
                        "payload": {
                            "message": f"Received {len(validation_ids)} validation IDs"
                        },
                    },
                    websocket=websocket,
                )

            elif message_type == "reset":
                manager.reset()
                await manager.send_personal_message(
                    message={
                        "type": "resetConfirmation",
                        "payload": {"message": "Reset successful"},
                    },
                    websocket=websocket,
                )
                await manager.broadcast(
                    message={
                        "type": "reset",
                        "payload": {"message": "Reset command received"},
                    }
                )

            elif message_type == "status":
                status_data = {
                    "active_connections": len(manager.active_connections),
                    "current_position": manager.current_position,
                    "validation_ids_count": len(manager.validation_ids),
                }

                await manager.send_personal_message(
                    message={"type": "status", "payload": status_data},
                    websocket=websocket,
                )

            elif message_type == "heartbeat":
                await manager.send_personal_message(
                    message={"type": "heartbeat"}, websocket=websocket
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    finally:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, ws_ping_timeout=600)
