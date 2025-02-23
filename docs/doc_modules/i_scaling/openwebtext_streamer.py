import argparse
import json
import os
from threading import Lock
from typing import Any

from datasets import load_dataset
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from eir.setup.streaming_data_setup.protocol import PROTOCOL_VERSION
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)

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
    def __init__(
        self,
        max_sequences: int = 200_000,
        sequence_length: int = 256,
        dataset_name: str = "Skylion007/openwebtext",
        dataset_split: str = "train",
    ):
        self.active_connections: dict[WebSocket, dict] = {}
        self.global_position = 0
        self._position_lock = Lock()

        self.dataset = None
        self.max_sequences = max_sequences
        self.sequence_length = sequence_length
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.validation_ids: set[str] = set()

        self.load_dataset()

    async def connect(self, websocket: WebSocket):
        try:
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
                return False

            worker_id = handshake_message.get("worker_id", 0)
            self.active_connections[websocket] = {
                "current_position": 0,
                "worker_id": worker_id,
            }

            await websocket.send_json(
                {"type": "handshake", "version": PROTOCOL_VERSION}
            )
            return True
        except Exception as e:
            logger.error(f"Error in connect: {e}")
            if websocket in self.active_connections:
                del self.active_connections[websocket]
            return False

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]

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

    def get_sequence_batch(
        self,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        batch = []
        min_words = 20

        with self._position_lock:
            start_position = self.global_position

            while len(batch) < batch_size and start_position < self.max_sequences:
                try:
                    text = self.dataset[start_position]["text"].strip()
                    words = text.split()

                    word_count = len(words)
                    ranges = list(range(0, word_count, self.sequence_length))

                    base_sample_id = f"sample_{start_position}_"

                    for chunk_idx, i in enumerate(ranges):
                        if len(batch) >= batch_size:
                            break

                        chunk_words = words[i : i + self.sequence_length]

                        if len(chunk_words) >= min_words:
                            chunk = " ".join(chunk_words)
                            sample_id = base_sample_id + str(chunk_idx)

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

                    start_position += 1

                except IndexError:
                    start_position = 0
                    break

            self.global_position = start_position

        return batch


def create_manager():
    max_sequences = int(os.getenv("MAX_SEQUENCES", "8000000"))
    sequence_length = int(os.getenv("SEQUENCE_LENGTH", "256"))
    dataset_name = os.getenv("DATASET_NAME", "Skylion007/openwebtext")
    dataset_split = os.getenv("DATASET_SPLIT", "train")

    logger.info(
        f"Creating ConnectionManager with max_sequences={max_sequences}, "
        f"sequence_length={sequence_length}, dataset_name={dataset_name}, "
        f"dataset_split={dataset_split}"
    )

    return ConnectionManager(
        max_sequences=max_sequences,
        sequence_length=sequence_length,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )


manager = create_manager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            try:
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
                    batch = manager.get_sequence_batch(
                        batch_size=batch_size,
                    )

                    if not batch:
                        await manager.send_personal_message(
                            message={"type": "data", "payload": ["terminate"]},
                            websocket=websocket,
                        )
                        break

                    await manager.send_personal_message(
                        message={"type": "data", "payload": batch},
                        websocket=websocket,
                    )

                elif message_type == "setValidationIds":
                    validation_ids = data.get("payload", {}).get("validation_ids", [])
                    manager.validation_ids = set(validation_ids)

                    await manager.send_personal_message(
                        message={
                            "type": "validationIdsConfirmation",
                            "payload": {
                                "message": f"Received {len(validation_ids)} "
                                f"validation IDs"
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
                        "global_position": manager.global_position,
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
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                continue

    except WebSocketDisconnect:
        logger.info("Client disconnected normally")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Error in websocket connection: {e}")
        manager.disconnect(websocket)
    finally:
        manager.disconnect(websocket)


def main():
    parser = argparse.ArgumentParser(description="Run the data streaming server")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )

    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, ws_ping_timeout=3600)


if __name__ == "__main__":
    main()
