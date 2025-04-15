import argparse
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
        sequence_length: int = 256,
        dataset_name: str = "HuggingFaceFW/fineweb",
        dataset_split: str = "train",
        max_iterations: int | None = None,
    ):
        self.active_connections: dict[WebSocket, dict] = {}
        self.global_position = 0
        self._position_lock = Lock()

        self.dataset = None
        self.sequence_length = sequence_length
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.max_iterations = max_iterations
        self.validation_ids: set[str] = set()

        logger.info(f"Loading dataset {dataset_name} with split {dataset_split}")
        logger.info(f"Using sequence_length={self.sequence_length}")
        if self.max_iterations is not None:
            logger.info(f"Will terminate after {self.max_iterations} iterations")

        self.dataset_iterator = None

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
        if self.dataset is not None:
            self.dataset_iterator = iter(self.dataset)
        self.global_position = 0
        if self.max_iterations is not None:
            logger.info(f"Reset: Will terminate after {self.max_iterations} iterations")

    def load_dataset(self):
        if self.dataset is None:
            name = None
            path = self.dataset_name
            if path == "HuggingFaceFW/fineweb":
                name = "sample-10BT"

            self.dataset = load_dataset(
                path,
                name=name,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            self.dataset_iterator = iter(self.dataset)

    def get_sequence_batch(self, batch_size: int) -> list[dict[str, Any]]:
        if (
            self.max_iterations is not None
            and self.global_position >= self.max_iterations
        ):
            logger.info(f"Reached max iterations ({self.max_iterations}), terminating")
            return []

        batch = []
        min_words = 20

        accumulated_text = []
        accumulated_words = 0

        with self._position_lock:
            while len(batch) < batch_size:
                try:
                    sample = next(self.dataset_iterator)
                    text = sample["text"].strip()

                    words = text.split()
                    word_count = len(words)

                    if word_count < min_words:
                        continue

                    if accumulated_words > 0:
                        accumulated_text.append("<|endoftext|>")

                    accumulated_text.extend(words)
                    accumulated_words += word_count

                    while accumulated_words >= self.sequence_length:
                        chunk_words = accumulated_text[: self.sequence_length]
                        chunk = " ".join(chunk_words)

                        sample_id = f"sample_{self.global_position}"

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

                        accumulated_text = accumulated_text[self.sequence_length :]
                        accumulated_words -= self.sequence_length

                        self.global_position += 1

                        if (
                            self.max_iterations is not None
                            and self.global_position >= self.max_iterations
                        ):
                            logger.info(
                                f"Reached max iterations ({self.max_iterations}) "
                                f"during batch creation"
                            )
                            return batch

                        if len(batch) >= batch_size:
                            break

                except StopIteration:
                    logger.info("Reached end of dataset stream, restarting iterator")
                    self.dataset_iterator = iter(self.dataset)
                    continue

            if accumulated_words >= min_words and len(batch) < batch_size:
                chunk = " ".join(accumulated_text)
                sample_id = f"sample_{self.global_position}"

                if sample_id not in self.validation_ids:
                    batch.append(
                        {
                            "inputs": {"text_output": chunk},
                            "target_labels": {"text_output": {"text_output": chunk}},
                            "sample_id": sample_id,
                        }
                    )

                self.global_position += 1

                if (
                    self.max_iterations is not None
                    and self.global_position >= self.max_iterations
                ):
                    logger.info(
                        f"Reached max iterations ({self.max_iterations}) after "
                        f"adding last chunk"
                    )

        return batch


def create_manager():
    sequence_length = int(os.getenv("SEQUENCE_LENGTH", "512"))
    dataset_name = os.getenv("DATASET_NAME", "HuggingFaceFW/fineweb")
    dataset_split = os.getenv("DATASET_SPLIT", "train")

    max_iterations_str = os.getenv("MAX_ITERATIONS")
    max_iterations = int(max_iterations_str) if max_iterations_str else None

    logger.info(
        f"Creating ConnectionManager with sequence_length={sequence_length}, "
        f"dataset_name={dataset_name}, dataset_split={dataset_split}, "
        f"max_iterations={max_iterations}"
    )

    return ConnectionManager(
        sequence_length=sequence_length,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        max_iterations=max_iterations,
    )


manager = create_manager()


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
                    "current_position": manager.global_position,
                    "validation_ids_count": len(manager.validation_ids),
                }

                if manager.max_iterations is not None:
                    status_data["max_iterations"] = manager.max_iterations
                    status_data["remaining_iterations"] = max(
                        0, manager.max_iterations - manager.global_position
                    )

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


def main():
    parser = argparse.ArgumentParser(description="Run the data streaming server")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of iterations before terminating (default: no limit)",
    )

    args = parser.parse_args()

    if args.max_iterations is not None:
        os.environ["MAX_ITERATIONS"] = str(args.max_iterations)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, ws_ping_timeout=3600)


if __name__ == "__main__":
    main()
