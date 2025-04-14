import argparse
import base64
import gc
import io
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from threading import Lock
from typing import Any

from datasets import load_dataset
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from eir.setup.streaming_data_setup.protocol import PROTOCOL_VERSION
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)

app = FastAPI()

MAX_WORKERS = 4
MAX_FETCH_WORKERS = 8
PREFETCH_SIZE = 8
RAW_QUEUE_SIZE = 1024
IMAGE_SIZE = (128, 128)


class InputInfo(BaseModel):
    type: str
    shape: list[int] | None = None


class OutputInfo(BaseModel):
    type: str
    shape: list[int] | None = None


class DatasetInfo(BaseModel):
    inputs: dict[str, InputInfo]
    outputs: dict[str, OutputInfo]


def process_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        background = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "RGBA":
            background.paste(image, mask=image.split()[3])
        else:
            background.paste(image)
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    image.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    result = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    return result


def _get_augmentations():
    augmentations = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
            transforms.RandomRotation(degrees=5),
            transforms.RandomResizedCrop(
                size=IMAGE_SIZE,
                scale=(0.95, 1.0),
                ratio=(0.95, 1.05),
            ),
        ]
    )

    return augmentations


class ConnectionManager:
    def __init__(
        self,
        dataset_name: str = "Artificio/WikiArt",
        subset_name: str = "",
        dataset_split: str = "train",
        image_column: str = "image",
        caption_column: str = "description",
    ):
        self.active_connections: dict[WebSocket, dict] = {}
        self.global_position = 0
        self._position_lock = Lock()
        self.running = True

        self.raw_sample_queue = Queue(maxsize=RAW_QUEUE_SIZE)
        self.batch_cache = Queue(maxsize=PREFETCH_SIZE)

        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

        self.dataset = None
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.dataset_split = dataset_split
        self.image_column = image_column
        self.caption_column = caption_column
        self.validation_ids: set[str] = set()
        self.dataset_iterator = None
        self._dataset_lock = Lock()

        logger.info(f"Loading dataset {dataset_name} with split {dataset_split}")
        logger.info(
            f"Using image_column={image_column}, caption_column={caption_column}"
        )

        self.load_dataset()

        self.fetch_threads = []
        for i in range(MAX_FETCH_WORKERS):
            fetch_thread = threading.Thread(
                target=self._fetch_worker,
                args=(i,),
                daemon=True,
            )
            fetch_thread.start()
            self.fetch_threads.append(fetch_thread)

        self.process_thread = threading.Thread(
            target=self._process_worker,
            daemon=True,
        )
        self.process_thread.start()

    def _fetch_worker(self, worker_id: int):
        augmentations = _get_augmentations()

        while self.running:
            try:
                if self.raw_sample_queue.qsize() < RAW_QUEUE_SIZE * 0.8:
                    try:
                        with self._dataset_lock:
                            try:
                                sample = next(self.dataset_iterator)
                            except StopIteration:
                                logger.info(
                                    f"Worker {worker_id}: Reached end of dataset, "
                                    f"restarting iterator"
                                )
                                self.dataset = self.dataset.shuffle(
                                    seed=random.randint(0, 10000)
                                )
                                self.dataset_iterator = iter(self.dataset)
                                sample = next(self.dataset_iterator)

                        image = sample[self.image_column]
                        if not isinstance(image, Image.Image):
                            if isinstance(image, str):
                                image = Image.open(image)
                            else:
                                continue

                        if random.random() < 0.8:
                            image = augmentations(image)
                        else:
                            image = image.resize(IMAGE_SIZE)

                        minimal_sample = {
                            self.image_column: image,
                            self.caption_column: sample[self.caption_column],
                        }

                        with self._position_lock:
                            position = self.global_position
                            self.global_position += 1

                        self.raw_sample_queue.put((position, minimal_sample))

                    except Exception as e:
                        logger.error(f"Error in fetch worker {worker_id}: {e}")
                        time.sleep(0.5)
                else:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Unexpected error in fetch worker {worker_id}: {e}")
                time.sleep(1)

    def _process_sample(self, sample_index: int, sample: dict) -> dict | None:
        try:
            image = sample[self.image_column]

            image_b64 = process_image(image=image)

            caption = sample[self.caption_column]
            sample_id = f"sample_{sample_index}"

            if sample_id not in self.validation_ids:
                return {
                    "inputs": {
                        "image": image_b64,
                        "caption": caption,
                    },
                    "target_labels": {
                        "image": {
                            "image": image_b64,
                        }
                    },
                    "sample_id": sample_id,
                }
            return None
        except Exception as e:
            logger.error(f"Error processing sample at index {sample_index}: {e}")
            return None
        finally:
            sample[self.image_column] = None

    def _process_worker(self):
        batch_size = 32
        while self.running:
            try:
                if self.batch_cache.qsize() < PREFETCH_SIZE:
                    samples_to_process = []
                    indices_to_process = []

                    while len(samples_to_process) < batch_size * 1.5:
                        try:
                            position, sample = self.raw_sample_queue.get(timeout=1)
                            samples_to_process.append(sample)
                            indices_to_process.append(position)
                        except Empty:
                            if samples_to_process:
                                break
                            time.sleep(0.1)
                            continue

                    if not samples_to_process:
                        continue

                    futures = [
                        self.executor.submit(self._process_sample, idx, sample)
                        for idx, sample in zip(
                            indices_to_process, samples_to_process, strict=False
                        )
                    ]

                    batch = []
                    for future in futures:
                        try:
                            result = future.result()
                            if result is not None:
                                batch.append(result)
                                if len(batch) >= batch_size:
                                    break
                        except Exception as e:
                            logger.error(f"Error in processing future: {e}")

                    if batch:
                        self.batch_cache.put(batch)

                    samples_to_process.clear()
                    indices_to_process.clear()

                    if self.global_position % 500 == 0:
                        gc.collect()
                else:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in process worker: {e}")
                time.sleep(0.5)

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
        with self._position_lock:
            self.global_position = 0

            while not self.raw_sample_queue.empty():
                try:
                    self.raw_sample_queue.get_nowait()
                except Empty:
                    break

            while not self.batch_cache.empty():
                try:
                    self.batch_cache.get_nowait()
                except Empty:
                    break

            gc.collect()

    def load_dataset(self):
        if self.dataset is None:
            self.dataset = load_dataset(
                self.dataset_name,
                split=self.dataset_split,
                streaming=False,
                trust_remote_code=True,
            )

            self.dataset = self.dataset.shuffle(seed=random.randint(0, 10000))

            self.dataset_iterator = iter(self.dataset)
            logger.info("Dataset loaded with selected columns only")

    def get_image_caption_batch(self, batch_size: int) -> list[dict[str, Any]]:
        try:
            try:
                return self.batch_cache.get(timeout=2)
            except Empty:
                logger.warning(
                    "Batch cache empty, waiting for processing to catch up..."
                )
                try:
                    return self.batch_cache.get(timeout=5)
                except Empty:
                    logger.error("Failed to get batch from cache after extended wait")
                    return []
        except Exception as e:
            logger.error(f"Error in get_image_caption_batch: {e}")
            return []

    def shutdown(self):
        self.running = False

        for thread in self.fetch_threads:
            if thread.is_alive():
                thread.join(timeout=2)

        if self.process_thread.is_alive():
            self.process_thread.join(timeout=2)

        self.executor.shutdown(wait=False)

        gc.collect()


def create_manager():
    dataset_name = os.getenv("DATASET_NAME", "Artificio/WikiArt")
    subset_name = os.getenv("SUBSET_NAME", "")
    dataset_split = os.getenv("DATASET_SPLIT", "train")
    image_column = os.getenv("IMAGE_COLUMN", "image")
    caption_column = os.getenv("CAPTION_COLUMN", "description")

    logger.info(
        f"Creating ConnectionManager with dataset_name={dataset_name}, "
        f"dataset_split={dataset_split}, image_column={image_column}, "
        f"caption_column={caption_column}"
    )

    return ConnectionManager(
        dataset_name=dataset_name,
        subset_name=subset_name,
        dataset_split=dataset_split,
        image_column=image_column,
        caption_column=caption_column,
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
                    inputs={
                        "image": InputInfo(type="image"),
                        "caption": InputInfo(type="sequence"),
                    },
                    outputs={
                        "image": OutputInfo(type="image"),
                    },
                )

                await manager.send_personal_message(
                    message={"type": "info", "payload": dataset_info.model_dump()},
                    websocket=websocket,
                )

            elif message_type == "getData":
                batch_size = data.get("payload", {}).get("batch_size", 32)
                batch = manager.get_image_caption_batch(batch_size=batch_size)

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
                    "raw_queue_size": manager.raw_sample_queue.qsize(),
                    "batch_cache_size": manager.batch_cache.qsize(),
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


@app.on_event("shutdown")
async def shutdown_event():
    manager.shutdown()


def main():
    global MAX_WORKERS, MAX_FETCH_WORKERS, PREFETCH_SIZE, RAW_QUEUE_SIZE

    parser = argparse.ArgumentParser(
        description="Run the memory-efficient image-caption data streaming server"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help="Number of worker threads for image processing",
    )
    parser.add_argument(
        "--fetch-workers",
        type=int,
        default=MAX_FETCH_WORKERS,
        help="Number of worker threads for fetching data",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=PREFETCH_SIZE,
        help="Number of batches to prefetch",
    )
    parser.add_argument(
        "--raw-queue-size",
        type=int,
        default=RAW_QUEUE_SIZE,
        help="Size of the raw sample queue",
    )

    args = parser.parse_args()

    MAX_WORKERS = args.workers
    MAX_FETCH_WORKERS = args.fetch_workers
    PREFETCH_SIZE = args.prefetch
    RAW_QUEUE_SIZE = args.raw_queue_size

    logger.info("Starting server with settings:")
    logger.info(f"- Workers: {MAX_WORKERS}")
    logger.info(f"- Fetch workers: {MAX_FETCH_WORKERS}")
    logger.info(f"- Prefetch size: {PREFETCH_SIZE}")
    logger.info(f"- Raw queue size: {RAW_QUEUE_SIZE}")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, ws_ping_timeout=3600)


if __name__ == "__main__":
    main()
