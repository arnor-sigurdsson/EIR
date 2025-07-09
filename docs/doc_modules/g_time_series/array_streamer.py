import base64
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from eir.setup.streaming_data_setup.protocol import PROTOCOL_VERSION
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


class StockDataManager:
    def __init__(
        self,
        *,
        csv_path: Path,
        sequence_length: int = 64,
        is_diffusion: bool = False,
    ):
        self.csv_path = csv_path
        self.sequence_length = sequence_length
        self.is_diffusion = is_diffusion
        self.current_index = 0

        logger.info(f"Loading stock data from {csv_path}")

        df = pd.read_csv(filepath_or_buffer=csv_path)
        self.df = df.sample(frac=1).reset_index(drop=True)

        logger.info(f"Loaded {len(self.df)} samples")

        self.active_connections: dict[WebSocket, dict] = {}

    async def connect_websocket(self, websocket: WebSocket) -> bool:
        try:
            await websocket.accept()

            handshake_message = await websocket.receive_json()

            if (
                handshake_message["type"] != "handshake"
                or handshake_message["version"] != PROTOCOL_VERSION
            ):
                await websocket.send_json(
                    {
                        "type": "error",
                        "payload": {"message": "Incompatible protocol version"},
                    }
                )
                await websocket.close()
                return False

            await websocket.send_json(
                {"type": "handshake", "version": PROTOCOL_VERSION}
            )

            self.active_connections[websocket] = {"position": 0}
            return True

        except Exception as e:
            logger.error(f"Error in connect: {e}")
            return False

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]

    def reset(self):
        self.current_index = 0
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        logger.info("Data manager reset")

    async def send_personal_message(
        self, *, message: dict[str, Any], websocket: WebSocket
    ):
        await websocket.send_json(message)

    def get_batch(self, *, batch_size: int) -> list[dict[str, Any]]:
        if self.current_index >= len(self.df):
            self.reset()

        batch = []
        for _ in range(batch_size):
            if self.current_index >= len(self.df):
                self.reset()

            row = self.df.iloc[self.current_index]

            input_array = np.array(
                [int(x) for x in row["InputSequence"].split(" ")], dtype=np.float32
            )
            output_array = np.array(
                [int(x) for x in row["OutputSequence"].split(" ")], dtype=np.float32
            )

            input_encoded = base64.b64encode(input_array.tobytes()).decode("utf-8")
            output_encoded = base64.b64encode(output_array.tobytes()).decode("utf-8")
            sample = {
                "inputs": {"stock_input": input_encoded},
                "target_labels": {"stock_output": {"stock_output": output_encoded}},
                "sample_id": row["ID"],
            }

            if self.is_diffusion:
                sample["inputs"]["stock_output"] = output_encoded

            batch.append(sample)
            self.current_index += 1

        return batch


@asynccontextmanager
async def lifespan(app: FastAPI):
    csv_path = Path(
        "eir_tutorials/g_time_series/02_time_series_stocks/data"
        "/stock_combined_train.csv"
    )

    config = app.state.config

    app.state.manager = StockDataManager(
        csv_path=csv_path,
        sequence_length=config.get("sequence_length", 64),
        is_diffusion=config.get("is_diffusion", False),
    )

    logger.info("Stock data manager initialized")
    yield

    logger.info("Application shutting down")


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    manager = websocket.app.state.manager

    connected = await manager.connect_websocket(websocket=websocket)
    if not connected:
        return

    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "status":
                await websocket.send_json(
                    {"type": "status", "payload": {"ready": True}}
                )

            elif message_type == "getInfo":
                dataset_info = {
                    "inputs": {
                        "stock_input": {
                            "type": "array",
                            "shape": [manager.sequence_length],
                        }
                    },
                    "outputs": {
                        "stock_output": {
                            "type": "array",
                            "shape": [manager.sequence_length],
                        }
                    },
                }

                is_diffusion = manager.is_diffusion
                if is_diffusion:
                    dataset_info["inputs"]["stock_output"] = {
                        "type": "array",
                        "shape": [manager.sequence_length],
                    }

                await websocket.send_json({"type": "info", "payload": dataset_info})

            elif message_type == "getData":
                batch_size = data.get("payload", {}).get("batch_size", 32)
                batch = manager.get_batch(batch_size=batch_size)

                if not batch:
                    await websocket.send_json(
                        {"type": "data", "payload": ["terminate"]}
                    )
                    break

                await websocket.send_json({"type": "data", "payload": batch})

            elif message_type == "reset":
                manager.reset()
                await websocket.send_json(
                    {
                        "type": "resetConfirmation",
                        "payload": {"message": "Reset successful"},
                    }
                )
                await websocket.send_json(
                    {"type": "reset", "payload": {"message": "Reset command received"}}
                )

            elif message_type == "setValidationIds":
                validation_ids = data.get("payload", {}).get("validation_ids", [])
                await websocket.send_json(
                    {
                        "type": "validationIdsConfirmation",
                        "payload": {
                            "message": f"Received {len(validation_ids)} validation IDs"
                        },
                    }
                )

            elif message_type == "heartbeat":
                await websocket.send_json({"type": "heartbeat"})

            else:
                logger.warning(f"Unknown message type: {message_type}")
                await websocket.send_json(
                    {
                        "type": "error",
                        "payload": {"message": f"Unknown message type: {message_type}"},
                    }
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket=websocket)
    finally:
        try:
            await websocket.close()
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")
        logger.info("WebSocket connection closed")


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the stock data streaming server")
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument(
        "--diffusion", action="store_true", help="Enable diffusion mode for the server"
    )
    parser.add_argument(
        "--sequence-length", type=int, default=64, help="Sequence length for the model"
    )

    args = parser.parse_args()

    app.state.config = {
        "is_diffusion": args.diffusion,
        "sequence_length": args.sequence_length,
    }

    uvicorn.run(app, host=args.host, port=args.port, ws_ping_timeout=3600)
