import argparse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from docs.doc_modules.g_time_series.stock_data_manager import (
    create_stock_manager,
)
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


async def connect_websocket(websocket: WebSocket) -> bool:
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

        await websocket.send_json({"type": "handshake", "version": PROTOCOL_VERSION})
        return True

    except Exception as e:
        logger.error(f"Error in connect: {e}")
        return False


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    connected = await connect_websocket(websocket=websocket)
    if not connected:
        return

    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "status":
                await websocket.send_json(
                    {"type": "status", "payload": manager.get_status()}
                )

            # DATASET-SPECIFIC: Define your data schema here
            elif message_type == "getInfo":
                dataset_info = DatasetInfo(
                    inputs={
                        "stock_input": InputInfo(
                            type="array",
                            shape=[manager.sequence_length],
                        )
                    },
                    outputs={
                        "stock_output": OutputInfo(
                            type="array",
                            shape=[manager.sequence_length],
                        )
                    },
                )

                # DATASET-SPECIFIC: Add diffusion inputs if needed
                if manager.is_diffusion:
                    dataset_info.inputs["stock_output"] = InputInfo(
                        type="array",
                        shape=[manager.sequence_length],
                    )

                await websocket.send_json(
                    {"type": "info", "payload": dataset_info.model_dump()}
                )

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
        logger.info("WebSocket disconnected")
    finally:
        try:
            await websocket.close()
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")
        logger.info("WebSocket connection closed")


def main():
    # DATASET-SPECIFIC: Customize CLI arguments for your use case
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

    # DATASET-SPECIFIC: Initialize your data manager here
    global manager
    manager = create_stock_manager(
        sequence_length=args.sequence_length,
        is_diffusion=args.diffusion,
    )

    logger.info("Stock data manager initialized")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, ws_ping_timeout=3600)


if __name__ == "__main__":
    main()
