import argparse
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from docs.doc_modules.user_guides.data_simulator import create_simulator
from eir.setup.streaming_data_setup.protocol import PROTOCOL_VERSION
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)

app = FastAPI()
simulator = create_simulator()


class InputInfo(BaseModel):
    type: str
    shape: list[int] | None = None


class OutputInfo(BaseModel):
    type: str
    shape: list[int] | None = None


class DatasetInfo(BaseModel):
    inputs: dict[str, InputInfo]
    outputs: dict[str, OutputInfo]


# start-connect-websocket
async def connect_websocket(websocket: WebSocket) -> bool:
    try:
        # P1 Step 1: Accept the WebSocket connection from the EIR-client.
        await websocket.accept()
        # P1 Step 2: Receive the handshake message from the EIR-client.
        handshake_message = await websocket.receive_json()

        # P1 Step 3: Validate the handshake message.
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

        # P1 Step 4: Send a handshake response back to the EIR-client.
        await websocket.send_json({"type": "handshake", "version": PROTOCOL_VERSION})
        return True

    except Exception as e:
        logger.error(f"Error in connect: {e}")
        return False


# end-connect-websocket


# start-websocket-endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # P1 Steps 1-4: Establish WebSocket connection and perform handshake.
    connected = await connect_websocket(websocket=websocket)
    if not connected:
        return

    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")

            # P2 Steps 1-2 & 9-10: Client (Gatherer) requests server status before
            # and after data gathering.
            if message_type == "status":
                await websocket.send_json(
                    {"type": "status", "payload": simulator.get_status()}
                )

            # P2 Steps 3-4: Client (Gatherer) requests the dataset schema.
            # NOTE: This is specific to the dataset
            elif message_type == "getInfo":
                dataset_info = DatasetInfo(
                    inputs={
                        "text_input": InputInfo(
                            type="sequence",
                        )
                    },
                    outputs={"text_output": OutputInfo(type="sequence")},
                )
                await websocket.send_json(
                    {"type": "info", "payload": dataset_info.model_dump()}
                )

            # P2, Steps 5-6: The Client (Gatherer) requests data, and the server
            # responds.
            # AND
            # P3, Steps 1-2: The Client (Trainer) requests data, and the server
            # responds.
            # Note: The server logic is identical for both phases.
            elif message_type == "getData":
                batch_size = data.get("payload", {}).get("batch_size", 32)
                batch = simulator.get_batch(batch_size=batch_size)

                if not batch:
                    await websocket.send_json(
                        {"type": "data", "payload": ["terminate"]}
                    )
                    break

                await websocket.send_json({"type": "data", "payload": batch})

            # P2 Steps 7-8: Client (Gatherer) requests to reset the server state
            # after gathering data.
            # This handles a reset command by performing two distinct actions
            # to support multi-client environments.
            # 1. A private 'resetConfirmation' is sent directly to the client
            #    that initiated the request, acknowledging their command was successful.
            # 2. A public 'reset' message is broadcast to all connected clients
            #    to notify them of the state change, ensuring synchronization.
            elif message_type == "reset":
                simulator.reset()
                await websocket.send_json(
                    {
                        "type": "resetConfirmation",
                        "payload": {"message": "Reset successful"},
                    }
                )
                await websocket.send_json(
                    {"type": "reset", "payload": {"message": "Reset command received"}}
                )

            # --- Ancillary Commands ---

            elif message_type == "setValidationIds":
                validation_ids = data.get("payload", {}).get("validation_ids", [])
                validation_ids = set(validation_ids)

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
        print("WebSocket disconnected")
        raise
    finally:
        try:
            await websocket.close()
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")
        logger.info("WebSocket connection closed")
        simulator.reset()


# end-websocket-endpoint


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
        help="Maximum number of iterations before terminating",
    )

    args = parser.parse_args()

    if args.max_iterations is not None:
        os.environ["MAX_ITERATIONS"] = str(args.max_iterations)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, ws_ping_timeout=3600)


if __name__ == "__main__":
    main()
