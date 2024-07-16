import asyncio
import json
from contextlib import asynccontextmanager

import websockets
from websockets import WebSocketClientProtocol

from eir.setup.streaming_data_setup.protocol import (
    FROM_SERVER_MESSAGE_TYPES,
    PROTOCOL_VERSION,
)
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


async def send_validation_ids(ws_url: str, valid_ids: list[str]):
    async with connect_to_server(
        websocket_url=ws_url,
        protocol_version=PROTOCOL_VERSION,
        max_size=10_000_000,
    ) as websocket:
        logger.info("Sending %d validation IDs to the server.", len(valid_ids))
        await websocket.send(
            json.dumps(
                {"type": "setValidationIds", "payload": {"validation_ids": valid_ids}}
            )
        )

        response_data = await receive_with_timeout(websocket=websocket)

        if response_data["type"] != "validationIdsConfirmation":
            raise ValueError(f"Unexpected response type: {response_data['type']}")
        logger.info(f"Server response: {response_data['payload']['message']}")


@asynccontextmanager
async def connect_to_server(
    websocket_url: str,
    protocol_version: str,
    max_size: int = 10_000_000,
):
    async with websockets.connect(uri=websocket_url, max_size=max_size) as websocket:
        try:
            await websocket.send(
                json.dumps(
                    {
                        "type": "handshake",
                        "version": protocol_version,
                    }
                )
            )

            response_data = await receive_with_timeout(websocket=websocket)
            is_not_handshake = response_data["type"] != "handshake"
            is_incompatible_version = response_data["version"] != protocol_version
            if is_not_handshake or is_incompatible_version:
                raise ValueError("Incompatible server version")

            logger.info(
                "Successfully to server with protocol version %s", protocol_version
            )
            yield websocket
        except Exception as e:
            logger.error(f"Error during connection or handshake: {e}")
            raise


async def receive_with_timeout(websocket: WebSocketClientProtocol, timeout: int = 30):
    try:
        raw_message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
        message_parsed = json.loads(s=raw_message)
        await handle_server_message(message=message_parsed)
        return message_parsed
    except asyncio.TimeoutError:
        logger.error(f"No message received from client in {timeout} seconds")
        return None
    except json.JSONDecodeError:
        logger.error("Received message was not valid JSON")
        return None


async def handle_server_message(message: dict):
    if message["type"] not in FROM_SERVER_MESSAGE_TYPES:
        logger.warning(f"Received unknown message type from server: {message['type']}")
