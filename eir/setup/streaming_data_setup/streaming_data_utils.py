import json
import logging
from contextlib import contextmanager

import websocket
from websocket._exceptions import WebSocketTimeoutException

from eir.setup.streaming_data_setup.protocol import (
    FROM_SERVER_MESSAGE_TYPES,
    PROTOCOL_VERSION,
)

logger = logging.getLogger(__name__)


def send_validation_ids(ws_url: str, valid_ids: list[str]):
    with connect_to_server(
        websocket_url=ws_url,
        protocol_version=PROTOCOL_VERSION,
        max_size=10_000_000,
    ) as ws:
        logger.info("Sending %d validation IDs to the server.", len(valid_ids))
        ws.send(
            json.dumps(
                {"type": "setValidationIds", "payload": {"validation_ids": valid_ids}}
            )
        )

        response_data = receive_with_timeout(websocket=ws)

        if response_data["type"] != "validationIdsConfirmation":
            raise ValueError(f"Unexpected response type: {response_data['type']}")
        logger.info(f"Server response: {response_data['payload']['message']}")


@contextmanager
def connect_to_server(
    websocket_url: str,
    protocol_version: str,
    max_size: int = 10_000_000,
):
    ws = websocket.create_connection(
        url=websocket_url,
        max_size=max_size,
    )

    try:
        ws.send(
            json.dumps(
                {
                    "type": "handshake",
                    "version": protocol_version,
                }
            )
        )

        response_data = receive_with_timeout(websocket=ws)
        is_not_handshake = response_data["type"] != "handshake"
        is_incompatible_version = response_data["version"] != protocol_version
        if is_not_handshake or is_incompatible_version:
            raise ValueError("Incompatible server version")

        logger.info(
            "Successfully connected to server with protocol version %s",
            protocol_version,
        )
        yield ws
    except Exception as e:
        logger.error(f"Error during connection or handshake: {e}")
        raise
    finally:
        ws.close()


def receive_with_timeout(websocket: websocket.WebSocket, timeout: int = 30):
    websocket.settimeout(timeout=timeout)
    try:
        raw_message = websocket.recv()
        message_parsed = json.loads(raw_message)
        handle_server_message(message=message_parsed)
        return message_parsed
    except WebSocketTimeoutException:
        logger.error("Server connection timed out")
        return None
    except json.JSONDecodeError:
        logger.error("Received message was not valid JSON")
        return None


def handle_server_message(message: dict):
    if message["type"] not in FROM_SERVER_MESSAGE_TYPES:
        logger.warning(f"Received unknown message type from server: {message['type']}")
