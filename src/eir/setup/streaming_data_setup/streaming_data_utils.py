import json
import logging
from contextlib import contextmanager
from typing import Generator

import websocket
from websocket._exceptions import WebSocketBadStatusException, WebSocketTimeoutException

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
) -> Generator[websocket.WebSocket, None, None]:
    logger.debug(f"Attempting WebSocket connection to: {websocket_url}")

    try:
        ws = websocket.create_connection(
            url=websocket_url,
            max_size=max_size,
        )
    except WebSocketBadStatusException as e:
        logger.error(f"Failed to establish WebSocket connection: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected connection error: {type(e).__name__} - {str(e)}")
        raise

    try:
        handshake_payload = {
            "type": "handshake",
            "version": protocol_version,
        }
        ws.send(json.dumps(handshake_payload))

        response_data = receive_with_timeout(websocket=ws)
        if response_data is None:
            raise ValueError("No handshake response received from server")

        if response_data["type"] != "handshake":
            raise ValueError(
                f"Expected handshake response, got: {response_data['type']}"
            )

        if response_data["version"] != protocol_version:
            raise ValueError(
                f"Protocol version mismatch. Expected: {protocol_version}, "
                f"Got: {response_data.get('version', 'unknown')}"
            )

        logger.info(f"Connected to server with protocol version {protocol_version}")
        yield ws
    except Exception as e:
        logger.error(f"Error during handshake: {str(e)}")
        raise
    finally:
        try:
            ws.close()
        except Exception as e:
            logger.warning(f"Error closing WebSocket connection: {str(e)}")


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
