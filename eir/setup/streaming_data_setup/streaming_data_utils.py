import json

import websockets

from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


async def send_validation_ids(ws_url: str, valid_ids: list[str]):
    async with websockets.connect(ws_url) as websocket:
        logger.info("Sending %d validation IDs to the server.", len(valid_ids))
        await websocket.send(
            json.dumps(
                {"type": "setValidationIds", "payload": {"validation_ids": valid_ids}}
            )
        )
        response = await websocket.recv()
        response_data = json.loads(response)
        if response_data["type"] != "validationIdsConfirmation":
            raise ValueError(f"Unexpected response type: {response_data['type']}")
        logger.info(f"Server response: {response_data['payload']['message']}")
