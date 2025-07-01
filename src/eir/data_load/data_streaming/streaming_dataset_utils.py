import json

from websocket import WebSocket

from eir.setup.input_setup_modules.setup_sequence import get_sequence_split_function
from eir.setup.streaming_data_setup.streaming_data_utils import receive_with_timeout
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


def streamline_sequence_manual_data(data: str, split_on: str | None) -> list[str] | str:
    """
    This is to specifically handle the case of an empty string / None being passed
    here. If e.g. we call the split_func on '', we will get [''], which will
    end up being encoded as a <unk> token. Instead, we want to return an empty
    list here. In e.g. the validation handler code, this is also set explicitly.
    """

    sequence_streamlined: list[str] | str
    if data == "" or data is None:
        sequence_streamlined = []
    else:
        split_func = get_sequence_split_function(split_on=split_on)
        split_data = split_func(data)
        sequence_streamlined = split_data

    return sequence_streamlined


def validate_server_protocol(ws: WebSocket) -> None:
    try:
        logger.debug("Validating server protocol...")

        # 1. Test getInfo
        ws.send(json.dumps({"type": "getInfo"}))
        response = receive_with_timeout(ws, timeout=5)
        if not response or response.get("type") != "info":
            raise RuntimeError(
                "Validation failed for 'getInfo'. "
                f"Expected 'info', got '"
                f"{response.get('type') if response else 'None'}'."
            )
        logger.debug("Endpoint 'getInfo' validated.")

        # 2. Test getData
        ws.send(json.dumps({"type": "getData", "payload": {"batch_size": 1}}))
        response = receive_with_timeout(ws, timeout=5)
        if not response or response.get("type") != "data":
            raise RuntimeError(
                "Validation failed for 'getData'. "
                f"Expected 'data', got '"
                f"{response.get('type') if response else 'None'}'."
            )
        logger.debug("Endpoint 'getData' validated.")

        # 3. Test heartbeat
        ws.send(json.dumps({"type": "heartbeat"}))
        response = receive_with_timeout(ws, timeout=5)
        if not response or response.get("type") != "heartbeat":
            raise RuntimeError(
                "Validation failed for 'heartbeat'. "
                f"Expected 'heartbeat', got '"
                f"{response.get('type') if response else 'None'}'."
            )
        logger.debug("Endpoint 'heartbeat' validated.")

        # 4. Test and execute reset to ensure a clean state
        # The protocol involves a private 'resetConfirmation' and a potential
        # public 'reset' broadcast. This test confirms the confirmation is received
        # while safely handling the optional broadcast message
        ws.send(json.dumps({"type": "reset"}))

        # Read up to two potential responses to clear the message queue
        response1 = receive_with_timeout(ws, timeout=5)  # direct reset confirmation
        response2 = receive_with_timeout(ws, timeout=5)  # potential reset broadcast

        # Check if the required confirmation was in either response
        responses = [response1, response2]
        received_types = {res.get("type") for res in responses if res}

        if "resetConfirmation" not in received_types:
            raise RuntimeError(
                "Validation failed for 'reset'. No 'resetConfirmation' received. "
                f"Got: {[t for t in received_types if t]}"
            )
        logger.debug("Endpoint 'reset' validated.")

        logger.info("Server protocol validation successful.")

    except Exception as e:
        logger.error(f"Server protocol validation failed: {e}")
        raise RuntimeError(
            f"Server does not seem to implement the required protocol. Failure on: "
            f"{str(e)}"
        ) from e
