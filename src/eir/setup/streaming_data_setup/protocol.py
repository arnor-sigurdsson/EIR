# protocol.py
from typing import Dict, List, TypedDict, Union


class GetInfoPayload(TypedDict):
    pass


class GetDataPayload(TypedDict):
    batch_size: int


class StatusPayload(TypedDict):
    pass


class ResetPayload(TypedDict):
    pass


class InfoPayload(TypedDict):
    dataset_info: Dict[str, Union[str, List[int]]]


class DataPayload(TypedDict):
    samples: List[Dict[str, Union[List[float], str]]]


class ResetConfirmationPayload(TypedDict):
    message: str


class ErrorPayload(TypedDict):
    message: str


class HeartbeatPayload(TypedDict):
    pass


class HandshakePayload(TypedDict):
    pass


class ValidationIdsConfirmationPayload(TypedDict):
    message: str


MESSAGE_TYPES = {
    "getInfo": {"direction": "client_to_server", "payload": GetInfoPayload},
    "getData": {"direction": "client_to_server", "payload": GetDataPayload},
    "status": {"direction": "bidirectional", "payload": StatusPayload},
    "reset": {"direction": "client_to_server", "payload": ResetPayload},
    "resetConfirmation": {
        "direction": "server_to_client",
        "payload": ResetConfirmationPayload,
    },
    "info": {"direction": "server_to_client", "payload": InfoPayload},
    "data": {"direction": "server_to_client", "payload": DataPayload},
    "error": {"direction": "bidirectional", "payload": ErrorPayload},
    "heartbeat": {"direction": "bidirectional", "payload": HeartbeatPayload},
    "handshake": {"direction": "bidirectional", "payload": HandshakePayload},
    "validationIdsConfirmation": {
        "direction": "server_to_client",
        "payload": ValidationIdsConfirmationPayload,
    },
}

FROM_SERVER_MESSAGE_TYPES = {
    "info",
    "data",
    "status",
    "heartbeat",
    "error",
    "handshake",
    "validationIdsConfirmation",
    "resetConfirmation",
}

PROTOCOL_VERSION = "1.0"
