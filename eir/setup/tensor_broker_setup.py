from typing import Optional

from eir.setup.schema_modules.tensor_broker_schemas import (
    TensorBrokerConfig,
    TensorMessageConfig,
)


def set_up_tensor_broker_config(
    tensor_broker_config: dict,
) -> Optional[TensorBrokerConfig]:
    if not tensor_broker_config:
        return None

    tensor_message_configs = []
    for message_config in tensor_broker_config["message_configs"]:
        tensor_message_configs.append(TensorMessageConfig(**message_config))

    return TensorBrokerConfig(message_configs=tensor_message_configs)
