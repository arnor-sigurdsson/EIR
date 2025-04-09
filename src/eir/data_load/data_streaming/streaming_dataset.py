import json
import time
from collections.abc import Iterator
from contextlib import _GeneratorContextManager
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import websocket
from PIL import Image
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, StandardScaler
from torch.utils.data import IterableDataset
from websocket import WebSocket

from eir.data_load.data_preparation_modules.imputation import (
    impute_missing_modalities_wrapper,
    impute_missing_output_modalities_wrapper,
)
from eir.data_load.data_preparation_modules.input_preparation_wrappers import (
    prepare_inputs_memory,
)
from eir.data_load.data_preparation_modules.output_preparation_wrappers import (
    prepare_outputs_memory,
)
from eir.data_load.data_utils import Sample
from eir.predict_modules.predict_tabular_input_setup import (
    ComputedPredictTabularInputInfo,
)
from eir.serve_modules.serve_network_utils import (
    deserialize_array,
    deserialize_image,
    prepare_request_input_data,
    streamline_sequence_manual_data,
)
from eir.serve_modules.serve_schemas import ComputedServeTabularInputInfo
from eir.setup.input_setup import al_input_objects_as_dict
from eir.setup.input_setup_modules.setup_sequence import ComputedSequenceInputInfo
from eir.setup.input_setup_modules.setup_tabular import ComputedTabularInputInfo
from eir.setup.output_setup import ComputedArrayOutputInfo, al_output_objects_as_dict
from eir.setup.output_setup_modules.image_output_setup import ComputedImageOutputInfo
from eir.setup.output_setup_modules.sequence_output_setup import (
    ComputedSequenceOutputInfo,
)
from eir.setup.output_setup_modules.survival_output_setup import (
    ComputedSurvivalOutputInfo,
)
from eir.setup.output_setup_modules.tabular_output_setup import (
    ComputedTabularOutputInfo,
)
from eir.setup.schemas import ImageOutputTypeConfig, SequenceOutputTypeConfig
from eir.setup.streaming_data_setup.protocol import PROTOCOL_VERSION
from eir.setup.streaming_data_setup.streaming_data_utils import (
    connect_to_server,
    receive_with_timeout,
)
from eir.target_setup.target_setup_utils import IdentityTransformer
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.data_load.datasets import al_getitem_return


al_outputs_prepared = dict[
    str,
    np.ndarray | torch.Tensor | list[str] | str | dict | Image.Image,
]

logger = get_logger(name=__name__)


class StreamingDataset(IterableDataset):
    def __init__(
        self,
        websocket_url: str,
        inputs: al_input_objects_as_dict,
        outputs: al_output_objects_as_dict,
        test_mode: bool,
        batch_size: int = 32,
        fetch_timeout: float = 10.0,
        max_consecutive_timeouts: int = 3,
        heartbeat_interval: float = 30.0,
    ):
        super().__init__()
        self.websocket_url = websocket_url
        self.inputs = inputs
        self.outputs = outputs
        self.test_mode = test_mode
        self.batch_size = batch_size
        self.fetch_timeout = fetch_timeout
        self.max_consecutive_timeouts = max_consecutive_timeouts
        self.heartbeat_interval = heartbeat_interval

        self.ws: WebSocket | None = None
        self.connection_context: _GeneratorContextManager[WebSocket] | None = None
        self.current_batch: list | None = None
        self._is_closed = False

    def __iter__(self) -> Iterator[Any]:
        self.ws = None
        self.current_batch = []
        self._is_closed = False
        self._connect()

        return self

    def _connect(self):
        self.connection_context = connect_to_server(
            websocket_url=self.websocket_url,
            protocol_version=PROTOCOL_VERSION,
        )
        self.ws = self.connection_context.__enter__()

        if self.ws:
            try:
                self.ws.send(json.dumps({"type": "heartbeat"}))
            except Exception as e:
                logger.error(f"Error sending initial heartbeat: {e}")

    def __next__(self) -> Any:
        if not self.current_batch:
            self._fetch_batch()

        if not self.current_batch:
            self.close()
            raise StopIteration

        sample_dict = self.current_batch.pop(0)
        sample = Sample(**sample_dict)
        return self._process_sample(sample)

    def _fetch_batch(self):
        consecutive_timeouts = 0
        assert self.ws is not None

        while consecutive_timeouts < self.max_consecutive_timeouts:
            try:
                if self.ws.sock is None:
                    logger.warning("Connection appears to be closed. Reconnecting...")
                    self._reconnect()

                self.ws.send(json.dumps({"type": "heartbeat"}))

                self.ws.send(
                    json.dumps(
                        {
                            "type": "getData",
                            "payload": {"batch_size": self.batch_size},
                        }
                    )
                )

                batch_msg = receive_with_timeout(websocket_=self.ws)
                if batch_msg is None:
                    consecutive_timeouts += 1
                    if consecutive_timeouts > 1:
                        extra_delay = min(2 ** (consecutive_timeouts - 1), 30)
                        logger.warning(
                            f"Received None message. "
                            f"Consecutive errors: {consecutive_timeouts}. "
                            f"Adding extra backoff delay of {extra_delay} seconds..."
                        )
                        time.sleep(extra_delay)
                    else:
                        logger.warning(
                            f"Received None message. "
                            f"Consecutive errors: {consecutive_timeouts}."
                        )
                    continue

                consecutive_timeouts = 0

                if batch_msg["type"] == "heartbeat":
                    continue

                if batch_msg["type"] == "error":
                    logger.error(f"Error message received: {batch_msg['payload']}")
                    continue

                if batch_msg["type"] != "data":
                    logger.warning(f"Unexpected message type: {batch_msg['type']}")
                    continue

                if not batch_msg["payload"] or batch_msg["payload"] == ["terminate"]:
                    logger.info(
                        "Received termination signal or empty payload, stopping."
                    )
                    raise StopIteration

                self.current_batch = batch_msg["payload"]
                return

            except websocket.WebSocketTimeoutException:
                consecutive_timeouts += 1
                logger.warning(
                    f"Timeout waiting for next batch. "
                    f"Consecutive timeouts: {consecutive_timeouts}"
                )
            except ConnectionResetError as e:
                logger.warning(
                    f"Connection reset by peer ({str(e)}). Attempting to reconnect."
                )
                self._reconnect()
                time.sleep(0.5)
            except websocket.WebSocketConnectionClosedException:
                logger.warning("WebSocket connection closed. Attempting to reconnect.")
                time.sleep(0.5)
            except OSError as e:
                if e.errno == 32:
                    logger.warning(
                        f"Broken pipe detected ({str(e)}). Attempting to reconnect."
                    )
                    self._reconnect()
                    time.sleep(0.5)
                else:
                    logger.error(f"Unexpected OSError: {e}", exc_info=True)
                    raise
            except StopIteration:
                raise
            except Exception as e:
                logger.error(f"Error fetching batch: {e}", exc_info=True)
                raise

        logger.error(
            f"Max consecutive timeouts ({self.max_consecutive_timeouts}) reached."
            f" Stopping iterator."
        )
        raise StopIteration

    def _reconnect(self):
        if self.ws and self.connection_context:
            self.connection_context.__exit__(None, None, None)
        self._connect()

    def close(self):
        if not self._is_closed:
            self._is_closed = True
            if self.ws and self.connection_context:
                self.connection_context.__exit__(None, None, None)

    def _process_sample(self, sample: Sample) -> "al_getitem_return":
        inputs = sample.inputs
        inputs_request_parsed = prepare_request_input_data(
            request_data=inputs,
            input_objects=self.inputs,
        )

        inputs_prepared_for_memory = prepare_inputs_for_in_memory_processing(
            inputs_request_parsed=inputs_request_parsed,
            input_objects=self.inputs,
        )

        inputs_prepared = prepare_inputs_memory(
            inputs=inputs_prepared_for_memory,
            inputs_objects=self.inputs,
            test_mode=self.test_mode,
        )

        inputs_final = impute_missing_modalities_wrapper(
            inputs_values=inputs_prepared,
            inputs_objects=self.inputs,
        )

        target_labels = sample.target_labels

        targets_request_parsed = prepare_request_output_data(
            request_data=target_labels,
            output_objects=self.outputs,
        )

        targets_prepared_for_memory = prepare_outputs_for_in_memory_processing(
            target_labels=targets_request_parsed,
            output_objects=self.outputs,
        )

        targets_prepared = prepare_outputs_memory(
            outputs=targets_prepared_for_memory,
            output_objects=self.outputs,
            test_mode=self.test_mode,
        )

        targets_final = impute_missing_output_modalities_wrapper(
            outputs_values=targets_prepared,
            output_objects=self.outputs,
        )

        sample_id = sample.sample_id

        return inputs_final, targets_final, sample_id


def prepare_inputs_for_in_memory_processing(
    inputs_request_parsed: dict[str, Any],
    input_objects: al_input_objects_as_dict,
) -> dict[str, Any]:
    inputs_prepared_for_memory = {}
    for name, cur_input in inputs_request_parsed.items():
        input_object = input_objects[name]

        match input_object:
            case ComputedSequenceInputInfo():
                cur_input = input_object.encode_func(cur_input)

            case (
                ComputedTabularInputInfo()
                | ComputedPredictTabularInputInfo()
                | ComputedServeTabularInputInfo()
            ):
                cur_input = _impute_missing_tabular_values(
                    input_object=input_object,
                    inputs_values=cur_input,
                )

        inputs_prepared_for_memory[name] = cur_input

    return inputs_prepared_for_memory


def prepare_request_output_data(
    request_data: dict[str, Any],
    output_objects: "al_output_objects_as_dict",
) -> dict[str, Any]:
    outputs_prepared: al_outputs_prepared = {}

    for output_name, serialized_data in request_data.items():
        output_object = output_objects[output_name]

        output_config = output_object.output_config
        output_type = output_config.output_info.output_type
        output_type_info = output_config.output_type_info

        match output_object:
            case ComputedSequenceOutputInfo():
                assert output_type == "sequence"
                assert isinstance(output_type_info, SequenceOutputTypeConfig)

                value = serialized_data[output_name]
                sequence_streamlined = streamline_sequence_manual_data(
                    data=value,
                    split_on=output_type_info.split_on,
                )

                outputs_prepared[output_name] = {output_name: sequence_streamlined}

            case ComputedImageOutputInfo():
                assert output_type == "image"
                assert isinstance(output_type_info, ImageOutputTypeConfig)

                value = serialized_data[output_name]
                image_data = deserialize_image(
                    image_str=value,
                    image_mode=output_type_info.mode,
                )

                outputs_prepared[output_name] = {output_name: image_data}

            case ComputedTabularOutputInfo() | ComputedSurvivalOutputInfo():
                assert output_type in ("tabular", "survival")
                tabular_data = apply_tabular_transformers(
                    cur_target=serialized_data,
                    target_transformers=output_object.target_transformers,
                    target_columns=output_object.target_columns,
                )

                outputs_prepared[output_name] = tabular_data

            case ComputedArrayOutputInfo():
                assert output_type == "array"

                value = serialized_data[output_name]
                array_np = deserialize_array(
                    array_str=value,
                    dtype=output_object.dtype,
                    shape=output_object.data_dimensions.full_shape(),
                )
                outputs_prepared[output_name] = {output_name: array_np}

            case _:
                raise ValueError(f"Unknown output type '{output_type}'")

    return outputs_prepared


def prepare_outputs_for_in_memory_processing(
    target_labels: dict[str, Any],
    output_objects: al_output_objects_as_dict,
) -> dict[str, Any]:
    targets_prepared_for_memory = {}
    for name, cur_target in target_labels.items():
        output_object = output_objects[name]

        match output_object:
            case ComputedSequenceOutputInfo():
                # This is set up on the fly in the batch step logic
                # for sequence outputs from the linked input data
                targets_prepared_for_memory[name] = {name: "DELAYED"}

            case _:
                targets_prepared_for_memory[name] = cur_target

    return targets_prepared_for_memory


def apply_tabular_transformers(
    cur_target: dict[str, int | float | torch.Tensor],
    target_transformers: dict[str, StandardScaler | LabelEncoder],
    target_columns: dict[Literal["con", "cat"], list[str]],
) -> dict[str, int | float | torch.Tensor]:
    transformed_target = {}

    for _col_type, columns in target_columns.items():
        for column in columns:
            if column not in cur_target:
                raise ValueError(f"Column {column} not found in target data.")

            if column not in target_transformers:
                raise ValueError(f"Transformer for column {column} not found.")

            transformer = target_transformers[column]
            value = cur_target[column]

            if isinstance(transformer, StandardScaler):
                transformed_value = transformer.transform([[value]])[0]
            elif isinstance(transformer, LabelEncoder):
                transformed_value = transformer.transform([value])
            elif isinstance(transformer, KBinsDiscretizer | IdentityTransformer):
                transformed_value = transformer.transform([[value]])[0]
            else:
                raise ValueError(f"Unknown transformer type for column {column}")

            transformed_target[column] = transformed_value.item()

    return transformed_target


def _impute_missing_tabular_values(
    input_object: (
        ComputedTabularInputInfo
        | ComputedPredictTabularInputInfo
        | ComputedServeTabularInputInfo
    ),
    inputs_values: dict[str, Any],
) -> dict[str, Any]:
    # TODO: Implement
    return inputs_values
