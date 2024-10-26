import atexit
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import websocket
from aislib.misc_utils import ensure_path_exists
from PIL import Image
from tqdm import tqdm

from eir.serve_modules.serve_network_utils import deserialize_array, deserialize_image
from eir.setup.config import Configs
from eir.setup.schemas import InputConfig, OutputConfig
from eir.setup.streaming_data_setup.protocol import PROTOCOL_VERSION
from eir.setup.streaming_data_setup.streaming_data_utils import (
    connect_to_server,
    receive_with_timeout,
)
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


def cleanup_streaming_setup(path: Path) -> None:
    if path.exists():
        logger.info(f"Cleaning up streaming setup folder: {path}")
        try:
            shutil.rmtree(path)
            logger.info("Cleanup completed successfully.")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    else:
        logger.info(f"No cleanup needed. Folder does not exist: {path}")


class StreamDataGatherer:
    def __init__(
        self,
        websocket_url: str,
        output_folder: str,
        input_configs: Dict[str, Any],
        output_configs: Dict[str, Any],
        batch_size: int = 32,
        max_samples: int = 1_000,
    ):
        self.websocket_url = websocket_url
        self.output_folder = output_folder
        self.inputs = input_configs
        self.outputs = output_configs
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.base_path = Path(f"{output_folder}/streaming_setup")
        self.dataset_info = None
        self.input_dataframes: dict[str, pd.DataFrame] = {}
        self.output_dataframes: dict[str, pd.DataFrame] = {}

        atexit.register(cleanup_streaming_setup, self.base_path)

    def get_dataset_info(self, ws: websocket.WebSocket):
        ws.send(json.dumps({"type": "getInfo"}))

        info_data = receive_with_timeout(websocket=ws)

        if info_data["type"] != "info":
            raise ValueError(f"Unexpected response type: {info_data['type']}")

        self.dataset_info = info_data["payload"]
        logger.info("Received dataset information.")

    def gather_and_save_data(self):
        self.base_path.mkdir(parents=True, exist_ok=True)

        with connect_to_server(
            websocket_url=self.websocket_url,
            protocol_version=PROTOCOL_VERSION,
        ) as ws:
            self.get_dataset_info(ws=ws)

            total_samples = 0
            pbar = tqdm(
                total=self.max_samples,
                desc="Gathering samples information and statistics",
                unit=" sample",
            )
            while total_samples < self.max_samples:
                ws.send(
                    json.dumps(
                        {
                            "type": "getData",
                            "payload": {"batch_size": self.batch_size},
                        }
                    )
                )

                batch_data = receive_with_timeout(websocket=ws)

                if batch_data["type"] != "data":
                    logger.error(f"Unexpected response type: {batch_data['type']}")
                    continue

                samples = batch_data["payload"]

                assert self.dataset_info is not None
                for sample in samples:
                    processed_input = process_inputs(
                        sample_input=sample["inputs"],
                        dataset_info=self.dataset_info,
                    )
                    processed_output = process_outputs(
                        sample_output=sample["target_labels"],
                        dataset_info=self.dataset_info,
                    )

                    save_inputs(
                        processed_input=processed_input,
                        sample_id=sample["sample_id"],
                        base_path=self.base_path,
                        inputs=self.dataset_info["inputs"],
                        dataframes=self.input_dataframes,
                    )
                    save_outputs(
                        processed_output=processed_output,
                        sample_id=sample["sample_id"],
                        base_path=self.base_path,
                        outputs=self.dataset_info["outputs"],
                        dataframes=self.output_dataframes,
                    )

                    total_samples += 1
                    pbar.update(1)
                    if total_samples >= self.max_samples:
                        break

            pbar.close()
            logger.info(f"Finished processing {total_samples} samples.")

        for name, df in self.input_dataframes.items():
            output_path = self.base_path / "input" / name
            ensure_path_exists(path=output_path, is_folder=True)
            df.to_csv(output_path / f"{name}.csv", index=False)

        for name, df in self.output_dataframes.items():
            output_path = self.base_path / "output" / name
            ensure_path_exists(path=output_path, is_folder=True)
            df.to_csv(output_path / f"{name}.csv", index=False)

    def reset(self):
        with connect_to_server(
            websocket_url=self.websocket_url,
            protocol_version=PROTOCOL_VERSION,
        ) as ws:
            ws.send(
                json.dumps(
                    {
                        "type": "reset",
                        "payload": {},
                    },
                )
            )

            for _ in range(2):
                reset_message = receive_with_timeout(websocket=ws)

                if reset_message["type"] == "reset":
                    logger.info(
                        f"Received broadcast: {reset_message['payload']['message']}"
                    )
                elif reset_message["type"] == "resetConfirmation":
                    logger.info("Reset command sent successfully.")
                else:
                    logger.error(
                        f"Unexpected response to reset command: {reset_message}"
                    )

    def get_status(self):
        with connect_to_server(
            websocket_url=self.websocket_url,
            protocol_version=PROTOCOL_VERSION,
        ) as ws:
            ws.send(
                json.dumps(
                    {
                        "type": "status",
                        "payload": {},
                    },
                )
            )

            status_data = receive_with_timeout(websocket=ws)

            if status_data["type"] == "status":
                logger.info(f"Current status: {status_data['payload']}")
                return status_data["payload"]
            else:
                logger.error(f"Unexpected response to status request: {status_data}")


def process_single_data(data: Any, info: Dict[str, Any]) -> Any:
    data_type = info["type"]

    if data_type in ["array", "omics"]:
        dtype = np.dtype(bool) if data_type == "omics" else np.dtype(np.float32)
        return process_array(data=data, shape=info["shape"], dtype=dtype)
    elif data_type == "image":
        return process_image(data=data, mode=info.get("mode"))
    elif data_type in ["tabular", "sequence", "survival"]:
        return data
    else:
        logger.warning(f"Unsupported data type: {data_type}")
        return data


def process_array(data: str, shape: list[int], dtype: np.dtype) -> np.ndarray:
    array = deserialize_array(
        array_str=data,
        dtype=dtype,
        shape=tuple(shape),
    )

    return array


def process_image(
    data: str, mode: Optional[Literal["L", "RGB", "RGBA"]]
) -> Image.Image:
    image = deserialize_image(
        image_str=data,
        image_mode=mode,
    )

    return image


def process_inputs(
    sample_input: dict[str, Any],
    dataset_info: dict[str, Any],
) -> dict[str, Any]:
    processed_inputs = {}
    for input_name, input_data in sample_input.items():
        input_info = dataset_info["inputs"].get(input_name)

        if input_info:
            processed_inputs[input_name] = process_single_data(
                data=input_data,
                info=input_info,
            )
        else:
            logger.warning(f"No info found for input {input_name}")
            processed_inputs[input_name] = input_data
    return processed_inputs


def process_outputs(
    sample_output: Dict[str, Any], dataset_info: Dict[str, Any]
) -> Dict[str, Any]:
    processed_outputs = {}
    for output_name, output_data in sample_output.items():
        output_info = dataset_info["outputs"].get(output_name)
        output_type = output_info["type"]

        if output_type in ["array", "sequence", "image"]:
            output_data = output_data[output_name]

        if output_info:
            processed_outputs[output_name] = process_single_data(
                data=output_data,
                info=output_info,
            )
        else:
            logger.warning(f"No info found for output {output_name}")
            processed_outputs[output_name] = output_data
    return processed_outputs


def save_data(
    data_type: str,
    data_name: str,
    data: Any,
    save_path: Path,
    sample_id: str,
    dataframes: Dict[str, pd.DataFrame],
):
    if data_type in ["omics", "array"]:
        assert isinstance(data, np.ndarray)
        np.save(save_path / f"{sample_id}.npy", data)
    elif data_type == "image":
        assert isinstance(data, Image.Image)
        data.save(save_path / f"{sample_id}.png")
    elif data_type in ["tabular", "sequence", "survival"]:
        if data_name not in dataframes:
            dataframes[data_name] = pd.DataFrame(columns=["ID"])

        df = dataframes[data_name]
        if data_type in ("tabular", "survival"):
            assert isinstance(data, dict)
            new_row = pd.DataFrame({"ID": [sample_id], **data})
        else:
            assert isinstance(data, str)
            new_row = pd.DataFrame({"ID": [sample_id], "Sequence": [data]})

        dataframes[data_name] = pd.concat([df, new_row], ignore_index=True)
    else:
        logger.warning(f"Unsupported data type: {data_type}")


def save_inputs(
    processed_input: Dict[str, Any],
    sample_id: str,
    base_path: Path,
    inputs: Dict[str, Dict],
    dataframes: Dict[str, pd.DataFrame],
):
    for input_name, input_config in inputs.items():
        input_type = input_config["type"]
        input_data = processed_input.get(input_name)

        if input_data is None:
            logger.warning(f"No data for input {input_name} in sample {sample_id}")
            continue

        save_path = base_path / "input" / input_name
        save_path.mkdir(parents=True, exist_ok=True)

        save_data(
            data_type=input_type,
            data_name=input_name,
            data=input_data,
            save_path=save_path,
            sample_id=sample_id,
            dataframes=dataframes,
        )


def save_outputs(
    processed_output: Dict[str, Any],
    sample_id: str,
    base_path: Path,
    outputs: Dict[str, Dict],
    dataframes: Dict[str, pd.DataFrame],
):
    for output_name, output_config in outputs.items():
        output_type = output_config["type"]
        output_data = processed_output.get(output_name)

        if output_data is None:
            logger.warning(f"No data for output {output_name} in sample {sample_id}")
            continue

        save_path = base_path / "output" / output_name
        save_path.mkdir(parents=True, exist_ok=True)

        save_data(
            data_type=output_type,
            data_name=output_name,
            data=output_data,
            save_path=save_path,
            sample_id=sample_id,
            dataframes=dataframes,
        )


def validate_streaming_setup(configs: Configs) -> Optional[str]:
    def is_websocket_url(url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme in ("ws", "wss")

    websocket_address = None
    sources_to_check = []

    for input_config in configs.input_configs:
        sources_to_check.append(input_config.input_info.input_source)

    for output_config in configs.output_configs:
        sources_to_check.append(output_config.output_info.output_source)

    for source in sources_to_check:
        if is_websocket_url(source):
            if websocket_address is None:
                websocket_address = source
            elif source != websocket_address:
                raise ValueError(
                    f"Inconsistent WebSocket addresses found. "
                    f"Expected {websocket_address}, but found {source}"
                )
        elif websocket_address is not None:
            raise ValueError(
                f"Mixed source types found. "
                f"Expected all sources to be WebSocket if one is present. "
                f"Found WebSocket {websocket_address} and non-WebSocket {source}"
            )

    return websocket_address


def gather_streaming_data_for_setup(
    websocket_url: str,
    output_folder: str,
    configs: Configs,
    batch_size: int,
    max_samples: int,
) -> tuple[str, Path]:
    input_configs = configs.input_configs
    output_configs = configs.output_configs

    inputs = {}
    for input_config in input_configs:
        input_name = input_config.input_info.input_name
        input_type = input_config.input_info.input_type
        inputs[input_name] = {"type": input_type}

    outputs = {}
    for output_config in output_configs:
        output_name = output_config.output_info.output_name
        output_type = output_config.output_info.output_type
        outputs[output_name] = {"type": output_type}

    gatherer = StreamDataGatherer(
        websocket_url=websocket_url,
        output_folder=output_folder,
        input_configs=inputs,
        output_configs=outputs,
        batch_size=batch_size,
        max_samples=max_samples,
    )

    gatherer.get_status()

    gatherer.gather_and_save_data()

    gatherer.reset()

    gatherer.get_status()

    return websocket_url, gatherer.base_path


def patch_configs_for_local_data(
    configs: Configs,
    local_data_path: Path,
) -> tuple[Configs, Configs]:
    patched_configs = deepcopy(configs)

    def update_source(
        config: Union[InputConfig, OutputConfig],
        config_type: str,
    ) -> None:
        match config:
            case InputConfig():
                name = config.input_info.input_name
                data_type = config.input_info.input_type
            case OutputConfig():
                name = config.output_info.output_name
                data_type = config.output_info.output_type
            case _:
                raise ValueError("Unsupported config type")

        new_source: str | Path
        if data_type in ["tabular", "sequence", "survival"]:
            new_source = local_data_path / config_type / name / f"{name}.csv"
        elif data_type in ["omics", "array", "image"]:
            new_source = str(local_data_path / config_type / name)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        if config_type == "input":
            assert isinstance(config, InputConfig)
            config.input_info.input_source = str(new_source)
        else:
            assert isinstance(config, OutputConfig)
            config.output_info.output_source = str(new_source)

    for input_config in patched_configs.input_configs:
        update_source(config=input_config, config_type="input")

    for output_config in patched_configs.output_configs:
        update_source(config=output_config, config_type="output")

    patched_configs.input_configs = _inject_correct_sequence_input_from_linked_output(
        input_configs=patched_configs.input_configs,
        output_configs=patched_configs.output_configs,
    )

    patched_configs.global_config.basic_experiment.n_epochs = 1

    return configs, patched_configs


def _inject_correct_sequence_input_from_linked_output(
    input_configs: Sequence[InputConfig],
    output_configs: Sequence[OutputConfig],
) -> Sequence[InputConfig]:
    """
    This is needed as EIR earlier creates a matching input sequence configuration
    for sequence outputs. While the source is correctly set at that point, we
    override that when patching the local data paths, so we need to re-inject the
    correct source for the input sequence
    """
    new_input_configs = deepcopy(input_configs)

    for output_config in output_configs:
        if output_config.output_info.output_type == "sequence":
            output_name = output_config.output_info.output_name
            matching_input = next(
                (
                    input_config
                    for input_config in new_input_configs
                    if input_config.input_info.input_name == output_name
                ),
                None,
            )
            assert (
                matching_input is not None
            ), f"No matching input found for {output_name}"

            matching_input.input_info.input_source = (
                output_config.output_info.output_source
            )

    return new_input_configs


def setup_and_gather_streaming_data(
    output_folder: str,
    configs: Configs,
    batch_size: int,
    max_samples: int,
) -> Optional[tuple[str, Path]]:
    try:
        websocket_url = validate_streaming_setup(configs=configs)
        if websocket_url:
            return gather_streaming_data_for_setup(
                websocket_url=websocket_url,
                output_folder=output_folder,
                configs=configs,
                batch_size=batch_size,
                max_samples=max_samples,
            )
        else:
            logger.info(
                "No streaming setup detected. Proceeding with standard data loading."
            )
            return None
    except ValueError as e:
        logger.error(f"Invalid streaming setup: {str(e)}")
        return None
