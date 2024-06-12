import atexit
import json
import subprocess
import threading
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import requests

from docs.doc_modules.experiments import save_command_as_text
from docs.doc_modules.serve_experiments_utils import send_request
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


@dataclass
class AutoDocServingInfo:
    name: str
    base_path: Path
    server_command: list[str]
    example_requests: list[dict[str, Any]]
    pre_run_command_modifications: Sequence[Callable[[List[str]], List[str]]] = ()
    post_run_functions: Sequence[Tuple[Callable, Dict]] = ()
    data_loading_function: Callable = None
    url: str = "http://localhost:8000/predict"


def make_serving_tutorial_data(
    auto_doc_experiment_info: AutoDocServingInfo,
) -> None:
    ade = auto_doc_experiment_info

    command = ade.server_command
    for command_modification in ade.pre_run_command_modifications:
        command = command_modification(command)

    predictions = run_serve_experiment_from_command(
        command=command,
        url=ade.url,
        example_requests=ade.example_requests,
        data_loading_function=ade.data_loading_function,
    )

    save_predictions_as_json(
        predictions=predictions,
        output_dir=ade.base_path / "serve_results",
    )

    save_command_as_text(
        command=command,
        output_path=(ade.base_path / "commands" / ade.name).with_suffix(".txt"),
    )

    for func, kwargs in ade.post_run_functions:
        func(**kwargs)


def run_serve_experiment_from_command(
    command: List[str],
    url: str,
    example_requests: list[list[dict[str, Any]]],
    data_loading_function: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    _run_server, process_storage = server_run_factory()

    logger.info(f"Running server with command: {command}")
    server_thread = threading.Thread(target=_run_server, args=(command,))
    server_thread.start()

    timeout = 120
    start_time = time.time()
    info_url = url.replace("/predict", "/info")
    while time.time() - start_time < timeout:
        if is_server_running(url=info_url):
            logger.info("Server is up and running.")
            break
        time.sleep(5)

    if not is_server_running(url=info_url):
        raise RuntimeError("Server failed to start within the allotted time.")

    responses = []
    for request in example_requests:
        if data_loading_function:

            loaded_items = []
            for sample in request:
                loaded_items.append(data_loading_function(sample))

            response = send_request(
                url=url,
                payload=loaded_items,
            )

            cur_info = {"request": request, "response": response}

            responses.append(cur_info)

    bound_terminate = partial(
        _terminate_process,
        process=process_storage["process"],
    )
    atexit.register(bound_terminate)
    process_storage["process"].terminate()
    return responses


def is_server_running(url: str) -> bool:
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.ConnectionError:
        return False


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()


def server_run_factory() -> (
    tuple[Callable[[list[str]], None], dict[str, subprocess.Popen]]
):
    cached = {}

    def _run_server(command: list[str]) -> None:
        process = subprocess.Popen(args=command)
        cached["process"] = process

    return _run_server, cached


def save_predictions_as_json(
    predictions: list,
    output_dir: Path,
    file_name: str = "predictions.json",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file_path = output_dir / file_name

    with open(output_file_path, "w", encoding="utf-8") as file:
        json.dump(predictions, file, ensure_ascii=False, indent=4)

    logger.info(f"Predictions saved to {output_file_path}")
