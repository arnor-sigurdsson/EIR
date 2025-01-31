import atexit
import inspect
import json
import subprocess
import textwrap
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import requests

from docs.doc_modules.experiments import save_command_as_text
from docs.doc_modules.serve_experiments_utils import send_request
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


@dataclass
class RequestExampleModule:
    function: Callable
    function_body: str
    name: str
    language: str = "python"


@dataclass
class AutoDocServingInfo:
    name: str
    base_path: Path
    server_command: list[str]
    example_requests: list[list[dict[str, Any]]]
    pre_run_command_modifications: Sequence[Callable[[list[str]], list[str]]] = ()
    post_run_functions: Sequence[tuple[Callable, dict]] = ()
    data_loading_function: Callable = None
    url: str = "http://localhost:8000/predict"
    request_example_modules: Sequence[RequestExampleModule] = ()


def build_request_example_module_from_function(
    function: Callable,
    name: str,
    language: str,
    custom_body: str | None = None,
) -> RequestExampleModule:
    if language == "python":
        function_body = extract_function_body(func=function)
        process = subprocess.run(
            ["ruff", "format", "-"],
            input=function_body,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            logger.warning(f"Ruff formatting failed: {process.stderr}")
            function_body = function_body
        else:
            function_body = process.stdout
    elif language == "shell":
        assert custom_body is not None
        function_body = custom_body
    else:
        raise ValueError(f"Unsupported language: {language}")

    return RequestExampleModule(
        function=function,
        function_body=function_body,
        name=name,
        language=language,
    )


def extract_function_body(func):
    source_lines = inspect.getsourcelines(func)[0]
    func_def_index = 0
    for i, line in enumerate(source_lines):
        if line.strip().startswith("def "):
            func_def_index = i
            break
    start = func_def_index + 1
    function_body_lines = source_lines[start:]

    function_body_lines = textwrap.dedent("".join(function_body_lines)).split("\n")

    parsed_lines = []
    for line in function_body_lines:
        if "# --skip-after" in line:
            break
        parsed_lines.append(line)

    return "\n".join(parsed_lines)


def test_and_save_request_example_module(
    request_example_module: RequestExampleModule,
    output_dir: Path,
) -> None:
    rem = request_example_module

    output_dir.mkdir(parents=True, exist_ok=True)

    module_output = rem.function()
    output_file_path = output_dir / f"{rem.name}_request_example.json"
    with open(output_file_path, "w", encoding="utf-8") as file:
        json.dump(module_output, file, ensure_ascii=False, indent=4)

    if rem.language == "python":
        suffix = ".py"
    elif rem.language == "shell":
        suffix = ".sh"
    else:
        raise ValueError(f"Unsupported language: {rem.language}")

    output_file_path = output_dir / f"{rem.name}_request_example_module{suffix}"
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(rem.function_body)

    logger.info(f"Request example module saved to {output_file_path}")


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
        base_path=ade.base_path,
        request_example_modules=ade.request_example_modules,
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
    command: list[str],
    url: str,
    example_requests: list[list[dict[str, Any]]],
    data_loading_function: Callable[[dict[str, Any]], dict[str, Any]],
    base_path: Path | None = None,
    request_example_modules: Sequence[RequestExampleModule] | None = None,
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

    bound_terminate = partial(
        _terminate_process,
        process=process_storage["process"],
    )
    atexit.register(bound_terminate)

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

    if request_example_modules:
        assert base_path is not None
        for request_example_module in request_example_modules:
            test_and_save_request_example_module(
                request_example_module=request_example_module,
                output_dir=base_path / "request_example",
            )

    process_storage["process"].terminate()
    process_storage["process"].wait()

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
        process.wait()


def server_run_factory() -> tuple[
    Callable[[list[str]], None], dict[str, subprocess.Popen]
]:
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
