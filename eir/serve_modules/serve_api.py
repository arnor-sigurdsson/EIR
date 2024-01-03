from typing import Any, Sequence

from aislib.misc_utils import get_logger
from fastapi import FastAPI

from eir.predict_modules.predict_tabular_input_setup import (
    ComputedPredictTabularInputInfo,
)
from eir.serve_modules.serve_api_models import ResponseModel, create_input_model
from eir.serve_modules.serve_experiment_io import ServeExperiment
from eir.serve_modules.serve_input_setup import general_pre_process
from eir.serve_modules.serve_post_process import general_post_process
from eir.serve_modules.serve_prediction import run_serve_prediction
from eir.serve_modules.serve_schemas import ComputedServeTabularInputInfo
from eir.setup.input_setup import al_input_objects_as_dict
from eir.setup.input_setup_modules.setup_array import ComputedArrayInputInfo
from eir.setup.input_setup_modules.setup_bytes import ComputedBytesInputInfo
from eir.setup.input_setup_modules.setup_image import ComputedImageInputInfo
from eir.setup.input_setup_modules.setup_omics import ComputedOmicsInputInfo
from eir.setup.input_setup_modules.setup_sequence import ComputedSequenceInputInfo
from eir.setup.input_setup_modules.setup_tabular import ComputedTabularInputInfo
from eir.setup.output_setup import al_output_objects_as_dict
from eir.setup.output_setup_modules.array_output_setup import ComputedArrayOutputInfo
from eir.setup.output_setup_modules.sequence_output_setup import (
    ComputedSequenceOutputInfo,
)
from eir.setup.output_setup_modules.tabular_output_setup import (
    ComputedTabularOutputInfo,
)
from eir.setup.schemas import InputConfig

logger = get_logger(name=__name__, tqdm_compatible=True)


async def process_request(data: Sequence, serve_experiment: ServeExperiment) -> Any:
    batch = general_pre_process(
        data=data,
        serve_experiment=serve_experiment,
    )

    prediction = run_serve_prediction(
        serve_experiment=serve_experiment,
        batch=batch,
    )

    response = general_post_process(
        outputs=prediction,
        input_objects=serve_experiment.inputs,
        output_objects=serve_experiment.outputs,
    )
    return response


def create_predict_endpoint(
    app: FastAPI,
    configs: Sequence[InputConfig],
    serve_experiment: ServeExperiment,
) -> None:
    input_model = create_input_model(configs=configs)

    @app.post("/predict", response_model=ResponseModel)
    async def predict(request: input_model) -> ResponseModel:  # type: ignore
        data = [request.model_dump()]  # type: ignore
        response_data = await process_request(
            data=data,
            serve_experiment=serve_experiment,
        )

        return ResponseModel(result=response_data[0])


def create_info_endpoint(app: FastAPI, serve_experiment: ServeExperiment) -> None:
    model_info: dict[str, Any] = get_model_info(
        input_objects=serve_experiment.inputs,
        output_objects=serve_experiment.outputs,
    )

    @app.get("/info")
    def info() -> dict[str, Any]:
        return model_info


def get_model_info(
    input_objects: al_input_objects_as_dict,
    output_objects: al_output_objects_as_dict,
) -> dict[str, Any]:
    model_info: dict[str, Any] = {
        "inputs": {},
        "outputs": {},
    }

    for name, input_object in input_objects.items():
        match input_object:
            case ComputedOmicsInputInfo():
                shape = input_object.data_dimensions.full_shape()[1:]
                model_info["inputs"][name] = {
                    "type": "omics",
                    "shape": shape,
                }

            case ComputedSequenceInputInfo() | ComputedBytesInputInfo():
                pass

            case ComputedImageInputInfo():
                pass

            case (
                ComputedTabularInputInfo()
                | ComputedPredictTabularInputInfo()
                | ComputedServeTabularInputInfo()
            ):
                pass

            case ComputedArrayInputInfo():
                shape = input_object.data_dimensions.full_shape()
                model_info["inputs"][name] = {
                    "type": "array",
                    "shape": shape,
                    "dtype": input_object.dtype.str,
                }

            case _:
                input_type = input_object.input_config.input_info.input_type
                raise ValueError(f"Unknown input type: {input_type}")

    for name, output_object in output_objects.items():
        match output_object:
            case ComputedTabularOutputInfo():
                pass

            case ComputedSequenceOutputInfo():
                pass

            case ComputedArrayOutputInfo():
                shape = output_object.data_dimensions.full_shape()
                model_info["outputs"][name] = {
                    "type": "array",
                    "shape": shape,
                    "dtype": output_object.dtype.str,
                }

            case _:
                output_type = output_object.output_config.output_info.output_type
                raise ValueError(f"Unknown output type: {output_type}")

    return model_info
