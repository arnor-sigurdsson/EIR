from functools import partial
from typing import Callable, Dict, Sequence, Union

from eir.experiment_io.experiment_io import load_serialized_input_object
from eir.predict_modules.predict_tabular_input_setup import (
    setup_tabular_input_for_testing,
)
from eir.setup import input_setup, schemas
from eir.setup.input_setup import al_input_objects_as_dict
from eir.setup.input_setup_modules import setup_array, setup_omics
from eir.setup.input_setup_modules.setup_bytes import ComputedBytesInputInfo
from eir.setup.input_setup_modules.setup_image import ComputedImageInputInfo
from eir.setup.input_setup_modules.setup_sequence import ComputedSequenceInputInfo
from eir.train_utils.step_logic import Hooks


def set_up_inputs_for_predict(
    test_inputs_configs: schemas.al_input_configs,
    ids: Sequence[str],
    hooks: Union["Hooks", None],
    output_folder: str,
) -> al_input_objects_as_dict:
    train_input_setup_kwargs = {
        "ids": ids,
        "output_folder": output_folder,
    }
    all_inputs = input_setup.set_up_inputs_general(
        inputs_configs=test_inputs_configs,
        hooks=hooks,
        setup_func_getter=get_input_setup_function_for_predict,
        setup_func_kwargs=train_input_setup_kwargs,
    )

    return all_inputs


def get_input_setup_function_for_predict(
    input_config: schemas.InputConfig,
) -> Callable[..., input_setup.al_input_objects]:
    mapping = get_input_setup_function_map_for_predict()
    input_type = input_config.input_info.input_type

    return mapping[input_type]


def get_input_setup_function_map_for_predict() -> (
    Dict[str, Callable[..., input_setup.al_input_objects]]
):
    setup_mapping: Dict[str, Callable[..., input_setup.al_input_objects]] = {
        "omics": setup_omics.set_up_omics_input,
        "tabular": setup_tabular_input_for_testing,
        "sequence": partial(
            load_serialized_input_object,
            input_class=ComputedSequenceInputInfo,
        ),
        "bytes": partial(
            load_serialized_input_object,
            input_class=ComputedBytesInputInfo,
        ),
        "image": partial(
            load_serialized_input_object,
            input_class=ComputedImageInputInfo,
        ),
        "array": partial(
            load_serialized_input_object,
            input_class=setup_array.ComputedArrayInputInfo,
        ),
    }

    return setup_mapping
