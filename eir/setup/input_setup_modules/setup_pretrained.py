from functools import partial
from pathlib import Path
from typing import Callable, Dict

from eir.experiment_io.input_object_io import load_serialized_input_object
from eir.setup.input_setup_modules.setup_array import set_up_array_input_object
from eir.setup.input_setup_modules.setup_bytes import ComputedBytesInputInfo
from eir.setup.input_setup_modules.setup_image import ComputedImageInputInfo
from eir.setup.input_setup_modules.setup_omics import ComputedOmicsInputInfo
from eir.setup.input_setup_modules.setup_sequence import ComputedSequenceInputInfo
from eir.setup.input_setup_modules.setup_tabular import (
    set_up_tabular_input_from_pretrained,
)


def get_input_setup_from_pretrained_function_map(
    run_folder: Path, load_module_name: str
) -> Dict[str, Callable]:
    pretrained_setup_mapping: Dict[str, Callable] = {
        "omics": partial(
            load_serialized_input_object,
            input_class=ComputedOmicsInputInfo,
            run_folder=run_folder,
            custom_input_name=load_module_name,
        ),
        "tabular": partial(
            set_up_tabular_input_from_pretrained,
            custom_input_name=load_module_name,
        ),
        "sequence": partial(
            load_serialized_input_object,
            input_class=ComputedSequenceInputInfo,
            run_folder=run_folder,
            custom_input_name=load_module_name,
        ),
        "bytes": partial(
            load_serialized_input_object,
            input_class=ComputedBytesInputInfo,
            run_folder=run_folder,
            custom_input_name=load_module_name,
        ),
        "image": partial(
            load_serialized_input_object,
            input_class=ComputedImageInputInfo,
            run_folder=run_folder,
            custom_input_name=load_module_name,
        ),
        "array": set_up_array_input_object,
    }

    return pretrained_setup_mapping
