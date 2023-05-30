from typing import (
    Dict,
    Union,
    Sequence,
    Callable,
    Type,
    TYPE_CHECKING,
    Any,
)

from aislib.misc_utils import get_logger

from eir.experiment_io.experiment_io import (
    get_run_folder_from_model_path,
)
from eir.setup import schemas
from eir.setup.input_setup_modules.setup_array import (
    set_up_array_input,
    ComputedArrayInputInfo,
)
from eir.setup.input_setup_modules.setup_bytes import (
    set_up_bytes_input_for_training,
    ComputedBytesInputInfo,
)
from eir.setup.input_setup_modules.setup_image import (
    ComputedImageInputInfo,
    set_up_image_input_for_training,
)
from eir.setup.input_setup_modules.setup_omics import (
    set_up_omics_input,
    ComputedOmicsInputInfo,
)
from eir.setup.input_setup_modules.setup_pretrained import (
    get_input_setup_from_pretrained_function_map,
)
from eir.setup.input_setup_modules.setup_sequence import (
    ComputedSequenceInputInfo,
    set_up_sequence_input_for_training,
)
from eir.setup.input_setup_modules.setup_tabular import (
    set_up_tabular_input_for_training,
    ComputedTabularInputInfo,
)

if TYPE_CHECKING:
    from eir.train_utils.step_logic import Hooks

logger = get_logger(__name__)

al_input_objects = Union[
    ComputedOmicsInputInfo,
    ComputedTabularInputInfo,
    ComputedSequenceInputInfo,
    ComputedBytesInputInfo,
    ComputedImageInputInfo,
    ComputedArrayInputInfo,
]
al_input_objects_as_dict = Dict[str, al_input_objects]

al_serializable_input_objects = Union[
    ComputedSequenceInputInfo,
    ComputedImageInputInfo,
    ComputedBytesInputInfo,
]

al_serializable_input_classes = Union[
    Type[ComputedSequenceInputInfo],
    Type[ComputedImageInputInfo],
    Type[ComputedBytesInputInfo],
]


def set_up_inputs_general(
    inputs_configs: schemas.al_input_configs,
    hooks: Union["Hooks", None],
    setup_func_getter: Callable[[schemas.InputConfig], Callable[..., al_input_objects]],
    setup_func_kwargs: Dict[str, Any],
) -> al_input_objects_as_dict:
    all_inputs = {}

    name_config_iter = get_input_name_config_iterator(input_configs=inputs_configs)

    for name, input_config in name_config_iter:
        setup_func = setup_func_getter(input_config=input_config)

        cur_input_data_config = input_config.input_info
        logger.info(
            "Setting up %s inputs '%s' from %s.",
            cur_input_data_config.input_type,
            cur_input_data_config.input_name,
            cur_input_data_config.input_source,
        )

        set_up_input = setup_func(
            input_config=input_config, hooks=hooks, **setup_func_kwargs
        )
        all_inputs[name] = set_up_input

    return all_inputs


def set_up_inputs_for_training(
    inputs_configs: schemas.al_input_configs,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    hooks: Union["Hooks", None],
) -> al_input_objects_as_dict:
    train_input_setup_kwargs = {
        "train_ids": train_ids,
        "valid_ids": valid_ids,
    }
    all_inputs = set_up_inputs_general(
        inputs_configs=inputs_configs,
        hooks=hooks,
        setup_func_getter=get_input_setup_function_for_train,
        setup_func_kwargs=train_input_setup_kwargs,
    )

    return all_inputs


def get_input_name_config_iterator(input_configs: schemas.al_input_configs):
    for input_config in input_configs:
        cur_input_data_config = input_config.input_info
        cur_name = cur_input_data_config.input_name
        yield cur_name, input_config


def get_input_setup_function_for_train(
    input_config: schemas.InputConfig,
) -> Callable[..., al_input_objects]:
    input_type = input_config.input_info.input_type
    pretrained_config = input_config.pretrained_config

    from_scratch_mapping = get_input_setup_function_map()

    if pretrained_config:
        pretrained_run_folder = get_run_folder_from_model_path(
            model_path=pretrained_config.model_path
        )
        from_pretrained_mapping = get_input_setup_from_pretrained_function_map(
            run_folder=pretrained_run_folder,
            load_module_name=pretrained_config.load_module_name,
        )
        return from_pretrained_mapping[input_type]

    return from_scratch_mapping[input_type]


def get_input_setup_function_map() -> Dict[str, Callable[..., al_input_objects]]:
    setup_mapping = {
        "omics": set_up_omics_input,
        "tabular": set_up_tabular_input_for_training,
        "sequence": set_up_sequence_input_for_training,
        "bytes": set_up_bytes_input_for_training,
        "image": set_up_image_input_for_training,
        "array": set_up_array_input,
    }

    return setup_mapping
