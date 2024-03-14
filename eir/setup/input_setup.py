from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
    Protocol,
    Sequence,
    Type,
    Union,
)

from eir.experiment_io.experiment_io import get_run_folder_from_model_path
from eir.predict_modules.predict_tabular_input_setup import (
    ComputedPredictTabularInputInfo,
)
from eir.serve_modules.serve_schemas import ComputedServeTabularInputInfo
from eir.setup import schemas
from eir.setup.input_setup_modules.setup_array import (
    ComputedArrayInputInfo,
    set_up_array_input,
)
from eir.setup.input_setup_modules.setup_bytes import (
    ComputedBytesInputInfo,
    set_up_bytes_input_for_training,
)
from eir.setup.input_setup_modules.setup_image import (
    ComputedImageInputInfo,
    set_up_image_input_for_training,
)
from eir.setup.input_setup_modules.setup_omics import (
    ComputedOmicsInputInfo,
    set_up_omics_input,
)
from eir.setup.input_setup_modules.setup_pretrained import (
    get_input_setup_from_pretrained_function_map,
)
from eir.setup.input_setup_modules.setup_sequence import (
    ComputedSequenceInputInfo,
    set_up_sequence_input_for_training,
)
from eir.setup.input_setup_modules.setup_tabular import (
    ComputedTabularInputInfo,
    set_up_tabular_input_for_training,
)
from eir.utils.logging import get_logger

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
    ComputedPredictTabularInputInfo,
    ComputedServeTabularInputInfo,
]
al_input_objects_as_dict = Dict[str, al_input_objects]

al_serializable_input_objects = Union[
    ComputedSequenceInputInfo,
    ComputedImageInputInfo,
    ComputedBytesInputInfo,
    ComputedArrayInputInfo,
]

al_serializable_input_classes = Union[
    Type[ComputedSequenceInputInfo],
    Type[ComputedImageInputInfo],
    Type[ComputedBytesInputInfo],
    Type[ComputedArrayInputInfo],
]


class InputSetupFunction(Protocol):
    def __call__(
        self,
        input_config: schemas.InputConfig,
        hooks: Optional["Hooks"],
        **kwargs,
    ) -> al_input_objects: ...


class InputSetupGetterFunction(Protocol):
    def __call__(self, input_config: schemas.InputConfig) -> InputSetupFunction: ...


def set_up_inputs_general(
    inputs_configs: schemas.al_input_configs,
    hooks: Optional["Hooks"],
    setup_func_getter: InputSetupGetterFunction,
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


def get_input_setup_function_map() -> dict[str, Callable[..., al_input_objects]]:
    setup_mapping: dict[str, Callable[..., al_input_objects]] = {
        "omics": set_up_omics_input,
        "tabular": set_up_tabular_input_for_training,
        "sequence": set_up_sequence_input_for_training,
        "bytes": set_up_bytes_input_for_training,
        "image": set_up_image_input_for_training,
        "array": set_up_array_input,
    }

    return setup_mapping
