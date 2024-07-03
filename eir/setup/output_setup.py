from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Protocol

from eir.data_load.label_setup import al_label_transformers
from eir.setup import schemas
from eir.setup.output_setup_modules.array_output_setup import (
    ComputedArrayOutputInfo,
    set_up_array_output,
)
from eir.setup.output_setup_modules.image_output_setup import (
    ComputedImageOutputInfo,
    set_up_image_output,
)
from eir.setup.output_setup_modules.sequence_output_setup import (
    ComputedSequenceOutputInfo,
    set_up_sequence_output,
)
from eir.setup.output_setup_modules.tabular_output_setup import (
    ComputedTabularOutputInfo,
    set_up_tabular_output,
)
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.setup.input_setup import al_input_objects_as_dict


logger = get_logger(name=__name__)

al_output_objects = (
    ComputedTabularOutputInfo
    | ComputedSequenceOutputInfo
    | ComputedArrayOutputInfo
    | ComputedImageOutputInfo
)
al_output_objects_as_dict = Dict[str, al_output_objects]


def set_up_outputs_for_training(
    output_configs: schemas.al_output_configs,
    input_objects: Optional["al_input_objects_as_dict"] = None,
    target_transformers: Optional[Dict[str, al_label_transformers]] = None,
) -> al_output_objects_as_dict:
    all_outputs = set_up_outputs_general(
        output_configs=output_configs,
        setup_func_getter=get_output_setup_function_for_train,
        setup_func_kwargs={
            "input_objects": input_objects,
            "target_transformers": target_transformers,
        },
    )

    return all_outputs


class OutputSetupFunction(Protocol):
    def __call__(
        self,
        input_config: schemas.OutputConfig,
        **kwargs,
    ) -> al_output_objects: ...


class OutputSetupGetterFunction(Protocol):
    def __call__(self, output_config: schemas.OutputConfig) -> OutputSetupFunction: ...


def set_up_outputs_general(
    output_configs: schemas.al_output_configs,
    setup_func_getter: OutputSetupGetterFunction,
    setup_func_kwargs: Dict[str, Any],
) -> al_output_objects_as_dict:
    all_outputs = {}

    name_config_iter = get_output_name_config_iterator(output_configs=output_configs)

    for name, output_config in name_config_iter:
        setup_func = setup_func_getter(output_config=output_config)

        cur_output_data_config = output_config.output_info
        logger.info(
            "Setting up %s outputs '%s' from %s.",
            cur_output_data_config.output_name,
            cur_output_data_config.output_type,
            cur_output_data_config.output_source,
        )

        set_up_output = setup_func(output_config=output_config, **setup_func_kwargs)
        all_outputs[name] = set_up_output

    return all_outputs


def get_output_setup_function_for_train(
    output_config: schemas.OutputConfig,
) -> Callable[..., al_output_objects]:
    output_type = output_config.output_info.output_type

    mapping = get_output_setup_function_map()

    return mapping[output_type]


def get_output_setup_function_map() -> dict[str, Callable[..., al_output_objects]]:
    setup_mapping: dict[str, Callable[..., al_output_objects]] = {
        "tabular": set_up_tabular_output,
        "sequence": set_up_sequence_output,
        "array": set_up_array_output,
        "image": set_up_image_output,
    }

    return setup_mapping


def get_output_name_config_iterator(output_configs: schemas.al_output_configs):
    """
    We do not allow '.' as it is used in the weighted sampling setup.
    """

    for output_config in output_configs:
        cur_input_data_config = output_config.output_info
        cur_name = cur_input_data_config.output_name

        if "." in cur_name:
            raise ValueError(
                "Having '.' in the output name is currently not supported. Got '%s'."
                "Kindly rename '%s' to not include any '.' symbols.",
                cur_name,
                cur_name,
            )

        yield cur_name, output_config
