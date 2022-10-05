from dataclasses import dataclass
from typing import Dict, Union, Callable, Any

from aislib.misc_utils import get_logger
from sklearn.preprocessing import StandardScaler

from eir.data_load.label_setup import (
    al_label_transformers,
    al_target_columns,
    merge_target_columns,
)
from eir.setup import schemas

al_num_outputs_per_target = Dict[str, int]

logger = get_logger(name=__name__)

al_output_objects = Union["TabularOutputInfo", Any]
al_output_objects_as_dict = Dict[str, al_output_objects]


@dataclass
class TabularOutputInfo:
    output_config: schemas.OutputConfig
    num_outputs_per_target: al_num_outputs_per_target
    target_columns: al_target_columns
    target_transformers: al_label_transformers


def set_up_outputs_for_training(
    output_configs: schemas.al_output_configs,
    target_transformers: Dict[str, al_label_transformers],
) -> al_output_objects_as_dict:

    all_inputs = set_up_outputs_general(
        output_configs=output_configs,
        setup_func_getter=get_output_setup_function_for_train,
        setup_func_kwargs={"target_transformers": target_transformers},
    )

    return all_inputs


def set_up_outputs_general(
    output_configs: schemas.al_output_configs,
    setup_func_getter: Callable[
        [Union[schemas.OutputConfig, Any]], Callable[..., al_output_objects]
    ],
    setup_func_kwargs: Dict[str, Any],
) -> al_output_objects_as_dict:
    all_inputs = {}

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
        all_inputs[name] = set_up_output

    return all_inputs


def get_output_setup_function_for_train(
    output_config: schemas.OutputConfig,
) -> Callable[..., al_output_objects]:

    output_type = output_config.output_info.output_type

    mapping = get_output_setup_function_map()

    return mapping[output_type]


def get_output_setup_function_map() -> Dict[str, Callable[..., al_output_objects]]:
    setup_mapping = {
        "tabular": set_up_tabular_output,
    }

    return setup_mapping


def set_up_tabular_output(
    output_config: schemas.OutputConfig,
    target_transformers: Dict[str, al_label_transformers],
    *args,
    **kwargs,
) -> TabularOutputInfo:
    cur_target_transformers = target_transformers[output_config.output_info.output_name]
    num_outputs_per_target = set_up_num_outputs_per_target(
        target_transformers=cur_target_transformers
    )

    target_columns = merge_target_columns(
        target_con_columns=list(output_config.output_type_info.target_con_columns),
        target_cat_columns=list(output_config.output_type_info.target_cat_columns),
    )

    tabular_output_info = TabularOutputInfo(
        output_config=output_config,
        num_outputs_per_target=num_outputs_per_target,
        target_columns=target_columns,
        target_transformers=cur_target_transformers,
    )

    return tabular_output_info


def set_up_num_outputs_per_target(
    target_transformers: al_label_transformers,
) -> al_num_outputs_per_target:

    num_outputs_per_target_dict = {}
    for target_column, transformer in target_transformers.items():
        if isinstance(transformer, StandardScaler):
            num_outputs = 1
        else:
            num_outputs = len(transformer.classes_)

            if num_outputs < 2:
                logger.warning(
                    f"Only {num_outputs} unique values found in categorical label "
                    f"column {target_column} (returned by {transformer}). This means "
                    f"that most likely an error will be raised if e.g. using "
                    f"nn.CrossEntropyLoss as it expects an output dimension of >=2."
                )

        num_outputs_per_target_dict[target_column] = num_outputs

    return num_outputs_per_target_dict


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
