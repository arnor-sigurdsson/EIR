from dataclasses import dataclass
from typing import Dict

from sklearn.preprocessing import StandardScaler

from eir.data_load.label_setup import (
    al_label_transformers,
    al_target_columns,
    merge_target_columns,
)
from eir.setup import schemas
from eir.setup.schema_modules.output_schemas_tabular import TabularOutputTypeConfig
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)

al_num_outputs_per_target = Dict[str, int]


@dataclass
class ComputedTabularOutputInfo:
    output_config: schemas.OutputConfig
    num_outputs_per_target: al_num_outputs_per_target
    target_columns: al_target_columns
    target_transformers: al_label_transformers


def set_up_tabular_output(
    output_config: schemas.OutputConfig,
    target_transformers: Dict[str, al_label_transformers],
    *args,
    **kwargs,
) -> ComputedTabularOutputInfo:
    cur_target_transformers = target_transformers[output_config.output_info.output_name]
    num_outputs_per_target = set_up_num_outputs_per_target(
        target_transformers=cur_target_transformers
    )

    output_type_info = output_config.output_type_info
    assert isinstance(output_type_info, TabularOutputTypeConfig)

    target_columns = merge_target_columns(
        target_con_columns=list(output_type_info.target_con_columns),
        target_cat_columns=list(output_type_info.target_cat_columns),
    )

    tabular_output_info = ComputedTabularOutputInfo(
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
