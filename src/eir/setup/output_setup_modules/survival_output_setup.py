from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

from eir.data_load.label_setup import (
    al_label_transformers,
    al_target_columns,
    merge_target_columns,
)
from eir.setup import schemas
from eir.setup.schema_modules.output_schemas_survival import SurvivalOutputTypeConfig
from eir.target_setup.target_setup_utils import IdentityTransformer
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)

al_num_outputs_per_target = dict[str, int]


@dataclass
class ComputedSurvivalOutputInfo:
    output_config: schemas.OutputConfig
    num_outputs_per_target: al_num_outputs_per_target
    target_columns: al_target_columns
    target_transformers: al_label_transformers

    baseline_hazard: np.ndarray | None = None
    baseline_unique_times: np.ndarray | None = None


def set_up_survival_output(
    output_config: schemas.OutputConfig,
    target_transformers: dict[str, al_label_transformers],
    baseline_hazard: np.ndarray | None = None,
    baseline_unique_times: np.ndarray | None = None,
    *args,
    **kwargs,
) -> ComputedSurvivalOutputInfo:
    output_name = output_config.output_info.output_name
    cur_target_transformers = target_transformers[output_name]
    num_outputs_per_target = set_up_num_survival_outputs(
        target_transformers=cur_target_transformers
    )

    output_type_info = output_config.output_type_info
    assert isinstance(output_type_info, SurvivalOutputTypeConfig)

    target_columns = merge_target_columns(
        target_con_columns=[output_type_info.time_column],
        target_cat_columns=[output_type_info.event_column],
    )

    tabular_output_info = ComputedSurvivalOutputInfo(
        output_config=output_config,
        num_outputs_per_target=num_outputs_per_target,
        target_columns=target_columns,
        target_transformers=cur_target_transformers,
        baseline_hazard=baseline_hazard,
        baseline_unique_times=baseline_unique_times,
    )

    return tabular_output_info


def set_up_num_survival_outputs(
    target_transformers: al_label_transformers,
) -> al_num_outputs_per_target:
    num_outputs_per_target_dict = {}
    assert len(target_transformers) == 2
    for column, transformer in target_transformers.items():
        if not isinstance(transformer, KBinsDiscretizer | IdentityTransformer):
            continue

        time_column = column
        event_column = next(i for i in target_transformers if i != time_column)

        match transformer:
            case IdentityTransformer():
                num_outputs_per_target_dict[event_column] = 1
            case KBinsDiscretizer(n_bins=transformer.n_bins):
                num_outputs_per_target_dict[event_column] = transformer.n_bins

    return num_outputs_per_target_dict
