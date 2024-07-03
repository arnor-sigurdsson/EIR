from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Sequence, Union

import torch
from torch.utils.data._utils.collate import default_collate

from eir.data_load.data_preparation_modules.imputation import (
    impute_missing_modalities_wrapper,
)
from eir.data_load.data_preparation_modules.input_preparation_wrappers import (
    prepare_inputs_memory,
)
from eir.experiment_io.experiment_io import load_transformers
from eir.predict_modules.predict_input_setup import (
    get_input_setup_function_map_for_predict,
)
from eir.predict_modules.predict_tabular_input_setup import (
    ComputedPredictTabularInputInfo,
)
from eir.serve_modules.serve_network_utils import prepare_request_input_data_wrapper
from eir.serve_modules.serve_schemas import ComputedServeTabularInputInfo, ServeLabels
from eir.setup.input_setup import (
    al_input_objects_as_dict,
    get_input_name_config_iterator,
)
from eir.setup.input_setup_modules.setup_sequence import ComputedSequenceInputInfo
from eir.setup.input_setup_modules.setup_tabular import ComputedTabularInputInfo
from eir.setup.schemas import InputConfig, TabularInputDataConfig, al_input_configs
from eir.train_utils.utils import call_hooks_stage_iterable
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.serve_modules.serve_experiment_io import ServeExperiment
    from eir.train import Experiment, Hooks

logger = get_logger(name=__name__)


@dataclass(frozen=True)
class ServeBatch:
    """
    Note, we keep the pre_hook_inputs around for cases like autoregressive
    generation where we use the actual input token ids. This is simpler for e.g.
    supervised learning where can use the final processed inputs (e.g. embeddings)
    directly.
    """

    pre_hook_inputs: Dict[str, torch.Tensor]
    inputs: Dict[str, torch.Tensor]
    inputs_split: Sequence[dict[str, torch.Tensor]]
    target_labels: Dict[str, Dict[str, torch.Tensor]]
    ids: list[str]


def set_up_inputs_for_serve(
    test_inputs_configs: al_input_configs,
    hooks: Union["Hooks", None],
    output_folder: str,
) -> al_input_objects_as_dict:
    all_inputs = {}

    name_config_iter = get_input_name_config_iterator(input_configs=test_inputs_configs)
    for name, input_config in name_config_iter:
        cur_input_data_config = input_config.input_info
        setup_func = get_input_setup_function_for_serve(
            input_type=cur_input_data_config.input_type
        )
        logger.info(
            "Setting up %s inputs '%s' from %s.",
            cur_input_data_config.input_type,
            cur_input_data_config.input_name,
            cur_input_data_config.input_source,
        )
        set_up_input = setup_func(
            input_config=input_config,
            ids=(),
            output_folder=output_folder,
            hooks=hooks,
        )
        all_inputs[name] = set_up_input

    return all_inputs


def general_pre_process(
    data: Sequence, serve_experiment: "ServeExperiment"
) -> ServeBatch:
    exp = serve_experiment

    inputs_parsed = parse_request_input_data_wrapper(
        data=data,
        input_objects=deepcopy(exp.inputs),
    )

    inputs_prepared = general_pre_process_raw_inputs_wrapper(
        raw_inputs=inputs_parsed,
        experiment=exp,
    )

    inputs_final = default_collate(inputs_prepared)
    inputs_final = inputs_final

    loader_batch = (inputs_final, None, None)

    batch_prep_hook_kwargs = {"experiment": exp}
    hooks = exp.hooks
    assert hooks is not None
    state = call_hooks_stage_iterable(
        hook_iterable=hooks.step_func_hooks.base_prepare_batch,
        common_kwargs={"loader_batch": loader_batch, **batch_prep_hook_kwargs},
        state=None,
    )
    batch = state["batch"]

    batch_final = ServeBatch(
        pre_hook_inputs=inputs_final,
        inputs=batch.inputs,
        inputs_split=inputs_prepared,
        target_labels={},
        ids=[f"Serve_{i}" for i in range(len(data))],
    )

    return batch_final


def parse_request_input_data_wrapper(
    data: Sequence, input_objects: al_input_objects_as_dict
) -> Sequence[dict[str, Any]]:
    loaded_data = _load_request_data(data=data)
    parsed_data = prepare_request_input_data_wrapper(
        request_data=loaded_data,
        input_objects=input_objects,
    )
    return parsed_data


def get_input_setup_function_for_serve(input_type: str) -> Callable:
    mapping_predict = get_input_setup_function_map_for_predict()

    mapping_predict["tabular"] = _setup_tabular_input_for_serve

    return mapping_predict[input_type]


def _setup_tabular_input_for_serve(
    input_config: InputConfig, output_folder: Path, *args, **kwargs
) -> ComputedServeTabularInputInfo:
    input_info = input_config.input_info
    input_type_info = input_config.input_type_info

    assert isinstance(input_type_info, TabularInputDataConfig)

    cat_columns = list(input_type_info.input_cat_columns)
    con_columns = list(input_type_info.input_con_columns)
    all_columns = cat_columns + con_columns

    transformers = load_transformers(
        transformers_to_load={input_info.input_name: all_columns},
        output_folder=str(output_folder),
    )
    serve_labels = ServeLabels(label_transformers=transformers[input_info.input_name])

    serve_tabular_info = ComputedServeTabularInputInfo(
        labels=serve_labels,
        input_config=input_config,
    )

    return serve_tabular_info


def _load_request_data(data: Sequence) -> Sequence[Dict[str, Any]]:
    input_data = data
    inputs_loaded = input_data

    return inputs_loaded


def general_pre_process_raw_inputs_wrapper(
    raw_inputs: Sequence[dict[str, Any]],
    experiment: Union["Experiment", "ServeExperiment"],
) -> Sequence[dict[str, torch.Tensor]]:

    all_preprocessed = []

    for raw_input in raw_inputs:
        preprocessed = general_pre_process_raw_inputs(
            raw_inputs=raw_input,
            experiment=experiment,
        )
        all_preprocessed.append(preprocessed)

    return all_preprocessed


def general_pre_process_raw_inputs(
    raw_inputs: dict[str, Any],
    experiment: Union["Experiment", "ServeExperiment"],
) -> dict[str, torch.Tensor]:
    inputs_prepared_for_memory = {}
    for name, cur_input in raw_inputs.items():
        input_object = experiment.inputs[name]

        match input_object:
            case ComputedSequenceInputInfo():
                cur_input = input_object.encode_func(cur_input)

            case (
                ComputedTabularInputInfo()
                | ComputedPredictTabularInputInfo()
                | ComputedServeTabularInputInfo()
            ):
                cur_input = _impute_missing_tabular_values(
                    input_object=input_object,
                    inputs_values=cur_input,
                )

        inputs_prepared_for_memory[name] = cur_input

    inputs_prepared = prepare_inputs_memory(
        inputs=inputs_prepared_for_memory,
        inputs_objects=experiment.inputs,
        test_mode=True,
    )

    inputs_final = impute_missing_modalities_wrapper(
        inputs_values=inputs_prepared,
        inputs_objects=experiment.inputs,
    )

    return inputs_final


def _impute_missing_tabular_values(
    input_object: (
        ComputedTabularInputInfo
        | ComputedPredictTabularInputInfo
        | ComputedServeTabularInputInfo
    ),
    inputs_values: dict[str, Any],
) -> dict[str, Any]:
    # TODO: Implement
    return inputs_values
