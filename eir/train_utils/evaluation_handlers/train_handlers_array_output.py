from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Generator, Iterator, Sequence, Tuple, Union

import numpy as np
import torch
from aislib.misc_utils import ensure_path_exists
from torch.utils.data import Dataset

from eir.data_load.data_preparation_modules.prepare_array import un_normalize_array
from eir.data_load.datasets import al_getitem_return
from eir.models.model_training_utils import predict_on_batch
from eir.setup.output_setup_modules.array_output_setup import ComputedArrayOutputInfo
from eir.setup.schemas import (
    ArrayOutputSamplingConfig,
    ArrayOutputTypeConfig,
    OutputConfig,
)
from eir.train_utils import utils
from eir.train_utils.evaluation_handlers.evaluation_handlers_utils import (
    convert_model_inputs_to_raw,
    general_pre_process_prepared_inputs,
    get_dataset_loader_single_sample_generator,
    prepare_base_input,
    prepare_manual_sample_data,
    serialize_raw_inputs,
)
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.predict import PredictExperiment, PredictHooks
    from eir.serve import ServeExperiment
    from eir.train import Experiment, Hooks, al_input_objects_as_dict

logger = get_logger(name=__name__)


@dataclass
class ArrayOutputEvalSamples:
    auto_samples: Dict[str, list["ArrayOutputEvalSample"]]
    manual_samples: Dict[str, list["ArrayOutputEvalSample"]]


@dataclass()
class ArrayOutputEvalSample:
    inputs_to_model: Dict[str, Any]
    target_labels: Dict[str, Any]
    sample_id: str


def array_out_single_sample_evaluation_wrapper(
    experiment: Union["Experiment", "PredictExperiment"],
    input_objects: "al_input_objects_as_dict",
    auto_dataset_to_load_from: Dataset,
    iteration: int,
    output_folder: str,
) -> None:
    default_eir_hooks = experiment.hooks

    output_configs = experiment.configs.output_configs

    if not any(i.sampling_config for i in output_configs):
        return

    manual_samples = get_array_output_manual_input_samples(
        output_configs=output_configs, input_objects=input_objects
    )

    auto_validation_generator = get_dataset_loader_single_sample_generator(
        dataset=auto_dataset_to_load_from
    )
    auto_samples = get_array_output_auto_validation_samples(
        output_configs=output_configs,
        eval_sample_iterator=auto_validation_generator,
    )
    eval_samples = ArrayOutputEvalSamples(
        auto_samples=auto_samples, manual_samples=manual_samples
    )

    for config in output_configs:
        if config.output_info.output_type != "array":
            continue

        cur_input_name = config.output_info.output_name
        cur_output_name = config.output_info.output_name

        not_in_manual_samples = cur_output_name not in eval_samples.manual_samples
        not_in_auto_samples = cur_output_name not in eval_samples.auto_samples
        if not_in_manual_samples and not_in_auto_samples:
            continue

        cur_sample_output_folder = utils.prepare_sample_output_folder(
            output_folder=output_folder,
            output_name=cur_output_name,
            column_name=cur_input_name,
            iteration=iteration,
        )

        output_type_info = config.output_type_info
        assert isinstance(output_type_info, ArrayOutputTypeConfig)

        assert config.sampling_config is not None

        sample_generator = _get_eval_sample_generator(
            eval_samples=eval_samples, output_name=cur_output_name
        )

        for idx, (eval_type, eval_sample) in enumerate(sample_generator):
            generated_array = array_generation(
                eval_sample=eval_sample,
                array_output_name=cur_output_name,
                experiment=experiment,
                default_eir_hooks=default_eir_hooks,
            )

            cur_output_path = (
                cur_sample_output_folder / eval_type / f"{idx}_generated.npy"
            )
            ensure_path_exists(path=cur_output_path)
            np.save(file=cur_output_path, arr=generated_array)

            cur_inputs_output_path = (
                cur_sample_output_folder / eval_type / f"{idx}_inputs"
            )

            raw_inputs = convert_model_inputs_to_raw(
                inputs_to_model=eval_sample.inputs_to_model,
                input_objects=input_objects,
            )

            serialize_raw_inputs(
                raw_inputs=raw_inputs,
                input_objects=input_objects,
                output_path=cur_inputs_output_path,
            )


@torch.no_grad()
def array_generation(
    eval_sample: ArrayOutputEvalSample,
    array_output_name: str,
    experiment: Union["Experiment", "PredictExperiment", "ServeExperiment"],
    default_eir_hooks: Union["Hooks", "PredictHooks"],
) -> np.ndarray:
    output_object = experiment.outputs[array_output_name]

    assert isinstance(output_object, ComputedArrayOutputInfo)

    prepared_sample_inputs = prepare_base_input(
        prepared_inputs=eval_sample.inputs_to_model,
    )

    batch = general_pre_process_prepared_inputs(
        prepared_inputs=prepared_sample_inputs,
        target_labels=eval_sample.target_labels,
        sample_id=eval_sample.sample_id,
        experiment=experiment,
        custom_hooks=default_eir_hooks,
    )

    outputs = predict_on_batch(
        model=experiment.model,
        inputs=batch.inputs,
    )

    array_output = outputs[array_output_name][array_output_name]

    array_output_raw = un_normalize_array(
        array=array_output,
        normalization_stats=output_object.normalization_stats,
    )

    array_output_raw_numpy = array_output_raw.cpu().numpy()

    return array_output_raw_numpy


def get_array_output_manual_input_samples(
    output_configs: Sequence[OutputConfig],
    input_objects: "al_input_objects_as_dict",
) -> Dict[str, list[ArrayOutputEvalSample]]:
    prepared_samples: dict[str, list[ArrayOutputEvalSample]] = {}

    for config_idx, config in enumerate(output_configs):
        if not config.sampling_config or config.output_info.output_type != "array":
            continue

        assert isinstance(config.sampling_config, ArrayOutputSamplingConfig)

        sample_data_from_yaml = config.sampling_config.manual_inputs
        output_name = config.output_info.output_name

        prepared_samples[output_name] = []

        for idx, single_sample_inputs in enumerate(sample_data_from_yaml):
            prepared_inputs = prepare_manual_sample_data(
                sample_inputs=single_sample_inputs, input_objects=input_objects
            )

            cur_eval_sample = ArrayOutputEvalSample(
                inputs_to_model=prepared_inputs,
                target_labels={},
                sample_id=f"manual_{idx}",
            )

            prepared_samples[output_name].append(cur_eval_sample)

    return prepared_samples


def get_array_output_auto_validation_samples(
    output_configs: Sequence[OutputConfig],
    eval_sample_iterator: Iterator[al_getitem_return],
) -> Dict[str, list[ArrayOutputEvalSample]]:
    prepared_eval_samples: dict[str, list[ArrayOutputEvalSample]] = {}

    for config_idx, config in enumerate(output_configs):
        if not config.sampling_config or config.output_info.output_type != "array":
            continue

        output_name = config.output_info.output_name

        prepared_eval_samples[output_name] = []

        assert isinstance(config.sampling_config, ArrayOutputSamplingConfig)
        for i in range(config.sampling_config.n_eval_inputs):
            input_to_model, target_labels, cur_id = next(eval_sample_iterator)

            cur_eval_sample = ArrayOutputEvalSample(
                inputs_to_model=input_to_model,
                target_labels=target_labels,
                sample_id=cur_id,
            )

            prepared_eval_samples[output_name].append(cur_eval_sample)

    return prepared_eval_samples


def _get_eval_sample_generator(
    eval_samples: ArrayOutputEvalSamples, output_name: str
) -> Generator[Tuple[str, ArrayOutputEvalSample], None, None]:
    cur_config_auto_samples = eval_samples.auto_samples[output_name]
    cur_config_manual_samples = eval_samples.manual_samples[output_name]

    for eval_sample in cur_config_auto_samples:
        yield "auto", eval_sample

    for eval_sample in cur_config_manual_samples:
        yield "manual", eval_sample
