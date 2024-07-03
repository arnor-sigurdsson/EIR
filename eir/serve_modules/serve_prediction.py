from typing import TYPE_CHECKING

import numpy as np
import torch

from eir.models.model_training_utils import predict_on_batch
from eir.serve_modules.serve_experiment_io import ServeExperiment
from eir.setup.schema_modules.output_schemas_array import ArrayOutputTypeConfig
from eir.setup.schema_modules.output_schemas_image import ImageOutputTypeConfig
from eir.setup.schema_modules.output_schemas_sequence import (
    SequenceOutputSamplingConfig,
)
from eir.setup.schemas import SequenceOutputTypeConfig
from eir.train_utils.evaluation_handlers.train_handlers_array_output import (
    ArrayOutputEvalSample,
    one_shot_array_generation,
    reverse_diffusion_array_generation,
)
from eir.train_utils.evaluation_handlers.train_handlers_sequence_output import (
    SequenceOutputEvalSample,
    autoregressive_sequence_generation,
)

if TYPE_CHECKING:
    from eir.serve_modules.serve_input_setup import ServeBatch

al_merged_predictions = list[
    dict[str, dict[str, torch.Tensor | list[int] | np.ndarray]]
]


@torch.inference_mode()
def run_serve_prediction(
    serve_experiment: ServeExperiment,
    batch: "ServeBatch",
) -> list[dict[str, dict[str, torch.Tensor | list[int] | np.ndarray]]]:
    one_shot_prediction = predict_on_batch(
        model=serve_experiment.model,
        inputs=batch.inputs,
    )

    sequence_predictions = _run_serve_sequence_generation(
        serve_experiment=serve_experiment,
        batch=batch,
    )

    array_predictions = _run_serve_array_generation(
        serve_experiment=serve_experiment,
        batch=batch,
    )

    predictions = merge_predictions_per_sample(
        one_shot_prediction=one_shot_prediction,
        sequence_predictions=sequence_predictions,
        array_predictions=array_predictions,
        batch_size=len(batch.ids),
    )

    return predictions


def merge_predictions_per_sample(
    one_shot_prediction: dict[str, dict[str, torch.Tensor]],
    sequence_predictions: dict[str, dict[str, list[list[int]]]],
    array_predictions: dict[str, dict[str, list[np.ndarray]]],
    batch_size: int,
) -> al_merged_predictions:

    merged_predictions: al_merged_predictions = [{} for _ in range(batch_size)]

    for os_key, os_value in one_shot_prediction.items():
        for sub_key, tensor in os_value.items():
            for i in range(batch_size):
                if os_key not in merged_predictions[i]:
                    merged_predictions[i][os_key] = {}
                merged_predictions[i][os_key][sub_key] = tensor[i]

    for seq_key, seq_value in sequence_predictions.items():
        for sub_key, seq_list in seq_value.items():
            for i in range(batch_size):
                if seq_key not in merged_predictions[i]:
                    merged_predictions[i][seq_key] = {}
                merged_predictions[i][seq_key][sub_key] = seq_list[i]

    for arr_key, arr_value in array_predictions.items():
        for sub_key, array_list in arr_value.items():
            for i in range(batch_size):
                if arr_key not in merged_predictions[i]:
                    merged_predictions[i][arr_key] = {}
                merged_predictions[i][arr_key][sub_key] = array_list[i]

    return merged_predictions


def _run_serve_sequence_generation(
    serve_experiment: ServeExperiment,
    batch: "ServeBatch",
) -> dict[str, dict[str, list[list[int]]]]:
    """
    Note that we always expect two levels for the predictions, in the case
    of tabular that's the output name and the column name, in the case of
    sequence we simply output name twice to conform to the same structure.
    """
    prepared = {}

    output_configs = serve_experiment.configs.output_configs
    for config in output_configs:
        if config.output_info.output_type != "sequence":
            continue

        cur_output_name = config.output_info.output_name

        output_type_info = config.output_type_info
        assert isinstance(output_type_info, SequenceOutputTypeConfig)

        assert config.sampling_config is not None

        if output_type_info.sequence_operation == "autoregressive":
            assert isinstance(config.sampling_config, SequenceOutputSamplingConfig)

            eval_samples = _build_sequence_eval_samples_batch(
                serve_batch=batch,
            )

            hooks = serve_experiment.hooks
            assert hooks is not None

            generated_tokens = autoregressive_sequence_generation(
                input_objects=serve_experiment.inputs,
                eval_samples=eval_samples,
                seq_output_name=cur_output_name,
                experiment=serve_experiment,
                default_eir_hooks=hooks,
                sampling_config=config.sampling_config,
            )

            prepared[cur_output_name] = {cur_output_name: generated_tokens}

    return prepared


def _build_sequence_eval_samples_batch(
    serve_batch: "ServeBatch",
) -> tuple[SequenceOutputEvalSample, ...]:
    eval_samples = []

    for i in range(len(serve_batch.ids)):

        prepared_inputs = serve_batch.inputs_split[i]

        eval_sample = SequenceOutputEvalSample(
            inputs_to_model=prepared_inputs,
            target_labels={},
            sample_id=serve_batch.ids[i],
        )
        eval_samples.append(eval_sample)

    return tuple(eval_samples)


def _run_serve_array_generation(
    serve_experiment: ServeExperiment,
    batch: "ServeBatch",
) -> dict[str, dict[str, list[np.ndarray]]]:
    prepared = {}

    output_configs = serve_experiment.configs.output_configs

    for config in output_configs:
        if config.output_info.output_type not in ("array", "image"):
            continue

        cur_output_name = config.output_info.output_name
        output_type_info = config.output_type_info
        assert isinstance(
            output_type_info, (ArrayOutputTypeConfig, ImageOutputTypeConfig)
        )

        assert config.sampling_config is not None

        eval_samples = _build_array_eval_samples_batch(serve_batch=batch)

        hooks = serve_experiment.hooks
        assert hooks is not None

        is_diffusion = output_type_info.loss == "diffusion"

        if is_diffusion:
            time_steps = output_type_info.diffusion_time_steps
            assert time_steps is not None
            array_outputs = reverse_diffusion_array_generation(
                eval_samples=eval_samples,
                array_output_name=cur_output_name,
                experiment=serve_experiment,
                default_eir_hooks=hooks,
                num_steps=time_steps,
            )
        else:
            array_outputs = one_shot_array_generation(
                eval_samples=eval_samples,
                array_output_name=cur_output_name,
                experiment=serve_experiment,
                default_eir_hooks=hooks,
            )

        prepared[cur_output_name] = {cur_output_name: array_outputs}

    return prepared


def _build_array_eval_samples_batch(
    serve_batch: "ServeBatch",
) -> tuple[ArrayOutputEvalSample, ...]:
    eval_samples = []

    for i in range(len(serve_batch.ids)):
        eval_sample = ArrayOutputEvalSample(
            inputs_to_model=serve_batch.inputs_split[i],
            target_labels={},
            sample_id=serve_batch.ids[i],
        )
        eval_samples.append(eval_sample)

    return tuple(eval_samples)
