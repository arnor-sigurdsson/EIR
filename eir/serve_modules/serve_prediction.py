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


@torch.inference_mode()
def run_serve_prediction(
    serve_experiment: ServeExperiment, batch: "ServeBatch"
) -> dict[str, dict[str, torch.Tensor | list[int] | np.ndarray]]:
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

    merged: dict[str, dict[str, torch.Tensor | list[int] | np.ndarray]]
    merged = {
        **one_shot_prediction,  # type: ignore
        **sequence_predictions,  # type: ignore
        **array_predictions,  # type: ignore
    }

    return merged


def _run_serve_sequence_generation(
    serve_experiment: ServeExperiment,
    batch: "ServeBatch",
) -> dict[str, dict[str, list[int]]]:
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

            eval_sample = SequenceOutputEvalSample(
                inputs_to_model=batch.pre_hook_inputs,
                target_labels={},
                sample_id="manual_request",
            )

            hooks = serve_experiment.hooks
            assert hooks is not None

            generated_tokens = autoregressive_sequence_generation(
                input_objects=serve_experiment.inputs,
                eval_sample=eval_sample,
                seq_output_name=cur_output_name,
                experiment=serve_experiment,
                default_eir_hooks=hooks,
                sampling_config=config.sampling_config,
            )

            prepared[cur_output_name] = {cur_output_name: generated_tokens}

    return prepared


def _run_serve_array_generation(
    serve_experiment: ServeExperiment,
    batch: "ServeBatch",
) -> dict[str, dict[str, np.ndarray]]:
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

        eval_sample = ArrayOutputEvalSample(
            inputs_to_model=batch.pre_hook_inputs,
            target_labels={},
            sample_id="manual_request",
        )

        hooks = serve_experiment.hooks
        assert hooks is not None

        is_diffusion = output_type_info.loss == "diffusion"

        if is_diffusion:
            time_steps = output_type_info.diffusion_time_steps
            assert time_steps is not None
            array_output = reverse_diffusion_array_generation(
                eval_sample=eval_sample,
                array_output_name=cur_output_name,
                experiment=serve_experiment,
                default_eir_hooks=hooks,
                num_steps=time_steps,
            )
        else:
            array_output = one_shot_array_generation(
                eval_sample=eval_sample,
                array_output_name=cur_output_name,
                experiment=serve_experiment,
                default_eir_hooks=hooks,
            )

        prepared[cur_output_name] = {cur_output_name: array_output}

    return prepared
