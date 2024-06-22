from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
from aislib.misc_utils import ensure_path_exists
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from eir.data_load.data_preparation_modules.imputation import (
    impute_missing_modalities_wrapper,
)
from eir.data_load.data_preparation_modules.prepare_array import un_normalize_array
from eir.data_load.datasets import al_getitem_return
from eir.interpretation.interpret_image import un_normalize_image
from eir.models.model_training_utils import predict_on_batch
from eir.setup.input_setup_modules.setup_array import ArrayNormalizationStats
from eir.setup.input_setup_modules.setup_image import ImageNormalizationStats
from eir.setup.output_setup_modules.array_output_setup import ComputedArrayOutputInfo
from eir.setup.output_setup_modules.image_output_setup import ComputedImageOutputInfo
from eir.setup.schemas import (
    ArrayOutputSamplingConfig,
    ArrayOutputTypeConfig,
    ImageOutputSamplingConfig,
    ImageOutputTypeConfig,
    OutputConfig,
)
from eir.train_utils import utils
from eir.train_utils.evaluation_handlers.evaluation_handlers_utils import (
    convert_model_inputs_to_raw,
    general_pre_process_prepared_inputs,
    get_batch_generator,
    get_dataset_loader_single_sample_generator,
    prepare_base_input,
    prepare_manual_sample_data,
    serialize_raw_inputs,
)
from eir.train_utils.step_modules.diffusion import p_sample_loop
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
        output_configs=output_configs,
        input_objects=input_objects,
    )

    auto_validation_generator = get_dataset_loader_single_sample_generator(
        dataset=auto_dataset_to_load_from
    )
    auto_samples = get_array_output_auto_validation_samples(
        output_configs=output_configs,
        eval_sample_iterator=auto_validation_generator,
    )
    eval_samples_base = ArrayOutputEvalSamples(
        auto_samples=auto_samples,
        manual_samples=manual_samples,
    )

    for config in output_configs:
        if config.output_info.output_type not in ("array", "image"):
            continue

        cur_input_name = config.output_info.output_name
        cur_output_name = config.output_info.output_name

        not_in_manual_samples = cur_output_name not in eval_samples_base.manual_samples
        not_in_auto_samples = cur_output_name not in eval_samples_base.auto_samples
        if not_in_manual_samples and not_in_auto_samples:
            continue

        cur_sample_output_folder = utils.prepare_sample_output_folder(
            output_folder=output_folder,
            output_name=cur_output_name,
            column_name=cur_input_name,
            iteration=iteration,
        )

        output_type_info = config.output_type_info
        assert isinstance(
            output_type_info, (ArrayOutputTypeConfig, ImageOutputTypeConfig)
        )

        assert config.sampling_config is not None

        sample_generator = _get_eval_sample_generator(
            eval_samples=eval_samples_base, output_name=cur_output_name
        )

        batch_generator = get_batch_generator(
            iterator=enumerate(sample_generator),
            batch_size=experiment.configs.global_config.batch_size,
        )

        for batch in batch_generator:
            batch_indices, batch_eval_data = zip(*batch)
            batch_eval_types, batch_eval_samples = zip(*batch_eval_data)

            if output_type_info.loss == "diffusion":
                time_steps = output_type_info.diffusion_time_steps
                assert time_steps is not None
                batch_generated_arrays = reverse_diffusion_array_generation(
                    eval_samples=batch_eval_samples,
                    array_output_name=cur_output_name,
                    experiment=experiment,
                    default_eir_hooks=default_eir_hooks,
                    num_steps=time_steps,
                )
            else:
                batch_generated_arrays = one_shot_array_generation(
                    eval_samples=batch_eval_samples,
                    array_output_name=cur_output_name,
                    experiment=experiment,
                    default_eir_hooks=default_eir_hooks,
                )

            for eval_type, idx, eval_sample, generated_array in zip(
                batch_eval_types,
                batch_indices,
                batch_eval_samples,
                batch_generated_arrays,
            ):

                cur_output_path = (
                    cur_sample_output_folder / eval_type / f"{idx}_generated.npy"
                )
                match output_type_info:
                    case ArrayOutputTypeConfig():
                        save_array_output(
                            array=generated_array, output_path=cur_output_path
                        )
                    case ImageOutputTypeConfig():
                        cur_output_path = cur_output_path.with_suffix(".png")
                        save_image_output(
                            array=generated_array, output_path=cur_output_path
                        )

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
def one_shot_array_generation(
    eval_samples: tuple[ArrayOutputEvalSample, ...],
    array_output_name: str,
    experiment: Union["Experiment", "PredictExperiment", "ServeExperiment"],
    default_eir_hooks: Union["Hooks", "PredictHooks"],
) -> list[np.ndarray]:
    output_object = experiment.outputs[array_output_name]

    assert isinstance(output_object, (ComputedArrayOutputInfo, ComputedImageOutputInfo))

    array_sampling_batch = prepare_array_sampling_batch(
        eval_samples=eval_samples, array_output_name=array_output_name
    )
    prepared_sample_inputs = array_sampling_batch.prepared_inputs
    prepared_targets = array_sampling_batch.prepared_targets

    all_inputs = default_collate(prepared_sample_inputs)
    all_targets = default_collate(prepared_targets)
    all_ids = [eval_sample.sample_id for eval_sample in eval_samples]

    batch = general_pre_process_prepared_inputs(
        prepared_inputs=all_inputs,
        target_labels=all_targets,
        sample_ids=all_ids,
        experiment=experiment,
        custom_hooks=default_eir_hooks,
    )

    outputs = predict_on_batch(
        model=experiment.model,
        inputs=batch.inputs,
    )

    batch_size = len(eval_samples)
    array_outputs = outputs[array_output_name][array_output_name]

    assert output_object.normalization_stats is not None

    final_numpy_outputs = []
    for batch_idx in range(batch_size):
        cur_output = array_outputs[batch_idx]
        cur_output_raw = un_normalize_wrapper(
            array=cur_output,
            normalization_stats=output_object.normalization_stats,
        )
        cur_output_numpy = cur_output_raw.cpu().numpy()
        final_numpy_outputs.append(cur_output_numpy)

    return final_numpy_outputs


@torch.inference_mode()
def reverse_diffusion_array_generation(
    eval_samples: tuple[ArrayOutputEvalSample, ...],
    array_output_name: str,
    experiment: Union["Experiment", "PredictExperiment", "ServeExperiment"],
    default_eir_hooks: Union["Hooks", "PredictHooks"],
    num_steps: int,
) -> list[np.ndarray]:
    """
    TODO: Consider not keeping / making configurable whether to keep intermediate
          states as this might blow up memory usage in some cases.
    """
    output_object = experiment.outputs[array_output_name]
    assert isinstance(output_object, (ComputedArrayOutputInfo, ComputedImageOutputInfo))

    dimensions = output_object.data_dimensions
    batch_size = len(eval_samples)
    shape = (batch_size,) + dimensions.full_shape()

    array_sampling_batch = prepare_array_sampling_batch(
        eval_samples=eval_samples,
        array_output_name=array_output_name,
    )
    prepared_sample_inputs = array_sampling_batch.prepared_inputs
    prepared_targets = array_sampling_batch.prepared_targets

    all_inputs = default_collate(prepared_sample_inputs)
    all_targets = default_collate(prepared_targets)
    all_ids = [eval_sample.sample_id for eval_sample in eval_samples]

    batch = general_pre_process_prepared_inputs(
        prepared_inputs=all_inputs,
        target_labels=all_targets,
        sample_ids=all_ids,
        experiment=experiment,
        custom_hooks=default_eir_hooks,
    )
    batch_inputs = deepcopy(batch.inputs)

    dc = output_object.diffusion_config
    assert dc is not None

    final_states = p_sample_loop(
        config=dc,
        batch_inputs=batch_inputs,
        output_name=array_output_name,
        model=experiment.model,
        output_shape=shape,
        time_steps=num_steps,
    )

    assert output_object.normalization_stats is not None
    final_numpy_outputs = []
    for batch_idx in range(batch_size):
        cur_sample_final_state = final_states[batch_idx]
        cur_final_output = un_normalize_wrapper(
            array=torch.from_numpy(cur_sample_final_state),
            normalization_stats=output_object.normalization_stats,
        )
        cur_final_output_numpy = cur_final_output.cpu().numpy()
        final_numpy_outputs.append(cur_final_output_numpy)

    return final_numpy_outputs


@dataclass(frozen=False)
class ArraySamplingBatch:
    prepared_inputs: list[dict[str, torch.Tensor]]
    prepared_targets: list[dict[str, torch.Tensor]]


def prepare_array_sampling_batch(
    eval_samples: Sequence[ArrayOutputEvalSample],
    array_output_name: str,
) -> ArraySamplingBatch:
    """
    Every sample has an extra batch dimension by default from e.g. the data
    loader and preparation. We squeeze here as we collate multiple samples
    later to avoid an extra dimension.
    """
    prepared_sample_inputs = []
    prepared_targets = []

    for eval_sample in eval_samples:
        cur_inputs = eval_sample.inputs_to_model
        cur_prepared = prepare_base_input(prepared_inputs=cur_inputs)

        prepared_sample_inputs.append(cur_prepared)
        prepared_targets.append(eval_sample.target_labels)

    return ArraySamplingBatch(
        prepared_inputs=prepared_sample_inputs,
        prepared_targets=prepared_targets,
    )


def un_normalize_wrapper(
    array: torch.Tensor,
    normalization_stats: ArrayNormalizationStats | ImageNormalizationStats,
) -> torch.Tensor:

    match normalization_stats:
        case ArrayNormalizationStats():
            un_normalized = un_normalize_array(
                array=array,
                normalization_stats=normalization_stats,
            )
        case ImageNormalizationStats():
            un_normalized_arr = un_normalize_image(
                normalized_img=array.cpu().numpy(),
                normalization_stats=normalization_stats,
            )
            un_normalized = torch.from_numpy(un_normalized_arr)
        case _:
            raise ValueError("Normalization stats not recognized.")

    return un_normalized


def save_array_output(array: np.ndarray, output_path: Path | str) -> None:
    ensure_path_exists(path=Path(output_path))
    np.save(file=output_path, arr=array)


def save_image_output(array: np.ndarray, output_path: Path | str) -> None:
    ensure_path_exists(path=Path(output_path))

    assert array.ndim == 3, "Input should be 3D"

    array_hwc = np.moveaxis(array, 0, -1)
    array_uint8 = (array_hwc * 255).astype(np.uint8)

    n_channels = array_uint8.shape[-1]
    mode: Optional[str]
    match n_channels:
        case 1:
            mode = "L"
            array_uint8 = array_uint8.squeeze(-1)
        case 3:
            mode = "RGB"
        case 4:
            mode = "RGBA"
        case _:
            mode = None

    pil_image = Image.fromarray(array_uint8, mode=mode)
    pil_image.save(output_path)


def get_array_output_manual_input_samples(
    output_configs: Sequence[OutputConfig],
    input_objects: "al_input_objects_as_dict",
) -> Dict[str, list[ArrayOutputEvalSample]]:
    prepared_samples: dict[str, list[ArrayOutputEvalSample]] = {}

    for config_idx, config in enumerate(output_configs):
        array_or_image = config.output_info.output_type in ("array", "image")
        if not config.sampling_config or not array_or_image:
            continue

        assert isinstance(
            config.sampling_config,
            (ArrayOutputSamplingConfig, ImageOutputSamplingConfig),
        )

        sample_data_from_yaml = config.sampling_config.manual_inputs
        output_name = config.output_info.output_name

        prepared_samples[output_name] = []

        for idx, single_sample_inputs in enumerate(sample_data_from_yaml):
            prepared_inputs = prepare_manual_sample_data(
                sample_inputs=single_sample_inputs,
                input_objects=input_objects,
            )

            imputed_inputs = impute_missing_modalities_wrapper(
                inputs_values=prepared_inputs,
                inputs_objects=input_objects,
            )

            cur_eval_sample = ArrayOutputEvalSample(
                inputs_to_model=imputed_inputs,
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
        array_or_image = config.output_info.output_type in ("array", "image")
        if not config.sampling_config or not array_or_image:
            continue

        output_name = config.output_info.output_name

        prepared_eval_samples[output_name] = []

        assert isinstance(
            config.sampling_config,
            (ArrayOutputSamplingConfig, ImageOutputSamplingConfig),
        )
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
