import json
from collections.abc import Generator, Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Union,
)

import numpy as np
import torch
from aislib.misc_utils import ensure_path_exists
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
)
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from eir.data_load.data_preparation_modules.imputation import (
    impute_missing_modalities_wrapper,
)
from eir.data_load.data_preparation_modules.prepare_array import un_normalize_array
from eir.data_load.datasets import al_getitem_return
from eir.interpretation.interpret_image import un_normalize_image
from eir.models.model_training_utils import predict_on_batch, recursive_to_device
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
from eir.train_utils.evaluation_modules.evaluation_handlers_utils import (
    convert_model_inputs_to_raw,
    general_pre_process_prepared_inputs,
    get_batch_generator,
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
    auto_samples: dict[str, list["ArrayOutputEvalSample"]]
    manual_samples: dict[str, list["ArrayOutputEvalSample"]]


@dataclass()
class ArrayOutputEvalSample:
    inputs_to_model: dict[str, Any]
    target_labels: dict[str, Any]
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

    auto_validation_dataloader = get_dataset_loader_single_sample_generator(
        dataset=auto_dataset_to_load_from,
        fabric=experiment.fabric,
    )

    auto_samples = get_array_output_auto_validation_samples(
        output_configs=output_configs,
        eval_sample_iterator=auto_validation_dataloader,
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
            output_type_info, ArrayOutputTypeConfig | ImageOutputTypeConfig
        )

        assert config.sampling_config is not None

        sample_generator = _get_eval_sample_generator(
            eval_samples=eval_samples_base, output_name=cur_output_name
        )

        batch_generator = get_batch_generator(
            iterator=enumerate(sample_generator),
            batch_size=experiment.configs.gc.be.batch_size,
        )

        meta = {}

        for batch in batch_generator:
            batch_indices, batch_eval_data = zip(*batch, strict=False)
            batch_eval_types, batch_eval_samples = zip(*batch_eval_data, strict=False)

            if output_type_info.loss == "diffusion":
                train_time_steps = output_type_info.diffusion_time_steps
                assert train_time_steps is not None

                sampling_config = config.sampling_config
                assert isinstance(
                    sampling_config,
                    ArrayOutputSamplingConfig | ImageOutputSamplingConfig,
                )

                inference_steps = sampling_config.diffusion_inference_steps
                scheduler_type = sampling_config.diffusion_sampler
                eta = sampling_config.diffusion_eta

                batch_generated_arrays = diffusion_generation_with_scheduler(
                    eval_samples=batch_eval_samples,
                    array_output_name=cur_output_name,
                    experiment=experiment,
                    default_eir_hooks=default_eir_hooks,
                    num_train_steps=train_time_steps,
                    num_inference_steps=inference_steps,
                    scheduler_type=scheduler_type,
                    eta=eta,
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
                strict=False,
            ):
                cur_output_path = (
                    cur_sample_output_folder / eval_type / f"{idx}_generated.npy"
                )
                match output_type_info:
                    case ArrayOutputTypeConfig():
                        save_array_output(
                            array=generated_array,
                            output_path=cur_output_path,
                        )
                    case ImageOutputTypeConfig():
                        cur_output_path = cur_output_path.with_suffix(".png")
                        save_image_output(
                            array=generated_array,
                            output_path=cur_output_path,
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

                cur_id = eval_sample.sample_id[0]
                meta[cur_id] = {
                    "generated": str(cur_output_path.relative_to(output_folder)),
                    "inputs": str(cur_inputs_output_path.relative_to(output_folder)),
                    "index": idx,
                }

            meta_path = cur_sample_output_folder / "meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=4)


@torch.no_grad()
def one_shot_array_generation(
    eval_samples: tuple[ArrayOutputEvalSample, ...],
    array_output_name: str,
    experiment: Union["Experiment", "PredictExperiment", "ServeExperiment"],
    default_eir_hooks: Union["Hooks", "PredictHooks"],
) -> list[np.ndarray]:
    output_object = experiment.outputs[array_output_name]

    assert isinstance(output_object, ComputedArrayOutputInfo | ComputedImageOutputInfo)

    device = str(experiment.fabric.device)
    array_sampling_batch = prepare_array_sampling_batch(
        eval_samples=eval_samples,
        array_output_name=array_output_name,
        device=device,
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


def get_scheduler(
    scheduler_type: str,
    num_train_steps: int,
    num_inference_steps: int,
    device: str,
    betas: torch.Tensor,
    eta: float | None = None,
) -> Any:
    betas_numpy = betas.cpu().numpy()

    scheduler: DDPMScheduler | DDIMScheduler | DPMSolverMultistepScheduler
    if scheduler_type == "ddpm":
        scheduler = DDPMScheduler(
            num_train_timesteps=num_train_steps,
            trained_betas=betas_numpy,
            clip_sample=False,
            prediction_type="v_prediction",
            timestep_spacing="trailing",
        )
        eta_for_step = None
    elif scheduler_type == "ddim":
        scheduler = DDIMScheduler(
            num_train_timesteps=num_train_steps,
            trained_betas=betas_numpy,
            clip_sample=False,
            prediction_type="v_prediction",
            timestep_spacing="trailing",
        )
        eta_for_step = eta if eta is not None else 0.0
    elif scheduler_type == "dpm_solver":
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=num_train_steps,
            trained_betas=betas_numpy,
            prediction_type="v_prediction",
            timestep_spacing="trailing",
        )
        eta_for_step = None
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, defaulting to DDPM")
        scheduler = DDPMScheduler(
            num_train_timesteps=num_train_steps,
            trained_betas=betas_numpy,
            clip_sample=False,
            prediction_type="v_prediction",
            timestep_spacing="trailing",
        )
        eta_for_step = eta if eta is not None else 0.0

    scheduler.set_timesteps(  # type: ignore[union-attr]
        num_inference_steps=num_inference_steps,
        device=device,
    )
    return scheduler, eta_for_step


def prepare_model_timestep(
    model: Any,
    output_name: str,
    t_batch: torch.Tensor,
    batch_inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    batch_inputs = deepcopy(batch_inputs)

    output_module = getattr(model.output_modules, output_name)
    t_emb = output_module.feature_extractor.timestep_embeddings(t_batch)
    batch_inputs[f"__extras_{output_name}"] = t_emb

    return batch_inputs


@torch.inference_mode()
def diffusion_generation_with_scheduler(
    eval_samples: tuple[ArrayOutputEvalSample, ...],
    array_output_name: str,
    experiment: Union["Experiment", "PredictExperiment", "ServeExperiment"],
    default_eir_hooks: Union["Hooks", "PredictHooks"],
    num_train_steps: int,
    num_inference_steps: int = 50,
    scheduler_type: str = "ddim",
    eta: float | None = None,
) -> list[np.ndarray]:
    output_object = experiment.outputs[array_output_name]
    assert isinstance(output_object, ComputedArrayOutputInfo | ComputedImageOutputInfo)

    dimensions = output_object.data_dimensions
    batch_size = len(eval_samples)
    shape = (batch_size,) + dimensions.full_shape()
    device = str(experiment.fabric.device)

    diffusion_config = output_object.diffusion_config
    assert diffusion_config is not None
    betas = diffusion_config.betas

    scheduler, eta_for_step = get_scheduler(
        scheduler_type=scheduler_type,
        num_train_steps=num_train_steps,
        num_inference_steps=num_inference_steps,
        device=device,
        eta=eta,
        betas=betas,
    )

    array_sampling_batch = prepare_array_sampling_batch(
        eval_samples=eval_samples,
        device=device,
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

    current_state = torch.randn(shape, device=device)

    for t in scheduler.timesteps:
        batch_inputs[array_output_name] = current_state

        t_batch = torch.full(
            (batch_size,),
            t,
            dtype=torch.long,
            device=device,
        )
        batch_inputs = prepare_model_timestep(
            model=experiment.model,
            output_name=array_output_name,
            t_batch=t_batch,
            batch_inputs=batch_inputs,
        )

        model_outputs = experiment.model(batch_inputs)
        model_diffusion_output = model_outputs[array_output_name][array_output_name]

        if scheduler_type == "ddim" and eta_for_step is not None:
            scheduler_output = scheduler.step(
                model_output=model_diffusion_output,
                timestep=t,
                sample=current_state,
                eta=eta_for_step,
            )
        else:
            scheduler_output = scheduler.step(
                model_output=model_diffusion_output,
                timestep=t,
                sample=current_state,
            )

        current_state = scheduler_output.prev_sample

    final_states = current_state.cpu().numpy()

    assert output_object.normalization_stats is not None
    final_numpy_outputs = []
    for batch_idx in range(batch_size):
        cur_sample_final_state = final_states[batch_idx]
        cur_final_output = un_normalize_wrapper(
            array=torch.from_numpy(cur_sample_final_state).to(device),
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
    device: str,
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
        cur_prepared = recursive_to_device(obj=cur_prepared, device=device)

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
    mode: str | None
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
) -> dict[str, list[ArrayOutputEvalSample]]:
    prepared_samples: dict[str, list[ArrayOutputEvalSample]] = {}

    for _config_idx, config in enumerate(output_configs):
        array_or_image = config.output_info.output_type in ("array", "image")
        if not config.sampling_config or not array_or_image:
            continue

        assert isinstance(
            config.sampling_config,
            ArrayOutputSamplingConfig | ImageOutputSamplingConfig,
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
) -> dict[str, list[ArrayOutputEvalSample]]:
    prepared_eval_samples: dict[str, list[ArrayOutputEvalSample]] = {}

    for _config_idx, config in enumerate(output_configs):
        array_or_image = config.output_info.output_type in ("array", "image")
        if not config.sampling_config or not array_or_image:
            continue

        output_name = config.output_info.output_name

        prepared_eval_samples[output_name] = []

        assert isinstance(
            config.sampling_config,
            ArrayOutputSamplingConfig | ImageOutputSamplingConfig,
        )
        for _i in range(config.sampling_config.n_eval_inputs):
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
) -> Generator[tuple[str, ArrayOutputEvalSample]]:
    cur_config_auto_samples = eval_samples.auto_samples[output_name]
    cur_config_manual_samples = eval_samples.manual_samples[output_name]

    for eval_sample in cur_config_auto_samples:
        yield "auto", eval_sample

    for eval_sample in cur_config_manual_samples:
        yield "manual", eval_sample
