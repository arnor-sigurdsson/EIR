import copy
import os
import sys
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generator,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

# Filter warnings from attribution calculation
warnings.filterwarnings(
    "ignore", message="Setting forward, backward hooks and attributes on non-linear.*"
)
warnings.filterwarnings(
    "ignore", message=".*Attempting to normalize by value approximately 0.*"
)


import numpy as np
import torch
from aislib.misc_utils import ensure_path_exists
from captum.attr import NoiseTunnel
from ignite.engine import Engine
from torch import nn
from torch.cuda import OutOfMemoryError
from torch.utils.data import DataLoader, Subset
from torch.utils.hooks import RemovableHandle

from eir.data_load.data_utils import Batch, get_output_info_generator
from eir.data_load.datasets import al_datasets
from eir.interpretation.interpret_array import (
    ArrayConsumerCallable,
    analyze_array_input_attributions,
    get_array_sum_consumer,
)
from eir.interpretation.interpret_image import analyze_image_input_attributions
from eir.interpretation.interpret_omics import (
    OmicsConsumerCallable,
    analyze_omics_input_attributions,
    get_omics_consumer,
)
from eir.interpretation.interpret_sequence import analyze_sequence_input_attributions
from eir.interpretation.interpret_tabular import analyze_tabular_input_attributions
from eir.interpretation.interpretation_utils import MyIntegratedGradients
from eir.models.input.omics.omics_models import CNNModel, LinearModel
from eir.models.model_training_utils import gather_data_loader_samples
from eir.setup.output_setup_modules.tabular_output_setup import (
    ComputedTabularOutputInfo,
)
from eir.setup.schemas import InputConfig, OutputConfig
from eir.target_setup.target_label_setup import MissingTargetsInfo
from eir.train_utils.evaluation import validation_handler
from eir.train_utils.utils import (
    call_hooks_stage_iterable,
    prepare_sample_output_folder,
    validate_handler_dependencies,
)
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.data_load.label_setup import al_label_transformers_object
    from eir.models.model_setup_modules.meta_setup import al_meta_model
    from eir.predict import PredictExperiment
    from eir.predict_modules.predict_attributions import (
        LoadedTrainExperimentMixedWithPredict,
    )
    from eir.setup.input_setup import al_input_objects_as_dict
    from eir.train import Experiment
    from eir.train_utils.train_handlers import HandlerConfig

logger = get_logger(name=__name__, tqdm_compatible=True)

# Type aliases
al_explainers = Union[MyIntegratedGradients, NoiseTunnel]


class WrapperModelForAttribution(nn.Module):
    """
    We need this wrapper module because libraries often only handle torch.Tensor or
    List[torch.Tensor] inputs. However, we do not want to restrict our modules to
    only accept those formats, rather using a dict. Hence, we use this module to
    accept the list libraries expect, but call the wrapped model with a matched dict.
    """

    def __init__(
        self,
        wrapped_model,
        input_names: Iterable[str],
        output_name: str,
        column_name: str,
        *args,
        **kwargs,
    ):
        super().__init__()
        assert not wrapped_model.training
        self.wrapped_model = wrapped_model
        self.input_names = input_names
        self.output_name = output_name
        self.column_name = column_name

    def match_tuple_inputs_and_names(
        self, input_sequence: Sequence[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(input_sequence, torch.Tensor):
            input_sequence = list(input_sequence)

        matched = {
            k: v
            for k, v in zip(self.input_names, input_sequence)
            if not k.startswith("__extras_")
        }

        return matched

    def forward(self, *inputs):
        """
        Note that have a slightly custom / different forward here compared with
        the method in MetaModel. This is because we only compute the attributions
        for tabular outputs, but we might have models that have e.g. diffusion
        outputs as well. In this case, the dynamic timestep embeddings do not
        contribute at all to the tabular output, which when computing the gradients
        in the Integrated Gradients method, will either lead to (a) differentiated
        tensors not being used in the graph and/or (b) None gradients showing up
        in the IG method.

        Hence, we manually only call the tabular output module we are currently
        interested in here, skipping e.g. the time embeddings completely.

        This also allows us to optimize slightly by only calling output modules
        that are actually needed.
        """
        inputs_matched = self.match_tuple_inputs_and_names(input_sequence=inputs)
        model = self.wrapped_model

        feature_extractors_out = {}
        for module_name, cur_input_module in model.input_modules.items():
            module_input = inputs_matched[module_name]
            feature_extractors_out[module_name] = cur_input_module(module_input)

        fused_features = {}
        for output_type, fusion_module in model.fusion_modules.items():
            fused_features[output_type] = fusion_module(feature_extractors_out)
        cur_fusion_target = model.fusion_to_output_mapping[self.output_name]
        fused_features = fused_features[cur_fusion_target]

        output_modules_out = {}
        cur_output = model.output_modules[self.output_name](fused_features)
        output_modules_out[self.output_name] = cur_output

        return output_modules_out[self.output_name][self.column_name]


@contextmanager
def suppress_stdout_and_stderr() -> Iterator[None]:
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


@validate_handler_dependencies([validation_handler])
def attribution_analysis_handler(
    engine: Engine, handler_config: "HandlerConfig"
) -> None:
    exp = handler_config.experiment
    gc = exp.configs.global_config
    iteration = engine.state.iteration

    types, should_run = check_if_should_run_analysis(
        input_configs=exp.configs.input_configs
    )
    if not should_run:
        logger.warning(
            "Got compute_attributions: %s but none of the input types in %s currently "
            "support attribution analysis. "
            "Attribution analysis will be skipped. ",
            gc.compute_attributions,
            types,
        )
        return

    logger.debug("Running attribution analysis.")

    attribution_output_folder_callable = partial(
        _prepare_eval_attribution_outfolder,
        output_folder=gc.output_folder,
        iteration=iteration,
    )

    background_loader = get_background_loader(experiment=exp)

    tabular_attribution_analysis_wrapper(
        model=exp.model,
        experiment=exp,
        output_folder_target_callable=attribution_output_folder_callable,
        dataset_to_interpret=exp.valid_dataset,
        background_loader=background_loader,
    )


def check_if_should_run_analysis(
    input_configs: Sequence[InputConfig],
) -> Tuple[set, bool]:
    all_types = {i.input_info.input_type for i in input_configs}

    if all_types == {"bytes"}:
        return all_types, False
    return all_types, True


def get_background_loader(experiment: "Experiment") -> torch.utils.data.DataLoader:
    background_loader = copy.deepcopy(experiment.train_loader)

    return background_loader


def tabular_attribution_analysis_wrapper(
    model: "al_meta_model",
    experiment: Union[
        "Experiment", "PredictExperiment", "LoadedTrainExperimentMixedWithPredict"
    ],
    output_folder_target_callable: Callable,
    dataset_to_interpret: al_datasets,
    background_loader: torch.utils.data.DataLoader,
) -> None:
    """
    We need to copy the model to avoid affecting the actual model during
    training (e.g. zero-ing out gradients).
    """

    exp = experiment
    gc = experiment.configs.global_config

    model_copy = copy.deepcopy(model)
    model_copy.eval()
    target_columns_gen = get_output_info_generator(outputs_as_dict=exp.outputs)

    for output_name, target_column_type, target_column_name in target_columns_gen:
        output_type = exp.outputs[output_name].output_config.output_info.output_type

        if output_type != "tabular":
            continue

        ao = get_attribution_object(
            experiment=exp,
            model=model_copy,
            column_name=target_column_name,
            output_name=output_name,
            background_loader=background_loader,
            n_background_samples=gc.attribution_background_samples,
        )

        act_callable = get_oom_adaptive_attribution_callable(
            explainer=ao.explainer,
            column_type=target_column_type,
            baseline_values=ao.baseline_values_ordered,
            baseline_names_ordered=ao.input_names_ordered,
            batch_size=gc.batch_size,
        )

        act_func = partial(
            get_attribution,
            attribution_callable=act_callable,
            input_names=ao.input_names_ordered,
        )

        data_producer = _get_interpretation_data_producer(
            experiment=exp,
            column_name=target_column_name,
            column_type=target_column_type,
            output_name=output_name,
            dataset=dataset_to_interpret,
        )

        act_producer = get_sample_attribution_producer(
            data_producer=data_producer,
            act_func=act_func,
            output_name=output_name,
            target_column_name=target_column_name,
            missing_ids_per_output=dataset_to_interpret.missing_ids_per_output,
        )

        input_names_and_types = _extract_input_names_and_types(
            input_objects=exp.inputs, input_names_ordered=ao.input_names_ordered
        )

        output_object = exp.outputs[output_name]
        assert isinstance(output_object, ComputedTabularOutputInfo)
        target_transformer = output_object.target_transformers[target_column_name]

        act_consumers = get_attribution_consumers(
            input_names_and_types=input_names_and_types,
            target_transformer=target_transformer,
            output_name=output_name,
            target_column=target_column_name,
            column_type=target_column_type,
        )

        all_attributions = process_attributions_for_all_modalities(
            attribution_producer=act_producer, attribution_consumers=act_consumers
        )

        for input_name in ao.input_names_ordered:
            input_object = exp.inputs[input_name]
            input_type = input_object.input_config.input_info.input_type

            if _do_skip_analyzing_input(
                input_name=input_name,
                input_type=input_type,
                output_configs=exp.configs.output_configs,
            ):
                continue

            act_output_folder = output_folder_target_callable(
                column_name=target_column_name,
                input_name=input_name,
                output_name=output_name,
            )
            common_kwargs = {
                "experiment": experiment,
                "input_name": input_name,
                "target_column_name": target_column_name,
                "target_column_type": target_column_type,
                "all_attributions": all_attributions[input_name],
                "attribution_outfolder": act_output_folder,
            }

            if input_type == "omics":
                analyze_omics_input_attributions(**common_kwargs)

            elif input_type == "tabular":
                analyze_tabular_input_attributions(
                    **common_kwargs, output_name=output_name
                )

            elif input_type == "sequence":
                # Currently we just run the full model here, despite being slightly
                # redundant, hence we pass in baselines_with_extras. Otherwise, we
                # would have to have a custom forward call skipping the time embeddings.
                expected_value = compute_expected_value(
                    model=model_copy,
                    baselines=ao.baselines_with_extras,
                )
                analyze_sequence_input_attributions(
                    **common_kwargs,
                    expected_target_classes_attributions=expected_value,
                    output_name=output_name,
                )
            elif input_type == "image":
                analyze_image_input_attributions(
                    **common_kwargs, output_name=output_name
                )

            elif input_type == "array":
                cur_array_attributions = all_attributions[input_name]
                assert isinstance(cur_array_attributions, dict)
                analyze_array_input_attributions(
                    attribution_outfolder=act_output_folder,
                    all_attributions=cur_array_attributions,
                )

        ao.hook_handle.remove()


def _extract_input_names_and_types(
    input_objects: "al_input_objects_as_dict",
    input_names_ordered: Sequence[str],
) -> dict[str, Literal["tabular", "omics", "sequence", "bytes", "image", "array"]]:
    input_names_and_types = {
        i: input_objects[i].input_config.input_info.input_type
        for i in input_names_ordered
        if not i.startswith("__extras_")
    }

    return input_names_and_types


def _do_skip_analyzing_input(
    input_name: str,
    input_type: str,
    output_configs: Sequence[OutputConfig],
) -> bool:
    """
    Second case is for when an input was generated for an output, e.g.
    in the case of sequence outputs.
    """
    if input_type == "bytes":
        return True

    input_name_in_output = any(
        input_name == output_config.output_info.output_name
        and output_config.output_info.output_type == "sequence"
        for output_config in output_configs
    )
    if input_name_in_output:
        return True

    return False


def compute_expected_value(
    model: nn.Module,
    baselines: Dict[str, torch.Tensor],
) -> np.ndarray:
    outputs = model(baselines)
    return outputs.mean(0).detach().cpu().numpy()


@dataclass
class AttributionObject:
    explainer: al_explainers
    hook_handle: RemovableHandle
    input_names_ordered: tuple[str, ...]
    baselines: dict[str, torch.Tensor]
    baseline_values_ordered: tuple[torch.Tensor, ...]
    baselines_with_extras: dict[str, torch.Tensor]


def get_attribution_object(
    experiment: Union[
        "Experiment",
        "PredictExperiment",
        "LoadedTrainExperimentMixedWithPredict",
    ],
    model: "al_meta_model",
    column_name: str,
    output_name: str,
    background_loader: DataLoader,
    n_background_samples: int,
) -> AttributionObject:
    """
    Note that currently we are always grabbing at least one batch of samples
    from the background loader (even when n_background_samples=0).
    """

    baseline_data, *_ = gather_data_loader_samples(
        batch_prep_hook=experiment.hooks.step_func_hooks.base_prepare_batch,
        batch_prep_hook_kwargs={"experiment": experiment},
        data_loader=background_loader,
        n_samples=n_background_samples,
    )

    baseline_data = _detach_all_inputs(tensor_inputs=baseline_data)
    baseline_data_no_extras = {
        k: v for k, v in baseline_data.items() if not k.startswith("__extras_")
    }

    hook_partial = partial(
        _grab_single_target_from_model_output_hook,
        output_target_column=column_name,
        output_name=output_name,
    )
    hook_handle = model.register_forward_hook(hook_partial)

    # Convert to list for wrapper model
    input_names, values_ordered = zip(*baseline_data_no_extras.items())
    input_names, values_ordered = tuple(input_names), tuple(values_ordered)

    assert not model.training
    wrapped_model = WrapperModelForAttribution(
        wrapped_model=model,
        input_names=input_names,
        output_name=output_name,
        column_name=column_name,
    )
    explainer = MyIntegratedGradients(forward_func=wrapped_model)

    attribution_object = AttributionObject(
        explainer=explainer,
        hook_handle=hook_handle,
        baselines=baseline_data_no_extras,
        input_names_ordered=input_names,
        baseline_values_ordered=values_ordered,
        baselines_with_extras=baseline_data,
    )

    return attribution_object


class AttributionCallable(Protocol):
    def __call__(
        self,
        *args,
        inputs: Dict[str, torch.Tensor],
        sample_label: torch.Tensor,
        **kwargs,
    ) -> list[np.ndarray]: ...


def get_attribution(
    attribution_callable: AttributionCallable,
    inputs: dict[str, torch.Tensor],
    input_names: Sequence[str],
    *args,
    **kwargs,
) -> Optional[dict[str, np.ndarray]]:
    input_attributions = attribution_callable(inputs=inputs, *args, **kwargs)

    if input_attributions is None:
        return None

    name_matched_attribution = {k: v for k, v in zip(input_names, input_attributions)}

    return name_matched_attribution


def get_oom_adaptive_attribution_callable(
    explainer: al_explainers,
    column_type: str,
    baseline_values: tuple[torch.Tensor, ...],
    baseline_names_ordered: tuple[str, ...],
    batch_size: int,
) -> AttributionCallable:
    n_steps = min(batch_size, 128)
    internal_batch_size = None
    has_successfully_run = False

    def calculate_attributions_adaptive(
        *args,
        inputs: Dict[str, torch.Tensor],
        sample_label: torch.Tensor,
        **kwargs,
    ) -> list[np.ndarray]:
        nonlocal has_successfully_run
        nonlocal internal_batch_size

        if has_successfully_run:
            return get_attributions(
                inputs=inputs,
                sample_label=sample_label,
                internal_batch_size=internal_batch_size,
                explainer=explainer,
                column_type=column_type,
                baselines=baseline_values,
                baseline_names_ordered=baseline_names_ordered,
                n_steps=n_steps,
            )
        else:
            while True:
                try:
                    res = get_attributions(
                        inputs=inputs,
                        sample_label=sample_label,
                        internal_batch_size=internal_batch_size,
                        explainer=explainer,
                        column_type=column_type,
                        baselines=baseline_values,
                        baseline_names_ordered=baseline_names_ordered,
                        n_steps=n_steps,
                    )
                    has_successfully_run = True

                    if internal_batch_size is not None:
                        logger.debug(
                            "Attribution IG internal batch size successfully set "
                            "to %d.",
                            internal_batch_size,
                        )

                    return res

                except OutOfMemoryError as e:  # type: ignore
                    if internal_batch_size == 1:
                        raise e

                    if internal_batch_size is None:
                        internal_batch_size = max(1, batch_size // 2)
                    else:
                        internal_batch_size = max(1, internal_batch_size // 2)

                    logger.debug(
                        "CUDA OOM error, reducing attribution IG "
                        "internal batch size to %d.",
                        internal_batch_size,
                    )
                    torch.cuda.empty_cache()
                    continue

    return calculate_attributions_adaptive


@dataclass
class SampleAttribution:
    sample_info: Batch
    sample_attributions: Dict[str, np.ndarray]
    raw_inputs: Dict


def get_attribution_consumers(
    input_names_and_types: dict[
        str, Literal["tabular", "omics", "sequence", "bytes", "image", "array"]
    ],
    target_transformer: "al_label_transformers_object",
    output_name: str,
    target_column: str,
    column_type: str,
) -> Dict[
    str, Union["BasicConsumerCallable", OmicsConsumerCallable, ArrayConsumerCallable]
]:
    consumers_dict = {}

    for input_name, input_type in input_names_and_types.items():
        if input_type == "bytes":
            continue

        consumer = _get_consumer_from_input_type(
            input_type=input_type,
            input_name=input_name,
            target_transformer=target_transformer,
            output_name=output_name,
            target_column=target_column,
            column_type=column_type,
        )
        if consumer is not None:
            consumers_dict[input_name] = consumer

    return consumers_dict


class BasicConsumerCallable(Protocol):
    def __call__(
        self,
        attribution: Optional["SampleAttribution"],
    ) -> Optional[Sequence["SampleAttribution"]]: ...


def _get_consumer_from_input_type(
    target_transformer: "al_label_transformers_object",
    input_type: str,
    input_name: str,
    output_name: str,
    target_column: str,
    column_type: str,
) -> BasicConsumerCallable | OmicsConsumerCallable | ArrayConsumerCallable:
    if input_type in ("sequence", "tabular", "image"):
        return get_basic_consumer()

    elif input_type == "omics":
        return get_omics_consumer(
            target_transformer=target_transformer,
            input_name=input_name,
            output_name=output_name,
            target_column=target_column,
            column_type=column_type,
        )

    elif input_type == "array":
        return get_array_sum_consumer(
            target_transformer=target_transformer,
            input_name=input_name,
            output_name=output_name,
            target_column=target_column,
            column_type=column_type,
        )
    else:
        raise ValueError(f"Unsupported input type {input_type}.")


def get_sample_attribution_producer(
    data_producer: Generator[tuple[Batch, dict[str, Any]], None, None],
    act_func: Callable,
    target_column_name: str,
    output_name: str,
    missing_ids_per_output: MissingTargetsInfo,
) -> Generator["SampleAttribution", None, None]:
    missing_for_modality = missing_ids_per_output.missing_ids_per_modality
    cur_missing_ids = missing_for_modality[output_name]

    missing_within_modality = missing_ids_per_output.missing_ids_within_modality
    cur_missing_ids_within = missing_within_modality[output_name][target_column_name]

    for batch, raw_inputs in data_producer:
        sample_target_labels = batch.target_labels
        cur_target_label = sample_target_labels[output_name][target_column_name]

        cur_id = batch.ids[0]
        if cur_id in cur_missing_ids or cur_id in cur_missing_ids_within:
            continue

        sample_all_modalities_attributions = act_func(
            inputs=batch.inputs,
            sample_label=cur_target_label,
        )

        if sample_all_modalities_attributions is None:
            continue

        batch_on_cpu = _convert_all_batch_tensors_to_cpu(batch=batch)
        cur_sample_attribution_info = SampleAttribution(
            sample_info=batch_on_cpu,
            sample_attributions=sample_all_modalities_attributions,
            raw_inputs=raw_inputs,
        )
        yield cur_sample_attribution_info


def get_basic_consumer() -> BasicConsumerCallable:
    results: list["SampleAttribution"] = []

    def _consumer(
        attribution: Optional["SampleAttribution"],
    ) -> Optional[Sequence["SampleAttribution"]]:
        if attribution is None:
            return results

        results.append(attribution)
        return None

    return _consumer


def process_attributions_for_all_modalities(
    attribution_producer: Generator["SampleAttribution", None, None],
    attribution_consumers: Dict[str, Callable],
) -> dict[str, Sequence["SampleAttribution"] | dict[str, np.ndarray]]:
    processed_attributions_all_modalities = {}

    for sample_attribution in attribution_producer:
        for consumer_name, consumer in attribution_consumers.items():
            parsed_act = _parse_out_non_target_modality_attributions(
                sample_attribution=sample_attribution, modality_key=consumer_name
            )
            consumer(parsed_act)

    for consumer_name, consumer in attribution_consumers.items():
        processed_attributions_all_modalities[consumer_name] = consumer(None)

    return processed_attributions_all_modalities


def _parse_out_non_target_modality_attributions(
    sample_attribution: SampleAttribution, modality_key: str
) -> SampleAttribution:
    parsed_act = {modality_key: sample_attribution.sample_attributions[modality_key]}
    parsed_inp = {modality_key: sample_attribution.raw_inputs[modality_key]}

    batch = sample_attribution.sample_info

    parsed_batch_inputs = {modality_key: batch.inputs[modality_key]}

    new_batch = Batch(
        inputs=parsed_batch_inputs, target_labels=batch.target_labels, ids=batch.ids
    )

    parsed_attribution = SampleAttribution(
        sample_info=new_batch, sample_attributions=parsed_act, raw_inputs=parsed_inp
    )

    return parsed_attribution


def _convert_all_batch_tensors_to_cpu(batch: Batch) -> Batch:
    """
    We need this to avoid blowing up GPU memory when gathering all the attributions and
    raw inputs, as the inputs are tensors which can be on the GPU.

    If needed later maybe we can use some fancy recursion here, but this works for now.
    """

    new_inputs = {k: v.cpu() for k, v in batch.inputs.items()}

    new_target_labels: Dict[str, Dict[str, torch.Tensor]] = {}
    for output_name, target_labels in batch.target_labels.items():
        target_labels_on_cpu = {k: v.cpu() for k, v in target_labels.items()}
        new_target_labels[output_name] = target_labels_on_cpu

    new_batch = Batch(inputs=new_inputs, target_labels=new_target_labels, ids=batch.ids)

    return new_batch


def _prepare_eval_attribution_outfolder(
    output_folder: str,
    input_name: str,
    column_name: str,
    output_name: str,
    iteration: int,
    *args,
    **kwargs,
):
    sample_outfolder = prepare_sample_output_folder(
        output_folder=output_folder,
        column_name=column_name,
        output_name=output_name,
        iteration=iteration,
    )
    attribution_outfolder = sample_outfolder / "attributions" / input_name
    ensure_path_exists(path=attribution_outfolder, is_folder=True)

    return attribution_outfolder


def _grab_single_target_from_model_output_hook(
    self: Union[CNNModel, LinearModel],
    input_: torch.Tensor,
    output: dict[str, dict[str, torch.Tensor]],
    output_target_column: str,
    output_name: str,
) -> torch.Tensor:
    return output[output_name][output_target_column]


def get_attributions(
    explainer: al_explainers,
    inputs: Dict[str, torch.Tensor],
    sample_label: torch.Tensor,
    column_type: str,
    baselines: tuple[torch.Tensor, ...],
    baseline_names_ordered: tuple[str, ...],
    n_steps: int,
    internal_batch_size: Union[int, None],
) -> list[np.ndarray]:

    tuple_inputs = tuple(inputs[n] for n in baseline_names_ordered)
    baselines_averaged = tuple(i.mean(0).unsqueeze(0) for i in baselines)

    common_kwargs = dict(
        inputs=tuple_inputs,
        baselines=baselines_averaged,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
    )

    if column_type == "con":
        output = explainer.attribute(**common_kwargs)
    else:
        assert sample_label >= 0, f"Got {sample_label} for {inputs} and {column_type}."

        output = explainer.attribute(target=sample_label, **common_kwargs)

    output = [o.detach().cpu().numpy() for o in output]

    if column_type == "con":
        assert isinstance(output[0], np.ndarray)
        return output

    return output


def _get_interpretation_data_producer(
    experiment: Union[
        "Experiment", "PredictExperiment", "LoadedTrainExperimentMixedWithPredict"
    ],
    column_name: str,
    column_type: str,
    output_name: str,
    dataset: al_datasets,
) -> Generator[Tuple[Batch, dict[str, Any]], None, None]:
    output_object = experiment.outputs[output_name]
    assert isinstance(output_object, ComputedTabularOutputInfo)

    target_transformer = output_object.target_transformers[column_name]
    gc = experiment.configs.global_config

    target_classes_numerical = _get_numerical_target_classes(
        target_transformer=target_transformer,
        column_type=column_type,
    )

    attributions_data_loader = _get_attributions_dataloader(
        dataset=dataset,
        max_attributions_per_class=gc.max_attributions_per_class,
        output_name=output_name,
        target_column=column_name,
        column_type=column_type,
        target_classes_numerical=target_classes_numerical,
    )

    for loader_batch in attributions_data_loader:
        state = call_hooks_stage_iterable(
            hook_iterable=experiment.hooks.step_func_hooks.base_prepare_batch,
            common_kwargs={"experiment": experiment, "loader_batch": loader_batch},
            state=None,
        )

        batch = state["batch"]

        inputs_detached = _detach_all_inputs(tensor_inputs=batch.inputs)
        batch_interpretation = Batch(
            inputs=inputs_detached,
            target_labels=batch.target_labels,
            ids=batch.ids,
        )

        raw_inputs = {k: v for k, v in loader_batch[0].items()}

        yield batch_interpretation, raw_inputs


def _detach_all_inputs(tensor_inputs: Dict[str, torch.Tensor]):
    inputs_detached = {}

    for input_name, value in tensor_inputs.items():
        inputs_detached[input_name] = value.detach()

    return inputs_detached


def _get_attributions_dataloader(
    dataset: al_datasets,
    max_attributions_per_class: Optional[int],
    output_name: str,
    target_column: str,
    column_type: str,
    target_classes_numerical: Sequence[int],
) -> DataLoader:
    if max_attributions_per_class is None:
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        return data_loader

    indices_func = _get_categorical_sample_indices_for_attributions
    if column_type == "con":
        indices_func = _get_continuous_sample_indices_for_attributions

    subset_indices = indices_func(
        dataset=dataset,
        max_attributions_per_class=max_attributions_per_class,
        target_column=target_column,
        output_name=output_name,
        target_classes_numerical=target_classes_numerical,
    )
    subset_dataset = _subsample_dataset(dataset=dataset, indices=subset_indices)
    data_loader = DataLoader(dataset=subset_dataset, batch_size=1, shuffle=False)
    return data_loader


def _get_categorical_sample_indices_for_attributions(
    dataset: al_datasets,
    max_attributions_per_class: int,
    output_name: str,
    target_column: str,
    target_classes_numerical: Sequence[int],
) -> Tuple[int, ...]:
    acc_label_counts: DefaultDict[int | float | torch.Tensor, int] = defaultdict(
        lambda: 0
    )
    acc_label_limit = max_attributions_per_class
    indices = []

    for index, sample in enumerate(dataset.samples):
        target_labels = sample.target_labels
        cur_sample_target_label = target_labels[output_name][target_column]

        is_over_limit = acc_label_counts[cur_sample_target_label] == acc_label_limit
        is_not_in_target_classes = (
            cur_sample_target_label not in target_classes_numerical
        )
        if is_over_limit or is_not_in_target_classes:
            continue

        indices.append(index)
        acc_label_counts[cur_sample_target_label] += 1

    return tuple(indices)


def _get_continuous_sample_indices_for_attributions(
    dataset: al_datasets, max_attributions_per_class: int, *args, **kwargs
) -> Tuple[int, ...]:
    acc_label_limit = max_attributions_per_class
    num_sample = len(dataset)
    indices = np.random.choice(num_sample, acc_label_limit)

    return tuple(indices)


def _subsample_dataset(dataset: al_datasets, indices: Sequence[int]):
    dataset_subset = Subset(dataset=dataset, indices=indices)
    return dataset_subset


def _get_numerical_target_classes(
    target_transformer,
    column_type: str,
):
    if column_type == "con":
        return [None]

    return target_transformer.transform(target_transformer.classes_)
