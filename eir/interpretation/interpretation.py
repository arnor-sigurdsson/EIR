import copy
import os
import sys
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import (
    Union,
    Dict,
    TYPE_CHECKING,
    Sequence,
    Iterable,
    Any,
    Tuple,
    Generator,
    Protocol,
    Callable,
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
from torch.cuda import OutOfMemoryError
from aislib.misc_utils import get_logger, ensure_path_exists
from ignite.engine import Engine
from captum.attr import IntegratedGradients, NoiseTunnel
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.hooks import RemovableHandle

from eir.data_load.data_utils import get_output_info_generator, Batch
from eir.data_load.datasets import al_datasets
from eir.interpretation.interpret_omics import (
    analyze_omics_input_attributions,
    get_omics_consumer,
    ParsedOmicsAttributions,
)
from eir.setup.schemas import InputConfig
from eir.interpretation.interpret_tabular import (
    analyze_tabular_input_attributions,
)
from eir.interpretation.interpret_sequence import analyze_sequence_input_attributions
from eir.interpretation.interpret_image import analyze_image_input_attributions
from eir.models.model_training_utils import gather_data_loader_samples
from eir.models.omics.models_cnn import CNNModel
from eir.models.omics.models_linear import LinearModel
from eir.train_utils.evaluation import validation_handler
from eir.train_utils.utils import (
    prep_sample_outfolder,
    validate_handler_dependencies,
    call_hooks_stage_iterable,
)

if TYPE_CHECKING:
    from eir.train_utils.train_handlers import HandlerConfig
    from eir.train import Experiment
    from eir.experiment_io.experiment_io import LoadedTrainExperiment
    from eir.data_load.label_setup import al_label_transformers_object

logger = get_logger(name=__name__, tqdm_compatible=True)

# Type aliases
al_explainers = Union[IntegratedGradients, NoiseTunnel]


class WrapperModelForAttribution(nn.Module):
    """
    We need this wrapper module because libraries often only handle torch.Tensor or
    List[torch.Tensor] inputs. However, we do not want to restrict our modules to
    only accept those formats, rather using a dict. Hence, we use this module to
    accept the list libraries expect, but call the wrapped model with a matched dict.
    """

    def __init__(self, wrapped_model, input_names: Iterable[str], *args, **kwargs):
        super().__init__()
        assert not wrapped_model.training
        self.wrapped_model = wrapped_model
        self.input_names = input_names

    def match_tuple_inputs_and_names(
        self, input_sequence: Sequence[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(input_sequence, torch.Tensor):
            input_sequence = list(input_sequence)

        matched = {k: v for k, v in zip(self.input_names, input_sequence)}

        return matched

    def forward(self, *inputs):
        matched_inputs = self.match_tuple_inputs_and_names(input_sequence=inputs)

        return self.wrapped_model(matched_inputs)


@contextmanager
def suppress_stdout_and_stderr() -> None:
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

    attribution_outfolder_callable = partial(
        _prepare_eval_attribution_outfolder,
        output_folder=gc.output_folder,
        iteration=iteration,
    )

    background_loader = get_background_loader(experiment=exp)

    attribution_analysis_wrapper(
        model=exp.model,
        experiment=exp,
        outfolder_target_callable=attribution_outfolder_callable,
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


def attribution_analysis_wrapper(
    model: nn.Module,
    experiment: Union["Experiment", "LoadedTrainExperiment"],
    outfolder_target_callable: Callable,
    dataset_to_interpret: al_datasets,
    background_loader: torch.utils.data.DataLoader,
) -> None:
    """
    We need to copy the model to avoid affecting the actual model during
    training (e.g. zero-ing out gradients).

    TODO: Refactor this function further.
    """

    exp = experiment
    gc = experiment.configs.global_config

    model_copy = copy.deepcopy(model)
    model_copy.eval()
    target_columns_gen = get_output_info_generator(outputs_as_dict=exp.outputs)

    for output_name, target_column_type, target_column_name in target_columns_gen:
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
        )

        input_names_and_types = {
            i: exp.inputs[i].input_config.input_info.input_type
            for i in ao.input_names_ordered
        }
        output_object = exp.outputs[output_name]
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

            if input_type == "bytes":
                continue

            act_output_folder = outfolder_target_callable(
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
                expected_value = compute_expected_value(
                    model=model_copy, baselines=ao.baselines
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

        ao.hook_handle.remove()


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
    input_names_ordered: tuple[str]
    baselines: Dict[str, torch.Tensor]
    baseline_values_ordered: tuple[torch.Tensor]


def get_attribution_object(
    experiment: "Experiment",
    model: nn.Module,
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

    hook_partial = partial(
        _grab_single_target_from_model_output_hook,
        output_target_column=column_name,
        output_name=output_name,
    )
    hook_handle = model.register_forward_hook(hook_partial)

    # Convert to list for wrapper model
    input_names, values_ordered = zip(*baseline_data.items())
    input_names, values_ordered = tuple(input_names), tuple(values_ordered)

    assert not model.training
    wrapped_model = WrapperModelForAttribution(
        wrapped_model=model,
        input_names=input_names,
    )
    explainer = IntegratedGradients(forward_func=wrapped_model)

    attribution_object = AttributionObject(
        explainer=explainer,
        hook_handle=hook_handle,
        baselines=baseline_data,
        input_names_ordered=input_names,
        baseline_values_ordered=values_ordered,
    )

    return attribution_object


class AttributionCallable(Protocol):
    def __call__(
        self,
        inputs: Dict[str, torch.Tensor],
        sample_label: torch.Tensor,
        *args,
        **kwargs,
    ) -> list[np.ndarray]:
        ...


def get_attribution(
    attribution_callable: AttributionCallable,
    inputs: Dict[str, torch.Tensor],
    input_names: Sequence[torch.Tensor],
    *args,
    **kwargs,
) -> Union[Dict[str, torch.Tensor], None]:
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

    base_kwargs = dict(
        explainer=explainer,
        column_type=column_type,
        baselines=baseline_values,
        baseline_names_ordered=baseline_names_ordered,
        n_steps=n_steps,
    )

    def calculate_attributions_adaptive(
        inputs: Dict[str, torch.Tensor], sample_label: torch.Tensor
    ) -> list[np.ndarray]:
        nonlocal has_successfully_run
        nonlocal internal_batch_size

        if has_successfully_run:
            return get_attributions(
                inputs=inputs,
                sample_label=sample_label,
                internal_batch_size=internal_batch_size,
                **base_kwargs,
            )
        else:
            while True:
                try:
                    res = get_attributions(
                        inputs=inputs,
                        sample_label=sample_label,
                        internal_batch_size=internal_batch_size,
                        **base_kwargs,
                    )
                    has_successfully_run = True

                    if internal_batch_size is not None:
                        logger.debug(
                            "Attribution IG internal batch size successfully set "
                            "to %d.",
                            internal_batch_size,
                        )

                    return res

                except OutOfMemoryError as e:
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


def get_attribution_callable(
    explainer: al_explainers,
    column_type: str,
    baseline_values: tuple[torch.Tensor, ...],
    baseline_names_ordered: tuple[str, ...],
    batch_size: int,
) -> Callable[
    [al_explainers, Dict[str, torch.Tensor], torch.Tensor, str],
    Union[None, np.ndarray],
]:
    n_steps = min(batch_size, 128)

    act_func_partial = partial(
        get_attributions,
        explainer=explainer,
        column_type=column_type,
        baselines=baseline_values,
        baseline_names_ordered=baseline_names_ordered,
        n_steps=n_steps,
    )
    return act_func_partial


@dataclass
class SampleAttribution:
    sample_info: "Batch"
    sample_attributions: Dict[str, np.ndarray]
    raw_inputs: Dict


def accumulate_all_attributions(
    data_producer: Iterable["Batch"],
    act_func: Callable,
    target_column_name: str,
    output_name: str,
) -> Sequence["SampleAttribution"]:
    all_attributions = []

    for batch, raw_inputs in data_producer:
        sample_target_labels = batch.target_labels

        sample_all_modalities_attributions = act_func(
            inputs=batch.inputs,
            sample_label=sample_target_labels[output_name][target_column_name],
        )

        if sample_all_modalities_attributions is None:
            continue

        batch_on_cpu = _convert_all_batch_tensors_to_cpu(batch=batch)
        cur_sample_attribution_info = SampleAttribution(
            sample_info=batch_on_cpu,
            sample_attributions=sample_all_modalities_attributions,
            raw_inputs=raw_inputs,
        )
        all_attributions.append(cur_sample_attribution_info)

    return all_attributions


def get_attribution_consumers(
    input_names_and_types: Dict[str, str],
    target_transformer: "al_label_transformers_object",
    output_name: str,
    target_column: str,
    column_type: str,
) -> Dict[str, Callable[[Union["SampleAttribution", None]], Any]]:
    consumers_dict = {}

    for input_name, input_type in input_names_and_types.items():
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


def _get_consumer_from_input_type(
    target_transformer: "al_label_transformers_object",
    input_type: str,
    input_name: str,
    output_name: str,
    target_column: str,
    column_type: str,
) -> Callable[
    [Union["SampleAttribution", None]],
    Union[Sequence["SampleAttribution"], ParsedOmicsAttributions],
]:
    if input_type in ("sequence", "tabular", "image"):
        return get_basic_sequence_consumer()

    elif input_type == "omics":
        return get_omics_consumer(
            target_transformer=target_transformer,
            input_name=input_name,
            output_name=output_name,
            target_column=target_column,
            column_type=column_type,
        )


def get_sample_attribution_producer(
    data_producer: Iterable["Batch"],
    act_func: Callable,
    target_column_name: str,
    output_name: str,
) -> Generator["SampleAttribution", None, None]:
    for batch, raw_inputs in data_producer:
        sample_target_labels = batch.target_labels

        sample_all_modalities_attributions = act_func(
            inputs=batch.inputs,
            sample_label=sample_target_labels[output_name][target_column_name],
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


def get_basic_sequence_consumer() -> (
    Callable[[Union["SampleAttribution", None]], Sequence["SampleAttribution"]]
):
    results = []

    def _consumer(
        attribution: Union["SampleAttribution", None]
    ) -> Sequence["SampleAttribution"]:
        if attribution is None:
            return results

        results.append(attribution)

    return _consumer


def process_attributions_for_all_modalities(
    attribution_producer: Generator["SampleAttribution", None, None],
    attribution_consumers: Dict[str, Callable],
) -> Dict[str, Union[Sequence["SampleAttribution"]]]:
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
    new_batch_kwargs = {}

    new_inputs = {k: v.cpu() for k, v in batch.inputs.items()}

    new_target_labels = {}
    for output_name, output_object in batch.target_labels.items():
        target_labels_on_cpu = {k: v.cpu() for k, v in output_object.items()}
        new_target_labels[output_name] = target_labels_on_cpu

    new_batch_kwargs["inputs"] = new_inputs
    new_batch_kwargs["target_labels"] = new_target_labels
    new_batch_kwargs["ids"] = batch.ids

    new_batch = Batch(**new_batch_kwargs)

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
    sample_outfolder = prep_sample_outfolder(
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
    output: Dict[str, torch.Tensor],
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
    list_inputs = tuple(inputs[n] for n in baseline_names_ordered)
    baselines_averaged = tuple(i.mean(0).unsqueeze(0) for i in baselines)

    common_kwargs = dict(
        inputs=list_inputs,
        baselines=baselines_averaged,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
    )

    with suppress_stdout_and_stderr():
        if column_type == "con":
            output = explainer.attribute(**common_kwargs)
        else:
            output = explainer.attribute(target=sample_label, **common_kwargs)

    output = [o.detach().cpu().numpy() for o in output]

    if column_type == "con":
        assert isinstance(output[0], np.ndarray)
        return output

    return output


def _get_interpretation_data_producer(
    experiment: "Experiment",
    column_name: str,
    column_type: str,
    output_name: str,
    dataset: al_datasets,
) -> Generator["Batch", None, None]:
    cur_output = experiment.outputs[output_name]
    target_transformer = cur_output.target_transformers[column_name]
    gc = experiment.configs.global_config

    target_classes_numerical = _get_numerical_target_classes(
        target_transformer=target_transformer,
        column_type=column_type,
        attribution_target_classes=gc.attribution_target_classes,
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
            inputs=inputs_detached, target_labels=batch.target_labels, ids=batch.ids
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
    max_attributions_per_class: int,
    output_name: str,
    target_column: str,
    column_type: str,
    target_classes_numerical: Sequence[int],
) -> DataLoader:
    common_args = {"batch_size": 1, "shuffle": False}

    if max_attributions_per_class is None:
        data_loader = DataLoader(dataset=dataset, **common_args)
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
    data_loader = DataLoader(dataset=subset_dataset, **common_args)
    return data_loader


def _get_categorical_sample_indices_for_attributions(
    dataset: al_datasets,
    max_attributions_per_class: int,
    output_name: str,
    target_column: str,
    target_classes_numerical: Sequence[int],
) -> Tuple[int, ...]:
    acc_label_counts = defaultdict(lambda: 0)
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
    attribution_target_classes: Union[Sequence[str], None],
):
    if column_type == "con":
        return [None]

    if attribution_target_classes:
        return target_transformer.transform(attribution_target_classes)

    return target_transformer.transform(target_transformer.classes_)
