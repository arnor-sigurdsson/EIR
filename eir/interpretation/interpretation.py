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
    List,
    TYPE_CHECKING,
    Sequence,
    Iterable,
    Any,
    Tuple,
    Generator,
    Protocol,
    Callable,
)

# Filter warnings from shap
# TODO: Possibly catch some of these these and log them?
warnings.filterwarnings(
    "ignore",
    "Using a non-full backward hook when the forward contains multiple autograd Nodes "
    "is deprecated and will be removed in future versions.",
)

import numpy as np
import torch
from aislib.misc_utils import get_logger, ensure_path_exists
from ignite.engine import Engine
from shap import DeepExplainer
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.hooks import RemovableHandle

from eir.data_load.data_utils import get_target_columns_generator, Batch
from eir.data_load.datasets import al_datasets
from eir.interpretation.interpret_omics import (
    analyze_omics_input_activations,
    get_omics_consumer,
    ParsedOmicsActivations,
)
from eir.interpretation.interpret_tabular import (
    analyze_tabular_input_activations,
)
from eir.interpretation.interpret_sequence import analyze_sequence_input_activations
from eir.models.model_training_utils import gather_dloader_samples
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
    from eir.predict import LoadedTrainExperiment
    from eir.data_load.label_setup import al_label_transformers_object

logger = get_logger(name=__name__, tqdm_compatible=True)

# Type aliases
# would be better to use Tuple here, but shap does literal type check for list, i.e.
# if type(data) == list:
al_model_inputs = List[Union[torch.Tensor, Union[torch.Tensor, None]]]


class WrapperModelForSHAP(nn.Module):
    """
    We need this wrapper module because SHAP only handles torch.Tensor or
    List[torch.Tensor] inputs (literally checks for list). However, we do not want to
    restrict our modules to only accept those formats, rather using a dict. Hence
    we use this module to accept the list SHAP expects, but call the wrapped model with
    a matched dict.
    """

    def __init__(self, wrapped_model, input_names: Iterable[str], *args, **kwargs):
        super().__init__()
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
def suppress_stdout() -> None:
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


@validate_handler_dependencies([validation_handler])
def activation_analysis_handler(
    engine: Engine, handler_config: "HandlerConfig"
) -> None:

    exp = handler_config.experiment
    gc = exp.configs.global_config
    iteration = engine.state.iteration

    activation_outfolder_callable = partial(
        _prepare_eval_activation_outfolder,
        run_name=gc.run_name,
        iteration=iteration,
    )

    background_loader = get_background_loader(experiment=exp)

    activation_analysis_wrapper(
        model=exp.model,
        experiment=exp,
        outfolder_target_callable=activation_outfolder_callable,
        dataset_to_interpret=exp.valid_dataset,
        background_loader=background_loader,
    )


def get_background_loader(experiment: "Experiment") -> torch.utils.data.DataLoader:
    background_loader = copy.deepcopy(experiment.train_loader)

    return background_loader


def activation_analysis_wrapper(
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
    target_columns_gen = get_target_columns_generator(target_columns=exp.target_columns)

    for target_column_type, target_column_name in target_columns_gen:

        explainer, hook_handle = get_shap_object(
            experiment=exp,
            model=model_copy,
            column_name=target_column_name,
            background_loader=background_loader,
            n_background_samples=gc.act_background_samples,
        )

        input_names = explainer.explainer.model.input_names

        act_callable = get_shap_activation_callable(
            explainer=explainer,
            column_type=target_column_type,
            act_samples_per_class_limit=gc.max_acts_per_class,
        )

        act_func = partial(
            get_activation,
            activation_callable=act_callable,
            input_names=input_names,
        )

        data_producer = _get_interpretation_data_producer(
            experiment=exp,
            column_name=target_column_name,
            column_type=target_column_type,
            dataset=dataset_to_interpret,
        )

        act_producer = get_sample_activation_producer(
            data_producer=data_producer,
            act_func=act_func,
            target_column_name=target_column_name,
        )
        act_consumers = get_activation_consumers(
            input_names=input_names,
            target_transformer=exp.target_transformers[target_column_name],
            target_column=target_column_name,
            column_type=target_column_type,
        )
        all_activations = process_activations_for_all_modalities(
            activation_producer=act_producer, activation_consumers=act_consumers
        )

        for input_name in input_names:

            activation_outfolder = outfolder_target_callable(
                column_name=target_column_name, input_name=input_name
            )
            common_kwargs = {
                "experiment": experiment,
                "input_name": input_name,
                "target_column_name": target_column_name,
                "target_column_type": target_column_type,
                "all_activations": all_activations[input_name],
                "activation_outfolder": activation_outfolder,
            }

            if input_name.startswith("omics_"):
                analyze_omics_input_activations(**common_kwargs)

            elif input_name.startswith("tabular_"):
                analyze_tabular_input_activations(**common_kwargs)

            elif input_name.startswith("sequence_"):
                analyze_sequence_input_activations(
                    **common_kwargs,
                    expected_target_classes_shap_values=explainer.expected_value,
                )

        hook_handle.remove()


def get_shap_object(
    experiment: "Experiment",
    model: nn.Module,
    column_name: str,
    background_loader: DataLoader,
    n_background_samples: int,
) -> Tuple[DeepExplainer, RemovableHandle]:
    background, *_ = gather_dloader_samples(
        batch_prep_hook=experiment.hooks.step_func_hooks.base_prepare_batch,
        batch_prep_hook_kwargs={"experiment": experiment},
        data_loader=background_loader,
        n_samples=n_background_samples,
    )

    background = _detach_all_inputs(tensor_inputs=background)

    hook_partial = partial(
        _grab_single_target_from_model_output_hook, output_target_column=column_name
    )
    hook_handle = model.register_forward_hook(hook_partial)

    # Convert to list for wrapper model
    input_names, input_values = zip(*background.items())
    input_names, input_values = list(input_names), list(input_values)

    wrapped_model = WrapperModelForSHAP(wrapped_model=model, input_names=input_names)
    explainer = DeepExplainer(model=wrapped_model, data=input_values)

    return explainer, hook_handle


class ActivationCallable(Protocol):
    def __call__(
        self, inputs: Sequence[torch.Tensor], *args, **kwargs
    ) -> Sequence[torch.Tensor]:
        ...


def get_activation(
    activation_callable: ActivationCallable,
    inputs: Sequence[torch.Tensor],
    input_names: Sequence[torch.Tensor],
    *args,
    **kwargs
) -> Union[Dict[str, torch.Tensor], None]:

    input_activations = activation_callable(inputs=inputs, *args, **kwargs)

    if input_activations is None:
        return None

    name_matched_activation = {k: v for k, v in zip(input_names, input_activations)}

    return name_matched_activation


def get_shap_activation_callable(
    explainer: DeepExplainer,
    column_type: str,
    act_samples_per_class_limit: Union[int, None],
) -> Callable[
    [DeepExplainer, al_model_inputs, torch.Tensor, str], Union[None, np.ndarray]
]:
    assert column_type in ("cat", "con")
    act_sample_func = get_shap_sample_acts_deep_correct_only

    if act_samples_per_class_limit is not None:
        act_sample_func = get_shap_sample_acts_deep_all_classes

    act_func_partial = partial(
        act_sample_func,
        explainer=explainer,
        column_type=column_type,
    )
    return act_func_partial


@dataclass
class SampleActivation:
    sample_info: "Batch"
    sample_activations: Dict[str, np.ndarray]
    raw_inputs: Dict


def accumulate_all_activations(
    data_producer: Iterable["Batch"], act_func: Callable, target_column_name: str
) -> Sequence["SampleActivation"]:

    all_activations = []

    for batch, raw_inputs in data_producer:
        sample_target_labels = batch.target_labels

        sample_all_modalities_activations = act_func(
            inputs=batch.inputs, sample_label=sample_target_labels[target_column_name]
        )

        if sample_all_modalities_activations is None:
            continue

        batch_on_cpu = _convert_all_batch_tensors_to_cpu(batch=batch)
        cur_sample_activation_info = SampleActivation(
            sample_info=batch_on_cpu,
            sample_activations=sample_all_modalities_activations,
            raw_inputs=raw_inputs,
        )
        all_activations.append(cur_sample_activation_info)

    return all_activations


def get_activation_consumers(
    input_names: Sequence[str],
    target_transformer: "al_label_transformers_object",
    target_column: str,
    column_type: str,
) -> Dict[str, Callable[[Union["SampleActivation", None]], Any]]:
    consumers_dict = {}

    for name in input_names:
        consumers_dict[name] = _get_consumer_from_input_name(
            input_name=name,
            target_transformer=target_transformer,
            target_column=target_column,
            column_type=column_type,
        )

    return consumers_dict


def _get_consumer_from_input_name(
    target_transformer: "al_label_transformers_object",
    input_name: str,
    target_column: str,
    column_type: str,
) -> Callable[
    [Union["SampleActivation", None]],
    Union[Sequence["SampleActivation"], ParsedOmicsActivations],
]:
    if input_name.startswith("sequence_") or input_name.startswith("tabular_"):
        return get_basic_sequence_consumer()
    elif input_name.startswith("omics_"):
        return get_omics_consumer(
            target_transformer=target_transformer,
            input_name=input_name,
            target_column=target_column,
            column_type=column_type,
        )


def get_sample_activation_producer(
    data_producer: Iterable["Batch"], act_func: Callable, target_column_name: str
):
    for batch, raw_inputs in data_producer:
        sample_target_labels = batch.target_labels

        sample_all_modalities_activations = act_func(
            inputs=batch.inputs, sample_label=sample_target_labels[target_column_name]
        )

        if sample_all_modalities_activations is None:
            continue

        batch_on_cpu = _convert_all_batch_tensors_to_cpu(batch=batch)
        cur_sample_activation_info = SampleActivation(
            sample_info=batch_on_cpu,
            sample_activations=sample_all_modalities_activations,
            raw_inputs=raw_inputs,
        )
        yield cur_sample_activation_info


def get_basic_sequence_consumer() -> Callable[
    [Union["SampleActivation", None]], Sequence["SampleActivation"]
]:

    results = []

    def _consumer(
        activation: Union["SampleActivation", None]
    ) -> Sequence["SampleActivation"]:

        if activation is None:
            return results

        results.append(activation)

    return _consumer


def process_activations_for_all_modalities(
    activation_producer: Generator["SampleActivation", None, None],
    activation_consumers: Dict[str, Callable],
) -> Dict[str, Union[Sequence["SampleActivation"]]]:
    processed_activations_all_modalities = {}

    for sample_activation in activation_producer:
        for consumer_name, consumer in activation_consumers.items():
            parsed_act = _parse_out_non_target_modality_activations(
                sample_activation=sample_activation, modality_key=consumer_name
            )
            consumer(parsed_act)

    for consumer_name, consumer in activation_consumers.items():
        processed_activations_all_modalities[consumer_name] = consumer(None)

    return processed_activations_all_modalities


def _parse_out_non_target_modality_activations(
    sample_activation: SampleActivation, modality_key: str
) -> SampleActivation:
    parsed_act = {modality_key: sample_activation.sample_activations[modality_key]}
    parsed_inp = {modality_key: sample_activation.raw_inputs[modality_key]}

    batch = sample_activation.sample_info

    parsed_batch_inputs = {modality_key: batch.inputs[modality_key]}

    new_batch = Batch(
        inputs=parsed_batch_inputs, target_labels=batch.target_labels, ids=batch.ids
    )

    parsed_activation = SampleActivation(
        sample_info=new_batch, sample_activations=parsed_act, raw_inputs=parsed_inp
    )

    return parsed_activation


def _convert_all_batch_tensors_to_cpu(batch: Batch) -> Batch:
    """
    We need this to avoid blowing up GPU memory when gathering all the activations and
    raw inputs, as the inputs are tensors which can be on the GPU.

    If needed later maybe we can use some fancy recursion here, but this works for now.
    """
    new_batch_kwargs = {}

    new_inputs = {k: v.cpu() for k, v in batch.inputs.items()}
    new_target_labels = {k: v.cpu() for k, v in batch.target_labels.items()}

    new_batch_kwargs["inputs"] = new_inputs
    new_batch_kwargs["target_labels"] = new_target_labels
    new_batch_kwargs["ids"] = batch.ids

    new_batch = Batch(**new_batch_kwargs)

    return new_batch


def _prepare_eval_activation_outfolder(
    run_name: str, input_name: str, column_name: str, iteration: int, *args, **kwargs
):
    sample_outfolder = prep_sample_outfolder(
        run_name=run_name, column_name=column_name, iteration=iteration
    )
    activation_outfolder = sample_outfolder / "activations" / input_name
    ensure_path_exists(path=activation_outfolder, is_folder=True)

    return activation_outfolder


def _grab_single_target_from_model_output_hook(
    self: Union[CNNModel, LinearModel],
    input_: torch.Tensor,
    output: Dict[str, torch.Tensor],
    output_target_column: str,
) -> torch.Tensor:
    return output[output_target_column]


def get_shap_sample_acts_deep_correct_only(
    explainer: DeepExplainer,
    inputs: al_model_inputs,
    sample_label: torch.Tensor,
    column_type: str,
) -> Union[np.ndarray, None]:
    """
    Note: We only get the grads for a correct prediction.

    Note: We need the [0] as ranked_outputs gives us a list of lists, where the inner
          is containing all the modalities.

    TODO: Add functionality to use ranked_outputs or all outputs.
    """
    with suppress_stdout():
        explained_input_order = explainer.explainer.model.input_names
        list_inputs = [inputs[n] for n in explained_input_order]
        output = explainer.shap_values(list_inputs, ranked_outputs=1)

    if column_type == "con":
        assert isinstance(output[0], np.ndarray)
        return output

    assert len(output) == 2
    shap_grads, pred_label = output
    if pred_label.item() == sample_label.item():
        return shap_grads[0]

    return None


def get_shap_sample_acts_deep_all_classes(
    explainer: DeepExplainer,
    inputs: al_model_inputs,
    sample_label: torch.Tensor,
    column_type: str,
) -> np.ndarray:
    with suppress_stdout():
        explained_input_order = explainer.explainer.model.input_names
        list_inputs = [inputs[n] for n in explained_input_order]
        output = explainer.shap_values(list_inputs)

    if column_type == "con":
        assert isinstance(output[0], np.ndarray)
        return output

    shap_grads = output[sample_label.item()]
    return shap_grads


def _get_interpretation_data_producer(
    experiment: "Experiment",
    column_name: str,
    column_type: str,
    dataset: al_datasets,
) -> Generator["Batch", None, None]:

    target_transformer = experiment.target_transformers[column_name]
    gc = experiment.configs.global_config

    target_classes_numerical = _get_numerical_target_classes(
        target_transformer=target_transformer,
        column_type=column_type,
        act_classes=gc.act_classes,
    )

    activations_data_loader = _get_activations_dataloader(
        dataset=dataset,
        max_acts_per_class=gc.max_acts_per_class,
        target_column=column_name,
        column_type=column_type,
        target_classes_numerical=target_classes_numerical,
    )

    for loader_batch in activations_data_loader:
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


def _get_activations_dataloader(
    dataset: al_datasets,
    max_acts_per_class: int,
    target_column: str,
    column_type: str,
    target_classes_numerical: Sequence[int],
) -> DataLoader:
    common_args = {"batch_size": 1, "shuffle": False}

    if max_acts_per_class is None:
        data_loader = DataLoader(dataset=dataset, **common_args)
        return data_loader

    indices_func = _get_categorical_sample_indices_for_activations
    if column_type == "con":
        indices_func = _get_continuous_sample_indices_for_activations

    subset_indices = indices_func(
        dataset=dataset,
        max_acts_per_class=max_acts_per_class,
        target_column=target_column,
        target_classes_numerical=target_classes_numerical,
    )
    subset_dataset = _subsample_dataset(dataset=dataset, indices=subset_indices)
    data_loader = DataLoader(dataset=subset_dataset, **common_args)
    return data_loader


def _get_categorical_sample_indices_for_activations(
    dataset: al_datasets,
    max_acts_per_class: int,
    target_column: str,
    target_classes_numerical: Sequence[int],
) -> Tuple[int, ...]:
    acc_label_counts = defaultdict(lambda: 0)
    acc_label_limit = max_acts_per_class
    indices = []

    for index, sample in enumerate(dataset.samples):
        target_labels = sample.target_labels
        cur_sample_target_label = target_labels[target_column]

        is_over_limit = acc_label_counts[cur_sample_target_label] == acc_label_limit
        is_not_in_target_classes = (
            cur_sample_target_label not in target_classes_numerical
        )
        if is_over_limit or is_not_in_target_classes:
            continue

        indices.append(index)
        acc_label_counts[cur_sample_target_label] += 1

    return tuple(indices)


def _get_continuous_sample_indices_for_activations(
    dataset: al_datasets, max_acts_per_class: int, *args, **kwargs
) -> Tuple[int, ...]:

    acc_label_limit = max_acts_per_class
    num_sample = len(dataset)
    indices = np.random.choice(num_sample, acc_label_limit)

    return tuple(indices)


def _subsample_dataset(dataset: al_datasets, indices: Sequence[int]):
    dataset_subset = Subset(dataset=dataset, indices=indices)
    return dataset_subset


def _get_numerical_target_classes(
    target_transformer, column_type: str, act_classes: Union[Sequence[str], None]
):
    if column_type == "con":
        return [None]

    if act_classes:
        return target_transformer.transform(act_classes)

    return target_transformer.transform(target_transformer.classes_)
