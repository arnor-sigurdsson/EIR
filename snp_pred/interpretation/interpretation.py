import copy
import os
import sys
from contextlib import contextmanager
from functools import partial
from typing import Union, Dict, List, TYPE_CHECKING, Sequence, Iterable

import numpy as np
import torch
from aislib.misc_utils import get_logger, ensure_path_exists
from ignite.engine import Engine
from shap import DeepExplainer
from torch import nn
from torch.utils.data import DataLoader

from snp_pred.data_load.data_utils import get_target_columns_generator
from snp_pred.interpretation.interpret_omics import analyze_omics_activations
from snp_pred.models import model_training_utils
from snp_pred.models.model_training_utils import gather_dloader_samples
from snp_pred.models.omics.models_cnn import CNNModel
from snp_pred.models.omics.models_mlp import MLPModel
from snp_pred.train_utils.evaluation import validation_handler
from snp_pred.train_utils.utils import (
    prep_sample_outfolder,
    validate_handler_dependencies,
)

if TYPE_CHECKING:
    from snp_pred.train_utils.train_handlers import HandlerConfig
    from snp_pred.train import Config

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
    """
    We need to copy the model to avoid affecting the actual model during
    training (e.g. zero-ing out gradients).

    TODO: Refactor this function further.
    """

    c = handler_config.config
    cl_args = c.cl_args
    iteration = engine.state.iteration

    model_copy = copy.deepcopy(c.model)
    target_columns_gen = get_target_columns_generator(target_columns=c.target_columns)

    for column_type, column_name in target_columns_gen:

        no_explainer_background_samples = _get_background_samples_for_shap_object(
            batch_size=cl_args.batch_size
        )

        explainer, hook_handle = get_shap_object(
            config=c,
            model=model_copy,
            column_name=column_name,
            train_loader=c.train_loader,
            n_background_samples=no_explainer_background_samples,
        )

        proc_funcs = {
            "pre": (
                partial(
                    _pre_transform_sample_before_activation,
                    column_name=column_name,
                    device=cl_args.device,
                    target_columns=c.target_columns,
                ),
            )
        }

        act_func = _get_shap_activation_function(
            explainer=explainer,
            column_type=column_type,
            act_samples_per_class_limit=cl_args.max_acts_per_class,
        )

        # for omics input ...j

        activation_outfolder = _prepare_activation_outfolder(
            run_name=cl_args.run_name, column_name=column_name, iteration=iteration
        )

        analyze_omics_activations(
            config=c,
            act_func=act_func,
            proc_funcs=proc_funcs,
            column_name=column_name,
            column_type=column_type,
            activation_outfolder=activation_outfolder,
        )

        # for tabular input ...

        hook_handle.remove()


def _get_background_samples_for_shap_object(batch_size: int):
    no_explainer_background_samples = np.max([int(batch_size / 8), 32])
    return no_explainer_background_samples


def get_shap_object(
    config: "Config",
    model: nn.Module,
    column_name: str,
    train_loader: DataLoader,
    n_background_samples: int = 64,
):
    c = config

    background, labels, ids = gather_dloader_samples(
        batch_prep_hook=c.hooks.step_func_hooks.base_prepare_batch,
        batch_prep_hook_kwargs={"config": c},
        data_loader=train_loader,
        n_samples=n_background_samples,
    )

    if "tabular_cl_args" in background:
        background["tabular_cl_args"] = background["tabular_cl_args"].detach()

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


def _pre_transform_sample_before_activation(
    single_sample, sample_label, column_name: str, device: str, target_columns
):
    single_sample = single_sample.to(device=device, dtype=torch.float32)

    sample_label = model_training_utils.parse_target_labels(
        target_columns=target_columns, device=device, labels=sample_label
    )[column_name]

    return single_sample, sample_label


def _get_shap_activation_function(
    explainer: DeepExplainer,
    column_type: str,
    act_samples_per_class_limit: Union[int, None],
):
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


def _prepare_activation_outfolder(run_name: str, column_name: str, iteration: int):
    sample_outfolder = prep_sample_outfolder(
        run_name=run_name, column_name=column_name, iteration=iteration
    )
    activation_outfolder = sample_outfolder / "activations"
    ensure_path_exists(path=activation_outfolder, is_folder=True)

    return activation_outfolder


def _grab_single_target_from_model_output_hook(
    self: Union[CNNModel, MLPModel],
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
):
    """
    Note: We only get the grads for a correct prediction.

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
):
    with suppress_stdout():
        explained_input_order = explainer.explainer.model.input_names
        list_inputs = [inputs[n] for n in explained_input_order]
        output = explainer.shap_values(list_inputs)

    if column_type == "con":
        assert isinstance(output[0], np.ndarray)
        return output

    shap_grads = output[sample_label.item()]
    return shap_grads[0]
