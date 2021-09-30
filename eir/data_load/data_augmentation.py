from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Union, Dict, Callable, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from eir.data_load.data_utils import Batch
from eir.data_load.label_setup import al_target_columns

if TYPE_CHECKING:
    from eir.train import al_training_labels_target, al_criterions, Experiment

al_target_values = Union[torch.LongTensor, torch.Tensor]
al_int_tensors = Union[
    torch.ByteTensor,
    torch.CharTensor,
    torch.ShortTensor,
    torch.IntTensor,
    torch.LongTensor,
    torch.BoolTensor,
]


@dataclass
class MixingObject:
    targets: "al_training_labels_target"
    targets_permuted: "al_training_labels_target"
    lambda_: float
    permuted_indexes: al_int_tensors


def get_mix_data_hook(mixing_type: str):
    mixing_func_mapping = _get_mixing_function_map()
    mixing_func = mixing_func_mapping.get(mixing_type)

    bound_hook = partial(hook_default_mix_data, mixing_func=mixing_func)

    return bound_hook


def hook_default_mix_data(
    experiment: "Experiment", state: Dict, mixing_func: Callable, *args, **kwargs
) -> Dict:

    gc = experiment.configs.global_config

    batch = state["batch"]

    mixing_info = get_mixing_info(
        mixing_alpha=gc.mixing_alpha,
        batch_size=gc.batch_size,
        target_labels=batch.target_labels,
        target_columns=experiment.target_columns,
    )

    mixed_inputs = {}

    for input_name, input_data in batch.inputs.items():
        if input_name.startswith("omics_"):

            mixed_omics = mixup_omics_data(
                inputs=input_data,
                mixing_func=mixing_func,
                mixing_info=mixing_info,
            )
            mixed_inputs[input_name] = mixed_omics

        elif input_name.startswith("tabular_"):

            mixed_tabular_input_tensor = mixup_tensor(
                tensor=input_data,
                lambda_=mixing_info.lambda_,
                random_batch_indices_to_mix=mixing_info.permuted_indexes,
            )
            mixed_inputs[input_name] = mixed_tabular_input_tensor

        elif input_name.startswith("sequence_"):
            mixed_sequence_embeddings = mixup_tensor(
                tensor=input_data,
                lambda_=mixing_info.lambda_,
                random_batch_indices_to_mix=mixing_info.permuted_indexes,
            )
            mixed_inputs[input_name] = mixed_sequence_embeddings

    batch_mixed = Batch(
        inputs=mixed_inputs,
        target_labels=batch.target_labels,
        ids=batch.ids,
    )

    state_updates = {
        "batch": batch_mixed,
        "mixing_info": mixing_info,
    }

    return state_updates


def get_mixing_info(
    mixing_alpha: float,
    batch_size: int,
    target_labels: "al_training_labels_target",
    target_columns: al_target_columns,
) -> MixingObject:
    lambda_ = _sample_lambda(mixing_alpha=mixing_alpha)

    permuted_indexes = get_random_batch_indices_to_mix(batch_size=batch_size)
    targets_permuted = mixup_all_targets(
        targets=target_labels,
        random_index_for_mixing=permuted_indexes,
        target_columns=target_columns,
    )

    mixing_info = MixingObject(
        targets=target_labels,
        targets_permuted=targets_permuted,
        lambda_=lambda_,
        permuted_indexes=permuted_indexes,
    )

    return mixing_info


def _sample_lambda(mixing_alpha: float) -> float:
    if mixing_alpha > 0:
        beta_object = torch.distributions.beta.Beta(mixing_alpha, mixing_alpha)
        lambda_ = beta_object.sample().item()
    else:
        lambda_ = 1.0

    return lambda_


def hook_mix_loss(experiment: "Experiment", state: Dict, *args, **kwargs) -> Dict:

    mixed_losses = calc_all_mixed_losses(
        target_columns=experiment.target_columns,
        criterions=experiment.criterions,
        outputs=state["model_outputs"],
        mixed_object=state["mixing_info"],
    )

    state_updates = {"per_target_train_losses": mixed_losses}

    return state_updates


def _get_mixing_function_map():
    mapping = {
        "cutmix-uniform": uniform_cutmix_omics_input,
        "cutmix-block": block_cutmix_omics_input,
        "mixup": mixup_input,
    }
    return mapping


def mixup_omics_data(
    inputs: torch.Tensor,
    mixing_func: Callable[[torch.Tensor, float, torch.Tensor], torch.Tensor],
    mixing_info: MixingObject,
) -> torch.Tensor:
    """
    NOTE: **This function will modify the inputs in-place**

    The original inputs (inputs) will be lost when calling this function, unless
    they have been explicitly copied and stored in another variable before this call.

    This is because we do not want to clone the input tensor in this (or any functions
    called within this) function as it might mean a large memory overhead if the inputs
    are large.

    An exception is when we use the "vanilla" MixUp, as that calculates a new tensor
    instead of cut-pasting inside an already existing tensor.
    """
    assert inputs.dim() == 4, "Should be called with 4 dimensions."

    mixed_inputs = mixing_func(
        input_batch=inputs,
        lambda_=mixing_info.lambda_,
        random_batch_indices_to_mix=mixing_info.permuted_indexes,
    )

    return mixed_inputs


def get_random_batch_indices_to_mix(batch_size: int) -> al_int_tensors:
    return torch.randperm(batch_size).to(dtype=torch.long)


def mixup_input(
    input_batch: torch.Tensor,
    lambda_: float,
    random_batch_indices_to_mix: al_int_tensors,
) -> torch.Tensor:
    """
    This function is to delegate arguments from mixup_snp_data to a general mixup
    function that is does not necessarily have an 'input_batch' argument.
    """
    mixed_input = mixup_tensor(
        tensor=input_batch,
        lambda_=lambda_,
        random_batch_indices_to_mix=random_batch_indices_to_mix,
    )

    return mixed_input


def mixup_tensor(
    tensor: torch.Tensor,
    lambda_: float,
    random_batch_indices_to_mix: Sequence[al_int_tensors],
) -> torch.Tensor:

    mixed_tensor = (
        lambda_ * tensor + (1.0 - lambda_) * tensor[random_batch_indices_to_mix, :]
    )
    return mixed_tensor


def block_cutmix_omics_input(
    input_batch: torch.Tensor, lambda_: float, random_batch_indices_to_mix: torch.Tensor
) -> torch.Tensor:

    cut_start, cut_end = get_block_cutmix_indices(
        input_length=input_batch.shape[-1], lambda_=lambda_
    )
    target_to_cut = input_batch[random_batch_indices_to_mix, :]
    cut_part = target_to_cut[..., cut_start:cut_end]

    # Caution: input_ will be modified as well since no .clone() below here
    cutmixed_x = input_batch
    cutmixed_x[..., cut_start:cut_end] = cut_part

    return cutmixed_x


def get_block_cutmix_indices(input_length: int, lambda_: float) -> Tuple[int, int]:
    mixin_coefficient = 1.0 - lambda_
    num_snps_to_mix = int(round(input_length * mixin_coefficient))
    random_index_start = np.random.choice(max(1, input_length - num_snps_to_mix))
    random_index_end = random_index_start + num_snps_to_mix
    return random_index_start, random_index_end


def uniform_cutmix_omics_input(
    input_batch: torch.Tensor,
    lambda_: float,
    random_batch_indices_to_mix: torch.Tensor,
) -> torch.Tensor:

    target_to_mix = input_batch[random_batch_indices_to_mix, :]

    random_snp_indices_to_mix = get_uniform_cutmix_indices(
        input_length=input_batch.shape[-1], lambda_=lambda_
    )
    cut_part = target_to_mix[..., random_snp_indices_to_mix]

    # Caution: input_ will be modified as well since no .clone() below here
    cutmixed_x = input_batch
    cutmixed_x[..., random_snp_indices_to_mix] = cut_part

    return cutmixed_x


def get_uniform_cutmix_indices(input_length: int, lambda_: float) -> torch.Tensor:
    mixin_coefficient = 1.0 - lambda_
    num_snps_to_mix = int(round(input_length * mixin_coefficient))
    random_to_mix = np.random.choice(input_length, num_snps_to_mix, replace=False)
    random_to_mix = torch.tensor(random_to_mix, dtype=torch.long)

    return random_to_mix


def mixup_all_targets(
    targets: "al_training_labels_target",
    random_index_for_mixing: al_int_tensors,
    target_columns: al_target_columns,
) -> "al_training_labels_target":
    targets_permuted = copy(targets)

    all_target_cols = target_columns["cat"] + target_columns["con"]
    for target_col in all_target_cols:

        cur_targets = targets_permuted[target_col]
        cur_targets_permuted = mixup_targets(
            targets=cur_targets, random_index_for_mixing=random_index_for_mixing
        )
        targets_permuted[target_col] = cur_targets_permuted

    return targets_permuted


def mixup_targets(
    targets: al_target_values,
    random_index_for_mixing: torch.Tensor,
) -> al_target_values:

    targets_permuted = copy(targets)
    targets_permuted = targets_permuted[random_index_for_mixing]

    return targets_permuted


def calc_all_mixed_losses(
    target_columns: al_target_columns,
    criterions: "al_criterions",
    outputs: Dict[str, torch.Tensor],
    mixed_object: MixingObject,
):

    losses = {}

    all_target_columns = target_columns["con"] + target_columns["cat"]
    for target_col in all_target_columns:
        cur_loss = calc_mixed_loss(
            criterion=criterions[target_col],
            outputs=outputs[target_col],
            targets=mixed_object.targets[target_col],
            targets_permuted=mixed_object.targets_permuted[target_col],
            lambda_=mixed_object.lambda_,
        )
        losses[target_col] = cur_loss

    return losses


def calc_mixed_loss(
    criterion: Union[nn.CrossEntropyLoss, nn.MSELoss],
    outputs: torch.Tensor,
    targets: torch.Tensor,
    targets_permuted: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:

    base_loss = lambda_ * criterion(input=outputs, target=targets)
    permuted_loss = (1.0 - lambda_) * criterion(input=outputs, target=targets_permuted)

    total_loss = base_loss + permuted_loss

    return total_loss


def make_random_omics_columns_missing(
    omics_array: torch.Tensor, percentage: float = 0.05, probability: float = 1.0
) -> torch.Tensor:
    random_draw = np.random.uniform()
    if random_draw > probability:
        return omics_array

    n_snps = omics_array.shape[2]
    n_to_drop = int(n_snps * percentage)
    random_to_drop = torch.randperm(n_snps)[:n_to_drop].to(dtype=torch.long)

    missing_arr = torch.tensor([False, False, False, True], dtype=torch.bool).reshape(
        -1, 1
    )
    omics_array[:, :, random_to_drop] = missing_arr

    return omics_array
