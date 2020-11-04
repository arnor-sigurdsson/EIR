from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Union, Dict, Callable, Sequence, Tuple

import numpy as np
import torch
from snp_pred.data_load.data_utils import Batch
from snp_pred.data_load.label_setup import al_target_columns
from torch import nn

if TYPE_CHECKING:
    from train import al_training_labels_target, al_criterions, Config

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
class MixupOutput:
    inputs: torch.Tensor
    targets: "al_training_labels_target"
    targets_permuted: "al_training_labels_target"
    lambda_: float
    permuted_indexes: Sequence[al_int_tensors]


def get_mix_data_hook(mixing_type: str):
    mixing_func_mapping = _get_mixing_function_map()
    mixing_func = mixing_func_mapping.get(mixing_type)

    bound_hook = partial(hook_mix_data, mixing_func=mixing_func)

    return bound_hook


def hook_mix_data(
    config: "Config", state: Dict, mixing_func: Callable, *args, **kwargs
) -> Dict:

    batch = state["batch"]

    mixed_object = mixup_snp_data(
        inputs=batch.inputs,
        targets=batch.target_labels,
        target_columns=config.target_columns,
        alpha=config.cl_args.mixing_alpha,
        mixing_func=mixing_func,
    )

    mixed_extra_input = batch.extra_inputs
    if batch.extra_inputs is not None:
        mixed_extra_input = mixup_tensor(
            tensor=batch.extra_inputs,
            lambda_=mixed_object.lambda_,
            random_batch_indices_to_mix=mixed_object.permuted_indexes,
        )

    batch_mixed = Batch(
        inputs=mixed_object.inputs,
        target_labels=batch.target_labels,
        extra_inputs=mixed_extra_input,
        ids=batch.ids,
    )

    state_updates = {"batch": batch_mixed, "mixed_snp_data": mixed_object}

    return state_updates


def hook_mix_loss(config: "Config", state: Dict, *args, **kwargs) -> Dict:

    mixed_losses = calc_all_mixed_losses(
        target_columns=config.target_columns,
        criterions=config.criterions,
        outputs=state["model_outputs"],
        mixed_object=state["mixed_snp_data"],
    )

    state_updates = {"per_target_train_losses": mixed_losses}

    return state_updates


def _get_mixing_function_map():
    mapping = {
        "cutmix-uniform": uniform_cutmix_input,
        "cutmix-block": block_cutmix_input,
        "mixup": mixup_input,
    }
    return mapping


def mixup_snp_data(
    inputs: torch.Tensor,
    targets: "al_training_labels_target",
    target_columns: al_target_columns,
    mixing_func: Callable[[torch.Tensor, float, torch.Tensor], torch.Tensor],
    alpha: float = 1.0,
) -> MixupOutput:
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

    if alpha > 0:
        lambda_ = np.random.beta(alpha, alpha)
    else:
        lambda_ = 1.0

    batch_size = inputs.size()[0]
    random_batch_indices_to_mix = get_random_batch_indices_to_mix(batch_size=batch_size)
    targets_permuted = mixup_all_targets(
        targets=targets,
        random_index_for_mixing=random_batch_indices_to_mix,
        target_columns=target_columns,
    )

    mixed_inputs = mixing_func(
        input_batch=inputs,
        lambda_=lambda_,
        random_batch_indices_to_mix=random_batch_indices_to_mix,
    )

    mixing_output = MixupOutput(
        inputs=mixed_inputs,
        targets=targets,
        targets_permuted=targets_permuted,
        lambda_=lambda_,
        permuted_indexes=random_batch_indices_to_mix,
    )

    return mixing_output


def get_random_batch_indices_to_mix(batch_size: int) -> torch.Tensor:
    return torch.randperm(batch_size)


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


def block_cutmix_input(
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

    assert (cutmixed_x.sum(dim=2) == 1).all()
    return cutmixed_x


def get_block_cutmix_indices(input_length: int, lambda_: float) -> Tuple[int, int]:
    mixin_coefficient = 1.0 - lambda_
    num_snps_to_mix = int(round(input_length * mixin_coefficient))
    random_index_start = np.random.choice(max(1, input_length - num_snps_to_mix))
    random_index_end = random_index_start + num_snps_to_mix
    return random_index_start, random_index_end


def uniform_cutmix_input(
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

    assert (cutmixed_x.sum(dim=2) == 1).all()
    return cutmixed_x


def get_uniform_cutmix_indices(input_length: int, lambda_: float) -> torch.Tensor:
    mixin_coefficient = 1.0 - lambda_
    num_snps_to_mix = int(round(input_length * mixin_coefficient))
    random_to_mix = np.random.choice(input_length, num_snps_to_mix, replace=False)
    random_to_mix = torch.tensor(random_to_mix, dtype=torch.long)

    return random_to_mix


def mixup_all_targets(
    targets: "al_training_labels_target",
    random_index_for_mixing: torch.Tensor,
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
    mixed_object: MixupOutput,
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


def make_random_snps_missing(
    array: torch.Tensor, percentage: float = 0.05, probability: float = 1.0
) -> torch.Tensor:
    random_draw = np.random.uniform()
    if random_draw > probability:
        return array

    n_snps = array.shape[2]
    n_to_drop = (int(n_snps * percentage),)
    random_to_drop = np.random.choice(n_snps, n_to_drop, replace=False)
    random_to_drop = torch.tensor(random_to_drop, dtype=torch.long)

    missing_arr = torch.tensor([False, False, False, True], dtype=torch.bool).reshape(
        -1, 1
    )
    array[:, :, random_to_drop] = missing_arr

    return array
