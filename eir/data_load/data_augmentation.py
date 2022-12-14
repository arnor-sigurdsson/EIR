from dataclasses import dataclass
from functools import partial
from typing import (
    TYPE_CHECKING,
    Union,
    Dict,
    Callable,
    Sequence,
    Tuple,
    Iterable,
    Generator,
)

import numpy as np
import torch
from aislib.misc_utils import get_logger
from timm.data.mixup import rand_bbox
from torch import nn

from eir.data_load.data_utils import Batch, get_output_info_generator

if TYPE_CHECKING:
    from eir.train import al_training_labels_target, al_criteria, Experiment
    from eir.setup.schemas import InputConfig

al_target_values = Union[torch.LongTensor, torch.Tensor]
al_int_tensors = Union[
    torch.ByteTensor,
    torch.CharTensor,
    torch.ShortTensor,
    torch.IntTensor,
    torch.LongTensor,
    torch.BoolTensor,
]

logger = get_logger(name=__name__)


@dataclass
class MixingObject:
    targets: "al_training_labels_target"
    targets_permuted: "al_training_labels_target"
    lambda_: float
    permuted_indexes: al_int_tensors


def get_mix_data_hook(input_configs: Iterable["InputConfig"]):
    mixing_func_mapping = _get_mixing_func_map()

    input_mixing_func_map = {}

    for config in input_configs:

        cur_input_type_config = config.input_type_info
        cur_mixing_type = getattr(cur_input_type_config, "mixing_subtype", "mixup")

        cur_input_type = config.input_info.input_type
        cur_mixing_callable = mixing_func_mapping[cur_input_type][cur_mixing_type]

        cur_input_name = config.input_info.input_name

        input_mixing_func_map[cur_input_name] = cur_mixing_callable

    bound_hook = partial(
        hook_default_mix_data,
        mixing_funcs=input_mixing_func_map,
    )

    return bound_hook


def _get_mixing_func_map() -> Dict[str, Dict[str, Callable]]:
    mapping = {
        "omics": {
            "cutmix-uniform": uniform_cutmix_omics_input,
            "cutmix-block": block_cutmix_omics_input,
            "mixup": mixup_tensor,
        },
        "tabular": {"mixup": mixup_tensor},
        "sequence": {"mixup": mixup_tensor},
        "bytes": {"mixup": mixup_tensor},
        "image": {"mixup": mixup_tensor, "cutmix": cutmix_image},
    }

    return mapping


def cutmix_image(
    tensor: torch.Tensor,
    lambda_: float,
    random_batch_indices_to_mix: Sequence[al_int_tensors],
):

    image_shape = tensor[0].shape
    y_bottom, y_top, x_bottom, x_top = rand_bbox(img_shape=image_shape, lam=lambda_)

    target_to_cut = tensor[random_batch_indices_to_mix, :]
    cut_part = target_to_cut[..., y_bottom:y_top, x_bottom:x_top]

    # Caution: input_ will be modified as well since no .clone() below here
    cutmixed_x = tensor
    cutmixed_x[..., y_bottom:y_top, x_bottom:x_top] = cut_part

    return cutmixed_x


def hook_default_mix_data(
    experiment: "Experiment",
    state: Dict,
    mixing_funcs: Dict[str, Callable],
    *args,
    **kwargs,
) -> Dict:

    gc = experiment.configs.global_config

    batch = state["batch"]

    target_columns_gen = get_output_info_generator(outputs_as_dict=experiment.outputs)

    mixing_info = get_mixing_info(
        mixing_alpha=gc.mixing_alpha,
        batch_size=gc.batch_size,
        target_labels=batch.target_labels,
        target_columns_gen=target_columns_gen,
    )

    mixed_inputs = {}

    for input_name, input_data in batch.inputs.items():

        cur_mixing_func = mixing_funcs.get(input_name)
        mixed_input_batch = cur_mixing_func(
            tensor=input_data,
            lambda_=mixing_info.lambda_,
            random_batch_indices_to_mix=mixing_info.permuted_indexes,
        )
        mixed_inputs[input_name] = mixed_input_batch

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
    target_columns_gen: Generator[Tuple[str, str, str], None, None],
) -> MixingObject:
    lambda_ = _sample_lambda(mixing_alpha=mixing_alpha)

    permuted_indexes = get_random_batch_indices_to_mix(batch_size=batch_size)
    targets_permuted = mixup_all_targets(
        targets=target_labels,
        random_index_for_mixing=permuted_indexes,
        target_columns_gen=target_columns_gen,
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

    target_columns_gen = get_output_info_generator(outputs_as_dict=experiment.outputs)

    mixed_losses = calc_all_mixed_losses(
        target_columns_gen=target_columns_gen,
        criteria=experiment.criteria,
        outputs=state["model_outputs"],
        mixed_object=state["mixing_info"],
    )

    state_updates = {"per_target_train_losses": mixed_losses}

    return state_updates


def get_random_batch_indices_to_mix(batch_size: int) -> al_int_tensors:
    return torch.randperm(batch_size).to(dtype=torch.long)


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
    tensor: torch.Tensor, lambda_: float, random_batch_indices_to_mix: torch.Tensor
) -> torch.Tensor:

    cut_start, cut_end = get_block_cutmix_indices(
        input_length=tensor.shape[-1], lambda_=lambda_
    )
    target_to_cut = tensor[random_batch_indices_to_mix, :]
    cut_part = target_to_cut[..., cut_start:cut_end]

    # Caution: input_ will be modified as well since no .clone() below here
    cutmixed_x = tensor
    cutmixed_x[..., cut_start:cut_end] = cut_part

    return cutmixed_x


def get_block_cutmix_indices(input_length: int, lambda_: float) -> Tuple[int, int]:
    mixin_coefficient = 1.0 - lambda_
    num_snps_to_mix = int(round(input_length * mixin_coefficient))
    random_index_start = np.random.choice(max(1, input_length - num_snps_to_mix))
    random_index_end = random_index_start + num_snps_to_mix
    return random_index_start, random_index_end


def uniform_cutmix_omics_input(
    tensor: torch.Tensor,
    lambda_: float,
    random_batch_indices_to_mix: torch.Tensor,
) -> torch.Tensor:

    target_to_mix = tensor[random_batch_indices_to_mix, :]

    random_snp_indices_to_mix = get_uniform_cutmix_indices(
        input_length=tensor.shape[-1], lambda_=lambda_
    )
    cut_part = target_to_mix[..., random_snp_indices_to_mix]

    # Caution: input_ will be modified as well since no .clone() below here
    cutmixed_x = tensor
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
    target_columns_gen: Generator[Tuple[str, str, str], None, None],
) -> "al_training_labels_target":

    targets_permuted = {}

    for output_name, target_type, target_name in target_columns_gen:

        if output_name not in targets_permuted:
            targets_permuted[output_name] = {}

        cur_targets = targets[output_name][target_name]
        cur_targets_permuted = mixup_targets(
            targets=cur_targets, random_index_for_mixing=random_index_for_mixing
        )
        targets_permuted[output_name][target_name] = cur_targets_permuted

    return targets_permuted


def mixup_targets(
    targets: al_target_values,
    random_index_for_mixing: torch.Tensor,
) -> al_target_values:

    targets_permuted = targets.detach().clone()
    targets_permuted = targets_permuted[random_index_for_mixing]

    return targets_permuted


def calc_all_mixed_losses(
    target_columns_gen: Generator[Tuple[str, str, str], None, None],
    criteria: "al_criteria",
    outputs: Dict[str, Dict[str, torch.Tensor]],
    mixed_object: MixingObject,
):

    losses = {output_name: {} for output_name in outputs.keys()}

    for output_name, target_type, target_name in target_columns_gen:
        cur_loss = calc_mixed_loss(
            criterion=criteria[output_name][target_name],
            outputs=outputs[output_name][target_name],
            targets=mixed_object.targets[output_name][target_name],
            targets_permuted=mixed_object.targets_permuted[output_name][target_name],
            lambda_=mixed_object.lambda_,
        )
        losses[output_name][target_name] = cur_loss

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
    random_draw = torch.rand(1).item()
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
