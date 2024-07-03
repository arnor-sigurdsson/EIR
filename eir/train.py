from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import torch
import torchtext

torchtext.disable_torchtext_deprecation_warning()
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
from aislib.misc_utils import ensure_path_exists
from ignite.engine import Engine
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler

from eir import __version__
from eir.data_load import datasets
from eir.data_load.data_utils import get_train_sampler
from eir.data_load.label_setup import split_ids
from eir.experiment_io.experiment_io import (
    get_default_experiment_keys_to_serialize,
    get_version_file,
    serialize_chosen_input_objects,
    serialize_experiment,
)
from eir.models.model_setup import get_model
from eir.models.model_setup_modules.meta_setup import al_meta_model
from eir.models.model_training_utils import run_lr_find
from eir.setup.config import Configs, get_configs
from eir.setup.input_setup import al_input_objects_as_dict, set_up_inputs_for_training
from eir.setup.input_setup_modules.setup_tabular import serialize_all_input_transformers
from eir.setup.output_setup import (
    al_output_objects_as_dict,
    set_up_outputs_for_training,
)
from eir.setup.output_setup_modules.sequence_output_setup import (
    converge_sequence_input_and_output,
)
from eir.target_setup.target_label_setup import (
    gather_all_ids_from_output_configs,
    read_manual_ids_if_exist,
    set_up_all_targets_wrapper,
)
from eir.train_utils import distributed, utils
from eir.train_utils.criteria import al_criteria_dict, get_criteria, get_loss_callable
from eir.train_utils.metrics import get_average_history_filepath, get_default_metrics
from eir.train_utils.optim import get_optimizer, maybe_wrap_model_with_swa
from eir.train_utils.step_logic import (
    Hooks,
    al_training_labels_target,
    get_default_hooks,
)
from eir.train_utils.train_handlers import configure_trainer
from eir.train_utils.utils import call_hooks_stage_iterable
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.train_utils.metrics import al_metric_record_dict, al_step_metric_dict


utils.seed_everything()
logger = get_logger(name=__name__, tqdm_compatible=True)


def main():
    torch.backends.cudnn.benchmark = True

    configs = get_configs()

    configs, local_rank = distributed.maybe_initialize_distributed_environment(
        configs=configs
    )

    default_hooks = get_default_hooks(configs=configs)
    default_experiment = get_default_experiment(
        configs=configs,
        hooks=default_hooks,
    )

    run_experiment(experiment=default_experiment)


@dataclass(frozen=True)
class Experiment:
    configs: Configs
    inputs: al_input_objects_as_dict
    outputs: al_output_objects_as_dict
    train_loader: torch.utils.data.DataLoader
    valid_loader: torch.utils.data.DataLoader
    valid_dataset: datasets.al_datasets
    model: al_meta_model
    optimizer: Optimizer
    criteria: al_criteria_dict
    loss_function: Callable
    metrics: "al_metric_record_dict"
    hooks: Hooks


def get_default_experiment(
    configs: Configs,
    hooks: Hooks,
) -> "Experiment":
    run_folder = _prepare_run_folder(output_folder=configs.global_config.output_folder)

    all_array_ids = gather_all_ids_from_output_configs(
        output_configs=configs.output_configs
    )
    manual_valid_ids = read_manual_ids_if_exist(
        manual_valid_ids_file=configs.global_config.manual_valid_ids_file
    )

    train_ids, valid_ids = split_ids(
        ids=all_array_ids,
        valid_size=configs.global_config.valid_size,
        manual_valid_ids=manual_valid_ids,
    )

    target_labels = set_up_all_targets_wrapper(
        train_ids=train_ids,
        valid_ids=valid_ids,
        run_folder=run_folder,
        output_configs=configs.output_configs,
        hooks=hooks,
    )

    inputs_as_dict = set_up_inputs_for_training(
        inputs_configs=configs.input_configs,
        train_ids=train_ids,
        valid_ids=valid_ids,
        hooks=hooks,
    )

    serialize_all_input_transformers(inputs_dict=inputs_as_dict, run_folder=run_folder)
    serialize_chosen_input_objects(inputs_dict=inputs_as_dict, run_folder=run_folder)

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=configs.output_configs,
        input_objects=inputs_as_dict,
        target_transformers=getattr(target_labels, "label_transformers", None),
    )
    inputs_as_dict = converge_sequence_input_and_output(
        inputs=inputs_as_dict, outputs=outputs_as_dict
    )

    train_dataset, valid_dataset = datasets.set_up_datasets_from_configs(
        configs=configs,
        target_labels=target_labels,
        inputs_as_dict=inputs_as_dict,
        outputs_as_dict=outputs_as_dict,
        train_ids_to_keep=train_ids,
        valid_ids_to_keep=valid_ids,
    )

    train_sampler = get_train_sampler(
        columns_to_sample=configs.global_config.weighted_sampling_columns,
        train_dataset=train_dataset,
    )

    train_dataloader, valid_dataloader = get_dataloaders(
        train_dataset=train_dataset,
        train_sampler=train_sampler,
        valid_dataset=valid_dataset,
        batch_size=configs.global_config.batch_size,
        num_workers=configs.global_config.dataloader_workers,
    )

    model = get_model(
        global_config=configs.global_config,
        inputs_as_dict=inputs_as_dict,
        fusion_config=configs.fusion_config,
        outputs_as_dict=outputs_as_dict,
    )

    model = maybe_wrap_model_with_swa(
        n_iter_before_swa=configs.global_config.n_iter_before_swa,
        model=model,
        device=torch.device(configs.global_config.device),
    )
    _log_model(
        model=model,
        structure_file=run_folder / "model_info.txt",
    )

    criteria = get_criteria(
        outputs_as_dict=outputs_as_dict,
    )

    loss_func = get_loss_callable(
        criteria=criteria,
    )

    optimizer = get_optimizer(
        model=model, loss_callable=loss_func, global_config=configs.global_config
    )

    metrics = get_default_metrics(
        target_transformers=target_labels.label_transformers,
        cat_averaging_metrics=configs.global_config.cat_averaging_metrics,
        con_averaging_metrics=configs.global_config.con_averaging_metrics,
    )

    experiment = Experiment(
        configs=configs,
        inputs=inputs_as_dict,
        outputs=outputs_as_dict,
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        valid_dataset=valid_dataset,
        model=model,
        optimizer=optimizer,
        criteria=criteria,
        loss_function=loss_func,
        metrics=metrics,
        hooks=hooks,
    )

    return experiment


def _prepare_run_folder(output_folder: str) -> Path:
    run_folder = utils.get_run_folder(output_folder=output_folder)
    history_file = get_average_history_filepath(
        run_folder=run_folder, train_or_val_target_prefix="train_"
    )
    if history_file.exists():
        raise FileExistsError(
            f"There already exists a run with that name: {history_file}. Please choose "
            f"a different run name or delete the folder."
        )

    ensure_path_exists(path=run_folder, is_folder=True)

    return run_folder


def get_dataloaders(
    train_dataset: datasets.DatasetBase,
    train_sampler: Union[None, WeightedRandomSampler, DistributedSampler],
    valid_dataset: datasets.DatasetBase,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple:
    check_dataset_and_batch_size_compatibility(
        dataset=train_dataset, batch_size=batch_size, name="Training"
    )

    check_dataset_and_batch_size_compatibility(
        dataset=valid_dataset, batch_size=batch_size, name="Validation"
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False if train_sampler else True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    return train_dataloader, valid_dataloader


def check_dataset_and_batch_size_compatibility(
    dataset: datasets.DatasetBase, batch_size: int, name: str = ""
):
    if len(dataset) < batch_size:
        raise ValueError(
            f"{name} dataset size ({len(dataset)}) can not be smaller than "
            f"batch size ({batch_size}). A fix can be increasing {name.lower()} sample "
            f"size or reducing the batch size. If predicting on few unknown samples, "
            f"a solution can be setting the batch size to 1 in the global configuration"
            f" passed to the prediction module. Future work includes making this "
            f"easier to work with."
        )


def _log_model(
    model: nn.Module,
    structure_file: Optional[Path],
    verbose: bool = False,
    parameter_file: Optional[str] = None,
) -> None:
    no_params = sum(p.numel() for p in model.parameters())
    no_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_summary = "Starting training with following model specifications:\n"
    model_summary += f"Total parameters: {format(no_params, ',.0f')}\n"
    model_summary += f"Trainable parameters: {format(no_trainable_params, ',.0f')}"

    logger.info(model_summary)

    if structure_file:
        with open(structure_file, "w") as structure_handle:
            structure_handle.write(str(model))

    layer_summary = ""
    if verbose:
        layer_summary = "\nModel layers:\n"
        for name, param in model.named_parameters():
            layer_summary += (
                f"Layer: {name}, "
                f"Shape: {list(param.size())}, "
                f"Parameters: {param.numel()}, "
                f"Trainable: {param.requires_grad}\n"
            )
        logger.info(layer_summary)

    if parameter_file:
        with open(parameter_file, "w") as f:
            f.write(model_summary)
            if verbose:
                f.write(layer_summary)


def _log_eir_version_info(outfile: Path) -> None:
    eir_version_info = f"{__version__}\n"

    ensure_path_exists(path=outfile)
    with open(outfile, "w") as f:
        f.write(eir_version_info)


def run_experiment(experiment: Experiment) -> None:
    gc = experiment.configs.global_config
    run_folder = utils.get_run_folder(output_folder=gc.output_folder)

    _log_eir_version_info(outfile=get_version_file(run_folder=run_folder))

    keys_to_serialize = get_default_experiment_keys_to_serialize()
    serialize_experiment(
        experiment=experiment,
        run_folder=run_folder,
        keys_to_serialize=keys_to_serialize,
    )

    train(experiment=experiment)


def train(experiment: Experiment) -> None:
    exp = experiment
    gc = experiment.configs.global_config

    trainer = get_base_trainer(experiment=experiment)

    if gc.find_lr:
        logger.info("Running LR find and exiting.")
        run_lr_find(
            trainer_engine=trainer,
            train_dataloader=exp.train_loader,
            model=exp.model,
            optimizer=exp.optimizer,
            output_folder=utils.get_run_folder(output_folder=gc.output_folder),
        )
        return

    trainer = configure_trainer(trainer=trainer, experiment=experiment)

    trainer.run(data=exp.train_loader, max_epochs=gc.n_epochs)


def get_base_trainer(experiment: Experiment) -> Engine:
    step_hooks = experiment.hooks.step_func_hooks

    def step(
        engine: Engine,
        loader_batch: Tuple[torch.Tensor, al_training_labels_target, List[str]],
    ) -> "al_step_metric_dict":
        """
        The output here goes to trainer.output.
        """
        experiment.model.train()
        experiment.optimizer.zero_grad()

        base_prepare_inputs_stage = step_hooks.base_prepare_batch
        state = call_hooks_stage_iterable(
            hook_iterable=base_prepare_inputs_stage,
            common_kwargs={"experiment": experiment, "loader_batch": loader_batch},
            state=None,
        )
        base_batch = state["batch"]

        post_prepare_inputs_stage = step_hooks.post_prepare_batch
        state = call_hooks_stage_iterable(
            hook_iterable=post_prepare_inputs_stage,
            common_kwargs={"experiment": experiment, "loader_batch": base_batch},
            state=state,
        )
        batch = state["batch"]

        model_forward_loss_stage = step_hooks.model_forward
        state = call_hooks_stage_iterable(
            hook_iterable=model_forward_loss_stage,
            common_kwargs={"experiment": experiment, "batch": batch},
            state=state,
        )

        loss_stage = step_hooks.loss
        state = call_hooks_stage_iterable(
            hook_iterable=loss_stage,
            common_kwargs={"experiment": experiment, "batch": batch},
            state=state,
        )

        optimizer_backward_stage = step_hooks.optimizer_backward
        state = call_hooks_stage_iterable(
            hook_iterable=optimizer_backward_stage,
            common_kwargs={"experiment": experiment, "batch": batch},
            state=state,
        )

        metrics_stage = step_hooks.metrics
        state = call_hooks_stage_iterable(
            hook_iterable=metrics_stage,
            common_kwargs={"experiment": experiment, "batch": batch},
            state=state,
        )

        return state["metrics"]

    trainer = Engine(process_function=step)

    return trainer


if __name__ == "__main__":
    main()
