import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from torch import nn
from torch.utils.data import DataLoader

import eir.models.models_utils
import eir.train_utils.accelerator
from eir import train
from eir.experiment_io.input_object_io import serialize_chosen_input_objects
from eir.experiment_io.output_object_io import serialize_output_objects
from eir.setup import config, input_setup, schemas
from eir.setup.input_setup_modules.setup_tabular import serialize_all_input_transformers
from eir.setup.output_setup import set_up_outputs_for_training
from eir.train import Experiment, converge_sequence_input_and_output
from eir.train_utils import metrics, optim, step_logic
from eir.train_utils.criteria import get_criteria
from eir.train_utils.utils import get_run_folder

al_modelling_test_configs = tuple[Experiment, "ModelTestConfig"]


def create_test_optimizer(
    global_config: schemas.GlobalConfig,
    model: nn.Module,
    criteria,
    extra_modules: dict[str, nn.Module] | None,
):
    """
    TODO: Refactor loss module construction out of this function.
    """
    loss_module = train.get_loss_callable(criteria=criteria)

    optimizer = optim.get_optimizer(
        model=model,
        loss_callable=loss_module,
        global_config=global_config,
        extra_modules=extra_modules,
    )

    return optimizer, loss_module


@dataclass
class ModelTestConfig:
    iteration: int
    run_path: Path
    last_sample_folders: dict[str, dict[str, Path]]
    attributions_paths: dict[str, dict[str, dict[str, Path]]]


@pytest.fixture()
def prep_modelling_test_configs(
    create_test_data,
    create_test_labels,
    create_test_config: config.Configs,
    create_test_dataloaders,
    create_test_model,
    create_test_datasets,
) -> tuple[Experiment, ModelTestConfig]:
    """
    Note that the fixtures used in this fixture get indirectly parametrized by
    test_classification and test_regression.
    """
    c = create_test_config
    gc = c.global_config
    train_loader, valid_loader, train_dataset, valid_dataset = create_test_dataloaders
    target_labels = create_test_labels

    model = create_test_model

    test_metrics = metrics.get_default_metrics(
        target_transformers=target_labels.label_transformers,
        cat_metrics=gc.met.cat_metrics,
        con_metrics=gc.met.con_metrics,
        cat_averaging_metrics=gc.met.cat_averaging_metrics,
        con_averaging_metrics=gc.met.con_averaging_metrics,
        output_configs=c.output_configs,
    )
    test_metrics = _patch_metrics(metrics_=test_metrics)
    test_metrics["general_metric_info"].all_are_val_only = False

    train_dataset, valid_dataset = create_test_datasets

    train_ids = tuple(target_labels.train_labels["ID"])
    valid_ids = tuple(target_labels.valid_labels["ID"])

    inputs_as_dict = input_setup.set_up_inputs_for_training(
        inputs_configs=c.input_configs,
        train_ids=train_ids,
        valid_ids=valid_ids,
        hooks=None,
    )

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=create_test_config.output_configs,
        input_objects=inputs_as_dict,
        target_transformers=target_labels.label_transformers,
    )
    serialize_output_objects(
        output_objects=outputs_as_dict,
        run_folder=Path(gc.be.output_folder),
    )
    inputs_as_dict = converge_sequence_input_and_output(
        inputs=inputs_as_dict, outputs=outputs_as_dict
    )

    criteria = get_criteria(outputs_as_dict=outputs_as_dict)

    hooks = step_logic.get_default_hooks(configs=c)

    extra_modules = None
    if hooks.extra_state is not None:
        extra_modules = hooks.extra_state.get("uncertainty_modules", {})

    optimizer, loss_module = create_test_optimizer(
        global_config=gc,
        model=model,
        criteria=criteria,
        extra_modules=extra_modules,
    )

    run_folder = get_run_folder(output_folder=gc.be.output_folder)
    eir.models.models_utils.log_model(
        model=model,
        structure_file=run_folder / "model_info.txt",
    )

    serialize_all_input_transformers(inputs_dict=inputs_as_dict, run_folder=run_folder)
    serialize_chosen_input_objects(inputs_dict=inputs_as_dict, run_folder=run_folder)

    fabric = eir.train_utils.accelerator.setup_accelerator(configs=c)
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, valid_loader = fabric.setup_dataloaders(train_loader, valid_loader)

    experiment = Experiment(
        configs=c,
        inputs=inputs_as_dict,
        outputs=outputs_as_dict,
        train_loader=train_loader,
        valid_loader=valid_loader,
        valid_dataset=valid_dataset,
        model=model,
        optimizer=optimizer,
        criteria=criteria,
        loss_function=loss_module,
        metrics=test_metrics,
        hooks=hooks,
        fabric=fabric,
    )

    targets = config.get_all_tabular_targets(output_configs=c.output_configs)
    test_config = get_cur_modelling_test_config(
        train_loader=train_loader,
        global_config=gc,
        tabular_targets=targets,
        output_configs=c.output_configs,
        input_names=inputs_as_dict.keys(),
    )

    return experiment, test_config


def _patch_metrics(
    metrics_: metrics.al_metric_record_dict,
) -> metrics.al_metric_record_dict:
    warnings.warn(
        "This function will soon be deprecated as conftest will need to "
        "create its own metrics when train.py default metrics will be "
        "minimal.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    for type_ in ("con",):
        for metric_record in metrics_[type_]:
            if metric_record.name == "r2":
                metric_record.only_val = False
    return metrics_


def get_cur_modelling_test_config(
    train_loader: DataLoader,
    global_config: schemas.GlobalConfig,
    tabular_targets: config.TabularTargets,
    output_configs: Sequence[schemas.OutputConfig],
    input_names: Iterable[str],
) -> ModelTestConfig:
    last_iter = len(train_loader) * global_config.be.n_epochs

    last_valid_iter = last_iter
    if global_config.tc.early_stopping_patience:
        last_valid_iter -= last_valid_iter % global_config.ec.sample_interval

    run_path = Path(f"{global_config.be.output_folder}/")

    last_sample_folders = _get_all_last_sample_folders(
        output_configs=output_configs,
        tabular_targets=tabular_targets,
        run_path=run_path,
        iteration=last_valid_iter,
    )

    last_act_folders = _get_all_last_sample_folders(
        output_configs=output_configs,
        tabular_targets=tabular_targets,
        run_path=run_path,
        iteration=last_iter,
    )

    all_attribution_paths = _get_all_attribution_paths(
        last_sample_folder_per_target_in_each_output=last_act_folders,
        input_names=input_names,
    )

    test_config = ModelTestConfig(
        iteration=last_valid_iter,
        run_path=run_path,
        last_sample_folders=last_sample_folders,
        attributions_paths=all_attribution_paths,
    )

    return test_config


def _get_all_attribution_paths(
    last_sample_folder_per_target_in_each_output: dict[str, dict[str, Path]],
    input_names: Iterable[str],
) -> dict[str, dict[str, dict[str, Path]]]:
    """
    output_name -> target_name -> input_name: path
    """

    all_attribution_paths = {}

    dict_to_iter = last_sample_folder_per_target_in_each_output
    for output_name, file_per_target_dict in dict_to_iter.items():
        if output_name not in all_attribution_paths:
            all_attribution_paths[output_name] = {}

        for target_name, last_sample_folder in file_per_target_dict.items():
            all_attribution_paths[output_name][target_name] = {}

            for input_name in input_names:
                path = last_sample_folder / "attributions" / input_name
                all_attribution_paths[output_name][target_name][input_name] = path

    return all_attribution_paths


def _get_all_last_sample_folders(
    output_configs: Sequence[schemas.OutputConfig],
    tabular_targets: config.TabularTargets,
    run_path: Path,
    iteration: int,
) -> dict[str, dict[str, Path]]:
    """
    output_name -> target_name: path
    """
    sample_folders = {}

    for output_name in tabular_targets.con_targets:
        if output_name not in sample_folders:
            sample_folders[output_name] = {}

        cur_con_columns = tabular_targets.con_targets[output_name]
        for con_column_name in cur_con_columns:
            sample_folders[output_name][con_column_name] = _get_test_sample_folder(
                run_path=run_path,
                iteration=iteration,
                output_name=output_name,
                output_head_name=con_column_name,
            )

    for output_name in tabular_targets.cat_targets:
        if output_name not in sample_folders:
            sample_folders[output_name] = {}

        cur_cat_columns = tabular_targets.cat_targets[output_name]
        for cat_column_name in cur_cat_columns:
            sample_folders[output_name][cat_column_name] = _get_test_sample_folder(
                run_path=run_path,
                iteration=iteration,
                output_name=output_name,
                output_head_name=cat_column_name,
            )

    for config_ in output_configs:
        if config_.output_info.output_type == "tabular":
            continue

        output_name = config_.output_info.output_name
        if output_name not in sample_folders:
            sample_folders[output_name] = {}

        sample_folders[output_name][output_name] = _get_test_sample_folder(
            run_path=run_path,
            iteration=iteration,
            output_name=output_name,
            output_head_name=output_name,
        )

    return sample_folders


def _get_test_sample_folder(
    run_path: Path, iteration: int, output_name: str, output_head_name: str
) -> Path:
    sample_folder = (
        run_path / f"results/{output_name}/{output_head_name}/samples/{iteration}"
    )

    return sample_folder
