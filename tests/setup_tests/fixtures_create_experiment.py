import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Iterable

import pytest
from torch import nn
from torch.utils.data import DataLoader

import eir.experiment_io
import eir.setup

from eir import train
from eir.experiment_io.experiment_io import (
    serialize_all_input_transformers,
    serialize_chosen_input_objects,
)
from eir.setup import schemas, config
from eir.setup.output_setup import set_up_outputs_for_training
from eir.train import Experiment
from eir.train_utils import optim, metrics
from eir.train_utils.utils import get_run_folder


def create_test_optimizer(
    global_config: schemas.GlobalConfig,
    model: nn.Module,
    criterions,
):
    """
    TODO: Refactor loss module construction out of this function.
    """

    loss_module = train._get_loss_callable(criteria=criterions)

    optimizer = optim.get_optimizer(
        model=model, loss_callable=loss_module, global_config=global_config
    )

    return optimizer, loss_module


@dataclass
class ModelTestConfig:
    iteration: int
    run_path: Path
    last_sample_folders: Dict[str, Dict[str, Path]]
    attributions_paths: Dict[str, Dict[str, Dict[str, Path]]]


@pytest.fixture()
def prep_modelling_test_configs(
    create_test_data,
    create_test_labels,
    create_test_config: config.Configs,
    create_test_dloaders,
    create_test_model,
    create_test_datasets,
) -> Tuple[Experiment, ModelTestConfig]:
    """
    Note that the fixtures used in this fixture get indirectly parametrized by
    test_classification and test_regression.
    """
    c = create_test_config
    gc = c.global_config
    train_loader, valid_loader, train_dataset, valid_dataset = create_test_dloaders
    target_labels = create_test_labels

    model = create_test_model

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=create_test_config.output_configs,
        target_transformers=target_labels.label_transformers,
    )

    criteria = train._get_criteria(outputs_as_dict=outputs_as_dict)
    test_metrics = metrics.get_default_metrics(
        target_transformers=target_labels.label_transformers,
        cat_averaging_metrics=gc.cat_averaging_metrics,
        con_averaging_metrics=gc.con_averaging_metrics,
    )
    test_metrics = _patch_metrics(metrics_=test_metrics)

    optimizer, loss_module = create_test_optimizer(
        global_config=gc,
        model=model,
        criterions=criteria,
    )

    train_dataset, valid_dataset = create_test_datasets

    train._log_model(model=model)

    inputs = eir.setup.input_setup.set_up_inputs_for_training(
        inputs_configs=c.input_configs,
        train_ids=tuple(target_labels.train_labels.keys()),
        valid_ids=tuple(target_labels.valid_labels.keys()),
        hooks=None,
    )
    run_folder = get_run_folder(output_folder=gc.output_folder)
    serialize_all_input_transformers(inputs_dict=inputs, run_folder=run_folder)
    serialize_chosen_input_objects(inputs_dict=inputs, run_folder=run_folder)

    hooks = train.get_default_hooks(configs=c)
    experiment = Experiment(
        configs=c,
        inputs=inputs,
        outputs=outputs_as_dict,
        train_loader=train_loader,
        valid_loader=valid_loader,
        valid_dataset=valid_dataset,
        model=model,
        optimizer=optimizer,
        criteria=criteria,
        loss_function=loss_module,
        metrics=test_metrics,
        writer=train.get_summary_writer(run_folder=Path(gc.output_folder)),
        hooks=hooks,
    )

    keys_to_serialize = (
        eir.experiment_io.experiment_io.get_default_experiment_keys_to_serialize()
    )
    eir.experiment_io.experiment_io.serialize_experiment(
        experiment=experiment,
        run_folder=get_run_folder(gc.output_folder),
        keys_to_serialize=keys_to_serialize,
    )

    targets = config.get_all_tabular_targets(output_configs=c.output_configs)
    test_config = _get_cur_modelling_test_config(
        train_loader=train_loader,
        global_config=gc,
        targets=targets,
        input_names=inputs.keys(),
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
    )
    for type_ in ("con",):
        for metric_record in metrics_[type_]:
            if metric_record.name == "r2":
                metric_record.only_val = False
    return metrics_


def _get_cur_modelling_test_config(
    train_loader: DataLoader,
    global_config: schemas.GlobalConfig,
    targets: config.TabularTargets,
    input_names: Iterable[str],
) -> ModelTestConfig:
    last_iter = len(train_loader) * global_config.n_epochs
    run_path = Path(f"{global_config.output_folder}/")

    last_sample_folders = _get_all_last_sample_folders(
        targets=targets, run_path=run_path, iteration=last_iter
    )

    all_attribution_paths = _get_all_attribution_paths(
        last_sample_folder_per_target_in_each_output=last_sample_folders,
        input_names=input_names,
    )

    test_config = ModelTestConfig(
        iteration=last_iter,
        run_path=run_path,
        last_sample_folders=last_sample_folders,
        attributions_paths=all_attribution_paths,
    )

    return test_config


def _get_all_attribution_paths(
    last_sample_folder_per_target_in_each_output: Dict[str, Dict[str, Path]],
    input_names: Iterable[str],
) -> Dict[str, Dict[str, Dict[str, Path]]]:
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
    targets: config.TabularTargets, run_path: Path, iteration: int
) -> Dict[str, Dict[str, Path]]:
    """
    output_name -> target_name: path
    """
    sample_folders = {}

    for output_name in targets.con_targets:
        if output_name not in sample_folders:
            sample_folders[output_name] = {}

        cur_con_columns = targets.con_targets[output_name]
        for con_column_name in cur_con_columns:
            sample_folders[output_name][con_column_name] = _get_test_sample_folder(
                run_path=run_path,
                iteration=iteration,
                output_name=output_name,
                column_name=con_column_name,
            )

    for output_name in targets.cat_targets:
        if output_name not in sample_folders:
            sample_folders[output_name] = {}

        cur_cat_columns = targets.cat_targets[output_name]
        for cat_column_name in cur_cat_columns:
            sample_folders[output_name][cat_column_name] = _get_test_sample_folder(
                run_path=run_path,
                iteration=iteration,
                output_name=output_name,
                column_name=cat_column_name,
            )

    return sample_folders


def _get_test_sample_folder(
    run_path: Path, iteration: int, output_name: str, column_name: str
) -> Path:
    sample_folder = (
        run_path / f"results/{output_name}/{column_name}/samples/{iteration}"
    )

    return sample_folder
