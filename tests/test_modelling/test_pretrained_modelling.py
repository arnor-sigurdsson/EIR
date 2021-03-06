from pathlib import Path
from typing import Tuple, Sequence, TYPE_CHECKING
from copy import deepcopy

import pytest

from eir import train
from eir.setup.schemas import BasicPretrainedConfig
from eir.setup.config import get_all_targets
from tests.test_modelling.test_modelling_utils import check_test_performance_results
from tests.conftest import _get_cur_modelling_test_config, cleanup

if TYPE_CHECKING:
    from tests.conftest import ModelTestConfig


@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "multi_task",
            "modalities": (
                "omics",
                "sequence",
                "image",
            ),
            "manual_test_data_creator": lambda: "test_multi_modal_multi_task",
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {
                    "output_folder": "multi_task_multi_modal",
                    "n_epochs": 1,
                    "act_background_samples": 8,
                    "sample_interval": 50,
                    "checkpoint_interval": 50,
                    "n_saved_models": 2,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {"l1": 1e-03},
                        },
                    },
                    {
                        "input_info": {"input_name": "test_sequence"},
                    },
                    {
                        "input_info": {"input_name": "test_bytes"},
                    },
                    {
                        "input_info": {"input_name": "test_image"},
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "input_cat_columns": ["OriginExtraCol"],
                            "input_con_columns": ["ExtraTarget"],
                        },
                        "model_config": {
                            "model_type": "tabular",
                            "model_init_config": {"l1": 1e-03},
                        },
                    },
                ],
                "predictor_configs": {
                    "model_config": {
                        "fc_task_dim": 64,
                        "fc_do": 0.10,
                        "rb_do": 0.10,
                    },
                },
                "target_configs": {
                    "target_cat_columns": ["Origin"],
                    "target_con_columns": ["Height"],
                },
            },
        }
    ],
    indirect=True,
)
def test_pre_trained_module_setup(
    prep_modelling_test_configs: Tuple[train.Experiment, "ModelTestConfig"],
):
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    _get_experiment_overloaded_for_pretrained_extractor(
        experiment=experiment, test_config=test_config, rename_pretrained_inputs=False
    )

    _get_experiment_overloaded_for_pretrained_extractor(
        experiment=experiment, test_config=test_config, rename_pretrained_inputs=True
    )

    _get_experiment_overloaded_for_pretrained_checkpoint(
        experiment=experiment, test_config=test_config
    )

    experiment_overwritten = _add_new_feature_extractor_to_experiment(
        experiment=experiment
    )
    _get_experiment_overloaded_for_pretrained_extractor(
        experiment=experiment_overwritten,
        test_config=test_config,
        rename_pretrained_inputs=False,
    )

    _get_experiment_overloaded_for_pretrained_extractor(
        experiment=experiment_overwritten,
        test_config=test_config,
        rename_pretrained_inputs=True,
    )


def _add_new_feature_extractor_to_experiment(
    experiment: train.Experiment,
) -> train.Experiment:
    """
    Used to check that we can do a partial loading of pretrained feature extractors.
    """

    experiment_attrs = experiment.__dict__
    experiment_configs = deepcopy(experiment.configs)
    first_config_modified = deepcopy(experiment_configs.input_configs[0])

    extra_input_name = first_config_modified.input_info.input_name + "_copy"
    first_config_modified.input_info.input_name = extra_input_name

    inputs_configs = list(experiment_configs.input_configs)
    inputs_configs_with_extra = inputs_configs + [first_config_modified]
    experiment_attrs["configs"].input_configs = inputs_configs_with_extra

    experiment_overwritten = train.Experiment(**experiment_attrs)

    return experiment_overwritten


@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "multi_task",
            "modalities": (
                "omics",
                "sequence",
                "image",
            ),
            "manual_test_data_creator": lambda: "test_multi_modal_multi_task",
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {
                    "output_folder": "multi_task_multi_modal",
                    "n_epochs": 2,
                    "act_background_samples": 8,
                    "sample_interval": 50,
                    "checkpoint_interval": 50,
                    "n_saved_models": 2,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {"l1": 1e-03},
                        },
                    },
                    {
                        "input_info": {"input_name": "test_sequence"},
                    },
                    {
                        "input_info": {"input_name": "test_bytes"},
                    },
                    {
                        "input_info": {"input_name": "test_image"},
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "input_cat_columns": ["OriginExtraCol"],
                            "input_con_columns": ["ExtraTarget"],
                        },
                        "model_config": {
                            "model_type": "tabular",
                            "model_init_config": {"l1": 1e-03},
                        },
                    },
                ],
                "predictor_configs": {
                    "model_config": {
                        "fc_task_dim": 64,
                        "fc_do": 0.10,
                        "rb_do": 0.10,
                    },
                },
                "target_configs": {
                    "target_cat_columns": ["Origin"],
                    "target_con_columns": ["Height"],
                },
            },
        }
    ],
    indirect=True,
)
def test_pre_training_and_loading(
    prep_modelling_test_configs: Tuple[train.Experiment, "ModelTestConfig"],
):
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    (
        pretrained_experiment,
        pretrained_test_config,
    ) = _get_experiment_overloaded_for_pretrained_extractor(
        experiment=experiment, test_config=test_config, rename_pretrained_inputs=True
    )

    train.train(experiment=pretrained_experiment)

    targets = get_all_targets(targets_configs=experiment.configs.target_configs)

    # Note we skip checking R2 for now as we patch the metrics in conftest.py
    # to check for both training and validation, but for now we will make do with
    # checking only the MCC for this
    for cat_column in targets.cat_targets:

        check_test_performance_results(
            run_path=pretrained_test_config.run_path,
            target_column=cat_column,
            metric="mcc",
            thresholds=(0.9, 0.9),
        )

    (
        pretrained_checkpoint_experiment,
        pretrained_checkpoint_test_config,
    ) = _get_experiment_overloaded_for_pretrained_checkpoint(
        experiment=experiment, test_config=test_config
    )

    train.train(experiment=pretrained_checkpoint_experiment)

    for cat_column in targets.cat_targets:

        check_test_performance_results(
            run_path=pretrained_checkpoint_test_config.run_path,
            target_column=cat_column,
            metric="mcc",
            thresholds=(0.9, 0.9),
        )


def _get_experiment_overloaded_for_pretrained_extractor(
    experiment: train.Experiment,
    test_config: "ModelTestConfig",
    rename_pretrained_inputs: bool,
    skip_pretrained_keys: Sequence[str] = tuple(),
) -> Tuple[train.Experiment, "ModelTestConfig"]:

    input_configs = deepcopy(experiment.configs.input_configs)
    pretrained_configs = deepcopy(experiment.configs)
    saved_model_path = next((test_config.run_path / "saved_models").iterdir())

    input_configs_with_pretrained = []
    for cur_input_config in input_configs:
        cur_name = cur_input_config.input_info.input_name

        if cur_name in skip_pretrained_keys:
            continue

        cur_pretrained_config = BasicPretrainedConfig(
            model_path=str(saved_model_path), load_module_name=cur_name
        )
        cur_input_config.pretrained_config = cur_pretrained_config

        # Check that the names can differ
        if rename_pretrained_inputs:
            cur_input_config.input_info.input_name = cur_name + "_pretrained_module"

        input_configs_with_pretrained.append(cur_input_config)

    pretrained_configs.input_configs = input_configs_with_pretrained

    pretrained_configs.global_config.n_epochs = 6
    pretrained_configs.global_config.sample_interval = 200
    pretrained_configs.global_config.checkpoint_interval = 200
    pretrained_configs.global_config.output_folder = (
        pretrained_configs.global_config.output_folder + "_with_pretrained"
    )

    run_path = Path(f"{pretrained_configs.global_config.output_folder}/")
    if run_path.exists():
        cleanup(run_path=run_path)

    default_hooks = train.get_default_hooks(configs=pretrained_configs)
    pretrained_experiment = train.get_default_experiment(
        configs=pretrained_configs, hooks=default_hooks
    )

    targets = get_all_targets(
        targets_configs=pretrained_experiment.configs.target_configs
    )
    pretrained_test_config = _get_cur_modelling_test_config(
        train_loader=pretrained_experiment.train_loader,
        global_config=pretrained_configs.global_config,
        targets=targets,
        input_names=pretrained_experiment.inputs.keys(),
    )

    return pretrained_experiment, pretrained_test_config


def _get_experiment_overloaded_for_pretrained_checkpoint(
    experiment: train.Experiment, test_config: "ModelTestConfig"
) -> Tuple[train.Experiment, "ModelTestConfig"]:

    pretrained_configs = deepcopy(experiment.configs)
    saved_model_path = next((test_config.run_path / "saved_models").iterdir())

    pretrained_configs.global_config.n_epochs = 6
    pretrained_configs.global_config.pretrained_checkpoint = str(saved_model_path)
    pretrained_configs.global_config.sample_interval = 200
    pretrained_configs.global_config.checkpoint_interval = 200
    pretrained_configs.global_config.output_folder = (
        pretrained_configs.global_config.output_folder + "_with_pretrained_checkpoint"
    )

    run_path = Path(f"{pretrained_configs.global_config.output_folder}/")
    if run_path.exists():
        cleanup(run_path=run_path)

    default_hooks = train.get_default_hooks(configs=pretrained_configs)
    pretrained_experiment = train.get_default_experiment(
        configs=pretrained_configs, hooks=default_hooks
    )

    targets = get_all_targets(
        targets_configs=pretrained_experiment.configs.target_configs
    )
    pretrained_test_config = _get_cur_modelling_test_config(
        train_loader=pretrained_experiment.train_loader,
        global_config=pretrained_configs.global_config,
        targets=targets,
        input_names=pretrained_experiment.inputs.keys(),
    )

    return pretrained_experiment, pretrained_test_config
