from pathlib import Path
from typing import Union, Tuple, Dict, List

import numpy as np
import pandas as pd
import pytest

from eir import train
from eir.setup.config import get_all_targets, Configs
from tests.conftest import cleanup, ModelTestConfig


@pytest.mark.parametrize(
    "create_test_data",
    [
        {"task_type": "binary"},
        {"task_type": "multi"},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: MLP
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "linear"},
                        "model_config": {"l1": 1e-03},
                    }
                ],
            },
        },
        # Case 2: CNN
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "cnn"},
                        "model_config": {
                            "rb_do": 0.25,
                            "channel_exp_base": 3,
                            "l1": 1e-03,
                        },
                    }
                ],
            },
        },
        # Case 3: Linear
        {
            "injections": {
                "global_configs": {"lr": 1e-03},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "identity"},
                    },
                ],
                "predictor_configs": {
                    "model_type": "linear",
                    "model_config": {"l1": 1e-03},
                },
            },
        },
    ],
    indirect=True,
)
def test_classification(keep_outputs, prep_modelling_test_configs):
    """
    NOTE:
        We probably cannot check directly if the gradients for a given SNP
        form are highest when that SNP form is present. E.g. if Asia has form
        [0, 0, 1, 0] in certain positions, it's not automatic that index 2
        in that position has the highest gradient, because the absence of a 1
        in at index 1 for example contributes to the decision as well.

        This can be circumvented by zero-ing out non-matches, i.e. multiplying
        the gradients with the one-hot inputs, but in that case we might
        be throwing important information away.

    NOTE:
        The indirect parametrization passes the arguments over to the fixtures used
        in _prep_modelling_test_config.
    """
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment)

    target_column = experiment.configs.target_configs[0].target_cat_columns[0]

    _check_test_performance_results(
        run_path=test_config.run_path,
        target_column=target_column,
        metric="mcc",
        thresholds=(0.8, 0.8),
    )

    top_row_grads_dict = {"Asia": [0] * 10, "Europe": [1] * 10, "Africa": [2] * 10}
    _check_snps_wrapper(
        test_config=test_config,
        target_column=target_column,
        top_row_grads_dict=top_row_grads_dict,
        at_least_n=8,
    )

    if not keep_outputs:
        cleanup(test_config.run_path)


def _check_snps_wrapper(
    test_config: ModelTestConfig,
    target_column: str,
    top_row_grads_dict: Dict[str, List[int]],
    at_least_n: Union[str, int] = "all",
):
    expected_top_indxs = list(range(50, 1000, 100))

    for paths in [test_config.activations_path, test_config.masked_activations_path]:
        check_types = True if paths == test_config.masked_activations_path else False
        cur_path = paths[target_column]
        _check_identified_snps(
            arrpath=cur_path,
            expected_top_indxs=expected_top_indxs,
            top_row_grads_dict=top_row_grads_dict,
            check_types=check_types,
            at_least_n=at_least_n,
        )


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "regression"}], indirect=True
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: Linear
        {
            "injections": {
                "global_configs": {"lr": 1e-03},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "identity"},
                    },
                ],
                "predictor_configs": {
                    "model_type": "linear",
                    "model_config": {"l1": 5e-03},
                },
                "target_configs": {
                    "target_cat_columns": [],
                    "target_con_columns": ["Height"],
                },
            },
        },
        # Case 2: CNN
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "cnn"},
                        "model_config": {"l1": 1e-03},
                    },
                ],
                "target_configs": {
                    "target_cat_columns": [],
                    "target_con_columns": ["Height"],
                },
            },
        },
        # Case 3: MLP
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "linear"},
                        "model_config": {"l1": 1e-03},
                    },
                ],
                "target_configs": {
                    "target_cat_columns": [],
                    "target_con_columns": ["Height"],
                },
            },
        },
        # Case 4: CNN Cycle
        {
            "injections": {
                "global_configs": {
                    "lr_schedule": "cycle",
                    "run_name": "test_lr-cycle",
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "cnn"},
                        "model_config": {"l1": 1e-03},
                    },
                ],
                "target_configs": {
                    "target_cat_columns": [],
                    "target_con_columns": ["Height"],
                },
            },
        },
    ],
    indirect=True,
)
def test_regression(keep_outputs, prep_modelling_test_configs):
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment)

    target_column = experiment.configs.target_configs[0].target_con_columns[0]
    model_type = (
        experiment.configs.input_configs[0].input_type_info.model_type == "linear"
    )

    # linear regression performs slightly worse, but we don't want to lower expectations
    # other models
    thresholds = (0.70, 0.70) if model_type == "linear" else (0.8, 0.8)
    _check_test_performance_results(
        run_path=test_config.run_path,
        target_column=target_column,
        metric="r2",
        thresholds=thresholds,
    )

    top_height_snp_index = 2
    top_row_grads_dict = {target_column: [top_height_snp_index] * 10}
    _check_snps_wrapper(
        test_config=test_config,
        target_column=target_column,
        top_row_grads_dict=top_row_grads_dict,
        at_least_n=8,
    )

    if not keep_outputs:
        cleanup(test_config.run_path)


def _check_test_performance_results(
    run_path: Path, target_column: str, metric: str, thresholds: Tuple[float, float]
):
    target_column_results_folder = run_path / "results" / target_column
    train_history_path = (
        target_column_results_folder / f"train_{target_column}_history.log"
    )
    valid_history_path = (
        target_column_results_folder / f"validation_{target_column}_history.log"
    )

    threshold_train, threshold_valid = thresholds

    df_train = pd.read_csv(train_history_path)
    assert df_train.loc[:, f"{target_column}_{metric}"].max() > threshold_train

    df_valid = pd.read_csv(valid_history_path)
    assert df_valid.loc[:, f"{target_column}_{metric}"].max() > threshold_valid


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "multi_task"}], indirect=True
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 0: Check that we add and use extra inputs.
        {
            "injections": {
                "global_configs": {
                    "run_name": "extra_inputs",
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "cnn"},
                        "model_config": {"l1": 1e-03},
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "model_type": "tabular",
                            "extra_cat_columns": ["OriginExtraCol"],
                            "extra_con_columns": ["ExtraTarget"],
                        },
                    },
                ],
                "target_configs": {
                    "target_cat_columns": ["Origin"],
                    "target_con_columns": ["Height"],
                },
            },
        },
        # Case 1: Normal multi task with CNN
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "cnn"},
                        "model_config": {
                            "channel_exp_base": 5,
                            "rb_do": 0.15,
                            "fc_repr_dim": 64,
                            "l1": 1e-03,
                        },
                    },
                ],
                "predictor_configs": {
                    "model_config": {"fc_task_dim": 64, "rb_do": 0.10, "fc_do": 0.10},
                },
                "target_configs": {
                    "target_cat_columns": ["Origin"],
                    "target_con_columns": ["Height", "ExtraTarget"],
                },
            },
        },
        # Case 2:  Normal multi task with MLP, note we have to reduce the LR for
        # stability and add L1 for regularization
        {
            "injections": {
                "global_configs": {"lr": 1e-03},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "linear"},
                        "model_config": {"l1": 1e-03},
                    },
                ],
                "predictor_configs": {
                    "model_config": {"fc_task_dim": 64, "rb_do": 0.10, "fc_do": 0.10},
                },
                "target_configs": {
                    "target_cat_columns": ["Origin"],
                    "target_con_columns": ["Height", "ExtraTarget"],
                },
            },
        },
        # Case 3: Using the Simple LCL model
        {
            "injections": {
                "global_configs": {"lr": 1e-03},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "mlp-split"},
                        "model_config": {
                            "fc_repr_dim": 8,
                            "split_mlp_num_splits": 64,
                            "l1": 1e-03,
                        },
                    },
                ],
                "target_configs": {
                    "target_cat_columns": ["Origin"],
                    "target_con_columns": ["Height", "ExtraTarget"],
                },
            },
        },
        # Case 4: Using the GLN
        {
            "injections": {
                "global_configs": {
                    "lr": 1e-03,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "genome-local-net"},
                        "model_config": {
                            "kernel_width": 8,
                            "channel_exp_base": 2,
                            "l1": 1e-03,
                            "rb_do": 0.20,
                        },
                    },
                ],
                "predictor_configs": {
                    "model_config": {
                        "fc_task_dim": 64,
                        "fc_do": 0.20,
                        "rb_do": 0.20,
                    },
                },
                "target_configs": {
                    "target_cat_columns": ["Origin"],
                    "target_con_columns": ["Height", "ExtraTarget"],
                },
            },
        },
        # Case 5: Using the MGMoE fusion
        {
            "injections": {
                "global_configs": {
                    "run_name": "mgmoe",
                    "lr": 1e-03,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "genome-local-net"},
                        "model_config": {
                            "kernel_width": 8,
                            "channel_exp_base": 2,
                            "l1": 1e-03,
                        },
                    },
                ],
                "predictor_configs": {
                    "model_type": "mgmoe",
                    "model_config": {"mg_num_experts": 2},
                },
                "target_configs": {
                    "target_cat_columns": ["Origin"],
                    "target_con_columns": ["Height", "ExtraTarget"],
                },
            },
        },
        # Case 6: Using the GLN with mixing
        {
            "injections": {
                "global_configs": {
                    "run_name": "mixing_multi",
                    "lr": 1e-03,
                    "mixing_alpha": 0.5,
                    "mixing_type": "cutmix-uniform",
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "genome-local-net"},
                        "model_config": {
                            "kernel_width": 8,
                            "channel_exp_base": 2,
                            "l1": 1e-03,
                        },
                    },
                ],
                "target_configs": {
                    "target_cat_columns": ["Origin"],
                    "target_con_columns": ["Height", "ExtraTarget"],
                },
            },
        },
        # Case 7: Using the GLN with limited activations
        {
            "injections": {
                "global_configs": {
                    "run_name": "limited_activations",
                    "lr": 1e-03,
                    "max_acts_per_class": 100,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "genome-local-net"},
                        "model_config": {
                            "kernel_width": 8,
                            "channel_exp_base": 2,
                            "l1": 1e-03,
                            "rb_do": 0.20,
                        },
                    },
                ],
                "target_configs": {
                    "target_cat_columns": ["Origin"],
                    "target_con_columns": ["Height", "ExtraTarget"],
                },
            },
        },
    ],
    indirect=True,
)
def test_multi_task(
    keep_outputs: bool,
    prep_modelling_test_configs: Tuple[train.Experiment, ModelTestConfig],
):
    experiment, test_config = prep_modelling_test_configs
    gc = experiment.configs.global_config

    train.train(experiment)

    targets = get_all_targets(targets_configs=experiment.configs.target_configs)
    extra_columns = get_all_tabular_input_columns(configs=experiment.configs)
    for cat_column in targets.cat_targets:
        threshold, at_least_n = _get_multi_task_test_args(
            extra_columns=extra_columns,
            target_copy="OriginExtraCol",
            mixing=gc.mixing_type,
        )

        _check_test_performance_results(
            run_path=test_config.run_path,
            target_column=cat_column,
            metric="mcc",
            thresholds=threshold,
        )

        top_row_grads_dict = {"Asia": [0] * 10, "Europe": [1] * 10, "Africa": [2] * 10}
        _check_snps_wrapper(
            test_config=test_config,
            target_column=cat_column,
            top_row_grads_dict=top_row_grads_dict,
            at_least_n=at_least_n,
        )

    for con_column in targets.con_targets:
        threshold, at_least_n = _get_multi_task_test_args(
            extra_columns=extra_columns,
            target_copy="ExtraTarget",
            mixing=gc.mixing_type,
        )

        _check_test_performance_results(
            run_path=test_config.run_path,
            target_column=con_column,
            metric="r2",
            thresholds=threshold,
        )

        top_height_snp_index = 2
        top_row_grads_dict = {con_column: [top_height_snp_index] * 10}
        _check_snps_wrapper(
            test_config=test_config,
            target_column=con_column,
            top_row_grads_dict=top_row_grads_dict,
            at_least_n=at_least_n,
        )

    if not keep_outputs:
        cleanup(test_config.run_path)


def get_all_tabular_input_columns(configs: Configs):
    extra_columns = []
    for input_config in configs.input_configs:
        if input_config.input_info.input_type == "tabular":
            extra_columns += input_config.input_type_info.extra_con_columns
            extra_columns += input_config.input_type_info.extra_cat_columns

    return extra_columns


def _get_multi_task_test_args(
    extra_columns: List[str], target_copy: str, mixing: Union[None, str]
) -> Tuple[Tuple[float, float], int]:
    """
    We use 0 for at_least_n in the case we have correlated input columns because
    in that case the model might not actually be using any of the SNPs (better to use
    the correlated column), hence we do not expect SNPs to be highly activated.

    When mixing, the train performance (loss is OK) for e.g. accuracy is expected to be
    relatively low, as we do not have discrete inputs but a mix.
    """

    an_extra_col_is_correlated_with_target = target_copy in extra_columns
    if an_extra_col_is_correlated_with_target:
        thresholds, at_least_n = (0.9, 0.9), 0
    else:
        thresholds, at_least_n = (0.8, 0.8), 7

    if mixing is not None:
        thresholds, at_least_n = (0.0, 0.8), 7

    return thresholds, at_least_n


def _check_identified_snps(
    arrpath: Path,
    expected_top_indxs: List[int],
    top_row_grads_dict: Dict[str, List[int]],
    check_types: bool,
    at_least_n: Union[str, int] = "all",
):
    """
    NOTE: We have the `at_least_n` to check for a partial match of found SNPs. Why?
    Because when doing these tests, we are making the inputs per class the same
    for multiple spots, in the case of regression this leads the network to only "need"
    to identify a part of the SNPs to create an output (possibly because we are not
    using a "correctness criteria" for regression, like we do with classification (i.e.
    only gather activations for correctly predicted classes).

    :param arrpath: Path to the accumulated grads / activation array.
    :param expected_top_indxs: Expected SNPs to be identified.
    :param top_row_grads_dict: What row is expected to be activated per class.
    :param check_types:  Whether to check the SNP types as well as the SNPs themselves
    (i.e. homozygous reference, etc).
    :param at_least_n: At least how many SNPs must be identified to pass the test.
    :return:
    """
    top_grads_array = np.load(str(arrpath), allow_pickle=True)

    # get dict from array
    top_grads_dict: dict = top_grads_array[()]

    for cls in top_grads_dict.keys():
        actual_top = np.array(sorted(top_grads_dict[cls]["top_n_idxs"]))
        expected_top = np.array(expected_top_indxs)
        if at_least_n == "all":
            assert (actual_top == expected_top).all()
        else:
            assert len(set(actual_top).intersection(set(expected_top))) >= at_least_n

        if check_types:
            expected_top_rows = top_row_grads_dict[cls]
            _check_snp_types(
                cls_name=cls,
                top_grads_msk=top_grads_dict,
                expected_idxs=expected_top_rows,
                at_least_n=at_least_n,
            )


def _check_snp_types(cls_name: str, top_grads_msk, expected_idxs, at_least_n: int):
    """
    Adds an additional check for SNP types (i.e. reference homozygous, heterozygous,
    alternative homozygous, missing).

    Used when we have masked out the SNPs (otherwise the 0s in the one hot might have
    a high activation, since they're saying the same thing as a 1 being in a spot).
    """
    top_idxs = np.array(top_grads_msk[cls_name]["top_n_grads"].argmax(0))
    expected_idxs = np.array(expected_idxs)

    if at_least_n == "all":
        assert (top_idxs == expected_idxs).all()
    else:
        assert (top_idxs == expected_idxs).sum() >= at_least_n
