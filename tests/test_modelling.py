from pathlib import Path
from typing import Union, Tuple, Dict, List

import numpy as np
import pandas as pd
import pytest

from eir import train
from tests.conftest import cleanup, ModelTestConfig


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
@pytest.mark.parametrize(
    "create_test_cl_args",
    [
        {"custom_cl_args": {"model_type": "mlp", "lr": 1e-03}},
        {"custom_cl_args": {"model_type": "cnn", "lr": 1e-03, "fc_do": 0.25}},
    ],
    indirect=True,
)
def test_classification_nonlinear(keep_outputs, prep_modelling_test_configs):
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
    config, test_config = prep_modelling_test_configs

    train.train(config)

    target_column = config.cl_args.target_cat_columns[0]

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


@pytest.mark.parametrize("create_test_data", [{"task_type": "binary"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_cl_args", [{"custom_cl_args": {"model_type": "linear"}}], indirect=True
)
def test_classification_linear(keep_outputs, prep_modelling_test_configs):
    """
    Largely duplicated from `test_classification_nonlinear` as currently linear model
    only support regression or binary classification.
    """
    config, test_config = prep_modelling_test_configs

    train.train(config)

    target_column = config.cl_args.target_cat_columns[0]

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
    "create_test_cl_args",
    [
        {
            "custom_cl_args": {
                "model_type": "linear",
                "lr": 1e-03,
                "target_cat_columns": [],
                "target_con_columns": ["Height"],
            }
        },
        {
            "custom_cl_args": {
                "model_type": "cnn",
                "target_cat_columns": [],
                "target_con_columns": ["Height"],
            }
        },
        {
            "custom_cl_args": {
                "model_type": "mlp",
                "target_cat_columns": [],
                "target_con_columns": ["Height"],
            }
        },
        {
            "custom_cl_args": {
                "model_type": "cnn",
                "target_cat_columns": [],
                "target_con_columns": ["Height"],
                "n_epochs": 10,
                "lr_schedule": "cycle",
                "run_name": "test_cycle",
            }
        },
    ],
    indirect=True,
)
def test_regression(keep_outputs, prep_modelling_test_configs):
    config, test_config = prep_modelling_test_configs

    train.train(config)

    target_column = config.cl_args.target_con_columns[0]

    # linear regression performs slightly worse, but we don't want to lower expectations
    # other models
    thresholds = (0.70, 0.70) if config.cl_args.model_type == "linear" else (0.8, 0.8)
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
    "create_test_cl_args",
    [
        {  # Case 0: Check that we add and use extra inputs.
            "custom_cl_args": {
                "model_type": "cnn",
                "target_cat_columns": ["Origin"],
                "extra_con_columns": ["ExtraTarget"],
                "extra_cat_columns": ["OriginExtraCol"],
                "target_con_columns": ["Height"],
                "run_name": "extra_inputs",
            }
        },
        {  # Case 1: Normal multi task with CNN
            "custom_cl_args": {
                "model_type": "cnn",
                "target_cat_columns": ["Origin"],
                "target_con_columns": ["Height", "ExtraTarget"],
            }
        },
        {  # Case 2: Normal multi task with MLP, note we have to reduce the LR for
            # stability and add L1 for regularization
            "custom_cl_args": {
                "model_type": "mlp",
                "l1": 1e-3,
                "lr": 1e-3,
                "target_cat_columns": ["Origin"],
                "target_con_columns": ["Height", "ExtraTarget"],
            }
        },
        {  # Case 3: Using the split model
            "custom_cl_args": {
                "model_type": "mlp-split",
                "l1": 1e-3,
                "lr": 1e-3,
                "fc_repr_dim": 8,
                "split_mlp_num_splits": 64,
                "target_cat_columns": ["Origin"],
                "target_con_columns": ["Height", "ExtraTarget"],
            }
        },
        {  # Case 4: Using the fully-split model
            "custom_cl_args": {
                "model_type": "genome-local-net",
                "lr": 1e-3,
                "kernel_width": 8,
                "channel_exp_base": 2,
                "layers": [1],
                "n_epochs": 6,
                "target_cat_columns": ["Origin"],
                "target_con_columns": ["Height", "ExtraTarget"],
            }
        },
        {  # Case 5: MGMOE
            "custom_cl_args": {
                "fusion_model_type": "mgmoe",
                "model_type": "mlp-split",
                "lr": 1e-3,
                "fc_repr_dim": 8,
                "split_mlp_num_splits": 64,
                "run_name": "test_mgmoe_fusion",
                "target_cat_columns": ["Origin"],
                "target_con_columns": ["Height", "ExtraTarget"],
            }
        },
        {  # Case 6: Fully split with mixup
            "custom_cl_args": {
                "model_type": "genome-local-net",
                "lr": 1e-3,
                "kernel_width": 8,
                "channel_exp_base": 2,
                "mixing_type": "cutmix-uniform",
                "layers": [1],
                "mixing_alpha": 1.0,
                "n_epochs": 10,
                "target_cat_columns": ["Origin"],
                "run_name": "test-mixing",
                "target_con_columns": ["Height", "ExtraTarget"],
            }
        },
    ],
    indirect=True,
)
def test_multi_task(keep_outputs, prep_modelling_test_configs):
    config, test_config = prep_modelling_test_configs
    cl_args = config.cl_args

    train.train(config)

    extra_columns = cl_args.extra_con_columns + cl_args.extra_cat_columns
    for cat_column in config.cl_args.target_cat_columns:
        threshold, at_least_n = _get_multi_task_test_args(
            extra_columns=extra_columns,
            target_copy="OriginExtraCol",
            mixing=cl_args.mixing_type,
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

    for con_column in config.cl_args.target_con_columns:
        threshold, at_least_n = _get_multi_task_test_args(
            extra_columns=extra_columns,
            target_copy="ExtraTarget",
            mixing=cl_args.mixing_type,
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
        thresholds, at_least_n = (0.8, 0.8), 8

    if mixing is not None:
        thresholds, at_least_n = (0.0, 0.8), 8

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
            _check_snp_types(cls, top_grads_dict, expected_top_rows, at_least_n)


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
