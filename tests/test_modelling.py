from pathlib import Path
from typing import Union, Tuple

import numpy as np
import pandas as pd
import pytest

from conftest import cleanup
from human_origins_supervised import train


@pytest.mark.parametrize(
    "create_test_data",
    [{"class_type": "binary"}, {"class_type": "multi"}],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_cl_args",
    [
        {"custom_cl_args": {"model_type": "cnn"}},
        {"custom_cl_args": {"model_type": "mlp"}},
    ],
    indirect=True,
)
def test_classification(keep_outputs, _prep_modelling_test_config):
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
    c = _prep_modelling_test_config
    cl_args = c.cl_args

    train.train_ignite(c)

    last_iter = len(c.train_loader) * cl_args.n_epochs
    column_name = cl_args.target_cat_columns[0]
    run_path = Path(f"runs/{cl_args.run_name}/")

    arrpath, arrpath_msk = _get_test_activation_arrs(
        run_path=run_path, iteration=last_iter, column_name=column_name
    )

    expected_top_indxs = list(range(50, 1000, 100))
    top_row_grads_dict = {"Asia": [1] * 10, "Europe": [2] * 10, "Africa": [0] * 10}

    for path in [arrpath, arrpath_msk]:
        check_types = True if path == arrpath_msk else False
        _check_identified_snps(
            path, expected_top_indxs, top_row_grads_dict, check_types
        )

    if not keep_outputs:
        cleanup(run_path)


@pytest.mark.parametrize(
    "create_test_data", [{"class_type": "regression"}], indirect=True
)
@pytest.mark.parametrize(
    "create_test_cl_args",
    [
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
    ],
    indirect=True,
)
def test_regression(keep_outputs, _prep_modelling_test_config):

    c = _prep_modelling_test_config
    cl_args = c.cl_args

    train.train_ignite(c)

    run_path = Path(f"runs/{cl_args.run_name}/")

    df_train = pd.read_csv(run_path / "t_average-loss_history.log")
    assert df_train.loc[:, "t_r2"].max() > 0.8

    df_valid = pd.read_csv(run_path / "v_average-loss_history.log")
    assert df_valid.loc[:, "v_r2"].max() > 0.8

    expected_top_indxs = list(range(50, 1000, 100))
    top_row_grads_dict = {"Regression": [0] * 10}

    last_iter = len(c.train_loader) * cl_args.n_epochs
    column_name = cl_args.target_con_columns[0]

    arrpath, arrpath_msk = _get_test_activation_arrs(
        run_path=run_path, iteration=last_iter, column_name=column_name
    )
    for path in [arrpath, arrpath_msk]:
        _check_identified_snps(path, expected_top_indxs, top_row_grads_dict, False, 1)

    if not keep_outputs:
        cleanup(run_path)


@pytest.fixture()
def _prep_modelling_test_config(
    create_test_data,
    create_test_cl_args,
    create_test_dloaders,
    create_test_model,
    create_test_optimizer,
    create_test_datasets,
):
    """
    Note that the fixtures used in this fixture get indirectly parametrized by
    test_classification and test_regression.
    """
    cl_args = create_test_cl_args
    train_dloader, valid_dloader, train_dataset, valid_dataset = create_test_dloaders
    model = create_test_model
    optimizer = create_test_optimizer
    criterions = train._get_criterions(train_dataset.target_columns)

    train_dataset, valid_dataset = create_test_datasets

    config = train.Config(
        cl_args=cl_args,
        train_loader=train_dloader,
        valid_loader=valid_dloader,
        valid_dataset=valid_dataset,
        model=model,
        optimizer=optimizer,
        criterions=criterions,
        labels_dict=train_dataset.labels_dict,
        target_transformers=train_dataset.target_transformers,
        target_columns=train_dataset.target_columns,
        data_width=cl_args.data_width,
    )

    return config


def _get_test_activation_arrs(
    run_path: Path, iteration: int, column_name: str
) -> Tuple[Path, Path]:
    results_path = run_path / f"results/{column_name}/samples/{iteration}"

    arrpath = results_path / "top_acts.npy"
    arrpath_masked = results_path / "top_acts_masked.npy"

    return arrpath, arrpath_masked


def _check_identified_snps(
    arrpath,
    expected_top_indxs,
    top_row_grads_dict,
    check_types,
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
    top_grads_array = np.load(arrpath, allow_pickle=True)

    # get dict from array
    top_grads_dict: dict = top_grads_array[()]

    for cls in top_grads_dict.keys():
        actual_top = np.array(sorted(top_grads_dict[cls]["top_n_idxs"]))
        expected_top = np.array(expected_top_indxs)
        if at_least_n == "all":
            assert (actual_top == expected_top).all()
        else:
            assert len(set(actual_top).intersection(set(expected_top))) >= at_least_n

        expected_top_rows = top_row_grads_dict[cls]

        if check_types:
            _check_snp_types(cls, top_grads_dict, expected_top_rows, at_least_n)


def _check_snp_types(cls_name, top_grads_msk, expected_idxs, at_least_n):
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
        assert len(set(top_idxs).intersection(set(expected_idxs))) >= at_least_n
