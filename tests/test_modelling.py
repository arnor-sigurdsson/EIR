from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pytest
from torch.nn import CrossEntropyLoss, MSELoss

from human_origins_supervised import train
from conftest import cleanup


def check_snp_types(cls_name, top_grads_msk, expected_idxs, at_least_n):
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


def check_identified_snps(
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
            check_snp_types(cls, top_grads_dict, expected_top_rows, at_least_n)


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
def test_classification_snp_identification(
    create_test_data,
    create_test_cl_args,
    create_test_dloaders,
    create_test_model,
    create_test_optimizer,
    create_test_dataset,
    keep_outputs,
):
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
    """
    cl_args = create_test_cl_args
    train_dloader, valid_dloader, train_dataset, valid_dataset = create_test_dloaders
    model = create_test_model
    optimizer = create_test_optimizer
    criterion = CrossEntropyLoss()

    train_dataset, valid_dataset = create_test_dataset
    target_transformer = train_dataset.target_transformer

    run_path = Path(f"runs/{cl_args.run_name}/")

    config = train.Config(
        cl_args,
        train_dloader,
        valid_dloader,
        valid_dataset,
        model,
        optimizer,
        criterion,
        train_dataset.labels_dict,
        target_transformer,
        cl_args.data_width,
    )

    train.train_ignite(config)

    last_iter = len(train_dloader) * cl_args.n_epochs
    arrpath = run_path / f"samples/{last_iter}/top_acts.npy"
    arrpath_msk = run_path / f"samples/{last_iter}/top_grads_masked.npy"
    expected_top_indxs = list(range(50, 1000, 100))
    top_row_grads_dict = {"Asia": [1] * 10, "Europe": [2] * 10, "Africa": [0] * 10}

    for path in [arrpath, arrpath_msk]:
        check_types = True if path == arrpath_msk else False
        check_identified_snps(path, expected_top_indxs, top_row_grads_dict, check_types)

    if not keep_outputs:
        cleanup(run_path)


@pytest.mark.parametrize(
    "create_test_data", [{"class_type": "regression"}], indirect=True
)
@pytest.mark.parametrize(
    "create_test_cl_args",
    [
        {"custom_cl_args": {"model_type": "cnn"}},
        {"custom_cl_args": {"model_type": "mlp"}},
    ],
    indirect=True,
)
def test_regression(
    create_test_data,
    create_test_cl_args,
    create_test_dloaders,
    create_test_model,
    create_test_optimizer,
    create_test_dataset,
    keep_outputs,
):

    cl_args = create_test_cl_args
    run_path = Path(f"runs/{cl_args.run_name}/")

    train_dloader, valid_dloader, train_dataset, valid_dataset = create_test_dloaders
    model = create_test_model
    optimizer = create_test_optimizer
    criterion = CrossEntropyLoss() if cl_args.model_task == "cls" else MSELoss()

    train_dataset, valid_dataset = create_test_dataset
    target_transformer = train_dataset.target_transformer

    config = train.Config(
        cl_args,
        train_dloader,
        valid_dloader,
        valid_dataset,
        model,
        optimizer,
        criterion,
        train_dataset.labels_dict,
        target_transformer,
        cl_args.data_width,
    )

    train.train_ignite(config)

    df = pd.read_csv(run_path / "training_history.log")

    assert df.loc[:, "t_r2"].max() > 0.8
    # lower due to overfitting on training set
    assert df.loc[:, "v_r2"].max() > 0.8

    last_iter = len(train_dloader) * cl_args.n_epochs
    arrpath = run_path / f"samples/{last_iter}/top_acts.npy"
    arrpath_msk = run_path / f"samples/{last_iter}/top_grads_masked.npy"
    expected_top_indxs = list(range(50, 1000, 100))
    top_row_grads_dict = {"Regression": [0] * 10}

    for path in [arrpath, arrpath_msk]:
        check_identified_snps(path, expected_top_indxs, top_row_grads_dict, False, 1)

    if not keep_outputs:
        cleanup(run_path)
