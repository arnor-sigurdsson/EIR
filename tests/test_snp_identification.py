from pathlib import Path
from shutil import rmtree

import numpy as np
import pytest
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from human_origins_supervised import train
from human_origins_supervised.data_load import datasets
from human_origins_supervised.models.models import Model


@pytest.fixture()
def create_test_dataset(create_test_data, create_test_cl_args):
    path, test_data_params = create_test_data

    cl_args = create_test_cl_args
    cl_args.data_folder = str(path / "test_arrays")
    cl_args.data_type = test_data_params["data_type"]

    train_dataset, valid_dataset = datasets.set_up_datasets(cl_args, valid_fraction=0.3)

    return train_dataset, valid_dataset


@pytest.fixture()
def create_test_dloaders(create_test_dataset):
    train_dataset, valid_dataset = create_test_dataset

    train_dloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    valid_dloader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

    return train_dloader, valid_dloader, train_dataset, valid_dataset


@pytest.fixture()
def create_test_model(create_test_cl_args, create_test_dataset):
    cl_args = create_test_cl_args
    train_dataset, _ = create_test_dataset

    model = Model(cl_args, train_dataset.num_classes)

    return model


@pytest.fixture()
def create_test_optimizer(create_test_cl_args, create_test_model):
    cl_args = create_test_cl_args
    model = create_test_model
    optimizer = optim.Adam(
        model.parameters(),
        lr=cl_args.lr,
        betas=(cl_args.b1, cl_args.b2),
        weight_decay=0.001,
    )

    return optimizer


def test_identification(
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
    label_encoder = train_dataset.label_encoder

    run_path = Path(f"models/{cl_args.run_name}/")

    config = train.Config(
        cl_args,
        train_dloader,
        valid_dloader,
        valid_dataset,
        model,
        optimizer,
        criterion,
        label_encoder,
        cl_args.data_width,
    )

    def cleanup():
        rmtree(run_path)

    def check_snp_types(cls_name, top_grads_msk, expected_idxs):
        top_idxs = list(top_grads_msk[cls_name]["top_n_grads"].argmax(0))
        assert top_idxs == expected_idxs

    if run_path.exists():
        cleanup()

    train.train_ignite(config)

    arrpath = run_path / f"samples/{cl_args.n_epochs}/top_acts.npy"
    top_grads_array = np.load(arrpath, allow_pickle=True)
    top_grads_dict: dict = top_grads_array[()]

    expected_top_indxs = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]

    arrpath_msk = run_path / f"samples/{cl_args.n_epochs}/top_grads_masked.npy"
    top_grads_msk_array = np.load(arrpath_msk, allow_pickle=True)
    top_grads_msk_dict: dict = top_grads_msk_array[()]

    top_row_grads_dict = {"Asia": [1] * 10, "Europe": [2] * 10, "Africa": [0] * 10}

    for cls in top_grads_dict.keys():
        assert sorted(top_grads_dict[cls]["top_n_idxs"]) == expected_top_indxs
        assert sorted(top_grads_msk_dict[cls]["top_n_idxs"]) == expected_top_indxs

        expected_top_rows = top_row_grads_dict[cls]
        check_snp_types(cls, top_grads_msk_dict, expected_top_rows)

    if not keep_outputs:
        cleanup()
