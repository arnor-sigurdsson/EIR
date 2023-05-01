from typing import Tuple

import pytest
import torch.utils
from torch.utils.data import DataLoader

import eir.setup
from eir.data_load import datasets
from eir.setup import config
from eir.setup.output_setup import set_up_outputs_for_training


@pytest.fixture()
def create_test_datasets(
    create_test_data,
    create_test_labels,
    create_test_config: config.Configs,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    configs = create_test_config
    target_labels = create_test_labels

    inputs = eir.setup.input_setup.set_up_inputs_for_training(
        inputs_configs=configs.input_configs,
        train_ids=tuple(target_labels.train_labels.keys()),
        valid_ids=tuple(target_labels.valid_labels.keys()),
        hooks=None,
    )

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=create_test_config.output_configs,
        target_transformers=target_labels.label_transformers,
    )

    train_dataset, valid_dataset = datasets.set_up_datasets_from_configs(
        configs=configs,
        target_labels=target_labels,
        inputs_as_dict=inputs,
        outputs_as_dict=outputs_as_dict,
    )

    return train_dataset, valid_dataset


@pytest.fixture()
def create_test_dloaders(create_test_config: config.Configs, create_test_datasets):
    c = create_test_config
    gc = c.global_config
    train_dataset, valid_dataset = create_test_datasets

    train_dloader = DataLoader(
        train_dataset, batch_size=gc.batch_size, shuffle=True, drop_last=True
    )

    valid_dloader = DataLoader(
        valid_dataset, batch_size=gc.batch_size, shuffle=False, drop_last=False
    )

    return train_dloader, valid_dloader, train_dataset, valid_dataset
