import pytest

from eir import train
from eir.setup import config
from eir.target_setup.target_label_setup import (
    MergedTargetLabels,
    gather_all_ids_from_output_configs,
    set_up_all_targets_wrapper,
)
from eir.train_utils.utils import get_run_folder


@pytest.fixture()
def create_test_labels(
    create_test_data, create_test_config: config.Configs
) -> MergedTargetLabels:
    c = create_test_config
    gc = c.global_config

    run_folder = get_run_folder(output_folder=gc.be.output_folder)

    all_array_ids = gather_all_ids_from_output_configs(output_configs=c.output_configs)
    train_ids, valid_ids = train.split_ids(
        ids=all_array_ids,
        valid_size=gc.be.valid_size,
    )

    target_labels = set_up_all_targets_wrapper(
        train_ids=train_ids,
        valid_ids=valid_ids,
        run_folder=run_folder,
        output_configs=c.output_configs,
        hooks=None,
    )

    return target_labels
