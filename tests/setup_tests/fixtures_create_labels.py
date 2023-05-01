import pytest

from eir import train
from eir.setup import config
from eir.train_utils.utils import get_run_folder


@pytest.fixture()
def create_test_labels(
    create_test_data, create_test_config: config.Configs
) -> train.MergedTargetLabels:
    c = create_test_config
    gc = c.global_config

    run_folder = get_run_folder(output_folder=gc.output_folder)

    all_array_ids = train.gather_all_ids_from_output_configs(
        output_configs=c.output_configs
    )
    train_ids, valid_ids = train.split_ids(ids=all_array_ids, valid_size=gc.valid_size)

    target_labels_info = train.get_tabular_target_file_infos(
        output_configs=c.output_configs
    )
    target_labels = train.set_up_tabular_target_labels_wrapper(
        tabular_target_file_infos=target_labels_info,
        custom_label_ops=None,
        train_ids=train_ids,
        valid_ids=valid_ids,
    )

    train.save_transformer_set(
        transformers_per_source=target_labels.label_transformers, run_folder=run_folder
    )

    return target_labels
