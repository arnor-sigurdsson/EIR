from typing import Optional

from eir.data_load import datasets
from eir.data_load.datasets import al_local_datasets
from eir.data_load.label_setup import al_target_labels
from eir.setup.config import Configs
from eir.setup.input_setup import al_input_objects_as_dict
from eir.setup.output_setup import al_output_objects_as_dict
from eir.target_setup.target_label_setup import MissingTargetsInfo


def set_up_default_dataset(
    configs: Configs,
    target_labels_df: Optional[al_target_labels],
    inputs_as_dict: al_input_objects_as_dict,
    outputs_as_dict: al_output_objects_as_dict,
    missing_ids_per_output: MissingTargetsInfo,
) -> al_local_datasets:
    test_dataset_kwargs = datasets.construct_default_dataset_kwargs_from_cl_args(
        target_labels_df=target_labels_df,
        outputs=outputs_as_dict,
        inputs=inputs_as_dict,
        test_mode=True,
        missing_ids_per_output=missing_ids_per_output,
    )

    test_dataset = datasets.DiskDataset(**test_dataset_kwargs)

    return test_dataset
