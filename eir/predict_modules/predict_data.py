from typing import Union

from eir.data_load import datasets
from eir.data_load.datasets import al_datasets
from eir.data_load.label_setup import al_label_dict
from eir.setup.config import Configs
from eir.setup.input_setup import al_input_objects_as_dict
from eir.setup.output_setup import al_output_objects_as_dict


def set_up_default_dataset(
    configs: Configs,
    target_labels_dict: Union[None, al_label_dict],
    inputs_as_dict: al_input_objects_as_dict,
    outputs_as_dict: al_output_objects_as_dict,
) -> al_datasets:
    test_dataset_kwargs = datasets.construct_default_dataset_kwargs_from_cl_args(
        target_labels_dict=target_labels_dict,
        outputs=outputs_as_dict,
        inputs=inputs_as_dict,
        test_mode=True,
    )

    test_dataset = datasets.DiskDataset(**test_dataset_kwargs)

    return test_dataset
