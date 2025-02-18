import pytest
from torch.utils.data import WeightedRandomSampler

from eir.data_load.data_utils import get_finite_train_sampler


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: Linear
        {
            "injections": {
                "global_configs": {
                    "optimization": {"lr": 1e-03},
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    }
                ],
            },
        },
    ],
    indirect=True,
)
def test_get_train_sampler(create_test_data, create_test_datasets, create_test_config):
    test_config = create_test_config
    gc = test_config.global_config

    train_dataset, *_ = create_test_datasets
    gc.weighted_sampling_columns = ["test_output_tabular__Origin"]

    test_sampler = get_finite_train_sampler(
        columns_to_sample=gc.weighted_sampling_columns,
        train_dataset=train_dataset,
    )
    assert isinstance(test_sampler, WeightedRandomSampler)

    gc.weighted_sampling_columns = None
    test_sampler = get_finite_train_sampler(
        columns_to_sample=gc.weighted_sampling_columns,
        train_dataset=train_dataset,
    )
    assert test_sampler is None
