from copy import deepcopy
from math import isclose

import numpy as np
import pytest
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

from eir.models.output.tabular.mlp_residual import ResidualMLPOutputModuleConfig
from eir.models.output.tabular.tabular_output_modules import TabularOutputModuleConfig
from eir.setup.output_setup_modules.tabular_output_setup import (
    ComputedTabularOutputInfo,
)
from eir.setup.schema_modules.output_schemas_tabular import TabularOutputTypeConfig
from eir.setup.schemas import OutputConfig, OutputInfoConfig
from eir.target_setup.target_label_setup import get_missing_targets_info
from eir.train_utils import metrics
from eir.train_utils.criteria import get_criteria


def test_calculate_batch_metrics():
    test_batch_metrics_kwargs = get_calculate_batch_metrics_data_test_kwargs()
    test_batch_metrics = metrics.calculate_batch_metrics(**test_batch_metrics_kwargs)

    loss_kwargs = _get_add_loss_to_metrics_kwargs(
        outputs_as_dict=test_batch_metrics_kwargs["outputs_as_dict"]
    )
    test_batch_metrics_w_loss = metrics.add_loss_to_metrics(
        metric_dict=test_batch_metrics, **loss_kwargs
    )

    test_batch_metrics_w_loss = test_batch_metrics_w_loss["test_output_tabular"]

    assert test_batch_metrics_w_loss["Origin"]["test_output_tabular_Origin_mcc"] == 1.0
    assert test_batch_metrics_w_loss["Origin"]["test_output_tabular_Origin_loss"] == 0.0

    assert test_batch_metrics_w_loss["BMI"]["test_output_tabular_BMI_r2"] == 1.0
    assert test_batch_metrics_w_loss["BMI"]["test_output_tabular_BMI_rmse"] == 0.0

    # sometimes slight numerical instability with scipy pearsonr
    assert isclose(test_batch_metrics_w_loss["BMI"]["test_output_tabular_BMI_pcc"], 1.0)
    assert test_batch_metrics_w_loss["BMI"]["test_output_tabular_BMI_loss"] == 0.0

    assert test_batch_metrics_w_loss["Height"]["test_output_tabular_Height_r2"] < 0
    assert test_batch_metrics_w_loss["Height"]["test_output_tabular_Height_rmse"] > 0.0
    assert isclose(
        test_batch_metrics_w_loss["Height"]["test_output_tabular_Height_pcc"], -1.0
    )
    assert test_batch_metrics_w_loss["Height"]["test_output_tabular_Height_loss"] == 1.0


def get_calculate_batch_metrics_data_test_kwargs():
    standard_scaler_fit_arr = [[0.0], [1.0], [2.0]]

    outputs = {
        "test_output_tabular": {
            "Origin": torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            "HairColor": torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            "BMI": torch.tensor([1.0, 2.0, 3.0]).unsqueeze(1),
            "Height": torch.tensor([1.0, 2.0, 3.0]).unsqueeze(1),
        }
    }

    labels = {
        "test_output_tabular": {
            "Origin": torch.tensor([0, 1, 2]),
            "HairColor": torch.tensor([0, 1, 2]),
            "BMI": torch.tensor([1.0, 2.0, 3.0]).unsqueeze(1),
            "Height": torch.tensor([-1.0, -2.0, -3.0]).unsqueeze(1),
        }
    }

    target_transformers = {
        "test_output_tabular": {
            "Origin": LabelEncoder().fit([1, 2, 3]),
            "HairColor": LabelEncoder().fit([1, 2, 3]),
            "BMI": StandardScaler().fit(standard_scaler_fit_arr),
            "Height": StandardScaler().fit(standard_scaler_fit_arr),
        }
    }
    metrics_ = metrics.get_default_metrics(
        target_transformers=target_transformers,
        cat_metrics=["mcc", "acc", "roc-auc-macro", "ap-macro"],
        con_metrics=["loss", "pcc", "r2", "rmse"],
        cat_averaging_metrics=None,
        con_averaging_metrics=None,
        output_configs=[],
    )

    test_outputs_as_dict = _get_metrics_test_module_test_outputs_as_dict()

    batch_metrics_function_kwargs = {
        "outputs_as_dict": test_outputs_as_dict,
        "outputs": outputs,
        "labels": labels,
        "mode": "val",
        "metric_record_dict": metrics_,
    }

    return batch_metrics_function_kwargs


def _get_metrics_test_module_test_outputs_as_dict():
    test_outputs_as_dict = {
        "test_output_tabular": ComputedTabularOutputInfo(
            output_config=OutputConfig(
                output_info=OutputInfoConfig(
                    output_name="test_output_tabular",
                    output_type="tabular",
                    output_source="None",
                ),
                output_type_info=TabularOutputTypeConfig(
                    target_con_columns=["Height", "BMI"],
                    target_cat_columns=["Origin", "HairColor"],
                ),
                model_config=TabularOutputModuleConfig(
                    model_init_config=ResidualMLPOutputModuleConfig()
                ),
            ),
            num_outputs_per_target={
                "Origin": 5,
                "HairColor": 5,
                "BMI": 1,
                "Height": 1,
            },
            target_columns={
                "con": ["Height", "BMI"],
                "cat": ["Origin", "HairColor"],
            },
            target_transformers={},
        )
    }
    return test_outputs_as_dict


def _get_add_loss_to_metrics_kwargs(outputs_as_dict):
    losses = {
        "test_output_tabular": {
            "Origin": torch.tensor(0.0),
            "HairColor": torch.tensor(0.0),
            "BMI": torch.tensor(0.0),
            "Height": torch.tensor(1.0),
        }
    }

    return {"losses": losses, "outputs_as_dict": outputs_as_dict}


def test_calc_rmse_multiple_samples():
    outputs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    labels = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    scaler = StandardScaler()
    scaler.fit(labels.reshape(-1, 1))
    target_transformers = {"output1": {"column1": scaler}}

    rmse = metrics.calc_rmse(
        outputs=outputs,
        labels=labels,
        target_transformers=target_transformers,
        output_name="output1",
        column_name="column1",
    )
    labels_transformed = scaler.inverse_transform(labels.reshape(-1, 1)).squeeze()
    outputs_transformed = scaler.inverse_transform(outputs.reshape(-1, 1)).squeeze()
    expected_rmse = np.sqrt(mean_squared_error(labels_transformed, outputs_transformed))

    assert np.isclose(rmse, expected_rmse)


def test_calc_rmse_single_sample():
    outputs = np.array([1.0])
    labels = np.array([1.1])
    scaler = StandardScaler()
    scaler.fit(labels.reshape(-1, 1))
    target_transformers = {"output1": {"column1": scaler}}

    rmse = metrics.calc_rmse(
        outputs=outputs,
        labels=labels,
        target_transformers=target_transformers,
        output_name="output1",
        column_name="column1",
    )
    expected_rmse = np.abs(labels[0] - outputs[0])

    assert np.isclose(rmse, expected_rmse)


def test_calc_rmse_no_samples():
    outputs = np.array([])
    labels = np.array([])
    scaler = StandardScaler()
    target_transformers = {"output1": {"column1": scaler}}

    rmse = metrics.calc_rmse(
        outputs=outputs,
        labels=labels,
        target_transformers=target_transformers,
        output_name="output1",
        column_name="column1",
    )

    assert np.isnan(rmse)


def test_calculate_losses_good():
    """
    Note that CrossEntropy applies LogSoftmax() before calculating the NLLLoss().

    We expect the CrossEntropyLosses to be around 0.9048

        >>> loss = torch.nn.CrossEntropyLoss()
        >>> input_ = torch.zeros(1, 5)
        >>> input_[0, 0] = 1
        >>> target = torch.zeros(1, dtype=torch.long)
        >>> loss(input_, target)
        tensor(0.9048)
    """

    common_values = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    test_criteria, test_labels, test_outputs = set_up_calculate_losses_data(
        label_values=common_values,
        output_values=common_values,
    )

    perfect_pred_loss = metrics.calculate_prediction_losses(
        criteria=test_criteria,
        targets=test_labels,
        inputs=test_outputs,
    )

    assert perfect_pred_loss["test_output_tabular"]["Height"].item() == 0.0
    assert perfect_pred_loss["test_output_tabular"]["BMI"].item() == 0.0

    assert 0.904 < perfect_pred_loss["test_output_tabular"]["Origin"].item() < 0.905


def test_calculate_losses_bad():
    # diff of 2 between each pair, RMSE expected to be 4.0
    label_values = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    output_values = torch.tensor([2, 3, 4, 5, 6], dtype=torch.int64)
    test_criteria, test_labels, test_outputs = set_up_calculate_losses_data(
        label_values=label_values,
        output_values=output_values,
    )

    bad_prediction_loss = metrics.calculate_prediction_losses(
        criteria=test_criteria,
        targets=test_labels,
        inputs=test_outputs,
    )

    expected_rmse = 4.0
    assert bad_prediction_loss["test_output_tabular"]["Height"].item() == expected_rmse
    assert bad_prediction_loss["test_output_tabular"]["BMI"].item() == expected_rmse

    # check that the loss is more than upper bound (0.905) in perfect case
    perfect_upper_bound = 0.905
    assert (
        bad_prediction_loss["test_output_tabular"]["Origin"].item()
        > perfect_upper_bound
    )


def set_up_calculate_losses_data(
    label_values: torch.Tensor,
    output_values: torch.Tensor,
):
    def generate_base_dict(values: torch.Tensor):
        base_dict = {
            "Height": deepcopy(values).to(dtype=torch.float32),
            "BMI": deepcopy(values).to(dtype=torch.float32),
            "Origin": deepcopy(values),
            "HairColor": deepcopy(values),
        }

        return base_dict

    test_outputs_as_dict = _get_metrics_test_module_test_outputs_as_dict()
    test_criteria = get_criteria(outputs_as_dict=test_outputs_as_dict)
    test_labels = generate_base_dict(values=label_values)

    test_outputs = generate_base_dict(output_values)

    one_hot = torch.nn.functional.one_hot
    test_outputs["Origin"] = one_hot(test_outputs["Origin"])
    test_outputs["Origin"] = test_outputs["Origin"].to(dtype=torch.float32)

    test_outputs["HairColor"] = one_hot(test_outputs["HairColor"])
    test_outputs["HairColor"] = test_outputs["HairColor"].to(dtype=torch.float32)

    return (
        test_criteria,
        {"test_output_tabular": test_labels},
        {"test_output_tabular": test_outputs},
    )


def test_aggregate_losses():
    # expected average of [0,1,2,3,4] = 2.0
    losses_dict = {
        "test_output_tabular": {
            str(i): torch.tensor(i, dtype=torch.float32) for i in range(5)
        }
    }

    test_aggregated_losses = metrics.aggregate_losses(losses_dict)
    assert test_aggregated_losses.item() == 2.0


@pytest.fixture()
def get_l1_test_model():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.fc_1 = torch.nn.Linear(10, 1)
            self.l1_penalized_weights = self.fc_1.weight

        def forward(self, x):
            return x

    return TestModel()


def test_get_model_l1_loss(get_l1_test_model):
    test_model = get_l1_test_model

    torch.nn.init.ones_(test_model.fc_1.weight)
    l1_loss = metrics.get_model_l1_loss(model=test_model, l1_weight=1.0)

    assert l1_loss == 10.0

    torch.nn.init.zeros_(test_model.fc_1.weight)
    l1_loss = metrics.get_model_l1_loss(model=test_model, l1_weight=1.0)

    assert l1_loss == 0.0


@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "binary",
            "modalities": (
                "omics",
                "tabular",
            ),
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: Omics Feature extractor: MLP
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "linear",
                            "model_init_config": {"l1": 1e-02},
                        },
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        },
        # Case 2: Omics Feature extractor: CNN
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {
                                "rb_do": 0.25,
                                "channel_exp_base": 3,
                                "l1": 1e-02,
                            },
                        },
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        },
        # Case 3: Omics feature extractor: Simple LCL
        {
            "injections": {
                "global_configs": {"optimization": {"lr": 1e-03}},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "lcl-simple",
                            "model_init_config": {
                                "fc_repr_dim": 8,
                                "num_lcl_chunks": 64,
                                "l1": 1e-03,
                            },
                        },
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        },
        # Case 4: Omics feature extractor: GLN
        {
            "injections": {
                "global_configs": {"optimization": {"lr": 1e-03}},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "genome-local-net",
                            "model_init_config": {
                                "kernel_width": 8,
                                "channel_exp_base": 2,
                                "l1": 1e-03,
                                "rb_do": 0.20,
                            },
                        },
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        },
        # Case 5: Omics feature extractor: Linear
        {
            "injections": {
                "global_configs": {"optimization": {"lr": 1e-03}},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "linear",
                            "model_init_config": {"l1": 1e-02},
                        },
                    },
                ],
                "fusion_configs": {
                    "model_type": "identity",
                },
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        },
        # Case 6: Tabular feature extractor
        {
            "injections": {
                "global_configs": {
                    "basic_experiment": {
                        "output_folder": "extra_inputs",
                    }
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {"l1": 0.0},
                        },
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "input_cat_columns": ["OriginExtraCol"],
                            "input_con_columns": ["ExtraTarget"],
                        },
                        "model_config": {
                            "model_type": "tabular",
                            "model_init_config": {"l1": 1e-02},
                        },
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        },
    ],
    indirect=True,
)
def test_hook_add_l1_loss(prep_modelling_test_configs):
    experiment, *_ = prep_modelling_test_configs

    test_state = {"loss": torch.tensor(0.0)}
    state_update = metrics.hook_add_l1_loss(experiment=experiment, state=test_state)

    l1_loss = state_update["loss"]
    assert l1_loss > 0.0


@pytest.fixture
def setup_data():
    batch_ids = ["id1", "id2", "id3"]
    model_outputs = {
        "output1": {
            "inner1": torch.randn(3, 10),
            "inner2": torch.randn(3, 20),
        }
    }
    target_labels = {
        "output1": {
            "inner1": torch.randint(0, 2, (3,)),
            "inner2": torch.randint(0, 2, (3,)),
        }
    }

    missing_per_modality = {"output1": set()}
    missing_ids_info = get_missing_targets_info(
        missing_ids_per_modality=missing_per_modality,
    )
    return batch_ids, model_outputs, target_labels, missing_ids_info


def test_no_missing_data(setup_data):
    batch_ids, model_outputs, target_labels, missing_ids_info = setup_data
    result = metrics.filter_missing_outputs_and_labels(
        batch_ids=batch_ids,
        model_outputs=model_outputs,
        target_labels=target_labels,
        missing_ids_info=missing_ids_info,
        with_labels=True,
    )
    assert isinstance(result, metrics.FilteredOutputsAndLabels), (
        "Result should be an instance of FilteredOutputsAndLabels"
    )
    assert "output1" in result.model_outputs, (
        "Output1 should be present in the filtered outputs"
    )
    assert "inner1" in result.model_outputs["output1"], (
        "Inner1 should be present in the filtered outputs of output1"
    )
    assert torch.equal(
        result.model_outputs["output1"]["inner1"], model_outputs["output1"]["inner1"]
    ), "Filtered output tensor should match the original"
    assert torch.equal(
        result.target_labels["output1"]["inner1"], target_labels["output1"]["inner1"]
    ), "Filtered labels should match the original"


def test_all_ids_missing(setup_data):
    batch_ids, model_outputs, target_labels, _ = setup_data
    missing_ids_info = get_missing_targets_info(
        missing_ids_per_modality={"output1": set(batch_ids)},
    )

    result = metrics.filter_missing_outputs_and_labels(
        batch_ids=batch_ids,
        model_outputs=model_outputs,
        target_labels=target_labels,
        missing_ids_info=missing_ids_info,
        with_labels=True,
    )

    assert len(result.model_outputs["output1"]["inner1"]) == 0, (
        "Should return an empty tensor for outputs when all IDs are missing"
    )
    assert len(result.target_labels["output1"]["inner1"]) == 0, (
        "Should return an empty tensor for labels when all IDs are missing"
    )
    assert len(result.model_outputs["output1"]["inner2"]) == 0, (
        "Should return an empty tensor for outputs when all IDs are missing"
    )
    assert len(result.target_labels["output1"]["inner2"]) == 0, (
        "Should return an empty tensor for labels when all IDs are missing"
    )
