from unittest.mock import ANY, MagicMock, call, create_autospec, patch

import pytest
import torch
from lightning.fabric import Fabric
from torch import nn

from eir.train_utils import step_logic
from eir.train_utils.optim import AttrDelegatedSWAWrapper


@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "binary",
            "modalities": ("omics",),
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Default optimizer setup
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "linear",
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
                    }
                ],
            },
        },
        # With gradient accumulation
        {
            "injections": {
                "global_configs": {
                    "optimization": {
                        "gradient_accumulation_steps": 4,
                    }
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "linear",
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
                    }
                ],
            },
        },
    ],
    indirect=True,
)
def test_hook_default_optimizer_backward(prep_modelling_test_configs):
    experiment, *_ = prep_modelling_test_configs

    experiment.__dict__["optimizer"] = MagicMock()

    state = {"iteration": 1, "loss": MagicMock()}
    num_test_steps = 12

    for _i in range(num_test_steps):
        step_logic.hook_default_optimizer_backward(experiment=experiment, state=state)
        state["iteration"] += 1

    grad_acc_steps = experiment.configs.gc.opt.gradient_accumulation_steps

    if grad_acc_steps:
        assert state["loss"].__truediv__.call_count == num_test_steps
        assert experiment.optimizer.step.call_count == num_test_steps // grad_acc_steps
    else:
        assert state["loss"].backward.call_count == num_test_steps
        assert experiment.optimizer.step.call_count == num_test_steps


@pytest.mark.parametrize(
    "loss, grad_acc_steps, expected",
    [
        (torch.tensor(2.0), 2, torch.tensor(1.0)),
        (torch.tensor(2.0), 1, torch.tensor(2.0)),
        (torch.tensor(2.0), 0, torch.tensor(2.0)),
    ],
)
def test_maybe_scale_loss_with_grad_accumulation_steps(
    loss: torch.Tensor, grad_acc_steps: int, expected: torch.Tensor
):
    result = step_logic.maybe_scale_loss_with_grad_accumulation_steps(
        loss=loss, grad_acc_steps=grad_acc_steps
    )
    assert torch.isclose(result, expected)


def test_maybe_apply_gradient_noise_to_model():
    model = nn.Sequential(
        nn.Linear(2, 3, bias=False),
        nn.Linear(3, 4, bias=False),
    )

    for param in model.parameters():
        param.grad = torch.zeros_like(param, requires_grad=True)

    gradient_noise = 0.1
    step_logic.maybe_apply_gradient_noise_to_model(
        model=model, gradient_noise=gradient_noise
    )

    for _name, param in model.named_parameters():
        assert (param.grad.data != torch.zeros_like(param)).all()


def test_maybe_apply_gradient_clipping_to_model():
    model = nn.Sequential(
        nn.Linear(2, 3, bias=False),
        nn.Linear(3, 4, bias=False),
    )

    for param in model.parameters():
        param.grad = torch.zeros_like(param)

    gradient_clipping = 0.1

    with patch("eir.train_utils.step_logic.clip_grad_norm_") as mock_clip_grad_norm:
        step_logic.maybe_apply_gradient_clipping_to_model(
            model=model, gradient_clipping=gradient_clipping
        )

        expected_parameters = list(model.parameters())
        actual_parameters = list(mock_clip_grad_norm.call_args[1]["parameters"])
        assert expected_parameters == actual_parameters

        mock_clip_grad_norm.assert_called_once_with(
            parameters=ANY,
            max_norm=gradient_clipping,
        )


def test_maybe_update_model_parameters_with_swa_basics():
    model = create_autospec(spec=Fabric, instance=True)

    swa_module = create_autospec(spec=AttrDelegatedSWAWrapper, instance=True)
    model.module = swa_module

    inner_model = MagicMock()
    model.module.module = inner_model

    n_iter_before_swa = 10
    sample_interval = 5

    step_logic.maybe_update_model_parameters_with_swa(
        n_iter_before_swa=None,
        model=model,
        iteration=12,
        sample_interval=sample_interval,
    )
    swa_module.update_parameters.assert_not_called()

    step_logic.maybe_update_model_parameters_with_swa(
        n_iter_before_swa=n_iter_before_swa,
        model=model,
        iteration=8,
        sample_interval=sample_interval,
    )
    swa_module.update_parameters.assert_not_called()

    step_logic.maybe_update_model_parameters_with_swa(
        n_iter_before_swa=n_iter_before_swa,
        model=model,
        iteration=11,
        sample_interval=sample_interval,
    )
    swa_module.update_parameters.assert_not_called()

    step_logic.maybe_update_model_parameters_with_swa(
        n_iter_before_swa=n_iter_before_swa,
        model=model,
        iteration=15,
        sample_interval=sample_interval,
    )
    swa_module.update_parameters.assert_called_once_with(swa_module.module)


def test_maybe_update_model_parameters_with_swa_multiple_iterations():
    model = create_autospec(spec=Fabric, instance=True)

    swa_module = create_autospec(spec=AttrDelegatedSWAWrapper, instance=True)
    model.module = swa_module

    inner_model = MagicMock()
    model.module.module = inner_model

    n_iter_before_swa = 10
    sample_interval = 5
    num_iterations = 30

    update_parameters_call_count = 0

    for iteration in range(num_iterations):
        step_logic.maybe_update_model_parameters_with_swa(
            n_iter_before_swa=n_iter_before_swa,
            model=model,
            iteration=iteration,
            sample_interval=sample_interval,
        )

        if iteration >= n_iter_before_swa and iteration % sample_interval == 0:
            update_parameters_call_count += 1

    swa_module.update_parameters.assert_has_calls(
        [call(swa_module.module)] * update_parameters_call_count
    )


@pytest.mark.parametrize(
    "iteration, grad_acc_steps, expected",
    [
        (1, 0, True),
        (1, 1, True),
        (1, 2, False),
        (2, 2, True),
        (3, 2, False),
    ],
)
def test_should_perform_optimizer_step(
    iteration: int, grad_acc_steps: int, expected: bool
):
    result = step_logic.should_perform_optimizer_step(
        iteration=iteration, grad_acc_steps=grad_acc_steps
    )
    assert result == expected
