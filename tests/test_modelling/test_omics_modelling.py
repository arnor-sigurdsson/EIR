from pathlib import Path
from typing import Union, Tuple, Dict, List, Sequence, TYPE_CHECKING

import numpy as np
import pytest

from eir import train
from eir.setup.config import Configs
from tests.conftest import should_skip_in_gha_macos
from tests.test_modelling.test_modelling_utils import (
    check_performance_result_wrapper,
)

if TYPE_CHECKING:
    from tests.conftest import ModelTestConfig


def _get_classification_output_configs() -> Sequence[Dict]:
    output_configs = [
        {
            "output_info": {"output_name": "test_output"},
            "output_type_info": {
                "target_cat_columns": ["Origin"],
                "target_con_columns": [],
            },
        }
    ]

    return output_configs


@pytest.mark.skipif(
    condition=should_skip_in_gha_macos(), reason="In GHA and platform is Darwin."
)
@pytest.mark.parametrize(
    "create_test_data",
    [
        {"task_type": "binary"},
        {"task_type": "multi"},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: MLP
        {
            "injections": {
                "global_configs": {
                    "weighted_sampling_columns": ["all"],
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"na_augment_prob": 0.8},
                        "model_config": {
                            "model_type": "linear",
                            "model_init_config": {"l1": 1e-04},
                        },
                    }
                ],
                "fusion_configs": {
                    "model_config": {
                        "fc_task_dim": 256,
                        "layers": [2],
                    }
                },
                "output_configs": _get_classification_output_configs(),
            },
        },
        # Case 2: CNN
        {
            "injections": {
                "global_configs": {
                    "weighted_sampling_columns": ["Origin"],
                    "gradient_noise": 0.01,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {
                                "rb_do": 0.25,
                                "channel_exp_base": 3,
                                "l1": 1e-04,
                            },
                        },
                    }
                ],
                "output_configs": _get_classification_output_configs(),
            },
        },
        # Case 3: Identity Fusion
        {
            "injections": {
                "global_configs": {"lr": 1e-03},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "identity"},
                    },
                ],
                "fusion_configs": {
                    "model_type": "identity",
                },
                "output_configs": _get_classification_output_configs(),
            },
        },
    ],
    indirect=True,
)
def test_classification(prep_modelling_test_configs):
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

    NOTE:
        The indirect parametrization passes the arguments over to the fixtures used
        in _prep_modelling_test_config.
    """
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    output_configs = experiment.configs.output_configs

    for output_config in output_configs:
        output_name = output_config.output_info.output_name

        check_performance_result_wrapper(
            outputs=experiment.outputs,
            run_path=test_config.run_path,
            thresholds=(0.8, 0.8),
        )

        top_row_grads_dict = {"Asia": [0] * 10, "Europe": [1] * 10, "Africa": [2] * 10}
        _check_snps_wrapper(
            test_config=test_config,
            output_name=output_name,
            target_name="Origin",
            top_row_grads_dict=top_row_grads_dict,
            at_least_n_snps=5,
            all_attribution_target_classes_must_pass=False,
        )


@pytest.mark.skipif(
    condition=should_skip_in_gha_macos(), reason="In GHA and platform is Darwin."
)
@pytest.mark.parametrize(
    "create_test_data",
    [
        {"task_type": "binary"},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: Identity Fusion, SNP subset
        {
            "injections": {
                "global_configs": {"lr": 1e-03},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"subset_snps_file": "auto"},
                        "model_config": {"model_type": "identity"},
                    },
                ],
                "fusion_configs": {
                    "model_type": "identity",
                },
                "output_configs": _get_classification_output_configs(),
            },
        },
        # Case 2: Identity Fusion, SNP subset, memory dataset
        {
            "injections": {
                "global_configs": {"lr": 1e-03, "memory_dataset": True},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"subset_snps_file": "auto"},
                        "model_config": {"model_type": "identity"},
                    },
                ],
                "fusion_configs": {
                    "model_type": "identity",
                },
                "output_configs": _get_classification_output_configs(),
            },
        },
    ],
    indirect=True,
)
def test_classification_subset(prep_modelling_test_configs):
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    output_configs = experiment.configs.output_configs

    for output_config in output_configs:
        output_name = output_config.output_info.output_name

        check_performance_result_wrapper(
            outputs=experiment.outputs,
            run_path=test_config.run_path,
            thresholds=(0.7, 0.7),
        )

        top_row_grads_dict = {"Asia": [0] * 10, "Europe": [1] * 10, "Africa": [2] * 10}
        _check_snps_wrapper(
            test_config=test_config,
            output_name=output_name,
            target_name="Origin",
            top_row_grads_dict=top_row_grads_dict,
            at_least_n_snps=2,
        )


def _check_snps_wrapper(
    test_config: "ModelTestConfig",
    output_name: str,
    target_name: str,
    top_row_grads_dict: Dict[str, List[int]],
    at_least_n_snps: Union[str, int] = "all",
    check_types_skip_cls_names: Sequence[str] = tuple(),
    all_attribution_target_classes_must_pass: bool = True,
):
    expected_top_indxs = list(range(50, 1000, 100))

    cur_output_act_paths = test_config.activations_paths[output_name]

    for target_folder_name, dict_with_path_to_input in cur_output_act_paths.items():
        if target_folder_name != target_name:
            continue

        omics_acts_generator = _get_snp_activations_generator(
            cur_output_act_paths=dict_with_path_to_input
        )

        for acts_array_path, is_masked in omics_acts_generator:
            check_types = True if is_masked else False
            _check_identified_snps(
                array_path=acts_array_path,
                expected_top_indices=expected_top_indxs,
                top_row_grads_dict=top_row_grads_dict,
                check_types=check_types,
                at_least_n=at_least_n_snps,
                check_types_skip_cls_names=check_types_skip_cls_names,
                all_classes_must_pass=all_attribution_target_classes_must_pass,
            )


def _get_snp_activations_generator(cur_output_act_paths: Dict[str, Path]):
    did_run = False

    for name, cur_path in cur_output_act_paths.items():
        if "genotype" in name:
            did_run = True

            top_acts_npy = cur_path / "top_acts.npy"
            assert top_acts_npy.exists()
            yield top_acts_npy, False

            top_acts_masked_npy = cur_path / "top_acts_masked.npy"
            assert top_acts_masked_npy.exists()
            yield top_acts_masked_npy, True

    assert did_run


def _get_regression_output_configs() -> Sequence[Dict]:
    output_configs = [
        {
            "output_info": {"output_name": "test_output"},
            "output_type_info": {
                "target_cat_columns": [],
                "target_con_columns": ["Height"],
            },
        }
    ]

    return output_configs


@pytest.mark.skipif(
    condition=should_skip_in_gha_macos(), reason="In GHA and platform is Darwin."
)
@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "regression"}], indirect=True
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: Identity Fusion
        {
            "injections": {
                "global_configs": {"lr": 1e-03},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "identity"},
                    },
                ],
                "fusion_configs": {
                    "model_type": "identity",
                },
                "output_configs": _get_regression_output_configs(),
            },
        },
        # Case 2: CNN
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {"l1": 1e-04, "channel_exp_base": 4},
                        },
                    },
                ],
                "output_configs": _get_regression_output_configs(),
            },
        },
        # Case 3: MLP with subset
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "linear",
                            "model_init_config": {"l1": 1e-04},
                        },
                    },
                ],
                "output_configs": _get_regression_output_configs(),
            },
        },
        # Case 4: CNN Cycle
        {
            "injections": {
                "global_configs": {
                    "lr_schedule": "cycle",
                    "output_folder": "test_lr-cycle",
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {"l1": 1e-04, "channel_exp_base": 4},
                        },
                    },
                ],
                "output_configs": _get_regression_output_configs(),
            },
        },
    ],
    indirect=True,
)
def test_regression(prep_modelling_test_configs):
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    output_configs = experiment.configs.output_configs

    for output_config in output_configs:
        output_name = output_config.output_info.output_name

        check_performance_result_wrapper(
            outputs=experiment.outputs,
            run_path=test_config.run_path,
            thresholds=(0.8, 0.8),
        )

        top_height_snp_index = 2
        top_row_grads_dict = {"Height": [top_height_snp_index] * 10}
        _check_snps_wrapper(
            test_config=test_config,
            output_name=output_name,
            target_name="Height",
            top_row_grads_dict=top_row_grads_dict,
            at_least_n_snps=5,
        )


def _get_multi_task_output_configs(
    label_smoothing: float = 0.0, uncertainty_mt_loss: bool = True
) -> Sequence[Dict]:
    output_configs = [
        {
            "output_info": {"output_name": "test_output_copy"},
            "output_type_info": {
                "target_cat_columns": [],
                "target_con_columns": ["Height"],
                "uncertainty_weighted_mt_loss": uncertainty_mt_loss,
            },
        },
        {
            "output_info": {"output_name": "test_output"},
            "output_type_info": {
                "target_cat_columns": ["Origin"],
                "target_con_columns": ["Height"],
                "cat_label_smoothing": label_smoothing,
                "uncertainty_weighted_mt_loss": uncertainty_mt_loss,
            },
        },
    ]

    return output_configs


@pytest.mark.skipif(
    condition=should_skip_in_gha_macos(), reason="In GHA and platform is Darwin."
)
@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "multi_task"}], indirect=True
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: Check that we add and use extra inputs.
        {
            "injections": {
                "global_configs": {
                    "output_folder": "extra_inputs",
                    "cat_averaging_metrics": ["mcc"],
                    "con_averaging_metrics": ["r2"],
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {
                                "l1": 2e-05,
                                "stochastic_depth_p": 0.2,
                            },
                        },
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "input_cat_columns": ["OriginExtraCol"],
                            "input_con_columns": ["ExtraTarget"],
                        },
                        "model_config": {"model_type": "tabular"},
                    },
                ],
                "output_configs": _get_multi_task_output_configs(),
            },
        },
        # Case 2: Normal multitask with CNN
        {
            "injections": {
                "global_configs": {
                    "mixing_alpha": 0.2,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {
                                "channel_exp_base": 5,
                                "rb_do": 0.10,
                                "fc_repr_dim": 256,
                                "l1": 2e-05,
                                "stochastic_depth_p": 0.1,
                            },
                        },
                    },
                ],
                "fusion_configs": {
                    "model_config": {
                        "fc_task_dim": 256,
                        "rb_do": 0.10,
                        "fc_do": 0.10,
                        "stochastic_depth_p": 0.1,
                    },
                },
                "output_configs": _get_multi_task_output_configs(),
            },
        },
        # Case 3:  Normal multitask with MLP, note we reduce the LR for
        # stability and add L1 for regularization
        {
            "injections": {
                "global_configs": {"lr": 1e-03},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "linear",
                            "model_init_config": {"l1": 2e-05},
                        },
                    },
                ],
                "fusion_configs": {
                    "model_config": {"fc_task_dim": 64, "rb_do": 0.10, "fc_do": 0.10},
                },
                "output_configs": _get_multi_task_output_configs(),
            },
        },
        # Case 4: Using the Simple LCL model
        {
            "injections": {
                "global_configs": {"lr": 1e-03},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "mlp-split",
                            "model_init_config": {
                                "fc_repr_dim": 8,
                                "split_mlp_num_splits": 64,
                                "l1": 2e-05,
                            },
                        },
                    },
                ],
                "output_configs": _get_multi_task_output_configs(),
            },
        },
        # Case 5: Using the GLN
        {
            "injections": {
                "global_configs": {
                    "lr": 1e-03,
                    "gradient_noise": 0.001,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "genome-local-net",
                            "model_init_config": {
                                "kernel_width": 8,
                                "channel_exp_base": 2,
                                "l1": 2e-05,
                                "rb_do": 0.20,
                                "attention_inclusion_cutoff": 512,
                            },
                        },
                    },
                ],
                "fusion_configs": {
                    "model_config": {
                        "fc_task_dim": 64,
                        "fc_do": 0.20,
                        "rb_do": 0.20,
                    },
                },
                "output_configs": _get_multi_task_output_configs(
                    label_smoothing=0.1, uncertainty_mt_loss=False
                ),
            },
        },
        # Case 6: Using the MGMoE fusion
        {
            "injections": {
                "global_configs": {
                    "output_folder": "mgmoe",
                    "lr": 1e-03,
                    "save_evaluation_sample_results": False,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "genome-local-net",
                            "model_init_config": {
                                "kernel_width": 8,
                                "channel_exp_base": 2,
                                "l1": 2e-05,
                            },
                        },
                    },
                ],
                "fusion_configs": {
                    "model_type": "mgmoe",
                    "model_config": {"mg_num_experts": 3, "stochastic_depth_p": 0.2},
                },
                "output_configs": _get_multi_task_output_configs(
                    uncertainty_mt_loss=False
                ),
            },
        },
        # Case 7: Using the GLN with mixing
        {
            "injections": {
                "global_configs": {
                    "output_folder": "mixing_multi",
                    "mixing_alpha": 0.2,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {
                            "mixing_subtype": "mixup",
                        },
                        "model_config": {
                            "model_type": "genome-local-net",
                            "model_init_config": {
                                "kernel_width": 8,
                                "channel_exp_base": 2,
                                "l1": 2e-05,
                            },
                        },
                    },
                ],
                "fusion_configs": {
                    "model_config": {
                        "fc_task_dim": 256,
                        "fc_do": 0.10,
                        "rb_do": 0.10,
                    },
                },
                "output_configs": _get_multi_task_output_configs(label_smoothing=0.1),
            },
        },
        # Case 8: Using the GLN with limited activations and gradient accumulation
        {
            "injections": {
                "global_configs": {
                    "output_folder": "limited_activations",
                    "lr": 1e-03 * 4,
                    "batch_size": 16,
                    "gradient_accumulation_steps": 4,
                    "max_attributions_per_class": 100,
                    "mixing_alpha": 0.2,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "genome-local-net",
                            "model_init_config": {
                                "kernel_width": 8,
                                "channel_exp_base": 2,
                                "l1": 2e-05,
                            },
                        },
                    },
                ],
                "fusion_configs": {
                    "model_config": {
                        "fc_task_dim": 256,
                        "fc_do": 0.10,
                        "rb_do": 0.10,
                    },
                },
                "output_configs": _get_multi_task_output_configs(),
            },
        },
    ],
    indirect=True,
)
def test_multi_task(
    prep_modelling_test_configs: Tuple[train.Experiment, "ModelTestConfig"],
):
    """
    Sometimes it seems we have the case that the model only gets activated by features
    in N-1 classes, and is not activated by features in the Nth class. I.e., possibly
    the default prediction is for the Nth class, and it only picks up features related
    to the other classes.
    """
    experiment, test_config = prep_modelling_test_configs
    gc = experiment.configs.global_config

    train.train(experiment=experiment)

    output_configs = experiment.configs.output_configs
    extra_columns = get_all_tabular_input_columns(configs=experiment.configs)

    for output_config in output_configs:
        output_name = output_config.output_info.output_name
        cat_targets = output_config.output_type_info.target_cat_columns
        con_targets = output_config.output_type_info.target_con_columns

        for target_name in cat_targets:
            target_copy = "OriginExtraCol"
            threshold, at_least_n = _get_multi_task_test_args(
                extra_columns=extra_columns,
                target_copy=target_copy,
                mixing=gc.mixing_alpha,
            )
            check_performance_result_wrapper(
                outputs=experiment.outputs,
                run_path=test_config.run_path,
                thresholds=threshold,
            )
            top_row_grads_dict = {
                "Asia": [0] * 10,
                "Europe": [1] * 10,
                "Africa": [2] * 10,
            }
            _check_snps_wrapper(
                test_config=test_config,
                output_name=output_name,
                target_name=target_name,
                top_row_grads_dict=top_row_grads_dict,
                at_least_n_snps=at_least_n,
                all_attribution_target_classes_must_pass=False,
            )

        for target_name in con_targets:
            target_copy = "ExtraTarget"
            threshold, at_least_n = _get_multi_task_test_args(
                extra_columns=extra_columns,
                target_copy=target_copy,
                mixing=gc.mixing_alpha,
            )
            check_performance_result_wrapper(
                outputs=experiment.outputs,
                run_path=test_config.run_path,
                thresholds=threshold,
            )
            top_height_snp_index = 2
            top_row_grads_dict = {"Height": [top_height_snp_index] * 10}
            _check_snps_wrapper(
                test_config=test_config,
                output_name=output_name,
                target_name=target_name,
                top_row_grads_dict=top_row_grads_dict,
                at_least_n_snps=at_least_n,
                all_attribution_target_classes_must_pass=True,
            )


def get_all_tabular_input_columns(configs: Configs) -> List[str]:
    extra_columns = []
    for input_config in configs.input_configs:
        if input_config.input_info.input_type == "tabular":
            extra_columns += input_config.input_type_info.input_con_columns
            extra_columns += input_config.input_type_info.input_cat_columns

    return extra_columns


def _get_multi_task_test_args(
    extra_columns: List[str], target_copy: str, mixing: float
) -> Tuple[Tuple[float, float], int]:
    """
    We use 0 for at_least_n in the case we have correlated input columns because
    in that case the model might not actually be using any of the SNPs (better to use
    the correlated column), hence we do not expect SNPs to be highly activated.

    When mixing, the train performance (loss is OK) for e.g. accuracy is expected to be
    relatively low, as we do not have discrete inputs but a mix.
    """

    an_extra_col_is_correlated_with_target = target_copy in extra_columns
    if an_extra_col_is_correlated_with_target:
        thresholds, at_least_n = (0.9, 0.9), 0
    else:
        thresholds, at_least_n = (0.8, 0.6), 5

    if mixing:
        thresholds, at_least_n = (0.0, 0.6), 5

    return thresholds, at_least_n


def _check_identified_snps(
    array_path: Path,
    expected_top_indices: List[int],
    top_row_grads_dict: Dict[str, List[int]],
    check_types: bool,
    check_types_skip_cls_names: Sequence[str] = tuple(),
    at_least_n: Union[str, int] = "all",
    all_classes_must_pass: bool = True,
) -> None:
    """
    NOTE: We have the `at_least_n` to check for a partial match of found SNPs. Why?
    Because when doing these tests, we are making the inputs per class the same
    for multiple spots, in the case of regression this leads the network to only "need"
    to identify a part of the SNPs to create an output (possibly because we are not
    using a "correctness criteria" for regression, like we do with classification (i.e.
    only gather activations for correctly predicted classes).

    :param array_path: Path to the accumulated grads / activation array.
    :param expected_top_indices: Expected SNPs to be identified.
    :param top_row_grads_dict: What row is expected to be activated per class.
    :param check_types:  Whether to check the SNP types as well as the SNPs themselves
    (i.e. homozygous reference, etc)
    :param at_least_n: At least how many SNPs must be identified to pass the test.
    :return:
    """

    top_grads_array = np.load(str(array_path), allow_pickle=True)

    # get dict from array
    top_grads_dict: dict = top_grads_array[()]

    classes_acts_success = []
    for cls in top_grads_dict.keys():
        actual_top = np.array(sorted(top_grads_dict[cls]["top_n_idxs"]))
        expected_top = np.array(expected_top_indices)

        if at_least_n == "all":
            snp_success = (actual_top == expected_top).all()
        else:
            matches = len(set(actual_top).intersection(set(expected_top)))
            snp_success = matches >= at_least_n

        if check_types and cls not in check_types_skip_cls_names:
            expected_top_rows = top_row_grads_dict[cls]
            snp_type_success = _check_snp_types(
                cls_name=cls,
                top_grads_msk=top_grads_dict,
                expected_idxs=expected_top_rows,
                at_least_n=at_least_n,
            )

            snp_success = snp_success and snp_type_success

        classes_acts_success.append(snp_success)

    if all_classes_must_pass:
        assert all(classes_acts_success)
    else:
        must_match_n = len(classes_acts_success) - 1
        must_match_n = max(must_match_n, 1)
        assert sum(classes_acts_success) >= must_match_n


def _check_snp_types(
    cls_name: str, top_grads_msk, expected_idxs, at_least_n: int
) -> bool:
    """
    Adds a check for SNP types (i.e. reference homozygous, heterozygous, alternative
    homozygous, missing).

    Used when we have masked out the SNPs (otherwise the 0s in the one hot might have
    a high activation, since they're saying the same thing as a 1 being in a spot).
    """
    top_idxs = np.array(top_grads_msk[cls_name]["top_n_grads"].argmax(0))
    expected_idxs = np.array(expected_idxs)

    if at_least_n == "all":
        snp_type_success = (top_idxs == expected_idxs).all()
    else:
        snp_type_success = (top_idxs == expected_idxs).sum() >= at_least_n

    return snp_type_success
