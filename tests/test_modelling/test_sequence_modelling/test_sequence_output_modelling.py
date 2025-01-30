from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from eir import train
from eir.train_utils.utils import seed_everything
from tests.conftest import should_skip_in_gha_macos
from tests.setup_tests.setup_modelling_test_data.setup_sequence_test_data import (
    get_continent_keyword_map,
    get_text_sequence_base,
)
from tests.test_modelling.test_modelling_utils import check_performance_result_wrapper

if TYPE_CHECKING:
    from tests.setup_tests.fixtures_create_experiment import (
        ModelTestConfig,
        al_modelling_test_configs,
    )

seed_everything(seed=0)


@pytest.mark.skipif(condition=should_skip_in_gha_macos(), reason="In GHA.")
@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "multi",
            "modalities": ("sequence",),
            "extras": {"sequence_csv_source": True},
            "split_to_test": True,
        },
        {
            "task_type": "multi",
            "modalities": ("sequence",),
            "split_to_test": True,
            "source": "local",
        },
        {
            "task_type": "multi",
            "modalities": ("sequence",),
            "split_to_test": True,
            "source": "deeplake",
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {
                    "basic_experiment": {
                        "output_folder": "test_generation",
                        "n_epochs": 15,
                        "memory_dataset": True,
                    }
                },
                "input_configs": [],
                "fusion_configs": {
                    "model_type": "pass-through",
                },
                "output_configs": [
                    {
                        "output_info": {
                            "output_name": "test_output_sequence",
                        },
                        "sampling_config": {
                            "manual_inputs": [
                                {
                                    "test_output_sequence": "arctic_",
                                },
                                {
                                    "test_output_sequence": "lyn",
                                },
                            ],
                        },
                    }
                ],
            },
        },
    ],
    indirect=True,
)
def test_sequence_output_modelling(
    prep_modelling_test_configs: "al_modelling_test_configs",
) -> None:
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    check_performance_result_wrapper(
        outputs=experiment.outputs,
        run_path=test_config.run_path,
        max_thresholds=(0.5, 0.5),
        min_thresholds=(1.5, 1.5),
    )

    _sequence_output_test_check_wrapper(experiment=experiment, test_config=test_config)


@pytest.mark.skipif(condition=should_skip_in_gha_macos(), reason="In GHA.")
@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "multi",
            "modalities": ("sequence",),
            "extras": {"sequence_csv_source": True},
            "split_to_test": True,
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {
                    "basic_experiment": {
                        "output_folder": "test_generation",
                        "n_epochs": 15,
                        "memory_dataset": True,
                    }
                },
                "input_configs": [],
                "fusion_configs": {
                    "model_type": "pass-through",
                },
                "output_configs": [
                    {
                        "output_info": {
                            "output_name": "test_output_sequence",
                        },
                        "output_type_info": {
                            "split_on": None,
                            "tokenizer": "bpe",
                            "adaptive_tokenizer_max_vocab_size": 128,
                        },
                        "sampling_config": {
                            "manual_inputs": [
                                {
                                    "test_output_sequence": "arctic_",
                                },
                                {
                                    "test_output_sequence": "lyn",
                                },
                            ],
                        },
                    }
                ],
            },
        },
    ],
    indirect=True,
)
def test_sequence_output_modelling_bpe(
    prep_modelling_test_configs: "al_modelling_test_configs",
) -> None:
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    check_performance_result_wrapper(
        outputs=experiment.outputs,
        run_path=test_config.run_path,
        max_thresholds=(0.5, 0.5),
        min_thresholds=(2.5, 2.5),
    )

    _sequence_output_test_check_wrapper(experiment=experiment, test_config=test_config)


def _sequence_output_test_check_wrapper(
    experiment: train.Experiment,
    test_config: "ModelTestConfig",
) -> None:
    output_configs = experiment.configs.output_configs
    set_all = get_expected_keywords_set()

    for output_config in output_configs:
        output_name = output_config.output_info.output_name
        output_type = output_config.output_info.output_type

        if output_type != "sequence":
            continue

        latest_sample = test_config.last_sample_folders[output_name][output_name]

        did_check = False
        for f in Path(latest_sample).rglob("*_generated.txt"):
            with open(f) as infile:
                content = infile.read().split()

                intersection = set(content).intersection(set_all)
                if not intersection:
                    raise AssertionError(f"No expected words found in file {f}")
                if len(intersection) < 2:
                    raise AssertionError(
                        f"Expected words found in file {f} do not match "
                        f"minimum length: {intersection}"
                    )

                did_check = True

        assert did_check, f"No .txt files found in {latest_sample}."


def get_expected_keywords_set() -> set[str]:
    expected_base = get_text_sequence_base()
    set_base = set(expected_base)
    expected_dynamic = get_continent_keyword_map()
    set_dynamic = set(chain.from_iterable(expected_dynamic.values()))
    set_all = set_base.union(set_dynamic)

    return set_all
