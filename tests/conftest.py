from typing import Any, Dict, Tuple

import pandas as pd
import pytest

from eir.train_utils.utils import seed_everything

pytest_plugins = [
    "tests.setup_tests.fixtures_create_data",
    "tests.setup_tests.fixtures_create_labels",
    "tests.setup_tests.fixtures_create_configs",
    "tests.setup_tests.fixtures_create_datasets",
    "tests.setup_tests.fixtures_create_models",
    "tests.setup_tests.fixtures_create_experiment",
]

seed_everything()


def pytest_addoption(parser):
    parser.addoption("--keep_outputs", action="store_true")
    parser.addoption(
        "--num_samples_per_class",
        type=int,
        default=1000,
        help="Number of samples per class.",
    )
    parser.addoption(
        "--num_snps", type=int, default=1000, help="Number of SNPs per sample."
    )


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.keep_outputs
    if "keep_outputs" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("keep_outputs", [option_value])


def get_system_info() -> Tuple[bool, str]:
    import os
    import platform

    in_gh_actions = os.environ.get("GITHUB_ACTIONS", False)
    if in_gh_actions:
        in_gh_actions = True

    system = platform.system()

    return in_gh_actions, system


def should_skip_in_gha():
    in_gha, _ = get_system_info()
    if in_gha:
        return True

    return False


def should_skip_in_gha_macos():
    """
    We use this to skip some modelling tests as the GHA MacOS runner can be very slow.
    """
    in_gha, platform = get_system_info()
    if in_gha and platform == "Darwin":
        return True

    return False


@pytest.fixture(scope="session")
def parse_test_cl_args(request) -> Dict[str, Any]:
    n_per_class = request.config.getoption("--num_samples_per_class")
    num_snps = request.config.getoption("--num_snps")

    parsed_args = {"n_per_class": n_per_class, "n_snps": num_snps}

    return parsed_args


@pytest.fixture()
def get_transformer_test_data():
    test_labels_dict = {
        "1": {"Origin": "Asia", "Height": 150},
        "2": {"Origin": "Africa", "Height": 190},
        "3": {"Origin": "Europe", "Height": 170},
    }
    test_labels_df = pd.DataFrame(test_labels_dict).T

    test_target_columns_dict = {"con": ["Height"], "cat": ["Origin"]}

    return test_labels_df, test_target_columns_dict
