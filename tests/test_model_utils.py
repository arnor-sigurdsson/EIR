import pytest

from human_origins_supervised.models import model_utils


@pytest.mark.parametrize(
    "test_input,expected",
    [((1000, 4), [2]), ((10000, 4), [2, 2]), ((1e6, 4), [2, 2, 3, 2])],
)
def find_no_resblocks_needed(test_input, expected):
    assert model_utils.find_no_resblocks_needed(*test_input) == expected
