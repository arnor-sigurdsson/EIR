import numpy as np
import torch
import pytest

from eir.data_load.data_preparation_modules.prepare_array import prepare_array_data


@pytest.mark.parametrize(
    "array_data, expected_result",
    [
        (np.array([1, 2, 3]), torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)),
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.float32),
        ),
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
            torch.tensor(
                [
                    [
                        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                    ]
                ],
                dtype=torch.float32,
            ),
        ),
    ],
)
def test_prepare_array_data(array_data, expected_result):
    result = prepare_array_data(array_data=array_data)

    assert torch.all(torch.eq(result, expected_result))


def test_prepare_array_data_invalid_dimensions():
    array_data = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]])

    with pytest.raises(ValueError) as exc_info:
        prepare_array_data(array_data=array_data)

    expected_error_message = (
        "Array has 4 dimensions, " "currently only 1, 2, or 3 are supported."
    )
    assert str(exc_info.value) == expected_error_message
