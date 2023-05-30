from copy import deepcopy
from unittest.mock import MagicMock

import numpy as np

from eir.data_load.data_preparation_modules import prepare_bytes


def test_prepare_bytes_data():
    test_input = np.array([i for i in range(100)])
    test_input_copy = deepcopy(test_input)

    input_config_mock = MagicMock()
    input_config_mock.input_config.input_type_info.max_length = 64
    input_config_mock.vocab.get.return_value = 0
    prepared_tensor = prepare_bytes.prepare_bytes_data(
        bytes_input_object=input_config_mock,
        bytes_data=test_input,
        test_mode=False,
    )

    assert (test_input == test_input_copy).all()
    assert prepared_tensor != test_input

    assert len(prepared_tensor) == 64
    assert len(test_input) == 100
