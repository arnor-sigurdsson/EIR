from copy import deepcopy
from unittest.mock import MagicMock

import numpy as np

from eir.data_load.data_preparation_modules import prepare_sequence


def test_prepare_sequence_data():
    test_input = np.array([i for i in range(100)])
    test_input_copy = deepcopy(test_input)

    input_config_mock = MagicMock()
    input_config_mock.computed_max_length = 64
    input_config_mock.encode_func.return_value = [0]
    prepared_tensor = prepare_sequence.prepare_sequence_data(
        sequence_input_object=input_config_mock,
        cur_file_content_tokenized=test_input,
        test_mode=False,
    )

    assert (test_input == test_input_copy).all()
    assert prepared_tensor != test_input

    assert len(prepared_tensor) == 64
    assert len(test_input) == 100
