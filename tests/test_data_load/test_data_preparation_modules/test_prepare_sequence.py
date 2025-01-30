from copy import deepcopy
from unittest.mock import MagicMock

import numpy as np

from eir.data_load.data_preparation_modules import prepare_sequence
from eir.setup.schemas import SequenceInputDataConfig


def test_prepare_sequence_data():
    test_input = np.array(list(range(100)))
    test_input_copy = deepcopy(test_input)

    encode_func_mock = MagicMock()
    encode_func_mock.return_value = [0]

    tokenizer_mock = MagicMock()
    tokenizer_mock.pad_token = "<pad>"

    sequence_input_data_config_mock = MagicMock(spec=SequenceInputDataConfig)
    sequence_input_data_config_mock.max_length = 64
    sequence_input_data_config_mock.sampling_strategy_if_longer = "uniform"

    sequence_input_object_mock = MagicMock()
    sequence_input_object_mock.computed_max_length = 64
    sequence_input_object_mock.encode_func = encode_func_mock
    sequence_input_object_mock.tokenizer = tokenizer_mock
    sequence_input_object_mock.input_config.input_type_info = (
        sequence_input_data_config_mock
    )

    prepared_tensor = prepare_sequence.prepare_sequence_data(
        sequence_input_object=sequence_input_object_mock,
        cur_file_content_tokenized=test_input,
        test_mode=False,
    )

    assert (test_input == test_input_copy).all()
    assert prepared_tensor.numpy().shape != test_input.shape
    assert len(prepared_tensor) == 64
    assert len(test_input) == 100
