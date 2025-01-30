from copy import deepcopy
from unittest.mock import MagicMock

import numpy as np

from eir.data_load.data_preparation_modules import prepare_bytes
from eir.setup.schemas import ByteInputDataConfig


def test_prepare_bytes_data():
    test_input = np.array(list(range(100)))
    test_input_copy = deepcopy(test_input)

    byte_input_data_config = ByteInputDataConfig(
        max_length=64,
        byte_encoding="uint8",
        sampling_strategy_if_longer="from_start",
        mixing_subtype="mixup",
    )

    input_config_mock = MagicMock()
    input_config_mock.input_type_info = byte_input_data_config

    vocab_mock = MagicMock()
    vocab_mock.get.return_value = 0

    bytes_input_object_mock = MagicMock()
    bytes_input_object_mock.input_config = input_config_mock
    bytes_input_object_mock.vocab = vocab_mock
    bytes_input_object_mock.computed_max_length = 64

    prepared_tensor = prepare_bytes.prepare_bytes_data(
        bytes_input_object=bytes_input_object_mock,
        bytes_data=test_input,
        test_mode=False,
    )

    assert (test_input == test_input_copy).all()
    assert prepared_tensor != test_input

    assert len(prepared_tensor) == 64
    assert len(test_input) == 100
