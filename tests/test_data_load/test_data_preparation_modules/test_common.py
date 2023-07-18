import torch

from eir.data_load.data_preparation_modules import common


def test_sample_sequence_uniform():
    test_tensor = torch.arange(0, 100)

    sampled_tensor = common._sample_sequence_uniform(
        tensor=test_tensor, tensor_length=len(test_tensor), max_length=50
    )
    assert len(sampled_tensor) == 50
    assert len(sampled_tensor) == len(set(sampled_tensor))


def test_process_tensor_to_length():
    test_tensor = torch.arange(0, 100)

    test_tensor_simple_trunc = common.process_tensor_to_length(
        tensor=test_tensor, max_length=50, sampling_strategy_if_longer="from_start"
    )
    assert len(test_tensor_simple_trunc) == 50
    assert (test_tensor_simple_trunc == test_tensor[:50]).all()

    test_tensor_unif_trunc = common.process_tensor_to_length(
        tensor=test_tensor, max_length=50, sampling_strategy_if_longer="uniform"
    )
    assert len(test_tensor_unif_trunc) == 50
    assert len(test_tensor_unif_trunc) == len(set(test_tensor_unif_trunc))

    test_tensor_padded = common.process_tensor_to_length(
        tensor=test_tensor, max_length=128, sampling_strategy_if_longer="uniform"
    )
    assert len(test_tensor_padded) == 128
