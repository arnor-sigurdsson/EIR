from typing import Sequence, List

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis.strategies import integers, lists, floats

from eir.data_load.data_utils import Batch
from eir.interpretation import interpret_tabular as it
from eir.interpretation.interpretation import SampleActivation


def generate_test_activation_sequence(
    n: int, tensor_elements: List[float]
) -> Sequence[SampleActivation]:
    return tuple(
        generate_test_sample_activation(id_=i, tensor_elements=tensor_elements)
        for i in range(n)
    )


def generate_test_sample_activation(
    id_: int, tensor_elements: List[float]
) -> SampleActivation:
    """
    We enforce two dimensions here, because that's the output dimension of attributions.
    """
    mock_batch = Batch(
        inputs={"test_input": torch.tensor(0)},
        target_labels={"test_target_label": torch.LongTensor(0)},
        ids=[str(id_)],
    )

    mock_activation = np.array(tensor_elements)
    if len(mock_activation.shape) == 1:
        mock_activation = np.expand_dims(mock_activation, axis=0)
    assert len(mock_activation.shape) == 2

    mock_sample_activations = {"test_activations": mock_activation}
    mock_raw_tabular_inputs = {"tabular_test_input": torch.tensor(tensor_elements)}

    mocked_sample_act = SampleActivation(
        sample_info=mock_batch,
        sample_activations=mock_sample_activations,
        raw_inputs=mock_raw_tabular_inputs,
    )

    return mocked_sample_act


@given(
    num_samples=integers(min_value=1, max_value=100),
    activation_inputs=lists(
        elements=floats(min_value=0, max_value=9), min_size=1, max_size=100
    ),
)
@settings(deadline=500)
def test_gather_continuous_attributions(num_samples, activation_inputs):
    test_sequence = generate_test_activation_sequence(
        n=num_samples, tensor_elements=activation_inputs
    )
    cat_to_con_cutoff = min(0, len(activation_inputs) // 2)

    gathered_values = it._gather_continuous_attributions(
        all_activations=test_sequence,
        cat_to_con_cutoff=cat_to_con_cutoff,
        input_name="test_activations",
    )

    assert len(gathered_values.shape) == 2
    assert gathered_values.shape[0] == num_samples
    assert gathered_values.shape[1] == len(activation_inputs) - cat_to_con_cutoff


@given(
    num_samples=integers(min_value=1, max_value=100),
    activation_inputs=lists(
        elements=floats(min_value=0, max_value=9), min_size=1, max_size=100
    ),
)
@settings(deadline=500)
def test_gather_categorical_attributions(num_samples, activation_inputs):
    """
    Note: We check for shape[1] == 1 here as we always sum up the categorical slices.
    """
    test_sequence = generate_test_activation_sequence(
        n=num_samples, tensor_elements=activation_inputs
    )
    test_slice_start = min(0, len(activation_inputs) // 2)
    test_slice_end = len(activation_inputs)
    test_slice = slice(test_slice_start, test_slice_end)

    gathered_values = it._gather_categorical_attributions(
        all_activations=test_sequence,
        cur_slice=test_slice,
        input_name="test_activations",
    )

    assert len(gathered_values.shape) == 2
    assert gathered_values.shape[0] == num_samples
    assert gathered_values.shape[1] == 1
