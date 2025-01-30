from collections.abc import Sequence

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis.strategies import floats, integers, lists

from eir.data_load.data_utils import Batch
from eir.interpretation import interpret_tabular as it
from eir.interpretation.interpretation import SampleAttribution


def generate_test_attribution_sequence(
    n: int, tensor_elements: list[float]
) -> Sequence[SampleAttribution]:
    return tuple(
        generate_test_sample_attribution(id_=i, tensor_elements=tensor_elements)
        for i in range(n)
    )


def generate_test_sample_attribution(
    id_: int, tensor_elements: list[float]
) -> SampleAttribution:
    """
    We enforce two dimensions here, because that's the output dimension of attributions.
    """
    mock_batch = Batch(
        inputs={"test_input": torch.tensor(0)},
        target_labels={"test_target_label": torch.LongTensor(0)},
        ids=[str(id_)],
    )

    mock_attribution = np.array(tensor_elements)
    if len(mock_attribution.shape) == 1:
        mock_attribution = np.expand_dims(mock_attribution, axis=0)
    assert len(mock_attribution.shape) == 2

    mock_sample_attributions = {"test_attributions": mock_attribution}
    mock_raw_tabular_inputs = {"tabular_test_input": torch.tensor(tensor_elements)}

    mocked_sample_act = SampleAttribution(
        sample_info=mock_batch,
        sample_attributions=mock_sample_attributions,
        raw_inputs=mock_raw_tabular_inputs,
    )

    return mocked_sample_act


@given(
    num_samples=integers(min_value=1, max_value=100),
    attribution_inputs=lists(
        elements=floats(min_value=0, max_value=9), min_size=1, max_size=100
    ),
)
@settings(deadline=500)
def test_gather_continuous_attributions(num_samples, attribution_inputs):
    test_sequence = generate_test_attribution_sequence(
        n=num_samples, tensor_elements=attribution_inputs
    )
    cat_to_con_cutoff = min(0, len(attribution_inputs) // 2)

    gathered_values = it._gather_continuous_attributions(
        all_attributions=test_sequence,
        cat_to_con_cutoff=cat_to_con_cutoff,
        input_name="test_attributions",
    )

    assert len(gathered_values.shape) == 2
    assert gathered_values.shape[0] == num_samples
    assert gathered_values.shape[1] == len(attribution_inputs) - cat_to_con_cutoff


@given(
    num_samples=integers(min_value=1, max_value=100),
    attribution_inputs=lists(
        elements=floats(min_value=0, max_value=9), min_size=1, max_size=100
    ),
)
@settings(deadline=500)
def test_gather_categorical_attributions(num_samples, attribution_inputs):
    """
    Note: We check for shape[1] == 1 here as we always sum up the categorical slices.
    """
    test_sequence = generate_test_attribution_sequence(
        n=num_samples, tensor_elements=attribution_inputs
    )
    test_slice_start = min(0, len(attribution_inputs) // 2)
    test_slice_end = len(attribution_inputs)
    test_slice = slice(test_slice_start, test_slice_end)

    gathered_values = it._gather_categorical_attributions(
        all_attributions=test_sequence,
        cur_slice=test_slice,
        input_name="test_attributions",
    )

    assert len(gathered_values.shape) == 2
    assert gathered_values.shape[0] == num_samples
    assert gathered_values.shape[1] == 1
