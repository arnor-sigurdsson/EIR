from copy import deepcopy
from typing import TYPE_CHECKING
from unittest.mock import create_autospec

import pytest
import torch

from eir.setup.input_setup_modules.torchtext_port.vocab import Vocab
from eir.train_utils.evaluation_modules.evaluation_handlers_utils import decode_tokens
from eir.train_utils.evaluation_modules.train_handlers_sequence_output import (
    SequenceOutputEvalSample,
    SequenceOutputSamplingConfig,
    _compute_target_index,
    _extract_base_generated_tokens,
    _mask_targets_for_auto_eval_generation,
    _prepare_current_autoregressive_input,
    autoregressive_sequence_generation,
    sample_next_token_index_from_output,
    top_k_top_p_filtering,
)

if TYPE_CHECKING:
    from eir.train import Experiment
    from tests.setup_tests.fixtures_create_experiment import ModelTestConfig


def test_mask_targets_for_auto_eval_generation():
    inputs = {
        "a": 1,
        "b": 2,
        "c": 3,
    }
    output_name = "b"
    input_types = {
        "a": "tabular",
        "b": "sequence",
        "c": "tabular",
    }

    expected_output = {
        "a": 1,
        "b": torch.tensor([], dtype=torch.long),
        "c": 3,
    }

    result = _mask_targets_for_auto_eval_generation(
        inputs=inputs, output_name=output_name, input_types=input_types
    )
    assert result["a"] == expected_output["a"]
    assert torch.equal(result["b"], expected_output["b"])
    assert result["c"] == expected_output["c"]

    with pytest.raises(NotImplementedError):
        output_name = "a"
        _mask_targets_for_auto_eval_generation(
            inputs=inputs, output_name=output_name, input_types=input_types
        )


def test_extract_base_generated_tokens():
    prepared_inputs = {"sequence_output": torch.tensor([1, 2, 3])}
    seq_output_name = "sequence_output"

    base_tokens = _extract_base_generated_tokens(
        prepared_inputs=prepared_inputs, seq_output_name=seq_output_name
    )

    assert base_tokens == [1, 2, 3]


def test_compute_target_index():
    current_generated_length = 5
    max_length = 10

    target_index = _compute_target_index(
        current_generated_length=current_generated_length, max_length=max_length
    )

    assert target_index == 5

    current_generated_length = 15

    target_index = _compute_target_index(
        current_generated_length=current_generated_length, max_length=max_length
    )

    assert target_index == 9


def test_prepare_current_autoregressive_input():
    prepared_sample_inputs = {"a": 1, "b": 2, "c": 3}
    generated_tokens = [1, 2, 3, 4, 5, 6]
    seq_output_name = "b"
    max_length = 5
    pad_idx = 0

    expected_output = deepcopy(prepared_sample_inputs)
    expected_output[seq_output_name] = torch.tensor([3, 4, 5, 6, 0])

    result = _prepare_current_autoregressive_input(
        prepared_sample_inputs=prepared_sample_inputs,
        generated_tokens=generated_tokens,
        seq_output_name=seq_output_name,
        max_length=max_length,
        pad_idx=pad_idx,
    )
    assert torch.equal(result[seq_output_name], expected_output[seq_output_name])


def test_sample_next_token_index_from_output():
    outputs = {"seq_output": {"seq_output": torch.rand(1, 5, 5)}}
    seq_output_name = "seq_output"
    sampling_config = SequenceOutputSamplingConfig(
        manual_inputs=[{"token": "hello"}],
        n_eval_inputs=10,
        generated_sequence_length=64,
        top_k=5,
        top_p=0.9,
    )
    current_target_indices = [3]

    result = sample_next_token_index_from_output(
        outputs=outputs,
        seq_output_name=seq_output_name,
        sampling_config=sampling_config,
        current_target_indices=current_target_indices,
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], int)


def test_top_k_top_p_filtering():
    """
    top_k -> tensor([  -inf,   -inf,   -inf, 0.4000, 0.5000, 0.6000])
    top_p -> tensor([  -inf,   -inf,   -inf,   -inf, 0.5000, 0.6000])
    """
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    logits = torch.tensor(values, dtype=torch.float).unsqueeze(0)

    top_k = 3
    top_p = 0.5
    filter_value = -float("Inf")

    expected_output = torch.tensor(
        [[-float("Inf"), -float("Inf"), -float("Inf"), -float("Inf"), 0.5, 0.6]]
    )

    result = top_k_top_p_filtering(
        logits=logits, top_k=top_k, top_p=top_p, filter_value=filter_value
    )

    assert torch.equal(result, expected_output)


@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "multi",
            "modalities": ("sequence",),
            "extras": {"sequence_csv_source": True},
            "split_to_test": True,
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {
                    "basic_experiment": {
                        "output_folder": "test_generation",
                        "n_epochs": 15,
                        "memory_dataset": True,
                    }
                },
                "input_configs": [],
                "fusion_configs": {
                    "model_type": "pass-through",
                },
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_sequence"},
                    }
                ],
            },
        },
    ],
    indirect=True,
)
def test_autoregressive_sequence_generation(
    prep_modelling_test_configs: tuple["Experiment", "ModelTestConfig"],
) -> None:
    experiment, test_config = prep_modelling_test_configs
    experiment.model.eval()

    sampling_config = SequenceOutputSamplingConfig(
        manual_inputs=[],
        n_eval_inputs=10,
        generated_sequence_length=20,
        top_k=5,
        top_p=0.5,
    )

    base = [5, 6, 7]
    test_input = torch.tensor(base, dtype=torch.long)
    eval_samples = (
        SequenceOutputEvalSample(
            inputs_to_model={"test_output_sequence": test_input},
            target_labels={"test_output_sequence": "World"},
            sample_id="0",
        ),
    )

    seq_output_name = "test_output_sequence"

    batch_result = autoregressive_sequence_generation(
        input_objects=experiment.inputs,
        eval_samples=eval_samples,
        seq_output_name=seq_output_name,
        experiment=experiment,
        default_eir_hooks=experiment.hooks,
        sampling_config=sampling_config,
    )

    assert isinstance(batch_result, list)

    item_result = batch_result[0]
    assert isinstance(item_result, list)

    assert len(item_result) <= sampling_config.generated_sequence_length
    assert item_result[:3] == base


@pytest.fixture
def tokens():
    return [0, 1, 2, 3]


def test_decode_tokens_with_vocab_and_split(tokens, decoded_tokens):
    mock_vocab = create_autospec(Vocab)
    mock_vocab.lookup_tokens.return_value = decoded_tokens
    split_on = " "

    result = decode_tokens(
        tokens=tokens,
        vocab=mock_vocab,
        split_on=split_on,
    )

    mock_vocab.lookup_tokens.assert_called_once_with(indices=tokens)
    assert result == " ".join(decoded_tokens)


def test_decode_tokens_with_vocab_and_no_split(tokens, decoded_tokens):
    mock_vocab = create_autospec(Vocab)
    mock_vocab.lookup_tokens.return_value = decoded_tokens
    split_on = None

    result = decode_tokens(
        tokens=tokens,
        vocab=mock_vocab,
        split_on=split_on,
    )

    mock_vocab.lookup_tokens.assert_called_once_with(indices=tokens)
    assert result == "".join(decoded_tokens)


@pytest.fixture
def decoded_tokens():
    return ["Hello,", "world"]
