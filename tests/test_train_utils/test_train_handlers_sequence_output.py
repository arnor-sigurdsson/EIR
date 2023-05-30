from copy import deepcopy
from pathlib import Path
from typing import Tuple, Any, TYPE_CHECKING
from unittest.mock import create_autospec

import numpy as np
import pytest
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torchtext.vocab import Vocab
from transformers import PreTrainedTokenizerBase

from eir.train_utils.train_handlers_sequence_output import (
    ImageNormalizationStats,
    convert_image_input_to_raw,
    convert_tabular_input_to_raw,
    _extract_base_generated_tokens,
    _compute_target_index,
    _mask_targets_for_auto_eval_generation,
    _prepare_current_autoregressive_input,
    sample_next_token_index_from_output,
    top_k_top_p_filtering,
    SequenceOutputSamplingConfig,
    decode_tokens,
    prepare_sequence_output_manual_sample_data,
    SequenceOutputEvalSample,
    autoregressive_sequence_generation,
)
from tests.setup_tests.setup_modelling_test_data.setup_array_test_data import (
    _set_up_base_test_array,
)
from tests.setup_tests.setup_modelling_test_data.setup_omics_test_data import (
    _set_up_base_test_omics_array,
)

if TYPE_CHECKING:
    from tests.setup_tests.fixtures_create_experiment import ModelTestConfig
    from eir.train import Experiment


def test_convert_image_input_to_raw():
    normalization_stats = ImageNormalizationStats(
        channel_means=[0.5, 0.5, 0.5], channel_stds=[0.5, 0.5, 0.5]
    )
    valid_input = torch.randn((1, 3, 64, 64))
    valid_output = convert_image_input_to_raw(
        data=valid_input, normalization_stats=normalization_stats
    )
    assert isinstance(valid_output, Image.Image)

    invalid_input = torch.randn((3, 64, 64))
    with pytest.raises(AssertionError):
        convert_image_input_to_raw(
            data=invalid_input, normalization_stats=normalization_stats
        )


def test_convert_tabular_input_to_raw():
    input_transformers = {}
    scaler = StandardScaler()
    scaler.fit(np.random.randn(10, 1))
    input_transformers["column1"] = scaler

    valid_input = {"column1": torch.tensor([0.5])}
    valid_output = convert_tabular_input_to_raw(
        data=valid_input, input_transformers=input_transformers
    )
    assert isinstance(valid_output, dict)

    invalid_input = {"invalid_column": torch.tensor([0.5])}
    with pytest.raises(KeyError):
        convert_tabular_input_to_raw(
            data=invalid_input, input_transformers=input_transformers
        )

    edge_input = {"column1": torch.tensor([])}
    with pytest.raises(AssertionError):
        convert_tabular_input_to_raw(
            data=edge_input, input_transformers=input_transformers
        )


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
        "b": torch.tensor([[]], dtype=torch.long),
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
    prepared_inputs = {"sequence_output": torch.tensor([[1, 2, 3]])}
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
    expected_output[seq_output_name] = torch.tensor([[3, 4, 5, 6, 0]])

    result = _prepare_current_autoregressive_input(
        prepared_sample_inputs=prepared_sample_inputs,
        generated_tokens=generated_tokens,
        seq_output_name=seq_output_name,
        max_length=max_length,
        pad_idx=pad_idx,
    )
    assert torch.equal(result[seq_output_name], expected_output[seq_output_name])


def test_sample_next_token_index_from_output():
    outputs = {"seq_output": {"seq_output": torch.rand(5, 5)}}
    seq_output_name = "seq_output"
    sampling_config = SequenceOutputSamplingConfig(
        manual_inputs=[{"token": "hello"}],
        n_eval_inputs=10,
        generated_sequence_length=64,
        top_k=5,
        top_p=0.9,
    )
    current_target_index = 3

    result = sample_next_token_index_from_output(
        outputs=outputs,
        seq_output_name=seq_output_name,
        sampling_config=sampling_config,
        current_target_index=current_target_index,
    )

    assert isinstance(result, int)


def test_top_k_top_p_filtering():
    """
    top_k -> tensor([  -inf,   -inf,   -inf, 0.4000, 0.5000, 0.6000])
    top_p -> tensor([  -inf,   -inf,   -inf,   -inf, 0.5000, 0.6000])
    """
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float)
    top_k = 3
    top_p = 0.5
    filter_value = -float("Inf")

    expected_output = torch.tensor(
        [-float("Inf"), -float("Inf"), -float("Inf"), -float("Inf"), 0.5, 0.6]
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
                    "output_folder": "test_generation",
                    "n_epochs": 15,
                    "memory_dataset": True,
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
    prep_modelling_test_configs: Tuple["Experiment", "ModelTestConfig"]
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
    test_input = torch.tensor([base], dtype=torch.long)
    eval_sample = SequenceOutputEvalSample(
        inputs_to_model={"test_output_sequence": test_input},
        target_labels={"test_output_sequence": "World"},
        sample_id="0",
    )

    seq_output_name = "test_output_sequence"

    result = autoregressive_sequence_generation(
        input_objects=experiment.inputs,
        eval_sample=eval_sample,
        seq_output_name=seq_output_name,
        experiment=experiment,
        default_eir_hooks=experiment.hooks,
        sampling_config=sampling_config,
    )

    assert isinstance(result, list)
    assert len(result) <= sampling_config.generated_sequence_length
    assert result[:3] == base


@pytest.fixture
def tokens():
    return [0, 1, 2, 3]


@pytest.fixture
def decoded_tokens():
    return ["Hello,", "world"]


def test_decode_tokens_with_pretrained_tokenizer(tokens, decoded_tokens):
    mock_tokenizer = create_autospec(PreTrainedTokenizerBase)
    mock_tokenizer.decode.return_value = " ".join(decoded_tokens)
    vocab = None
    split_on = None

    result = decode_tokens(
        tokens=tokens, tokenizer=mock_tokenizer, vocab=vocab, split_on=split_on
    )

    mock_tokenizer.decode.assert_called_once_with(token_ids=tokens)
    assert result == " ".join(decoded_tokens)


def test_decode_tokens_with_vocab_and_split(tokens, decoded_tokens):
    tokenizer = None
    mock_vocab = create_autospec(Vocab)
    mock_vocab.lookup_tokens.return_value = decoded_tokens
    split_on = " "

    result = decode_tokens(
        tokens=tokens, tokenizer=tokenizer, vocab=mock_vocab, split_on=split_on
    )

    mock_vocab.lookup_tokens.assert_called_once_with(indices=tokens)
    assert result == " ".join(decoded_tokens)


def test_decode_tokens_with_vocab_and_no_split(tokens, decoded_tokens):
    tokenizer = None
    mock_vocab = create_autospec(Vocab)
    mock_vocab.lookup_tokens.return_value = decoded_tokens
    split_on = None

    result = decode_tokens(
        tokens=tokens, tokenizer=tokenizer, vocab=mock_vocab, split_on=split_on
    )

    mock_vocab.lookup_tokens.assert_called_once_with(indices=tokens)
    assert result == "".join(decoded_tokens)


def _generate_manual_sample_test_data(tmp_path) -> dict[str, Any]:
    sample_inputs = {}

    # 1. Omics
    omics_array, *_ = _set_up_base_test_omics_array(n_snps=1000)
    omics_array = omics_array.astype(np.uint8)
    omics_file_path = tmp_path / "omics.npy"
    np.save(str(omics_file_path), omics_array)
    sample_inputs["test_genotype"] = str(omics_file_path)

    # 2. Sequence
    sequence_data = "hello world"
    sample_inputs["test_sequence"] = sequence_data

    # 3. Bytes
    byte_data = b"some byte data"
    byte_data_file_path = tmp_path / "byte_data.bin"
    with byte_data_file_path.open("wb") as f:
        f.write(byte_data)
    sample_inputs["test_bytes"] = str(byte_data_file_path)

    # 4. Image
    image_base = np.zeros((16, 16), dtype=np.uint8)
    img = Image.fromarray(image_base)
    image_file_path = tmp_path / "image.png"
    img.save(image_file_path)
    sample_inputs["test_image"] = str(image_file_path)

    # 5. Tabular
    tabular_data = {
        "OriginExtraCol": ["Europe"],
        "ExtraTarget": [0.1337],
    }
    sample_inputs["test_tabular"] = tabular_data

    # 6. Array
    array_data, _ = _set_up_base_test_array(dims=1, class_integer=0)
    array_file_path = tmp_path / "array.npy"
    np.save(array_file_path, array_data)
    sample_inputs["test_array"] = str(array_file_path)

    return sample_inputs


@pytest.mark.parametrize(
    argnames="create_test_data",
    argvalues=[
        {
            "task_type": "multi",
            "split_to_test": True,
            "modalities": (
                "omics",
                "sequence",
                "image",
                "array",
            ),
            "source": "local",
            "extras": {"array_dims": 1},
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {
                    "output_folder": "test_manual_samples_preparation",
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "genome-local-net"},
                    },
                    {
                        "input_info": {"input_name": "test_sequence"},
                    },
                    {
                        "input_info": {"input_name": "test_bytes"},
                    },
                    {
                        "input_info": {"input_name": "test_image"},
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "input_cat_columns": ["OriginExtraCol"],
                            "input_con_columns": ["ExtraTarget"],
                        },
                        "model_config": {"model_type": "tabular"},
                    },
                    {
                        "input_info": {"input_name": "test_array"},
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {
                                "rb_do": 0.25,
                                "channel_exp_base": 3,
                                "l1": 1e-04,
                                "kernel_height": 1,
                            },
                        },
                    },
                ],
                "fusion_configs": {
                    "model_config": {
                        "fc_task_dim": 256,
                        "fc_do": 0.10,
                        "rb_do": 0.10,
                    },
                },
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        },
    ],
    indirect=True,
)
def test_prepare_sequence_output_manual_sample_data(
    prep_modelling_test_configs: Tuple["Experiment", "ModelTestConfig"],
    tmp_path: Path,
):
    experiment, test_config = prep_modelling_test_configs

    input_objects = experiment.inputs

    test_data = _generate_manual_sample_test_data(tmp_path=tmp_path)
    prepared_test_data = prepare_sequence_output_manual_sample_data(
        sample_inputs=test_data, input_objects=input_objects
    )

    expected_keys = [
        "test_genotype",
        "test_tabular",
        "test_sequence",
        "test_bytes",
        "test_image",
        "test_array",
    ]
    assert set(prepared_test_data.keys()) == set(expected_keys)

    assert prepared_test_data["test_genotype"].shape == (1, 4, 1000)

    assert prepared_test_data["test_image"].shape == (3, 16, 16)

    assert prepared_test_data["test_sequence"].shape == (63,)

    assert prepared_test_data["test_array"].shape == (1, 1, 100)

    assert set(prepared_test_data["test_tabular"].keys()) == {
        "OriginExtraCol",
        "ExtraTarget",
    }
    assert prepared_test_data["test_tabular"]["OriginExtraCol"].dtype == int
    assert prepared_test_data["test_tabular"]["ExtraTarget"].dtype == float

    assert prepared_test_data["test_bytes"].shape == (128,)
