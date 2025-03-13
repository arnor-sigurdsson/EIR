import json
import os
import random
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

import pytest
from aislib.misc_utils import ensure_path_exists
from transformers import PreTrainedTokenizer

from eir.setup import schemas
from eir.setup.input_setup_modules import setup_sequence
from eir.setup.input_setup_modules.torchtext_port.vocab import Vocab


def _get_mock_target():
    return "eir.setup.input_setup_modules.setup_sequence"


def set_up_simple_sequence_test_data(
    path: Path,
    pool: Sequence[str],
    n_samples: int = 100,
    min_length: int = 5,
    max_length: int = 20,
    sep: str = " ",
) -> Path:
    ensure_path_exists(path=path, is_folder=True)

    n_words_seq = tuple(range(min_length, max_length))
    for i in range(n_samples):
        cur_n_words = random.choice(seq=n_words_seq)
        cur_words = random.choices(population=pool, k=cur_n_words)
        outpath = path / f"test_sample_{i}.txt"

        cur_out_text = sep.join(cur_words)
        outpath.write_text(data=cur_out_text)

    return path


def set_up_simple_vocab_file(
    vocab: Sequence[str],
    outpath: Path,
) -> Path:
    ensure_path_exists(path=outpath, is_folder=False)
    vocab_dict = {word: i for i, word in enumerate(vocab)}
    with open(outpath, "w") as out_file_handle:
        json.dump(vocab_dict, out_file_handle)

    return outpath


def _get_simple_sample_pool() -> Sequence[str]:
    return (
        "dog",
        "cat",
        "mouse",
        "cow",
        "dolphin",
        "fox",
    )


def test_get_tokenizer():
    test_input = [
        "the",
        "lazy",
        "dog",
        "JUMPED",
        "over::",
        "the",
        "red",
        "fox",
        "or",
        "whatever",
    ]

    identity_tokenizer = setup_sequence.get_basic_tokenizer(
        tokenizer_name=None, tokenizer_language=None
    )
    assert identity_tokenizer(test_input) == test_input

    with pytest.raises(ValueError):
        setup_sequence.get_basic_tokenizer(
            tokenizer_name="revtok", tokenizer_language="is"
        )

    basic_english_tokenizer = setup_sequence.get_basic_tokenizer(
        tokenizer_name="basic_english", tokenizer_language="en"
    )

    test_input_tokenized_expected = [
        "the",
        "lazy",
        "dog",
        "jumped",
        "over",
        "the",
        "red",
        "fox",
        "or",
        "whatever",
    ]
    test_input_tokenized = basic_english_tokenizer(test_input)
    assert test_input_tokenized == test_input_tokenized_expected


def test_get_tokenized_vocab_iterator():
    def _test_iterator():
        yield [
            "the",
            "lazy",
            "dog",
            "JUMPED",
            "over::",
            "the",
            "red",
            "fox",
            "or",
            "whatever",
        ]

    basic_english_tokenizer = setup_sequence.get_basic_tokenizer(
        tokenizer_name="basic_english",
        tokenizer_language="en",
    )

    tokenized_vocab_iterator = setup_sequence.get_tokenized_vocab_iterator(
        vocab_iterator=_test_iterator(),
        tokenizer=basic_english_tokenizer,
        is_from_file=False,
    )
    results = list(tokenized_vocab_iterator)
    assert len(results) == 1

    assert results[0] == [
        "the",
        "lazy",
        "dog",
        "jumped",
        "over",
        "the",
        "red",
        "fox",
        "or",
        "whatever",
    ]


def test_get_vocab_iterator_basic(tmp_path: Path):
    seq_path = tmp_path / "test_sequence"
    test_pool = _get_simple_sample_pool()
    set_up_simple_sequence_test_data(
        path=seq_path, pool=test_pool, sep=" ", n_samples=100, max_length=20
    )

    gathered_stats = setup_sequence.GatheredSequenceStats()
    vocab_iter = setup_sequence.get_vocab_iterator(
        input_source=str(seq_path), split_on=" ", gathered_stats=gathered_stats
    )
    vocab = {word for sequence in vocab_iter for word in sequence}
    assert vocab == set(test_pool)
    assert gathered_stats.total_files == 100


def test_get_bpe_tokenizer(tmp_path: Path):
    seq_path = tmp_path / "test_sequence"
    test_pool = _get_simple_sample_pool()
    set_up_simple_sequence_test_data(
        path=seq_path, pool=test_pool, sep=" ", n_samples=100, max_length=20
    )

    # Core functionality
    vocab_iter_test_training = setup_sequence.get_vocab_iterator(
        input_source=str(seq_path),
        split_on=" ",
        gathered_stats=setup_sequence.GatheredSequenceStats(),
    )
    bpe_tokenizer_from_scratch = setup_sequence.get_bpe_tokenizer(
        vocab_iterator=vocab_iter_test_training,
        vocab_file=None,
        vocab_size=None,
        split_on=" ",
    )
    known_sequence = ["cat", "dog", "mouse"]
    known_sequence_tokenized = bpe_tokenizer_from_scratch(known_sequence)
    assert known_sequence_tokenized == ["cat", "dog", "mouse"]

    unknown_sequence = ["edge", "knot", "city"]
    unknown_sequence_tokenized = bpe_tokenizer_from_scratch(unknown_sequence)
    assert unknown_sequence_tokenized == [
        "e",
        "d",
        "g",
        "e",
        "<unk>",
        "n",
        "o",
        "t",
        "c",
        "i",
        "t",
        "<unk>",
    ]

    # Saving and loading
    vocab_iter_test_saving = setup_sequence.get_vocab_iterator(
        input_source=str(seq_path),
        split_on=" ",
        gathered_stats=setup_sequence.GatheredSequenceStats(),
    )
    bpe_tokenizer_object = setup_sequence._get_bpe_tokenizer_object(
        vocab_iterator=vocab_iter_test_saving,
        vocab_file=None,
        vocab_size=None,
    )
    saved_bpe_path = tmp_path / "test_bpe.json"
    bpe_tokenizer_object.save(str(saved_bpe_path))

    bpe_tokenizer_from_pretrained = setup_sequence.get_bpe_tokenizer(
        vocab_iterator=None,
        vocab_file=str(saved_bpe_path),
        vocab_size=None,
        split_on=" ",
    )
    known_sequence_tokenized_pretrained = bpe_tokenizer_from_pretrained(known_sequence)
    assert known_sequence_tokenized_pretrained == ["cat", "dog", "mouse"]
    unknown_sequence_tokenized_pretrained = bpe_tokenizer_from_pretrained(
        unknown_sequence
    )
    assert unknown_sequence_tokenized_pretrained == [
        "e",
        "d",
        "g",
        "e",
        "<unk>",
        "n",
        "o",
        "t",
        "c",
        "i",
        "t",
        "<unk>",
    ]

    # General checks
    gathered_stats_general = setup_sequence.GatheredSequenceStats()
    vocab_iter_test_general = setup_sequence.get_vocab_iterator(
        input_source=str(seq_path),
        split_on=" ",
        gathered_stats=gathered_stats_general,
    )
    vocab = {word for sequence in vocab_iter_test_general for word in sequence}
    assert vocab == set(test_pool)
    assert gathered_stats_general.total_files == 100


def test_get_vocab_iterator_basic_diff_split(tmp_path: Path):
    seq_path_diff_split = tmp_path / "test_sequence_no_split"
    test_pool = _get_simple_sample_pool()

    set_up_simple_sequence_test_data(
        path=seq_path_diff_split,
        pool=test_pool,
        sep="---",
        n_samples=100,
        min_length=20,
        max_length=21,
    )
    gathered_stats_diff_split = setup_sequence.GatheredSequenceStats()
    vocab_iter_diff_split = setup_sequence.get_vocab_iterator(
        input_source=str(seq_path_diff_split),
        split_on="---",
        gathered_stats=gathered_stats_diff_split,
    )
    vocab_diff_split = {word for sequence in vocab_iter_diff_split for word in sequence}
    assert vocab_diff_split == set(test_pool)
    assert gathered_stats_diff_split.total_files == 100
    assert gathered_stats_diff_split.max_length == 20


def test_get_vocab_iterator_vocab_file(tmp_path: Path):
    seq_path = tmp_path / "test_sequence_using_vocab"
    vocab_path = tmp_path / "vocab.json"
    test_pool = _get_simple_sample_pool()
    vocab_file = set_up_simple_vocab_file(vocab=test_pool, outpath=vocab_path)

    gathered_stats_vocab = setup_sequence.GatheredSequenceStats()
    set_up_simple_sequence_test_data(
        path=seq_path, pool=test_pool, sep=" ", n_samples=100, max_length=20
    )
    vocab_iter_diff_split = setup_sequence.get_vocab_iterator(
        input_source=str(vocab_file),
        split_on=" ",
        gathered_stats=gathered_stats_vocab,
        vocab_file=str(vocab_file),
    )
    vocab = set(vocab_iter_diff_split)
    assert vocab == set(test_pool)
    assert gathered_stats_vocab.total_count == len(vocab)


def test_get_max_length(tmp_path):
    """
    Probabilistic guarantee of max length in this test, might fail once in a while.
    """
    seq_path = tmp_path / "test_sequence_checking_max_length"
    test_pool = _get_simple_sample_pool()

    set_up_simple_sequence_test_data(
        path=seq_path,
        pool=test_pool,
        sep=" ",
        n_samples=200,
        min_length=15,
        max_length=21,
    )
    gathered_stats_max_length = setup_sequence.GatheredSequenceStats()
    vocab_iter_max_length = setup_sequence.get_vocab_iterator(
        input_source=str(seq_path),
        split_on=" ",
        gathered_stats=gathered_stats_max_length,
    )
    vocab_max_length = {word for sequence in vocab_iter_max_length for word in sequence}
    assert vocab_max_length == set(test_pool)
    assert gathered_stats_max_length.total_files == 200
    assert gathered_stats_max_length.max_length == 20

    max_length_from_func = setup_sequence.get_max_length(
        max_length_config_value="max", gathered_stats=gathered_stats_max_length
    )
    assert max_length_from_func == 20

    avg_length_from_func = setup_sequence.get_max_length(
        max_length_config_value="average", gathered_stats=gathered_stats_max_length
    )
    assert avg_length_from_func < 20


def test_possibly_gather_all_stats_from_input(tmp_path):
    seq_path = tmp_path / "test_sequence_using_vocab"
    vocab_path = tmp_path / "vocab.json"
    test_pool = _get_simple_sample_pool()
    vocab_file = set_up_simple_vocab_file(vocab=test_pool, outpath=vocab_path)

    gathered_stats_vocab = setup_sequence.GatheredSequenceStats()
    set_up_simple_sequence_test_data(
        path=seq_path,
        pool=test_pool,
        sep=" ",
        n_samples=100,
        min_length=20,
        max_length=21,
    )
    vocab_iter_diff_split = setup_sequence.get_vocab_iterator(
        input_source=str(vocab_file),
        split_on=" ",
        gathered_stats=gathered_stats_vocab,
        vocab_file=str(vocab_file),
    )
    vocab = set(vocab_iter_diff_split)
    assert vocab == set(test_pool)
    assert gathered_stats_vocab.total_count == len(vocab)

    new_gathered_stats = setup_sequence.possibly_gather_all_stats_from_input(
        prev_gathered_stats=gathered_stats_vocab,
        input_source=str(seq_path),
        vocab_file=str(vocab_file),
        split_on=" ",
        max_length="max",
    )
    assert new_gathered_stats.total_files == 100
    assert new_gathered_stats.max_length == 20
    assert new_gathered_stats.total_count == 100 * 20


@mock.patch(f"{_get_mock_target()}.AutoTokenizer")
def test_get_hf_tokenizer(auto_tokenizer_mock):
    setup_sequence._get_hf_tokenizer(hf_model_name="bert-base-uncased")
    assert os.getenv("TOKENIZERS_PARALLELISM") == "false"
    assert auto_tokenizer_mock.from_pretrained.called


def test_add_specials_to_hf_tokenizer():
    hf_tokenizer = mock.create_autospec(spec=PreTrainedTokenizer)
    hf_tokenizer.special_tokens_map = {}
    hf_tokenizer_with_specials = setup_sequence._add_specials_to_hf_tokenizer(
        hf_tokenizer=hf_tokenizer
    )
    assert hf_tokenizer_with_specials.add_special_tokens.called
    assert not hf_tokenizer_with_specials.add_tokens.called


def test_sync_hf_and_pytorch_vocab():
    hf_tokenizer = mock.create_autospec(spec=PreTrainedTokenizer)
    hf_tokenizer.get_vocab.return_value = {"a": 1, "b": 0, "c": 2}
    vocab = setup_sequence.sync_hf_and_pytorch_vocab(hf_tokenizer=hf_tokenizer)
    assert isinstance(vocab, Vocab)
    assert vocab.get_stoi() == {"a": 1, "b": 0, "c": 2}


@mock.patch(f"{_get_mock_target()}._get_hf_tokenizer")
@mock.patch(f"{_get_mock_target()}.sync_hf_and_pytorch_vocab")
def test_get_sequence_input_objects_from_pretrained(sync_vocab_mock, tokenizer_mock):
    input_config = mock.create_autospec(spec=schemas.InputConfig)
    input_config.input_type_info = mock.create_autospec(
        spec=schemas.SequenceInputDataConfig
    )
    input_config.input_type_info.vocab_file = None

    input_config.model_config = mock.Mock()
    input_config.model_config.model_type = "bert-base-uncased"
    tokenizer_mock.return_value.encode.return_value = [1, 2, 3]

    func = setup_sequence.get_sequence_input_objects_from_pretrained
    (vocab, stats, tokenizer, encode) = func(input_config=input_config, mode="train")

    assert tokenizer_mock.called
    assert sync_vocab_mock.called
    assert isinstance(stats, setup_sequence.GatheredSequenceStats)
    assert encode("input string") == [1, 2, 3]


def test_yield_tokens_from_file(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "vocab.json"
    p.write_text("a\nb\nc\nd")

    gathered_stats = setup_sequence.GatheredSequenceStats()

    result = list(
        setup_sequence.yield_tokens_from_file(
            file_path=str(p),
            split_on=None,
            gathered_stats=gathered_stats,
        )
    )

    assert gathered_stats.total_files == 1
    assert gathered_stats.total_count == 4
    assert gathered_stats.max_length == 1
    assert result == ["a", "b", "c", "d"]


def test_yield_tokens_from_file_handle_different_splits(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "vocab.txt"

    p.write_text("a b c\nd e f")

    gathered_stats = setup_sequence.GatheredSequenceStats()

    result = list(
        setup_sequence.yield_tokens_from_file(
            file_path=str(p),
            split_on=" ",
            gathered_stats=gathered_stats,
        )
    )

    assert result == [["a", "b", "c"], ["d", "e", "f"]]


def test_yield_tokens_from_file_retain_newline(tmp_path):
    tmp_dir = tmp_path / "sub"
    tmp_dir.mkdir()
    vocab_file = tmp_dir / "vocab.txt"

    vocab_file.write_text("a\n\nb")
    gathered_stats = setup_sequence.GatheredSequenceStats()
    result = list(
        setup_sequence.yield_tokens_from_file(
            file_path=str(vocab_file),
            split_on=None,
            gathered_stats=gathered_stats,
        )
    )

    assert result == ["a", "", "b"]


def test_update_max_length(tmp_path):
    tmp_dir = tmp_path / "sub"
    tmp_dir.mkdir()
    vocab_file = tmp_dir / "vocab.txt"

    vocab_file.write_text("a\nbc\ndef")

    gathered_stats = setup_sequence.GatheredSequenceStats()
    _ = list(
        setup_sequence.yield_tokens_from_file(
            file_path=str(vocab_file),
            split_on="",
            gathered_stats=gathered_stats,
        )
    )

    assert gathered_stats.max_length == 3
