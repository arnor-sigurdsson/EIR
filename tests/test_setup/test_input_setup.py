import random
from pathlib import Path
from typing import Sequence, TYPE_CHECKING

import pytest
from aislib.misc_utils import ensure_path_exists

from eir.setup import input_setup

if TYPE_CHECKING:
    from eir.setup.config import Configs


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


def set_up_simple_vocab_file(vocab: Sequence[str], outpath: Path) -> Path:
    ensure_path_exists(path=outpath, is_folder=False)
    with open(outpath, "w") as out_file_handle:
        for word in vocab:
            out_file_handle.write(word + "\n")

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


@pytest.mark.parametrize(
    "create_test_data",
    [
        {"task_type": "binary", "modalities": ["omics", "sequence"]},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "cnn"},
                        "model_config": {"l1": 1e-03},
                    },
                    {
                        "input_info": {"input_name": "test_sequence"},
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "model_type": "tabular",
                            "extra_cat_columns": ["OriginExtraCol"],
                            "extra_con_columns": ["ExtraTarget"],
                        },
                    },
                ],
            },
        }
    ],
    indirect=True,
)
def test_get_input_name_config_iterator(create_test_config: "Configs"):
    test_configs = create_test_config

    named_input_configs_iterator = input_setup.get_input_name_config_iterator(
        input_configs=test_configs.input_configs
    )
    for name, config in named_input_configs_iterator:
        input_type = config.input_info.input_type
        assert name.startswith(input_type + "_")


def test_get_tokenizer():

    test_input = "the lazy dog JUMPED over:: the red fox or whatever".split()

    identity_tokenizer = input_setup.get_basic_tokenizer(
        tokenizer_name=None, tokenizer_language=None
    )
    assert identity_tokenizer(test_input) == test_input

    with pytest.raises(ValueError):
        input_setup.get_basic_tokenizer(
            tokenizer_name="revtok", tokenizer_language="is"
        )

    basic_english_tokenizer = input_setup.get_basic_tokenizer(
        tokenizer_name="basic_english", tokenizer_language="en"
    )

    test_input_tokenized_expected = (
        "the lazy dog jumped over the red fox or whatever".split()
    )
    test_input_tokenized = basic_english_tokenizer(test_input)
    assert test_input_tokenized == test_input_tokenized_expected


def test_get_tokenized_vocab_iterator():
    def _test_iterator():
        yield "the lazy dog JUMPED over:: the red fox or whatever".split()

    basic_english_tokenizer = input_setup.get_basic_tokenizer(
        tokenizer_name="basic_english", tokenizer_language="en"
    )

    tokenized_vocab_iterator = input_setup.get_tokenized_vocab_iterator(
        vocab_iterator=_test_iterator(), tokenizer=basic_english_tokenizer
    )
    results = [i for i in tokenized_vocab_iterator]
    assert len(results) == 1

    assert results[0] == "the lazy dog jumped over the red fox or whatever".split()


def test_get_vocab_iterator_basic(tmp_path: Path):

    seq_path = tmp_path / "test_sequence"
    test_pool = _get_simple_sample_pool()
    set_up_simple_sequence_test_data(
        path=seq_path, pool=test_pool, sep=" ", n_samples=100, max_length=20
    )

    gathered_stats = input_setup.GatheredSequenceStats()
    vocab_iter = input_setup.get_vocab_iterator(
        input_source=str(seq_path), split_on=" ", gathered_stats=gathered_stats
    )
    vocab = set(word for sequence in vocab_iter for word in sequence)
    assert vocab == set(test_pool)
    assert gathered_stats.total_files == 100


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
    gathered_stats_diff_split = input_setup.GatheredSequenceStats()
    vocab_iter_diff_split = input_setup.get_vocab_iterator(
        input_source=str(seq_path_diff_split),
        split_on="---",
        gathered_stats=gathered_stats_diff_split,
    )
    vocab_diff_split = set(
        word for sequence in vocab_iter_diff_split for word in sequence
    )
    assert vocab_diff_split == set(test_pool)
    assert gathered_stats_diff_split.total_files == 100
    assert gathered_stats_diff_split.max_length == 20


def test_get_vocab_iterator_vocab_file(tmp_path: Path):
    seq_path = tmp_path / "test_sequence_using_vocab"
    vocab_path = tmp_path / "vocab.txt"
    test_pool = _get_simple_sample_pool()
    vocab_file = set_up_simple_vocab_file(vocab=test_pool, outpath=vocab_path)

    gathered_stats_vocab = input_setup.GatheredSequenceStats()
    set_up_simple_sequence_test_data(
        path=seq_path, pool=test_pool, sep=" ", n_samples=100, max_length=20
    )
    vocab_iter_diff_split = input_setup.get_vocab_iterator(
        input_source=str(vocab_file),
        split_on=" ",
        gathered_stats=gathered_stats_vocab,
        vocab_file=str(vocab_file),
    )
    vocab = set(word for sequence in vocab_iter_diff_split for word in sequence)
    assert vocab == set(test_pool)
    assert gathered_stats_vocab.total_count == len(vocab)


def test_possibly_gather_all_stats_from_input(tmp_path):
    seq_path = tmp_path / "test_sequence_using_vocab"
    vocab_path = tmp_path / "vocab.txt"
    test_pool = _get_simple_sample_pool()
    vocab_file = set_up_simple_vocab_file(vocab=test_pool, outpath=vocab_path)

    gathered_stats_vocab = input_setup.GatheredSequenceStats()
    set_up_simple_sequence_test_data(
        path=seq_path,
        pool=test_pool,
        sep=" ",
        n_samples=100,
        min_length=20,
        max_length=21,
    )
    vocab_iter_diff_split = input_setup.get_vocab_iterator(
        input_source=str(vocab_file),
        split_on=" ",
        gathered_stats=gathered_stats_vocab,
        vocab_file=str(vocab_file),
    )
    vocab = set(word for sequence in vocab_iter_diff_split for word in sequence)
    assert vocab == set(test_pool)
    assert gathered_stats_vocab.total_count == len(vocab)

    new_gathered_stats = input_setup.possibly_gather_all_stats_from_input(
        prev_gathered_stats=gathered_stats_vocab,
        input_source=str(seq_path),
        vocab_file=str(vocab_file),
        split_on=" ",
        max_length="max",
    )
    assert new_gathered_stats.total_files == 100
    assert new_gathered_stats.max_length == 20
    assert new_gathered_stats.total_count == 100 * 20


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
    gathered_stats_max_length = input_setup.GatheredSequenceStats()
    vocab_iter_max_length = input_setup.get_vocab_iterator(
        input_source=str(seq_path),
        split_on=" ",
        gathered_stats=gathered_stats_max_length,
    )
    vocab_max_length = set(
        word for sequence in vocab_iter_max_length for word in sequence
    )
    assert vocab_max_length == set(test_pool)
    assert gathered_stats_max_length.total_files == 200
    assert gathered_stats_max_length.max_length == 20

    max_length_from_func = input_setup.get_max_length(
        max_length_config_value="max", gathered_stats=gathered_stats_max_length
    )
    assert max_length_from_func == 20

    avg_length_from_func = input_setup.get_max_length(
        max_length_config_value="average", gathered_stats=gathered_stats_max_length
    )
    assert avg_length_from_func < 20
