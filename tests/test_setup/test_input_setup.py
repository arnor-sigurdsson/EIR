import random
from pathlib import Path
from typing import Sequence, TYPE_CHECKING, List

import pandas as pd
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
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {"l1": 1e-03},
                        },
                    },
                    {
                        "input_info": {"input_name": "test_sequence"},
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "input_cat_columns": ["OriginExtraCol"],
                            "input_con_columns": ["ExtraTarget"],
                        },
                        "model_config": {"model_type": "tabular"},
                    },
                ],
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
        name_from_config = config.input_info.input_name
        assert name == name_from_config


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


def test_get_bpe_tokenizer(tmp_path: Path):

    seq_path = tmp_path / "test_sequence"
    test_pool = _get_simple_sample_pool()
    set_up_simple_sequence_test_data(
        path=seq_path, pool=test_pool, sep=" ", n_samples=100, max_length=20
    )

    # Core functionality
    vocab_iter_test_training = input_setup.get_vocab_iterator(
        input_source=str(seq_path),
        split_on=" ",
        gathered_stats=input_setup.GatheredSequenceStats(),
    )
    bpe_tokenizer_from_scratch = input_setup.get_bpe_tokenizer(
        vocab_iterator=vocab_iter_test_training, vocab_file=None
    )
    known_sequence = "cat dog mouse".split()
    known_sequence_tokenized = bpe_tokenizer_from_scratch(known_sequence)
    assert known_sequence_tokenized == ["cat", "dog", "mouse"]

    unknown_sequence = "edge knot city".split()
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
    vocab_iter_test_saving = input_setup.get_vocab_iterator(
        input_source=str(seq_path),
        split_on=" ",
        gathered_stats=input_setup.GatheredSequenceStats(),
    )
    bpe_tokenizer_object = input_setup._get_bpe_tokenizer_object(
        vocab_iterator=vocab_iter_test_saving, vocab_file=None
    )
    saved_bpe_path = tmp_path / "test_bpe.json"
    bpe_tokenizer_object.save(str(saved_bpe_path))

    bpe_tokenizer_from_pretrained = input_setup.get_bpe_tokenizer(
        vocab_iterator=None, vocab_file=str(saved_bpe_path)
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
    gathered_stats_general = input_setup.GatheredSequenceStats()
    vocab_iter_test_general = input_setup.get_vocab_iterator(
        input_source=str(seq_path),
        split_on=" ",
        gathered_stats=gathered_stats_general,
    )
    vocab = set(word for sequence in vocab_iter_test_general for word in sequence)
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


@pytest.mark.parametrize(
    "indices_to_subset,expected",
    [
        ([0, 10, 20, 30, 40, 50, 60, 70], [0, 10, 20, 30, 40, 50, 60, 70]),
        ([0, 10, 20, 30, 40, 50, 60, 70, 120, 999], [0, 10, 20, 30, 40, 50, 60, 70]),
    ],
)
def test_setup_subset_indices(indices_to_subset: List[int], expected: List[int]):
    test_data = []
    for i in range(100):
        cur_row_data = ["1", str(i), "0.1", str(i), "A", "T"]
        test_data.append(cur_row_data)

    df_bim_test = pd.DataFrame(test_data, columns=input_setup._get_bim_headers())

    snps_to_subset = [str(i) for i in indices_to_subset]

    subset_indices = input_setup._setup_snp_subset_indices(
        df_bim=df_bim_test, snps_to_subset=snps_to_subset
    )
    assert subset_indices.tolist() == expected


def test_read_snp_df(tmp_path):
    snp_file_str = """
            1     rs3094315        0.020130          752566 G A
            1    rs7419119         0.022518          842013 G T
            1   rs13302957         0.024116          891021 G A
            1    rs6696609         0.024457          903426 T C
            1       rs8997         0.025727          949654 A G
            1    rs9442372         0.026288         1018704 A G
            1    rs4970405         0.026674         1048955 G A
            1   rs11807848         0.026711         1061166 C T
            1    rs4970421         0.028311         1108637 A G
            1    rs1320571         0.028916         1120431 A G
               """
    file_ = tmp_path / "data_final.bim"
    file_.write_text(snp_file_str)

    df_bim = input_setup.read_bim(bim_file_path=str(file_))
    snp_arr = df_bim["VAR_ID"].array
    assert len(snp_arr) == 10
    assert snp_arr[0] == "rs3094315"
    assert snp_arr[-1] == "rs1320571"
