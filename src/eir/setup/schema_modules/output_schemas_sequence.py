from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from eir.setup.schemas import al_max_sequence_length, al_tokenizer_choices

al_sequence_operations = Literal["autoregressive", "mlm"]


@dataclass
class SequenceOutputTypeConfig:
    """
    :param vocab_file:
        An optional text file containing pre-defined vocabulary to use
        for the training. If this is not passed in, the framework will automatically
        build the vocabulary from the training data. Passing in a vocabulary file is
        therefore useful if (a) you want to manually specify / limit the vocabulary used
        and/or (b) you want to save time by pre-computing the vocabulary.

        Here, there are two formats supported:

        - A ``.json`` file containing a dictionary with the vocabulary as keys and
         the corresponding token IDs as values. For example:
         ``{"the": 0, "cat": 1, "sat": 2, "on": 3, "the": 4, "mat": 5}``

        - A ``.json`` file with the results of training and saving the vocabulary of
         a Huggingface BPE tokenizer. This is the file create by calling
         ``hf_tokenizer.save()``. This is only valid when using the ``bpe`` tokenizer.

    :param max_length:
        Maximum length to truncate/pad sequences to. This can be an integer or the
        values 'max' or 'average'. The 'max' keyword will use the maximum sequence
        length found in the training data, while the 'average' will use the average
        length across all training samples.

    :param sampling_strategy_if_longer:
        Controls how sequences are truncated if they are longer than the specified
        ``max_length`` parameter. Using 'from_start' will always truncate from the
        beginning of the sequence, ensuring the the samples will always be the same
        during training. Setting this parameter to ``uniform`` will uniformly sample
        a slice of a given sample sequence during training. Note that for consistency,
        the validation/test set samples always use the ``from_start`` setting when
        truncating.

    :param min_freq:
        Minimum number of times a token must appear in the total training data to be
        included in the vocabulary. Note that this setting will not do anything if
        passing in ``vocab_file``.

    :param split_on:
        Which token to split the sequence on to generate separate tokens for the
        vocabulary.

    :param tokenizer:
        Which tokenizer to use. Relevant if modelling on language, but not as much when
        doing it on other arbitrary sequences.

    :param tokenizer_language:
        Which language rules the tokenizer should apply when tokenizing the raw data.

    :param adaptive_tokenizer_max_vocab_size:
        If using an adaptive tokenizer (``"bpe"``), this parameter controls the maximum
        size of the vocabulary.

    :param sequence_operation:
        Which operation to perform on the sequence. Currently only ``autoregressive``
        is supported, which means that the model will be trained to predict the next
        token in the sequence given the previous tokens.
    """

    vocab_file: None | str = None
    max_length: "al_max_sequence_length" = "average"
    sampling_strategy_if_longer: Literal["from_start", "uniform"] = "uniform"
    min_freq: int = 10
    split_on: str | None = " "
    tokenizer: "al_tokenizer_choices" = None  # type: ignore
    tokenizer_language: str | None = None
    adaptive_tokenizer_max_vocab_size: int | None = None

    sequence_operation: al_sequence_operations = "autoregressive"


@dataclass
class SequenceOutputSamplingConfig:
    """
    :param manual_inputs:
        Manually specified inputs to use for sequence generation. This is useful
        if you want to generate sequences based on a specific input. Depending
        on the input type, different formats are expected:

        - ``sequence``: A string written directly in the ``.yaml`` file.
        - ``omics``: A file path to NumPy array of shape ``(4, n_SNPs)`` on disk.
        - ``image``: An image file path on disk.
        - ``tabular``: A mapping of :literal:`(column key: value)` written directly
          in the ``.yaml`` file.
        - ``array``: A file path to NumPy array on disk.
        - ``bytes``: A file path to a file on disk.


    :param n_eval_inputs:
        The number of inputs automatically sampled from the validation set for
        sequence generation.

    :param generated_sequence_length:
        The length of the output sequences that are generated.

    :param temperature:
        Controls the randomness of predictions by scaling the logits before applying
        softmax. A higher temperature results in more random predictions, while a
        lower temperature results in more deterministic predictions.

    :param repetition_penalty:
        Discourages repetition by reducing the probability of tokens that have
        already appeared in the generated text. Values greater than 1.0 apply
        the penalty, with higher values (1.2-1.5) reducing repetition more
        aggressively. A value of 1.0 disables this feature.

    :param repetition_penalty_max_window:
        The maximum number of most recent tokens to consider when applying the
        repetition penalty. A smaller window focuses on preventing local repetition,
        while a larger window prevents repetition across the entire sequence.

    :param frequency_penalty:
        Reduces the probability of tokens proportional to how frequently they've
        appeared in the generated text. Unlike repetition penalty, this scales with
        usage count. Positive values (0.1-0.3) increase diversity, with higher values
        producing more varied vocabulary.

    :param frequency_penalty_max_window:
        The maximum number of most recent tokens to track when calculating token
        frequencies for the frequency penalty. Larger windows maintain longer-term
        memory of word usage patterns.

    :param top_k:
        The number of top candidates to consider when sampling the next token
        in an output sequence. By default, the model considers the top 20 candidates

    :param top_p:
        The cumulative probability of the top candidates to consider when sampling
        the next token in an output sequence. For example, if top_p is 0.9, the model
        will stop sampling candidates once the cumulative probability of the most
        likely candidates reaches 0.9.

    :param tau:
        Controls locally typical sampling by filtering tokens based on how close their
        probabilities are to the expected distribution. Values range from 0.0 to 1.0,
        where 1.0 disables the filter. Lower values produce more consistent text by
        removing outlier tokens.
    """

    manual_inputs: Sequence[dict[str, str]] = ()
    n_eval_inputs: int = 10

    generated_sequence_length: int = 64

    repetition_penalty: float = 1.1
    repetition_penalty_max_window: int = 64
    frequency_penalty: float = 0.1
    frequency_penalty_max_window: int = 128
    temperature: float = 0.7
    top_k: int = 20
    top_p: float = 0.9
    tau: float = 0.95
