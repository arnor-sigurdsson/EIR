from collections import Counter, OrderedDict
from typing import Iterable, Optional

from eir.setup.input_setup_modules.torchtext_port.vocab import Vocab


def vocab(
    ordered_dict: dict[str, int],
    min_freq: int = 1,
    specials: Optional[list[str]] = None,
    special_first: bool = True,
) -> Vocab:
    """
    Factory method for creating a vocab object which maps tokens to indices.

    Note that the ordering in which key-value pairs were inserted in the
    `ordered_dict` will be respected when building the vocab.
    Therefore, if sorting by token frequency is important to the user, the
    `ordered_dict` should be created in a way to reflect this.

    Args:
        ordered_dict: Ordered Dictionary mapping tokens to their
        corresponding occurrence frequencies.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
        specials: Special symbols to add. The order of supplied tokens will be
        preserved.
        special_first: Indicates whether to insert special symbols at the beginning or
        at the end.

    Returns:
        Vocab: A `Vocab` object

    Examples:
        >>> ordered_dict = OrderedDict([('a', 5), ('b', 3), ('c', 1)])
        >>> v1 = vocab(ordered_dict)
        >>> print(v1['a'])  # prints 0
        >>> print(v1['c'])  # prints 2
        >>> v2 = vocab(ordered_dict, specials=['<unk>'])
        >>> print(v2['<unk>'])  # prints 0
        >>> print(v2['a'])  # prints 1
        >>> v2.set_default_index(v2['<unk>'])
        >>> print(v2['out of vocab'])  # prints 0 (index of <unk>)
    """
    specials = specials or []
    tokens = []

    # Add special tokens
    if special_first:
        tokens.extend(specials)

    # Add tokens that meet the minimum frequency
    for token, freq in ordered_dict.items():
        if freq >= min_freq and token not in specials:
            tokens.append(token)

    # Add special tokens at the end if not added at the beginning
    if not special_first:
        tokens.extend(specials)

    # Create and return the Vocab object
    return Vocab(tokens)


def build_vocab_from_iterator(
    iterator: Iterable,
    min_freq: int = 1,
    specials: Optional[list[str]] = None,
    special_first: bool = True,
    max_tokens: Optional[int] = None,
) -> Vocab:
    """
    Build a Vocab from an iterator.

    Args:
        iterator: Iterator used to build Vocab. Must yield list or iterator of tokens.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
        specials: Special symbols to add. The order of supplied tokens will be
        preserved.
        special_first: Indicates whether to insert special symbols at the beginning
        or at the end.
        max_tokens: If provided, creates the vocab from the
        `max_tokens - len(specials)` most frequent tokens.

    Returns:
        Vocab: A `Vocab` object

    Examples:
        >>> def yield_tokens(texts):
        ...     for text in texts:
        ...         yield text.split()
        >>> texts = ["the quick brown fox", "jumps over the lazy dog", ... ]
        >>> vocab = build_vocab_from_iterator(yield_tokens(texts), specials=["<unk>"])
        >>> print(vocab.get_itos())  # prints ['<unk>', 'the', 'lazy', 'dog', ...]
        >>> vocab.set_default_index(vocab['<unk>'])
        >>> print(vocab['river'])  # prints 0 (index of <unk>)
    """
    counter: Counter = Counter()
    for tokens in iterator:
        counter.update(tokens)

    specials = specials or []

    # Sort by descending frequency, then lexicographically
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    if max_tokens is not None:
        if len(specials) >= max_tokens:
            raise ValueError(
                "len(specials) >= max_tokens, so the "
                "vocab will be entirely special tokens."
            )
        sorted_by_freq_tuples = sorted_by_freq_tuples[: max_tokens - len(specials)]

    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    return vocab(
        ordered_dict=ordered_dict,
        min_freq=min_freq,
        specials=specials,
        special_first=special_first,
    )
