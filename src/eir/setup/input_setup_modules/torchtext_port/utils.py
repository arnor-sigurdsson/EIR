import re
from functools import partial


def _split_tokenizer(x):  # noqa: F821
    # type: (str) -> list[str]
    return x.split()


def _spacy_tokenize(x, spacy):
    return [tok.text for tok in spacy.tokenizer(x)]


_patterns = [
    r"\'",
    r"\"",
    r"\.",
    r"<br \/>",
    r",",
    r"\(",
    r"\)",
    r"\!",
    r"\?",
    r"\;",
    r"\:",
    r"\s+",
]

_replacements = [
    " '  ",
    "",
    " . ",
    " ",
    " , ",
    " ( ",
    " ) ",
    " ! ",
    " ? ",
    " ",
    " ",
    " ",
]

_patterns_dict = [
    (re.compile(p), r) for p, r in zip(_patterns, _replacements, strict=False)
]


def _basic_normalize(line):
    r"""
    Basic normalization for a line of text.
    Normalization includes
    - lowercasing
    - complete some basic text normalization for English words as follows:
        add spaces before and after '\''
        remove '\"',
        add spaces before and after '.'
        replace '<br \/>'with single space
        add spaces before and after ','
        add spaces before and after '('
        add spaces before and after ')'
        add spaces before and after '!'
        add spaces before and after '?'
        replace ';' with single space
        replace ':' with single space
        replace multiple spaces with single space

    Returns a list of tokens after splitting on whitespace.
    """

    line = line.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line.split()


def get_tokenizer(tokenizer, language="en"):
    r"""
    Generate tokenizer function for a string sentence.

    Args:
        tokenizer: the name of tokenizer function. If None, it returns split()
            function, which splits the string sentence by space.
            If basic_english, it returns _basic_english_normalize() function,
            which normalize the string first and split by space. If a callable
            function, it will return the function. If a tokenizer library
            (e.g. spacy, moses, toktok, revtok, subword), it returns the
            corresponding library.
        language: Default en

    Examples:
        >>> tokenizer = get_tokenizer("basic_english")
        >>> tokens = tokenizer("You can now install TorchText using pip!")
        >>> tokens
        >>> ['you', 'can', 'now', 'install', 'torchtext', 'using', 'pip', '!']

    """

    # default tokenizer is string.split(), added as a module function for serialization
    if tokenizer is None:
        return _split_tokenizer

    if tokenizer == "basic":
        return _basic_normalize

    if tokenizer == "basic_english":
        if language != "en":
            raise ValueError("Basic normalization is only available for English(en)")
        return _basic_normalize

    # simply return if a function is passed
    if callable(tokenizer):
        return tokenizer

    if tokenizer == "spacy":
        try:
            import spacy

            try:
                spacy = spacy.load(language)
            except OSError:
                # Model shortcuts no longer work in spaCy 3.0+, try using fullnames
                # List is from
                # https://github.com/explosion/spaCy/blob/
                # b903de3fcb56df2f7247e5b6cfa6b66f4ff02b62/spacy/errors.py#L789
                old_model_shortcuts = (
                    spacy.errors.OLD_MODEL_SHORTCUTS
                    if hasattr(spacy.errors, "OLD_MODEL_SHORTCUTS")
                    else {}
                )
                if language not in old_model_shortcuts:
                    raise
                import warnings

                warnings.warn(
                    f'Spacy model "{language}" could not be loaded,'
                    f' trying "{old_model_shortcuts[language]}" instead',
                    stacklevel=2,
                )
                spacy = spacy.load(old_model_shortcuts[language])
            return partial(_spacy_tokenize, spacy=spacy)
        except ImportError:
            print(
                "Please install SpaCy. "
                "See the docs at https://spacy.io for more information."
            )
            raise
        except AttributeError:
            print(
                f"Please install SpaCy and the SpaCy {language} tokenizer. "
                "See the docs at https://spacy.io for more "
                "information."
            )
            raise
    elif tokenizer == "moses":
        try:
            from sacremoses import MosesTokenizer

            moses_tokenizer = MosesTokenizer()
            return moses_tokenizer.tokenize
        except ImportError:
            print(
                "Please install SacreMoses. "
                "See the docs at https://github.com/alvations/sacremoses "
                "for more information."
            )
            raise
    elif tokenizer == "toktok":
        try:
            from nltk.tokenize.toktok import ToktokTokenizer

            toktok = ToktokTokenizer()
            return toktok.tokenize
        except ImportError:
            print(
                "Please install NLTK. "
                "See the docs at https://nltk.org  for more information."
            )
            raise
    elif tokenizer == "revtok":
        try:
            import revtok

            return revtok.tokenize
        except ImportError:
            print("Please install revtok.")
            raise
    elif tokenizer == "subword":
        try:
            import revtok

            return partial(revtok.tokenize, decap=True)
        except ImportError:
            print("Please install revtok.")
            raise
    raise ValueError(
        f"Requested tokenizer {tokenizer}, valid choices are a "
        "callable that takes a single string as input, "
        '"revtok" for the revtok reversible tokenizer, '
        '"subword" for the revtok caps-aware tokenizer, '
        '"spacy" for the SpaCy English tokenizer, or '
        '"moses" for the NLTK port of the Moses tokenization '
        "script."
    )
