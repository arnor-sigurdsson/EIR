import argparse
import os
from collections.abc import Iterator

from datasets import load_dataset

from eir.setup.input_setup_modules.setup_sequence import (
    _get_bpe_tokenizer_object,
)
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


def create_dataset_text_iterator(
    dataset_name: str,
    name: str | None = None,
    max_samples: int | None = 100000,
    text_field: str = "text",
) -> Iterator[str]:
    logger.info(f"Loading dataset {dataset_name} (name: {name if name else 'all'})...")

    dataset_args = {
        "split": "train",
        "streaming": True,
        "trust_remote_code": True,
    }

    if name:
        dataset_args["name"] = name

    dataset = load_dataset(dataset_name, **dataset_args)

    sample_count = 0

    for sample in dataset:
        if max_samples and sample_count >= max_samples:
            break

        text = sample.get(text_field, "")

        if text and isinstance(text, str) and len(text.strip()) > 0:
            yield text.strip()
            sample_count += 1


def main():
    parser = argparse.ArgumentParser(
        description="Train BPE tokenizer from a streaming dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="sample-10BT",
        help="Name of subset to use.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=49152,
        help="Target vocabulary size",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100_000,
        help="Maximum number of samples to use for training",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for tokenizer (.json)",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Field name containing text in the dataset",
    )

    args = parser.parse_args()

    if not args.output.endswith(".json"):
        args.output = args.output + ".json"

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    logger.info(
        f"Creating text iterator from {args.dataset} with {args.max_samples} samples..."
    )
    vocab_iterator = create_dataset_text_iterator(
        dataset_name=args.dataset,
        name=args.name,
        max_samples=args.max_samples,
        text_field=args.text_field,
    )

    logger.info(
        f"Training BPE tokenizer with target vocabulary size {args.vocab_size}..."
    )
    tokenizer = _get_bpe_tokenizer_object(
        vocab_iterator=vocab_iterator,
        vocab_file=None,
        vocab_size=args.vocab_size,
        raise_on_validation_error=False,
    )

    logger.info(f"Saving tokenizer to {args.output}...")
    tokenizer.save(args.output)

    logger.info("Tokenizer trained successfully")
    logger.info(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    test_text = "Hello, this is a test of the newly trained tokenizer."
    encoding = tokenizer.encode(test_text)
    logger.info(f"Test encoding: '{test_text}'")
    logger.info(f"Tokens: {encoding.ids[:10]}... (showing first 10)")
    logger.info(f"Decoded: '{tokenizer.decode(encoding.ids)}'")


if __name__ == "__main__":
    main()
