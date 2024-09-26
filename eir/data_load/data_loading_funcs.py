from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from statistics import mean
from typing import (
    TYPE_CHECKING,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.data_load.data_utils import Sample
    from eir.data_load.datasets import DatasetBase  # noqa: F401

logger = get_logger(name=__name__, tqdm_compatible=True)


al_sample_weight_and_counts = Dict[str, torch.Tensor | list[int]]


def get_weighted_random_sampler(
    samples: Iterable["Sample"],
    columns_to_sample: List[str],
) -> WeightedRandomSampler:
    """
    Labels spec:

    {
        {
        ID1:
            {
                output_name:
                {
                    Label Column: Target Value,
                    Extra Column 1: Extra Column 1 Value
                    Extra Column 2: Extra Column 2 Value}
                }
            },
        },
        ID2: {...}
    }

    The list comprehension is going over all the label dicts associated with the IDs,
    then just parsing the label (converting to int in the case of classification).
    """

    parsed_weighted_sample_columns = _build_weighted_sample_dict_from_config_sequence(
        config_list=columns_to_sample
    )

    all_column_weights = {}
    samples_list = list(samples)
    n_samples = len(samples_list)

    if n_samples < 10_000:
        max_workers = 1
    else:
        max_workers = None

    logger.debug(
        "Using %s workers for weighted sampling setup based on %d samples.",
        max_workers,
        n_samples,
    )

    process_output_partial = partial(_process_output, samples_list)
    logger.debug("Setting up weighted sampling statistics.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        task_iter = parsed_weighted_sample_columns.items()
        for output_name, weighted_columns_list in task_iter:
            logger.debug(f"Setting up weighted sampling for output '{output_name}'.")
            futures.append(
                executor.submit(
                    process_output_partial,
                    output_name,
                    weighted_columns_list,
                )
            )

        for future in as_completed(futures):
            output_name, cur_column_weights = future.result()
            for cur_target, cur_weight_object in cur_column_weights.items():
                all_column_weights[f"{output_name}.{cur_target}"] = cur_weight_object

    samples_weighted, num_sample_per_epoch = _aggregate_column_sampling_weights(
        all_target_columns_weights_and_counts=all_column_weights
    )

    logger.debug(
        "Num samples per epoch according to average target class counts in %s: %d",
        columns_to_sample,
        num_sample_per_epoch,
    )
    sampler = WeightedRandomSampler(
        weights=samples_weighted, num_samples=num_sample_per_epoch, replacement=True
    )

    return sampler


def _process_output(
    samples: Iterable["Sample"], output_name: str, weighted_columns_list: List[str]
) -> Tuple[str, Dict[str, al_sample_weight_and_counts]]:
    logger.debug(f"Setting up weighted sampling for output {output_name}")
    cur_column_weights = _gather_column_sampling_weights(
        samples=samples,
        output_name=output_name,
        columns_to_sample=weighted_columns_list,
    )
    return output_name, cur_column_weights


def _build_weighted_sample_dict_from_config_sequence(
    config_list: List[str],
) -> Dict[str, List[str]]:
    weighted_sample_dict = {}

    for weighted_sample_config_string in config_list:
        if weighted_sample_config_string == "all":
            return {"all": ["all"]}

        output_name, sample_column = weighted_sample_config_string.split(".", 1)
        if output_name not in weighted_sample_dict:
            weighted_sample_dict[output_name] = [sample_column]
        else:
            weighted_sample_dict[output_name].append(sample_column)

    return weighted_sample_dict


def _gather_column_sampling_weights(
    samples: Iterable["Sample"],
    output_name: str,
    columns_to_sample: Iterable[str],
) -> Dict[str, al_sample_weight_and_counts]:
    all_target_label_weight_dicts: dict[str, al_sample_weight_and_counts] = {}

    for column in columns_to_sample:
        try:
            cur_label_iterable = (
                (
                    i.target_labels[output_name][column]
                    if output_name in i.target_labels
                    and column in i.target_labels[output_name]
                    else np.nan
                )
                for i in samples
            )
            cur_label_iterable_int: Generator[int | float, None, None] = (
                int(i) if not np.isnan(i) else np.nan for i in cur_label_iterable
            )
            cur_weight_dict = _get_column_label_weights_and_counts(
                label_iterable=cur_label_iterable_int,
                column_name=column,
            )

            logger.debug(
                "Label counts in column %s: %s", column, cur_weight_dict["label_counts"]
            )

            all_target_label_weight_dicts[column] = cur_weight_dict
        except ValueError as e:
            logger.warning(f"Skipping column {column} due to error: {str(e)}.")
        except KeyError:
            logger.warning(f"Skipping column {column} due to missing data.")

    if not all_target_label_weight_dicts:
        logger.warning("No valid columns found for sampling weights")

    return all_target_label_weight_dicts


def _get_column_label_weights_and_counts(
    label_iterable: Iterable[int | float], column_name: Optional[str] = None
) -> al_sample_weight_and_counts:
    """
    We have the assertion to make sure we have a unique integer for each label, starting
    with 0 as we use it to index into the weights directly.

    TODO:   Optimize so we do just one pass over `train_dataset.samples` if this becomes
            a bottleneck.
    """

    def _check_labels(label_list: list[int | float]):
        labels_set = set(label_list)
        found_labels = sorted(list(labels_set))
        expected_labels = list(range(len(found_labels)))

        if found_labels != expected_labels:
            raise ValueError(
                f"When setting up weighed sampling for column {column_name}, "
                f"all labels must be present in the training set. "
                f"Expected at least {max(expected_labels)} labels, "
                f"but got {len(found_labels)}. "
                "This is likely due to a mismatch between the training set and the "
                "validation set, possibly due to rare labels in the data that e.g. "
                "only appear in the validation set after splitting. "
                "Weighed sampling is therefore not supported for this "
                "column."
            )

    labels = list(label_iterable)
    valid_labels = [label for label in labels if not np.isnan(label)]

    _check_labels(label_list=valid_labels)

    label_counts: list[int] = [int(i[1]) for i in sorted(Counter(valid_labels).items())]

    weights = 1.0 / torch.tensor(label_counts, dtype=torch.float32)
    samples_weighted = torch.tensor(
        [weights[int(label)] if not np.isnan(label) else np.nan for label in labels]
    )

    output_dict: al_sample_weight_and_counts = {
        "samples_weighted": samples_weighted,
        "label_counts": label_counts,
    }
    return output_dict


def _aggregate_column_sampling_weights(
    all_target_columns_weights_and_counts: dict[str, al_sample_weight_and_counts]
) -> Tuple[Sequence[float], int]:
    """
    We sum up the normalized weights for each target column to create the final sampling
    weights.

    As for the samples per epoch, we take the average of the class counts per target,
    then sum those up.
    """

    all_weights = torch.stack(
        [
            torch.Tensor(i["samples_weighted"])
            for i in all_target_columns_weights_and_counts.values()
        ],
        dim=1,
    )

    all_weights = torch.nan_to_num(all_weights, nan=0.0)
    all_weights_summed = all_weights.sum(dim=1)

    samples_per_epoch = int(
        mean(
            mean(i["label_counts"])
            for i in all_target_columns_weights_and_counts.values()
        )
    )
    samples_per_epoch = min(len(all_weights_summed), samples_per_epoch)

    all_weights_summed_list = all_weights_summed.tolist()

    return all_weights_summed_list, samples_per_epoch
