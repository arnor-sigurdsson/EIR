from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol

import numpy as np

from eir.data_load.label_setup import al_label_transformers_object
from eir.interpretation.interpretation_utils import get_target_class_name

if TYPE_CHECKING:
    from eir.interpretation.interpretation import SampleAttribution


def analyze_array_input_attributions(
    attribution_outfolder: Path,
    all_attributions: dict[str, np.ndarray],
):
    for target_class, value in all_attributions.items():
        np.save(
            file=str(attribution_outfolder / f"{target_class}.npy"),
            arr=value,
            allow_pickle=True,
        )


class ArrayConsumerCallable(Protocol):
    def __call__(
        self,
        attribution: Optional["SampleAttribution"],
    ) -> Optional[dict[str, np.ndarray]]: ...


def get_array_sum_consumer(
    target_transformer: "al_label_transformers_object",
    input_name: str,
    output_name: str,
    target_column: str,
    column_type: str,
) -> ArrayConsumerCallable:
    results: dict[str, np.ndarray] = {}
    n_samples: dict[str, int] = {}

    def _consumer(
        attribution: Optional["SampleAttribution"],
    ) -> Optional[dict[str, np.ndarray]]:
        nonlocal results
        nonlocal n_samples

        if attribution is None:
            for key, value in results.items():
                results[key] = value / n_samples[key]
            return results

        sample_target_labels = attribution.sample_info.target_labels

        cur_label_name = get_target_class_name(
            sample_label=sample_target_labels[output_name][target_column],
            target_transformer=target_transformer,
            column_type=column_type,
            target_column_name=target_column,
        )

        sample_acts = attribution.sample_attributions[input_name].squeeze()
        if cur_label_name not in results:
            results[cur_label_name] = sample_acts
            n_samples[cur_label_name] = 1
        else:
            results[cur_label_name] += sample_acts
            n_samples[cur_label_name] += 1

        return None

    return _consumer
