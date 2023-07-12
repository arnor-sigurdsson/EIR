from typing import TYPE_CHECKING, DefaultDict, Optional, Union

from tqdm import tqdm

from eir.data_load.data_source_modules.common_utils import add_id_to_samples
from eir.data_load.label_setup import al_label_dict

if TYPE_CHECKING:
    from eir.data_load.datasets import Sample


def add_tabular_data_to_samples(
    tabular_dict: al_label_dict,
    samples: DefaultDict[str, "Sample"],
    ids_to_keep: Optional[set[str]],
    source_name: str = "Tabular Data",
) -> DefaultDict[str, "Sample"]:
    def _get_tabular_iterator(ids_to_keep_: Union[None, set[str]]):
        for sample_id_, tabular_inputs_ in tabular_dict.items():
            if ids_to_keep_ and sample_id_ not in ids_to_keep_:
                continue

            yield sample_id_, tabular_inputs_

    known_length = None if not ids_to_keep else len(ids_to_keep)
    tabular_iterator = tqdm(
        _get_tabular_iterator(ids_to_keep_=ids_to_keep),
        desc=source_name,
        total=known_length,
    )

    for sample_id, tabular_inputs in tabular_iterator:
        samples = add_id_to_samples(samples=samples, sample_id=sample_id)

        samples[sample_id].inputs[source_name] = tabular_inputs

    return samples
