from typing import TYPE_CHECKING, DefaultDict, Optional

import pandas as pd
from tqdm import tqdm

from eir.data_load.data_source_modules.common_utils import add_id_to_samples

if TYPE_CHECKING:
    from eir.data_load.datasets import Sample


def add_tabular_data_to_samples(
    df_tabular: pd.DataFrame,
    samples: DefaultDict[str, "Sample"],
    ids_to_keep: Optional[set[str]] = None,
    source_name: str = "Tabular Data",
) -> DefaultDict[str, "Sample"]:

    def _get_tabular_iterator(ids_to_keep_: Optional[set[str]] = None):
        fields = df_tabular.columns
        for row in df_tabular.itertuples():
            sample_id_ = str(getattr(row, "Index"))
            if ids_to_keep_ is not None and sample_id_ not in ids_to_keep_:
                continue

            tabular_inputs_ = {field: getattr(row, field) for field in fields}

            yield sample_id_, tabular_inputs_

    known_length = None if ids_to_keep is None else len(ids_to_keep)
    tabular_iterator = tqdm(
        _get_tabular_iterator(ids_to_keep_=ids_to_keep),
        desc=source_name,
        total=known_length,
    )

    for sample_id, tabular_inputs in tabular_iterator:
        samples = add_id_to_samples(samples=samples, sample_id=sample_id)

        samples[sample_id].inputs[source_name] = tabular_inputs

    return samples
