from typing import Dict, Sequence

from eir.models.input.tabular.tabular import (
    SimpleTabularModel,
    SimpleTabularModelConfig,
)


def get_tabular_model(
    model_init_config: SimpleTabularModelConfig,
    cat_columns: Sequence[str],
    con_columns: Sequence[str],
    device: str,
    unique_label_values: Dict[str, set[int]],
) -> SimpleTabularModel:
    tabular_model = SimpleTabularModel(
        model_init_config=model_init_config,
        cat_columns=cat_columns,
        con_columns=con_columns,
        unique_label_values_per_column=unique_label_values,
        device=device,
    )

    return tabular_model
