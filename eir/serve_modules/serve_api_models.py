from typing import Any, Sequence, Type

from pydantic import BaseModel, create_model

from eir.setup.schemas import InputConfig, TabularInputDataConfig


def create_tabular_model(
    name: str,
    cat_columns: Sequence[str],
    con_columns: Sequence[str],
) -> Type[BaseModel]:
    fields: dict[str, Any] = {col: (str, ...) for col in cat_columns}
    fields.update({col: (float, ...) for col in con_columns})
    return create_model(name, **fields)


def create_input_model(configs: Sequence[InputConfig]) -> Type[BaseModel]:
    fields: dict[str, Any] = {}

    for config in configs:
        input_type = config.input_info.input_type
        input_type_info = config.input_type_info

        if input_type in {"sequence", "bytes", "omics", "array", "image"}:
            fields[config.input_info.input_name] = (str, ...)
        elif input_type == "tabular":
            assert isinstance(input_type_info, TabularInputDataConfig)
            tabular_model = create_tabular_model(
                name=f"{config.input_info.input_name}_Model",
                cat_columns=list(input_type_info.input_cat_columns),
                con_columns=list(input_type_info.input_con_columns),
            )
            fields[config.input_info.input_name] = (tabular_model, ...)
        else:
            raise ValueError(f"Unknown input type {input_type}")

    return create_model("DynamicInputModel", **fields)


class ResponseModel(BaseModel):
    result: list[dict[str, Any]]
