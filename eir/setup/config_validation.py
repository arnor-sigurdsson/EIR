import reprlib
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import pandas as pd

from eir.setup import schemas
from eir.setup.schema_modules.output_schemas_array import ArrayOutputTypeConfig
from eir.setup.schema_modules.output_schemas_image import ImageOutputTypeConfig
from eir.setup.schema_modules.output_schemas_tabular import TabularOutputTypeConfig

if TYPE_CHECKING:
    from eir.setup.config import Configs


def validate_train_configs(configs: "Configs") -> None:
    validate_input_configs(input_configs=configs.input_configs)
    validate_output_configs(output_configs=configs.output_configs)

    validate_config_sync(
        input_configs=configs.input_configs, output_configs=configs.output_configs
    )

    validate_tensor_broker_configs(
        input_configs=configs.input_configs,
        fusion_configs=[configs.fusion_config],
        output_configs=configs.output_configs,
    )


def validate_input_configs(input_configs: Sequence[schemas.InputConfig]) -> None:
    for input_config in input_configs:
        base_validate_input_info(input_info=input_config.input_info)
        input_source = Path(input_config.input_info.input_source)

        match input_config.input_type_info:
            case schemas.TabularInputDataConfig(
                input_cat_columns, input_con_columns, _, _
            ):
                expected_columns = list(input_cat_columns) + list(input_con_columns)
                validate_tabular_file(
                    file_to_check=input_source,
                    expected_columns=expected_columns,
                    name="Tabular input",
                )


def base_validate_input_info(input_info: schemas.InputDataConfig) -> None:
    if not Path(input_info.input_source).exists():
        raise ValueError(
            f"Input source {input_info.input_source} does not exist. "
            f"Please check the path is correct."
        )


def validate_output_configs(output_configs: Sequence[schemas.OutputConfig]) -> None:
    for output_config in output_configs:
        base_validate_output_info(output_info=output_config.output_info)
        name = output_config.output_info.output_name

        match output_config.output_type_info:
            case TabularOutputTypeConfig(
                target_cat_columns,
                target_con_columns,
                _,
                _,
                _,
            ):
                output_source = output_config.output_info.output_source

                if output_source is None:
                    continue

                output_source_path = Path(output_source)

                expected_columns = list(target_cat_columns) + list(target_con_columns)
                validate_tabular_file(
                    file_to_check=output_source_path,
                    expected_columns=expected_columns,
                    name="Tabular output",
                )
            case ArrayOutputTypeConfig(_, _, loss, _):
                model_type = output_config.model_config.model_type
                if loss == "diffusion":
                    if model_type not in ("cnn",):
                        raise ValueError(
                            "Currently, diffusion loss is only supported for output "
                            "array model type 'cnn'. Please check the model type for "
                            f"output '{name}'."
                        )
            case ImageOutputTypeConfig(_, _, loss, _):
                model_type = output_config.model_config.model_type
                if loss == "diffusion":
                    if model_type not in ("cnn",):
                        raise ValueError(
                            "Currently, diffusion loss is only supported for output "
                            "image model type 'cnn'. Please check the model type for "
                            f"output '{name}'."
                        )


def base_validate_output_info(output_info: schemas.OutputInfoConfig) -> None:
    if output_info.output_source is None:
        return

    if not Path(output_info.output_source).exists():
        raise ValueError(
            f"Output source {output_info.output_source} does not exist. "
            f"Please check the path is correct."
        )


def validate_tabular_file(
    file_to_check: Path,
    expected_columns: Sequence[str],
    name: str,
) -> None:
    header = pd.read_csv(file_to_check, nrows=0).columns.tolist()
    if "ID" not in header:
        raise ValueError(
            f"{name} file {file_to_check} does not contain a column named 'ID'. "
            f"Please check the input file."
        )

    if not set(expected_columns).issubset(header):
        raise ValueError(
            f"{name} file {file_to_check} does not contain all expected columns: "
            f"{reprlib.repr(expected_columns)}. "
            f"Please check the input file."
        )

    series_ids = pd.read_csv(
        filepath_or_buffer=file_to_check,
        usecols=["ID"],
        engine="pyarrow",
        dtype_backend="pyarrow",
    ).squeeze()
    if not series_ids.is_unique:
        non_unique = series_ids[series_ids.duplicated()].tolist()
        raise ValueError(
            f"{name} file {file_to_check} contains non-unique"
            f" IDs: {reprlib.repr(non_unique)}. "
            f"Please check the input file."
        )


def validate_config_sync(
    input_configs: Sequence[schemas.InputConfig],
    output_configs: Sequence[schemas.OutputConfig],
) -> None:
    input_names = {input_config.input_info.input_name for input_config in input_configs}

    diff_out_configs = [
        i for i in output_configs if i.output_info.output_type in ("array", "image")
    ]
    for output_config in diff_out_configs:
        output_name = output_config.output_info.output_name
        output_type_info = output_config.output_type_info
        assert isinstance(
            output_type_info, (ArrayOutputTypeConfig, ImageOutputTypeConfig)
        )
        is_diffusion = output_type_info.loss == "diffusion"
        if is_diffusion and output_name not in input_names:
            raise ValueError(
                f"Diffusion loss is only supported when the corresponding input data "
                f"for the output '{output_name}' is provided. "
                "In practice, this means adding an input configuration with the "
                "same name and input_source as the output. "
                "This is needed to calculate the diffusion loss."
            )


def validate_tensor_broker_configs(
    input_configs: Sequence[schemas.InputConfig],
    fusion_configs: Sequence[schemas.FusionConfig],
    output_configs: Sequence[schemas.OutputConfig],
) -> None:
    all_tensor_broker_configs: list[schemas.TensorBrokerConfig] = []
    all_configs = list(input_configs) + list(fusion_configs) + list(output_configs)

    for config in all_configs:
        if (
            hasattr(config, "tensor_broker_config")
            and config.tensor_broker_config is not None
        ):
            all_tensor_broker_configs.append(config.tensor_broker_config)

    if not all_tensor_broker_configs:
        return

    all_message_names = []
    for broker_config in all_tensor_broker_configs:
        all_message_names.extend([msg.name for msg in broker_config.message_configs])

    if len(all_message_names) != len(set(all_message_names)):
        counts = {name: all_message_names.count(name) for name in all_message_names}
        raise ValueError(
            f"All tensor message names must be unique across all configs."
            f"Got counts: {counts}"
        )

    for broker_config in all_tensor_broker_configs:
        for msg in broker_config.message_configs:
            if (
                msg.cache_fusion_type == "cross-attention"
                and msg.projection_type != "sequence"
            ):
                raise ValueError(
                    f"Cross-attention is only allowed with 'sequence' projection. "
                    f"Message '{msg.name}' uses {msg.projection_type}."
                )

    cached_messages = set()
    used_messages = set()
    for broker_config in all_tensor_broker_configs:
        for msg in broker_config.message_configs:
            if msg.cache_tensor:
                cached_messages.add(msg.name)
            if msg.use_from_cache:
                used_messages.update(msg.use_from_cache)

    unused_caches = cached_messages - used_messages
    if unused_caches:
        raise ValueError(
            f"The following tensor messages are never used: {unused_caches}"
        )

    unused_needing_cache = used_messages - cached_messages
    if unused_needing_cache:
        raise ValueError(
            f"The following tensor messages are used but not cached: "
            f"{unused_needing_cache}"
        )
