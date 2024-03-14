import tempfile
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Dict, Generator, Iterable, Literal, Protocol, Sequence, Tuple

from eir.predict_modules.predict_config_validation import (
    validate_predict_configs_and_args,
)
from eir.predict_modules.predict_utils import (
    log_and_raise_missing_or_multiple_config_matching_general,
    log_and_raise_missing_or_multiple_tabular_output_matches,
)
from eir.setup import config, schemas
from eir.setup.config import Configs
from eir.setup.config_setup_modules.config_setup_utils import (
    get_yaml_to_dict_iterator,
    object_to_primitives,
    recursive_dict_replace,
)
from eir.setup.schema_modules.output_schemas_tabular import TabularOutputTypeConfig
from eir.utils.logging import get_logger

al_named_dict_configs = Dict[
    Literal["global_configs", "fusion_configs", "input_configs", "output_configs"],
    Iterable[Dict],
]


logger = get_logger(name=__name__)


def converge_train_and_predict_configs(
    train_configs: Configs, predict_cl_args: Namespace
) -> Configs:
    train_configs_copy = deepcopy(train_configs)

    named_dict_iterators = get_named_predict_dict_iterators(
        predict_cl_args=predict_cl_args
    )

    matched_dict_iterator = get_train_predict_matched_config_generator(
        train_configs=train_configs_copy, named_dict_iterators=named_dict_iterators
    )

    predict_configs_overloaded = overload_train_configs_for_predict(
        matched_dict_iterator=matched_dict_iterator
    )
    validate_predict_configs_and_args(
        predict_configs=predict_configs_overloaded, predict_cl_args=predict_cl_args
    )

    return predict_configs_overloaded


def get_named_predict_dict_iterators(
    predict_cl_args: Namespace,
) -> dict[str, tuple[dict, ...]]:
    target_keys = {
        "global_configs",
        "fusion_configs",
        "input_configs",
        "output_configs",
    }
    dict_of_generators = {}
    for key, value in predict_cl_args.__dict__.items():
        if key in target_keys:
            if not value:
                value = ()
            cur_gen = get_yaml_to_dict_iterator(yaml_config_files=value)
            dict_of_generators[key] = tuple(cur_gen)

    return dict_of_generators


def get_train_predict_matched_config_generator(
    train_configs: Configs,
    named_dict_iterators: dict[str, tuple[dict, ...]],
) -> Generator[Tuple[str, Dict, Dict], None, None]:
    train_keys = set(train_configs.__dict__.keys())

    single_configs = {
        "global_configs": "global_config",
        "fusion_configs": "fusion_config",
    }

    sequence_configs = {"input_configs", "output_configs"}

    for predict_argument_name, predict_dict_iter in named_dict_iterators.items():
        name_in_configs_object = single_configs.get(
            predict_argument_name, predict_argument_name
        )
        assert name_in_configs_object in train_keys

        if predict_dict_iter is None:
            predict_dict_iter = []

        # If not a sequence we can yield directly
        if predict_argument_name in single_configs.keys():
            train_config = getattr(train_configs, name_in_configs_object)
            train_config_as_dict = object_to_primitives(obj=train_config)

            predict_config_as_dict = config.combine_dicts(dicts=predict_dict_iter)

            yield (
                name_in_configs_object,
                train_config_as_dict,
                predict_config_as_dict,
            )

        # Otherwise we have to match the respective ones with each other
        elif predict_argument_name in sequence_configs:
            train_config_sequence = getattr(train_configs, name_in_configs_object)

            for cur_config in train_config_sequence:
                is_delayed = _predict_input_config_should_not_exist_yet(
                    predict_argument_name=predict_argument_name,
                    output_configs=train_configs.output_configs,
                    cur_config=cur_config,
                )

                if is_delayed:
                    continue

                matching_func = get_config_sequence_matching_func(
                    name=name_in_configs_object
                )
                predict_dict_match_from_iter = matching_func(
                    train_config=cur_config, predict_dict_iterator=predict_dict_iter
                )

                cur_train_config_as_dict = object_to_primitives(obj=cur_config)

                yield (
                    name_in_configs_object,
                    cur_train_config_as_dict,
                    predict_dict_match_from_iter,
                )


def _predict_input_config_should_not_exist_yet(
    predict_argument_name: str,
    output_configs: Sequence[schemas.OutputConfig],
    cur_config: schemas.InputConfig,
) -> bool:
    """
    There is strict matching functionality for input and output configs for
    e.g. tabular outputs, but for sequence outputs the input is automatically
    generated in the configuration setup, so we don't enforce that the input
    is being passed to the predict module as a yaml config.
    """
    is_delayed = False
    if predict_argument_name == "input_configs":
        matching_output_configs = _find_output_configs_matching_input(
            output_configs=output_configs,
            input_config=cur_config,
        )

        if matching_output_configs:
            assert len(matching_output_configs) == 1
            matching_output_config = matching_output_configs[0]
            if matching_output_config.output_info.output_type == "sequence":
                is_delayed = True

    return is_delayed


def _find_output_configs_matching_input(
    output_configs: Iterable[schemas.OutputConfig], input_config: schemas.InputConfig
) -> list[schemas.OutputConfig]:
    matching_output_configs = [
        i
        for i in output_configs
        if i.output_info.output_name == input_config.input_info.input_name
    ]
    assert len(matching_output_configs) <= 1

    return matching_output_configs


class InputSequenceMatchingFunction(Protocol):
    def __call__(
        self,
        train_config: schemas.InputConfig,
        predict_dict_iterator: Iterable[dict],
    ) -> dict: ...


class OutputSequenceMatchingFunction(Protocol):
    def __call__(
        self,
        train_config: schemas.OutputConfig,
        predict_dict_iterator: Iterable[dict],
    ) -> dict: ...


def get_config_sequence_matching_func(
    name: str,
) -> InputSequenceMatchingFunction | OutputSequenceMatchingFunction:
    assert name in ("input_configs", "output_configs")

    def _input_configs(
        train_config: schemas.InputConfig,
        predict_dict_iterator: Iterable[dict],
    ) -> dict:
        if not isinstance(train_config, schemas.InputConfig):
            raise TypeError("Expected InputConfig for _input_configs")

        matches = []
        predict_names_and_types = []

        train_input_info = train_config.input_info
        train_input_name = train_input_info.input_name
        train_input_type = train_input_info.input_type

        for predict_input_config_dict in predict_dict_iterator:
            predict_feature_extractor_info = predict_input_config_dict["input_info"]
            predict_input_name = predict_feature_extractor_info["input_name"]
            predict_input_type = predict_feature_extractor_info["input_type"]

            predict_names_and_types.append((predict_input_name, predict_input_type))

            cond_1 = predict_input_name == train_input_name
            cond_2 = predict_input_type == train_input_type

            if all((cond_1, cond_2)):
                matches.append(predict_input_config_dict)

        if len(matches) != 1:
            log_and_raise_missing_or_multiple_config_matching_general(
                train_name=train_input_name,
                train_type=train_input_type,
                matches=matches,
                predict_names_and_types=predict_names_and_types,
                name="input",
            )

        assert len(matches) == 1, matches
        return matches[0]

    def _output_configs(
        train_config: schemas.OutputConfig,
        predict_dict_iterator: Iterable[dict],
    ) -> dict:
        if not isinstance(train_config, schemas.OutputConfig):
            raise TypeError("Expected OutputConfig for _output_configs")

        match train_config.output_info.output_type:
            case "tabular":
                match = _check_matching_tabular_output_configs(
                    train_config=train_config,
                    predict_dict_iterator=predict_dict_iterator,
                )
            case "sequence":
                predict_dict_iterator = (
                    _get_maybe_patched_null_sequence_output_source_for_generation(
                        predict_dict_iterator=predict_dict_iterator
                    )
                )
                match = _check_matching_general_output_configs(
                    train_config=train_config,
                    predict_dict_iterator=predict_dict_iterator,
                )

            case "array":
                match = _check_matching_general_output_configs(
                    train_config=train_config,
                    predict_dict_iterator=predict_dict_iterator,
                )

            case _:
                raise NotImplementedError(
                    f"Output type '{train_config.output_info.output_type}' "
                    f"not implemented."
                )

        return match

    if name == "input_configs":
        return _input_configs
    return _output_configs


def _get_maybe_patched_null_sequence_output_source_for_generation(
    predict_dict_iterator: Iterable[dict],
) -> Iterable[dict]:
    for output_config in predict_dict_iterator:
        if output_config["output_info"]["output_type"] == "sequence":
            if output_config["output_info"]["output_source"] is None:
                output_name = output_config["output_info"]["output_name"]

                logger.info(
                    f"Got 'None' as output source for sequence output: {output_name}. "
                    f"Patching output source to a temporary directory "
                    f"for generation."
                )

                tmp_dir = tempfile.mkdtemp()
                tmp_file = Path(tmp_dir) / f"{output_name}_seed_file.txt"
                tmp_file.touch()

                output_config["output_info"]["output_source"] = str(tmp_dir)

        yield output_config


def _check_matching_general_output_configs(
    train_config: schemas.OutputConfig, predict_dict_iterator: Iterable[dict]
) -> dict:
    train_output_info = train_config.output_info
    train_output_name = train_output_info.output_name
    train_output_type = train_output_info.output_type

    matches = []
    predict_names_and_types = []

    for predict_output_config_dict in predict_dict_iterator:
        predict_output_info = predict_output_config_dict["output_info"]
        predict_output_name = predict_output_info["output_name"]
        predict_output_type = predict_output_info["output_type"]

        predict_names_and_types.append((predict_output_name, predict_output_type))

        cond_1 = predict_output_name == train_output_name
        cond_2 = predict_output_type == train_output_type

        if all((cond_1, cond_2)):
            matches.append(predict_output_config_dict)

    if len(matches) != 1:
        log_and_raise_missing_or_multiple_config_matching_general(
            train_name=train_output_name,
            train_type=train_output_type,
            matches=matches,
            predict_names_and_types=predict_names_and_types,
            name="output",
        )

    assert len(matches) == 1, matches
    return matches[0]


def _check_matching_tabular_output_configs(
    train_config: schemas.OutputConfig,
    predict_dict_iterator: Iterable[dict],
):
    matches = []
    predict_names_and_types = []

    output_name = train_config.output_info.output_name
    output_type = train_config.output_info.output_type

    output_type_info = train_config.output_type_info
    assert isinstance(output_type_info, TabularOutputTypeConfig)

    train_cat_columns = output_type_info.target_cat_columns
    train_con_columns = output_type_info.target_con_columns

    for predict_output_config_dict in predict_dict_iterator:
        predict_output_info = predict_output_config_dict["output_info"]
        predict_output_name = predict_output_info["output_name"]
        predict_output_type = predict_output_info["output_type"]

        cat_cols = predict_output_config_dict.get("output_type_info", {}).get(
            "target_cat_columns", []
        )
        con_cols = predict_output_config_dict.get("output_type_info", {}).get(
            "target_con_columns", []
        )

        predict_names_and_types.append(
            (predict_output_name, predict_output_type, cat_cols, con_cols)
        )

        cond_1 = predict_output_name == output_name
        cond_2 = predict_output_type == output_type

        cond_3 = train_cat_columns == cat_cols
        cond_4 = train_con_columns == con_cols

        if all((cond_1, cond_2, cond_3, cond_4)):
            matches.append(predict_output_config_dict)

    if len(matches) != 1:
        log_and_raise_missing_or_multiple_tabular_output_matches(
            train_name=output_name,
            train_type=output_type,
            train_cat_columns=train_cat_columns,
            train_con_columns=train_con_columns,
            matches=matches,
            predict_names_and_types=predict_names_and_types,
        )

    assert len(matches) == 1, matches
    return matches[0]


def overload_train_configs_for_predict(
    matched_dict_iterator: Generator[Tuple[str, Dict, Dict], None, None],
) -> Configs:
    main_overloaded_kwargs: dict = {}

    for name, train_config_dict, predict_config_dict_to_inject in matched_dict_iterator:
        _maybe_warn_about_output_folder_overload_from_predict(
            name=name,
            predict_config_dict_to_inject=predict_config_dict_to_inject,
            train_config_dict=train_config_dict,
        )

        overloaded_dict = recursive_dict_replace(
            dict_=train_config_dict, dict_to_inject=predict_config_dict_to_inject
        )
        if name in ("global_config", "fusion_config"):
            main_overloaded_kwargs[name] = overloaded_dict
        elif name in ("input_configs", "output_configs"):
            main_overloaded_kwargs.setdefault(name, [])
            main_overloaded_kwargs.get(name, []).append(overloaded_dict)

    global_configs: list[dict] = [main_overloaded_kwargs.get("global_config", {})]
    global_config_overloaded = config.get_global_config(
        global_configs=global_configs,
    )

    if main_overloaded_kwargs.get("input_configs"):
        input_configs_overloaded = config.get_input_configs(
            input_configs=main_overloaded_kwargs.get("input_configs", [])
        )
    else:
        input_configs_overloaded = []

    fusion_config_overloaded = config.load_fusion_configs(
        [main_overloaded_kwargs.get("fusion_config", {})]
    )

    output_configs_overloaded = config.load_output_configs(
        output_configs=main_overloaded_kwargs.get("output_configs", []),
    )

    train_configs_overloaded = config.Configs(
        global_config=global_config_overloaded,
        input_configs=input_configs_overloaded,
        fusion_config=fusion_config_overloaded,
        output_configs=output_configs_overloaded,
    )

    train_configs_overloaded_with_seq_outputs = (
        config.get_configs_object_with_seq_output_configs(
            configs=train_configs_overloaded,
        )
    )

    return train_configs_overloaded_with_seq_outputs


def _maybe_warn_about_output_folder_overload_from_predict(
    name: str, predict_config_dict_to_inject: Dict, train_config_dict: Dict
) -> None:
    if name == "global_config" and "output_folder" in predict_config_dict_to_inject:
        output_folder_from_predict = predict_config_dict_to_inject["output_folder"]
        output_folder_from_train = train_config_dict["output_folder"]

        if output_folder_from_predict == output_folder_from_train:
            return

        logger.warning(
            "output_folder in '%s' will be replaced with '%s' from given prediction "
            "output configuration ('%s'). If this is intentional, you can ignore this "
            "message but most likely this is a mistake and will cause an error. "
            "Resolution: Remove 'output_folder' from "
            "the global output config as it will "
            "be automatically looked up based on the experiment.",
            train_config_dict,
            output_folder_from_predict,
            predict_config_dict_to_inject,
        )
