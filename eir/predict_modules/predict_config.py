from argparse import Namespace
from copy import deepcopy
from typing import Generator, Tuple, Dict, Literal, Callable, Iterable

from aislib.misc_utils import get_logger

from eir.predict_modules.predict_utils import (
    log_and_raise_missing_or_multiple_input_matches,
    log_and_raise_missing_or_multiple_output_matches,
)
from eir.setup import config, schemas
from eir.setup.config import Configs, object_to_primitives, recursive_dict_replace

al_named_dict_configs = Dict[
    Literal["global_configs", "fusion_configs", "input_configs", "output_configs"],
    Iterable[Dict],
]


logger = get_logger(name=__name__)


def converge_train_and_predict_configs(
    train_configs: Configs, predict_cl_args: Namespace
) -> Configs:

    train_configs_copy = deepcopy(train_configs)

    named_dict_iterators = get_named_pred_dict_iterators(
        predict_cl_args=predict_cl_args
    )

    matched_dict_iterator = get_train_predict_matched_config_generator(
        train_configs=train_configs_copy, named_dict_iterators=named_dict_iterators
    )

    predict_configs_overloaded = overload_train_configs_for_predict(
        matched_dict_iterator=matched_dict_iterator
    )

    return predict_configs_overloaded


def get_named_pred_dict_iterators(
    predict_cl_args: Namespace,
) -> al_named_dict_configs:
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
            cur_gen = config.get_yaml_to_dict_iterator(yaml_config_files=value)
            dict_of_generators[key] = tuple(cur_gen)
    return dict_of_generators


def get_train_predict_matched_config_generator(
    train_configs: Configs,
    named_dict_iterators: al_named_dict_configs,
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


def get_config_sequence_matching_func(
    name: Literal["input_configs", "output_configs"]
) -> Callable:
    assert name in ("input_configs", "output_configs")

    def _input_configs(
        train_config: schemas.InputConfig,
        predict_dict_iterator: Iterable[Dict],
    ):
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
            log_and_raise_missing_or_multiple_input_matches(
                train_name=train_input_name,
                train_type=train_input_type,
                matches=matches,
                predict_names_and_types=predict_names_and_types,
            )

        assert len(matches) == 1, matches
        return matches[0]

    def _output_configs(
        train_config: schemas.OutputConfig,
        predict_dict_iterator: Iterable[Dict],
    ):

        matches = []
        predict_names_and_types = []

        output_name = train_config.output_info.output_name
        output_type = train_config.output_info.output_type

        train_cat_columns = train_config.output_type_info.target_cat_columns
        train_con_columns = train_config.output_type_info.target_con_columns

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
            log_and_raise_missing_or_multiple_output_matches(
                train_name=output_name,
                train_type=output_type,
                train_cat_columns=train_cat_columns,
                train_con_columns=train_con_columns,
                matches=matches,
                predict_names_and_types=predict_names_and_types,
            )

        assert len(matches) == 1, matches
        return matches[0]

    if name == "input_configs":
        return _input_configs
    return _output_configs


def overload_train_configs_for_predict(
    matched_dict_iterator: Generator[Tuple[str, Dict, Dict], None, None],
) -> Configs:

    main_overloaded_kwargs = {}

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
            main_overloaded_kwargs.get(name).append(overloaded_dict)

    global_config_overloaded = config.get_global_config(
        global_configs=[main_overloaded_kwargs.get("global_config")]
    )
    input_configs_overloaded = config.get_input_configs(
        input_configs=main_overloaded_kwargs.get("input_configs")
    )
    fusion_config_overloaded = config.load_fusion_configs(
        [main_overloaded_kwargs.get("fusion_config")]
    )

    tabular_output_setup = config.DynamicOutputSetup(
        output_types_schema_map=config.get_outputs_types_schema_map(),
        output_module_config_class_getter=config.get_output_module_config_class,
        output_module_init_class_map=config.get_output_config_type_init_callable_map(),
    )

    output_configs_overloaded = config.load_output_configs(
        output_configs=main_overloaded_kwargs.get("output_configs"),
        dynamic_output_setup=tabular_output_setup,
    )

    train_configs_overloaded = config.Configs(
        global_config=global_config_overloaded,
        input_configs=input_configs_overloaded,
        fusion_config=fusion_config_overloaded,
        output_configs=output_configs_overloaded,
    )

    return train_configs_overloaded


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
