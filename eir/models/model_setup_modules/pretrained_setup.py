from pathlib import Path
from typing import TYPE_CHECKING, Sequence, Tuple, Union

from eir.experiment_io.experiment_io import (
    get_run_folder_from_model_path,
    load_serialized_train_experiment,
)
from eir.models.model_setup_modules.meta_setup import (
    MetaClassGetterCallable,
    get_default_meta_class,
    get_meta_model_class_and_kwargs_from_configs,
)
from eir.models.model_setup_modules.model_io import load_model
from eir.setup import schemas
from eir.setup.input_setup import al_input_objects_as_dict
from eir.setup.output_setup import al_output_objects_as_dict
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.models.meta.meta_utils import al_input_modules

logger = get_logger(name=__name__)


def overload_meta_model_feature_extractors_with_pretrained(
    input_modules: "al_input_modules",
    inputs_as_dict: al_input_objects_as_dict,
    outputs_as_dict: "al_output_objects_as_dict",
    meta_class_getter: MetaClassGetterCallable = get_default_meta_class,
) -> "al_input_modules":
    """
    Note that `inputs_as_dict` here are coming from the current experiment, arguably
    it would be more robust / better to have them loaded from the pretrained experiment,
    but then we have to setup things from there such as hooks, valid_ids, train_ids,
    etc.

    For now, we will enforce that the feature extractor architecture that is set-up
    and then uses pre-trained weights from a previous experiment must match that of
    the feature extractor that did the pre-training. Simply put, we must ensure
    that all input setup parameters that have to do with architecture match exactly
    between the (a) pretrained input config and (b) the input config loading the
    pretrained model.

    The strict parameter being set to false is due to the fact that we can have e.g.
    different fusion modules in the current model vs the pretrained model, which should
    raise an error e.g. if continuing training from a checkpoint. However in this case,
    it acceptable to have different fusion modules, as long as the input modules are
    matching.
    """

    any_pretrained = any(
        i.input_config.pretrained_config for i in inputs_as_dict.values()
    )
    if not any_pretrained:
        return input_modules

    input_configs = tuple(i.input_config for i in inputs_as_dict.values())
    replace_pattern = _build_all_replacements_tuples_for_loading_pretrained_module(
        input_configs=input_configs
    )
    for input_name, input_object in inputs_as_dict.items():
        input_config = input_object.input_config

        pretrained_config = input_config.pretrained_config
        if not pretrained_config:
            continue

        load_model_path = Path(pretrained_config.model_path)
        load_run_folder = get_run_folder_from_model_path(
            model_path=str(load_model_path)
        )
        load_experiment = load_serialized_train_experiment(run_folder=load_run_folder)
        load_configs = load_experiment.configs

        func = get_meta_model_class_and_kwargs_from_configs
        meta_model_class, meta_model_kwargs = func(
            global_config=load_configs.global_config,
            fusion_config=load_configs.fusion_config,
            inputs_as_dict=inputs_as_dict,
            outputs_as_dict=outputs_as_dict,
            meta_class_getter=meta_class_getter,
            strict=False,
        )

        pretrained_name = pretrained_config.load_module_name
        loaded_and_renamed_meta_model = load_model(
            model_path=load_model_path,
            model_class=meta_model_class,
            model_init_kwargs=meta_model_kwargs,
            device="cpu",
            test_mode=False,
            state_dict_key_rename=replace_pattern,
            state_dict_keys_to_keep=(pretrained_name,),
        )
        loaded_and_renamed_fusion_extractors = (
            loaded_and_renamed_meta_model.input_modules
        )

        module_name_to_load = pretrained_config.load_module_name
        module_to_overload = loaded_and_renamed_fusion_extractors[input_name]

        logger.info(
            "Replacing '%s' in current model with '%s' from %s.",
            input_name,
            module_name_to_load,
            load_model_path,
        )

        input_modules[input_name] = module_to_overload

    return input_modules


def _build_all_replacements_tuples_for_loading_pretrained_module(
    input_configs: Sequence[schemas.InputConfig],
) -> Sequence[Tuple[str, str]]:
    replacement_patterns = []
    for input_config in input_configs:
        if input_config.pretrained_config:
            cur_replacement = _build_replace_tuple_when_loading_pretrained_module(
                load_module_name=input_config.pretrained_config.load_module_name,
                current_input_name=input_config.input_info.input_name,
            )
            if cur_replacement:
                replacement_patterns.append(cur_replacement)

    return replacement_patterns


def _build_replace_tuple_when_loading_pretrained_module(
    load_module_name: str, current_input_name: str
) -> Union[None, Tuple[str, str]]:
    if load_module_name == current_input_name:
        return None

    load_module_name_parsed = f"modules_to_fuse.{load_module_name}."
    current_input_name_parsed = f"modules_to_fuse.{current_input_name}."

    replace_pattern = (load_module_name_parsed, current_input_name_parsed)

    return replace_pattern
