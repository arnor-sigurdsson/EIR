from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from random import sample
from typing import TYPE_CHECKING, Literal, Union

import polars as pl
from aislib.misc_utils import ensure_path_exists
from lightning.fabric import Fabric
from torch.utils.data import DataLoader

from eir.data_load import label_setup
from eir.data_load.data_utils import consistent_nan_collate
from eir.experiment_io.experiment_io import (
    LoadedTrainExperiment,
    load_serialized_train_experiment,
)
from eir.interpretation.interpretation import tabular_attribution_analysis_wrapper
from eir.models.model_setup_modules.meta_setup import al_meta_model
from eir.predict_modules.predict_data import set_up_default_dataset
from eir.predict_modules.predict_input_setup import set_up_inputs_for_predict
from eir.predict_modules.predict_target_setup import (
    MergedPredictTargetLabels,
    get_target_labels_for_testing,
)
from eir.setup.config import Configs
from eir.setup.input_setup import al_input_objects_as_dict
from eir.setup.output_setup import al_output_objects_as_dict
from eir.train import check_dataset_and_batch_size_compatibility
from eir.train_utils.step_logic import Hooks
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.predict import PredictExperiment

logger = get_logger(name=__name__)


def compute_predict_attributions(
    run_folder: Path,
    loaded_train_experiment: "LoadedTrainExperiment",
    predict_config: "PredictExperiment",
) -> None:
    """
    The reason why we have the different sources to load the configs from is that
    when the source is 'train', we need to have the full configs to set up to access
    the various paths used for training (e.g. input_source), which has been stripped
    from the serialized configs. In the case of loading from 'predict', we use
    the already set up paths / objects, e.g. selecting the predict output objects,
    overloading the train experiment with the predict model, inputs, etc.
    """
    gc = predict_config.configs.global_config

    background_source = (
        predict_config.predict_specific_cl_args.attribution_background_source
    )

    configs_source: Literal["configs", "configs_stripped"] = "configs"
    if background_source == "predict":
        configs_source = "configs_stripped"

    loaded_train_experiment = load_serialized_train_experiment(
        run_folder=run_folder,
        device=gc.be.device,
        source_folder=configs_source,
    )

    if configs_source == "configs":
        output_objects = loaded_train_experiment.outputs
    elif configs_source == "configs_stripped":
        output_objects = predict_config.outputs
    else:
        raise ValueError(
            f"Invalid configs source '{configs_source}'. "
            f"Expected 'configs' or 'configs_stripped'."
        )

    background_source_config = get_background_source_config(
        background_source_in_predict_cl_args=background_source,
        train_configs=loaded_train_experiment.configs,
        predict_configs=predict_config.configs,
    )
    background_dataloader_base = _get_predict_background_loader(
        batch_size=gc.be.batch_size,
        num_attribution_background_samples=gc.aa.attribution_background_samples,
        outputs_as_dict=output_objects,
        configs=background_source_config,
        dataloader_workers=gc.be.dataloader_workers,
        loaded_hooks=loaded_train_experiment.hooks,
    )

    background_dataloader = predict_config.fabric.setup_dataloaders(
        background_dataloader_base
    )
    if isinstance(background_dataloader, list):
        raise ValueError("Got a list of dataloaders, but expected a single dataloader.")

    overloaded_train_experiment = _overload_train_experiment_for_predict_attributions(
        train_config=loaded_train_experiment,
        predict_config=predict_config,
    )

    attribution_output_folder_callable = partial(
        _get_predict_attribution_output_folder_target,
        predict_output_folder=Path(
            predict_config.predict_specific_cl_args.output_folder
        ),
    )

    tabular_attribution_analysis_wrapper(
        model=predict_config.model,
        experiment=overloaded_train_experiment,
        output_folder_target_callable=attribution_output_folder_callable,
        dataset_to_interpret=predict_config.test_dataset,
        background_loader=background_dataloader,
    )


def get_background_source_config(
    background_source_in_predict_cl_args: Literal["train", "predict"],
    train_configs: Configs,
    predict_configs: Configs,
) -> Configs:
    """
    TODO:   In the case of predict, make sure background and samples analysed are
            separated.
    """
    if background_source_in_predict_cl_args == "predict":
        logger.info(
            "Background for attribution analysis will be loaded from sources "
            "passed to predict.py."
        )
        return predict_configs

    if background_source_in_predict_cl_args == "train":
        logger.info(
            "Background for attribution analysis will be loaded from sources "
            "previously used for training run with name '%s'.",
            train_configs.gc.be.output_folder,
        )
        return train_configs

    raise ValueError("Invalid background source. Expected 'train' or 'predict'.")


@dataclass
class LoadedTrainExperimentMixedWithPredict(LoadedTrainExperiment):
    model: al_meta_model
    inputs: al_input_objects_as_dict
    fabric: Fabric


def _overload_train_experiment_for_predict_attributions(
    train_config: LoadedTrainExperiment,
    predict_config: "PredictExperiment",
) -> "LoadedTrainExperimentMixedWithPredict":
    """
    TODO:   Possibly set inputs=None as a field in LoadedTrainExperiment that then gets
            filled with test_inputs. When we do not need the weird monkey-patching here
            of the batch_prep_hooks, as the LoadedTrainExperiment will have the
            inputs attribute.
    """
    train_experiment_copy = copy(train_config)

    mixed_experiment_kwargs = train_experiment_copy.__dict__
    mixed_experiment_kwargs["model"] = predict_config.model
    mixed_experiment_kwargs["configs"] = predict_config.configs
    mixed_experiment_kwargs["inputs"] = predict_config.inputs
    mixed_experiment_kwargs["fabric"] = predict_config.fabric

    mixed_experiment = LoadedTrainExperimentMixedWithPredict(**mixed_experiment_kwargs)

    return mixed_experiment


def _get_predict_background_loader(
    batch_size: int,
    num_attribution_background_samples: int,
    dataloader_workers: int,
    configs: Configs,
    outputs_as_dict: al_output_objects_as_dict,
    loaded_hooks: Union["Hooks", None],
) -> DataLoader:
    """
    Note we have the target filtering there to ensure that the final
    dataset size matches that of sampled IDs. If not, there is a filtering step
    applied to the tabular inputs that come from set_up_inputs_from_predict,
    where the input hook `setup_tabular_input_for_testing` is called and filters
    on the IDs.

    As part of the basic processing of the tabular inputs, missing values have
    already been imputed. But, as setup_tabular_input_for_testing applies filtering,
    whereas other input loading funcs might not, we end up with the case where
    the joining of the tabular inputs and other modalities (done in the
    dataset setup) results in multiple NaNs for the tabular inputs,
    """

    background_ids_pool = label_setup.gather_all_ids_from_all_inputs(
        input_configs=configs.input_configs
    )

    target_labels = get_target_labels_for_testing(
        configs_overloaded_for_predict=configs,
        ids=background_ids_pool,
    )

    background_ids_sampled = _get_background_ids_sample(
        target_labels=target_labels,
        ids_inputs=background_ids_pool,
        num_attribution_background_samples=num_attribution_background_samples,
    )

    background_inputs_as_dict = set_up_inputs_for_predict(
        test_inputs_configs=configs.input_configs,
        ids=background_ids_sampled,
        hooks=loaded_hooks,
        output_folder=configs.gc.be.output_folder,
    )

    df_target = target_labels.predict_labels
    df_target_filtered = df_target.filter(pl.col("ID").is_in(background_ids_sampled))

    background_dataset = set_up_default_dataset(
        configs=configs,
        outputs_as_dict=outputs_as_dict,
        target_labels_df=df_target_filtered,
        inputs_as_dict=background_inputs_as_dict,
        missing_ids_per_output=target_labels.missing_ids_per_output,
    )

    assert len(background_dataset) <= num_attribution_background_samples

    check_dataset_and_batch_size_compatibility(
        dataset=background_dataset,
        batch_size=batch_size,
        name="Test attribution background",
    )
    background_loader = DataLoader(
        dataset=background_dataset,
        collate_fn=consistent_nan_collate,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_workers,
    )

    return background_loader


def _get_predict_attribution_output_folder_target(
    predict_output_folder: Path, output_name: str, column_name: str, input_name: str
) -> Path:
    attribution_output_folder = (
        predict_output_folder / output_name / column_name / "attributions" / input_name
    )
    ensure_path_exists(path=attribution_output_folder, is_folder=True)

    return attribution_output_folder


def _get_background_ids_sample(
    target_labels: MergedPredictTargetLabels,
    ids_inputs: Sequence[str],
    num_attribution_background_samples: int,
) -> list[str]:
    ids_target_set = set(target_labels.predict_labels["ID"])
    ids_inputs_set = set(ids_inputs)

    ids_common: list[str] = list(ids_target_set.intersection(ids_inputs_set))

    background_ids_sampled: list[str] = sample(
        population=ids_common,
        k=num_attribution_background_samples,
    )

    return background_ids_sampled
