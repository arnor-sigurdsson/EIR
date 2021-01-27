import argparse
import json
from argparse import Namespace
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Dict, Sequence, Callable, Iterable, Tuple, Type, Any

import dill
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from aislib.misc_utils import get_logger
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader

import snp_pred.visualization.visualization_funcs as vf
from snp_pred.configuration import append_data_source_prefixes
from snp_pred.data_load import datasets, label_setup
from snp_pred.data_load.data_utils import get_target_columns_generator
from snp_pred.data_load.datasets import (
    al_datasets,
)
from snp_pred.data_load.label_setup import (
    al_label_dict,
    al_label_transformers_object,
    al_target_columns,
    al_label_transformers,
    al_all_column_ops,
    gather_ids_from_tabular_file,
    transform_label_df,
    TabularFileInfo,
)
from snp_pred.models.model_training_utils import gather_pred_outputs_from_dloader
from snp_pred.models.tabular.tabular import al_emb_lookup_dict
from snp_pred.train import (
    get_train_config_serialization_path,
    prepare_base_batch_default,
    Hooks,
    get_tabular_target_label_data,
    get_tabular_inputs_label_data,
    get_fusion_class_from_cl_args,
    get_fusion_kwargs_from_cl_args,
    al_num_outputs_per_target,
    DataDimensions,
)
from snp_pred.train_utils.evaluation import PerformancePlotConfig
from snp_pred.train_utils.utils import get_run_folder

torch.manual_seed(0)
np.random.seed(0)

logger = get_logger(name=__name__, tqdm_compatible=True)


def main(predict_cl_args: argparse.Namespace, predict_config: "PredictConfig") -> None:

    predict(predict_config=predict_config, predict_cl_args=predict_cl_args)


def predict(
    predict_config: "PredictConfig",
    predict_cl_args: Namespace,
) -> None:

    all_preds, all_labels, all_ids = gather_pred_outputs_from_dloader(
        data_loader=predict_config.test_dataloader,
        batch_prep_hook=predict_config.hooks.predict_stages.base_prepare_batch,
        batch_prep_hook_kwargs={"predict_config": predict_config},
        model=predict_config.model,
        with_labels=predict_cl_args.evaluate,
    )

    target_columns_gen = get_target_columns_generator(
        target_columns=predict_config.test_dataset.target_columns
    )

    for target_column_type, target_column in target_columns_gen:

        target_preds = all_preds[target_column]

        cur_target_transformer = predict_config.target_transformers[target_column]
        preds_sm = F.softmax(input=target_preds, dim=1).cpu().numpy()

        classes = _get_target_classnames(
            transformer=cur_target_transformer, target_column=target_column
        )

        output_folder = Path(predict_cl_args.output_folder)
        _save_predictions(
            preds=preds_sm,
            test_dataset=predict_config.test_dataset,
            classes=classes,
            outfolder=output_folder,
        )

        if predict_cl_args.evaluate:
            cur_labels = all_labels[target_column].cpu().numpy()

            plot_config = PerformancePlotConfig(
                val_outputs=preds_sm,
                val_labels=cur_labels,
                val_ids=all_ids,
                iteration=0,
                column_name=target_column,
                column_type=target_column_type,
                target_transformer=cur_target_transformer,
                output_folder=output_folder,
            )

            vf.gen_eval_graphs(plot_config=plot_config)


@dataclass
class PredictConfig:
    train_cl_args_overloaded: Namespace
    test_dataset: datasets.DiskDataset
    target_columns: al_target_columns
    target_transformers: al_label_transformers
    test_dataloader: DataLoader
    model: nn.Module
    hooks: "PredictHooks"


@dataclass
class PredictHooks:

    predict_stages: "PredictHookStages"
    custom_column_label_parsing_ops: al_all_column_ops = None


@dataclass
class PredictHookStages:

    al_hook = Callable[..., Dict]
    al_hooks = [Iterable[al_hook]]

    base_prepare_batch: al_hooks
    model_forward: al_hooks


@dataclass
class PredictLabels:
    label_dict: al_label_dict
    transformers: al_label_transformers


def get_default_predict_config(
    run_folder: Path,
    predict_cl_args: Namespace,
) -> PredictConfig:

    train_config = _load_serialized_train_config(run_folder=run_folder)

    train_cl_args_overloaded, predict_cl_args = _converge_train_and_predict_cl_args(
        train_cl_args=train_config.cl_args, predict_cl_args=predict_cl_args
    )

    default_train_hooks = train_config.hooks

    target_labels, tabular_input_labels = None, None
    if predict_cl_args.evaluate:
        test_ids = gather_ids_from_tabular_file(
            file_path=train_cl_args_overloaded.label_file
        )

        label_ops = default_train_hooks.custom_column_label_parsing_ops
        target_labels, tabular_input_labels = get_target_and_extra_labels_for_predict(
            train_cl_args_overloaded=train_cl_args_overloaded,
            custom_column_label_parsing_ops=label_ops,
            ids=test_ids,
        )

    test_dataset = _set_up_default_test_dataset(
        train_cl_args_overloaded=train_cl_args_overloaded,
        data_dimensions=train_config.data_dimensions,
        test_labels_dict=target_labels.label_dict,
        tabular_inputs_labels_dict=tabular_input_labels.label_dict,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=train_cl_args_overloaded.batch_size,
        shuffle=False,
        num_workers=train_cl_args_overloaded.dataloader_workers,
    )

    func = _get_fusion_model_class_and_kwargs_from_cl_args
    fusion_model_class, fusion_model_kwargs = func(
        train_cl_args=train_cl_args_overloaded,
        num_outputs_per_target=train_config.num_outputs_per_target,
        tabular_input_transformers=tabular_input_labels.transformers,
        omics_data_dimensions=train_config.data_dimensions,
    )

    model = _load_model(
        model_path=Path(predict_cl_args.model_path),
        model_class=fusion_model_class,
        model_init_kwargs=fusion_model_kwargs,
        device=train_cl_args_overloaded.device,
    )
    assert not model.training

    default_predict_hooks = _get_default_predict_hooks(train_hooks=default_train_hooks)
    test_config = PredictConfig(
        train_cl_args_overloaded=train_cl_args_overloaded,
        test_dataset=test_dataset,
        target_columns=train_config.target_columns,
        target_transformers=train_config.target_transformers,
        test_dataloader=test_dataloader,
        model=model,
        hooks=default_predict_hooks,
    )
    return test_config


def _load_serialized_train_config(run_folder: Path):
    train_config_path = get_train_config_serialization_path(run_folder=run_folder)
    with open(train_config_path, "rb") as infile:
        train_config = dill.load(file=infile)

    return train_config


def get_target_and_extra_labels_for_predict(
    train_cl_args_overloaded: Namespace,
    custom_column_label_parsing_ops: al_all_column_ops,
    ids: Sequence[str],
) -> Tuple[PredictLabels, PredictLabels]:
    """
    NOTE:   This can be extended to more tabular data, including other files, if we
            update the parameters slightly.
    """

    target_info = get_tabular_target_label_data(cl_args=train_cl_args_overloaded)

    target_labels = get_labels_for_predict(
        run_name=train_cl_args_overloaded.run_name,
        tabular_file_info=target_info,
        custom_column_label_parsing_ops=custom_column_label_parsing_ops,
        ids=ids,
    )

    tabular_input_info = get_tabular_inputs_label_data(cl_args=train_cl_args_overloaded)

    if tabular_input_info.con_columns or tabular_input_info.cat_columns:
        tabular_input_labels = get_labels_for_predict(
            run_name=train_cl_args_overloaded.run_name,
            tabular_file_info=tabular_input_info,
            custom_column_label_parsing_ops=custom_column_label_parsing_ops,
            ids=ids,
        )
    else:
        tabular_input_labels = PredictLabels(label_dict={}, transformers={})

    return target_labels, tabular_input_labels


def get_labels_for_predict(
    run_name: str,
    tabular_file_info: TabularFileInfo,
    custom_column_label_parsing_ops: al_all_column_ops,
    ids: Sequence[str],
) -> PredictLabels:

    all_columns = list(tabular_file_info.cat_columns) + list(
        tabular_file_info.con_columns
    )

    if not all_columns:
        raise ValueError(f"No columns specified in {tabular_file_info}.")

    transformers = _load_transformers(
        run_name=run_name,
        transformers_to_load=all_columns,
    )

    df_labels = _load_labels_for_predict(
        tabular_info=tabular_file_info,
        ids_to_keep=ids,
        custom_label_ops=custom_column_label_parsing_ops,
    )
    labels_dict = parse_labels_for_testing(
        tabular_info=tabular_file_info,
        df_labels_test=df_labels,
        label_transformers=transformers,
    )

    labels = PredictLabels(label_dict=labels_dict, transformers=transformers)

    return labels


def _get_default_predict_hooks(train_hooks: "Hooks") -> PredictHooks:
    stages = PredictHookStages(
        base_prepare_batch=[_hook_default_predict_prepare_batch],
        model_forward=[train_hooks.step_func_hooks.model_forward],
    )
    predict_hooks = PredictHooks(
        predict_stages=stages,
        custom_column_label_parsing_ops=train_hooks.custom_column_label_parsing_ops,
    )

    return predict_hooks


def _hook_default_predict_prepare_batch(
    predict_config: "PredictConfig",
    loader_batch,
    *args,
    **kwargs,
):
    batch = prepare_base_batch_default(
        loader_batch=loader_batch,
        cl_args=predict_config.train_cl_args_overloaded,
        target_columns=predict_config.target_columns,
        model=predict_config.model,
    )

    state_updates = {"batch": batch}

    return state_updates


def _get_target_classnames(
    transformer: al_label_transformers_object, target_column: str
):
    if isinstance(transformer, LabelEncoder):
        return transformer.classes_
    return target_column


def _load_cl_args_config(cl_args_config_path: Path) -> Namespace:
    with open(str(cl_args_config_path), "r") as infile:
        loaded_cl_args = json.load(infile)

    return Namespace(**loaded_cl_args)


def _get_fusion_model_class_and_kwargs_from_cl_args(
    train_cl_args: Namespace,
    num_outputs_per_target: al_num_outputs_per_target,
    tabular_input_transformers: al_label_transformers,
    omics_data_dimensions: Dict[str, DataDimensions],
) -> Tuple[Type[nn.Module], Dict[str, Any]]:

    fusion_model_class = get_fusion_class_from_cl_args(
        fusion_model_type=train_cl_args.fusion_model_type
    )

    fusion_model_kwargs = get_fusion_kwargs_from_cl_args(
        cl_args=train_cl_args,
        omics_data_dimensions=omics_data_dimensions,
        num_outputs_per_target=num_outputs_per_target,
        tabular_label_transformers=tabular_input_transformers,
    )

    return fusion_model_class, fusion_model_kwargs


def _load_model(
    model_path: Path,
    model_class: Type[nn.Module],
    model_init_kwargs: Dict,
    device: str,
) -> torch.nn.Module:

    model = model_class(**model_init_kwargs)

    model = _load_model_weights(
        model=model, model_state_dict_path=model_path, device=device
    )

    model.eval()

    return model


def _load_model_weights(
    model: nn.Module, model_state_dict_path: Path, device: str
) -> nn.Module:
    device_for_load = torch.device(device)
    model.load_state_dict(
        state_dict=torch.load(model_state_dict_path, map_location=device_for_load)
    )
    model = model.to(device=device_for_load)

    return model


def _load_saved_embeddings_dict(
    embed_columns: Union[None, List[str]], run_name: str
) -> Union[None, al_emb_lookup_dict]:
    embeddings_dict = None

    run_folder = get_run_folder(run_name)

    if embed_columns:
        model_embeddings_path = run_folder / "extra_inputs" / "embeddings.save"
        embeddings_dict = joblib.load(model_embeddings_path)

    return embeddings_dict


def _converge_train_and_predict_cl_args(
    train_cl_args: Namespace, predict_cl_args: Namespace
) -> Tuple[Namespace, Namespace]:

    train_cl_args_copy = copy(train_cl_args)

    train_cl_args_overloaded = _overload_train_cl_args_for_predict(
        train_cl_args=train_cl_args_copy, predict_cl_args=predict_cl_args
    )
    train_cl_args_overloaded_prefix_renamed = append_data_source_prefixes(
        cl_args=train_cl_args_overloaded
    )

    train_keys = train_cl_args_overloaded_prefix_renamed.__dict__.keys()
    predict_cl_args_filtered = _remove_keys_from_namespace(
        namespace=predict_cl_args, keys_to_remove=train_keys
    )

    return train_cl_args_overloaded_prefix_renamed, predict_cl_args_filtered


def _overload_train_cl_args_for_predict(
    train_cl_args: Namespace, predict_cl_args: Namespace
) -> Namespace:

    train_overloaded = copy(train_cl_args)
    train_keys = set(train_overloaded.__dict__.keys())

    predict_arg_iter = predict_cl_args.__dict__.items()
    for predict_argument_name, predict_argument_value in predict_arg_iter:

        if predict_argument_name in train_keys:
            train_overloaded.__setattr__(predict_argument_name, predict_argument_value)

    return train_overloaded


def _remove_keys_from_namespace(
    namespace: Namespace, keys_to_remove: Iterable[str]
) -> Namespace:
    new_namespace_kwargs = {}

    keys_to_remove_set = set(keys_to_remove)

    for key_name, key_value in namespace.__dict__.items():
        if key_name not in keys_to_remove_set:
            new_namespace_kwargs[key_name] = key_value

    filtered_namespace = Namespace(**new_namespace_kwargs)

    return filtered_namespace


def _load_labels_for_predict(
    tabular_info: TabularFileInfo,
    ids_to_keep: Sequence[str],
    custom_label_ops: al_all_column_ops = None,
) -> pd.DataFrame:

    parse_wrapper = label_setup.get_label_parsing_wrapper(
        label_parsing_chunk_size=tabular_info.parsing_chunk_size
    )
    df_labels_test = parse_wrapper(
        label_file_tabular_info=tabular_info,
        ids_to_keep=ids_to_keep,
        custom_label_ops=custom_label_ops,
    )

    return df_labels_test


def parse_labels_for_testing(
    tabular_info: TabularFileInfo,
    df_labels_test: pd.DataFrame,
    label_transformers: al_label_transformers,
) -> al_label_dict:

    con_transformers = {
        k: v for k, v in label_transformers.items() if k in tabular_info.con_columns
    }
    train_con_column_means = _prep_missing_con_dict(con_transformers=con_transformers)

    df_labels_test = label_setup.handle_missing_label_values_in_df(
        df=df_labels_test,
        cat_label_columns=tabular_info.cat_columns,
        con_label_columns=tabular_info.con_columns,
        con_manual_values=train_con_column_means,
        name="test_df",
    )

    df_labels_test_transformed = transform_label_df(
        df_labels=df_labels_test, label_transformers=label_transformers
    )

    test_labels_dict = df_labels_test_transformed.to_dict("index")

    return test_labels_dict


def _prep_missing_con_dict(con_transformers: al_label_transformers) -> Dict[str, float]:

    train_means = {
        column: transformer.mean_[0] for column, transformer in con_transformers.items()
    }

    return train_means


def _set_up_default_test_dataset(
    train_cl_args_overloaded: Namespace,
    data_dimensions: Dict[str, "DataDimensions"],
    test_labels_dict: Union[None, al_label_dict],
    tabular_inputs_labels_dict: Union[None, al_label_dict],
) -> al_datasets:
    """
    :param train_cl_args_overloaded: Training CL arguments with the data_source
    replaced with testing one.
    :param test_labels_dict: None if we are predicting on unknown data,
    otherwise a dictionary of labels (if evaluating on test set).
    :return: Dataset instance to be used for loading test samples.
    """

    test_dataset_kwargs = datasets.construct_default_dataset_kwargs_from_cl_args(
        cl_args=train_cl_args_overloaded,
        target_labels_dict=test_labels_dict,
        data_dimensions=data_dimensions,
        tabular_labels_dict=tabular_inputs_labels_dict,
        na_augment=False,
    )

    test_dataset = datasets.DiskDataset(**test_dataset_kwargs)

    return test_dataset


def _load_transformers(
    run_name: str, transformers_to_load: Union[Sequence[str], None]
) -> al_label_transformers:

    run_folder = get_run_folder(run_name=run_name)
    all_transformers = (i.stem for i in (run_folder / "transformers").iterdir())

    iterable = transformers_to_load if transformers_to_load else all_transformers

    label_transformers = {}
    for transformer_name in iterable:
        target_transformer_path = label_setup.get_transformer_path(
            run_path=run_folder, transformer_name=transformer_name
        )
        target_transformer_object = joblib.load(filename=target_transformer_path)
        label_transformers[transformer_name] = target_transformer_object

    return label_transformers


def _save_predictions(
    preds: torch.Tensor, test_dataset: al_datasets, classes: List[str], outfolder: Path
) -> None:
    test_ids = [i.sample_id for i in test_dataset.samples]
    df_preds = pd.DataFrame(data=preds, index=test_ids, columns=classes)
    df_preds.index.name = "ID"

    df_preds.to_csv(outfolder / "predictions.csv")


def get_predict_cl_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model to use for predictions.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=64, help="size of the batches"
    )

    parser.add_argument(
        "--label_file", type=str, help="Which file to load labels from."
    )

    parser.add_argument(
        "--omics_sources",
        type=str,
        nargs="*",
        help="Which one-hot omics sources to load samples from for predicting. Can "
        "either be (a) a folder in which files will be gathered from the folder "
        "recursively or (b) a simple text file with each line having a path for "
        "a sample array",
    )

    parser.add_argument(
        "--omics_names",
        type=str,
        nargs="*",
        help="Names for the omics sources passed in the --omics_sources argument.",
    )

    parser.add_argument("--evaluate", dest="evaluate", action="store_true")
    parser.set_defaults(evaluate=False)

    parser.add_argument(
        "--output_folder", type=str, help="Where to save prediction results."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["gpu", "cpu"],
        help="Which device (CPU or GPU) to run" "the inference on.",
    )

    parser.add_argument(
        "--gpu_num",
        type=str,
        default="0",
        help="Which GPU to run (according to CUDA) if running with flat "
        "'--device gpu'.",
    )

    parser.add_argument(
        "--dataloader_workers",
        type=int,
        default=0,
        help="Number of workers for dataloader.",
    )

    cl_args = parser.parse_args()

    device = _parse_predict_device_cl_argument(cl_args=cl_args)
    cl_args.device = device

    return cl_args


def _parse_predict_device_cl_argument(cl_args: argparse.Namespace) -> str:
    if cl_args.device == "gpu":
        device = "cuda:" + cl_args.gpu_num if torch.cuda.is_available() else "cpu"
        if cl_args.device == "cpu":
            logger.warning(
                "Device CL input was 'gpu' with gpu_num '%s', "
                "but number of available GPUs is %d. Reverting to CPU for "
                "now.",
                cl_args.gpu_num,
                torch.cuda.device_count(),
            )
    else:
        device = "cpu"

    return device


if __name__ == "__main__":

    predict_cl_args_ = get_predict_cl_args()

    run_folder_ = Path(predict_cl_args_.model_path).parents[1]
    predict_config_ = get_default_predict_config(
        run_folder=run_folder_,
        predict_cl_args=predict_cl_args_,
    )

    predict(predict_config=predict_config_, predict_cl_args=predict_cl_args_)
