import argparse
import json
from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Dict, Sequence, Callable, Iterable, Tuple

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
    merge_target_columns,
    gather_ids_from_data_source,
    transform_label_df,
    TabularFileInfo,
)
from snp_pred.models.extra_inputs_module import al_emb_lookup_dict
from snp_pred.models.model_training_utils import gather_pred_outputs_from_dloader
from snp_pred.models.models import get_model_class
from snp_pred.train import (
    prepare_base_batch_default,
    Hooks,
    get_tabular_target_label_data,
    get_tabular_inputs_label_data,
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
    train_cl_args: Namespace
    train_args_modified_for_testing: Namespace
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


def set_up_all_label_data(cl_args: argparse.Namespace) -> TabularFileInfo:

    table_info = TabularFileInfo(
        file_path=cl_args.label_file,
        con_columns=cl_args.target_con_columns + cl_args.extra_con_columns,
        cat_columns=cl_args.target_cat_columns + cl_args.extra_cat_columns,
        parsing_chunk_size=cl_args.label_parsing_chunk_size,
    )

    return table_info


def get_default_predict_config(
    run_folder: Path,
    predict_cl_args: Namespace,
) -> PredictConfig:

    with open(
        str(run_folder / "serializations" / "filtered_config.dill"), "rb"
    ) as infile:
        train_config = dill.load(file=infile)

    train_cl_args = train_config.cl_args

    test_train_mixed_cl_args = _modify_train_cl_args_for_testing(
        train_cl_args=train_cl_args, predict_cl_args=predict_cl_args
    )

    default_train_hooks = train_config.hooks

    target_labels_dict, extra_labels_dict = None, None
    if predict_cl_args.evaluate:
        test_ids = gather_ids_from_data_source(
            data_source=Path(predict_cl_args.data_source)
        )

        label_ops = default_train_hooks.custom_column_label_parsing_ops
        target_labels_dict, extra_labels_dict = get_target_and_extra_labels_for_predict(
            train_cl_args=train_cl_args,
            custom_column_label_parsing_ops=label_ops,
            ids=test_ids,
        )

    test_dataset = _set_up_test_dataset(
        test_train_cl_args_mix=test_train_mixed_cl_args,
        test_labels_dict=target_labels_dict,
        tabular_inputs_labels_dict=extra_labels_dict,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=predict_cl_args.batch_size,
        shuffle=False,
        num_workers=predict_cl_args.num_workers,
    )

    model = _load_model(
        model_path=Path(predict_cl_args.model_path),
        num_outputs_per_target=train_config.num_outputs_per_target,
        train_cl_args=train_cl_args,
        device=predict_cl_args.device,
    )
    assert not model.training

    default_predict_hooks = _get_default_predict_hooks(train_hooks=default_train_hooks)
    test_config = PredictConfig(
        train_cl_args=train_cl_args,
        train_args_modified_for_testing=test_train_mixed_cl_args,
        test_dataset=test_dataset,
        target_columns=train_config.target_columns,
        target_transformers=train_config.target_transformers,
        test_dataloader=test_dataloader,
        model=model,
        hooks=default_predict_hooks,
    )
    return test_config


def get_target_and_extra_labels_for_predict(
    train_cl_args: Namespace,
    custom_column_label_parsing_ops: al_all_column_ops,
    ids: Sequence[str],
) -> Tuple[Dict, Dict]:

    target_cols = train_cl_args.target_cat_columns + train_cl_args.target_con_columns
    target_transformers = _load_transformers(
        run_name=train_cl_args.run_name,
        transformers_to_load=target_cols,
    )

    target_labels_info = get_tabular_target_label_data(cl_args=train_cl_args)
    df_target_labels_predict = _load_labels_for_testing(
        tabular_info=target_labels_info,
        ids_to_keep=ids,
        custom_label_ops=custom_column_label_parsing_ops,
    )
    target_labels_dict = parse_labels_for_testing(
        tabular_info=target_labels_info,
        df_labels_test=df_target_labels_predict,
        label_transformers=target_transformers,
    )

    tabular_cols = train_cl_args.extra_cat_columns + train_cl_args.extra_con_columns
    extra_transformers = _load_transformers(
        run_name=train_cl_args.run_name,
        transformers_to_load=tabular_cols,
    )
    extra_labels_info = get_tabular_inputs_label_data(cl_args=train_cl_args)
    df_extra_labels_predict = _load_labels_for_testing(
        tabular_info=extra_labels_info,
        ids_to_keep=ids,
        custom_label_ops=custom_column_label_parsing_ops,
    )
    extra_labels_dict = parse_labels_for_testing(
        tabular_info=extra_labels_info,
        df_labels_test=df_extra_labels_predict,
        label_transformers=extra_transformers,
    )

    return target_labels_dict, extra_labels_dict


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
        cl_args=predict_config.train_cl_args,
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


def _load_model(
    model_path: Path,
    num_outputs_per_target: Dict[str, int],
    train_cl_args: Namespace,
    device: str,
) -> torch.nn.Module:
    """
    :param model_path: Path to the model as passed in CL arguments.
    :param num_outputs_per_target: Number of classes the model was trained on,
    used to set up last layer neurons.
    :param train_cl_args: CL arguments used during training, used to set up various
    aspects of model architecture.
    :param device: Which device to cast the model to.
    :return: Loaded model initialized with saved weights.
    """

    embeddings_dict = _load_saved_embeddings_dict(
        embed_columns=train_cl_args.extra_cat_columns, run_name=train_cl_args.run_name
    )

    model_class = get_model_class(train_cl_args.model_type)

    model: torch.nn.Module = model_class(
        cl_args=train_cl_args,
        num_outputs_per_target=num_outputs_per_target,
        embeddings_dict=embeddings_dict,
        extra_continuous_inputs_columns=train_cl_args.extra_con_columns,
    )

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


def _modify_train_cl_args_for_testing(
    train_cl_args: Namespace, predict_cl_args: Namespace
) -> Namespace:
    """
    When initalizing the datasets and model classes, we want to make sure we have the
    same configuration as when training the model, with the exception of which
    data_source to get observations from (i.e. here we want the test set folder).

    We use deepcopy to make sure the training configuration stays frozen.
    """
    train_cl_args_mod = deepcopy(train_cl_args)
    train_cl_args_mod.data_source = predict_cl_args.data_source

    return train_cl_args_mod


def _load_labels_for_testing(
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


def _set_up_test_dataset(
    test_train_cl_args_mix: Namespace,
    test_labels_dict: Union[None, al_label_dict],
    tabular_inputs_labels_dict: Union[None, al_label_dict],
) -> al_datasets:
    """
    :param test_train_cl_args_mix: Training CL arguments with the data_source
    replaced with testing one.
    :param test_labels_dict: None if we are predicting on unknown data,
    otherwise a dictionary of labels (if evaluating on test set).
    :return: Dataset instance to be used for loading test samples.
    """
    a = test_train_cl_args_mix

    target_columns = merge_target_columns(
        target_con_columns=a.target_con_columns, target_cat_columns=a.target_cat_columns
    )

    test_dataset = datasets.DiskDataset(
        data_source=a.data_source,
        target_columns=target_columns,
        target_width=a.target_width,
        target_labels_dict=test_labels_dict,
        tabular_inputs_labels_dict=tabular_inputs_labels_dict,
    )

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

    parser.add_argument("--evaluate", dest="evaluate", action="store_true")
    parser.set_defaults(evaluate=False)

    parser.add_argument(
        "--data_source",
        type=str,
        required=True,
        help="Path to folder with samples to predict on.",
    )

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
        "--num_workers",
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
