import argparse
import json
from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Dict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from aislib.misc_utils import get_logger
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

import snp_pred.visualization.visualization_funcs as vf
from snp_pred.data_load import datasets, label_setup
from snp_pred.data_load.data_utils import get_target_columns_generator
from snp_pred.data_load.datasets import (
    al_datasets,
    merge_target_columns,
)
from snp_pred.data_load.label_setup import (
    al_label_dict,
    al_label_transformers_object,
    al_label_transformers,
    al_all_column_ops,
)
from snp_pred.models.extra_inputs_module import al_emb_lookup_dict
from snp_pred.models.model_training_utils import gather_pred_outputs_from_dloader
from snp_pred.models.models import get_model_class
from snp_pred.models.models_cnn import CNNModel
from snp_pred.models.models_mlp import MLPModel
from snp_pred.train_utils.evaluation import PerformancePlotConfig
from snp_pred.train_utils.utils import get_run_folder

torch.manual_seed(0)
np.random.seed(0)

logger = get_logger(name=__name__, tqdm_compatible=True)


def predict(
    predict_cl_args: Namespace, custom_label_ops: al_all_column_ops = None
) -> None:
    outfolder = Path(predict_cl_args.output_folder)
    run_folder = Path(predict_cl_args.model_path).parents[1]

    c = get_test_config(
        run_folder=run_folder,
        predict_cl_args=predict_cl_args,
        custom_label_ops=custom_label_ops,
    )

    all_preds, all_labels, all_ids = gather_pred_outputs_from_dloader(
        data_loader=c.test_dataloader,
        cl_args=c.train_args_modified_for_testing,
        model=c.model,
        device=predict_cl_args.device,
        with_labels=predict_cl_args.evaluate,
    )

    target_columns_gen = get_target_columns_generator(
        target_columns=c.test_dataset.target_columns
    )

    for target_column_type, target_column in target_columns_gen:
        target_preds = all_preds[target_column]

        cur_target_transformer = c.test_dataset.target_transformers[target_column]
        preds_sm = F.softmax(input=target_preds, dim=1).cpu().numpy()

        classes = _get_target_classnames(
            transformer=cur_target_transformer, target_column=target_column
        )
        _save_predictions(
            preds=preds_sm,
            test_dataset=c.test_dataset,
            classes=classes,
            outfolder=outfolder,
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
                output_folder=outfolder,
            )

            vf.gen_eval_graphs(plot_config=plot_config)


@dataclass
class TestConfig:
    train_cl_args: Namespace
    train_args_modified_for_testing: Namespace
    test_dataset: datasets.DiskArrayDataset
    test_dataloader: DataLoader
    model: Union[CNNModel, MLPModel]


def get_test_config(
    run_folder: Path,
    predict_cl_args: Namespace,
    custom_label_ops: al_all_column_ops = None,
):
    train_cl_args = _load_cl_args_config(
        cl_args_config_path=run_folder / "cl_args.json"
    )

    test_train_mixed_cl_args = _modify_train_cl_args_for_testing(
        train_cl_args=train_cl_args, predict_cl_args=predict_cl_args
    )

    test_labels_dict = None
    if predict_cl_args.evaluate:
        test_labels_dict = _load_labels_for_testing(
            test_train_cl_args_mix=test_train_mixed_cl_args,
            custom_label_ops=custom_label_ops,
        )

    test_dataset = _set_up_test_dataset(
        test_train_cl_args_mix=test_train_mixed_cl_args,
        test_labels_dict=test_labels_dict,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=predict_cl_args.batch_size,
        shuffle=False,
        num_workers=predict_cl_args.num_workers,
    )

    model = _load_model(
        model_path=Path(predict_cl_args.model_path),
        num_classes=test_dataset.num_classes,
        train_cl_args=train_cl_args,
        device=predict_cl_args.device,
    )
    assert not model.training

    test_config = TestConfig(
        train_cl_args=train_cl_args,
        train_args_modified_for_testing=test_train_mixed_cl_args,
        test_dataset=test_dataset,
        test_dataloader=test_dataloader,
        model=model,
    )
    return test_config


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
    model_path: Path, num_classes: int, train_cl_args: Namespace, device: str
) -> torch.nn.Module:
    """
    :param model_path: Path to the model as passed in CL arguments.
    :param num_classes: Number of classes the model was trained on, used to set up last
    layer neurons.
    :param train_cl_args: CL arguments used during training, used to set up various
    aspects of model architecture.
    :param device: Which device to cast the model to.
    :return: Loaded model initialized with saved weights.
    """

    embeddings_dict = _load_saved_embeddings_dict(
        embed_columns=train_cl_args.extra_cat_columns, run_name=train_cl_args.run_name
    )

    mode_class = get_model_class(train_cl_args.model_type)

    model: torch.nn.Module = mode_class(
        cl_args=train_cl_args,
        num_classes=num_classes,
        embeddings_dict=embeddings_dict,
        extra_continuous_inputs_columns=train_cl_args.extra_con_columns,
    )

    model = _load_model_weights(
        model=model, model_state_dict_path=model_path, device=device
    )

    model.eval()

    return model


def _load_model_weights(
    model: Union[CNNModel, MLPModel], model_state_dict_path: Path, device: str
) -> Union[CNNModel, MLPModel]:
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
    test_train_cl_args_mix: Namespace, custom_label_ops: al_all_column_ops = None
) -> al_label_dict:
    """
    Used when doing an evaluation on test set.

    :param test_train_cl_args_mix: Training CL arguments with the data_source
    replaced with testing one.
    :return: Testing labels for performance measurement.
    """

    a = test_train_cl_args_mix

    parse_wrapper = label_setup.get_label_parsing_wrapper(
        cl_args=test_train_cl_args_mix
    )
    df_labels_test = parse_wrapper(cl_args=a, custom_label_ops=custom_label_ops)

    train_con_column_means = _prep_missing_con_dict(test_train_cl_args_mix=a)
    df_labels_test = label_setup.handle_missing_label_values_in_df(
        df=df_labels_test,
        cl_args=test_train_cl_args_mix,
        con_manual_values=train_con_column_means,
        name="test_df",
    )
    test_labels_dict = df_labels_test.to_dict("index")

    return test_labels_dict


def _prep_missing_con_dict(test_train_cl_args_mix: Namespace) -> Dict[str, float]:
    a = test_train_cl_args_mix
    con_columns = a.extra_con_columns + a.target_con_columns

    extra_con_transformers = _load_transformers(
        run_name=a.run_name, transformers_to_load=con_columns
    )

    train_means = {
        column: transformer.mean_[0]
        for column, transformer in extra_con_transformers.items()
    }

    return train_means


def _set_up_test_dataset(
    test_train_cl_args_mix: Namespace, test_labels_dict: Union[None, al_label_dict]
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

    target_columns_flat = target_columns["con"] + target_columns["cat"]
    target_transformers = _load_transformers(
        run_name=a.run_name, transformers_to_load=target_columns_flat
    )

    extra_con_transformers = _load_transformers(
        run_name=a.run_name, transformers_to_load=a.extra_con_columns
    )

    test_dataset = datasets.DiskArrayDataset(
        data_source=a.data_source,
        target_columns=target_columns,
        labels_dict=test_labels_dict,
        target_transformers=target_transformers,
        extra_con_transformers=extra_con_transformers,
        target_width=a.target_width,
    )

    return test_dataset


def _load_transformers(
    run_name: str, transformers_to_load: List[str]
) -> al_label_transformers:
    run_folder = get_run_folder(run_name)

    label_transformers = {}
    for transformer_name in transformers_to_load:
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


if __name__ == "__main__":
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

    if cl_args.device == "gpu":
        cl_args.device = (
            "cuda:" + cl_args.gpu_num if torch.cuda.is_available() else "cpu"
        )
        if cl_args.device == "cpu":
            logger.warning(
                "Device CL input was 'gpu' with gpu_num '%s', "
                "but number of available GPUs is %d. Reverting to CPU for "
                "now.",
                cl_args.gpu_num,
                torch.cuda.device_count(),
            )

    predict(predict_cl_args=cl_args)