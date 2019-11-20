import argparse
import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aislib.misc_utils import get_logger

import human_origins_supervised.visualization.visualization_funcs as vf
from human_origins_supervised.data_load import datasets, label_setup
from human_origins_supervised.data_load.datasets import al_datasets
from human_origins_supervised.data_load.label_setup import al_label_dict
from human_origins_supervised.models.model_utils import gather_pred_outputs_from_dloader
from human_origins_supervised.models.models import Model

torch.manual_seed(0)
np.random.seed(0)

logger = get_logger(__name__)


def load_cl_args_config(cl_args_config_path: Path) -> Namespace:
    with open(str(cl_args_config_path), "r") as infile:
        loaded_cl_args = json.load(infile)

    return Namespace(**loaded_cl_args)


def load_model(
    model_path: Path, n_classes: int, train_cl_args: Namespace, device: str
) -> torch.nn.Module:
    embeddings_dict = None
    if train_cl_args.embed_columns:
        model_embeddings_path = (
            model_path.parents[1] / "extra_inputs" / "embeddings.save"
        )
        embeddings_dict = joblib.load(model_embeddings_path)

    model: torch.nn.Module = Model(
        train_cl_args, n_classes, embeddings_dict, train_cl_args.contn_columns
    )
    device_for_load = torch.device(device)
    model.load_state_dict(torch.load(model_path, map_location=device_for_load))
    model.eval()
    model = model.to(device=device)

    return model


def modify_train_cl_args_for_testing(
    train_cl_args: Namespace, predict_cl_args: Namespace
) -> Namespace:
    """
    When initalizing the datasets and model classes, we want to make sure we have the
    same configuration as when training the model, with the exception of which
    data_folder to get observations from (i.e. here we want the test set folder).

    We use deepcopy to make sure the training configuration stays frozen.
    """
    train_cl_args_mod = deepcopy(train_cl_args)
    train_cl_args_mod.data_folder = predict_cl_args.data_folder

    return train_cl_args_mod


def load_labels_for_testing(test_train_cl_args_mix: Namespace) -> al_label_dict:
    df_labels_test = label_setup.label_df_parse_wrapper(test_train_cl_args_mix)

    continuous_columns = test_train_cl_args_mix.contn_columns[:]

    for continuous_column in continuous_columns:
        scaler_path = label_setup.get_transformer_path(
            test_train_cl_args_mix.run_name,
            test_train_cl_args_mix.target_column,
            "standard_scaler",
        )
        df_labels_test, _ = label_setup.scale_non_target_continuous_columns(
            df_labels_test,
            continuous_column,
            test_train_cl_args_mix.run_name,
            scaler_path,
        )

    df_labels_test = label_setup.handle_missing_label_values(
        df_labels_test, test_train_cl_args_mix, "test_df"
    )
    test_labels_dict = df_labels_test.to_dict("index")

    return test_labels_dict


def set_up_test_dataset(
    test_train_cl_args_mix: Namespace, test_labels_dict: al_label_dict
) -> al_datasets:
    dataset_class_common_args = datasets.construct_dataset_init_params_from_cl_args(
        test_train_cl_args_mix
    )

    # TODO: Update this, currently we need to have the model in a hardcoded runs
    #       folder when loading.
    target_transformer_path = label_setup.get_transformer_path(
        test_train_cl_args_mix.run_name,
        test_train_cl_args_mix.target_column,
        "target_transformer",
    )
    target_transformer = joblib.load(target_transformer_path)

    test_dataset = datasets.DiskArrayDataset(
        **dataset_class_common_args,
        labels_dict=test_labels_dict,
        target_transformer=target_transformer,
    )

    return test_dataset


def save_predictions(
    preds: torch.Tensor, test_dataset: al_datasets, outfolder: Path
) -> None:
    test_ids = [i.sample_id for i in test_dataset.samples]
    classes = test_dataset.target_transformer.classes_
    df_preds = pd.DataFrame(data=preds, index=test_ids, columns=classes)
    df_preds.index.name = "ID"

    df_preds.to_csv(outfolder / "predictions.csv")


def predict(predict_cl_args: Namespace) -> None:
    outfolder = Path(predict_cl_args.output_folder)

    # Set up CL arguments
    train_cl_args = load_cl_args_config(
        Path(predict_cl_args.model_path).parents[1] / "cl_args.json"
    )
    test_train_mixed_cl_args = modify_train_cl_args_for_testing(
        train_cl_args, predict_cl_args
    )

    # Set up data loading
    test_labels_dict = None
    if predict_cl_args.evaluate:
        test_labels_dict = load_labels_for_testing(test_train_mixed_cl_args)

    test_dataset = set_up_test_dataset(test_train_mixed_cl_args, test_labels_dict)
    test_dloader = DataLoader(
        test_dataset, batch_size=predict_cl_args.batch_size, shuffle=False
    )

    # Set up model
    model = load_model(
        Path(predict_cl_args.model_path),
        len(test_dataset.target_transformer.classes_),
        train_cl_args,
        predict_cl_args.device,
    )
    assert not model.training

    # Get predictions
    preds, labels, ids = gather_pred_outputs_from_dloader(
        test_dloader,
        test_train_mixed_cl_args,
        model,
        device=predict_cl_args.device,
        labels_dict=test_dataset.labels_dict,
        with_labels=predict_cl_args.evaluate,
    )
    preds_sm = F.softmax(preds, dim=1).cpu().numpy()

    # Evaluate / analyse predictions
    save_predictions(preds_sm, test_dataset, outfolder)

    if predict_cl_args.evaluate:
        vf.gen_eval_graphs(
            labels.cpu().numpy(),
            preds_sm,
            ids,
            outfolder,
            test_dataset.target_transformer,
            test_train_mixed_cl_args.model_task,
        )


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
        "--data_folder",
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

    predict(cl_args)
