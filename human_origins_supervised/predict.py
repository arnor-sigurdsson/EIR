import argparse
import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import human_origins_supervised.visualization.visualization_funcs as vf
from human_origins_supervised.data_load import datasets, label_setup
from human_origins_supervised.data_load.label_setup import al_label_dict
from human_origins_supervised.models.model_utils import gather_pred_outputs_from_dloader
from human_origins_supervised.models.models import Model

torch.manual_seed(0)
np.random.seed(0)


def load_cl_args_config(cl_args_config_path: Path) -> Namespace:
    with open(str(cl_args_config_path), "r") as infile:
        loaded_cl_args = json.load(infile)

    return Namespace(**loaded_cl_args)


def load_model(model_path: Path, n_classes: int, train_cl_args: Namespace):

    embeddings_dict = None
    if train_cl_args.embed_columns:
        model_embeddings_path = (
            model_path.parents[1] / "extra_inputs" / "embeddings.save"
        )
        embeddings_dict = joblib.load(model_embeddings_path)

    model = Model(
        train_cl_args, n_classes, embeddings_dict, train_cl_args.contn_columns
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def modify_train_cl_args_for_testing(train_cl_args: Namespace, test_cl_args: Namespace):
    """
    TODO:
        Make this clearer, maybe something like test_cl_args, which is mixed
        from train/test, and predict_cl_args, which refers to CL args given to this
        script.
    """
    train_cl_args_mod = deepcopy(train_cl_args)
    train_cl_args_mod.data_folder = test_cl_args.data_folder

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
) -> Union[datasets.DiskArrayDataset, datasets.MemoryArrayDataset]:
    dataset_class_common_args = datasets.construct_dataset_init_params_from_cl_args(
        test_train_cl_args_mix
    )

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


def predict(test_cl_args):
    outfolder = Path(test_cl_args.output_folder)

    # Set up CL arguments
    train_cl_args = load_cl_args_config(
        Path(test_cl_args.model_path).parents[1] / "cl_args.json"
    )
    test_train_mixed_cl_args = modify_train_cl_args_for_testing(
        train_cl_args, test_cl_args
    )

    # Set up data loading
    test_labels_dict = load_labels_for_testing(test_train_mixed_cl_args)
    test_dataset = set_up_test_dataset(test_train_mixed_cl_args, test_labels_dict)
    test_dloader = DataLoader(
        test_dataset, batch_size=test_cl_args.batch_size, shuffle=False
    )

    # Set up model
    model = load_model(
        Path(test_cl_args.model_path), test_dataset.num_classes, train_cl_args
    )
    model = model.to(test_cl_args.device)
    assert not model.training

    # Get predictions
    preds, labels, ids = gather_pred_outputs_from_dloader(
        test_dloader,
        test_train_mixed_cl_args,
        model,
        device=test_train_mixed_cl_args.device,
        labels_dict=test_dataset.labels_dict,
        with_labels=test_cl_args.evaluate,
    )

    preds_sm = F.softmax(preds, dim=1)
    preds = preds_sm.cpu().numpy()

    # Evaluate / analyse predictions
    test_ids = [i.sample_id for i in test_dataset.samples]
    classes = test_dataset.target_transformer.classes_
    df_preds = pd.DataFrame(data=preds, index=test_ids, columns=classes)
    df_preds.index.name = "ID"

    df_preds.to_csv(outfolder / "predictions.csv")

    if test_cl_args.evaluate:
        vf.gen_eval_graphs(
            labels.cpu().numpy(),
            preds,
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
        "--labels_folder",
        type=str,
        required=False,
        help="Path to label file for samples. Used when evaluating on test data and/or "
        "using extra label inputs..",
    )

    parser.add_argument(
        "--classes_path",
        type=str,
        required=True,
        help="Path to a .npy file with class names in same"
        "order as the numerical encoding in the model.",
    )

    parser.add_argument(
        "--output_folder", type=str, help="Where to save prediction results."
    )

    parser.add_argument(
        "--gpu_num", type=str, default="0", help="Which GPU to run (according to CUDA)."
    )

    cl_args = parser.parse_args()

    cl_args.device = "cuda:" + cl_args.gpu_num if torch.cuda.is_available() else "cpu"

    predict(cl_args)
