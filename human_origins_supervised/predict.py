import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

import human_origins_supervised.visualization.visualization_funcs as vf
from human_origins_supervised.models import data_load
from human_origins_supervised.models.model_utils import gather_pred_outputs_from_dloader
from human_origins_supervised.models.models import Model

torch.manual_seed(0)
np.random.seed(0)


def load_run_config(run_config_path: Path) -> SimpleNamespace:
    with open(str(run_config_path), "r") as infile:
        run_config = json.load(infile)

    return SimpleNamespace(**run_config)


def load_model(model_path: Path, n_classes):
    model_run_config_path = model_path.parents[1] / "run_config.json"
    run_config = load_run_config(model_run_config_path)

    model = Model(run_config, n_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def predict(cl_args):
    outfolder = Path(cl_args.output_folder)

    test_dataset = data_load.MemoryArrayDataset(
        data_folder=cl_args.data_folder, with_labels=cl_args.evaluate
    )

    test_dloader = DataLoader(
        test_dataset, batch_size=cl_args.batch_size, shuffle=False
    )

    classes = np.load(cl_args.classes_path)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = classes

    model = load_model(Path(cl_args.model_path), len(classes))
    model = model.to(cl_args.device)

    preds, labels = gather_pred_outputs_from_dloader(
        test_dloader, model, cl_args.device, with_labels=cl_args.evaluate
    )
    preds_sm = F.softmax(preds, dim=1)
    preds = preds_sm.cpu().numpy()

    df_preds = pd.DataFrame(data=preds, index=test_dataset.ids, columns=classes)

    df_preds.to_csv(outfolder / "predictions.csv")

    if cl_args.evaluate:
        vf.gen_eval_graphs(labels.cpu().numpy(), preds, outfolder, label_encoder)


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

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Whether to test performance on the data."
        "Assumes that there are labels.",
    )

    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Path observations to predict on.",
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
