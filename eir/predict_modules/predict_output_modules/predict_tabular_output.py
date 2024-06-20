from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from aislib.misc_utils import ensure_path_exists
from sklearn.preprocessing import LabelEncoder, StandardScaler

from eir.data_load.data_utils import get_output_info_generator
from eir.data_load.label_setup import al_label_transformers_object
from eir.setup.output_setup_modules.tabular_output_setup import (
    ComputedTabularOutputInfo,
)
from eir.train_utils.evaluation import PerformancePlotConfig
from eir.visualization import visualization_funcs as vf

if TYPE_CHECKING:
    from eir.predict import PredictExperiment


def predict_tabular_wrapper_with_labels(
    predict_config: "PredictExperiment",
    all_predictions: dict[str, dict[str, torch.Tensor]],
    all_labels: dict[str, dict[str, torch.Tensor]],
    all_ids: dict[str, dict[str, list[str]]],
    predict_cl_args: Namespace,
) -> None:
    target_columns_gen = get_output_info_generator(
        outputs_as_dict=predict_config.outputs
    )

    for output_name, target_head_name, target_column_name in target_columns_gen:
        if target_head_name == "general":
            continue

        target_predictions = all_predictions[output_name][target_column_name]
        predictions = _parse_predictions(target_predictions=target_predictions)

        cur_output_object = predict_config.outputs[output_name]
        assert isinstance(cur_output_object, ComputedTabularOutputInfo)

        target_transformers = cur_output_object.target_transformers
        cur_target_transformer = target_transformers[target_column_name]
        classes = _get_target_class_names(
            transformer=cur_target_transformer, target_column=target_column_name
        )

        output_folder = Path(
            predict_cl_args.output_folder, output_name, target_column_name
        )
        ensure_path_exists(path=output_folder, is_folder=True)

        target_labels = None
        if all_labels:
            target_labels = all_labels[output_name][target_column_name].cpu().numpy()

        cur_ids = all_ids[output_name][target_column_name]

        df_merged_predictions = _merge_ids_predictions_and_labels(
            ids=cur_ids,
            predictions=predictions,
            labels=target_labels,
            prediction_classes=classes,
        )

        df_predictions = _add_inverse_transformed_columns_to_predictions(
            df=df_merged_predictions,
            target_column_name=target_column_name,
            target_column_type=target_head_name,
            transformer=cur_target_transformer,
            evaluation=predict_cl_args.evaluate,
        )

        _save_predictions(
            df_predictions=df_predictions,
            output_folder=output_folder,
        )

        if predict_cl_args.evaluate:
            cur_labels = all_labels[output_name][target_column_name].cpu().numpy()

            plot_config = PerformancePlotConfig(
                val_outputs=predictions,
                val_labels=cur_labels,
                val_ids=cur_ids,
                iteration=0,
                column_name=target_column_name,
                column_type=target_head_name,
                target_transformer=cur_target_transformer,
                output_folder=output_folder,
            )

            vf.gen_eval_graphs(plot_config=plot_config)


def _merge_ids_predictions_and_labels(
    ids: Sequence[str],
    predictions: np.ndarray,
    labels: Optional[np.ndarray],
    prediction_classes: Union[Sequence[str], None] = None,
    label_column_name: str = "True Label",
) -> pd.DataFrame:
    df = pd.DataFrame()

    df["ID"] = ids
    df = df.set_index("ID")

    if labels is not None:
        df[label_column_name] = labels

    if prediction_classes is None:
        prediction_classes = [f"Score Class {i}" for i in range(predictions.shape[1])]

    df[prediction_classes] = predictions

    return df


def predict_tabular_wrapper_no_labels(
    predict_config: "PredictExperiment",
    all_predictions: dict[str, dict[str, torch.Tensor]],
    all_ids: dict[str, dict[str, list[str]]],
    predict_cl_args: Namespace,
) -> None:
    target_columns_gen = get_output_info_generator(
        outputs_as_dict=predict_config.outputs
    )

    for output_name, target_head_name, target_column_name in target_columns_gen:
        if target_head_name == "general":
            continue

        target_predictions = all_predictions[output_name][target_column_name]
        predictions = _parse_predictions(target_predictions=target_predictions)

        cur_output_object = predict_config.outputs[output_name]
        assert isinstance(cur_output_object, ComputedTabularOutputInfo)

        target_transformers = cur_output_object.target_transformers
        cur_target_transformer = target_transformers[target_column_name]
        classes = _get_target_class_names(
            transformer=cur_target_transformer, target_column=target_column_name
        )

        output_folder = Path(
            predict_cl_args.output_folder, output_name, target_column_name
        )
        ensure_path_exists(path=output_folder, is_folder=True)

        cur_ids = all_ids[output_name][target_column_name]

        df_predictions = _merge_ids_and_predictions(
            ids=cur_ids,
            predictions=predictions,
            prediction_classes=classes,
        )

        df_predictions = _add_inverse_transformed_columns_to_predictions(
            df=df_predictions,
            target_column_name=target_column_name,
            target_column_type=target_head_name,
            transformer=cur_target_transformer,
            evaluation=predict_cl_args.evaluate,
        )

        _save_predictions(
            df_predictions=df_predictions,
            output_folder=output_folder,
        )


def _merge_ids_and_predictions(
    ids: Sequence[str],
    predictions: np.ndarray,
    prediction_classes: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    df = pd.DataFrame()

    df["ID"] = ids
    df = df.set_index("ID")

    if prediction_classes is None:
        prediction_classes = [f"Score Class {i}" for i in range(predictions.shape[1])]

    df[prediction_classes] = predictions

    return df


def _add_inverse_transformed_columns_to_predictions(
    df: pd.DataFrame,
    target_column_name: str,
    target_column_type: str,
    transformer: al_label_transformers_object,
    evaluation: bool,
) -> pd.DataFrame:
    df_copy = df.copy()

    assert target_column_type in ["con", "cat"], target_column_type

    if evaluation:
        df = _add_inverse_transformed_column(
            df=df_copy,
            column_name="True Label",
            transformer=transformer,
        )

    if target_column_type == "con":
        df = _add_inverse_transformed_column(
            df=df,
            column_name=target_column_name,
            transformer=transformer,
        )

    return df


def _add_inverse_transformed_column(
    df: pd.DataFrame,
    column_name: str,
    transformer: al_label_transformers_object,
) -> pd.DataFrame:
    df_copy = df.copy()

    tt_it = transformer.inverse_transform
    values = np.asarray(df[column_name].values)
    col_name = f"{column_name} Untransformed"

    if isinstance(transformer, LabelEncoder):
        df_copy.insert(loc=0, column=col_name, value=tt_it(values))
    elif isinstance(transformer, StandardScaler):
        values_parsed = tt_it(values.reshape(-1, 1)).flatten()
        df_copy.insert(loc=0, column=col_name, value=values_parsed)
    else:
        raise NotImplementedError(f"Transformer {transformer} not supported.")

    return df_copy


def _parse_predictions(target_predictions: torch.Tensor) -> np.ndarray:
    predictions = target_predictions.cpu().numpy()
    return predictions


def _get_target_class_names(
    transformer: al_label_transformers_object, target_column: str
):
    if isinstance(transformer, LabelEncoder):
        return transformer.classes_
    return [target_column]


def _save_predictions(df_predictions: pd.DataFrame, output_folder: Path) -> None:
    df_predictions.to_csv(path_or_buf=str(output_folder / "predictions.csv"))
