from pathlib import Path
from typing import Tuple, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from eir.setup.output_setup import al_output_objects_as_dict


def check_performance_result_wrapper(
    outputs: "al_output_objects_as_dict",
    run_path: Path,
    thresholds: Tuple[float, float],
) -> None:
    for output_name, output_object in outputs.items():
        cat_target_columns = output_object.target_columns["cat"]
        con_target_columns = output_object.target_columns["con"]

        for cat_target_column in cat_target_columns:
            check_test_performance_results(
                run_path=run_path,
                target_column=cat_target_column,
                output_name=output_name,
                metric="mcc",
                thresholds=thresholds,
            )

        for con_target_column in con_target_columns:
            check_test_performance_results(
                run_path=run_path,
                output_name=output_name,
                target_column=con_target_column,
                metric="r2",
                thresholds=thresholds,
            )


def check_test_performance_results(
    run_path: Path,
    output_name: str,
    target_column: str,
    metric: str,
    thresholds: Tuple[float, float],
):
    target_column_results_folder = run_path / "results" / output_name / target_column

    train_history_path = (
        target_column_results_folder / f"train_{target_column}_history.log"
    )
    valid_history_path = (
        target_column_results_folder / f"validation_{target_column}_history.log"
    )

    threshold_train, threshold_valid = thresholds

    df_train = pd.read_csv(train_history_path)
    key = f"{output_name}_{target_column}_{metric}"
    assert df_train.loc[:, key].max() > threshold_train

    df_valid = pd.read_csv(valid_history_path)
    assert df_valid.loc[:, key].max() > threshold_valid
