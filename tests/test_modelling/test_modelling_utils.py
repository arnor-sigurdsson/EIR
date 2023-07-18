from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import pandas as pd

if TYPE_CHECKING:
    from eir.setup.output_setup import al_output_objects_as_dict


def check_performance_result_wrapper(
    outputs: "al_output_objects_as_dict",
    run_path: Path,
    max_thresholds: Tuple[float, float],
    min_thresholds: Tuple[float, float] = (1.5, 1.0),
    cat_metric: Optional[str] = "mcc",
    con_metric: Optional[str] = "r2",
) -> None:
    for output_name, output_object in outputs.items():
        if output_object.output_config.output_info.output_type == "tabular":
            cat_target_columns = output_object.target_columns["cat"]
            con_target_columns = output_object.target_columns["con"]

            if cat_metric is not None:
                for cat_target_column in cat_target_columns:
                    check_test_performance_results(
                        run_path=run_path,
                        target_column=cat_target_column,
                        output_name=output_name,
                        metric=cat_metric,
                        thresholds=max_thresholds,
                    )
            if con_metric is not None:
                for con_target_column in con_target_columns:
                    check_test_performance_results(
                        run_path=run_path,
                        output_name=output_name,
                        target_column=con_target_column,
                        metric=con_metric,
                        thresholds=max_thresholds,
                    )

        elif output_object.output_config.output_info.output_type == "sequence":
            check_test_performance_results(
                run_path=run_path,
                output_name=output_name,
                target_column=output_name,
                metric="loss",
                thresholds=min_thresholds,
                direction="min",
            )


def check_test_performance_results(
    run_path: Path,
    output_name: str,
    target_column: str,
    metric: str,
    thresholds: Tuple[float, float],
    direction: str = "max",
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
    df_valid = pd.read_csv(valid_history_path)
    key = f"{output_name}_{target_column}_{metric}"
    if direction == "max":
        assert df_train.loc[:, key].max() > threshold_train
        assert df_valid.loc[:, key].max() > threshold_valid
    elif direction == "min":
        assert df_train.loc[:, key].min() < threshold_train
        assert df_valid.loc[:, key].min() < threshold_valid
