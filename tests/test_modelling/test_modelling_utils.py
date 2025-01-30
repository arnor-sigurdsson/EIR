from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from eir.setup.output_setup import al_output_objects_as_dict


def check_performance_result_wrapper(
    outputs: "al_output_objects_as_dict",
    run_path: Path,
    max_thresholds: tuple[float, float],
    min_thresholds: tuple[float, float] = (1.5, 1.0),
    survival_max_thresholds: tuple[float, float] = (0.6, 0.6),
    survival_min_thresholds: tuple[float, float] = (2.0, 2.0),
    cat_metric: str | None = "mcc",
    con_metric: str | None = "r2",
) -> None:
    for output_name, output_object in outputs.items():
        output_type = output_object.output_config.output_info.output_type
        if output_type == "tabular":
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

        elif output_type in ("sequence",):
            check_test_performance_results(
                run_path=run_path,
                output_name=output_name,
                target_column=output_name,
                metric="loss",
                thresholds=min_thresholds,
                direction="min",
            )

        elif output_type in ("survival",):
            check_test_performance_results(
                run_path=run_path,
                output_name=output_name,
                target_column="BinaryOrigin",
                metric="loss",
                thresholds=survival_min_thresholds,
                direction="min",
            )

            check_test_performance_results(
                run_path=run_path,
                output_name=output_name,
                target_column="BinaryOrigin",
                metric="c-index",
                thresholds=survival_max_thresholds,
                direction="max",
                check_train=False,
            )


def check_test_performance_results(
    run_path: Path,
    output_name: str,
    target_column: str,
    metric: str,
    thresholds: tuple[float, float],
    direction: str = "max",
    check_train: bool = True,
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

    fail_msg = (
        f"Failed for {output_name}, {target_column}, {metric} with "
        f"direction {direction} and thresholds {thresholds} for "
        f"run path {run_path}."
    )

    key = f"{output_name}_{target_column}_{metric}"
    if direction == "max":
        if check_train and key in df_train.columns:
            assert df_train.loc[:, key].max() > threshold_train, fail_msg
        assert df_valid.loc[:, key].max() > threshold_valid, fail_msg
    elif direction == "min":
        if check_train and key in df_train.columns:
            assert df_train.loc[:, key].min() < threshold_train, fail_msg
        assert df_valid.loc[:, key].min() < threshold_valid, fail_msg
