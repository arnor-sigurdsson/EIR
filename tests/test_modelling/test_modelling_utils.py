from pathlib import Path
from typing import Tuple

import pandas as pd


def check_test_performance_results(
    run_path: Path, target_column: str, metric: str, thresholds: Tuple[float, float]
):
    target_column_results_folder = run_path / "results" / target_column
    train_history_path = (
        target_column_results_folder / f"train_{target_column}_history.log"
    )
    valid_history_path = (
        target_column_results_folder / f"validation_{target_column}_history.log"
    )

    threshold_train, threshold_valid = thresholds

    df_train = pd.read_csv(train_history_path)
    assert df_train.loc[:, f"{target_column}_{metric}"].max() > threshold_train

    df_valid = pd.read_csv(valid_history_path)
    assert df_valid.loc[:, f"{target_column}_{metric}"].max() > threshold_valid
