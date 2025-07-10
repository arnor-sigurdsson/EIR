import base64
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


class StockDataManager:
    def __init__(
        self,
        *,
        csv_path: Path,
        sequence_length: int = 64,
        is_diffusion: bool = False,
    ):
        self.csv_path = csv_path
        self.sequence_length = sequence_length
        self.is_diffusion = is_diffusion
        self.current_index = 0

        logger.info(f"Loading stock data from {csv_path}")

        df = pd.read_csv(filepath_or_buffer=csv_path)
        self.df = df.sample(frac=1).reset_index(drop=True)

        logger.info(f"Loaded {len(self.df)} samples")

    def reset(self):
        self.current_index = 0
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        logger.info("Data manager reset")

    def get_batch(self, *, batch_size: int) -> list[dict[str, Any]]:
        if self.current_index >= len(self.df):
            self.reset()

        batch = []
        for _ in range(batch_size):
            if self.current_index >= len(self.df):
                self.reset()

            row = self.df.iloc[self.current_index]

            input_array = np.array(
                [int(x) for x in row["InputSequence"].split(" ")], dtype=np.float32
            )
            output_array = np.array(
                [int(x) for x in row["OutputSequence"].split(" ")], dtype=np.float32
            )

            input_encoded = base64.b64encode(input_array.tobytes()).decode("utf-8")
            output_encoded = base64.b64encode(output_array.tobytes()).decode("utf-8")
            sample = {
                "inputs": {"stock_input": input_encoded},
                "target_labels": {"stock_output": {"stock_output": output_encoded}},
                "sample_id": row["ID"],
            }

            if self.is_diffusion:
                sample["inputs"]["stock_output"] = output_encoded

            batch.append(sample)
            self.current_index += 1

        return batch

    def get_status(self) -> dict[str, Any]:
        return {
            "current_index": self.current_index,
            "total_samples": len(self.df),
        }


def create_stock_manager(
    *, sequence_length: int = 64, is_diffusion: bool = False
) -> StockDataManager:
    csv_path = Path(
        "eir_tutorials/g_time_series/02_time_series_stocks/data"
        "/stock_combined_train.csv"
    )

    logger.info(
        f"Creating StockDataManager with sequence_length={sequence_length}, "
        f"is_diffusion={is_diffusion}"
    )

    return StockDataManager(
        csv_path=csv_path,
        sequence_length=sequence_length,
        is_diffusion=is_diffusion,
    )
