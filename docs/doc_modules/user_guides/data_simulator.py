import os
from typing import Any

from docs.doc_modules.user_guides.single_sample_simulation import simulate_health_sample
from eir.utils.logging import get_logger

logger = get_logger(__name__)


class DataSimulator:
    def __init__(
        self,
        sequence_length: int = 64,
        max_iterations: int | None = None,
    ):
        self.sequence_length = sequence_length
        self.max_iterations = max_iterations
        self.position = 0

        logger.info(f"Using sequence_length={self.sequence_length}")
        if self.max_iterations is not None:
            logger.info(f"Will terminate after {self.max_iterations} iterations")

    def reset(self):
        self.position = 0
        if self.max_iterations is not None:
            logger.info(f"Reset: Will terminate after {self.max_iterations} iterations")

    def get_batch(self, batch_size: int) -> list[dict[str, Any]]:
        if self.max_iterations is not None and self.position >= self.max_iterations:
            logger.info(f"Reached max iterations ({self.max_iterations}), terminating")
            return []

        batch = []
        for _ in range(batch_size):
            if self.max_iterations is not None and self.position >= self.max_iterations:
                break

            text_input, text_sequence = simulate_health_sample(sequence_length=64)
            sample_id = f"sample_{self.position}"

            batch.append(
                {
                    "inputs": {
                        "text_output": text_sequence,
                        "text_input": text_input,
                    },
                    "target_labels": {"text_output": {"text_output": text_sequence}},
                    "sample_id": sample_id,
                }
            )

            self.position += 1

        return batch

    def get_status(self) -> dict[str, Any]:
        status_data = {
            "current_position": self.position,
        }

        if self.max_iterations is not None:
            status_data["max_iterations"] = self.max_iterations
            status_data["remaining_iterations"] = max(
                0, self.max_iterations - self.position
            )

        return status_data


def create_simulator() -> DataSimulator:
    sequence_length = int(os.getenv("SEQUENCE_LENGTH", "512"))

    max_iterations_str = os.getenv("MAX_ITERATIONS")
    max_iterations = int(max_iterations_str) if max_iterations_str else None

    logger.info(
        f"Creating DataSimulator with sequence_length={sequence_length}, "
        f"max_iterations={max_iterations}"
    )

    return DataSimulator(
        sequence_length=sequence_length,
        max_iterations=max_iterations,
    )
