from dataclasses import dataclass, field
from typing import Literal

SurvivalLossNames = Literal["NegativeLogLikelihood", "CoxPHLoss"]


@dataclass
class SurvivalOutputTypeConfig:
    """
    Basic configuration for survival analysis output.

    :param time_columns:
        The names of the columns in the label file that contain the time-to-event or
        censoring time. Each time column should be aligned with the corresponding
        event column at the same index in ``event_columns``.

    :param event_columns:
        The names of the columns in the label file that indicate whether an event
        occurred (``1``) or the observation was censored (``0``). Each event column
        should be aligned with the corresponding time column at the same index in
        ``time_columns``.

    :param num_durations:
        The number of discrete time intervals to use in the model. This determines the
        size of the output layer per event.

    :param loss_function:
        The loss function to use for training the survival model.

    :param max_duration:
        The maximum duration to consider. Times beyond this will be censored at this
        point. If None, use the maximum observed time.

    :param label_parsing_chunk_size:
        Number of rows to process at a time when loading the input_source.
        Useful when RAM is limited.
    """

    time_columns: list[str] = field(default_factory=list)
    event_columns: list[str] = field(default_factory=list)
    num_durations: int = 10
    loss_function: SurvivalLossNames = "NegativeLogLikelihood"
    max_duration: None | float = None
    label_parsing_chunk_size: None | int = None

    def __post_init__(self) -> None:
        if len(self.time_columns) != len(self.event_columns):
            raise ValueError(
                f"time_columns and event_columns must have the same length, "
                f"got {len(self.time_columns)} and {len(self.event_columns)}."
            )

    @property
    def time_column(self) -> str:
        if len(self.time_columns) != 1:
            raise ValueError(
                "time_column property is only valid when there is exactly one "
                f"time column, got {len(self.time_columns)}."
            )
        return self.time_columns[0]

    @property
    def event_column(self) -> str:
        if len(self.event_columns) != 1:
            raise ValueError(
                "event_column property is only valid when there is exactly one "
                f"event column, got {len(self.event_columns)}."
            )
        return self.event_columns[0]

    def get_time_column_for_event(self, event_column: str) -> str:
        idx = self.event_columns.index(event_column)
        return self.time_columns[idx]
