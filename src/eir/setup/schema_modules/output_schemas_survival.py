from dataclasses import dataclass
from typing import Literal

SurvivalLossNames = Literal["NegativeLogLikelihood", "CoxPHLoss"]


@dataclass
class SurvivalOutputTypeConfig:
    """
    Basic configuration for survival analysis output.

    :param time_column:
        The name of the column in the label file that contains the time-to-event or
        censoring time.

    :param event_column:
        The name of the column in the label file that indicates whether an event
        occurred (``1``) or the observation was censored (``0``).

    :param num_durations:
        The number of discrete time intervals to use in the model. This determines the
        size of the output layer.

    :param loss_function:
        The loss function to use for training the survival model.

    :param max_duration:
        The maximum duration to consider. Times beyond this will be censored at this
        point. If None, use the maximum observed time.

    :param label_parsing_chunk_size:
        Number of rows to process at a time when loading the input_source.
        Useful when RAM is limited.
    """

    time_column: str
    event_column: str
    num_durations: int = 10
    loss_function: SurvivalLossNames = "NegativeLogLikelihood"
    max_duration: None | float = None
    label_parsing_chunk_size: None | int = None
