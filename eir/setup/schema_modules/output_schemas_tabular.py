from dataclasses import dataclass, field
from typing import Literal, Sequence, Union

al_cat_loss_names = Literal["CrossEntropyLoss"]
al_con_loss_names = Literal[
    "MSELoss",
    "L1Loss",
    "SmoothL1Loss",
    "PoissonNLLLoss",
    "HuberLoss",
]


@dataclass
class TabularOutputTypeConfig:
    """
    :param target_cat_columns:
        Which columns from ``label_file`` to use as categorical targets.

    :param target_con_columns:
        Which columns from ``label_file`` to use as continuous targets.

    :param label_parsing_chunk_size:
        Number of rows to process at time when loading in the ``input_source``. Useful
        when RAM is limited.

    :param cat_label_smoothing:
        Label smoothing to apply to categorical targets.

    :param uncertainty_weighted_mt_loss:
        Whether to use uncertainty weighted loss for multitask / multilabel learning.
    """

    target_cat_columns: Sequence[str] = field(default_factory=list)
    target_con_columns: Sequence[str] = field(default_factory=list)
    label_parsing_chunk_size: Union[None, int] = None
    cat_label_smoothing: float = 0.0
    cat_loss_name: al_cat_loss_names = "CrossEntropyLoss"
    con_loss_name: al_con_loss_names = "MSELoss"
    uncertainty_weighted_mt_loss: bool = True
