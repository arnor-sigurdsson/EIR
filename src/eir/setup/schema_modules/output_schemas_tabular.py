from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

al_cat_loss_names = Literal["CrossEntropyLoss", "BCEWithLogitsLoss"]
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

    :param cat_loss_name:
        Loss function to use for categorical targets. If using ``BCEWithLogitsLoss``,
        the targets should all be binary.

    :param con_loss_name:
        Loss function to use for continuous targets.

    :param cat_loss_class_balanced:
        Whether to use class-balanced weighting for categorical loss. Uses the
        effective number of samples (Cui et al., 2019) to compute per-class weights,
        helping with imbalanced targets. Only applicable with ``BCEWithLogitsLoss``.

    :param uncertainty_weighted_mt_loss:
        Whether to use uncertainty weighted loss for multitask / multilabel learning.
    """

    target_cat_columns: Sequence[str] = field(default_factory=list)
    target_con_columns: Sequence[str] = field(default_factory=list)
    label_parsing_chunk_size: None | int = None
    cat_label_smoothing: float = 0.0
    cat_loss_name: al_cat_loss_names = "CrossEntropyLoss"
    con_loss_name: al_con_loss_names = "MSELoss"
    cat_loss_class_balanced: bool = False
    uncertainty_weighted_mt_loss: bool = False

    def __post_init__(self) -> None:
        if self.cat_loss_class_balanced and self.cat_loss_name != "BCEWithLogitsLoss":
            raise ValueError(
                "cat_loss_class_balanced is only supported with BCEWithLogitsLoss. "
                f"Got cat_loss_name='{self.cat_loss_name}'."
            )
