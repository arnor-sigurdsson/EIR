from typing import Union, Type, Dict

from snp_pred.models.fusion_mgmoe import MGMoEModel
from snp_pred.models.models_cnn import CNNModel
from snp_pred.models.models_mlp import MLPModel
from snp_pred.models.models_split_mlp import SplitMLPModel, FullySplitMLPModel

al_models_classes = Union[
    Type["CNNModel"],
    Type["MLPModel"],
    Type["SplitMLPModel"],
    Type["FullySplitMLPModel"],
    Type["MGMoEModel"],
]

al_models = Union[
    "CNNModel",
    "MLPModel",
    "SplitMLPModel",
    "FullySplitMLPModel",
    "MGMoEModel",
]


def _get_model_mapping() -> Dict[str, al_models_classes]:
    mapping = {
        "cnn": CNNModel,
        "mlp": MLPModel,
        "mlp-split": SplitMLPModel,
        "mlp-fully-split": FullySplitMLPModel,
        "mlp-mgmoe": MGMoEModel,
    }

    return mapping


def get_model_class(model_type: str) -> al_models_classes:
    mapping = _get_model_mapping()
    return mapping[model_type]
