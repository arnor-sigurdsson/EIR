from typing import Union, Type, Dict

from snp_pred.models.models_cnn import CNNModel
from snp_pred.models.models_linear import LinearModel
from snp_pred.models.models_mgmoe import MGMoEModel
from snp_pred.models.models_mlp import MLPModel
from snp_pred.models.models_split_mlp import SplitMLPModel

al_models_classes = Union[
    Type["CNNModel"],
    Type["MLPModel"],
    Type["LinearModel"],
    Type["SplitMLPModel"],
    Type["MGMoEModel"],
]


def _get_model_mapping() -> Dict[str, al_models_classes]:
    mapping = {
        "cnn": CNNModel,
        "mlp": MLPModel,
        "mlp-split": SplitMLPModel,
        "mlp-mgmoe": MGMoEModel,
        "linear": LinearModel,
    }

    return mapping


def get_model_class(model_type: str) -> al_models_classes:
    mapping = _get_model_mapping()
    return mapping[model_type]


al_models = Union["CNNModel", "MLPModel", "LinearModel", "SplitMLPModel", "MGMoEModel"]
