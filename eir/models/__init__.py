from typing import Union

from torch.nn import DataParallel

from eir.models.fusion import FusionModel
from eir.models.fusion_linear import LinearFusionModel
from eir.models.fusion_mgmoe import MGMoEModel

al_fusion_models = Union[FusionModel, LinearFusionModel, MGMoEModel, DataParallel]
