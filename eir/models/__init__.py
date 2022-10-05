from typing import Union

from torch.nn import DataParallel

from eir.models.fusion.fusion_identity import IdentityFusionModel
from eir.models.fusion.fusion_mgmoe import MGMoEModel
from eir.models.meta.meta import MetaModel

al_fusion_models = Union[MetaModel, IdentityFusionModel, MGMoEModel, DataParallel]
