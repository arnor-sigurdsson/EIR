from typing import Union

from eir.models.meta.meta import MetaModel
from eir.train_utils.distributed import AttrDelegatedDistributedDataParallel
from eir.train_utils.optim import AttrDelegatedSWAWrapper

al_meta_model = Union[
    MetaModel, AttrDelegatedDistributedDataParallel, AttrDelegatedSWAWrapper
]
