from dataclasses import dataclass, field
from typing import Union, Literal, List, Optional, Sequence, Type

from eir.models.fusion import FusionModelConfig
from eir.models.fusion_mgmoe import MGMoEModelConfig
from eir.models.omics.omics_models import (
    MLPModel,
    CNNModel,
    LCLModel,
    SimpleLCLModel,
    IdentityModel,
    CNNModelConfig,
    LCLModelConfig,
    MLPModelConfig,
    SimpleLCLModelConfig,
    IdentityModelConfig,
)
from eir.models.tabular.tabular import SimpleTabularModel, TabularModelConfig

al_input_configs = Sequence["InputConfig"]


al_model_configs = [
    Type[FusionModelConfig],
    Type[MGMoEModelConfig],
    Type[CNNModelConfig],
    Type[MLPModelConfig],
    Type[SimpleLCLModelConfig],
    Type[LCLModelConfig],
    Type[TabularModelConfig],
    Type[IdentityModelConfig],
]

al_models_classes = Union[
    Type[CNNModel],
    Type[MLPModel],
    Type[LCLModel],
    Type[SimpleLCLModel],
    Type[SimpleTabularModel],
    Type[IdentityModel],
]


@dataclass
class GlobalConfig:
    run_name: str
    n_epochs: int = 10
    batch_size: int = 64
    valid_size: Union[float, int] = 0.1
    dataloader_workers: int = 0
    device: str = "cpu"
    gpu_num: str = "0"
    weighted_sampling_column: Union[None, str] = None
    lr: float = 1e-03
    lr_lb: float = 0.0
    find_lr: bool = False
    lr_schedule: Literal["cycle", "plateau", "same", "cosine"] = "plateau"
    lr_plateau_patience: int = 10
    lr_plateau_factor: float = 0.2
    early_stopping_patience: int = 10
    early_stopping_buffer: Union[None, int] = None
    warmup_steps: Union[Literal["auto"], int] = "auto"
    optimizer: str = "adam"
    b1: float = 0.9
    b2: float = 0.99
    wd: float = 1e-04
    memory_dataset: bool = False
    sample_interval: int = 200
    checkpoint_interval: Union[None, int] = None
    n_saved_models: int = 1
    multi_gpu: bool = False
    get_acts: bool = False
    act_classes: Union[None, List[str]] = None
    max_acts_per_class: Union[None, int] = None
    act_every_sample_factor: int = 1
    debug: bool = False
    no_pbar: bool = False
    mixing_alpha: float = 0.0
    mixing_type: Union[None, Literal["mixup", "cutmix-block", "cutmix-uniform"]] = None
    plot_skip_steps: int = 200


@dataclass
class PredictorConfig:
    model_type: str
    model_config: Union[FusionModelConfig, MGMoEModelConfig]


@dataclass
class InputConfig:
    input_info: "InputDataConfig"
    input_type_info: Union["OmicsInputDataConfig", "TabularInputDataConfig"]
    model_config: al_model_configs


@dataclass
class InputDataConfig:
    input_source: str
    input_name: str
    input_type: str


@dataclass
class OmicsInputDataConfig:
    snp_file: Optional[str] = None
    na_augment_perc: float = 0.0
    na_augment_prob: float = 0.0
    model_type: Literal["cnn", "mlp", "mlp-split", "genome-local-net", "linear"] = "gln"
    omics_format: Literal["one-hot"] = "one-hot"


@dataclass
class TabularInputDataConfig:
    model_type: Literal["tabular"] = "tabular"
    extra_cat_columns: Sequence[str] = field(default_factory=list)
    extra_con_columns: Sequence[str] = field(default_factory=list)
    label_parsing_chunk_size: Union[None, int] = None


@dataclass
class TargetConfig:
    label_file: str
    label_parsing_chunk_size: Union[None, int] = None
    target_cat_columns: Sequence[str] = field(default_factory=list)
    target_con_columns: Sequence[str] = field(default_factory=list)
