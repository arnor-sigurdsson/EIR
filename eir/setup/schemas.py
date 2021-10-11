from dataclasses import dataclass, field
from typing import Union, Literal, List, Optional, Sequence, Type

from eir.models.fusion import FusionModelConfig
from eir.models.fusion_linear import LinearFusionModelConfig
from eir.models.fusion_mgmoe import MGMoEModelConfig
from eir.models.omics.omics_models import (
    LinearModel,
    CNNModel,
    LCLModel,
    SimpleLCLModel,
    IdentityModel,
    CNNModelConfig,
    LCLModelConfig,
    LinearModelConfig,
    SimpleLCLModelConfig,
    IdentityModelConfig,
    Dataclass,
)
from eir.models.tabular.tabular import SimpleTabularModel, TabularModelConfig

al_input_configs = Sequence["InputConfig"]


al_model_configs = Union[
    Type[FusionModelConfig],
    Type[MGMoEModelConfig],
    Type[CNNModelConfig],
    Type[LinearModelConfig],
    Type[SimpleLCLModelConfig],
    Type[LCLModelConfig],
    Type[TabularModelConfig],
    Type[IdentityModelConfig],
    Type[Dataclass],
]

al_models_classes = Union[
    Type[CNNModel],
    Type[LinearModel],
    Type[LCLModel],
    Type[SimpleLCLModel],
    Type[SimpleTabularModel],
    Type[IdentityModel],
]

al_tokenizer_choices = (
    Union[
        Literal["basic_english"],
        Literal["spacy"],
        Literal["moses"],
        Literal["toktok"],
        Literal["revtok"],
        Literal["subword"],
        None,
    ],
)

al_max_sequence_length = Union[int, Literal["max", "average"]]


@dataclass
class GlobalConfig:
    """
    Global configurations that are common / relevant for the whole experiment to run.

    :param run_name:
        What to name the experiment and output folder where results are saved.

    :param n_epochs:
        Number of epochs for training.

    :param batch_size:
        Size of batches during training.

    :param valid_size:
        Size if the validaton set, if float then uses a percentage. If int,
        then raw counts.

    :param dataloader_workers:
        Number of workers for multi-process training and validation data loading.

    :param device:
        Device to run the training on (i.e. GPU / CPU).

    :param gpu_num:
        Which GPU to run (according to CUDA order).

    :param weighted_sampling_columns:
        Target column to apply weighted sampling on. Only applies to categorical
        columns. Passing in 'all' here will use an average of all the target columns.

    :param lr:
        Base learning rate for optimizer.

    :param lr_lb:
        Lower bound for learning rate when using LR scheduling

    :param find_lr:
        Whether to perform a range test of different learning rates, with
        the lower limit being what is passed in for the --lr flag.
        Produces a plot and exits with status 0 before training if this flag
        is active.

    :param lr_schedule:
        Whether to use cyclical, cosine or reduce on plateau learning rate
        schedule. Otherwise keeps same learning rate

    :param lr_plateau_patience:
        Number of validation performance steps without improvement over
        best performance before reducing LR (only relevant when --lr_schedule is
        'plateau'.

    :param lr_plateau_factor:
        Factor to reduce LR when running with plateau schedule.

    :param early_stopping_patience:
        Number of validation performance steps without improvement over
        best performance before terminating run.

    :param early_stopping_buffer:
        Number of iterations to run before activation early stopping checks,
        useful if networks take a while to 'kick into gear'.

    :param warmup_steps:
        How many steps to use in warmup. If not set, will automatically compute the
        number of steps if using an adaptive optimizer, otherwise use 2000.

    :param optimizer:
        What optimizer to use.

    :param b1:
        Decay of first order momentum of gradient for relevant optimizers.

    :param b2:
        Decay of second order momentum of gradient for relevant optimizers.

    :param wd:
        Weight decay.

    :param memory_dataset:
        Whether to load all sample into memory during training.

    :param sample_interval:
        Iteration interval to perform validation and possibly activation analysis if
        set.

    :param checkpoint_interval:
        Iteration interval to checkpoint (i.e. save) model.

    :param n_saved_models:
        Number of top N models to saved during training.

    :param multi_gpu:
        Whether to run the training on multiple GPUs for the current node.

    :param get_acts:
        Whether to compute activations w.r.t. inputs.

    :param max_acts_per_class:
        Maximum number of samples per class to gather for activation analysis.
        Good to use when modelling on imbalanced data.

    :param act_every_sample_factor:
        Controls whether the activations are computed at every sample interval
        (=1), every other sample interval (=2), etc. Useful when computing the
        activations takes a long time and we don't want to do it every time we
        evaluate.

    :param act_background_samples:
        Number of samples to use for the background in activation computations.

    :param debug:
        Whether to run in debug mode.

    :param no_pbar:
        Whether to not use progress bars. Useful when stdout/stderr is written
        to files.

    :param mixing_alpha:
        Alpha parameter used for mixing (higher means more mixing).

    :param mixing_type:
        Type of mixing to apply when using mixup and similar approaches.

    :param plot_skip_steps:
        How many iterations to skip in in plots.
    """

    run_name: str
    n_epochs: int = 10
    batch_size: int = 64
    valid_size: Union[float, int] = 0.1
    dataloader_workers: int = 0
    device: str = "cpu"
    gpu_num: str = "0"
    weighted_sampling_columns: Union[None, Sequence[str]] = None
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
    b2: float = 0.999
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
    act_background_samples: int = 64
    debug: bool = False
    no_pbar: bool = False
    mixing_alpha: float = 0.0
    mixing_type: Union[None, Literal["mixup", "cutmix-block", "cutmix-uniform"]] = None
    plot_skip_steps: int = 200


@dataclass
class PredictorConfig:
    """
    :param model_type:
        Which type of fusion model to use.

    :param model_config:
        Predictor model configuration.
    """

    model_type: Literal["default", "linear", "mgmoe"]
    model_config: Union[FusionModelConfig, LinearFusionModelConfig, MGMoEModelConfig]


@dataclass
class InputConfig:
    """
    :param input_info:
        Information about the input source, name and type.

    :param input_type_info:
       Information specific to the input type, e.g. some augmentations are only relevant
       for omics input. Another example is the type of model to apply to the input.

    :param model_config:
        Configuration for the chosen model (i.e. feature extractor) for this input.

    :param interpretation_config:
        Configuration for interpretation analysis when applicable.

    """

    input_info: "InputDataConfig"
    input_type_info: Union[
        "OmicsInputDataConfig", "TabularInputDataConfig", "SequenceInputDataConfig"
    ]
    model_config: al_model_configs
    interpretation_config: Union[None, "SequenceInterpretationConfig"] = None


@dataclass
class InputDataConfig:
    """
    :param input_source:
        Where on the filesystem to locate the input.

    :param input_name:
        Name to identify the input.

    :param input_type:
        Type of the input.
    """

    input_source: str
    input_name: str
    input_type: Literal["omics", "tabular", "sequence"]


@dataclass
class OmicsInputDataConfig:
    """
    :param snp_file:
        Path to the relevant ``.bim`` file, used for activation analysis.

    :param na_augment_perc:
        Percentage of the input (i.e. percentage of SNPs) to augment by setting the
        SNPs to 'missing' (i.e. ``[0, 0, 0, 1]`` in one-hot encoding).

    :param na_augment_prob:
        Probability of applying NA augmentation to a given sample.

    :param model_type:
        Type of omics feature extractor to use.

    :param omics_format:
        Currently unsupported (i.e. does nothing), which format the omics data is in.
    """

    snp_file: Optional[str] = None
    na_augment_perc: float = 0.0
    na_augment_prob: float = 0.0
    model_type: Literal[
        "cnn", "linear", "mlp-split", "genome-local-net", "linear"
    ] = "gln"
    omics_format: Literal["one-hot"] = "one-hot"


@dataclass
class TabularInputDataConfig:
    """
    :param model_type:
        Type of tabular model to use. Currently only one type ("tabular") is supported.

    :param extra_cat_columns:
        Which columns to use as a categorical inputs from the ``input_source`` specified
        in the ``input_info`` field of the relevant ``.yaml``.

    :param extra_con_columns:
        Which columns to use as a continuous inputs from the ``input_source`` specified
        in the ``input_info`` field of the relevant ``.yaml``.

    :param label_parsing_chunk_size:
        Number of rows to process at time when loading in the ``input_source``. Useful
        when RAM is limited.
    """

    model_type: Literal["tabular"] = "tabular"
    extra_cat_columns: Sequence[str] = field(default_factory=list)
    extra_con_columns: Sequence[str] = field(default_factory=list)
    label_parsing_chunk_size: Union[None, int] = None


@dataclass
class SequenceInputDataConfig:
    """
    :param model_type:
        Type of sequence model to use. Currently only one type ("sequence-default") is
        supported, which is a vanilla transformer model.

    :param vocab_file:
        An optional text file containing pre-defined vocabulary to use
        for the training. If this is not passed in, the framework will automatically
        build the vocabulary from the training data. Passing in a vocabulary file is
        therefore useful if (a) you want to manually specify / limit the vocabulary used
        and/or (b) you want to save time by pre-computing the vocabulary.

    :param max_length:
        Maximum length to truncate/pad sequences to. This can be an integer or the
        values 'max' or 'average'. The 'max' keyword will use the maximum sequence
        length found in the training data, while the 'average' will use the average
        length across all training samples.

    :param sampling_strategy_if_longer:
        Controls how sequences are truncated if they are longer than the specified
        ``max_length`` parameter. Using 'from_start' will always truncate from the
        beginning of the sequence, ensuring the the samples will always be the same
        during training. Setting this parameter to ``uniform`` will uniformly sample
        a slice of a given sample sequence during training. Note that for consistency,
        the validation/test set samples always use the ``from_start`` setting when
        truncating.

    :param min_freq:
        Minimum number of times a token must appear in the total training data to be
        included in the vocabulary. Note that this setting will not do anything if
        passing in ``vocab_file``.


    :param split_on:
        Which token to split the sequence on to generate separate tokens for the
        vocabulary.

    :param tokenizer:
        Which tokenizer to use. Relevant if modelling on language, but not as much when
        doing it on other arbitrary sequences.

    :param tokenizer_language:
        Which language rules the tokenizer should apply when tokenizing the raw data.
    """

    model_type: Literal["sequence-default"] = "sequence-default"
    vocab_file: Union[None, str] = None
    max_length: al_max_sequence_length = "average"
    sampling_strategy_if_longer: Literal["from_start", "uniform"] = "uniform"
    min_freq: int = 10
    split_on: str = " "
    tokenizer: al_tokenizer_choices = None
    tokenizer_language: Union[str, None] = None


@dataclass
class SequenceInterpretationConfig:
    """
    :param interpretation_sampling_strategy:
        How to sample sequences for activation analysis. `first_n` always grabs the
        same first n values from the beginning of the dataset to interpret, while
        `random_sample` will sample uniformly from the whole dataset without
        replacement.

    :param num_samples_to_interpret:
        How many samples to interpret.

    :param manual_samples_to_interpret:
        IDs of samples to always interpret, irrespective of
        `interpretation_sampling_strategy` and `num_samples_to_interpret`. A caveat
        here is that they must be present in the dataset that is being interpreted
        (e.g. validation / test dataset), meaning that adding IDs here that happen to
        be in the training dataset will not work.
    """

    interpretation_sampling_strategy: Literal["first_n", "random_sample"] = "first_n"
    num_samples_to_interpret: int = 10
    manual_samples_to_interpret: Union[Sequence[str], None] = None


@dataclass
class TargetConfig:
    """
    :param label_file:
        Label ``.csv`` file to load targets from.

    :param label_parsing_chunk_size:
        Number of rows to process at time when loading in the ``input_source``. Useful
        when RAM is limited.

    :param target_cat_columns:
        Which columns from ``label_file`` to use as categorical targets.

    :param target_con_columns:
        Which columns from ``label_file`` to use as continuous targets.
    """

    label_file: str
    label_parsing_chunk_size: Union[None, int] = None
    target_cat_columns: Sequence[str] = field(default_factory=list)
    target_con_columns: Sequence[str] = field(default_factory=list)
