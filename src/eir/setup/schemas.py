from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Union

from eir.models.fusion.fusion_attention import AttentionFusionConfig
from eir.models.fusion.fusion_identity import IdentityConfig
from eir.models.fusion.fusion_mgmoe import MGMoEModelConfig
from eir.models.input.array.array_models import ArrayModelConfig
from eir.models.input.image.image_models import ImageModelConfig
from eir.models.input.omics.omics_models import (
    CNNModel,
    IdentityModel,
    LCLModel,
    LinearModel,
    OmicsModelConfig,
    SimpleLCLModel,
)
from eir.models.input.sequence.transformer_models import SequenceModelConfig
from eir.models.input.tabular.tabular import SimpleTabularModel, TabularModelConfig
from eir.models.layers.mlp_layers import ResidualMLPConfig
from eir.models.output.array.array_output_modules import ArrayOutputModuleConfig
from eir.models.output.sequence.sequence_output_modules import (
    SequenceOutputModuleConfig,
)
from eir.models.output.tabular.tabular_output_modules import TabularOutputModuleConfig
from eir.setup.schema_modules.latent_analysis_schemas import LatentSamplingConfig
from eir.setup.schema_modules.output_schemas_array import (
    ArrayOutputSamplingConfig,
    ArrayOutputTypeConfig,
)
from eir.setup.schema_modules.output_schemas_image import (
    ImageOutputSamplingConfig,
    ImageOutputTypeConfig,
)
from eir.setup.schema_modules.output_schemas_sequence import (
    SequenceOutputSamplingConfig,
    SequenceOutputTypeConfig,
)
from eir.setup.schema_modules.output_schemas_survival import SurvivalOutputTypeConfig
from eir.setup.schema_modules.output_schemas_tabular import TabularOutputTypeConfig
from eir.setup.schema_modules.tensor_broker_schemas import TensorBrokerConfig

al_input_configs = Sequence["InputConfig"]
al_output_configs = Sequence["OutputConfig"]

al_input_type_info = Union[
    "OmicsInputDataConfig",
    "TabularInputDataConfig",
    "SequenceInputDataConfig",
    "ByteInputDataConfig",
    "ImageInputDataConfig",
    "ArrayInputDataConfig",
]

al_base_optimizer_choices = Literal[
    "sgdm",
    "adam",
    "adamw",
    "adahessian",
    "adabelief",
    "adabeliefw",
]


al_cat_metric_choices = Sequence[
    Literal[
        "mcc",
        "acc",
        "roc-auc-macro",
        "ap-macro",
        "f1-macro",
        "precision-macro",
        "recall-macro",
        "cohen-kappa",
    ]
]
al_con_metric_choices = Sequence[
    Literal[
        "r2",
        "pcc",
        "loss",
        "rmse",
        "mae",
        "mape",
        "explained-variance",
    ]
]

all_survival_metric_choices = Sequence[
    Literal[
        "c-index",
        "ibs",
        "td-cindex",
        "td-auc",
    ]
]

al_feature_extractor_configs = (
    OmicsModelConfig
    | TabularModelConfig
    | ImageModelConfig
    | SequenceModelConfig
    | ArrayModelConfig
)

al_feature_extractor_configs_classes = (
    type[OmicsModelConfig]
    | type[TabularModelConfig]
    | type[ImageModelConfig]
    | type[SequenceModelConfig]
    | type[ArrayModelConfig]
)

al_models_classes = (
    type[CNNModel]
    | type[LinearModel]
    | type[LCLModel]
    | type[SimpleLCLModel]
    | type[SimpleTabularModel]
    | type[IdentityModel]
)


al_output_module_configs_classes = (
    type[TabularOutputModuleConfig]
    | type[SequenceOutputModuleConfig]
    | type[ArrayOutputModuleConfig]
)

al_output_type_configs = (
    SequenceOutputTypeConfig
    | TabularOutputTypeConfig
    | ArrayOutputTypeConfig
    | ImageOutputTypeConfig
    | SurvivalOutputTypeConfig
)

al_output_module_configs = (
    TabularOutputModuleConfig | SequenceOutputModuleConfig | ArrayOutputModuleConfig
)


al_tokenizer_choices = (
    Literal["basic_english"]
    | Literal["basic"]
    | Literal["spacy"]
    | Literal["moses"]
    | Literal["toktok"]
    | Literal["revtok"]
    | Literal["subword"]
    | Literal["bpe"]
    | str
    | None
)

al_max_sequence_length = int | Literal["max", "average"]


@dataclass
class BasicExperimentConfig:
    """
    :param output_folder:
        What to name the experiment and output folder where results are saved.

    :param n_epochs:
        Number of epochs for training.

    :param batch_size:
        Size of batches during training.

    :param valid_size:
        Size if the validation set, if float then uses a percentage. If int, then raw
        counts.

    :param manual_valid_ids_file:
        File with IDs of those samples to manually use as the validation set.
        Should be one ID per line in the file.

    :param dataloader_workers:
        Number of workers for multiprocess training and validation data loading.

    :param device:
        DEPRECATED: Use ``AcceleratorConfig.hardware`` instead.

    :param memory_dataset:
        Whether to load all sample into memory during training.
    """

    output_folder: str
    n_epochs: int = 10
    batch_size: int = 64
    valid_size: float | int = 0.1
    manual_valid_ids_file: str | None = None
    dataloader_workers: int = 0
    device: str = "cpu"
    memory_dataset: bool = False


@dataclass
class GlobalModelConfig:
    """
    :param compile_model:
        Whether to compile the model before training. This can be useful to speed
        up training, but may not work for all models.

    :param n_iter_before_swa:
        Number of iterations to run before activating Stochastic Weight Averaging (SWA).

    :param pretrained_checkpoint:
        Path to a pretrained checkpoint model file (under saved_models/ in the
        experiment output folder) to load and use as a starting point for training.

    :param strict_pretrained_loading:
        Whether to enforce that the loaded pretrained model exactly the same
        architecture as the current model. If ``False``, will only load the layers that
        match between the two models.
    """

    compile_model: bool = False
    n_iter_before_swa: None | int = None
    pretrained_checkpoint: None | str = None
    strict_pretrained_loading: bool = True


@dataclass
class OptimizationConfig:
    """
    :param optimizer:
        What optimizer to use.

    :param lr:
        Base learning rate for optimizer.

    :param lr_lb:
        Lower bound for learning rate when using LR scheduling

    :param b1:
        Decay of first order momentum of gradient for relevant optimizers.

    :param b2:
        Decay of second order momentum of gradient for relevant optimizers.

    :param wd:
        Weight decay.

    :param gradient_clipping:
        Max norm used for gradient clipping, with ``p=2``.

    :param gradient_accumulation_steps:
        Number of steps to use for gradient accumulation.

    :param gradient_noise:
        Gradient noise to inject during training.
    """

    optimizer: al_base_optimizer_choices = "adamw"
    lr: float = 3e-04
    lr_lb: float = 0.0
    b1: float = 0.9
    b2: float = 0.999
    wd: float = 1e-04
    gradient_clipping: float = 1.0
    gradient_accumulation_steps: None | int = None
    gradient_noise: float = 0.0


@dataclass
class LRScheduleConfig:
    """
    :param lr_schedule:
        Whether to use "same" "cyclical", "cosine" or "reduce on plateau" learning
        rate schedule. The "reduce on plateau" schedule will reduce the learning rate
        when the validation performance does not improve for a number of steps.

    :param lr_plateau_patience:
        Number of validation performance steps without improvement over the best
        performance before reducing LR (only relevant when ``lr_schedule`` is
        ``"plateau"``.

    :param lr_plateau_factor:
        Factor to reduce LR when running with a plateau schedule.

    :param warmup_steps:
        How many steps to use in warmup. If not set, will automatically compute the
        number of steps if using an adaptive optimizer, otherwise use 2000.

    :param plot_lr_schedule:
        Whether to plot the learning rate schedule expected during training, useful
        to e.g. visualize the effect of warmup and "cosine" / "cyclical" schedules.
        Not relevant when using the "plateau" schedule, since the LR is not fixed
        beforehand.
    """

    lr_schedule: Literal["cycle", "plateau", "same", "cosine"] = "plateau"
    lr_plateau_patience: int = 10
    lr_plateau_factor: float = 0.2
    warmup_steps: Literal["auto"] | int = "auto"
    plot_lr_schedule: bool = False


@dataclass
class TrainingControlConfig:
    """
    :param early_stopping_patience:
        Number of validation performance steps without improvement over the best
        performance before terminating the run.

    :param early_stopping_buffer:
        Number of iterations to run before activating early stopping checks, useful
        if networks take a while to 'kick into gear'.

    :param weighted_sampling_columns:
        Target column to apply weighted sampling on, relevant for supervised
        (tabular) targets.
        Only applies to categorical columns.
        Passing in ``- 'all'`` (note this is still passed in as an array, just
        containing a single string) here will use an average of all the
        categorical target columns to compute the weights.

    :param mixing_alpha:
        Alpha parameter used for mixing (higher means more mixing).
        See `Mixup: Beyond Empirical
        Risk Minimization <https://arxiv.org/abs/1710.09412>`_ for details.
    """

    early_stopping_patience: int = 10
    early_stopping_buffer: None | int = None
    weighted_sampling_columns: None | Sequence[str] = None
    mixing_alpha: float = 0.0


@dataclass
class EvaluationCheckpointConfig:
    """
    :param sample_interval:
        Iteration interval to perform validation and possibly attribution analysis
        if set.

    :param saved_result_detail_level:
        Level of detail to save in the results file. Higher levels will save more
        information, but will take up more space and might be slow especially
        when using many tabular targets. The details are as follows
        (each step skips a level of detail in addition to the previous):

        - ``5``: The default, save all metrics and plots.
        - ``4``: Skip validation plots and generated samples under the
          ``/samples`` folder.
        - ``3``: Skip plots for individual targets.
        - ``2``: Skip individual target plots (e.g. R2 training curve for regression
          targets).
        - ``1``: Skip individual target metrics (including loss).

    :param checkpoint_interval:
        Iteration interval to checkpoint (i.e. save) model.

    :param n_saved_models:
        Number of top N models to saved during training.
    """

    sample_interval: int = 200
    saved_result_detail_level: int = 5
    checkpoint_interval: None | int = None
    n_saved_models: int = 1


@dataclass
class AttributionAnalysisConfig:
    """
    :param compute_attributions:
        Whether to compute attributions / feature importance scores
        (using integrated gradients) assigned by the model with respect to the
        input features.

    :param max_attributions_per_class:
        Maximum number of samples per class to gather for
        attribution / feature importance
        analysis. Good to use when modelling on imbalanced data.

    :param attributions_every_sample_factor:
        Controls whether the attributions / feature importance
        values are computed at every
        sample interval (set to ``1``), every other sample interval (set to ``2``), etc.
        Useful when computing the attributions takes a long time, and
        we don't want to do it every time we evaluate.

    :param attribution_background_samples:
        Number of samples to use for the background in attribution / feature importance
        computations.
    """

    compute_attributions: bool = False
    max_attributions_per_class: None | int = None
    attributions_every_sample_factor: int = 1
    attribution_background_samples: int = 256


@dataclass
class SupervisedMetricsConfig:
    """

    :param cat_metrics:
        Which metrics to calculate for categorical targets.

    :param con_metrics:
        Which metrics to calculate for continuous targets.

    :param cat_averaging_metrics:
        Which metrics to use for averaging categorical targets. If not set, will use
        the average of MCC, ROC-AUC and AP.

    :param con_averaging_metrics:
        Which metrics to use for averaging continuous targets. If not set,
        will use the average of 1.0-LOSS, PCC and R2.
    """

    cat_metrics: al_cat_metric_choices = ("mcc", "acc", "roc-auc-macro", "ap-macro")
    con_metrics: al_con_metric_choices = ("r2", "pcc", "loss")

    cat_averaging_metrics: al_cat_metric_choices | None = None
    con_averaging_metrics: al_con_metric_choices | None = None


@dataclass
class VisualizationLoggingConfig:
    """
    :param plot_skip_steps:
        How many iterations to skip in plots. Useful to get a zoomed-in view of the
        training process after the initial 'burn-in' period.

    :param no_pbar:
        Whether to not use progress bars. Useful when stdout/stderr is
        written to files.

    :param log_level:
        Logging level to use. Can be one of ``'debug'``, ``'info'``,
        ``'warning'``, ``'error'``, ``'critical'``.
    """

    plot_skip_steps: int = 200
    no_pbar: bool = False
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"


@dataclass
class DataPreparationConfig:
    """
    :param streaming_setup_samples:
        Number of samples to use during streaming setup (e.g., for training tokenizers).
        If ``None``, uses all available samples.

    :param streaming_batch_size:
        Batch size to use during streaming setup. If ``None``, uses the training
        batch size.
    """

    streaming_setup_samples: int = 10000
    streaming_batch_size: int | None = None


@dataclass
class AcceleratorConfig:
    """
    Configuration for distributed and accelerated training setup.

    :param hardware:
        The hardware accelerator to use.
        Options include ``'cpu'``, ``'cuda'``, ``'mps'``, ``'tpu'``, etc.

    :param precision:
        Numerical precision for training.
        Options include ``'32-true'``, ``'16-mixed'``, ``'bf16-mixed'``, etc.
        Using mixed precision can significantly speed up training on supported hardware.

    :param strategy:
        The parallelization strategy to use. Common options:

        - ``'dp'``: Data Parallelism
        - ``'ddp'``: Distributed Data Parallelism
        - ``'ddp_spawn'``: Distributed Data Parallelism with spawn
        - ``'deepspeed'``: DeepSpeed
        - ``'fsdp'``: Fully Sharded Data Parallelism
        - ``'auto'``: Automatically choose the best strategy based on the
          hardware available.

    :param devices:
        Number of devices to use or specific device indices.
        Can be an integer (e.g., 4 for 4 GPUs) or a list of indices (e.g., ``[0, 1]``).

    :param num_nodes:
        Number of compute nodes to use for distributed training.
    """

    hardware: str = "auto"
    precision: Literal[
        "64",
        "32",
        "16",
        "bf16",
        "64-true",
        "32-true",
        "16-true",
        "bf16-true",
        "16-mixed",
        "bf16-mixed",
        "transformer-engine",
        "transformer-engine-float16",
    ] = "32-true"
    strategy: str = "auto"
    devices: int | list[int] | str = "auto"
    num_nodes: int = 1


@dataclass
class GlobalConfig:
    """
    Global configurations that are common / relevant for the whole experiment to run.
    """

    basic_experiment: BasicExperimentConfig
    model: GlobalModelConfig
    optimization: OptimizationConfig
    lr_schedule: LRScheduleConfig
    training_control: TrainingControlConfig
    evaluation_checkpoint: EvaluationCheckpointConfig
    attribution_analysis: AttributionAnalysisConfig
    metrics: SupervisedMetricsConfig
    visualization_logging: VisualizationLoggingConfig
    data_preparation: DataPreparationConfig
    accelerator: AcceleratorConfig
    latent_sampling: LatentSamplingConfig | None = None

    be: BasicExperimentConfig = field(init=False, repr=False)
    m: GlobalModelConfig = field(init=False, repr=False)
    opt: OptimizationConfig = field(init=False, repr=False)
    lr: LRScheduleConfig = field(init=False, repr=False)
    tc: TrainingControlConfig = field(init=False, repr=False)
    ec: EvaluationCheckpointConfig = field(init=False, repr=False)
    aa: AttributionAnalysisConfig = field(init=False, repr=False)
    met: SupervisedMetricsConfig = field(init=False, repr=False)
    vl: VisualizationLoggingConfig = field(init=False, repr=False)
    dp: DataPreparationConfig = field(init=False, repr=False)
    ac: AcceleratorConfig = field(init=False, repr=False)
    ls: LatentSamplingConfig | None = field(init=False, repr=False)

    def __post_init__(self):
        self.be = self.basic_experiment
        self.m = self.model
        self.opt = self.optimization
        self.lr = self.lr_schedule
        self.tc = self.training_control
        self.ec = self.evaluation_checkpoint
        self.aa = self.attribution_analysis
        self.met = self.metrics
        self.vl = self.visualization_logging
        self.dp = self.data_preparation
        self.ac = self.accelerator
        self.ls = self.latent_sampling


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

    :param pretrained_config:
        Configuration for using leveraging pretraining from a previous experiment.

    :param interpretation_config:
        Configuration for interpretation analysis when applicable.

    :param tensor_broker_config:
        Configuration for tensor broker when applicable. Note that this is an
        experimental feature.

    """

    input_info: "InputDataConfig"
    input_type_info: Union[
        "OmicsInputDataConfig",
        "TabularInputDataConfig",
        "SequenceInputDataConfig",
        "ByteInputDataConfig",
        "ImageInputDataConfig",
        "ArrayInputDataConfig",
    ]
    model_config: al_feature_extractor_configs
    pretrained_config: Union[None, "BasicPretrainedConfig"] = None
    interpretation_config: Union[None, "BasicInterpretationConfig"] = None
    tensor_broker_config: None | TensorBrokerConfig = None


@dataclass
class InputDataConfig:
    """
    :param input_source:
        Where on the filesystem to locate the input. Altenratively,
        a websocket URL to connect to for streaming data.

    :param input_name:
        Name to identify the input. These must be unique across all inputs.

    :param input_type:
        Type of the input.

    :param input_inner_key:
        Inner key to use for the input. Only used when ``input_source`` is a deeplake
        dataset.
    """

    input_source: str
    input_name: str
    input_type: Literal["omics", "tabular", "sequence", "image", "bytes", "array"]
    input_inner_key: None | str = None


@dataclass
class OmicsInputDataConfig:
    """
    :param snp_file:
        Path to the relevant ``.bim`` file, used for attribution analysis. If not
        computing attributions, this can be set to ``None``.

    :param subset_snps_file:
        Path to a file with corresponding SNP IDs to subset from the main
        arrays for the modeling. Requires the ``snp_file`` parameter to
        be passed in.

    :param na_augment_alpha:
        Used to control the extent of missing data augmentation in the omics data.
        A value is sampled from a beta distribution, and the sampled value is used
        to set a percentage of the SNPs to be 'missing'.

        The alpha (α) parameter of the beta distribution influences the shape of the
        distribution towards 1. Higher values of alpha (compared to beta) bias the
        distribution to sample larger percentages of SNPs to be set as 'missing',
        leading to a higher likelihood of missingness.

        Conversely, lower values of alpha (compared to beta) result in sampling lower
        percentages, thus reducing the probability and extent of missingness.

        For example, setting alpha to 1.0 and beta to 5.0 will skew the distribution
        towards lower percentages of missingness, since beta is significantly larger.
        Setting alpha to 5.0 and beta to 1.0 will skew the distribution towards higher
        percentages of missingness, since alpha is significantly larger.

        **Examples:**

        - ``alpha = 1.0, beta = 9.0``: μ=E(X)=0.05, σ=SD(X)=0.0476 (avg 5% missing)
        - ``alpha = 1.0, beta = 4.0``: μ=E(X)=0.2, σ=SD(X)=0.1633 (avg 20% missing)

    :param na_augment_beta:
        Used to control the extent of missing data augmentation in the omics data.
        A value is sampled from a beta distribution, and the sampled value is used
        to set a percentage of the SNPs to be 'missing'.

        Beta (β) parameter of the beta distribution, influencing the shape of the
        distribution towards 0. Higher values of beta (compared to alpha) bias the
        distribution to sample smaller percentages of SNPs to be set as 'missing',
        leading to a lower likelihood and extent of missingness.

        Conversely, lower values of beta (compared to alpha) result in sampling
        larger percentages, thus increasing the probability and extent of missingness.

    :param shuffle_augment_alpha:
        Used to control the extent of shuffling data augmentation in the omics data.
        A value is sampled from a beta distribution, and the sampled value is used to
        determine the percentage of the SNPs to be shuffled.

        The alpha (α) parameter of the beta distribution influences the shape of
        the distribution towards 1. Higher values of alpha (compared to beta) bias
        the distribution to sample larger percentages of SNPs to be shuffled, leading
        to a higher likelihood of extensive shuffling.

        Conversely, lower values of alpha (compared to beta) result in sampling lower
        percentages, thus reducing the extent of shuffling. Setting alpha to a
        significantly larger value than beta will skew the distribution towards higher
        percentages of shuffling.

        **Examples:**

        - ``alpha = 1.0, beta = 9.0``: μ=E(X)=0.05, σ=SD(X)=0.0476 (avg 5% shuffled)
        - ``alpha = 1.0, beta = 4.0``: μ=E(X)=0.2, σ=SD(X)=0.1633 (avg 20% shuffled))

    :param shuffle_augment_beta:
        Used to control the extent of shuffling data augmentation in the omics data.
        A value is sampled from a beta distribution, and the sampled value is used to
        determine the percentage of the SNPs to be shuffled.

        Beta (β) parameter of the beta distribution, influencing the shape of the
        distribution towards 0. Higher values of beta (compared to alpha) bias the
        distribution to sample smaller percentages of SNPs to be shuffled, leading to
        a lower likelihood and extent of shuffling. Conversely, lower values of beta
        (compared to alpha) result in sampling larger percentages, thus increasing
        the likelihood and extent of shuffling.

    :param omics_format:
        Currently unsupported (i.e. does nothing), which format the omics data is in.

    :param mixing_subtype:
        Which type of mixing to use on the omics data given that ``mixing_alpha`` is
        set >0.0 in the global configuration.

    :param modality_dropout_rate:
        Dropout rate to apply to the modality, e.g., ``0.2`` means that 20% of the time,
        this modality will be dropped out during training.
    """

    snp_file: str | None = None
    subset_snps_file: str | None = None
    na_augment_alpha: float = 1.0
    na_augment_beta: float = 5.0
    shuffle_augment_alpha: float = 0.0
    shuffle_augment_beta: float = 0.0
    omics_format: Literal["one-hot"] = "one-hot"
    mixing_subtype: Literal["mixup", "cutmix-block", "cutmix-uniform"] = "mixup"
    modality_dropout_rate: float = 0.0


@dataclass
class TabularInputDataConfig:
    """
    :param input_cat_columns:
        Which columns to use as a categorical inputs from the ``input_source`` specified
        in the ``input_info`` field of the relevant ``.yaml``.

    :param input_con_columns:
        Which columns to use as a continuous inputs from the ``input_source`` specified
        in the ``input_info`` field of the relevant ``.yaml``.

    :param label_parsing_chunk_size:
        Number of rows to process at time when loading in the ``input_source``. Useful
        when RAM is limited.

    :param mixing_subtype:
        Which type of mixing to use on the tabular data given that ``mixing_alpha`` is
        set >0.0 in the global configuration.

    :param modality_dropout_rate:
        Dropout rate to apply to the modality, e.g., ``0.2`` means that 20% of the time,
        this modality will be dropped out during training.
    """

    input_cat_columns: Sequence[str] = field(default_factory=list)
    input_con_columns: Sequence[str] = field(default_factory=list)
    label_parsing_chunk_size: None | int = None
    mixing_subtype: Literal["mixup"] = "mixup"
    modality_dropout_rate: float = 0.0


@dataclass
class SequenceInputDataConfig:
    """
    :param vocab_file:
        An optional text file containing pre-defined vocabulary to use
        for the training. If this is not passed in, the framework will automatically
        build the vocabulary from the training data. Passing in a vocabulary file is
        therefore useful if (a) you want to manually specify / limit the vocabulary used
        and/or (b) you want to save time by pre-computing the vocabulary.

        Here, there are two formats supported:

        - A ``.json`` file containing a dictionary with the vocabulary as keys and
         the corresponding token IDs as values. For example:
         ``{"the": 0, "cat": 1, "sat": 2, "on": 3, "the": 4, "mat": 5}``

        - A ``.json`` file with the results of training and saving the vocabulary of
         a Huggingface BPE tokenizer. This is the file create by calling
         ``hf_tokenizer.save()``. This is only valid when using the ``bpe`` tokenizer.

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
        vocabulary. Setting this to ``None`` will split on every character in the
        sequence.

    :param tokenizer:
        Which tokenizer to use. Relevant if modeling on language, but not as much when
        doing it on other arbitrary sequences.

    :param tokenizer_language:
        Which language rules the tokenizer should apply when tokenizing the raw data.
        Only relevant when using a tokenizer that supports language-specific rules,
        such as ``spacy`` (which you have to install separately).

    :param adaptive_tokenizer_max_vocab_size:
        If using an adaptive tokenizer (``"bpe"``), this parameter controls the maximum
        size of the vocabulary.

    :param mixing_subtype:
        Which type of mixing to use on the sequence data given that ``mixing_alpha`` is
        set >0.0 in the global configuration.

    :param modality_dropout_rate:
        Dropout rate to apply to the modality, e.g., ``0.2`` means that 20% of the time,
        this modality will be dropped out during training.
    """

    vocab_file: None | str = None
    max_length: al_max_sequence_length = "average"
    sampling_strategy_if_longer: Literal["from_start", "uniform"] = "uniform"
    min_freq: int = 10
    split_on: str | None = " "
    tokenizer: al_tokenizer_choices = None  # type: ignore
    tokenizer_language: str | None = None
    adaptive_tokenizer_max_vocab_size: int | None = None
    mixing_subtype: Literal["mixup"] = "mixup"
    modality_dropout_rate: float = 0.0


@dataclass
class BasicPretrainedConfig:
    """
    :param model_path:
        Path save model from an EIR training run to load. Note that currently this
        only supports if it's in a folder from an EIR training run, as the current
        prototype functionality uses that to e.g., find configuration from the
        pretrained model run.

    :param load_module_name:
        Name of the module to extract and use in the respective input feature
        extraction.
    """

    model_path: str
    load_module_name: str


@dataclass
class BasicInterpretationConfig:
    """
    :param interpretation_sampling_strategy:
        How to sample sequences for attribution analysis. ``first_n`` always grabs the
        same first n values from the beginning of the dataset to interpret, while
        ``random_sample`` will sample uniformly from the whole dataset without
        replacement.

    :param num_samples_to_interpret:
        How many samples to interpret.

    :param manual_samples_to_interpret:
        IDs of samples to always interpret, irrespective of
        ``interpretation_sampling_strategy`` and ``num_samples_to_interpret``. A caveat
        here is that they must be present in the dataset that is being interpreted
        (e.g., validation / test dataset), meaning that adding IDs here that happen to
        be in the training dataset will not work.
    """

    interpretation_sampling_strategy: Literal["first_n", "random_sample"] = "first_n"
    num_samples_to_interpret: int = 10
    manual_samples_to_interpret: Sequence[str] | None = None


@dataclass
class ByteInputDataConfig:
    """
    :param byte_encoding:
        Which byte encoding to use when reading the binary data, currently only
        support ``"uint8"``.

    :param max_length:
        Maximum length to truncate/pad sequences to. While in sequence models this
        generally refers to words, here we are referring to the number of bytes.

    :param sampling_strategy_if_longer:
        Controls how sequences are truncated if they are longer than the specified
        ``max_length`` parameter. Using 'from_start' will always truncate from the
        beginning of the byte sequence, ensuring the samples will always be the same
        during training. Setting this parameter to ``uniform`` will uniformly sample
        a slice of a given sample sequence during training. Note that for consistency,
        the validation/test set samples always use the ``from_start`` setting when
        truncating.

    :param mixing_subtype:
        Which type of mixing to use on the bytes data given that ``mixing_alpha`` is
        set >0.0 in the global configuration.

    :param modality_dropout_rate:
        Dropout rate to apply to the modality, e.g., ``0.2`` means that 20% of the time,
        this modality will be dropped out during training.
    """

    max_length: int = 256
    byte_encoding: Literal["uint8"] = "uint8"
    sampling_strategy_if_longer: Literal["from_start", "uniform"] = "uniform"
    mixing_subtype: Literal["mixup"] = "mixup"
    modality_dropout_rate: float = 0.0


@dataclass
class ImageInputDataConfig:
    """
    :param auto_augment:
        Setting this to ``True`` will use TrivialAugment Wide augmentation.

    :param size:
        Target size of the images for training.  If size is a sequence like
        (h, w), output size will be matched to this. If size is an int,
        the image will be resized to (size, size).

    :param resize_approach:
        The method used for resizing the images. Options are:

        - ``resize``: Directly resize the image to the target size.
        - ``randomcrop``: Resize the image to a larger size than the target and then
          apply a random crop to the target size.
        - ``centercrop``: Resize the image to a larger size than the target and then
          apply a center crop to the target size.

    :param adaptive_normalization_max_samples:
        If using adaptive normalization (channel),
        how many samples to use to compute the normalization parameters.
        If ``None``, uses all samples.

    :param mean_normalization_values:
        Average channel values to normalize images with. This can be a sequence matching
        the number of channels, or ``None``. If ``None`` and using a pretrained model,
        the values used for the model pretraining will be used. If ``None`` and training
        from scratch, will iterate over training data and compute the running average
        per channel.

    :param stds_normalization_values:
        Standard deviation channel values to normalize images with. This can be a
        sequence mathing the number of channels, or ``None``. If ``None`` and using a
        pretrained model, the values used for the model pretraining will be used.
        If ``None`` and training from scratch, will iterate over training data and
        compute the running average per channel.

    :param mode:
        An explicit mode to convert loaded images to. Useful when working with
        input data with a mixed number of channels, or you want to convert
        images to a specific mode.
        Options are

        - ``RGB``: Red, Green, Blue (channels=3)
        - ``L``: Grayscale (channels=1)
        - ``RGBA``: Red, Green, Blue, Alpha (channels=4)

    :param num_channels:
        Number of channels in the images. If ``None``, tries to infer the number of
        channels from a random image in the training data. Useful when known
        ahead of time how many channels the images have, will raise an error if
        an image with a different number of channels is encountered.

    :param mixing_subtype:
        Which type of mixing to use on the image data given that ``mixing_alpha`` is
        set >0.0 in the global configuration.

    :param modality_dropout_rate:
        Dropout rate to apply to the modality, e.g., ``0.2`` means that 20% of the time,
        this modality will be dropped out during training.
    """

    auto_augment: bool = True
    size: Sequence[int] = (64,)
    resize_approach: Literal["resize", "randomcrop", "centercrop"] = "resize"
    adaptive_normalization_max_samples: int | None = None
    mean_normalization_values: None | Sequence[float] = None
    stds_normalization_values: None | Sequence[float] = None
    mode: Literal["RGB", "L", "RGBA"] | None = None
    num_channels: int | None = None
    mixing_subtype: Literal["mixup"] | Literal["cutmix"] = "mixup"
    modality_dropout_rate: float = 0.0


@dataclass
class ArrayInputDataConfig:
    """
    :param mixing_subtype:
        Which type of mixing to use on the image data given that ``mixing_alpha`` is
        set >0.0 in the global configuration.

    :param modality_dropout_rate:
        Dropout rate to apply to the modality, e.g., ``0.2`` means that 20% of the time,
        this modality will be dropped out during training.

    :param normalization:
        Which type of normalization to apply to the array data. If ``element``, will
        normalize each element in the array independently. If ``channel``, will
        normalize each channel in the array independently.
        For 'channel', assumes PyTorch format where the channel dimension is the
        first dimension.

    :param adaptive_normalization_max_samples:
        If using adaptive normalization (channel / element),
        how many samples to use to compute the normalization parameters.
        If ``None``, uses all samples.
    """

    mixing_subtype: Literal["mixup"] = "mixup"
    modality_dropout_rate: float = 0.0
    normalization: Literal["element", "channel"] | None = "channel"
    adaptive_normalization_max_samples: int | None = None


@dataclass
class FusionConfig:
    """
    :param model_type:
        Which type of fusion model to use.

    :param model_config:
        Fusion model configuration.
    """

    model_type: Literal[
        "mlp-residual",
        "identity",
        "mgmoe",
        "pass-through",
        "attention",
    ]
    model_config: (
        ResidualMLPConfig | IdentityConfig | MGMoEModelConfig | AttentionFusionConfig
    )
    tensor_broker_config: None | TensorBrokerConfig = None


@dataclass
class OutputInfoConfig:
    """
    :param output_source:
        Where on the filesystem to locate the output (if applicable).

    :param output_name:
        Name to identify the output.

    :param output_type:
        Type of the output.
    """

    output_source: str
    output_name: str
    output_type: Literal["tabular", "sequence", "array"]
    output_inner_key: str | None = None


@dataclass
class OutputConfig:
    """
    :param output_info:
        Information about the output source, name and type.

    :param output_type_info:
       Information specific to the output type, e.g. which columns to predict
       from a tabular file.

    :param model_config:
        Configuration for the chosen model (i.e. output module after fusion) for this
        output.

    :param sampling_config:
        Configuration for how to sample results from the output module.
    """

    output_info: OutputInfoConfig
    output_type_info: (
        TabularOutputTypeConfig
        | SequenceOutputTypeConfig
        | ArrayOutputTypeConfig
        | ImageOutputTypeConfig
        | SurvivalOutputTypeConfig
    )
    model_config: (
        TabularOutputModuleConfig | SequenceOutputModuleConfig | ArrayOutputModuleConfig
    )

    sampling_config: (
        SequenceOutputSamplingConfig
        | ArrayOutputSamplingConfig
        | ImageOutputSamplingConfig
        | dict
        | None
    ) = None
    tensor_broker_config: None | TensorBrokerConfig = None
