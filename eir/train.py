import argparse
import sys
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import (
    Union,
    Tuple,
    List,
    Dict,
    overload,
    TYPE_CHECKING,
    Callable,
    Iterable,
    Sequence,
    Any,
    Set,
    Type,
)

import dill
import numpy as np
import torch
from aislib.misc_utils import ensure_path_exists
from aislib.misc_utils import get_logger
from ignite.engine import Engine
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from eir.configuration import get_default_cl_args
from eir.data_load import data_utils
from eir.data_load import datasets
from eir.data_load.data_augmentation import hook_mix_loss, get_mix_data_hook
from eir.data_load.data_loading_funcs import get_weighted_random_sampler
from eir.data_load.data_utils import Batch
from eir.data_load.label_setup import (
    al_target_columns,
    al_label_transformers,
    al_all_column_ops,
    get_array_path_iterator,
    set_up_train_and_valid_tabular_data,
    gather_ids_from_tabular_file,
    split_ids,
    TabularFileInfo,
    save_transformer_set,
    Labels,
)
from eir.models import fusion, fusion_mgmoe
from eir.models import model_training_utils
from eir.models.model_training_utils import run_lr_find
from eir.models.models_linear import LinearModel, LinearModelConfig
from eir.models.omics.omics_models import (
    get_model_class,
    get_omics_model_init_kwargs,
    match_namespace_to_dataclass,
)
from eir.models.tabular.tabular import (
    get_tabular_inputs,
    TabularModel,
    get_unique_values_from_transformers,
)
from eir.train_utils import utils
from eir.train_utils.metrics import (
    calculate_batch_metrics,
    calculate_prediction_losses,
    aggregate_losses,
    add_multi_task_average_metrics,
    get_average_history_filepath,
    get_default_metrics,
    hook_add_l1_loss,
    get_uncertainty_loss_hook,
    add_loss_to_metrics,
)
from eir.train_utils.optimizers import (
    get_optimizer,
    get_optimizer_backward_kwargs,
)
from eir.train_utils.train_handlers import HandlerConfig
from eir.train_utils.train_handlers import configure_trainer
from eir.train_utils.utils import (
    call_hooks_stage_iterable,
)

if TYPE_CHECKING:
    from eir.train_utils.metrics import (
        al_step_metric_dict,
        al_metric_record_dict,
    )

# aliases
al_criterions = Dict[str, Union[nn.CrossEntropyLoss, nn.MSELoss]]
# these are all after being collated by torch dataloaders
al_training_labels_target = Dict[str, Union[torch.LongTensor, torch.Tensor]]
al_training_labels_extra = Dict[str, Union[List[str], torch.Tensor]]
al_training_labels_batch = Dict[
    str, Union[al_training_labels_target, al_training_labels_extra]
]
al_dataloader_getitem_batch = Tuple[
    Union[Dict[str, torch.Tensor], Dict[str, Any]],
    al_training_labels_target,
    List[str],
]
al_num_outputs_per_target = Dict[str, int]

torch.manual_seed(0)
np.random.seed(0)

logger = get_logger(name=__name__, tqdm_compatible=True)


def main():
    default_cl_args = get_default_cl_args()
    utils.configure_root_logger(run_name=default_cl_args.run_name)

    default_hooks = get_default_hooks(cl_args_=default_cl_args)
    default_config = get_default_config(cl_args=default_cl_args, hooks=default_hooks)

    run_experiment(cl_args=default_cl_args, config=default_config)


def run_experiment(cl_args: argparse.Namespace, config: "Config") -> None:

    _log_model(model=config.model, l1_weight=cl_args.l1)

    run_folder = utils.get_run_folder(run_name=cl_args.run_name)
    keys_to_serialize = get_default_config_keys_to_serialize()
    serialize_config(
        config=config, run_folder=run_folder, keys_to_serialize=keys_to_serialize
    )

    train(config=config)


@dataclass(frozen=True)
class Config:
    """
    The idea of this class is to keep track of objects that need to be used
    in multiple contexts in different parts of the code (e.g. the train
    dataloader is used to load samples during training, but also as background
    for SHAP activation calculations).
    """

    cl_args: argparse.Namespace
    data_dimensions: Dict[str, "DataDimensions"]
    train_loader: torch.utils.data.DataLoader
    valid_loader: torch.utils.data.DataLoader
    valid_dataset: torch.utils.data.Dataset
    labels_dict: Dict
    target_transformers: al_label_transformers
    target_columns: al_target_columns
    num_outputs_per_target: al_num_outputs_per_target
    model: Union[fusion.FusionModel, nn.DataParallel]
    optimizer: Optimizer
    criterions: al_criterions
    loss_function: Callable
    writer: SummaryWriter
    metrics: "al_metric_record_dict"
    hooks: Union["Hooks", None]


def serialize_config(
    config: "Config", run_folder: Path, keys_to_serialize: Union[Iterable[str], None]
) -> None:
    serialization_path = get_train_config_serialization_path(run_folder=run_folder)
    ensure_path_exists(path=serialization_path)

    filtered_config = filter_config_by_keys(config=config, keys=keys_to_serialize)
    serialize_namespace(namespace=filtered_config, output_path=serialization_path)


def get_train_config_serialization_path(run_folder: Path) -> Path:
    train_config_path = run_folder / "serializations" / "filtered_config.dill"

    return train_config_path


def get_default_config_keys_to_serialize() -> Iterable[str]:
    return (
        "cl_args",
        "data_dimensions",
        "num_outputs_per_target",
        "target_transformers",
        "target_columns",
        "metrics",
        "hooks",
    )


def filter_config_by_keys(
    config: "Config", keys: Union[None, Iterable[str]] = None
) -> SimpleNamespace:
    filtered = {}

    config_fields = (f.name for f in fields(config))
    iterable = keys if keys is not None else config_fields

    for k in iterable:
        filtered[k] = getattr(config, k)

    namespace = SimpleNamespace(**filtered)

    return namespace


def serialize_namespace(namespace: SimpleNamespace, output_path: Path) -> None:
    with open(output_path, "wb") as outfile:
        dill.dump(namespace, outfile)


def get_default_config(
    cl_args: argparse.Namespace, hooks: Union["Hooks", None] = None
) -> "Config":
    run_folder = _prepare_run_folder(run_name=cl_args.run_name)

    target_labels, tabular_input_labels = get_target_and_tabular_input_labels(
        cl_args=cl_args,
        custom_label_parsing_operations=hooks.custom_column_label_parsing_ops,
    )
    save_transformer_set(
        transformers=target_labels.label_transformers, run_folder=run_folder
    )
    save_transformer_set(
        transformers=tabular_input_labels.label_transformers, run_folder=run_folder
    )

    data_dimensions = _gather_all_omics_data_dimensions(
        omics_sources=cl_args.omics_sources, omics_names=cl_args.omics_names
    )

    train_dataset, valid_dataset = datasets.set_up_datasets_from_cl_args(
        cl_args=cl_args,
        data_dimensions=data_dimensions,
        target_labels=target_labels,
        tabular_inputs_labels=tabular_input_labels,
    )

    batch_size = _modify_bs_for_multi_gpu(
        multi_gpu=cl_args.multi_gpu, batch_size=cl_args.batch_size
    )

    train_sampler = get_train_sampler(
        columns_to_sample=cl_args.weighted_sampling_column, train_dataset=train_dataset
    )

    train_dloader, valid_dloader = get_dataloaders(
        train_dataset=train_dataset,
        train_sampler=train_sampler,
        valid_dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=cl_args.dataloader_workers,
    )

    num_outputs_per_target = set_up_num_outputs_per_target(
        target_transformers=target_labels.label_transformers
    )

    model = get_model_from_cl_args(
        cl_args=cl_args,
        omics_data_dimensions=data_dimensions,
        num_outputs_per_target=num_outputs_per_target,
        tabular_label_transformers=tabular_input_labels.label_transformers,
    )

    criterions = _get_criterions(
        target_columns=train_dataset.target_columns, model_type=cl_args.model_type
    )

    writer = get_summary_writer(run_folder=run_folder)

    loss_func = _get_loss_callable(
        criterions=criterions,
    )

    optimizer = get_optimizer(model=model, loss_callable=loss_func, cl_args=cl_args)

    metrics = get_default_metrics(target_transformers=target_labels.label_transformers)

    config = Config(
        cl_args=cl_args,
        data_dimensions=data_dimensions,
        train_loader=train_dloader,
        valid_loader=valid_dloader,
        valid_dataset=valid_dataset,
        labels_dict=train_dataset.target_labels_dict,
        target_transformers=target_labels.label_transformers,
        num_outputs_per_target=num_outputs_per_target,
        target_columns=train_dataset.target_columns,
        model=model,
        optimizer=optimizer,
        criterions=criterions,
        loss_function=loss_func,
        writer=writer,
        metrics=metrics,
        hooks=hooks,
    )

    return config


def get_target_and_tabular_input_labels(
    cl_args: argparse.Namespace, custom_label_parsing_operations: al_all_column_ops
) -> Tuple[Labels, Labels]:
    """
    TODO: Set up support for multiple tabular files.
    """
    all_array_ids = gather_ids_from_tabular_file(file_path=cl_args.label_file)
    train_ids, valid_ids = split_ids(ids=all_array_ids, valid_size=cl_args.valid_size)

    logger.info("Setting up target labels.")
    target_labels_info = get_tabular_target_label_data(cl_args=cl_args)
    target_labels = set_up_train_and_valid_tabular_data(
        tabular_info=target_labels_info,
        custom_label_ops=custom_label_parsing_operations,
        train_ids=train_ids,
        valid_ids=valid_ids,
    )

    tabular_inputs_info = get_tabular_inputs_label_data(cl_args=cl_args)

    n_cat_tabular = len(tabular_inputs_info.cat_columns)
    n_con_tabular = len(tabular_inputs_info.con_columns)
    if n_cat_tabular + n_con_tabular > 0:
        logger.info("Setting up tabular inputs.")
        tabular_inputs = set_up_train_and_valid_tabular_data(
            tabular_info=tabular_inputs_info,
            custom_label_ops=custom_label_parsing_operations,
            train_ids=train_ids,
            valid_ids=valid_ids,
        )
    else:
        tabular_inputs = Labels(train_labels={}, valid_labels={}, label_transformers={})

    return target_labels, tabular_inputs


def get_tabular_target_label_data(cl_args: argparse.Namespace) -> TabularFileInfo:

    table_info = TabularFileInfo(
        file_path=cl_args.label_file,
        con_columns=cl_args.target_con_columns,
        cat_columns=cl_args.target_cat_columns,
        parsing_chunk_size=cl_args.label_parsing_chunk_size,
    )

    return table_info


def get_tabular_inputs_label_data(cl_args: argparse.Namespace) -> TabularFileInfo:

    table_info = TabularFileInfo(
        file_path=cl_args.label_file,
        con_columns=cl_args.extra_con_columns,
        cat_columns=cl_args.extra_cat_columns,
        parsing_chunk_size=cl_args.label_parsing_chunk_size,
    )

    return table_info


def _gather_all_omics_data_dimensions(
    omics_sources: Sequence[str], omics_names: Sequence[str]
) -> Dict[str, "DataDimensions"]:

    data_dimensions = {}

    if not omics_sources and not omics_names:
        return data_dimensions

    assert len(omics_sources) == len(omics_names)

    for source, name in zip(omics_sources, omics_names):
        cur_dimension = _get_data_dimension_from_data_source(data_source=Path(source))
        data_dimensions[name] = cur_dimension

    return data_dimensions


@dataclass
class DataDimensions:
    channels: int
    height: int
    width: int

    def num_elements(self):
        return self.channels * self.height * self.width


def _get_data_dimension_from_data_source(
    data_source: Path,
) -> DataDimensions:
    """
    TODO: Make more dynamic / robust. Also weird to say "width" for a 1D vector.
    """

    iterator = get_array_path_iterator(data_source=data_source)
    path = next(iterator)
    shape = np.load(file=path).shape

    if len(shape) == 1:
        channels, height, width = 1, 1, shape[0]
    elif len(shape) == 2:
        channels, height, width = 1, shape[0], shape[1]
    elif len(shape) == 3:
        channels, height, width = shape
    else:
        raise ValueError("Currently max 3 dimensional inputs supported")

    return DataDimensions(channels=channels, height=height, width=width)


def set_up_num_outputs_per_target(
    target_transformers: al_label_transformers,
) -> al_num_outputs_per_target:

    num_outputs_per_target_dict = {}
    for target_column, transformer in target_transformers.items():
        if isinstance(transformer, StandardScaler):
            num_outputs = 1
        else:
            num_outputs = len(transformer.classes_)

            if num_outputs < 2:
                logger.warning(
                    f"Only {num_outputs} unique values found in categorical label "
                    f"column {target_column} (returned by {transformer}). This means "
                    f"that most likely an error will be raised if e.g. using "
                    f"nn.CrossEntropyLoss as it expects an output dimension of >=2."
                )

        num_outputs_per_target_dict[target_column] = num_outputs

    return num_outputs_per_target_dict


def _prepare_run_folder(run_name: str) -> Path:
    run_folder = utils.get_run_folder(run_name=run_name)
    history_file = get_average_history_filepath(
        run_folder=run_folder, train_or_val_target_prefix="train_"
    )
    if history_file.exists():
        raise FileExistsError(
            f"There already exists a run with that name: {history_file}. Please choose "
            f"a different run name or delete the folder."
        )

    ensure_path_exists(path=run_folder, is_folder=True)

    return run_folder


def _modify_bs_for_multi_gpu(multi_gpu: bool, batch_size: int) -> int:
    if multi_gpu:
        batch_size = torch.cuda.device_count() * batch_size
        logger.info(
            "Batch size set to %d to account for %d GPUs.",
            batch_size,
            torch.cuda.device_count(),
        )

    return batch_size


@overload
def get_train_sampler(
    columns_to_sample: None, train_dataset: datasets.DatasetBase
) -> None:
    ...


@overload
def get_train_sampler(
    columns_to_sample: List[str], train_dataset: datasets.DatasetBase
) -> WeightedRandomSampler:
    ...


def get_train_sampler(columns_to_sample, train_dataset):
    """
    TODO:   Refactor, remove dependency on train_dataset and use instead
            Iterable[Samples], and target_columns directly.
    """
    if columns_to_sample is None:
        return None

    loaded_target_columns = (
        train_dataset.target_columns["con"] + train_dataset.target_columns["cat"]
    )

    is_sample_column_loaded = set(columns_to_sample).issubset(
        set(loaded_target_columns)
    )
    is_sample_all_cols = columns_to_sample == ["all"]

    if not is_sample_column_loaded and not is_sample_all_cols:
        raise ValueError(
            "Weighted sampling from non-loaded columns not supported yet "
            f"(could not find {columns_to_sample})."
        )

    if is_sample_all_cols:
        columns_to_sample = train_dataset.target_columns["cat"]

    train_sampler = get_weighted_random_sampler(
        samples=train_dataset.samples, target_columns=columns_to_sample
    )
    return train_sampler


def get_dataloaders(
    train_dataset: datasets.DatasetBase,
    train_sampler: Union[None, WeightedRandomSampler],
    valid_dataset: datasets.DatasetBase,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple:

    train_dloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False if train_sampler else True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )

    valid_dloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )

    return train_dloader, valid_dloader


class GetAttrDelegatedDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def get_model_from_cl_args(
    cl_args: argparse.Namespace,
    omics_data_dimensions: Dict[str, DataDimensions],
    num_outputs_per_target: al_num_outputs_per_target,
    tabular_label_transformers: al_label_transformers,
) -> Union[nn.Module, nn.DataParallel]:
    """
    Note:   If we have a linear model, we just return that because that takes care of
            fusing tabular and genotype data by itself.

    Note:   This function currently assumes 1 omics source, and one tabular source.
            If we wanted to expand it, we would have to pass in nested configurations
            for each data source (i.e. model), instead of just cl_args. Also we would
            need to have nested data_dimensions, linking to models.
    """

    if cl_args.model_type == "linear":
        assert len(omics_data_dimensions) == 1

        linear_model = _get_linear_model(
            target_cat_columns=cl_args.target_cat_columns,
            target_con_columns=cl_args.target_con_columns,
            data_dimensions=omics_data_dimensions,
            device=cl_args.device,
        )
        return linear_model

    fusion_class = get_fusion_class_from_cl_args(
        fusion_model_type=cl_args.fusion_model_type
    )
    fusion_kwargs = get_fusion_kwargs_from_cl_args(
        cl_args=cl_args,
        omics_data_dimensions=omics_data_dimensions,
        num_outputs_per_target=num_outputs_per_target,
        tabular_label_transformers=tabular_label_transformers,
    )
    fusion_model = fusion_class(**fusion_kwargs)
    fusion_model = fusion_model.to(device=cl_args.device)

    if cl_args.multi_gpu:
        fusion_model = GetAttrDelegatedDataParallel(module=fusion_model)

    return fusion_model


def _get_linear_model(
    target_cat_columns: Sequence[str],
    target_con_columns: Sequence[str],
    data_dimensions: Dict[str, DataDimensions],
    device: str,
) -> LinearModel:
    """
    TODO: Update / fix when we make a more general linear model. Currently it only
          supports one omics input and tabular data.
    """

    assert len(data_dimensions) == 1

    single_input_name = list(data_dimensions.keys())[0]
    single_input_data_dimension = list(data_dimensions.values())[0]

    model_config = LinearModelConfig(
        input_name=single_input_name,
        data_dimensions=single_input_data_dimension,
        target_cat_columns=target_cat_columns,
        target_con_columns=target_con_columns,
    )

    model = LinearModel(model_config=model_config)
    model = model.to(device=device)

    return model


def get_modules_to_fuse_from_cl_args(
    cl_args: argparse.Namespace,
    omics_data_dimensions: Dict[str, DataDimensions],
    tabular_label_transformers: al_label_transformers,
):
    models = nn.ModuleDict()

    for name, dimensions in omics_data_dimensions.items():
        cur_omics_model = get_omics_model_from_cl_args(
            cl_args=cl_args, data_dimensions=dimensions
        )

        models[name] = cur_omics_model

    if cl_args.extra_con_columns or cl_args.extra_cat_columns:
        unique_tabular_values = get_unique_values_from_transformers(
            transformers=tabular_label_transformers,
            keys_to_use=cl_args.extra_cat_columns,
        )

        tabular_model = get_tabular_model(
            cat_columns=cl_args.extra_cat_columns,
            con_columns=cl_args.extra_con_columns,
            device=cl_args.device,
            unique_label_values=unique_tabular_values,
        )
        models["tabular_cl_args"] = tabular_model

    return models


def get_tabular_model(
    cat_columns: Sequence[str],
    con_columns: Sequence[str],
    device: str,
    unique_label_values: Dict[str, Set[str]],
) -> TabularModel:
    tabular_model = TabularModel(
        cat_columns=cat_columns,
        con_columns=con_columns,
        unique_label_values=unique_label_values,
        device=device,
    )

    return tabular_model


def get_omics_model_from_cl_args(
    cl_args: argparse.Namespace, data_dimensions: DataDimensions
):

    omics_model_class = get_model_class(model_type=cl_args.model_type)
    model_init_kwargs = get_omics_model_init_kwargs(
        model_type=cl_args.model_type, cl_args=cl_args, data_dimensions=data_dimensions
    )
    omics_model = omics_model_class(**model_init_kwargs)

    if cl_args.model_type == "cnn":
        assert omics_model.data_size_after_conv >= 8

    omics_model = omics_model.to(device=cl_args.device)

    return omics_model


def get_fusion_class_from_cl_args(
    fusion_model_type: str,
) -> Type[nn.Module]:
    if fusion_model_type == "mgmoe":
        return fusion_mgmoe.MGMoEModel
    elif fusion_model_type == "default":
        return fusion.FusionModel
    raise ValueError(f"Unrecognized fusion model type: {fusion_model_type}.")


def get_fusion_kwargs_from_cl_args(
    cl_args: argparse.Namespace,
    omics_data_dimensions: Dict[str, DataDimensions],
    num_outputs_per_target: al_num_outputs_per_target,
    tabular_label_transformers: al_label_transformers,
) -> Dict[str, Any]:

    kwargs = {}
    modules_to_fuse = get_modules_to_fuse_from_cl_args(
        cl_args=cl_args,
        omics_data_dimensions=omics_data_dimensions,
        tabular_label_transformers=tabular_label_transformers,
    )
    kwargs["modules_to_fuse"] = modules_to_fuse

    kwargs["num_outputs_per_target"] = num_outputs_per_target

    model_dataclass_config_class = fusion.FusionModelConfig
    if cl_args.fusion_model_type == "mgmoe":
        model_dataclass_config_class = fusion_mgmoe.MGMoEModelConfig
    model_dataclass_config_kwargs = match_namespace_to_dataclass(
        namespace=cl_args, data_class=model_dataclass_config_class
    )
    model_config = model_dataclass_config_class(**model_dataclass_config_kwargs)

    kwargs["model_config"] = model_config

    return kwargs


def _check_linear_model_columns(cl_args: argparse.Namespace) -> None:
    num_label_cols = len(cl_args.target_cat_columns + cl_args.target_con_columns)
    if num_label_cols != 1:
        raise NotImplementedError(
            "Linear model only supports one target column currently."
        )

    num_extra_cols = len(cl_args.extra_cat_columns + cl_args.extra_con_columns)
    if num_extra_cols != 0:
        raise NotImplementedError(
            "Extra columns not supported for linear model currently."
        )


def _get_criterions(
    target_columns: al_target_columns, model_type: str
) -> al_criterions:
    criterions_dict = {}

    def get_criterion(column_type_):

        if model_type == "linear":
            if column_type_ == "cat":
                return _calc_bce
            else:
                return partial(_calc_mse, mse_loss_func=nn.MSELoss(reduction="mean"))

        if column_type_ == "con":
            return partial(_calc_mse, mse_loss_func=nn.MSELoss())
        elif column_type_ == "cat":
            return nn.CrossEntropyLoss()

    target_columns_gen = data_utils.get_target_columns_generator(
        target_columns=target_columns
    )

    for column_type, column_name in target_columns_gen:
        criterion = get_criterion(column_type_=column_type)
        criterions_dict[column_name] = criterion

    return criterions_dict


def _calc_bce(input, target):
    # note we use input and not e.g. input_ here because torch uses name "input"
    # in loss functions for compatibility
    bce_loss_func = nn.BCELoss()
    return bce_loss_func(input[:, 1], target.to(dtype=torch.float))


def _calc_mse(input, target, mse_loss_func: nn.MSELoss):
    return mse_loss_func(input=input.squeeze(), target=target.squeeze())


def _get_loss_callable(criterions: al_criterions):

    single_task_loss_func = partial(calculate_prediction_losses, criterions=criterions)
    return single_task_loss_func


def get_summary_writer(run_folder: Path) -> SummaryWriter:
    log_dir = Path(run_folder / "tensorboard_logs")
    writer = SummaryWriter(log_dir=str(log_dir))

    return writer


def _log_model(model: nn.Module, l1_weight: float) -> None:
    """
    TODO: Add summary of parameters
    TODO: Add verbosity option
    """
    no_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if l1_weight:
        logger.debug(
            "Penalizing weights of shape %s with L1 loss with weight %f.",
            model.l1_penalized_weights.shape,
            l1_weight,
        )

    logger.info(
        "Starting training with a %s parameter model.", format(no_params, ",.0f")
    )


def train(config: Config) -> None:
    c = config
    cl_args = config.cl_args
    step_hooks = c.hooks.step_func_hooks

    def step(
        engine: Engine,
        loader_batch: Tuple[torch.Tensor, al_training_labels_batch, List[str]],
    ) -> "al_step_metric_dict":
        """
        The output here goes to trainer.output.
        """
        c.model.train()
        c.optimizer.zero_grad()

        base_prepare_inputs_stage = step_hooks.base_prepare_batch
        state = call_hooks_stage_iterable(
            hook_iterable=base_prepare_inputs_stage,
            common_kwargs={"config": config, "loader_batch": loader_batch},
            state=None,
        )
        base_batch = state["batch"]

        post_prepare_inputs_stage = step_hooks.post_prepare_batch
        state = call_hooks_stage_iterable(
            hook_iterable=post_prepare_inputs_stage,
            common_kwargs={"config": config, "loader_batch": base_batch},
            state=state,
        )
        batch = state["batch"]

        model_forward_loss_stage = step_hooks.model_forward
        state = call_hooks_stage_iterable(
            hook_iterable=model_forward_loss_stage,
            common_kwargs={"config": config, "batch": batch},
            state=state,
        )

        loss_stage = step_hooks.loss
        state = call_hooks_stage_iterable(
            hook_iterable=loss_stage,
            common_kwargs={"config": config, "batch": batch},
            state=state,
        )

        optimizer_backward_stage = step_hooks.optimizer_backward
        state = call_hooks_stage_iterable(
            hook_iterable=optimizer_backward_stage,
            common_kwargs={"config": config, "batch": batch},
            state=state,
        )

        metrics_stage = step_hooks.metrics
        state = call_hooks_stage_iterable(
            hook_iterable=metrics_stage,
            common_kwargs={"config": config, "batch": batch},
            state=state,
        )

        return state["metrics"]

    trainer = Engine(process_function=step)

    if cl_args.find_lr:
        logger.info("Running LR find and exiting.")
        run_lr_find(
            trainer_engine=trainer,
            train_dataloader=c.train_loader,
            model=c.model,
            optimizer=c.optimizer,
            output_folder=utils.get_run_folder(run_name=cl_args.run_name),
        )
        sys.exit(0)

    trainer = configure_trainer(trainer=trainer, config=config)

    trainer.run(data=c.train_loader, max_epochs=cl_args.n_epochs)


def get_default_hooks(cl_args_: argparse.Namespace):
    step_func_hooks = _get_default_step_function_hooks(cl_args=cl_args_)
    hooks_object = Hooks(step_func_hooks=step_func_hooks)

    return hooks_object


@dataclass
class Hooks:
    al_handler_attachers = Iterable[Callable[[Engine, HandlerConfig], Engine]]

    step_func_hooks: "StepFunctionHookStages"
    custom_column_label_parsing_ops: al_all_column_ops = None
    custom_handler_attachers: Union[None, al_handler_attachers] = None


def _get_default_step_function_hooks(cl_args: argparse.Namespace):
    """
    TODO: Add validation, inspect that outputs have correct names.
    TODO: Refactor, split into smaller functions e.g. for L1, mixing and uncertainty.
    """

    init_kwargs = _get_default_step_function_hooks_init_kwargs(cl_args=cl_args)

    step_func_hooks = StepFunctionHookStages(**init_kwargs)

    return step_func_hooks


def _get_default_step_function_hooks_init_kwargs(
    cl_args: argparse.Namespace,
) -> Dict[str, Sequence[Callable]]:

    init_kwargs = {
        "base_prepare_batch": [hook_default_prepare_batch],
        "post_prepare_batch": [],
        "model_forward": [hook_default_model_forward],
        "loss": [hook_default_per_target_loss],
        "optimizer_backward": [hook_default_optimizer_backward],
        "metrics": [hook_default_compute_metrics],
    }

    if cl_args.mixing_type is not None:
        logger.debug(
            "Setting up hooks for mixing with %s with α=%.2g.",
            cl_args.mixing_type,
            cl_args.mixing_alpha,
        )
        mix_hook = get_mix_data_hook(mixing_type=cl_args.mixing_type)

        init_kwargs["post_prepare_batch"].append(mix_hook)
        init_kwargs["loss"][0] = hook_mix_loss

    if len(cl_args.target_con_columns + cl_args.target_cat_columns) > 1:
        logger.debug(
            "Setting up hook for uncertainty weighted loss for multi task modelling."
        )
        uncertainty_hook = get_uncertainty_loss_hook(
            target_cat_columns=cl_args.target_cat_columns,
            target_con_columns=cl_args.target_con_columns,
            device=cl_args.device,
        )
        init_kwargs["loss"].append(uncertainty_hook)

    init_kwargs["loss"].append(hook_default_aggregate_losses)

    if cl_args.l1:
        init_kwargs["loss"].append(hook_add_l1_loss)

    return init_kwargs


@dataclass
class StepFunctionHookStages:

    al_hook = Callable[..., Dict]
    al_hooks = [Iterable[al_hook]]

    base_prepare_batch: al_hooks
    post_prepare_batch: al_hooks
    model_forward: al_hooks
    loss: al_hooks
    optimizer_backward: al_hooks
    metrics: al_hooks


def hook_default_prepare_batch(
    config: "Config",
    loader_batch: al_dataloader_getitem_batch,
    *args,
    **kwargs,
) -> Dict:

    batch = prepare_base_batch_default(
        loader_batch=loader_batch,
        cl_args=config.cl_args,
        target_columns=config.target_columns,
        model=config.model,
    )

    state_updates = {"batch": batch}

    return state_updates


def prepare_base_batch_default(
    loader_batch: al_dataloader_getitem_batch,
    cl_args: argparse.Namespace,
    target_columns: al_target_columns,
    model: fusion.FusionModel,
):

    inputs, target_labels, train_ids = loader_batch

    inputs_prepared = {}
    for input_name in inputs:

        if input_name.startswith("omics_"):
            cur_omics = inputs[input_name]
            cur_omics = cur_omics.to(device=cl_args.device)
            cur_omics = cur_omics.to(dtype=torch.float32)

            inputs_prepared[input_name] = cur_omics

        elif input_name.startswith("tabular_"):

            tabular_source_input: Dict[str, torch.Tensor] = inputs[input_name]
            for name, tensor in tabular_source_input.items():
                tabular_source_input[name] = tensor.to(device=cl_args.device)

            tabular = get_tabular_inputs(
                extra_cat_columns=cl_args.extra_cat_columns,
                extra_con_columns=cl_args.extra_con_columns,
                tabular_model=model.modules_to_fuse[input_name],
                tabular_input=tabular_source_input,
                device=cl_args.device,
            )
            inputs_prepared[input_name] = tabular
        else:
            raise ValueError(f"Unrecognized input type {input_name}.")

    if target_labels:
        target_labels = model_training_utils.parse_target_labels(
            target_columns=target_columns,
            device=cl_args.device,
            labels=target_labels,
        )

    batch = Batch(
        inputs=inputs_prepared,
        target_labels=target_labels,
        ids=train_ids,
    )

    return batch


def hook_default_model_forward(
    config: "Config", batch: "Batch", *args, **kwargs
) -> Dict:

    inputs = batch.inputs
    train_outputs = config.model(inputs=inputs)

    state_updates = {"model_outputs": train_outputs}

    return state_updates


def hook_default_optimizer_backward(
    config: "Config", state: Dict, *args, **kwargs
) -> Dict:

    optimizer_backward_kwargs = get_optimizer_backward_kwargs(
        optimizer_name=config.cl_args.optimizer
    )

    state["loss"].backward(**optimizer_backward_kwargs)
    config.optimizer.step()

    return {}


def hook_default_compute_metrics(
    config: "Config", batch: "Batch", state: Dict, *args, **kwargs
):

    train_batch_metrics = calculate_batch_metrics(
        target_columns=config.target_columns,
        outputs=state["model_outputs"],
        labels=batch.target_labels,
        mode="train",
        metric_record_dict=config.metrics,
    )

    train_batch_metrics_w_loss = add_loss_to_metrics(
        target_columns=config.target_columns,
        losses=state["per_target_train_losses"],
        metric_dict=train_batch_metrics,
    )

    train_batch_metrics_with_averages = add_multi_task_average_metrics(
        batch_metrics_dict=train_batch_metrics_w_loss,
        target_columns=config.target_columns,
        loss=state["loss"].item(),
        performance_average_functions=config.metrics["averaging_functions"],
    )

    state_updates = {"metrics": train_batch_metrics_with_averages}

    return state_updates


def hook_default_per_target_loss(
    config: "Config", batch: "Batch", state: Dict, *args, **kwargs
) -> Dict:

    per_target_train_losses = config.loss_function(
        inputs=state["model_outputs"], targets=batch.target_labels
    )

    state_updates = {"per_target_train_losses": per_target_train_losses}

    return state_updates


def hook_default_aggregate_losses(state: Dict, *args, **kwargs) -> Dict:

    train_loss_avg = aggregate_losses(losses_dict=state["per_target_train_losses"])
    state_updates = {"loss": train_loss_avg}

    return state_updates


if __name__ == "__main__":
    main()
