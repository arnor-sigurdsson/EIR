from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Dict, Literal
from typing import Set, overload, TYPE_CHECKING, Sequence, Iterable, Any

import joblib
import torch
from aislib.misc_utils import ensure_path_exists
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn

if TYPE_CHECKING:
    from eir.data_load.label_setup import al_label_dict

# Aliases
al_tabular_input = Dict[str, Union[List[str], torch.Tensor]]
al_unique_embed_vals = Dict[str, Set[str]]
al_emb_lookup_dict = Dict[str, Dict[str, Dict[int, int]]]


@dataclass
class TabularModelConfig:

    """
    :param model_type:
         Which type of image model to use.

    :param model_init_config:
          Configuration / arguments used to initialise model.
    """

    model_init_config: "SimpleTabularModelConfig"
    model_type: Literal["tabular"] = "tabular"


@dataclass
class SimpleTabularModelConfig:
    """
    :param l1:
        L1 regularization applied to the embeddings for categorical tabular inputs.

    :param fc_layer:
        Whether to add a single fully-connected layer to the model, alternative
        to looking up and passing the inputs through directly.
    """

    l1: float = 0.00

    fc_layer: bool = False


class SimpleTabularModel(nn.Module):
    def __init__(
        self,
        model_init_config: SimpleTabularModelConfig,
        cat_columns: Sequence[str],
        con_columns: Sequence[str],
        unique_label_values_per_column: Dict[str, Set[str]],
        device: str,
    ):
        """
        Note: It would be more natural maybe to do the lookup here
        (using self.embeddings_dict), but then we also have to do all tensor
        preparation (e.g. mixing) here. Perhaps for now better to do it outside in
        preparation hook. However, this way also keeps a common interface between all
        current models, where they are taking a tensor as input (otherwise, we would
        be taking a dict here).

        Note: Currently the only thing this model does is looking up the trainable
        embeddings. Other types of tabular models can be added later, with their own
        layers.
        """

        super().__init__()

        self.cat_columns = cat_columns
        self.con_columns = con_columns
        self.unique_label_values = unique_label_values_per_column
        self.device = device

        self.embeddings_dict = set_up_embedding_dict(
            unique_label_values=unique_label_values_per_column
        )

        emb_total_dim = 0
        con_total_dim = 0

        if self.embeddings_dict:
            emb_total_dim = attach_embeddings(self, self.embeddings_dict)
        if self.con_columns:
            con_total_dim = len(self.con_columns)

        self.input_dim = emb_total_dim + con_total_dim

        self.layer = nn.Identity()
        if model_init_config.fc_layer:
            self.layer = nn.Linear(
                in_features=self.input_dim, out_features=self.input_dim, bias=False
            )

    @property
    def num_out_features(self) -> int:
        return self.input_dim

    def script_submodules_for_tracing(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Embedding):
                scripted_module = torch.jit.script(obj=module)
                setattr(self, name, scripted_module)

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        if not self.cat_columns:
            return torch.empty(0)

        return torch.cat([torch.flatten(i) for i in self.parameters()])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layer(input)


def set_up_embedding_dict(
    unique_label_values: al_unique_embed_vals,
) -> al_emb_lookup_dict:
    emb_lookup_dict = {}

    for emb_col, emb_uniq_values in unique_label_values.items():
        sorted_unique_values = sorted(emb_uniq_values)
        lookup_table = {k: idx for idx, k in enumerate(sorted_unique_values)}
        emb_lookup_dict[emb_col] = {"lookup_table": lookup_table}

    return emb_lookup_dict


def attach_embeddings(model: nn.Module, embeddings_dict: al_emb_lookup_dict) -> int:
    total_emb_dimension = 0

    for emb_col in sorted(embeddings_dict.keys()):
        n_categories = len(embeddings_dict[emb_col]["lookup_table"])
        cur_embedding_dim = calc_embedding_dimension(n_categories=n_categories)

        embedding_module = nn.Embedding(
            num_embeddings=n_categories, embedding_dim=cur_embedding_dim
        )
        setattr(model, "embed_" + emb_col, embedding_module)

        total_emb_dimension += cur_embedding_dim

    return total_emb_dimension


def calc_embedding_dimension(n_categories: int) -> int:
    return min(600, round(1.6 * n_categories**0.56))


def get_tabular_inputs(
    input_cat_columns: Sequence[str],
    input_con_columns: Sequence[str],
    tabular_model: SimpleTabularModel,
    tabular_input: "al_tabular_input",
    device: str,
) -> Union[torch.Tensor, None]:
    """
    We want to have a wrapper function to gather all extra inputs needed by the model.
    """
    extra_embeddings = None
    if input_cat_columns:

        extra_embeddings = get_embeddings_from_labels(
            categorical_columns=input_cat_columns,
            labels=tabular_input,
            model=tabular_model,
        )

        if not input_con_columns:
            return extra_embeddings.to(device=device)

    extra_continuous = None
    if input_con_columns:

        extra_continuous = get_extra_continuous_inputs_from_labels(
            labels=tabular_input, continuous_columns=input_con_columns
        )

        if not input_cat_columns:
            return extra_continuous.to(device=device)

    if extra_continuous is not None and extra_embeddings is not None:
        concat_emb_and_con = torch.cat((extra_embeddings, extra_continuous), dim=1)
        return concat_emb_and_con.to(device=device)

    return None


def get_embeddings_from_labels(
    categorical_columns: Sequence[str],
    labels: Dict[str, Sequence[torch.Tensor]],
    model: SimpleTabularModel,
) -> torch.Tensor:

    """
    Note that the extra_embeddings is a list of tensors, where each tensor is a batch
    of embeddings for a given extra categorical column.

    Note also that we do not need categorical_columns here, we can grab them from the
    embeddings dict directly. But we use that so the order of embeddings in the
    concatenated tensor is clear.
    """

    if labels is None:
        raise ValueError("No extra labels found for when looking up embeddings.")

    extra_embeddings = []
    for col_key in categorical_columns:
        cur_embedding = lookup_embeddings(
            model=model,
            embedding_col=col_key,
            labels=labels[col_key],
        )
        extra_embeddings.append(cur_embedding)

    extra_embeddings = torch.cat(extra_embeddings, dim=1)

    return extra_embeddings


def lookup_embeddings(
    model: SimpleTabularModel,
    embedding_col: str,
    labels: Sequence[torch.Tensor],
) -> torch.Tensor:
    """
    This produces a batch of embeddings, with dimensions N x embed_dim.
    """

    cur_embedding_module: nn.Embedding = getattr(model, "embed_" + embedding_col)
    cur_embedding = cur_embedding_module(labels)

    return cur_embedding


def get_unique_embed_values(
    labels_dict: "al_label_dict", embedding_cols: Sequence[str]
) -> al_unique_embed_vals:
    unique_embeddings_dict = OrderedDict((i, set()) for i in sorted(embedding_cols))

    for sample_labels in labels_dict.values():
        for key, value in sample_labels.items():
            if key in embedding_cols:
                assert isinstance(value, int)
                unique_embeddings_dict[key].add(value)

    return unique_embeddings_dict


def get_extra_continuous_inputs_from_labels(
    labels: Dict[str, torch.Tensor], continuous_columns: Iterable[str]
) -> torch.Tensor:

    extra_continuous = []
    for col in continuous_columns:
        cur_con_labels = labels[col].unsqueeze(1).to(dtype=torch.float)
        extra_continuous.append(cur_con_labels)

    extra_continuous = torch.cat(extra_continuous, dim=1)

    return extra_continuous


@overload
def set_up_and_save_embeddings_dict(
    embedding_columns: None, labels_dict: "al_label_dict", run_folder: Path
) -> None:
    ...


@overload
def set_up_and_save_embeddings_dict(
    embedding_columns: List[str], labels_dict: "al_label_dict", run_folder: Path
) -> al_emb_lookup_dict:
    ...


def set_up_and_save_embeddings_dict(embedding_columns, labels_dict, run_folder):
    """
    We need to save it for test time models to be able to load.
    """
    if embedding_columns:
        embedding_dict = get_embedding_dict(
            labels_dict=labels_dict, embedding_cols=embedding_columns
        )
        outpath = run_folder / "extra_inputs" / "embeddings.save"
        ensure_path_exists(path=outpath)
        joblib.dump(value=embedding_dict, filename=outpath)
        return embedding_dict

    return None


def get_embedding_dict(
    labels_dict: "al_label_dict", embedding_cols: List[str]
) -> al_emb_lookup_dict:
    """
    Simple wrapper function to call other embedding functions to create embedding
    dictionary.
    """
    unique_embs = get_unique_embed_values(
        labels_dict=labels_dict, embedding_cols=embedding_cols
    )
    emb_lookup_dict = set_up_embedding_dict(unique_label_values=unique_embs)

    return emb_lookup_dict


def get_unique_values_from_transformers(
    transformers: Dict[str, Union[LabelEncoder, StandardScaler]],
    keys_to_use: Union[str, Iterable[str]],
) -> Dict[str, Any]:

    out = {}

    if not keys_to_use:
        return out

    if keys_to_use == "all":
        iterable = transformers.keys()
    else:
        iterable = keys_to_use

    for k in iterable:
        out[k] = set(transformers[k].classes_)

    return out
