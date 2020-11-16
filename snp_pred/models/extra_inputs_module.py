from argparse import Namespace
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict, Set, Union, overload, TYPE_CHECKING

import joblib
import torch
from aislib.misc_utils import ensure_path_exists
from torch import nn

from snp_pred.data_load.label_setup import al_label_dict

if TYPE_CHECKING:
    from snp_pred.train import al_training_labels_extra

# Aliases
al_unique_embed_vals = Dict[str, Set[str]]
al_emb_lookup_dict = Dict[str, Dict[str, Dict[str, int]]]


def get_unique_embed_values(
    labels_dict: al_label_dict, embedding_cols: List[str]
) -> al_unique_embed_vals:
    unique_embeddings_dict = OrderedDict((i, set()) for i in sorted(embedding_cols))

    for sample_labels in labels_dict.values():
        for key, value in sample_labels.items():
            if key in embedding_cols:
                unique_embeddings_dict[key].add(value)

    return unique_embeddings_dict


def set_up_embedding_lookups(
    unique_emb_dict: al_unique_embed_vals,
) -> al_emb_lookup_dict:
    emb_lookup_dict = {}

    for emb_col, emb_uniq_values in unique_emb_dict.items():
        sorted_unique_values = sorted(emb_uniq_values)
        lookup_table = {k: idx for idx, k in enumerate(sorted_unique_values)}
        emb_lookup_dict[emb_col] = {"lookup_table": lookup_table}

    return emb_lookup_dict


def get_embedding_dict(
    labels_dict: al_label_dict, embedding_cols: List[str]
) -> al_emb_lookup_dict:
    """
    Simple wrapper function to call other embedding functions to create embedding
    dictionary.
    """
    unique_embs = get_unique_embed_values(labels_dict, embedding_cols)
    emb_lookup_dict = set_up_embedding_lookups(unique_embs)

    return emb_lookup_dict


@overload
def set_up_and_save_embeddings_dict(
    embedding_columns: None, labels_dict: al_label_dict, run_folder: Path
) -> None:
    ...


@overload
def set_up_and_save_embeddings_dict(
    embedding_columns: List[str], labels_dict: al_label_dict, run_folder: Path
) -> al_emb_lookup_dict:
    ...


def set_up_and_save_embeddings_dict(embedding_columns, labels_dict, run_folder):
    """
    We need to save it for test time models to be able to load.
    """
    if embedding_columns:
        embedding_dict = get_embedding_dict(labels_dict, embedding_columns)
        outpath = run_folder / "extra_inputs" / "embeddings.save"
        ensure_path_exists(outpath)
        joblib.dump(embedding_dict, outpath)
        return embedding_dict

    return None


def calc_embedding_dimension(n_categories: int) -> int:
    return min(600, round(1.6 * n_categories ** 0.56))


def attach_embeddings(model: nn.Module, embeddings_dict: al_emb_lookup_dict) -> int:
    total_emb_dimension = 0

    for emb_col in sorted(embeddings_dict.keys()):
        n_categories = len(embeddings_dict[emb_col]["lookup_table"])
        cur_embedding_dim = calc_embedding_dimension(n_categories)

        embedding_module = nn.Embedding(n_categories, cur_embedding_dim)
        setattr(model, "embed_" + emb_col, embedding_module)

        total_emb_dimension += cur_embedding_dim

    return total_emb_dimension


def get_extra_continuous_inputs_from_labels(
    extra_labels: Dict[str, torch.Tensor], continuous_columns: List[str]
) -> torch.Tensor:

    extra_continuous = []
    for col in continuous_columns:
        cur_con_labels = extra_labels[col].unsqueeze(1).to(dtype=torch.float)
        extra_continuous.append(cur_con_labels)

    extra_continuous = torch.cat(extra_continuous, dim=1)

    return extra_continuous


def lookup_embeddings(
    model: nn.Module,
    embeddings_dict: al_emb_lookup_dict,
    embedding_col: str,
    extra_labels: List[str],
    device: str,
) -> torch.Tensor:
    """
    This produces a batch of embeddings, with dimensions N x embed_dim.
    """
    cur_lookup_table = embeddings_dict[embedding_col]["lookup_table"]
    cur_lookup_indexes = [cur_lookup_table.get(i) for i in extra_labels]
    cur_lookup_indexes = torch.tensor(cur_lookup_indexes, dtype=torch.long)
    cur_lookup_indexes = cur_lookup_indexes.to(device)

    cur_embedding_module = getattr(model, "embed_" + embedding_col)
    cur_embedding = cur_embedding_module(cur_lookup_indexes)

    return cur_embedding


def get_embeddings_from_labels(
    extra_labels: Dict[str, List[str]], model: nn.Module, device: str
) -> torch.Tensor:

    """
    Note that the extra_embeddings is a list of tensors, where each tensor is a batch
    of embeddings for a given extra categorical column.
    """

    if not extra_labels:
        raise ValueError("No extra labels found for when looking up embeddings.")

    extra_embeddings = []
    for col_key in model.embeddings_dict:
        cur_embedding = lookup_embeddings(
            model=model,
            embeddings_dict=model.embeddings_dict,
            embedding_col=col_key,
            extra_labels=extra_labels[col_key],
            device=device,
        )
        extra_embeddings.append(cur_embedding)

    extra_embeddings = torch.cat(extra_embeddings, dim=1)

    return extra_embeddings


def get_extra_inputs(
    cl_args: Namespace, model: nn.Module, labels: "al_training_labels_extra"
) -> Union[torch.Tensor, None]:
    """
    We want to have a wrapper function to gather all extra inputs needed by the model.
    """
    extra_embeddings = None
    if cl_args.extra_cat_columns:

        extra_embeddings = get_embeddings_from_labels(
            extra_labels=labels, model=model, device=cl_args.device
        )

        if not cl_args.extra_con_columns:
            return extra_embeddings.to(device=cl_args.device)

    extra_continuous = None
    if cl_args.extra_con_columns:

        extra_continuous = get_extra_continuous_inputs_from_labels(
            extra_labels=labels, continuous_columns=cl_args.extra_con_columns
        )

        if not cl_args.extra_cat_columns:
            return extra_continuous.to(device=cl_args.device)

    if extra_continuous is not None and extra_embeddings is not None:
        concat_emb_and_con = torch.cat((extra_embeddings, extra_continuous), dim=1)
        return concat_emb_and_con.to(device=cl_args.device)

    return None