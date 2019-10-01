from typing import List, Dict, Set

import torch
from torch import nn
from human_origins_supervised.data_load.label_setup import al_label_dict

# Aliases
al_unique_embed_vals = Dict[str, Set[str]]
al_emb_lookup_dict = Dict[str, Dict[str, Dict[str, int]]]


def get_unique_embed_values(
    labels_dict: al_label_dict, embedding_cols: List[str]
) -> al_unique_embed_vals:
    unique_embeddings_dict = {i: set() for i in embedding_cols}

    for sample_labels in labels_dict.values():
        for key, value in sample_labels.items():
            if key in embedding_cols:
                unique_embeddings_dict[key].add(value)

    return unique_embeddings_dict


def set_up_embedding_lookups(
    unique_emb_dict: al_unique_embed_vals
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


def calc_embedding_dimension(n_categories: int) -> int:
    return min(600, round(1.6 * n_categories ** 0.56))


def attach_embeddings(model: nn.Module, embeddings_dict: al_emb_lookup_dict) -> int:
    total_emb_dimension = 0

    for emb_col in embeddings_dict:
        n_categories = len(embeddings_dict[emb_col]["lookup_table"])
        cur_embedding_dim = calc_embedding_dimension(n_categories)

        embedding_module = nn.Embedding(n_categories, cur_embedding_dim)
        setattr(model, "embed_" + emb_col, embedding_module)

        total_emb_dimension += cur_embedding_dim

    return total_emb_dimension


def lookup_embeddings(
    model: nn.Module,
    embeddings_dict: al_emb_lookup_dict,
    embedding_col: str,
    extra_labels: List[Dict[str, str]],
    device: str,
):
    cur_lookup_table = embeddings_dict[embedding_col]["lookup_table"]
    cur_lookup_indexes = [cur_lookup_table.get(i[embedding_col]) for i in extra_labels]
    cur_lookup_indexes = torch.tensor(cur_lookup_indexes, dtype=torch.long)
    cur_lookup_indexes = cur_lookup_indexes.to(device)

    cur_embedding_module = getattr(model, "embed_" + embedding_col)
    cur_embedding = cur_embedding_module(cur_lookup_indexes)

    return cur_embedding
