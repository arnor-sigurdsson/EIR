from collections import OrderedDict
from collections.abc import Sequence

import pytest
import torch
from torch import nn

from eir.data_load.label_setup import al_label_dict
from eir.models.input.tabular import tabular as tab
from eir.models.input.tabular.tabular import (
    al_emb_lookup_dict,
    al_unique_embed_vals,
    set_up_embedding_dict,
)


@pytest.fixture
def create_emb_test_label_data():
    test_label_dict = {
        "ID1": {
            "Origin": 0,  # Iceland
            "Climate": 0,  # Cool
            "Population": 0,  # Small
            "Food": 0,  # Fish
        },
        "ID2": {
            "Origin": 1,  # Mexico
            "Climate": 1,  # Warm
            "Population": 1,  # Large
            "Food": 1,  # Tacos
        },
    }

    emb_cols = ["Climate", "Population", "Food"]
    return test_label_dict, emb_cols


@pytest.fixture
def create_test_emb_model():
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x

    model = TestModel()

    return model


def test_get_unique_embed_values(create_emb_test_label_data):
    test_label_dict, emb_cols = create_emb_test_label_data

    unique_emb_dict = get_unique_embed_values(
        labels_dict=test_label_dict, embedding_cols=emb_cols
    )
    assert len(unique_emb_dict) == 3
    assert unique_emb_dict["Climate"] == {0, 1}  # Cool, Warm
    assert unique_emb_dict["Population"] == {0, 1}  # Small, Large
    assert unique_emb_dict["Food"] == {0, 1}  # Fish, Tacos


def test_set_up_embedding_lookups(create_emb_test_label_data):
    test_label_dict, emb_cols = create_emb_test_label_data

    unique_emb_dict = get_unique_embed_values(test_label_dict, emb_cols)
    emb_lookup_dict = tab.set_up_embedding_dict(unique_emb_dict)
    assert len(emb_lookup_dict) == 3

    for key, value_dict in emb_lookup_dict.items():
        assert key in emb_cols
        assert "lookup_table" in value_dict
        assert len(value_dict["lookup_table"]) == 2
        assert set(value_dict["lookup_table"].values()) == {0, 1}


def test_calc_embedding_dimension():
    assert tab.calc_embedding_dimension(2) == 2
    assert tab.calc_embedding_dimension(60) > 10


def test_attach_embeddings(create_emb_test_label_data, create_test_emb_model):
    test_label_dict, emb_cols = create_emb_test_label_data
    test_model = create_test_emb_model

    emb_dict = get_embedding_dict(test_label_dict, emb_cols)

    total_emb_dimensions = tab.attach_embeddings(test_model, emb_dict)

    # 2 for each column
    assert total_emb_dimensions == 6

    for col in emb_cols:
        embedding_attr_name = "embed_" + col
        assert hasattr(test_model, embedding_attr_name)

        cur_embedding = getattr(test_model, embedding_attr_name)
        assert isinstance(cur_embedding, nn.Embedding)
        assert cur_embedding.num_embeddings == 2
        assert cur_embedding.embedding_dim == 2


def test_lookup_embeddings(create_emb_test_label_data, create_test_emb_model):
    test_label_dict, emb_cols = create_emb_test_label_data
    test_model = create_test_emb_model

    emb_dict = get_embedding_dict(labels_dict=test_label_dict, embedding_cols=emb_cols)
    tab.attach_embeddings(model=test_model, embeddings_dict=emb_dict)

    cur_lookup_table = emb_dict["Population"]["lookup_table"]
    extra_label_emb_index = torch.tensor(cur_lookup_table[0], dtype=torch.long)

    test_extra_labels = torch.LongTensor([0])
    cur_embedding = tab.lookup_embeddings(
        model=test_model,
        embedding_col="Population",
        labels=test_extra_labels,
    )

    model_embedding = test_model.embed_Population(extra_label_emb_index)
    assert (cur_embedding == model_embedding).all()


def test_get_embeddings_from_labels(create_emb_test_label_data, create_test_emb_model):
    """
    id:

    "ID1": {
            "Origin": 0 # "Iceland",
            "Climate": 0 # "Cool",
            "Population": 0 # "Small",
            "Food": 0 # "Fish",
        }


    emb_dict, label meaning:

    {'Climate':{'lookup_table': {'Cool': 0, 'Warm': 1}},
    'Population': {'lookup_table': {'Large': 0, 'Small': 1}},
    'Food': {'lookup_table': {'Fish': 0, 'Tacos': 1}}}

    emb_dict, actual numeric values:

    {'Climate':{'lookup_table': {0: 0, 1: 1}},
    'Population': {'lookup_table': {0: 0, 1: 1}},
    'Food': {'lookup_table': {0: 0, 1: 1}}}
    """
    test_label_dict, emb_cols = create_emb_test_label_data
    test_model = create_test_emb_model

    emb_dict = get_embedding_dict(labels_dict=test_label_dict, embedding_cols=emb_cols)
    tab.attach_embeddings(model=test_model, embeddings_dict=emb_dict)

    test_model.embeddings_dict = emb_dict

    test_extra_labels = {
        k: torch.LongTensor([0]) for k in ("Climate", "Food", "Population")
    }

    test_embeddings = tab.get_embeddings_from_labels(
        categorical_columns=emb_cols, labels=test_extra_labels, model=test_model
    )

    assert test_embeddings.shape[1] == 6

    # check climate, "Cool" at index 0
    id1_emb_climate = test_embeddings[:, :2]
    assert (
        (id1_emb_climate == test_model.embed_Climate(torch.LongTensor([0])))
        .all()
        .item()
    )

    # check population, "Small" at index 1
    id1_emb_population = test_embeddings[:, 2:4]
    assert (
        (id1_emb_population == test_model.embed_Population(torch.LongTensor([0])))
        .all()
        .item()
    )

    # check food, "Fish" at index 0
    id_emb_food = test_embeddings[:, 4:]
    assert (id_emb_food == test_model.embed_Food(torch.LongTensor([0]))).all().item()


def get_embedding_dict(
    labels_dict: "al_label_dict", embedding_cols: list[str]
) -> al_emb_lookup_dict:
    """
    Simple wrapper function to call other embedding functions to create embedding
    dictionary.
    """
    unique_embeddings = get_unique_embed_values(
        labels_dict=labels_dict,
        embedding_cols=embedding_cols,
    )
    embedding_lookup_dict = set_up_embedding_dict(
        unique_label_values=unique_embeddings,
    )

    return embedding_lookup_dict


def get_unique_embed_values(
    labels_dict: "al_label_dict", embedding_cols: Sequence[str]
) -> al_unique_embed_vals:
    unique_embeddings_dict: al_unique_embed_vals = OrderedDict(
        (i, set()) for i in sorted(embedding_cols)
    )

    for sample_labels in labels_dict.values():
        for key, value in sample_labels.items():
            if key in embedding_cols:
                assert isinstance(value, int)
                unique_embeddings_dict[key].add(value)

    return unique_embeddings_dict
