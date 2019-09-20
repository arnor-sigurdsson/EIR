import pytest

import torch
from torch import nn
from human_origins_supervised.models import embeddings as emb


@pytest.fixture
def create_emb_test_label_data():
    test_label_dict = {
        "ID1": {
            "Origin": "Iceland",
            "Climate": "Cool",
            "Population": "Small",
            "Food": "Fish",
        },
        "ID2": {
            "Origin": "Mexico",
            "Climate": "Warm",
            "Population": "Large",
            "Food": "Tacos",
        },
    }

    emb_cols = ["Climate", "Population", "Food"]
    return test_label_dict, emb_cols


@pytest.fixture
def create_test_model():
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x

    model = TestModel()

    return model


def test_get_unique_embed_values(create_emb_test_label_data):
    test_label_dict, emb_cols = create_emb_test_label_data

    unique_emb_dict = emb.get_unique_embed_values(test_label_dict, emb_cols)
    assert len(unique_emb_dict) == 3
    assert unique_emb_dict["Climate"] == {"Cool", "Warm"}
    assert unique_emb_dict["Population"] == {"Small", "Large"}
    assert unique_emb_dict["Food"] == {"Tacos", "Fish"}


def test_set_up_embedding_lookups(create_emb_test_label_data):
    test_label_dict, emb_cols = create_emb_test_label_data

    unique_emb_dict = emb.get_unique_embed_values(test_label_dict, emb_cols)
    emb_lookup_dict = emb.set_up_embedding_lookups(unique_emb_dict)
    assert len(emb_lookup_dict) == 3

    for key, value_dict in emb_lookup_dict.items():
        assert key in emb_cols
        assert "lookup_table" in value_dict
        assert len(value_dict["lookup_table"]) == 2
        assert set(value_dict["lookup_table"].values()) == {0, 1}


def test_calc_embedding_dimension():
    assert emb.calc_embedding_dimension(2) == 2
    assert emb.calc_embedding_dimension(60) > 10


def test_attach_embeddings(create_emb_test_label_data, create_test_model):
    test_label_dict, emb_cols = create_emb_test_label_data
    test_model = create_test_model

    emb_dict = emb.get_embedding_dict(test_label_dict, emb_cols)

    total_emb_dimensions = emb.attach_embeddings(test_model, emb_dict)

    # 2 for each column
    assert total_emb_dimensions == 6

    for col in emb_cols:
        embedding_attr_name = "embed_" + col
        assert hasattr(test_model, embedding_attr_name)

        cur_embedding = getattr(test_model, embedding_attr_name)
        assert isinstance(cur_embedding, nn.Embedding)
        assert cur_embedding.num_embeddings == 2
        assert cur_embedding.embedding_dim == 2


def test_lookup_embeddings(create_emb_test_label_data, create_test_model):
    test_label_dict, emb_cols = create_emb_test_label_data
    test_model = create_test_model

    emb_dict = emb.get_embedding_dict(test_label_dict, emb_cols)
    emb.attach_embeddings(test_model, emb_dict)

    test_extra_labels = [{"Population": "Small"}]

    cur_lookup_table = emb_dict["Population"]["lookup_table"]
    extra_label_emb_index = torch.tensor(cur_lookup_table["Small"], dtype=torch.long)
    cur_embedding = emb.lookup_embeddings(
        test_model, emb_dict, "Population", test_extra_labels, "cpu"
    )

    model_embedding = test_model.embed_Population(extra_label_emb_index)
    assert (cur_embedding == model_embedding).all()
