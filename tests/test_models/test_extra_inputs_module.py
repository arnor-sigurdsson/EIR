import pytest

import torch
from torch import nn
from human_origins_supervised.models import extra_inputs_module as emb


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


def test_attach_embeddings(create_emb_test_label_data, create_test_emb_model):
    test_label_dict, emb_cols = create_emb_test_label_data
    test_model = create_test_emb_model

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


def test_lookup_embeddings(create_emb_test_label_data, create_test_emb_model):
    test_label_dict, emb_cols = create_emb_test_label_data
    test_model = create_test_emb_model

    emb_dict = emb.get_embedding_dict(
        labels_dict=test_label_dict, embedding_cols=emb_cols
    )
    emb.attach_embeddings(model=test_model, embeddings_dict=emb_dict)

    cur_lookup_table = emb_dict["Population"]["lookup_table"]
    extra_label_emb_index = torch.tensor(cur_lookup_table["Small"], dtype=torch.long)

    test_extra_labels = ["Small"]
    cur_embedding = emb.lookup_embeddings(
        model=test_model,
        embeddings_dict=emb_dict,
        embedding_col="Population",
        extra_labels=test_extra_labels,
        device="cpu",
    )

    model_embedding = test_model.embed_Population(extra_label_emb_index)
    assert (cur_embedding == model_embedding).all()


def test_get_embeddings_from_labels(create_emb_test_label_data, create_test_emb_model):
    """
    id:

    "ID1": {
            "Origin": "Iceland",
            "Climate": "Cool",
            "Population": "Small",
            "Food": "Fish",
        }


    emb_dict:

    {'Climate':{'lookup_table': {'Cool': 0, 'Warm': 1}},
    'Population': {'lookup_table': {'Large': 0, 'Small': 1}},
    'Food': {'lookup_table': {'Fish': 0, 'Tacos': 1}}}
    """
    test_label_dict, emb_cols = create_emb_test_label_data
    test_model = create_test_emb_model

    emb_dict = emb.get_embedding_dict(
        labels_dict=test_label_dict, embedding_cols=emb_cols
    )
    emb.attach_embeddings(model=test_model, embeddings_dict=emb_dict)

    test_model.embeddings_dict = emb_dict

    test_extra_labels = {"Climate": ["Cool"], "Population": ["Small"], "Food": ["Fish"]}
    test_embeddings = emb.get_embeddings_from_labels(
        extra_labels=test_extra_labels, model=test_model, device="cpu"
    )

    assert test_embeddings.shape[1] == 6

    # check climate, "Cool" at index 0
    id1_emb_climate = test_embeddings[:, :2]
    assert (id1_emb_climate == test_model.embed_Climate(torch.tensor(0))).all().item()

    # check food, "Fish" at index 0
    id1_emb_food = test_embeddings[:, 2:4]
    assert (id1_emb_food == test_model.embed_Food(torch.tensor(0))).all().item()

    # check population, "Small" at index 1
    id_emb_pop = test_embeddings[:, 4:]
    assert (id_emb_pop == test_model.embed_Population(torch.tensor(1))).all().item()
