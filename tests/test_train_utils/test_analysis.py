from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from human_origins_supervised.train_utils import evaluation


def test_get_most_wrong_preds():
    test_val_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    test_val_preds = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0])
    test_val_probs = np.array(
        [
            [0.1, 0.9],  # wrong, 1
            [0.2, 0.8],  # wrong, 2
            [0.3, 0.7],  # wrong, 3
            [0.9, 0.1],  # correct, 4
            [0.9, 0.1],  # correct, 5
            [0.1, 0.9],  # correct, 6
            [0.2, 0.8],  # correct, 7
            [0.95, 0.05],  # wrong, 8
            [0.85, 0.15],  # wrong, 9
            [0.75, 0.25],  # wrong, 10
        ]
    )
    test_ids = np.array(["Test_ID_" + str(i) for i in range(1, 11)])

    def dummy_function(x, *args, **kwargs):
        return x

    # patch so we get back the probs above after softmax
    with patch(
        "human_origins_supervised.train_utils.evaluation.softmax",
        side_effect=dummy_function,
        autospec=True,
    ):
        df_most_wrong = evaluation.get_most_wrong_cls_preds(
            test_val_true, test_val_preds, test_val_probs, test_ids
        )

    assert df_most_wrong.shape[0] == 6
    assert list(df_most_wrong["Sample_ID"]) == [
        "Test_ID_" + str(i) for i in (8, 1, 9, 2, 10, 3)
    ]
    assert df_most_wrong["True_Prob"].iloc[0] == 0.05
    assert df_most_wrong["True_Prob"].iloc[-1] == 0.3

    assert ((df_most_wrong["True_Prob"] + df_most_wrong["Wrong_Prob"]) == 1.0).all()

    assert (df_most_wrong["True_Label"] != df_most_wrong["Wrong_Label"]).all()


def test_inverse_numerical_labels_hook():
    test_target_transformer = LabelEncoder()
    test_target_transformer.fit(["Asia", "Europe"])

    test_df = pd.DataFrame(
        columns=["True_Label", "Wrong_Label"], data=[[0, 1], [0, 1], [1, 0], [1, 0]]
    )

    test_df_encoded = evaluation.inverse_numerical_labels_hook(
        test_df, test_target_transformer
    )

    assert list(test_df_encoded["True_Label"]) == ["Asia"] * 2 + ["Europe"] * 2
    assert list(test_df_encoded["Wrong_Label"]) == ["Europe"] * 2 + ["Asia"] * 2


def test_anno_meta_hook(tmp_path):
    test_anno_data = [
        ["S1", "GL_Asia", "Tokyo", "Japan"],
        ["S2", "GL_Europe", "Hafnarfjordur", "Iceland"],
        ["S3", "GL_Africa", "Nairobi", "Kenya"],
        ["S4", "GL_Asia", "Tokyo", "Japan"],
        ["S5", "GL_Europe", "Hafnarfjordur", "Iceland"],
    ]

    anno_columns = ["Instance ID", "Group Label", "Location", "Country"]

    test_df_anno = pd.DataFrame(columns=anno_columns, data=test_anno_data)
    test_anno_fpath = tmp_path / "test_anno.csv"
    test_df_anno.to_csv(test_anno_fpath, sep="\t", index=False)

    data_base = ["S1_-_Asia", "S2_-_Europe", "S3_-_Africa", "S4_-_Asia", "S5_-_Europe"]
    data = [[i, "TestValue"] for i in data_base]
    test_df = pd.DataFrame(columns=["Sample_ID", "TestColumn"], data=data)

    anno_metad_df = evaluation.anno_meta_hook(test_df, anno_fpath=test_anno_fpath)

    assert anno_metad_df.shape[0] == 5

    assert set(anno_metad_df["TestColumn"]) == {"TestValue"}

    s1_row = anno_metad_df.loc[anno_metad_df["Sample_ID"] == "S1"]
    assert s1_row.Location.values[0] == "Tokyo"
    assert s1_row.Country.values[0] == "Japan"

    s5_row = anno_metad_df.loc[anno_metad_df["Sample_ID"] == "S5"]
    assert s5_row.Location.values[0] == "Hafnarfjordur"
    assert s5_row.Country.values[0] == "Iceland"
