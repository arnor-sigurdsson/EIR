import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from eir.train_utils import evaluation


def test_parse_valid_classification_preds():
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
    val_classes = ["Ember", "Ash"]

    test_ids = np.array(["Test_ID_" + str(i) for i in range(1, 11)])

    df_valid_preds = evaluation._parse_valid_classification_predictions(
        val_true=test_val_true,
        val_predictions=test_val_preds,
        val_outputs=test_val_probs,
        val_classes=val_classes,
        ids=test_ids,
    )

    assert df_valid_preds.shape[0] == 10
    assert set(df_valid_preds["ID"]) == {"Test_ID_" + str(i) for i in range(1, 11)}
    assert (df_valid_preds["True_Label"] == test_val_true).all()
    assert (df_valid_preds["Predicted"] == test_val_preds).all()

    assert (df_valid_preds["Score Class Ember"] == test_val_probs[:, 0]).all()
    assert (df_valid_preds["Score Class Ash"] == test_val_probs[:, 1]).all()


def test_inverse_numerical_labels_hook():
    test_target_transformer = LabelEncoder()
    test_target_transformer.fit(["Asia", "Europe"])

    test_df = pd.DataFrame(
        columns=["True_Label", "Predicted"], data=[[0, 1], [0, 1], [1, 0], [1, 0]]
    )

    test_df_encoded = evaluation._inverse_numerical_labels_hook(
        test_df, test_target_transformer
    )

    assert list(test_df_encoded["True_Label"]) == ["Asia"] * 2 + ["Europe"] * 2
    assert list(test_df_encoded["Predicted"]) == ["Europe"] * 2 + ["Asia"] * 2
