from eir.data_load import label_setup
from eir.setup.output_setup import set_up_num_outputs_per_target


def test_set_up_num_classes(get_transformer_test_data):
    df_test, test_target_columns_dict = get_transformer_test_data

    test_transformers = label_setup._get_fit_label_transformers(
        df_labels=df_test, label_columns=test_target_columns_dict
    )

    num_classes = set_up_num_outputs_per_target(target_transformers=test_transformers)

    assert num_classes["Height"] == 1
    assert num_classes["Origin"] == 3
