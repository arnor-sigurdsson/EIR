import pytest

from human_origins_supervised.data_load import label_setup
from human_origins_supervised.data_load.common_ops import ColumnOperation


@pytest.fixture()
def get_test_column_ops():
    def test_column_op_1(df, column_name, replace_dict):
        df = df.replace({column_name: replace_dict})
        return df

    def test_column_op_2(df, column_name, multiplier):
        df[column_name] = df[column_name] * multiplier
        return df

    def test_column_op_3(df, column_name, replace_with_col):
        df[column_name] = df[replace_with_col]
        return df

    replace_dict_args = {"replace_dict": {"Europe": "Iceland"}}
    multiplier_dict_arg = {"multiplier": 2}
    replace_column_dict_arg = {"replace_with_col": "ExtraCol3"}
    test_column_ops = {
        "Origin": [
            ColumnOperation(test_column_op_1, replace_dict_args),
            ColumnOperation(test_column_op_2, multiplier_dict_arg),
        ],
        "OriginExtraColumns": [
            ColumnOperation(
                test_column_op_1, replace_dict_args, ("ExtraCol1", "ExtraCol2")
            ),
            ColumnOperation(test_column_op_3, replace_column_dict_arg, ("ExtraCol3",)),
        ],
    }

    return test_column_ops


def test_get_extra_columns(get_test_column_ops):
    test_column_ops = get_test_column_ops

    extra_columns = label_setup.get_extra_columns("OriginExtraColumns", test_column_ops)
    assert extra_columns == ["ExtraCol1", "ExtraCol2", "ExtraCol3"]

    extra_columns_empty = label_setup.get_extra_columns("Origin", test_column_ops)
    assert extra_columns_empty == []


@pytest.mark.parametrize(
    "create_test_data",
    [
        {"class_type": "binary", "data_type": "uint8"},
        {"class_type": "multi", "data_type": "uint8"},
    ],
    indirect=True,
)
def test_load_label_df(request, create_test_data):
    path, test_data_params = create_test_data
    label_fpath = path / "labels.csv"
    n_classes = 2 if test_data_params.get("class_type") == "binary" else 3

    n_per_class = request.config.getoption("--num_samples_per_class")

    df_label = label_setup.load_label_df(label_fpath, "Origin")

    assert df_label.shape[0] == n_per_class * n_classes
    assert df_label.index.name == "ID"
    assert [i for i in df_label.Origin.value_counts()] == [n_per_class] * n_classes

    df_label["ExtraCol"] = "ExtraVal"
    label_extra_fpath = path / "labels_extracol.csv"
    df_label.to_csv(label_extra_fpath)

    df_label_extra = label_setup.load_label_df(
        label_extra_fpath, "Origin", extra_columns=("ExtraCol",)
    )

    assert df_label_extra.shape[1] == 2
    assert df_label_extra["ExtraCol"].unique().item() == "ExtraVal"

    df_label_ids = label_setup.load_label_df(
        label_fpath, "Origin", ids_to_keep=("95_Europe", "96_Europe", "97_Europe")
    )
    assert df_label_ids.shape[0] == 3


@pytest.mark.parametrize(
    "create_test_data", [{"class_type": "binary", "data_type": "uint8"}], indirect=True
)
def test_parse_label_df(create_test_data, get_test_column_ops):
    path, test_data_params = create_test_data
    label_fpath = path / "labels.csv"

    test_column_ops = get_test_column_ops

    df_label = label_setup.load_label_df(label_fpath, "Origin")
    df_label_parsed = label_setup.parse_label_df(df_label, test_column_ops, "Origin")

    assert set(df_label_parsed.Origin.unique()) == {"Iceland" * 2, "Asia" * 2}

    extra_cols = ("ExtraCol3",)
    for col in extra_cols:
        df_label[col] = "Iceland"

    df_label = df_label.rename(columns={"Origin": "OriginExtraColumns"})
    df_label_parsed = label_setup.parse_label_df(df_label, test_column_ops, "Origin")
    assert df_label_parsed["OriginExtraColumns"].unique().item() == "Iceland"


@pytest.mark.parametrize(
    "create_test_data", [{"class_type": "binary", "data_type": "uint8"}], indirect=True
)
def test_label_df_parse_wrapper(request, create_test_data, create_test_cl_args):
    cl_args = create_test_cl_args
    df_labels = label_setup.label_df_parse_wrapper(cl_args)

    n_per_class = request.config.getoption("--num_samples_per_class")
    # since we're only testing binary case here
    n_total = n_per_class * 2

    assert df_labels.shape == (n_total, 1)
    assert set(df_labels[cl_args.target_column].unique()) == {"Asia", "Europe"}


@pytest.mark.parametrize(
    "create_test_data",
    [
        {"class_type": "binary", "data_type": "uint8"},
        {"class_type": "multi", "data_type": "uint8"},
    ],
    indirect=True,
)
def test_split_df(create_test_data, create_test_cl_args):
    cl_args = create_test_cl_args

    df_labels = label_setup.label_df_parse_wrapper(cl_args)

    for valid_fraction in (0.1, 0.5, 0.7):

        df_train, df_valid = label_setup.split_df(df_labels, valid_fraction)
        expected_train = df_labels.shape[0] * (1 - valid_fraction)
        expected_valid = df_labels.shape[0] * valid_fraction

        assert df_train.shape[0] == int(expected_train)
        assert df_valid.shape[0] == int(expected_valid)


@pytest.mark.parametrize(
    "create_test_data",
    [
        {"class_type": "binary", "data_type": "uint8"},
        {"class_type": "multi", "data_type": "uint8"},
    ],
    indirect=True,
)
def test_scale_regression_labels(create_test_data, create_test_cl_args):
    path, _ = create_test_data
    cl_args = create_test_cl_args

    df_labels = label_setup.label_df_parse_wrapper(cl_args)

    for column_value, new_value in zip(["Africa", "Asia", "Europe"], [150, 170, 190]):
        mask = df_labels[cl_args.target_column] == column_value
        df_labels[mask] = new_value

    df_train, df_valid = label_setup.split_df(df_labels, 0.1)

    df_train, scaler_path = label_setup.scale_non_target_continuous_columns(
        df_train, cl_args.target_column, path
    )
    df_valid, _ = label_setup.scale_non_target_continuous_columns(
        df_valid, cl_args.target_column, path, scaler_path
    )

    assert df_train[cl_args.target_column].between(-2, 2).all()
    assert df_valid[cl_args.target_column].between(-2, 2).all()


@pytest.mark.parametrize(
    "create_test_data",
    [
        {"class_type": "binary", "data_type": "uint8"},
        {"class_type": "multi", "data_type": "uint8"},
    ],
    indirect=True,
)
def test_set_up_train_and_valid_labels(request, create_test_data, create_test_cl_args):
    path, test_data_params = create_test_data
    cl_args = create_test_cl_args
    n_classes = 2 if test_data_params.get("class_type") == "binary" else 3

    n_per_class = request.config.getoption("--num_samples_per_class")

    train_labels_dict, valid_labels_dict = label_setup.set_up_train_and_valid_labels(
        cl_args
    )

    assert len(train_labels_dict) + len(valid_labels_dict) == n_classes * n_per_class
    assert len(train_labels_dict) > len(valid_labels_dict)

    train_ids_set = set(train_labels_dict.keys())
    valid_ids_set = set(valid_labels_dict.keys())

    assert len(train_ids_set) == len(train_labels_dict)
    assert len(valid_ids_set) == len(valid_labels_dict)

    assert valid_ids_set.isdisjoint(train_ids_set)
