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
        "OriginExtraColumnsAll": [
            ColumnOperation(
                test_column_op_1, replace_dict_args, ("ExtraCol1", "ExtraCol2")
            ),
            ColumnOperation(test_column_op_3, replace_column_dict_arg, ("ExtraCol3",)),
        ],
        "OriginExtraColumnsPartial1": [
            ColumnOperation(
                test_column_op_1, replace_dict_args, ("ExtraCol1", "ExtraCol2")
            )
        ],
        "OriginExtraColumnsPartial2": [
            ColumnOperation(test_column_op_3, replace_column_dict_arg, ("ExtraCol3",))
        ],
    }

    return test_column_ops


@pytest.mark.parametrize(
    "create_test_data",
    [{"class_type": "binary"}, {"class_type": "multi"}],
    indirect=True,
)
def test_set_up_train_and_valid_labels(
    parse_test_cl_args, create_test_data, create_test_cl_args
):
    c = create_test_data
    cl_args = create_test_cl_args
    n_classes = len(c.target_classes)

    train_labels_dict, valid_labels_dict = label_setup.set_up_train_and_valid_labels(
        cl_args
    )

    assert len(train_labels_dict) + len(valid_labels_dict) == n_classes * c.n_per_class
    assert len(train_labels_dict) > len(valid_labels_dict)

    train_ids_set = set(train_labels_dict.keys())
    valid_ids_set = set(valid_labels_dict.keys())

    assert len(train_ids_set) == len(train_labels_dict)
    assert len(valid_ids_set) == len(valid_labels_dict)

    assert valid_ids_set.isdisjoint(train_ids_set)


@pytest.mark.parametrize("create_test_data", [{"class_type": "binary"}], indirect=True)
def test_label_df_parse_wrapper(
    parse_test_cl_args, create_test_data, create_test_cl_args
):
    c = create_test_data
    cl_args = create_test_cl_args
    test_target_column = cl_args.target_cat_columns[0]  # Origin

    df_labels = label_setup.label_df_parse_wrapper(cl_args)

    # since we're only testing binary case here
    n_total = c.n_per_class * 2

    assert df_labels.shape == (n_total, 1)
    assert set(df_labels[test_target_column].unique()) == {"Asia", "Europe"}


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (["Origin"], []),
        (["OriginExtraColumnsAll"], ["ExtraCol1", "ExtraCol2", "ExtraCol3"]),
        (["OriginExtraColumnsPartial1", "Origin"], ["ExtraCol1", "ExtraCol2"]),
        (
            ["OriginExtraColumnsPartial1", "OriginExtraColumnsPartial2"],
            ["ExtraCol1", "ExtraCol2", "ExtraCol3"],
        ),
    ],
)
def test_get_extra_columns(test_input, expected, get_test_column_ops):
    test_column_ops = get_test_column_ops

    test_output = label_setup._get_extra_columns(test_input, test_column_ops)
    assert test_output == expected


@pytest.mark.parametrize(
    "create_test_data",
    [{"class_type": "binary"}, {"class_type": "multi"}],
    indirect=True,
)
def test_load_label_df_one_target_no_extra_col(parse_test_cl_args, create_test_data):
    c = create_test_data

    label_fpath = c.scoped_tmp_path / "labels.csv"
    n_classes = len(c.target_classes)

    target_column_single = ["Origin"]
    df_label = label_setup._load_label_df(label_fpath, target_column_single)

    assert df_label.shape[0] == c.n_per_class * n_classes
    assert df_label.index.name == "ID"
    assert [i for i in df_label.Origin.value_counts()] == [c.n_per_class] * n_classes


@pytest.mark.parametrize(
    "create_test_data",
    [{"class_type": "binary"}, {"class_type": "multi"}],
    indirect=True,
)
def test_load_label_df_one_target_one_extra_col(parse_test_cl_args, create_test_data):
    c = create_test_data

    label_fpath = c.scoped_tmp_path / "labels.csv"

    target_column_single = ["Origin"]

    df_label_extra = label_setup._load_label_df(
        label_fpath, target_column_single, extra_columns=("OriginExtraCol",)
    )

    assert df_label_extra.shape[1] == 2
    # OriginExtraCol is same as Origin by definiton
    assert (df_label_extra["OriginExtraCol"] == df_label_extra["Origin"]).all()


@pytest.mark.parametrize(
    "create_test_data",
    [{"class_type": "binary"}, {"class_type": "multi"}],
    indirect=True,
)
def test_load_label_df_filter_ids(parse_test_cl_args, create_test_data):
    c = create_test_data

    label_fpath = c.scoped_tmp_path / "labels.csv"

    target_column_single = ["Origin"]

    df_label_ids = label_setup._load_label_df(
        label_fpath,
        target_column_single,
        ids_to_keep=("95_Europe", "96_Europe", "97_Europe"),
    )
    assert df_label_ids.shape[0] == 3


@pytest.mark.parametrize(
    "create_test_data",
    [{"class_type": "binary"}, {"class_type": "multi"}],
    indirect=True,
)
def test_load_label_extra_target_extra_col(parse_test_cl_args, create_test_data):
    c = create_test_data

    label_fpath = c.scoped_tmp_path / "labels.csv"

    target_column_multi = ["Origin", "ExtraTarget"]
    df_label_multi_target = label_setup._load_label_df(
        label_fpath, target_column_multi, extra_columns=("OriginExtraCol",)
    )

    assert df_label_multi_target.shape[1] == 3

    # Check that they're all the same, as defined
    part_1 = df_label_multi_target["Origin"]
    part_2 = df_label_multi_target["OriginExtraCol"]
    part_3 = df_label_multi_target["ExtraTarget"]
    assert (part_1 == part_2).all()
    assert (part_2 == part_3).all()


@pytest.mark.parametrize("create_test_data", [{"class_type": "binary"}], indirect=True)
def test_parse_label_df(create_test_data, get_test_column_ops):
    c = create_test_data
    label_fpath = c.scoped_tmp_path / "labels.csv"

    test_column_ops = get_test_column_ops

    df_label = label_setup._load_label_df(label_fpath, ["Origin"])
    df_label_parsed = label_setup._parse_label_df(df_label, test_column_ops, ["Origin"])

    assert set(df_label_parsed.Origin.unique()) == {"Iceland" * 2, "Asia" * 2}

    extra_cols = ("ExtraCol3",)
    for col in extra_cols:
        df_label[col] = "Iceland"

    df_label = df_label.rename(columns={"Origin": "OriginExtraColumnsAll"})
    df_label_parsed = label_setup._parse_label_df(df_label, test_column_ops, ["Origin"])
    assert df_label_parsed["OriginExtraColumnsAll"].unique().item() == "Iceland"


@pytest.mark.parametrize(
    "create_test_data",
    [{"class_type": "binary"}, {"class_type": "multi"}],
    indirect=True,
)
def test_split_df(create_test_data, create_test_cl_args):
    cl_args = create_test_cl_args

    df_labels = label_setup.label_df_parse_wrapper(cl_args)

    for valid_fraction in (0.1, 0.5, 0.7):

        df_train, df_valid = label_setup._split_df(df_labels, valid_fraction)
        expected_train = df_labels.shape[0] * (1 - valid_fraction)
        expected_valid = df_labels.shape[0] * valid_fraction

        assert df_train.shape[0] == int(expected_train)
        assert df_valid.shape[0] == int(expected_valid)


@pytest.mark.parametrize(
    "create_test_data",
    [{"class_type": "binary"}, {"class_type": "multi"}],
    indirect=True,
)
def test_scale_regression_labels(create_test_data, create_test_cl_args):
    c = create_test_data
    cl_args = create_test_cl_args
    test_target_column = cl_args.target_cat_columns[0]  # Origin

    df_labels = label_setup.label_df_parse_wrapper(cl_args=cl_args)

    for column_value, new_value in zip(["Africa", "Asia", "Europe"], [150, 170, 190]):
        mask = df_labels[test_target_column] == column_value
        df_labels[mask] = new_value

    df_train, df_valid = label_setup._split_df(df_labels, 0.1)

    df_train, scaler_path = label_setup.scale_non_target_continuous_columns(
        df=df_train, continuous_column=test_target_column, run_folder=c.scoped_tmp_path
    )
    df_valid, _ = label_setup.scale_non_target_continuous_columns(
        df=df_valid,
        continuous_column=test_target_column,
        run_folder=c.scoped_tmp_path,
        scaler_path=scaler_path,
    )

    assert df_train[test_target_column].between(-2, 2).all()
    assert df_valid[test_target_column].between(-2, 2).all()
