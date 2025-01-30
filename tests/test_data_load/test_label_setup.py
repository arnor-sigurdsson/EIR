from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.preprocessing import LabelEncoder, StandardScaler

from eir import train
from eir.data_load import label_setup
from eir.data_load.label_setup import merge_target_columns
from eir.setup.config import Configs
from eir.target_setup.target_label_setup import (
    gather_all_ids_from_output_configs,
    get_tabular_target_file_infos,
    set_up_all_target_labels_wrapper,
)


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {
                    "basic_experiment": {
                        "output_folder": "tabular_only",
                    },
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "label_parsing_chunk_size": 50,
                            "input_cat_columns": [],
                            "input_con_columns": ["ExtraTarget"],
                        },
                        "model_config": {"model_type": "tabular"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                            "label_parsing_chunk_size": 50,
                        },
                    },
                ],
            },
        },
        {
            "injections": {
                "global_configs": {
                    "basic_experiment": {
                        "output_folder": "tabular_only",
                    }
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "label_parsing_chunk_size": None,
                            "input_cat_columns": [],
                            "input_con_columns": ["ExtraTarget"],
                        },
                        "model_config": {"model_type": "tabular"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                            "label_parsing_chunk_size": None,
                        },
                    },
                ],
            },
        },
    ],
    indirect=True,
)
def test_set_up_train_and_valid_tabular_data(
    parse_test_cl_args,
    create_test_data,
    create_test_config,
):
    test_configs = create_test_config
    gc = test_configs.global_config

    dc = create_test_data
    n_classes = len(dc.target_classes)

    all_array_ids = gather_all_ids_from_output_configs(
        output_configs=test_configs.output_configs
    )
    train_ids, valid_ids = train.split_ids(
        ids=all_array_ids,
        valid_size=gc.be.valid_size,
    )

    target_labels = set_up_all_target_labels_wrapper(
        output_configs=test_configs.output_configs,
        train_ids=train_ids,
        valid_ids=valid_ids,
    )

    train_labels_df = target_labels.train_labels
    valid_labels_df = target_labels.valid_labels

    assert len(train_labels_df) + len(valid_labels_df) == n_classes * dc.n_per_class
    assert len(train_labels_df) > len(valid_labels_df)

    train_ids_set = set(train_labels_df["ID"])
    valid_ids_set = set(valid_labels_df["ID"])

    assert len(train_ids_set) == len(train_labels_df)
    assert len(valid_ids_set) == len(valid_labels_df)

    assert valid_ids_set.isdisjoint(train_ids_set)


def test_set_up_all_target_transformers(get_transformer_test_data):
    df_test_labels, test_target_columns_dict = get_transformer_test_data

    all_target_transformers = label_setup._get_fit_label_transformers(
        df_labels_train=df_test_labels,
        df_labels_full=df_test_labels,
        label_columns=test_target_columns_dict,
        impute_missing=False,
    )

    height_transformer = all_target_transformers["Height"]
    assert isinstance(height_transformer, StandardScaler)

    origin_transformer = all_target_transformers["Origin"]
    assert isinstance(origin_transformer, LabelEncoder)


def test_fit_scaler_transformer_on_target_column(get_transformer_test_data):
    df_test_labels, test_target_columns_dict = get_transformer_test_data

    transformer = label_setup._get_transformer(column_type="con")

    height_transformer = label_setup._fit_transformer_on_label_column(
        column_series=df_test_labels["Height"],
        transformer=transformer,
        impute_missing=False,
    )

    assert height_transformer.n_samples_seen_ == 3
    assert height_transformer.mean_ == 170
    assert height_transformer.transform([[170]]) == 0


def test_fit_label_encoder_transformer_on_target_column(get_transformer_test_data):
    df_test_labels, test_target_columns_dict = get_transformer_test_data

    transformer = label_setup._get_transformer(column_type="cat")

    origin_transformer = label_setup._fit_transformer_on_label_column(
        column_series=df_test_labels["Origin"],
        transformer=transformer,
        impute_missing=False,
    )

    assert origin_transformer.transform(["Africa"]).item() == 0
    assert origin_transformer.transform(["Europe"]).item() == 2


def test_streamline_values_for_transformer():
    test_values = np.array([1, 2, 3, 4, 5])

    scaler_transformer = StandardScaler()
    streamlined_values_scaler = label_setup.streamline_values_for_transformers(
        transformer=scaler_transformer, values=test_values
    )
    assert streamlined_values_scaler.shape == (5, 1)

    encoder_transformer = LabelEncoder()
    streamlined_values_encoder = label_setup.streamline_values_for_transformers(
        transformer=encoder_transformer, values=test_values
    )
    assert streamlined_values_encoder.shape == (5,)


@pytest.mark.parametrize(
    "test_input_key,expected",
    [
        ("1", {"Origin_as_int": 1, "Scaled_height_int": -1}),  # asia
        ("2", {"Origin_as_int": 0, "Scaled_height_int": 1}),  # africa
        ("3", {"Origin_as_int": 2, "Scaled_height_int": 0}),  # europe
    ],
)
def test_transform_all_labels_in_sample_targets_only(
    test_input_key, expected, get_transformer_test_data
):
    df_test_labels, test_target_columns_dict = get_transformer_test_data

    target_transformers = label_setup._get_fit_label_transformers(
        df_labels_train=df_test_labels,
        df_labels_full=df_test_labels,
        label_columns=test_target_columns_dict,
        impute_missing=False,
    )

    transformed_df = label_setup.transform_label_df(
        df_labels=df_test_labels,
        label_transformers=target_transformers,
        impute_missing=False,
    )

    transformed_sample_labels = transformed_df.filter(pl.col("ID") == test_input_key)

    assert transformed_sample_labels["Origin"].item() == expected["Origin_as_int"]
    assert (
        int(transformed_sample_labels["Height"].item()) == expected["Scaled_height_int"]
    )


@pytest.mark.parametrize(
    "test_input_key,expected",
    [
        ("1", {"Extra_con_int": -1}),  # asia
        ("2", {"Extra_con_int": 1}),  # africa
        ("3", {"Extra_con_int": 0}),  # europe
    ],
)
def test_transform_all_labels_in_sample_with_extra_con(
    test_input_key, expected, get_transformer_test_data
):
    df_test_labels, test_target_columns_dict = get_transformer_test_data

    df_test_labels = df_test_labels.with_columns(
        pl.when(pl.col("ID") == "1")
        .then(130)
        .when(pl.col("ID") == "2")
        .then(170)
        .when(pl.col("ID") == "3")
        .then(150)
        .otherwise(None)
        .alias("Extra_Con")
    )

    test_target_columns_dict["con"].append("Extra_Con")
    label_transformers = label_setup._get_fit_label_transformers(
        df_labels_train=df_test_labels,
        df_labels_full=df_test_labels,
        label_columns=test_target_columns_dict,
        impute_missing=False,
    )

    df_test_labels_transformed = label_setup.transform_label_df(
        df_labels=df_test_labels,
        label_transformers=label_transformers,
        impute_missing=False,
    )

    dtlt = df_test_labels_transformed
    transformed_sample_labels = dtlt.filter(pl.col("ID") == test_input_key).row(
        index=0, named=True
    )

    assert int(transformed_sample_labels["Extra_Con"]) == expected["Extra_con_int"]


@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {
                    "basic_experiment": {
                        "output_folder": "tabular_only",
                    }
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "label_parsing_chunk_size": 50,
                            "input_cat_columns": [],
                            "input_con_columns": ["ExtraTarget"],
                        },
                        "model_config": {"model_type": "tabular"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                            "label_parsing_chunk_size": 50,
                        },
                    },
                ],
            },
        },
        {
            "injections": {
                "global_configs": {
                    "basic_experiment": {
                        "output_folder": "tabular_only",
                    }
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "label_parsing_chunk_size": None,
                            "input_cat_columns": [],
                            "input_con_columns": ["ExtraTarget"],
                        },
                        "model_config": {"model_type": "tabular"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                            "label_parsing_chunk_size": None,
                        },
                    },
                ],
            },
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize("create_test_data", [{"task_type": "binary"}], indirect=True)
def test_label_df_parse_wrapper(
    parse_test_cl_args, create_test_data, create_test_config: Configs
):
    dc = create_test_data
    test_configs = create_test_config

    assert len(test_configs.output_configs) == 1
    main_target_info = test_configs.output_configs[0].output_type_info

    test_target_column = main_target_info.target_cat_columns[0]  # ["Origin"]

    target_file_infos = get_tabular_target_file_infos(
        output_configs=test_configs.output_configs
    )
    assert len(target_file_infos) == 1

    parse_wrapper = label_setup.get_label_parsing_wrapper(
        label_parsing_chunk_size=main_target_info.label_parsing_chunk_size
    )
    df_labels = parse_wrapper(
        label_file_tabular_info=target_file_infos["test_output_tabular"],
        ids_to_keep=None,
    )

    # since we're only testing binary case here
    n_total = dc.n_per_class * 2

    assert df_labels.shape == (n_total, 2)
    assert set(df_labels[test_target_column].unique()) == {"Asia", "Europe"}


def test_ensure_categorical_columns_are_str(get_test_nan_df):
    df_test = get_test_nan_df
    df_converted = label_setup.ensure_categorical_columns_and_format(df=df_test)

    column_dtypes = {col: df_converted[col].dtype for col in df_converted.columns}

    for column, dtype in column_dtypes.items():
        assert isinstance(dtype, object) or dtype == pl.Float32

        if isinstance(dtype, object) and dtype != pl.Float32:
            categories = df_converted[column].unique()
            for i in categories:
                assert isinstance(i, str) or i is None, f"Column: {column}, Value: {i}"


@pytest.mark.parametrize("create_test_data", [{"task_type": "binary"}], indirect=True)
def test_gather_ids_from_data_source(create_test_data):
    c = create_test_data

    test_path = c.scoped_tmp_path / "omics"
    expected_num_samples = c.n_per_class * len(c.target_classes)

    test_ids = label_setup.gather_ids_from_data_source(data_source=test_path)

    assert len(test_ids) == expected_num_samples

    # check that ids are properly formatted, not paths
    assert not any(".npy" in i for i in test_ids)


@pytest.mark.parametrize("create_test_data", [{"task_type": "binary"}], indirect=True)
def test_get_array_path_iterator_file(create_test_data):
    c = create_test_data

    test_path = c.scoped_tmp_path / "omics"
    test_label_file_path = c.scoped_tmp_path / "test_paths.txt"

    with open(test_label_file_path, "w") as test_label_file:
        for path in test_path.iterdir():
            test_label_file.write(str(path) + "\n")

    expected_num_samples = c.n_per_class * len(c.target_classes)
    text_file_iterator = label_setup.get_file_path_iterator(
        data_source=test_label_file_path
    )

    assert len(list(text_file_iterator)) == expected_num_samples


@pytest.mark.parametrize("create_test_data", [{"task_type": "binary"}], indirect=True)
def test_get_array_path_iterator_folder(create_test_data):
    c = create_test_data

    test_path = c.scoped_tmp_path / "omics"

    expected_num_samples = c.n_per_class * len(c.target_classes)
    folder_iterator = label_setup.get_file_path_iterator(data_source=test_path)

    assert len(list(folder_iterator)) == expected_num_samples


@pytest.mark.parametrize("create_test_data", [{"task_type": "binary"}], indirect=True)
def test_get_array_path_iterator_fail(create_test_data):
    c = create_test_data

    with pytest.raises(FileNotFoundError):
        label_setup.get_file_path_iterator(data_source=Path("does/not/exist"))

    test_label_file_path = c.scoped_tmp_path / "test_paths_fail.txt"

    with open(test_label_file_path, "w") as test_label_file:
        for _i in range(5):
            test_label_file.write("non/existent/path.npy" + "\n")

    with pytest.raises(FileNotFoundError):
        iterator = label_setup.get_file_path_iterator(data_source=test_label_file_path)
        _ = list(iterator)


@pytest.mark.parametrize(
    "test_input_args,expected",
    [
        ({"target_cat_columns": ["Origin"]}, ["Origin"]),
        (
            {"target_cat_columns": ["Origin", "OriginExtraColumnsAll"]},
            ["Origin", "OriginExtraColumnsAll"],
        ),
        (
            {
                "target_cat_columns": ["Origin"],
                "input_con_columns": ["OriginExtraColumnsPartial1"],
            },
            ["Origin", "OriginExtraColumnsPartial1"],
        ),
        (
            {
                "target_con_columns": ["Origin"],
                "target_cat_columns": ["OriginExtraColumnsAll"],
                "input_con_columns": ["OriginExtraColumnsPartial1"],
                "input_cat_columns": ["OriginExtraColumnsPartial2"],
            },
            [
                "Origin",
                "OriginExtraColumnsAll",
                "OriginExtraColumnsPartial1",
                "OriginExtraColumnsPartial2",
            ],
        ),
    ],
)
def test_get_all_label_columns_and_dtypes(
    test_input_args,
    expected,
):
    cat_columns = []
    con_columns = []
    for key, value in test_input_args.items():
        if "_cat_" in key:
            cat_columns += value
        elif "_con_" in key:
            con_columns += value

    columns, dtypes = label_setup._get_all_label_columns_and_dtypes(
        cat_columns=cat_columns,
        con_columns=con_columns,
    )

    assert set(columns) == set(expected)
    for column in cat_columns:
        assert isinstance(dtypes[column], object)
    for column in con_columns:
        assert dtypes[column] == pl.Float32


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_load_label_df_one_target_no_extra_col(parse_test_cl_args, create_test_data):
    c = create_test_data

    label_fpath = c.scoped_tmp_path / "labels.csv"
    n_classes = len(c.target_classes)

    label_columns = ["Origin"]
    df_label = label_setup._load_label_df(
        label_fpath=label_fpath,
        columns=label_columns,
    )

    assert df_label.height == c.n_per_class * n_classes
    assert df_label.get_column("ID") is not None

    value_counts = (
        df_label.get_column("Origin")
        .value_counts()
        .sort(by="count", descending=True)
        .get_column("count")
        .to_list()
    )
    assert value_counts == [c.n_per_class] * n_classes


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_load_label_df_one_target_one_extra_col(parse_test_cl_args, create_test_data):
    c = create_test_data

    label_fpath = c.scoped_tmp_path / "labels.csv"

    label_columns = ["Origin", "OriginExtraCol"]

    df_label_extra = label_setup._load_label_df(
        label_fpath=label_fpath,
        columns=label_columns,
    )

    assert df_label_extra.width == 3

    # OriginExtraCol is same as Origin by definition
    assert (df_label_extra["OriginExtraCol"] == df_label_extra["Origin"]).all()


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_load_label_df_missing_col_fail(parse_test_cl_args, create_test_data):
    c = create_test_data

    label_fpath = c.scoped_tmp_path / "labels.csv"

    label_columns = ["Origin", "NonExistentColumn"]

    with pytest.raises(ValueError):
        label_setup._load_label_df(
            label_fpath=label_fpath,
            columns=label_columns,
        )


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_load_label_extra_target_extra_col(parse_test_cl_args, create_test_data):
    c = create_test_data

    label_fpath = c.scoped_tmp_path / "labels.csv"

    label_columns = ["Origin", "OriginExtraCol", "Height", "ExtraTarget"]
    df_label_multi_target = label_setup._load_label_df(
        label_fpath=label_fpath,
        columns=label_columns,
    )

    assert df_label_multi_target.width == 5

    part_1 = df_label_multi_target.get_column("Origin")
    part_2 = df_label_multi_target.get_column("OriginExtraCol")
    assert (part_1 == part_2).all()

    part_3 = df_label_multi_target.get_column("Height")
    part_4 = df_label_multi_target.get_column("ExtraTarget")
    assert (part_3 - 50).cast(pl.Int64).equals(part_4.cast(pl.Int64))


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_get_currently_available_columns_pass(
    parse_test_cl_args,
    create_test_data,
):
    c = create_test_data

    label_fpath = c.scoped_tmp_path / "labels.csv"
    label_columns = ["Origin"]

    available_columns = label_setup._get_currently_available_columns(
        label_fpath=label_fpath,
        requested_columns=label_columns,
    )

    assert available_columns == ["Origin"]


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_get_currently_available_columns_fail(parse_test_cl_args, create_test_data):
    c = create_test_data

    label_fpath = c.scoped_tmp_path / "labels.csv"
    label_columns = ["Origin", "NotExisting1", "NotExisting2"]

    with pytest.raises(ValueError):
        label_setup._get_currently_available_columns(
            label_fpath=label_fpath,
            requested_columns=label_columns,
        )


@pytest.mark.parametrize("create_test_data", [{"task_type": "binary"}], indirect=True)
def test_filter_ids_from_label_df(create_test_data):
    c = create_test_data
    label_fpath = c.scoped_tmp_path / "labels.csv"

    label_columns = ["Origin"]
    df_labels = label_setup._load_label_df(
        label_fpath=label_fpath,
        columns=label_columns,
    )

    df_no_filter = label_setup._filter_ids_from_label_df(df_labels=df_labels)
    assert df_no_filter.equals(df_labels)

    ids_to_keep = ("0_Asia", "1_Asia", "998_Europe")
    df_filtered = label_setup._filter_ids_from_label_df(
        df_labels=df_labels, ids_to_keep=ids_to_keep
    )

    assert df_filtered.height == 3

    assert tuple(df_filtered.get_column("ID").to_list()) == ids_to_keep

    assert tuple(df_filtered.get_column("Origin").to_list()) == (
        "Asia",
        "Asia",
        "Europe",
    )


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_check_parsed_label_df_pass(parse_test_cl_args, create_test_data):
    c = create_test_data
    label_fpath = c.scoped_tmp_path / "labels.csv"

    label_columns = ["Origin", "ExtraTarget"]

    df_labels = label_setup._load_label_df(
        label_fpath=label_fpath,
        columns=label_columns,
        dtypes={
            "Origin": pl.Categorical,
            "ExtraTarget": pl.Float32,
        },
    )

    df_labels_checked = label_setup._check_parsed_label_df(
        df_labels=df_labels,
        supplied_label_columns=label_columns,
    )
    assert df_labels is df_labels_checked


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_check_parsed_label_df_fail(
    parse_test_cl_args,
    create_test_data,
):
    c = create_test_data
    label_fpath = c.scoped_tmp_path / "labels.csv"

    label_columns = ["Origin"]
    fail_label_columns = ["Origin", "ExtraTarget", "NotExisting"]

    df_labels = label_setup._load_label_df(
        label_fpath=label_fpath,
        columns=label_columns,
    )

    with pytest.raises(ValueError):
        label_setup._check_parsed_label_df(
            df_labels=df_labels,
            supplied_label_columns=fail_label_columns,
        )


@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_split_df_by_ids(create_test_data, create_test_config):
    test_configs = create_test_config

    target_file_infos = get_tabular_target_file_infos(
        output_configs=test_configs.output_configs
    )
    assert len(target_file_infos) == 1

    main_target_file_info = target_file_infos["test_output_tabular"]

    df_labels = label_setup.label_df_parse_wrapper(
        label_file_tabular_info=main_target_file_info,
        ids_to_keep=None,
    )

    ids = tuple(df_labels.get_column("ID").to_list())

    for valid_fraction in (0.1, 0.5, 0.7):
        ids_train, ids_valid = label_setup.split_ids(ids=ids, valid_size=valid_fraction)
        df_train, df_valid = label_setup._split_df_by_ids(
            df=df_labels,
            train_ids=list(ids_train),
            valid_ids=list(ids_valid),
        )

        assert set(df_train.get_column("ID").to_list()) == set(ids_train)
        assert set(df_valid.get_column("ID").to_list()) == set(ids_valid)

        expected_no_train = df_labels.height * (1 - valid_fraction)
        expected_no_valid = df_labels.height * valid_fraction

        assert len(ids_train) == int(expected_no_train) == df_train.height
        assert len(ids_valid) == int(expected_no_valid) == df_valid.height


def test_validate_df_pass():
    df = pl.DataFrame({"ID": ["a", "b", "c"], "A": [1, 2, 3], "B": [4, 5, 6]})

    label_setup._validate_df(df=df)


def test_validate_df_fail():
    df = pl.DataFrame(
        {"ID": ["a", "b", "b", "c"], "A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}
    )

    with pytest.raises(ValueError) as excinfo:
        label_setup._validate_df(df)

    assert "b" in str(excinfo.value)

    df = pl.DataFrame(
        {
            "ID": ["a", "b", "b", "c", "c", "c"],
            "A": [1, 2, 3, 4, 5, 6],
            "B": [7, 8, 9, 10, 11, 12],
        }
    )

    with pytest.raises(ValueError) as excinfo:
        label_setup._validate_df(df)

    assert "b" in str(excinfo.value)
    assert "c" in str(excinfo.value)


@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_split_ids(create_test_data, create_test_config):
    test_configs = create_test_config

    target_file_infos = get_tabular_target_file_infos(
        output_configs=test_configs.output_configs
    )
    assert len(target_file_infos) == 1

    main_target_file_info = target_file_infos["test_output_tabular"]

    df_labels = label_setup.label_df_parse_wrapper(
        label_file_tabular_info=main_target_file_info,
        ids_to_keep=None,
    )

    ids = tuple(df_labels.get_column("ID").to_list())

    for valid_fraction in (0.1, 0.5, 0.7):
        ids_train, ids_valid = label_setup.split_ids(ids=ids, valid_size=valid_fraction)
        expected_train = df_labels.height * (1 - valid_fraction)
        expected_valid = df_labels.height * valid_fraction

        assert len(ids_train) == int(expected_train)
        assert len(ids_valid) == int(expected_valid)

    manual_ids = ids[:10]

    ids_train, ids_valid = label_setup.split_ids(
        ids=ids, valid_size=valid_fraction, manual_valid_ids=manual_ids
    )
    assert set(ids_valid) == set(manual_ids)


@pytest.fixture
def get_test_nan_df():
    """
    >>> df
         A    B   C  D
    0  NaN  2   NaN  0.0
    1  3    4   NaN  1.0
    2  NaN  NaN NaN  5.0
    3  NaN  3   NaN  4.0

            [
            [np.nan, 2, np.nan, 0],
            [3, 4, np.nan, 1],
            [np.nan, np.nan, np.nan, 5],
            [np.nan, 3, np.nan, 4],
        ],
        columns=list("ABCD"),
    """
    return pl.DataFrame(
        {
            "ID": ["0", "1", "2", "3"],
            "A": [None, "3", None, None],
            "B": ["2", "4", None, "3"],
            "C": [None, None, None, None],
            "D": [0.0, 1.0, 5.0, 4.0],
        }
    ).with_columns([pl.col("C").cast(pl.Float32), pl.col("D").cast(pl.Float32)])


@pytest.fixture
def get_test_nan_tabular_file_info():
    label_info = label_setup.TabularFileInfo(
        file_path=Path("fake_file"),
        cat_columns=["A", "B"],
        con_columns=["C", "D"],
    )

    return label_info


def test_process_train_and_label_dfs(get_test_nan_df, get_test_nan_tabular_file_info):
    """
    NOTE:   Here we have the situation where we have NA in valid, but not in train. This
            means that we have to make sure we have manually added the 'NA' to train.
    """
    test_df = get_test_nan_df
    label_info = get_test_nan_tabular_file_info

    train_df = test_df.clone()
    valid_df = test_df.clone()

    train_df = train_df.with_columns(
        [
            pl.col("A").fill_null(5).cast(pl.String),
            pl.col("B").fill_null(5).cast(pl.String),
        ]
    )

    train_df = label_setup.ensure_categorical_columns_and_format(df=train_df)
    valid_df = label_setup.ensure_categorical_columns_and_format(df=valid_df)

    func = label_setup._process_train_and_label_dfs
    train_df_filled, valid_df_filled, label_transformers = func(
        tabular_info=label_info,
        df_labels_train=train_df,
        df_labels_valid=valid_df,
        impute_missing=True,
    )

    assert set(label_transformers["A"].classes_) == {"5", "3", "__NULL__"}
    assert set(label_transformers["B"].classes_) == {"2", "3", "4", "5", "__NULL__"}

    a_t = label_transformers["A"]
    assert set(train_df_filled.get_column("A").unique().to_list()) == {0, 1}
    assert set(
        a_t.inverse_transform(train_df_filled.get_column("A").unique().to_list())
    ) == {"5", "3"}

    assert set(valid_df_filled.get_column("A").unique().to_list()) == {0, 2}
    assert set(
        a_t.inverse_transform(valid_df_filled.get_column("A").unique().to_list())
    ) == {"__NULL__", "3"}

    b_t = label_transformers["B"]
    assert set(train_df_filled.get_column("B").unique().to_list()) == {0, 1, 2, 3}
    assert set(
        b_t.inverse_transform(train_df_filled.get_column("B").unique().to_list())
    ) == {"2", "3", "4", "5"}

    assert set(valid_df_filled.get_column("B").unique().to_list()) == {0, 1, 2, 4}
    assert set(
        b_t.inverse_transform(valid_df_filled.get_column("B").unique().to_list())
    ) == {"2", "3", "4", "__NULL__"}

    assert (train_df_filled.get_column("C") == 0.0).all()
    assert (valid_df_filled.get_column("C") == 0.0).all()

    assert train_df_filled.get_column("D").equals(valid_df_filled.get_column("D"))


def test_handle_missing_label_values_in_df(
    get_test_nan_df, get_test_nan_tabular_file_info
):
    test_df = get_test_nan_df
    label_info = get_test_nan_tabular_file_info

    test_df_filled = label_setup.handle_missing_label_values_in_df(
        df=test_df,
        cat_label_columns=label_info.cat_columns,
        con_label_columns=label_info.con_columns,
        con_manual_values={"C": 3.0},
        impute_missing=True,
    )

    assert set(test_df_filled.get_column("A").unique().to_list()) == {"__NULL__", "3"}
    assert set(test_df_filled.get_column("B").unique().to_list()) == {
        "__NULL__",
        "2",
        "3",
        "4",
    }

    assert (test_df_filled.get_column("C") == 3.0).all()
    assert test_df_filled.get_column("D").equals(test_df.get_column("D"))


@pytest.mark.parametrize("impute_missing", [False, True])
def test_fill_categorical_nans(
    impute_missing: bool, get_test_nan_df: pl.DataFrame
) -> None:
    """
    Only when we impute missing do we call fill_null('nan'), this is the case for
    inputs where we e.g. actually want to encode them in the tabular input. For
    targets, we keep them as None as they are never actually used.
    """
    test_df = get_test_nan_df
    test_df_filled = label_setup._fill_categorical_nans(
        df=test_df,
        column_names=["A", "B"],
        impute_missing=impute_missing,
    )

    if impute_missing:
        assert set(test_df_filled.get_column("A").unique().to_list()) == {
            "__NULL__",
            "3",
        }
        assert set(test_df_filled.get_column("B").unique().to_list()) == {
            "__NULL__",
            "2",
            "3",
            "4",
        }
    else:
        assert {
            x
            for x in test_df_filled.get_column("A").unique().to_list()
            if x is not None
        } == {"3"}
        assert {
            x
            for x in test_df_filled.get_column("B").unique().to_list()
            if x is not None
        } == {"2", "3", "4"}

    assert test_df_filled.get_column("C").is_null().all()
    assert test_df_filled.get_column("D").is_null().sum() == 0


def test_get_con_manual_vals_dict(get_test_nan_df):
    test_df = get_test_nan_df.with_columns(
        [pl.col(["A", "B", "C", "D"]).cast(pl.Float32)]
    )

    means_dict = label_setup._get_con_manual_vals_dict(
        df=test_df,
        con_columns=[
            "A",
            "B",
            "C",
            "D",
        ],
    )
    assert means_dict["A"] == 3.0
    assert means_dict["B"] == 3.0
    assert means_dict["C"] == 0.0  # all are nan
    assert means_dict["D"] == 2.5


def test_fill_continuous_nans(get_test_nan_df):
    test_df = get_test_nan_df.with_columns(
        [pl.col(["A", "B", "C", "D"]).cast(pl.Float32)]
    )

    manual_values = {"A": 1.0, "B": 2.0, "C": 3.0}
    test_df_filled = label_setup._fill_continuous_nans(
        df=test_df,
        column_names=["A", "B", "C"],
        con_means_dict=manual_values,
        impute_missing=True,
    )

    assert test_df_filled.filter(pl.col("ID") == "0").get_column("A")[0] == 1.0
    assert test_df_filled.filter(pl.col("ID") == "1").get_column("A")[0] == 3.0
    assert test_df_filled.filter(pl.col("ID") == "2").get_column("B")[0] == 2.0
    assert (test_df_filled.get_column("C") == 3.0).all()


@pytest.mark.parametrize(
    "test_input,expected",
    [  # test case 1
        (
            (["con_1", "con_2"], ["cat_1", "cat_2"]),
            {"con": ["con_1", "con_2"], "cat": ["cat_1", "cat_2"]},
        ),
        # test case 2
        ((["con_1", "con_2"], []), {"con": ["con_1", "con_2"], "cat": []}),
    ],
)
def test_merge_target_columns_pass(test_input, expected):
    test_output = merge_target_columns(*test_input)
    assert test_output == expected


def test_merge_target_columns_fail():
    with pytest.raises(ValueError):
        merge_target_columns([], [])
