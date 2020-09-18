import pandas as pd
import torch

from snp_pred.data_load import common_ops


def test_filter_ids_from_array_and_id_lists():
    test_ids_to_filter = ["a", "b", "c"]
    test_arrays = torch.tensor([torch.tensor(i) for i in [1, 2, 3, 4, 5]])
    test_ids = ["a", "b", "c", "d", "e"]

    filter_func = common_ops.filter_ids_from_array_and_id_lists
    filtered_ids_skip, filtered_arrays_skip = filter_func(
        test_ids_to_filter, test_ids, test_arrays, "skip"
    )

    assert filtered_ids_skip == ["d", "e"]
    assert (filtered_arrays_skip == torch.tensor([4, 5])).all()

    filtered_ids_keep, filtered_arrays_keep = filter_func(
        test_ids_to_filter, test_ids, test_arrays, "keep"
    )

    assert filtered_ids_keep == ["a", "b", "c"]
    assert (filtered_arrays_keep == torch.tensor([1, 2, 3])).all()


def test_bucket_column():
    testcol = "TestCol"
    test_df = pd.DataFrame(columns=[testcol], data=range(0, 11))

    test_df_out = common_ops.bucket_column(test_df, testcol, n_buckets=5)

    assert str(test_df_out[testcol].loc[0]) == "(-0.011, 2.0]"
    assert str(test_df_out[testcol].loc[3]) == "(2.0, 4.0]"
    assert str(test_df_out[testcol].loc[10]) == "(8.0, 10.0]"


def test_get_low_count_ids():
    testcol = "TestCol"
    col_values = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3]
    test_df = pd.DataFrame(columns=[testcol], data=col_values)

    low_count_ids = common_ops.get_low_count_ids(test_df, testcol, 2)
    assert low_count_ids == [3, 4, 5, 6]
