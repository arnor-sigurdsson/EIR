import pandas as pd
import pytest

from eir.setup.input_setup_modules import setup_omics


@pytest.mark.parametrize(
    "indices_to_subset,expected",
    [
        ([0, 10, 20, 30, 40, 50, 60, 70], [0, 10, 20, 30, 40, 50, 60, 70]),
        ([0, 10, 20, 30, 40, 50, 60, 70, 120, 999], [0, 10, 20, 30, 40, 50, 60, 70]),
    ],
)
def test_setup_subset_indices(indices_to_subset: list[int], expected: list[int]):
    test_data = []
    for i in range(100):
        cur_row_data = ["1", str(i), "0.1", str(i), "A", "T"]
        test_data.append(cur_row_data)

    df_bim_test = pd.DataFrame(test_data, columns=setup_omics._get_bim_headers())

    snps_to_subset = [str(i) for i in indices_to_subset]

    subset_indices = setup_omics._setup_snp_subset_indices(
        df_bim=df_bim_test, snps_to_subset=snps_to_subset
    )
    assert subset_indices.tolist() == expected


def test_read_snp_df(tmp_path):
    snp_file_str = """
            1     rs3094315        0.020130          752566 G A
            1    rs7419119         0.022518          842013 G T
            1   rs13302957         0.024116          891021 G A
            1    rs6696609         0.024457          903426 T C
            1       rs8997         0.025727          949654 A G
            1    rs9442372         0.026288         1018704 A G
            1    rs4970405         0.026674         1048955 G A
            1   rs11807848         0.026711         1061166 C T
            1    rs4970421         0.028311         1108637 A G
            1    rs1320571         0.028916         1120431 A G
               """
    file_ = tmp_path / "data_final.bim"
    file_.write_text(snp_file_str)

    df_bim = setup_omics.read_bim(bim_file_path=str(file_))
    snp_arr = df_bim["VAR_ID"].array
    assert len(snp_arr) == 10
    assert snp_arr[0] == "rs3094315"
    assert snp_arr[-1] == "rs1320571"
