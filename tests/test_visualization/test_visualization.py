from typing import Dict

import numpy as np
import pytest

import eir.interpretation.interpret_omics


def test_rescale_gradients():
    input_array = np.array([[0, 0, 1], [0, 0, 2], [0, 0, 4]])

    rescaled_array = eir.interpretation.interpret_omics.rescale_gradients(
        gradients=input_array
    )

    assert (
        rescaled_array == np.array([[0, 0, 0.25], [0, 0, 0.50], [0, 0, 1.00]])
    ).all()


@pytest.fixture()
def acc_grads_inp():
    asia_arr = np.zeros((4, 10))
    asia_arr[0, 0] = 0.5
    asia_arr[1, 2] = 1.0
    asia_arr[2, 4] = 2.0

    europe_arr = np.zeros((4, 10))
    europe_arr[0, 1] = 1.5
    europe_arr[1, 3] = 2.0
    europe_arr[2, 5] = 3.0

    accumulated_grads_test = {"Asia": asia_arr, "Europe": europe_arr}

    return accumulated_grads_test


def test_get_top_gradients(acc_grads_inp: Dict[str, np.ndarray]):
    top_snps_per_class = eir.interpretation.interpret_omics.get_snp_cols_w_top_grads(
        acc_grads_inp, 3
    )
    assert top_snps_per_class["Asia"]["top_n_idxs"] == [0, 2, 4]
    asia_grads = top_snps_per_class["Asia"]["top_n_grads"]
    assert (
        asia_grads == np.array([[0.5, 0, 0], [0, 1.0, 0], [0, 0, 2.0], [0, 0, 0]])
    ).all()

    assert top_snps_per_class["Europe"]["top_n_idxs"] == [1, 3, 5]
    eur_grads = top_snps_per_class["Europe"]["top_n_grads"]
    assert (
        eur_grads == np.array([[1.5, 0, 0], [0, 2.0, 0], [0, 0, 3.0], [0, 0, 0]])
    ).all()


def test_read_snp_df(tmp_path):
    snp_file_str = """
                rs3094315    1        0.020130          752566 G A
               rs7419119     1        0.022518          842013 G T
              rs13302957     1        0.024116          891021 G A
               rs6696609     1        0.024457          903426 T C
                  rs8997     1        0.025727          949654 A G
               rs9442372     1        0.026288         1018704 A G
               rs4970405     1        0.026674         1048955 G A
              rs11807848     1        0.026711         1061166 C T
               rs4970421     1        0.028311         1108637 A G
               rs1320571     1        0.028916         1120431 A G
               """
    file_ = tmp_path / "data_final.snp"
    file_.write_text(snp_file_str)

    snp_df = eir.interpretation.interpret_omics.read_snp_df(snp_file_path=file_)
    snp_arr = snp_df["VAR_ID"].array
    assert len(snp_arr) == 10
    assert snp_arr[0] == "rs3094315"
    assert snp_arr[-1] == "rs1320571"


def test_gather_and_rescale_snps(acc_grads_inp):
    top_gradients_dict = eir.interpretation.interpret_omics.get_snp_cols_w_top_grads(
        accumulated_grads=acc_grads_inp, n=3
    )
    classes = ["Asia", "Europe"]

    top_snps_dict = eir.interpretation.interpret_omics.gather_and_rescale_snps(
        all_gradients_dict=acc_grads_inp,
        top_gradients_dict=top_gradients_dict,
        classes=classes,
    )

    top_asia_idx_by_asia = top_snps_dict["Asia"]["Asia"]
    expected_asia_idx = np.array([[0.25, 0, 0], [0, 0.5, 0], [0, 0, 1.0], [0, 0, 0]])
    assert (top_asia_idx_by_asia == expected_asia_idx).all()

    top_eur_idx_by_asia = top_snps_dict["Asia"]["Europe"]
    assert (top_eur_idx_by_asia == np.zeros((4, 3))).all()

    top_eur_idx_by_eur = top_snps_dict["Europe"]["Europe"]
    expected_europe_idx = np.array([[0.5, 0, 0], [0, 2 / 3, 0], [0, 0, 1.0], [0, 0, 0]])
    assert (top_eur_idx_by_eur == expected_europe_idx).all()

    top_asia_idx_by_eur = top_snps_dict["Europe"]["Asia"]
    assert (top_asia_idx_by_eur == np.zeros((4, 3))).all()
