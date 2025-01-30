import numpy as np
import pytest

from eir.interpretation import interpret_omics


def test_rescale_gradients():
    input_array = np.array([[0, 0, 1], [0, 0, 2], [0, 0, 4]])

    rescaled_array = interpret_omics.rescale_gradients(gradients=input_array)

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


def test_get_top_gradients(acc_grads_inp: dict[str, np.ndarray]):
    top_snps_per_class = interpret_omics.get_snp_cols_w_top_grads(acc_grads_inp, 3)
    assert top_snps_per_class["Asia"]["top_n_idxs"].tolist() == [0, 2, 4]

    asia_grads = top_snps_per_class["Asia"]["top_n_grads"]
    asia_expected = np.array([[0.5, 0, 0], [0, 1.0, 0], [0, 0, 2.0], [0, 0, 0]])
    assert (asia_grads == asia_expected).all()

    assert top_snps_per_class["Europe"]["top_n_idxs"].tolist() == [1, 3, 5]

    eur_grads = top_snps_per_class["Europe"]["top_n_grads"]
    eur_expected = np.array([[1.5, 0, 0], [0, 2.0, 0], [0, 0, 3.0], [0, 0, 0]])
    assert (eur_grads == eur_expected).all()


def test_gather_and_rescale_snps(acc_grads_inp):
    top_gradients_dict = interpret_omics.get_snp_cols_w_top_grads(
        accumulated_grads=acc_grads_inp, n=3
    )
    classes = ["Asia", "Europe"]

    top_snps_dict = interpret_omics.gather_and_rescale_snps(
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
