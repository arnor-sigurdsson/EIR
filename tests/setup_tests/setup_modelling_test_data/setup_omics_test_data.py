from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tests.setup_tests.setup_modelling_test_data.setup_targets_test_data import (
    get_current_test_label_values,
    get_test_label_file_fieldnames,
    set_up_label_file_writing,
    set_up_label_line_dict,
)
from tests.setup_tests.setup_modelling_test_data.setup_test_data_utils import (
    set_up_test_data_root_outpath,
)

if TYPE_CHECKING:
    from tests.setup_tests.fixtures_create_data import TestDataConfig


def create_test_omics_data_and_labels(
    test_data_config: "TestDataConfig", array_outfolder: Path
) -> Path:
    c = test_data_config

    fieldnames = get_test_label_file_fieldnames()
    label_file_handle, label_file_writer = set_up_label_file_writing(
        base_path=c.scoped_tmp_path,
        fieldnames=fieldnames,
        extra_name="_omics",
    )

    array_outfolder = set_up_test_data_root_outpath(base_folder=array_outfolder)

    for cls, snp_row_idx in c.target_classes.items():
        for sample_idx in range(c.n_per_class):
            sample_outpath = array_outfolder / f"{sample_idx}_{cls}"

            num_active_snps_in_sample = _create_and_save_test_array_omics(
                test_data_config=c,
                active_snp_row_idx=snp_row_idx,
                sample_outpath=sample_outpath,
            )

            label_line_base = set_up_label_line_dict(
                sample_name=sample_outpath.stem, fieldnames=fieldnames
            )

            label_line_dict = get_current_test_label_values(
                values_dict=label_line_base,
                num_active_elements_in_sample=len(num_active_snps_in_sample),
                cur_class=cls,
            )
            label_file_writer.writerow(label_line_dict)

    label_file_handle.close()

    write_test_data_snp_file(base_folder=c.scoped_tmp_path, n_snps=c.n_snps)

    return array_outfolder


def _create_and_save_test_array_omics(
    test_data_config: "TestDataConfig", active_snp_row_idx, sample_outpath: Path
):
    c = test_data_config

    base_array, snp_idxs_candidates = _set_up_base_test_omics_array(n_snps=c.n_snps)

    cur_test_array, snps_this_sample = _create_test_array(
        base_array=base_array,
        snp_idxs_candidates=snp_idxs_candidates,
        snp_row_idx=active_snp_row_idx,
    )

    np.save(str(sample_outpath), cur_test_array)

    return snps_this_sample


def _set_up_base_test_omics_array(n_snps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    IMPORTANT NOTE ON MEMORY LAYOUT in DeepLake V4:
    This function ensures proper memory layout for storage in DeepLake. The PyTorch
    transpose operation creates arrays in Fortran order (column-major) memory layout:
        - Original data: C_CONTIGUOUS=True, F_CONTIGUOUS=False
        - After transpose: C_CONTIGUOUS=False, F_CONTIGUOUS=True

    DeepLake assumes C-order (row-major) when storing/loading arrays. If we store
    a Fortran-ordered array, Deep Lake will read the memory in the wrong order,
    corrupting the data. Consider this example:

    Fortran-ordered memory of a one-hot array:
        Memory: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        Intended shape (4x4):    When Deep Lake reads in C-order:
        1 0 0 0                  1 1 0 0
        0 1 0 0      -->         0 0 0 0
        0 0 1 0                  0 0 0 0
        0 0 0 1                  0 0 1 1

    To prevent this, we use np.ascontiguousarray() to ensure C-ordered memory
    layout before yielding the arrays for storage.
    """
    # create random one hot array
    base_array = np.eye(4, dtype=bool)[np.random.choice(4, n_snps)].T

    # ensure C order
    base_array = np.ascontiguousarray(base_array)

    # set up 10 candidates
    step_size = n_snps // 10
    snp_idxs_candidates = np.array(range(50, n_snps, step_size))

    return base_array, snp_idxs_candidates


def _create_test_array(
    base_array: np.ndarray,
    snp_idxs_candidates: np.ndarray,
    snp_row_idx: int,
) -> tuple[np.ndarray, list[int]]:
    # make samples have missing for chosen, otherwise might have alleles chosen
    # below by random, without having the phenotype
    base_array[:, snp_idxs_candidates] = 0
    base_array[3, snp_idxs_candidates] = 1

    lower_bound, upper_bound = 4, 11  # between 4 and 10 snps

    np.random.shuffle(snp_idxs_candidates)
    num_snps_this_sample = np.random.randint(lower_bound, upper_bound)
    snp_idxs = sorted(snp_idxs_candidates[:num_snps_this_sample])

    base_array[:, snp_idxs] = 0
    base_array[snp_row_idx, snp_idxs] = 1

    base_array = base_array.astype(np.uint8)
    return base_array, snp_idxs


def write_test_data_snp_file(base_folder: Path, n_snps: int) -> None:
    """
    BIM specs:
        0. Chromosome code
        1. Variant ID
        2. Position in centi-morgans
        3. Base-pair coordinate (1-based)
        4. ALT allele cod
        5. REF allele code
    """
    snp_file = base_folder / "test_snps.bim"
    base_snp_string_list = ["1", "REPLACE_W_IDX", "0.1", "REPLACE_W_IDX", "A", "T"]

    with open(str(snp_file), "w") as snpfile:
        for snp_idx in range(n_snps):
            cur_snp_list = base_snp_string_list[:]
            cur_snp_list[1] = str(snp_idx)
            cur_snp_list[3] = str(snp_idx)

            cur_snp_string = "\t".join(cur_snp_list)
            snpfile.write(cur_snp_string + "\n")

    subset_file = base_folder / "test_subset_snps.txt"

    n_subset_snps = n_snps // 5
    with open(str(subset_file), "w") as subset_snp_file:
        for snp_idx in range(n_subset_snps):
            subset_snp_file.write(str(snp_idx) + "\n")
