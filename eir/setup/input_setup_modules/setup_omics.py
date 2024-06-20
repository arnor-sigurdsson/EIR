from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from eir.setup import schemas
from eir.setup.input_setup_modules.common import (
    DataDimensions,
    get_data_dimension_from_data_source,
)
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


@dataclass
class ComputedOmicsInputInfo:
    input_config: schemas.InputConfig
    data_dimensions: "DataDimensions"
    subset_indices: Optional[np.ndarray]


def set_up_omics_input(
    input_config: schemas.InputConfig, *args, **kwargs
) -> ComputedOmicsInputInfo:
    data_dimensions = get_data_dimension_from_data_source(
        data_source=Path(input_config.input_info.input_source),
        deeplake_inner_key=input_config.input_info.input_inner_key,
    )

    subset_indices = None
    input_type_info = input_config.input_type_info
    assert isinstance(input_type_info, schemas.OmicsInputDataConfig)

    if input_type_info.subset_snps_file:
        assert input_type_info.snp_file is not None
        df_bim = read_bim(bim_file_path=input_type_info.snp_file)
        snps_to_subset = read_subset_file(
            subset_snp_file_path=input_type_info.subset_snps_file
        )
        subset_indices = _setup_snp_subset_indices(
            df_bim=df_bim,
            snps_to_subset=snps_to_subset,
            snp_file_name=input_type_info.snp_file,
            subset_file_name=input_type_info.subset_snps_file,
        )
        data_dimensions = DataDimensions(
            channels=data_dimensions.channels,
            height=data_dimensions.height,
            width=len(subset_indices),
        )

    omics_input_info = ComputedOmicsInputInfo(
        input_config=input_config,
        data_dimensions=data_dimensions,
        subset_indices=subset_indices,
    )

    return omics_input_info


def _setup_snp_subset_indices(
    df_bim: pd.DataFrame,
    snps_to_subset: List[str],
    subset_file_name: str = "",
    snp_file_name: str = "",
) -> np.ndarray:
    """
    .bim columns: ["CHR_CODE", "VAR_ID", "POS_CM", "BP_COORD", "ALT", "REF"]
    """

    df_subset = df_bim[df_bim["VAR_ID"].isin(snps_to_subset)]

    if len(df_subset) < len(snps_to_subset):
        num_missing = len(snps_to_subset) - len(df_subset)
        missing = [i for i in snps_to_subset if i not in df_subset["VAR_ID"].values]
        logger.warning(
            "Did not find all SNPs in subset file '%s' in base .bim file '%s'. "
            "Number of missing SNPs: %d. Example: '%s'.",
            subset_file_name,
            snp_file_name,
            num_missing,
            missing[:3],
        )
    else:
        logger.info(
            "Using %d SNPs from subset file %s.", len(df_subset), subset_file_name
        )

    return np.asarray(df_subset.index)


def read_subset_file(subset_snp_file_path: str) -> List[str]:
    with open(subset_snp_file_path, "r") as infile:
        snps_to_subset = infile.read().split()

    return snps_to_subset


def read_bim(bim_file_path: str) -> pd.DataFrame:
    bim_headers = _get_bim_headers()
    df_bim = pd.read_csv(bim_file_path, names=bim_headers, sep=r"\s+")
    df_bim["VAR_ID"] = df_bim["VAR_ID"].astype(str)

    if not len(df_bim.columns) == 6:
        raise ValueError(
            "Expected 6 columns in bim file '%s', got %d.",
            bim_file_path,
            len(df_bim.columns),
        )

    return df_bim


def _get_bim_headers() -> List[str]:
    bim_headers = ["CHR_CODE", "VAR_ID", "POS_CM", "BP_COORD", "ALT", "REF"]
    return bim_headers
