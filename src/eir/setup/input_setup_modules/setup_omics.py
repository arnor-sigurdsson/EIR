from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

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
    subset_indices: np.ndarray | None
    expert_snp_indices: dict[str, np.ndarray] | None = None


def set_up_omics_input(
    input_config: schemas.InputConfig,
    data_dimensions: Optional["DataDimensions"] = None,
    *args,
    **kwargs,
) -> ComputedOmicsInputInfo:
    if data_dimensions is None:
        data_dimensions = get_data_dimension_from_data_source(
            data_source=Path(input_config.input_info.input_source),
        )

    subset_indices = None
    expert_snp_indices = None
    input_type_info = input_config.input_type_info
    assert isinstance(input_type_info, schemas.OmicsInputDataConfig)

    if input_type_info.expert_snp_groups_file:
        assert input_type_info.snp_file is not None, (
            "snp_file must be set when using expert_snp_groups_file "
            "(needed for BIM lookup)."
        )
        df_bim = read_bim(bim_file_path=input_type_info.snp_file)

        expert_groups = _read_expert_snp_groups(
            path=input_type_info.expert_snp_groups_file,
        )

        union_snps = _compute_expert_union(expert_groups=expert_groups)

        subset_indices = _setup_snp_subset_indices(
            df_bim=df_bim,
            snps_to_subset=union_snps,
            snp_file_name=input_type_info.snp_file,
            subset_file_name=input_type_info.expert_snp_groups_file,
        )

        expert_snp_indices = _setup_expert_snp_indices(
            df_bim=df_bim,
            expert_groups=expert_groups,
            subset_indices=subset_indices,
        )

        data_dimensions = DataDimensions(
            channels=data_dimensions.channels,
            height=data_dimensions.height,
            width=len(subset_indices),
        )

        logger.info(
            "Set up %d expert SNP groups from '%s'. "
            "Union contains %d SNPs. Expert sizes: %s.",
            len(expert_groups),
            input_type_info.expert_snp_groups_file,
            len(subset_indices),
            {name: len(indices) for name, indices in expert_snp_indices.items()},
        )

    elif input_type_info.subset_snps_file:
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
        expert_snp_indices=expert_snp_indices,
    )

    return omics_input_info


def _read_expert_snp_groups(path: str) -> dict[str, list[str]]:
    with open(path) as f:
        groups = yaml.safe_load(f)

    if not isinstance(groups, dict):
        raise ValueError(
            f"Expected expert_snp_groups_file '{path}' to contain a YAML mapping "
            f"of expert_name -> list of SNP IDs, got {type(groups).__name__}."
        )

    for name, snp_list in groups.items():
        if not isinstance(snp_list, list) or not all(
            isinstance(s, str) for s in snp_list
        ):
            raise ValueError(
                f"Expert group '{name}' must map to a list of SNP ID strings."
            )

    return groups


def _compute_expert_union(expert_groups: dict[str, list[str]]) -> list[str]:
    seen: set[str] = set()
    union: list[str] = []
    for snp_list in expert_groups.values():
        for snp in snp_list:
            if snp not in seen:
                seen.add(snp)
                union.append(snp)
    return union


def _setup_expert_snp_indices(
    df_bim: pd.DataFrame,
    expert_groups: dict[str, list[str]],
    subset_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    subset_set = set(subset_indices.tolist())
    bim_var_ids = df_bim["VAR_ID"].values

    subset_index_to_position: dict[int, int] = {
        idx: pos for pos, idx in enumerate(subset_indices)
    }

    result: dict[str, np.ndarray] = {}
    for name, snp_list in expert_groups.items():
        positions: list[int] = []
        for snp_id in snp_list:
            bim_matches = np.where(bim_var_ids == snp_id)[0]
            for bim_idx in bim_matches:
                if bim_idx in subset_set:
                    positions.append(subset_index_to_position[bim_idx])
        result[name] = np.array(sorted(set(positions)), dtype=np.int64)

    return result


def _setup_snp_subset_indices(
    df_bim: pd.DataFrame,
    snps_to_subset: list[str],
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


def read_subset_file(subset_snp_file_path: str) -> list[str]:
    with open(subset_snp_file_path) as infile:
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


def _get_bim_headers() -> list[str]:
    bim_headers = ["CHR_CODE", "VAR_ID", "POS_CM", "BP_COORD", "ALT", "REF"]
    return bim_headers
