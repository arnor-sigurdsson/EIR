import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import deeplake
import numpy as np
import pandas as pd
import pytest
from _pytest.fixtures import SubRequest

from eir.setup.input_setup_modules.setup_image import default_image_loader
from tests.setup_tests.setup_modelling_test_data.setup_array_test_data import (
    create_test_array_data_and_labels,
)
from tests.setup_tests.setup_modelling_test_data.setup_image_test_data import (
    create_test_image_data,
)
from tests.setup_tests.setup_modelling_test_data.setup_omics_test_data import (
    create_test_omics_data_and_labels,
)
from tests.setup_tests.setup_modelling_test_data.setup_sequence_test_data import (
    create_test_sequence_data,
)
from tests.setup_tests.setup_modelling_test_data.setup_test_data_utils import (
    common_split_test_data_wrapper,
    set_up_test_data_root_outpath,
)


@pytest.fixture()
def create_test_data(request, tmp_path_factory, parse_test_cl_args) -> "TestDataConfig":
    test_data_config = _create_test_data_config(
        create_test_data_fixture_request=request,
        tmp_path_factory=tmp_path_factory,
        parsed_test_cl_args=parse_test_cl_args,
    )

    base_outfolder = set_up_test_data_root_outpath(
        base_folder=test_data_config.scoped_tmp_path
    )

    drop_random_samples = test_data_config.random_samples_dropped_from_modalities

    omics_path = base_outfolder / "omics"
    if "omics" in test_data_config.modalities and not omics_path.exists():
        omics_sample_path = create_test_omics_data_and_labels(
            test_data_config=test_data_config,
            array_outfolder=omics_path,
        )

        if drop_random_samples:
            _delete_random_files_from_folder(folder=omics_sample_path, n_to_drop=50)

    image_path = base_outfolder / "image"
    if "image" in test_data_config.modalities and not image_path.exists():
        image_sample_folder = create_test_image_data(
            test_data_config=test_data_config,
            image_output_folder=image_path,
        )
        if drop_random_samples:
            _delete_random_files_from_folder(folder=image_sample_folder, n_to_drop=50)

    sequence_path = base_outfolder / "sequence"
    if "sequence" in test_data_config.modalities and not sequence_path.exists():
        sequence_sample_folder = create_test_sequence_data(
            test_data_config=test_data_config,
            sequence_outfolder=sequence_path,
        )
        if drop_random_samples:
            _delete_random_files_from_folder(
                folder=sequence_sample_folder, n_to_drop=50
            )
            sequence_csv = base_outfolder / "sequence.csv"
            _delete_random_rows_from_csv(csv_file=sequence_csv, n_to_drop=50)

    arrays_path = base_outfolder / "array"
    if "array" in test_data_config.modalities and not arrays_path.exists():
        array_sample_folder = create_test_array_data_and_labels(
            test_data_config=test_data_config,
            array_output_folder=arrays_path,
        )
        if drop_random_samples:
            _delete_random_files_from_folder(folder=array_sample_folder, n_to_drop=50)

    _merge_labels_from_modalities(base_path=base_outfolder)

    if drop_random_samples:
        label_file = test_data_config.scoped_tmp_path / "labels.csv"
        _delete_random_rows_from_csv(csv_file=label_file, n_to_drop=50)
        _delete_random_cells_from_csv(csv_file=label_file, fraction_to_drop=0.10)
        _delete_random_cells_from_csv(
            csv_file=label_file,
            fraction_to_drop=0.20,
            columns=["SparseHeight", "SparseOrigin"],
        )

    if test_data_config.request_params.get("split_to_test", False):
        post_split_callables = _get_test_post_split_callables()
        for modality in test_data_config.modalities:
            common_split_test_data_wrapper(
                test_folder=test_data_config.scoped_tmp_path,
                name=modality,
                post_split_callables=post_split_callables,
            )

    if test_data_config.request_params.get("split_to_test", False):
        _make_deeplake_test_dataset(
            base_output_folder=base_outfolder, sub_folder_name="train_set"
        )
        _make_deeplake_test_dataset(
            base_output_folder=base_outfolder, sub_folder_name="test_set"
        )
    else:
        _make_deeplake_test_dataset(
            base_output_folder=base_outfolder, sub_folder_name=None
        )

    return test_data_config


def _get_test_post_split_callables() -> dict[str, Callable]:
    def _sequence_post_split(
        test_root_folder: Path,
        train_ids: Sequence[str],
        test_ids: Sequence[str],
    ) -> None:
        df_sequence = pd.read_csv(test_root_folder / "sequence.csv")

        df_sequence_train = df_sequence[df_sequence["ID"].isin(train_ids)]
        df_sequence_test = df_sequence[df_sequence["ID"].isin(test_ids)]

        df_sequence_train.to_csv(test_root_folder / "sequence_train.csv", index=False)
        df_sequence_test.to_csv(test_root_folder / "sequence_test.csv", index=False)

        (test_root_folder / "sequence.csv").unlink()

    callables = {"sequence": _sequence_post_split}

    return callables


def _merge_labels_from_modalities(base_path: Path) -> None:
    dfs = []

    for file in base_path.iterdir():
        # if we have already merged the labels
        if file.name in ("labels.csv", "labels_train.csv", "labels_test.csv"):
            return

        if file.name.startswith("labels_"):
            assert file.suffix == ".csv"
            dfs.append(pd.read_csv(file, index_col="ID"))

    df_final = dfs[0].copy()

    if len(dfs) == 1:
        df_final.to_csv(base_path / "labels.csv")
        return

    for df in dfs[1:]:
        assert df["Origin"].equals(df_final["Origin"])
        assert df["OriginExtraCol"].equals(df_final["Origin"])
        assert df.index.equals(df_final.index)

        df_final["Height"] += df["Height"]
        df_final["ExtraTarget"] += df["ExtraTarget"]

    df_final["Height"] /= len(dfs)
    df_final["ExtraTarget"] /= len(dfs)

    df_final.to_csv(base_path / "labels.csv")


def _make_deeplake_test_dataset(
    base_output_folder: Path,
    sub_folder_name: None | Literal["train_set", "test_set"],
) -> None:
    suffix = f"_{sub_folder_name}" if sub_folder_name is not None else ""
    ds_path = base_output_folder / f"deeplake{suffix}"

    if deeplake.exists(str(ds_path)):
        return

    shapes = {}
    for f in base_output_folder.iterdir():
        if not f.is_dir() or "deeplake" in f.name:
            continue

        file_iterator = f.iterdir()
        if sub_folder_name is not None:
            file_iterator = (f / sub_folder_name).iterdir()

        for sample_file in file_iterator:
            match f.name:
                case "omics":
                    data = np.load(str(sample_file))
                    shapes["test_genotype"] = data.shape
                case "image":
                    data = np.array(default_image_loader(str(sample_file)))

                    # we add an axis here to make it a 3D image, otherwise
                    # deeplake will complain as it expects a channel dimension
                    data = data[..., np.newaxis]

                    shapes["test_image"] = data.shape
                case "array":
                    data = np.load(str(sample_file))
                    shapes["test_array"] = data.shape
            if len(shapes) == 3:
                break
        if len(shapes) == 3:
            break

    ds = deeplake.create(str(ds_path))

    ds.add_column("ID", deeplake.types.Text())
    ds.add_column("test_sequence", deeplake.types.Text())

    if "test_genotype" in shapes:
        ds.add_column(
            "test_genotype",
            deeplake.types.Array(dtype="uint8", shape=shapes["test_genotype"]),
        )
        ds.add_column("test_genotype_exists", deeplake.types.Bool())

    if "test_image" in shapes:
        ds.add_column(
            "test_image",
            deeplake.types.Image(dtype="uint8", sample_compression="png"),
        )
        ds.add_column("test_image_exists", deeplake.types.Bool())

    if "test_array" in shapes:
        ds.add_column(
            "test_array",
            deeplake.types.Array(dtype="float32", shape=shapes["test_array"]),
        )
        ds.add_column("test_array_exists", deeplake.types.Bool())

    ds.commit("Created dataset schema")

    empty_data = {
        "test_genotype": (
            np.zeros(shapes["test_genotype"], dtype="bool")
            if "test_genotype" in shapes
            else None
        ),
        "test_image": (
            np.zeros(shapes["test_image"], dtype=np.uint8)
            if "test_image" in shapes
            else None
        ),
        "test_sequence": "",
        "test_array": (
            np.zeros(shapes["test_array"], dtype=np.float32)
            if "test_array" in shapes
            else None
        ),
    }

    samples = {}
    for f in base_output_folder.iterdir():
        if not f.is_dir() or "deeplake" in f.name:
            continue

        file_iterator = f.iterdir()
        if sub_folder_name is not None:
            file_iterator = (f / sub_folder_name).iterdir()

        for sample_file in file_iterator:
            sample_id = sample_file.stem
            if sample_id not in samples:
                sample = {"ID": [sample_id], "test_sequence": [""]}

                for col_name, zero_data in empty_data.items():
                    if col_name != "test_sequence":
                        if zero_data is not None:
                            sample[col_name] = [zero_data]
                            sample[f"{col_name}_exists"] = [False]

                samples[sample_id] = sample

            match f.name:
                case "omics":
                    data = np.load(str(sample_file))
                    assert (data.sum(0) == 1).all()

                    order = "C" if data.flags["C_CONTIGUOUS"] else "F"
                    assert order == "C"

                    samples[sample_id]["test_genotype"] = [data]
                    samples[sample_id]["test_genotype_exists"] = [True]
                case "image":
                    data = default_image_loader(str(sample_file))
                    data_np = np.array(data)
                    # we add an axis here to make it a 3D image, otherwise
                    # deeplake will complain as it expects a channel dimension
                    data_np = data_np[..., np.newaxis]

                    samples[sample_id]["test_image"] = [data_np]
                    samples[sample_id]["test_image_exists"] = [True]
                case "sequence":
                    data = sample_file.read_text().strip()
                    samples[sample_id]["test_sequence"] = [data if data else ""]
                case "array":
                    data = np.load(str(sample_file))

                    order = "C" if data.flags["C_CONTIGUOUS"] else "F"
                    assert order == "C"

                    samples[sample_id]["test_array"] = [data]
                    samples[sample_id]["test_array_exists"] = [True]
                case _:
                    raise ValueError(f"Unknown directory type: {f.name}")

    for _sample_id, sample in samples.items():
        ds.append(sample)

    ds.commit("Added all samples")


def _delete_random_rows_from_csv(csv_file: Path, n_to_drop: int):
    df = pd.read_csv(filepath_or_buffer=csv_file, index_col=0)
    drop_indices = np.random.choice(df.index, n_to_drop, replace=False)
    df_subset = df.drop(drop_indices)
    df_subset.to_csv(path_or_buf=csv_file)


def _delete_random_cells_from_csv(
    csv_file: Path,
    fraction_to_drop: float = 0.10,
    columns: Sequence[str] | None = None,
):
    df = pd.read_csv(filepath_or_buffer=csv_file, index_col=0)

    col_iterator = columns if columns is not None else df.columns
    for col in col_iterator:
        df.loc[df.sample(frac=fraction_to_drop).index, col] = np.nan

    df.to_csv(path_or_buf=csv_file)


def _delete_random_files_from_folder(folder: Path, n_to_drop: int):
    all_files = tuple(folder.iterdir())
    to_drop = random.sample(population=all_files, k=n_to_drop)

    for f in to_drop:
        f.unlink()


@dataclass
class TestDataConfig:
    request_params: dict
    task_type: str
    scoped_tmp_path: Path
    target_classes: dict[str, int]
    n_per_class: int
    n_snps: int
    modalities: Sequence[Literal["omics"] | Literal["sequence"]] = ("omics",)
    random_samples_dropped_from_modalities: bool = False
    source: Literal["local", "deeplake"] = "local"
    extras: dict[str, Any] = field(default_factory=dict)


def _create_test_data_config(
    create_test_data_fixture_request: SubRequest,
    tmp_path_factory,
    parsed_test_cl_args,
) -> TestDataConfig:
    request_params = create_test_data_fixture_request.param
    task_type = request_params["task_type"]

    if request_params.get("manual_test_data_creator", None):
        manual_name_creator_callable = request_params["manual_test_data_creator"]
        basename = str(manual_name_creator_callable())
    else:
        basename = "test_data_" + str(
            _hash_dict(dict_to_hash={**request_params, **parsed_test_cl_args})
        )

    scoped_tmp_path = tmp_path_factory.getbasetemp().joinpath(basename)

    if not scoped_tmp_path.exists():
        scoped_tmp_path.mkdir(mode=0o700)

    target_classes = {"Asia": 0, "Europe": 1}
    if task_type != "binary":
        target_classes["Africa"] = 2

    test_data_config = TestDataConfig(
        request_params=request_params,
        task_type=task_type,
        scoped_tmp_path=scoped_tmp_path,
        target_classes=target_classes,
        n_per_class=parsed_test_cl_args["n_per_class"],
        n_snps=parsed_test_cl_args["n_snps"],
        modalities=request_params.get("modalities", ("omics",)),
        random_samples_dropped_from_modalities=request_params.get(
            "random_samples_dropped_from_modalities", False
        ),
        source=request_params.get("source", "local"),
        extras=request_params.get("extras", {}),
    )

    return test_data_config


def _hash_dict(dict_to_hash: dict) -> int:
    dict_hash = hash(json.dumps(dict_to_hash, sort_keys=True))
    return dict_hash
